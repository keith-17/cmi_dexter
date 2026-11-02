"""Prepare, submit once, monitor and collect the matched private NOOP comparison.

No submission is performed by prepare, watch, collect, or plotting. Each variant
has its own immutable source package and acknowledgement/attempt records.
"""
from __future__ import annotations

import argparse
import ast
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "experiments/noop_candidate21"
VARIANTS = ("baseline", "enhanced")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def branch():
    return subprocess.run(["git", "branch", "--show-current"], cwd=ROOT,
        capture_output=True, check=True, text=True).stdout.strip()


def build_root(run_id):
    if not re.fullmatch(r"[a-z0-9-]+", run_id):
        raise ValueError("run_id must contain lowercase letters, digits and hyphens")
    return ROOT / "kaggle_build" / ("noop-comparison-" + run_id)


def results_root(run_id):
    return ROOT / "results" / ("noop-comparison-" + run_id)


def source_cells(nb):
    return ["".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in nb["cells"]]


def notebook_literals(notebook):
    values = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    continue
                values[node.targets[0].id] = value
    return values


def prepare(run_id, *, docker_image, enhanced_options=None):
    """Package the actual notebook through the existing GPU workflow."""
    import nbformat
    import kaggle_experiment as workflow
    assert branch() == "multibranch/v2/develop"
    notebook_path = ROOT / "notebooks/multibranch_pretraining.ipynb"
    original = read(notebook_path)
    controls = notebook_literals(original)
    reference = read(REFERENCE / "reference_params.json")
    best = controls["BEST_PARAMS"]
    assert all(best[key] == value for key, value in reference.items())
    assert controls["single_noop_experiment"] is True
    assert controls["search_mode"] == "best params"
    for family in ("motion_stats", "periodicity", "spectral_stats", "quality"):
        assert best["estimator__extractor__noop__" + family] is None
    requested = {key.removeprefix("estimator__extractor__noop__"): value
                 for key, value in controls["NOOP_EXPERIMENT_PARAMS"].items()}
    if enhanced_options is not None:
        assert requested == enhanced_options, "Use the visible notebook preset."
    sources = sorted((ROOT / "src").glob("*.py"))
    source_hashes = {path.name: sha(path) for path in sources}
    root = build_root(run_id)
    report = {"passed": False, "notebook": str(notebook_path), "variants": {}}
    for variant in VARIANTS:
        target = root / variant
        assert not (target / "submission_attempt.json").exists()
        target.mkdir(parents=True, exist_ok=True)
        slug = f"cmi-noop-{variant}-{run_id}"
        metadata = dict(id="keithmarange/" + slug, title=slug,
            code_file="multibranch_pretraining.ipynb", language="python", kernel_type="notebook",
            is_private=True, enable_gpu=True, enable_tpu=False, enable_internet=False,
            machine_shape="NvidiaTeslaT4", docker_image=docker_image,
            dataset_sources=[], competition_sources=["cmi-detect-behavior-with-sensor-data"],
            kernel_sources=[], model_sources=[], keywords=["gpu"])
        metadata_path = target / "configuration/kernel-metadata.json"
        write(metadata_path, metadata)
        old_build, old_metadata = workflow.BUILD, workflow.METADATA
        try:
            workflow.BUILD, workflow.METADATA = target / "submission", metadata_path
            workflow.prepare(notebook_name="multibranch_pretraining.ipynb", use_attached_source=False)
        finally:
            workflow.BUILD, workflow.METADATA = old_build, old_metadata
        path = target / "submission" / metadata["code_file"]
        packaged = read(path)
        assert source_cells(packaged)[1:] == source_cells(original)
        changed_cells = []
        if variant == "enhanced":
            for index, cell in enumerate(packaged["cells"]):
                if cell["cell_type"] != "code":
                    continue
                source, count = re.subn(r"^search_mode = 'best params'", "search_mode = 'grid'",
                                       "".join(cell["source"]), flags=re.MULTILINE)
                if count:
                    assert count == 1
                    cell["source"] = source.splitlines(True)
                    changed_cells.append(index)
            assert len(changed_cells) == 1
        assert notebook_literals(packaged)["BEST_PARAMS"] == best
        for cell in packaged["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
        nbformat.validate(nbformat.from_dict(packaged))
        write(path, packaged)
        config = dict(variant=variant, search_mode="best params" if variant == "baseline" else "grid",
                      candidates=1, epochs=best["estimator__epochs"], native_cv_and_refit=True,
                      reference=best, noop_options={} if variant == "baseline" else requested)
        write(target / "assets/config.json", config)
        manifest = dict(kernel=metadata["id"], notebook_sha256=sha(path),
            metadata_sha256=sha(target / "submission/kernel-metadata.json"),
            source_sha256=source_hashes, runtime_sha256={},
            original_notebook_sha256=sha(notebook_path), changed_notebook_cells=changed_cells,
            original_notebook="notebooks/multibranch_pretraining.ipynb", native_search_flow=True)
        write(target / "build_manifest.json", manifest)
        report["variants"][variant] = config
    report["passed"] = True
    write(root / "validation_report.json", report)
    print(json.dumps({"passed": True, "directory": str(root), "native_notebook": str(notebook_path),
                      "variants": {key: value["search_mode"] for key, value in report["variants"].items()}}, indent=2))
    return root


def api_client():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    return api


def submit(run_id, variant):
    root = build_root(run_id)
    assert read(root / "validation_report.json")["passed"]
    target = root / variant
    attempt_path, receipt_path = target / "submission_attempt.json", target / "submission_receipt.json"
    assert not attempt_path.exists() and not receipt_path.exists(), "Attempt already recorded; inspect it rather than resubmit"
    submission = target / "submission"
    metadata, manifest = read(submission / "kernel-metadata.json"), read(target / "build_manifest.json")
    assert sha(submission / metadata["code_file"]) == manifest["notebook_sha256"]
    assert sha(submission / "kernel-metadata.json") == manifest["metadata_sha256"]
    for name, digest in manifest["source_sha256"].items():
        assert sha(ROOT / "src" / name) == digest, "Source changed after packaging: " + name
    for name, digest in manifest["runtime_sha256"].items():
        assert sha(ROOT / "scripts" / name) == digest, "Runtime changed after packaging: " + name
    assert sha(ROOT / manifest["original_notebook"]) == manifest["original_notebook_sha256"], "Notebook changed after packaging"
    assert metadata["enable_gpu"] and metadata["is_private"]
    api = api_client()
    existing = api.kernels_list(user="keithmarange", search=metadata["title"], page_size=100)
    assert not any(getattr(k, "ref", "") == metadata["id"] for k in (existing or [])), "Slug already exists"
    write(attempt_path, dict(time_utc=datetime.now(timezone.utc).isoformat(), kernel=metadata["id"], push_request_count=1))
    try:
        receipt = api.kernels_push(str(submission)).to_dict(False)
        receipt.update(submitted_at_utc=datetime.now(timezone.utc).isoformat(), push_request_count=1,
                       submitted_notebook_sha256=manifest["notebook_sha256"])
        write(receipt_path, receipt)
        assert not receipt.get("error") and receipt.get("kernelId")
        assert not any(receipt.get(k) for k in ("invalidCompetitionSources", "invalidDatasetSources", "invalidKernelSources"))
        print(json.dumps(receipt, indent=2), flush=True)
    except BaseException as e:
        record = read(attempt_path); record.update(error_type=type(e).__name__, error=str(e))
        write(attempt_path, record)
        raise


def acknowledged_kernel(target):
    receipt = read(target / "submission_receipt.json")
    return receipt.get("ref", "").removeprefix("/code/") or read(target / "submission/kernel-metadata.json")["id"]


def verify_remote(api, target):
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    kernel = acknowledged_kernel(target)
    receipt = read(target / "submission_receipt.json")
    request = ApiGetKernelRequest()
    request.user_name, request.kernel_slug = kernel.split("/")
    request.version_label = "v" + str(receipt["versionNumber"])
    with api.build_kaggle_client() as client:
        remote = client.kernels.kernels_api_client.get_kernel(request)
    meta = read(target / "submission/kernel-metadata.json")
    assert source_cells(read(target / "submission" / meta["code_file"])) == source_cells(json.loads(remote.blob.source))
    assert remote.metadata.id == receipt["kernelId"]
    assert remote.metadata.current_version_number == receipt["versionNumber"]
    assert remote.metadata.enable_gpu and remote.metadata.is_private
    assert remote.metadata.machine_shape == "NvidiaTeslaT4"
    assert remote.metadata.docker_image == meta["docker_image"]
    write(target / "remote_verification.json", dict(kernel=kernel, source_equal=True,
        version=receipt["versionNumber"], gpu=True, private=True, machine_shape=remote.metadata.machine_shape))
    return kernel


def collect(api, run_id, variant):
    target = build_root(run_id) / variant
    kernel = verify_remote(api, target)
    status = api.kernels_status(kernel).to_dict(False)
    write(target / "latest_status.json", status)
    destination = results_root(run_id) / variant / "downloaded"
    destination.mkdir(parents=True, exist_ok=True)
    api.kernels_output(kernel, path=str(destination), file_pattern=r"(?:.*\.(?:csv|json|txt|log|png))$", quiet=True)
    assert status["status"] == "COMPLETE", status
    summaries = list(destination.rglob("*_summary.json"))
    assert len(summaries) == 1, summaries
    summary = read(summaries[0])
    assert summary["status"] == "complete", summary
    assert summary["experiment_count"] == summary["valid_candidate_count"] == 1
    assert summary["failed_candidate_count"] == 0
    assert len(summary["protocols"]) == 1
    config = read(target / "assets/config.json")
    params = summary["best_params"]
    expected = dict(config["reference"])
    expected.update({"estimator__extractor__noop__" + key: value for key, value in config["noop_options"].items()})
    for key, value in expected.items():
        assert params[key] == value, key
    history = summary["training_info"]["history"]["finetuning"]
    assert len(history["loss"]) == 2000
    assert list(destination.rglob("*internal_permutation.csv")), "Importance output missing"
    write(results_root(run_id) / variant / "history.json", history)
    write(results_root(run_id) / variant / "completed_results.json", dict(
        verified=True, kernel=kernel, native_notebook=True, epochs=2000,
        cv_score=summary["best_cv_score"], holdout_metrics=summary["holdout_metrics"],
        feature_importance=True, source_verified=True))
    print("VERIFIED_COMPLETE " + variant + " " + json.dumps(summary["holdout_metrics"]), flush=True)
    return summary


def parse_native_training_log(text, variant):
    """Observe existing Keras verbose=2 output without changing model callbacks."""
    rows, fit, previous_epoch, pending = [], 0, 0, None
    for line in text.splitlines():
        epoch = re.search(r"\bEpoch\s+(\d+)/(\d+)", line)
        if epoch:
            number = int(epoch.group(1))
            if fit == 0 or number <= previous_epoch:
                fit += 1
            previous_epoch, pending = number, (fit, number)
            continue
        if pending is None:
            continue
        metrics = {key: float(value) for key, value in re.findall(
            r"\b([a-z][a-z0-9_]*):\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", line)}
        if "loss" in metrics and any(key.endswith("accuracy") for key in metrics):
            rows.append(dict(variant=variant, fit=pending[0], epoch=pending[1], **metrics))
            pending = None
    return rows


def watch(run_id, seconds):
    api = api_client()
    targets = {v: build_root(run_id) / v for v in VARIANTS}
    kernels = {v: verify_remote(api, p) for v, p in targets.items()}
    events = queue.Queue()
    stop_streams = threading.Event()
    statuses = {}
    def stream(variant, kernel):
        failures = 0
        seen_events, emitted = set(), set()
        native_log = ""
        while not stop_streams.is_set():
            try:
                for event in api.kernels_logs_stream(kernel):
                    event_key = hashlib.sha256(json.dumps(event, sort_keys=True).encode()).hexdigest()
                    if event_key in seen_events:
                        continue
                    seen_events.add(event_key)
                    content = str(event.get("data", ""))
                    events.put((variant, content))
                    native_log += content
                    for row in parse_native_training_log(native_log, variant):
                        key = (row["fit"], row["epoch"])
                        if key not in emitted:
                            emitted.add(key)
                            events.put((variant, "\nNOOP_EPOCH " + json.dumps(row) + "\n"))
                    if stop_streams.is_set():
                        return
                failures = 0
                if statuses.get(variant, {}).get("status") in {"COMPLETE", "ERROR", "CANCELLED", "CANCEL_ACKNOWLEDGED"}:
                    return
                # A queued kernel can return an empty persisted log before its
                # live stream exists. Reconnect without resubmitting training.
                stop_streams.wait(10)
            except Exception as e:
                events.put((variant, "\nSTREAM_NOTE " + repr(e) + "\n"))
                failures += 1
                if "10013" in str(e) or "Permission" in str(e) or failures >= 4:
                    return
                stop_streams.wait(3)
    for v, k in kernels.items(): threading.Thread(target=stream, args=(v, k), daemon=True).start()
    buffers, epochs = {v: "" for v in VARIANTS}, {v: {} for v in VARIANTS}
    for v in VARIANTS:
        p = results_root(run_id) / v / "epoch_telemetry.jsonl"
        if p.exists():
            for line in p.read_text().splitlines():
                row = json.loads(line); epochs[v][(int(row.get("fit", 1)), int(row["epoch"]))] = row
    start, last_status = time.monotonic(), 0.
    complete, failures = set(), {}
    while time.monotonic() - start < seconds:
        try:
            variant, content = events.get(timeout=1)
            dest = results_root(run_id) / variant
            dest.mkdir(parents=True, exist_ok=True)
            with (dest / "live.log").open("a", encoding="utf-8") as handle: handle.write(content)
            buffers[variant] = (buffers[variant] + content)[-100_000:]
            for match in re.finditer(r"NOOP_EPOCH (\{[^\r\n]*\})", buffers[variant]):
                try: row = json.loads(match.group(1))
                except ValueError: continue
                number = (int(row.get("fit", 1)), int(row["epoch"]))
                if number not in epochs[variant]:
                    epochs[variant][number] = row
                    with (dest / "epoch_telemetry.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row) + "\n")
                    write(dest / "progress.json", row)
        except queue.Empty:
            pass
        if time.monotonic() - last_status >= 45:
            for variant, kernel in kernels.items():
                if variant in complete: continue
                status = api.kernels_status(kernel).to_dict(False)
                statuses[variant] = status
                write(targets[variant] / "latest_status.json", status)
                latest = epochs[variant].get(max(epochs[variant], default=0), {})
                print("PROGRESS " + json.dumps(dict(variant=variant, status=status, latest=latest)), flush=True)
                if status["status"] in {"COMPLETE", "ERROR", "CANCELLED", "CANCEL_ACKNOWLEDGED"}:
                    try:
                        collect(api, run_id, variant)
                    except Exception as error:
                        failures[variant] = dict(error_type=type(error).__name__, error=str(error))
                        write(results_root(run_id) / variant / "collection_failure.json", failures[variant])
                        print("COLLECTION_FAILURE " + variant + " " + str(error), flush=True)
                    complete.add(variant)
            if complete == set(VARIANTS): break
            last_status = time.monotonic()
    stop_streams.set()
    if failures:
        raise RuntimeError("Finished monitoring both runs; result failures: " + json.dumps(failures))
    return statuses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "submit", "watch", "collect", "verify"))
    parser.add_argument("--run-id", default="20261102-notebook")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--docker-image")
    parser.add_argument("--enhanced-options", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        assert args.docker_image and args.enhanced_options
        prepare(args.run_id, docker_image=args.docker_image, enhanced_options=read(args.enhanced_options))
    elif args.command == "submit":
        assert args.variant; submit(args.run_id, args.variant)
    elif args.command == "watch":
        watch(args.run_id, args.seconds)
    elif args.command == "collect":
        api = api_client()
        for v in ([args.variant] if args.variant else VARIANTS): collect(api, args.run_id, v)
    else:
        api = api_client()
        for v in VARIANTS: print(verify_remote(api, build_root(args.run_id) / v))


if __name__ == "__main__":
    main()
