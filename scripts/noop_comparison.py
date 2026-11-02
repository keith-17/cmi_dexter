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


def prepare(run_id, *, docker_image, enhanced_options):
    assert branch() == "multibranch/v2/develop", "Use the new multibranch/v2/develop branch"
    root = build_root(run_id)
    reference, expected = read(REFERENCE / "reference_params.json"), read(REFERENCE / "expected_split.json")
    assert len(reference) == 215 and reference["estimator__epochs"] == 2000
    assert reference["estimator__augmentor__prob"] == 0
    assert reference["estimator__early_stopping_patience"] == 0
    assert reference["estimator__restore_best_weights"] is False
    assert expected["target_holdout_sequences"] == 2539
    sources = sorted((ROOT / "src").glob("*.py"))
    assert {"noop_sequence_extractor.py", "noop_importance.py"} <= {p.name for p in sources}
    sys.path.insert(0, str(ROOT / "src"))
    from noop_sequence_extractor import NoopOptions
    from multibranch_utils import candidate_error
    from sklearn.base import clone
    import noop_runtime
    source_hashes = {p.name: sha(p) for p in sources}
    runtime_hashes = {name: sha(ROOT / "scripts" / name)
                      for name in ("noop_runtime.py", "plot_noop_comparison.py")}
    report = {"passed": False, "branch": branch(), "reference_parameter_count": len(reference),
              "source_sha256": source_hashes, "variants": {}}
    for variant in VARIANTS:
        target = root / variant
        assert not (target / "submission_attempt.json").exists(), "Preserve an already-attempted source package"
        target.mkdir(parents=True, exist_ok=True)
        noop = NoopOptions(**(enhanced_options if variant == "enhanced" else {}))
        noop.validate(float(reference["estimator__extractor__imu_native_sampling_rate"]))
        noop_params = noop.get_params(deep=False)
        config = dict(variant=variant, epochs=2000, noop_options=noop_params,
            comparison="Frozen Candidate21; only optional NOOP inputs and their required static branch differ",
            seed=42, feature_importance_domain="internal validation only", feature_importance_repeats=3,
            holdout_scope="Target-only, sequence-disjoint; binary F1 is not detection evidence",
            sampling_rate_note="Uses original Candidate21 configured100Hz unchanged; acquisition cadence not re-measured")
        pipeline = noop_runtime.make_pipeline(config, reference)
        error = candidate_error({}, base_params=pipeline.get_params(deep=True))
        assert error is None, error
        copied = clone(pipeline)
        assert copied.get_params()["estimator__extractor__noop"].get_params() == noop_params
        for key, value in reference.items(): assert copied.get_params()[key] == value, key
        # Numerical/no-fit checks precede submission; fitted CPU tests are separately required.
        for p in sources: ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        provenance = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
            source_sha256=source_hashes, runtime_sha256=runtime_hashes, git_branch=branch(),
            git_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                capture_output=True, check=True, text=True).stdout.strip(),
            reference_candidate=21, reference_params_sha256=sha(REFERENCE / "reference_params.json"),
            expected_split_sha256=sha(REFERENCE / "expected_split.json"),
            no_search=True, fits=1, epochs=2000, final_weight_holdout_evaluations=1,
            old_parameter_changes={}, noop_options=noop_params)
        files = {"src/" + p.name: p.read_bytes() for p in sources}
        files.update({"noop_runtime.py": (ROOT / "scripts/noop_runtime.py").read_bytes(),
                      "plot_noop_comparison.py": (ROOT / "scripts/plot_noop_comparison.py").read_bytes()})
        for name, value in {"config": config, "reference_params": reference,
                            "expected_split": expected, "provenance": provenance}.items():
            files["assets/" + name + ".json"] = json.dumps(value, indent=2, default=str).encode()
            write(target / ("assets/" + name + ".json"), value)
        archive_bytes = io.BytesIO()
        with zipfile.ZipFile(archive_bytes, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, data in sorted(files.items()): archive.writestr(name, data)
        payload = base64.b64encode(archive_bytes.getvalue()).decode("ascii")
        bootstrap = ("import base64, io, sys, zipfile\nfrom pathlib import Path\n"
            f"with zipfile.ZipFile(io.BytesIO(base64.b64decode({payload!r}))) as z:\n"
            "    z.extractall(Path.cwd())\n"
            "sys.path.insert(0, str(Path.cwd() / 'src'))\n")
        def cell(kind, text, ident):
            result = dict(cell_type=kind, id=ident, metadata={}, source=text.splitlines(True))
            if kind == "code": result.update(execution_count=None, outputs=[])
            return result
        notebook = dict(nbformat=4, nbformat_minor=5,
            metadata={"kernelspec": {"display_name": "Python3", "language": "python", "name": "python3"}},
            cells=[cell("markdown", f"# CMI NOOP v2 - {variant}\n"
                "Candidate21 fixed2000epochs; identical original data/split/settings.\n"
                "Final weights evaluated once. Importance uses internal validation only.\n", "intro"),
                cell("code", bootstrap, "source-snapshot"),
                cell("code", "import noop_runtime\nnoop_runtime.run()\n", "fixed-fit"),
                cell("code", "from IPython.display import Image, display\n"
                     "display(Image(filename='experiment_results/accuracy_loss.png'))\n", "curves"),
                cell("markdown", "## Feature importance\nPermutation importance of the trained model on internal validation; not attention weights.\n", "importance-title"),
                cell("code", "from pathlib import Path\n"
                     "for path in sorted(Path('experiment_results').glob('validation_permutation*.png')):\n"
                     "    display(Image(filename=str(path)))\n", "importance-plot")])
        for c in notebook["cells"]:
            if c["cell_type"] == "code": ast.parse("".join(c["source"]))
        try:
            import nbformat
            nbformat.validate(nbformat.from_dict(notebook))
        except ImportError:
            raise RuntimeError("nbformat is required for submission preflight")
        submission = target / "submission"
        submission.mkdir(exist_ok=True)
        notebook_path = submission / f"noop_{variant}.ipynb"
        write(notebook_path, notebook)
        slug = f"cmi-noop-v2-{variant}-{run_id}"
        metadata = dict(id="keithmarange/" + slug, title=slug,
            code_file=notebook_path.name, language="python", kernel_type="notebook",
            is_private=True, enable_gpu=True, enable_tpu=False, enable_internet=False,
            machine_shape="NvidiaTeslaT4", docker_image=docker_image,
            dataset_sources=[], competition_sources=["cmi-detect-behavior-with-sensor-data"],
            kernel_sources=[], model_sources=[], keywords=["gpu"])
        write(submission / "kernel-metadata.json", metadata)
        manifest = dict(kernel=metadata["id"], notebook_sha256=sha(notebook_path),
            metadata_sha256=sha(submission / "kernel-metadata.json"), source_sha256=source_hashes,
            runtime_sha256=runtime_hashes,
            payload_sha256=hashlib.sha256(archive_bytes.getvalue()).hexdigest())
        write(target / "build_manifest.json", manifest)
        report["variants"][variant] = dict(kernel=metadata["id"], clone_preserved=True,
            all_original_parameters_preserved=True, noop_options=noop_params)
    report["passed"] = True
    write(root / "validation_report.json", report)
    print(json.dumps(report, indent=2), flush=True)
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
    if status["status"] != "COMPLETE":
        raise RuntimeError(f"{kernel}: {status}; failure artifacts saved under {destination}")
    output = destination / "experiment_results"
    state, history, metrics = read(output / "execution_state.json"), read(output / "history.json"), read(output / "holdout_metrics.json")
    assert state["status"] == "COMPLETE" and state["fit_count"] == state["holdout_evaluation_count"] == 1
    assert state["final_weight_evaluation_used"] and len(history["loss"]) == 2000
    assert metrics["exclusions"]["evaluated_sequences"] == 2539
    assert any(output.glob("validation_permutation*.csv")), "Feature importance output missing"
    write(results_root(run_id) / variant / "completed_results.json", dict(
        verified=True, kernel=kernel, epochs=2000, metrics=metrics,
        validation_importance=True, source_verified=True))
    print("VERIFIED_COMPLETE " + variant + " " + json.dumps(metrics), flush=True)
    return state


def watch(run_id, seconds):
    api = api_client()
    targets = {v: build_root(run_id) / v for v in VARIANTS}
    kernels = {v: verify_remote(api, p) for v, p in targets.items()}
    events = queue.Queue()
    stop_streams = threading.Event()
    statuses = {}
    def stream(variant, kernel):
        failures = 0
        while not stop_streams.is_set():
            try:
                for event in api.kernels_logs_stream(kernel):
                    events.put((variant, str(event.get("data", ""))))
                    if stop_streams.is_set():
                        return
                failures = 0
                if statuses.get(variant, {}).get("status") in {"COMPLETE", "ERROR", "CANCELLED"}:
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
                row = json.loads(line); epochs[v][int(row["epoch"])] = row
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
                number = int(row["epoch"])
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
                if status["status"] in {"COMPLETE", "ERROR", "CANCELLED"}:
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
    parser.add_argument("--run-id", default="20261102")
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
