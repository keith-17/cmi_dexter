"""Kaggle retrieval, candidate histories and isolated experiment preparation."""
from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
import re
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_REPO = Path(r"C:\Users\maran\OneDrive\Documents\Git Profile\cmi_dexter")
DEFAULT_KERNEL = "keithmarange/multibranch-pretraining"
AUGMENTATIONS = ["jitter", "gaussian_noise", "scaling", "time_shift", "crop_resize",
                 "temporal_mask", "channel_dropout", "sensor_dropout", "magnitude_warp"]


def api_client():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def as_dict(value):
    if isinstance(value, dict):
        return value
    try:
        return json.loads(str(value))
    except (ValueError, TypeError):
        return {"status": str(value)}


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return default


def coerce(value):
    if not isinstance(value, str):
        return value
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(value)
        except (ValueError, TypeError, SyntaxError):
            pass
    return value


def refresh_snapshot(kernel, capture_seconds=12):
    if not re.fullmatch(r"[\w-]+/[\w-]+", kernel):
        raise ValueError("Use an owner/notebook-slug Kaggle reference.")
    api = api_client()
    root = ROOT / "data" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    root.mkdir(parents=True)
    status = as_dict(api.kernels_status(kernel))
    record = {"kernel": kernel, "status": status, "checked_at": datetime.now(timezone.utc).isoformat()}
    (root / "status.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    api.kernels_pull(kernel, str(root / "remote"), metadata=True)
    api.kernels_output(kernel, str(root / "output"), file_pattern=r"(?:.*\.(csv|json|txt|log))$", quiet=True)
    if status.get("status") == "RUNNING" and hasattr(api, "kernels_logs_stream"):
        events = root / "output" / "live_events.jsonl"
        events.parent.mkdir(exist_ok=True)
        with events.open("w", encoding="utf-8") as stream:
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            process = subprocess.Popen([sys.executable, "-u", str(ROOT / "capture_logs.py"), kernel],
                                       stdout=stream, stderr=subprocess.DEVNULL, creationflags=flags)
            try:
                process.wait(timeout=capture_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        rendered = []
        for line in events.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                rendered.append(str(json.loads(line).get("data", "")))
            except ValueError:
                continue
        if rendered:
            (events.parent / "live_execution.log").write_text("".join(rendered), encoding="utf-8")
    return root


def flatten_log(text):
    try:
        events = json.loads(text)
        if isinstance(events, list) and all(isinstance(e, dict) for e in events):
            text = "".join(str(e.get("data", "")) for e in events)
    except ValueError:
        pass
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text).replace("\r", "\n")


def parse_epoch_log(text):
    """Associate completed CV records with their preceding, separate epoch blocks."""
    trials, blocks, current = [], [], {}
    epoch, fits = None, 0
    protocol = "unreported"
    for line in flatten_log(text).splitlines():
        if line.startswith("Validation protocol:"):
            protocol = line.split(":", 1)[1].split("|", 1)[0].strip()
        if line.startswith("Fitting "):
            fits += 1
        match = re.match(r"^Epoch (\d+)/(\d+)", line)
        if match:
            epoch = int(match[1])
            if epoch == 1 and current:
                blocks.append(list(current.values()))
                current = {}
            continue
        if epoch and re.search(r"(?:^|\s)(?:\w+_)?accuracy:", line):
            metrics = {key: float(value) for key, value in
                       re.findall(r"\b([\w]+):\s*([-+\d.eE]+)(?=\s|$)", line)
                       if key not in {"s", "ms"}}
            if "loss" in metrics:
                current[epoch] = {"epoch": epoch, **metrics}
        if line.startswith("[CV ") and "] END " in line:
            body = line.split("] END ", 1)[1]
            matches = list(re.finditer(r"(?:^|, )(estimator__[\w]+)=", body))
            params = {}
            footer = body.find(";, score=")
            for i, item in enumerate(matches):
                end = matches[i + 1].start() if i + 1 < len(matches) else footer
                params[item[1]] = coerce(body[item.end():end])
            score = re.search(r"score=\(train=([-+\d.eE]+), test=([-+\d.eE]+)\)", body)
            if not score:
                score = re.search(r"score=([-+\d.eE]+)", body)
            if score:
                histories = blocks + ([list(current.values())] if current else [])
                trials.append({"id": len(trials) + 1, "params": params,
                               "score": float(score[2] if score.lastindex == 2 else score[1]),
                               "histories": histories, "protocol": protocol,
                               "source": "Live log (rounded score)"})
            blocks, current, epoch = [], {}, None
    return trials, blocks + ([list(current.values())] if current else []), fits


def normal(value):
    value = coerce(value)
    return json.dumps(value, sort_keys=True, default=str)


def get_config(notebook):
    result = {}
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        try:
            tree = ast.parse("".join(cell.get("source", [])))
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.Assign):
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        result[target.id] = value
    return result


def load_snapshot(root):
    root = Path(root)
    status = read_json(root / "status.json", {})
    if isinstance(status.get("status"), str):
        status["status"] = as_dict(status["status"])
    metadata = read_json(root / "remote" / "kernel-metadata.json", {})
    notebook_path = root / "remote" / metadata.get("code_file", "missing.ipynb")
    notebook = read_json(notebook_path, {})
    config = get_config(notebook)
    output = root / "output"
    logs = list(output.rglob("*.log")) if output.exists() else []
    text = ""
    if logs:
        text = max((flatten_log(f.read_text(encoding="utf-8", errors="replace")) for f in logs), key=len)
    trials, running, fits = parse_epoch_log(text)
    available_protocols = list(dict.fromkeys(t.get("protocol", "unreported") for t in trials))
    selected_protocol = "sequence" if "sequence" in available_protocols else next(iter(available_protocols), "unreported")
    trials = [t for t in trials if t.get("protocol", "unreported") == selected_protocol]
    cv_paths = sorted(output.rglob("raw_cv.csv")) if output.exists() else []
    if not cv_paths:
        cv_paths = sorted(output.rglob("partial_cv.csv")) if output.exists() else []
    if not cv_paths and output.exists():
        cv_paths = [p for p in output.rglob("*_cv.csv") if "partial" not in p.name]
    if cv_paths:
        frames = []
        for path in cv_paths:
            try:
                frame = pd.read_csv(path)
            except (OSError, pd.errors.EmptyDataError):
                continue
            if "mean_test_score" in frame:
                frames.append((path, frame))
        if frames:
            # Preserve separate validation protocols rather than merging their ranks.
            frames.sort(key=lambda item: ("sequence" not in str(item[0]), -len(item[1])))
            path, frame = frames[0]
            selected_protocol = path.parent.name
            recorded = []
            for index, row in frame.iterrows():
                params = coerce(row.get("params", "{}"))
                if not isinstance(params, dict):
                    params = {key[6:]: coerce(value) for key, value in row.items() if key.startswith("param_")}
                matches = []
                for trial in trials:
                    common = set(trial["params"]) & set(params)
                    if common and all(normal(trial["params"][key]) == normal(params[key]) for key in common):
                        matches.append(trial)
                histories = [block for match in matches for block in match["histories"]]
                if not histories and config.get("n_splits", 1) == 1 and len(frame) == len(trials):
                    candidate = trials[index]
                    if abs(candidate["score"] - float(row["mean_test_score"])) <= .00051:
                        histories = candidate["histories"]
                recorded.append({"id": index + 1, "score": float(row["mean_test_score"]),
                                 "params": params, "histories": histories, "source": str(path.name)})
            trials = recorded
    protocol_records = []
    if output.exists():
        for path in output.rglob("validation_summary.json"):
            value = read_json(path, [])
            if isinstance(value, list):
                protocol_records.extend(value)
    holdout = next((v.get("holdout_score") for v in protocol_records
                    if v.get("validation_mode") == "sequence" and v.get("holdout_score") is not None), None)
    failure_files = list(output.rglob("candidate_failures.jsonl")) if output.exists() else []
    failures = sum(len(p.read_text(encoding="utf-8", errors="replace").splitlines()) for p in failure_files)
    failures = max(failures, sum(not pd.notna(t["score"]) or t["score"] < 0 for t in trials))
    budget = config.get("n_iter")
    if config.get("search_mode") == "evo" and config.get("population_size") and config.get("generations") is not None:
        budget = config["population_size"] * (config["generations"] + 1)
    return {"root": str(root), "status": status, "metadata": metadata, "config": config,
            "notebook_path": str(notebook_path), "trials": trials, "running_histories": running,
            "fits": fits, "log": text, "failures": failures, "protocols": protocol_records,
            "holdout": holdout, "budget": budget,
            "protocol": selected_protocol,
            "experiment": config.get("experiment_name", metadata.get("title", root.name))}


def snapshots():
    paths = [path.parent for path in (ROOT / "data").glob("*/status.json")]
    return sorted(paths, key=lambda p: p.name, reverse=True)


def leaderboard(snapshot):
    records = []
    for trial in snapshot["trials"]:
        p = trial["params"]
        blocks = trial["histories"]
        last = blocks[-1][-1] if blocks and blocks[-1] else {}
        records.append({"Trial": trial["id"], "CV score": trial["score"],
                        "Acceleration": p.get("estimator__extractor__acc_modes", "—"),
                        "CNN dropout": p.get("estimator__conv_dropout", "—"),
                        "Branch dropout": p.get("estimator__branch_dropout", "—"),
                        "Aug. probability": p.get("estimator__augmentor__prob", "—"),
                        "Learning rate": p.get("estimator__learning_rate", "—"),
                        "Epochs logged": last.get("epoch"),
                        "Last val. accuracy": last.get("val_accuracy", last.get("val_predictions_accuracy"))})
    if not records:
        return pd.DataFrame(columns=["Trial", "CV score", "Acceleration", "CNN dropout", "Aug. probability"])
    return pd.DataFrame(records).sort_values("CV score", ascending=False, na_position="last")


def curve_summary(trial):
    blocks = trial["histories"]
    if not blocks:
        return "Epoch-level history is not published for this candidate yet."
    h = blocks[-1]
    loss_rows = [r for r in h if "val_loss" in r]
    if not loss_rows:
        return f"{len(h)} epochs logged. Internal validation loss is unavailable."
    best = min(loss_rows, key=lambda r: r["val_loss"])
    last = loss_rows[-1]
    message = f"Minimum validation loss {best['val_loss']:.4f} at epoch {best['epoch']}; {len(h)} epochs logged."
    if last["val_loss"] > best["val_loss"] * 1.05 and last["loss"] < last["val_loss"]:
        message += " The widening loss gap suggests overfitting after the minimum."
    return message


def patch_literals(notebook, updates):
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell.get("source", []))
        lines = source.splitlines(keepends=True)
        replacements = []
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in updates:
                        replacements.append((node.lineno - 1, node.end_lineno,
                                             f"{target.id} = {updates[target.id]!r}\n"))
        for start, end, replacement in sorted(replacements, reverse=True):
            lines[start:end] = [replacement]
        cell["source"] = "".join(lines).splitlines(keepends=True)


def prepare_experiment(snapshot, settings, repository=DEFAULT_REPO):
    """Create a new notebook without touching any source notebook or repository file."""
    notebook = read_json(snapshot["notebook_path"])
    if not notebook:
        raise ValueError("Refresh Kaggle first to retrieve a notebook template.")
    notebook = json.loads(json.dumps(notebook))
    config = get_config(notebook)
    space = dict(config.get("NEXT_RUN_SPACE", {}))
    space.update(settings["space"])
    patch_literals(notebook, {"experiment_name": settings["name"], "search_mode": settings["search"],
                             "n_iter": settings["iterations"], "TARGET_COL": settings["target"],
                             "FIRST_RUN": False, "BASE_LEARNING_RATE": settings["space"]["estimator__learning_rate"][0],
                             "FINETUNING_EPOCHS": settings["epochs"], "orientation_filter_list": settings["orientations"],
                             "NEXT_RUN_SPACE": space})
    # Inject the final override immediately before sklearn/skopt search-space conversion.
    # Apply it to both search modes, including stage settings that were merged earlier.
    injected = False
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        text = "".join(cell["source"])
        if "prepare_multibranch_bayesian_space" not in text:
            continue
        markers = list(re.finditer(r"^if SKOPT_AVAILABLE:", text, re.M))
        if markers:
            position = markers[0].start()
            override = ("# Dashboard configuration is applied after shared-stage settings.\n"
                        f"_dashboard_space = {space!r}\n"
                        "GRID_PARAM_SPACE.update(_dashboard_space)\n"
                        "EVOLUTIONARY_PARAM_SPACE.update({key: ECat(values) for key, values in _dashboard_space.items()})\n"
                        "if 'param_space' in globals() and isinstance(param_space, dict):\n"
                        "    param_space.update(_dashboard_space)\n")
            text = text[:position] + override + text[position:]
            cell["source"] = text.splitlines(keepends=True)
            injected = True
            break
    if not injected:
        raise ValueError("The notebook search construction could not be verified. Download the configuration; no run was submitted.")
    source = Path(repository) / "src"
    required = ["base_utils.py", "multibranch_utils.py", "pretraining_utils.py", "pretraining_search.py"]
    if any(not (source / name).is_file() for name in required):
        raise ValueError("Choose a repository containing the four current CMI source modules.")
    data = io.BytesIO()
    hashes = {}
    with zipfile.ZipFile(data, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.glob("*.py")):
            content = path.read_bytes()
            ast.parse(content.decode("utf-8-sig"))
            archive.writestr(path.name, content)
            hashes[path.name] = hashlib.sha256(content).hexdigest()
    encoded = base64.b64encode(data.getvalue()).decode("ascii")
    bootstrap = f'''# Exact source snapshot prepared by CMI Dexter dashboard.
import base64,io,sys,zipfile
from pathlib import Path
_source_root=Path('/kaggle/working/src');_source_root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode({encoded!r}))) as _archive:
    for _entry in _archive.namelist():
        if Path(_entry).name != _entry: raise ValueError('Invalid source member')
        (_source_root/_entry).write_bytes(_archive.read(_entry))
sys.path.insert(0,str(_source_root))
import tensorflow as tf
if not tf.config.list_physical_devices('GPU'): raise RuntimeError('GPU required for this experiment')
try: import skopt
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable,'-m','pip','install','scikit-optimize'])
'''
    notebook["cells"] = [c for c in notebook["cells"] if not (c.get("cell_type") == "code" and "base64.b64decode(" in "".join(c.get("source", [])))]
    notebook["cells"].insert(0, {"cell_type": "code", "metadata": {}, "execution_count": None,
                                   "outputs": [], "source": bootstrap.splitlines(keepends=True)})
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            cell.update(execution_count=None, outputs=[])
    notebook.get("metadata", {}).pop("papermill", None)
    meta = dict(snapshot["metadata"])
    meta.update(code_file="experiment.ipynb", enable_gpu=True, enable_tpu=False)
    build = ROOT / "builds" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    build.mkdir(parents=True)
    (build / "experiment.ipynb").write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    (build / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (build / "configuration.json").write_text(json.dumps({**settings, "source_sha256": hashes}, indent=2), encoding="utf-8")
    return build


def submit_experiment(build):
    metadata = read_json(Path(build) / "kernel-metadata.json")
    api = api_client()
    state = as_dict(api.kernels_status(metadata["id"]))
    if state.get("status") in {"RUNNING", "QUEUED", "STARTING"}:
        raise ValueError("This notebook already has an active run. Wait for it to finish before submitting another version.")
    response = as_dict(api.kernels_push(str(build)))
    if response.get("error") or response.get("failureMessage"):
        raise ValueError(str(response))
    record = {"response": response, "submitted_at": datetime.now(timezone.utc).isoformat(),
              "status": as_dict(api.kernels_status(metadata["id"]))}
    (Path(build) / "submitted.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record
