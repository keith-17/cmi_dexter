"""Sequence-aligned permutation importance for the fitted multibranch network.

Inputs are extracted once from the model's saved internal validation sequences.
Whole sequence blocks are exchanged only between recordings with the same
number of chunks and the same valid-timestep masks. No timesteps, padding or
model weights are changed, and no model is fitted by this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


_MOTION_BRANCHES = ("acc", "derived_imu", "rotation", "stft", "cwt", "noop")


def _unwrap_fitted(pipeline, X):
    """Use fitted preprocessing, if any, without refitting or mutating it."""
    if hasattr(pipeline, "best_estimator_"):
        pipeline = pipeline.best_estimator_
    if hasattr(pipeline, "steps"):
        steps = list(pipeline.steps)
        for _, transformer in steps[:-1]:
            if transformer not in (None, "passthrough"):
                X = transformer.transform(X)
        estimator = steps[-1][1]
    else:
        estimator = pipeline
    required = ("extractor_", "model_", "le_", "input_order_", "_model_inputs")
    missing = [name for name in required if not hasattr(estimator, name)]
    if missing:
        raise ValueError(f"Expected a fitted multibranch estimator; missing {missing}.")
    return estimator, X


def _validate_scope(estimator, X, sequence_col, require_internal_validation):
    if not isinstance(X, pd.DataFrame) or sequence_col not in X:
        raise ValueError(f"X_internal must contain {sequence_col!r}.")
    if X[sequence_col].isna().any():
        raise ValueError("Internal validation sequence IDs must not be missing.")
    ids = pd.Index(X[sequence_col].unique())
    if not len(ids):
        raise ValueError("Internal validation input is empty.")
    validation_ids = pd.Index(getattr(estimator, "validation_sequence_ids_", []))
    if require_internal_validation:
        if validation_ids.empty:
            raise ValueError("The fitted estimator has no saved internal validation IDs.")
        outside = ids.difference(validation_ids)
        if len(outside):
            raise ValueError(
                "Permutation importance accepts only the saved internal validation "
                f"sequences; {len(outside)} supplied IDs are outside that partition."
            )
    for attr in ("train_sequence_ids_", "preprocessing_sequence_ids_"):
        trained = pd.Index(getattr(estimator, attr, []))
        if len(ids.intersection(trained)):
            raise ValueError("Permutation importance must not include fitted training sequences.")
    return ids


def _sequence_truth(y, sequence_ids, sequence_col, target):
    if isinstance(y, pd.Series):
        if not y.index.is_unique:
            raise ValueError("Sequence-indexed labels must have a unique index.")
        labels = y
    elif isinstance(y, pd.DataFrame):
        yy = y.reset_index() if sequence_col not in y and y.index.name == sequence_col else y
        if sequence_col not in yy or target not in yy:
            raise ValueError(f"y_internal must contain {sequence_col!r} and {target!r}.")
        if (yy.groupby(sequence_col, sort=False)[target].nunique(dropna=False) > 1).any():
            raise ValueError("Target labels must be constant within each sequence.")
        labels = yy.drop_duplicates(sequence_col).set_index(sequence_col)[target]
    else:
        raise TypeError("Use a label DataFrame or a sequence-indexed Series.")
    labels = labels.reindex(sequence_ids)
    if labels.isna().any():
        raise ValueError("Missing labels for one or more extracted validation sequences.")
    return labels


def _sequence_layout(sequence_ids, mask):
    """Group donors by exact chunk count/order and padding support."""
    ids = pd.Index(sequence_ids)
    unique = ids.unique()
    if ids.isna().any() or len(mask) != len(ids):
        raise ValueError("Chunk IDs and masks must be non-missing and aligned.")
    rows = {sid: np.flatnonzero(ids == sid) for sid in unique}
    strata = {}
    for sid, ix in rows.items():
        support = np.ascontiguousarray(mask[ix], dtype=np.bool_)
        signature = (support.shape, support.tobytes())
        strata.setdefault(signature, []).append(sid)
    return unique, rows, list(strata.values())


def _donor_rows(rows, strata, rng, n_chunks):
    """A random cycle in each non-singleton stratum has no fixed points."""
    donor_rows = np.arange(n_chunks)
    moved = 0
    for stratum in strata:
        if len(stratum) < 2:
            continue
        order = rng.permutation(len(stratum))
        for i, pos in enumerate(order):
            recipient = stratum[int(pos)]
            donor = stratum[int(order[(i + 1) % len(order)])]
            donor_rows[rows[recipient]] = rows[donor]
            moved += 1
    return donor_rows, moved


def _prediction_probabilities(estimator, branch_arrays, mask, sequence_ids):
    inputs = estimator._model_inputs(branch_arrays, mask)
    probabilities = estimator.model_.predict(inputs, verbose=0)
    if isinstance(probabilities, dict):
        probabilities = probabilities["predictions"]
    elif isinstance(probabilities, (tuple, list)):
        names = list(getattr(estimator.model_, "output_names", []))
        if "predictions" not in names:
            raise ValueError("Cannot identify the trained model's predictions output.")
        probabilities = probabilities[names.index("predictions")]
    probabilities = np.asarray(probabilities)
    expected_shape = (len(sequence_ids), len(estimator.le_.classes_))
    if probabilities.shape != expected_shape or not np.isfinite(probabilities).all():
        raise ValueError(f"Expected finite model probabilities of shape {expected_shape}.")
    frame = pd.DataFrame(probabilities, index=pd.Index(sequence_ids), columns=estimator.le_.classes_)
    return frame.groupby(level=0, sort=False).mean()


def _default_groups(branch_arrays):
    groups = {}
    motion = [name for name in _MOTION_BRANCHES if name in branch_arrays]
    if motion:
        # Raw and derived motion change together, from the same donor sequence.
        groups["motion_coherent"] = motion
    for name in ("tof", "thermo", "noop"):
        if name in branch_arrays:
            groups[name] = [name]
    known = {name for names in groups.values() for name in names}
    for name in branch_arrays:
        if name not in known:
            groups[name] = [name]
    return groups


def compute_permutation_importance(
    pipeline,
    X_internal,
    y_internal,
    repeats=3,
    random_state=42,
    include_noop_individual=True,
    output_dir=None,
    prefix="noop_permutation",
    require_internal_validation=True,
    groups: Optional[Mapping[str, list]] = None,
):
    """Measure sequence macro-F1 drop in the actual fitted neural network.

    The returned DataFrame contains one row per coherent branch group and one
    row per NOOP feature (when enabled). ``importance_mean`` is baseline minus
    permuted macro-F1, averaged over ``repeats``. Macro-F1 uses the fitted label
    encoder's fixed class set. Negative values are retained.

    By default every input ID must belong to ``validation_sequence_ids_`` and
    none may belong to the model's training/preprocessing partition. Do not
    pass the final holdout. Repeated inspection still makes internal validation
    exploratory; these scores are model dependence, not causal importance.

    Donors are restricted to identical per-sequence window/mask layouts. The
    saved coverage identifies sequences that could actually move; singleton
    layouts remain unchanged. At least two compatible sequences are required.
    Arrays are copied only for the currently permuted group; originals and
    fitted model weights are never modified, including if prediction raises.

    If ``output_dir`` is provided, save CSV, strict JSON and a PNG bar plot.
    ``result.attrs`` contains baseline score, scope, coverage and saved paths.
    """
    if isinstance(repeats, bool) or not isinstance(repeats, (int, np.integer)) or repeats < 1:
        raise ValueError("repeats must be a positive integer.")
    if not isinstance(prefix, str) or not prefix or Path(prefix).name != prefix:
        raise ValueError("prefix must be a simple nonempty filename stem.")
    estimator, X = _unwrap_fitted(pipeline, X_internal)
    sequence_col = getattr(estimator.extractor_, "sequence_col", "sequence_id")
    requested = _validate_scope(estimator, X, sequence_col, require_internal_validation)
    chunks = estimator.extractor_.transform_chunks(X)
    mask = np.asarray(chunks["mask"], dtype=np.bool_)
    sequence_ids = np.asarray(chunks["sequence_ids"])
    if not pd.Index(sequence_ids).isin(requested).all():
        raise ValueError("Extractor returned sequences outside the supplied internal validation partition.")
    if not len(sequence_ids):
        raise ValueError("No internal validation chunks survived extraction.")
    if hasattr(estimator, "_branches_from_chunks"):
        branch_arrays = estimator._branches_from_chunks(chunks)
    else:
        if "noop" in getattr(estimator, "input_order_", []):
            raise ValueError("NOOP-enabled models require _branches_from_chunks support.")
        branch_arrays = estimator._split_branches(chunks["X"], mask, estimator.branch_groups_)
    branch_arrays = {name: np.asarray(value) for name, value in branch_arrays.items()}
    if set(estimator.input_order_) != set(branch_arrays):
        raise ValueError("Extracted branch arrays do not match the trained model inputs.")
    for name, array in branch_arrays.items():
        if array.shape[0] != len(sequence_ids):
            raise ValueError(f"Branch {name!r} is not aligned to extracted chunks.")
    del chunks

    unique_ids, rows, strata = _sequence_layout(sequence_ids, mask)
    permutable = sum(len(group) for group in strata if len(group) > 1)
    if not permutable:
        raise ValueError("No two validation sequences share compatible window and padding layouts.")
    target = getattr(estimator, "primary_target", "bfrb")
    labels = _sequence_truth(y_internal, unique_ids, sequence_col, target)
    classes = np.asarray(estimator.le_.classes_)
    if not labels.isin(classes).all():
        raise ValueError("Internal validation labels include a class absent from the trained classifier.")

    def score(arrays):
        probabilities = _prediction_probabilities(estimator, arrays, mask, sequence_ids).reindex(unique_ids)
        predictions = classes[np.argmax(probabilities.to_numpy(), axis=1)]
        return float(f1_score(labels.to_numpy(), predictions, labels=classes, average="macro", zero_division=0))

    baseline = score(branch_arrays)
    group_spec = _default_groups(branch_arrays) if groups is None else dict(groups)
    operations = []
    for name, branches in group_spec.items():
        branches = list(branches)
        if not branches or len(branches) != len(set(branches)) or set(branches) - set(branch_arrays):
            raise ValueError(f"Invalid branch group {name!r}: {branches}.")
        operations.append(("group", str(name), branches, None))
    if not operations:
        raise ValueError("At least one nonempty branch group is required.")
    noop_names = list(getattr(estimator, "noop_feature_names_", []))
    if "noop" in branch_arrays:
        if branch_arrays["noop"].ndim != 2 or len(noop_names) != branch_arrays["noop"].shape[1]:
            raise ValueError("NOOP feature names must match the trained static branch columns.")
        if len(noop_names) != len(set(noop_names)):
            raise ValueError("NOOP feature names must be unique.")
        if include_noop_individual:
            operations.extend(("noop_feature", name, ["noop"], index)
                              for index, name in enumerate(noop_names))
    # Reuse each repeat's donor map for every operation to reduce Monte Carlo noise.
    rng = np.random.default_rng(random_state)
    permutations = [_donor_rows(rows, strata, rng, len(sequence_ids))[0] for _ in range(repeats)]
    records = []
    for kind, name, branches, column in operations:
        scores = []
        for donor_rows in permutations:
            permuted = dict(branch_arrays)
            for branch in branches:
                original = branch_arrays[branch]
                if column is None:
                    permuted[branch] = original[donor_rows].copy()
                else:
                    block = original.copy()
                    block[:, column] = original[donor_rows, column]
                    permuted[branch] = block
            scores.append(score(permuted))
        drops = baseline - np.asarray(scores)
        records.append({
            "kind": kind, "name": name, "branches": "|".join(branches),
            "baseline_macro_f1": baseline,
            "permuted_macro_f1_mean": float(np.mean(scores)),
            "importance_mean": float(np.mean(drops)),
            "importance_std": float(np.std(drops)),
            "repeats": int(repeats), "n_sequences": int(len(unique_ids)),
            "n_permutable_sequences": int(permutable),
            "permutable_fraction": float(permutable / len(unique_ids)),
            "repeat_macro_f1": json.dumps(scores),
            "repeat_f1_drop": json.dumps(drops.tolist()),
        })
    result = pd.DataFrame(records).sort_values("importance_mean", ascending=False, kind="stable").reset_index(drop=True)
    result.attrs.update({
        "scope": "saved internal validation sequences",
        "metric": "macro-F1 over the fitted classifier's fixed class set",
        "baseline_macro_f1": baseline,
        "requested_sequences": int(len(requested)),
        "evaluated_sequences": int(len(unique_ids)),
        "excluded_sequence_ids": requested.difference(unique_ids).tolist(),
        "permutable_sequences": int(permutable),
        "permutable_fraction": float(permutable / len(unique_ids)),
        "permutation_unit": "whole sequence, exact chunk-count and mask-layout strata",
        "sequence_ids": unique_ids.tolist(),
        "random_state": int(random_state),
        "interpretation": "Trained-model dependence; correlated inputs can share or mask importance. Not causal attribution.",
    })
    if output_dir is not None:
        save_permutation_importance(result, output_dir, prefix=prefix)
    return result


def save_permutation_importance(result, output_dir, prefix="noop_permutation"):
    """Save the scored table, scope/coverage metadata, and a clearly labelled plot."""
    if not isinstance(prefix, str) or not prefix or Path(prefix).name != prefix:
        raise ValueError("prefix must be a simple nonempty filename stem.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{prefix}.csv"
    json_path = output_dir / f"{prefix}.json"
    plot_path = output_dir / f"{prefix}.png"
    result.to_csv(csv_path, index=False)
    paths = {"csv": str(csv_path), "json": str(json_path), "plot": str(plot_path)}
    metadata = dict(result.attrs)
    metadata["paths"] = paths
    payload = {"metadata": metadata, "importance": result.to_dict(orient="records")}
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False, default=_json_scalar), encoding="utf-8")
    import matplotlib.pyplot as plt
    ordered = result.sort_values("importance_mean", kind="stable")
    fig, ax = plt.subplots(figsize=(10, max(3, .32 * len(ordered))))
    labels = [f"{kind}: {name}" for kind, name in zip(ordered["kind"], ordered["name"])]
    ax.barh(labels, ordered["importance_mean"], xerr=ordered["importance_std"], capsize=3)
    ax.axvline(0, color="0.4", linewidth=.8)
    ax.set_xlabel("Decrease in internal-validation sequence macro-F1 (mean ± repeat SD)")
    ax.set_title("Permutation importance of the trained multibranch network")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    result.attrs["paths"] = paths
    return paths


def _json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")
