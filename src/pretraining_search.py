"""Run comparable candidate searches separately for each validation protocol.

Validation changes which recordings may be learned from, so it is an outer
experiment choice rather than a candidate scored against a different test set.
Each protocol gets its own holdout, CV splitter and stage validation grouping.
Results share one ranked experiment CSV and one JSON training summary.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


_GROUP_PARAMS = (
    "estimator__validation_group_column",
    "estimator__pretraining_validation_group_column",
)


def _choices(value):
    if hasattr(value, "categories"):
        return list(value.categories)
    if hasattr(value, "choices"):
        return list(value.choices)
    if isinstance(value, (tuple, list)):
        return list(value)
    return [value]


def validation_protocols(validation_param_space=None):
    """Expand categorical protocol choices, ignoring group names for sequence mode."""
    space = validation_param_space or {"validation_mode": ["sequence"], "group_column": ["subject"]}
    unknown = set(space) - {"validation_mode", "group_column"}
    if unknown:
        raise ValueError(f"Unknown validation parameters: {sorted(unknown)}")
    modes = _choices(space.get("validation_mode", ["sequence"]))
    columns = _choices(space.get("group_column", ["subject"]))
    if not modes or not columns:
        raise ValueError("Validation parameter choices must not be empty.")
    protocols = []
    for mode in modes:
        if mode not in {"sequence", "subject"}:
            raise ValueError("validation_mode choices must be 'sequence' or 'subject'.")
        for column in ([None] if mode == "sequence" else columns):
            if mode == "subject" and (not isinstance(column, str) or not column):
                raise ValueError("Subject validation requires a nonempty group_column.")
            item = {"validation_mode": mode, "group_column": column}
            if item not in protocols:
                protocols.append(item)
    return protocols


def _json_value(value):
    """Keep structured parameters and histories readable in strict JSON."""
    if value is None or value is pd.NA or value is np.ma.masked:
        return None
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (list, tuple, pd.Index)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _write_json(path, value):
    path.write_text(json.dumps(_json_value(value), indent=2, allow_nan=False), encoding="utf-8")


def split_validation_protocol(frame, protocol, *, train_size, n_splits, random_state):
    """Split complete sequences before pretraining, fine-tuning or extraction."""
    if "sequence_id" not in frame:
        raise ValueError("Validation requires sequence_id.")
    if frame["sequence_id"].isna().any():
        raise ValueError("sequence_id must not be missing.")
    if not 0 < float(train_size) < 1:
        raise ValueError("train_size must be between 0 and 1.")
    if not isinstance(n_splits, (int, np.integer)) or n_splits < 1:
        raise ValueError("n_splits must be a positive integer.")
    mode, column = protocol["validation_mode"], protocol["group_column"]
    split_group = "sequence_id" if mode == "sequence" else column
    if split_group not in frame:
        raise ValueError(f"Missing group column {split_group!r} for {mode} validation.")
    metadata = frame[list(dict.fromkeys(["sequence_id", split_group]))].drop_duplicates()
    if metadata["sequence_id"].duplicated().any() or metadata[split_group].isna().any():
        raise ValueError("Validation groups must be non-missing and constant per sequence.")
    metadata = metadata.sort_values("sequence_id")
    if mode == "subject" and metadata[split_group].nunique() < 3:
        raise ValueError("Need at least three groups for independent holdout and CV training.")

    train, holdout = None, None
    if mode == "sequence":
        if __package__:
            from .base_utils import sample_balanced_split
        else:
            from base_utils import sample_balanced_split
        try:
            train, holdout = sample_balanced_split(
                frame, train_pct=train_size, test_pct=1 - train_size, random_state=random_state,
            )
        except (KeyError, ValueError) as exc:
            print(f"Balanced split unavailable ({exc}); using sequence-grouped random split.")
    if train is None:
        splitter = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(metadata, groups=metadata[split_group]))
        train_ids = metadata.iloc[train_idx]["sequence_id"]
        holdout_ids = metadata.iloc[test_idx]["sequence_id"]
        train = frame.loc[frame["sequence_id"].isin(train_ids)]
        holdout = frame.loc[frame["sequence_id"].isin(holdout_ids)]
    if set(train[split_group]) & set(holdout[split_group]):
        raise ValueError(f"Training and holdout overlap on {split_group!r}.")
    groups = train[split_group]
    needed = max(2, int(n_splits))
    if groups.nunique() < needed:
        raise ValueError(f"CV requires at least {needed} training groups for {mode} validation.")
    cv = (GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
          if n_splits == 1 else GroupKFold(n_splits=n_splits))
    return train, holdout, groups, cv


def _fixed_protocol_spaces(spaces, fixed):
    """Preserve every unrelated range and per-subspace Bayesian budget."""
    if isinstance(spaces, dict):
        result = dict(spaces)
        for key, value in fixed.items():
            previous = result.get(key)
            if hasattr(previous, "categories") or hasattr(previous, "choices"):
                result[key] = type(previous)([value])
            else:
                result[key] = [value]
        return result
    if isinstance(spaces, list):
        return [(_fixed_protocol_spaces(item[0], fixed), item[1])
                if isinstance(item, tuple) and len(item) == 2 else _fixed_protocol_spaces(item, fixed)
                for item in spaces]
    raise TypeError("Search spaces must be a dictionary or list of search-space dictionaries.")


def _refit_candidate(search, X, y):
    if __package__:
        from .multibranch_utils import refit_valid_candidate
    else:
        from multibranch_utils import refit_valid_candidate
    return refit_valid_candidate(search, X, y)


def _holdout_results(search, X_test, y_test, target_col):
    if __package__:
        from .base_utils import evaluate_holdout
    else:
        from base_utils import evaluate_holdout
    model = search.best_estimator_
    estimator = model.named_steps["estimator"]
    X_final = estimator.extractor_.select_finetuning_sequences(X_test, y_test)
    y_final = y_test.loc[X_final.index].copy()
    predictions = model.predict(X_final)
    expected = pd.Index(X_final["sequence_id"].unique(), name="sequence_id")
    if (not isinstance(predictions, pd.Series) or predictions.empty
            or not predictions.index.is_unique or not predictions.index.isin(expected).all()
            or predictions.isna().any()):
        raise ValueError("Holdout predictions must contain distinct IDs from the final-task holdout domain.")
    predictions.index.name = "sequence_id"
    excluded = expected.difference(predictions.index).tolist()
    y_final = y_final.loc[y_final["sequence_id"].isin(predictions.index)]
    exclusions = {
        "reason": "excluded by fitted extractor during prediction", "sequence_ids": excluded,
        "requested_sequences": len(expected), "evaluated_sequences": len(predictions),
        "evaluation_scope": "configured fine-tuning domain accepted by fitted extractor",
    }
    evaluation = evaluate_holdout(y_final, predictions, target_col=target_col, verbose=True)
    evaluation["exclusions"] = exclusions
    return evaluation


def _training_info(estimator):
    histories = {
        "pretraining": getattr(estimator, "pretraining_history_", {}),
        "finetuning_head": getattr(estimator, "finetuning_head_history_", {}),
        "finetuning": estimator.get_history_dict(),
    }
    return {
        "history": histories,
        "epochs_ran": {stage: max((len(values) for values in history.values()), default=0)
                       for stage, history in histories.items()},
        "pretraining_completed": getattr(estimator, "pretraining_completed_", False),
        "sequence_counts": {
            name.removesuffix("_sequence_ids_"): len(getattr(estimator, name))
            for name in ("pretraining_sequence_ids_", "pretraining_validation_sequence_ids_",
                         "finetuning_sequence_ids_", "validation_sequence_ids_")
            if hasattr(estimator, name)
        },
    }


def _experiment_rows(search, protocol, holdout_score, refitted, first_candidate):
    """Export one row per candidate, including grid-search parameters and failures."""
    rows = pd.DataFrame(search.cv_results_).copy()
    params = rows.get("params", pd.Series([{} for _ in range(len(rows))])).tolist()
    for key in sorted({key for candidate in params for key in candidate}):
        rows[f"param_{key}"] = [candidate.get(key) for candidate in params]
    rows["params"] = [json.dumps(_json_value(candidate), allow_nan=False) for candidate in params]
    for column in rows.columns:
        if column.startswith("param_"):
            rows[column] = rows[column].map(
                lambda value: json.dumps(_json_value(value), allow_nan=False)
                if isinstance(value, (dict, list, tuple, np.ndarray)) else value)
    scores = pd.to_numeric(rows.get("mean_test_score", pd.Series(index=rows.index, dtype=float)), errors="coerce")
    rows["mean_test_score"] = scores.where(np.isfinite(scores))
    rows["candidate_id"] = np.arange(first_candidate, first_candidate + len(rows))
    rows["validation_mode"] = protocol["validation_mode"]
    rows["group_column"] = protocol["group_column"]
    rows["selected_for_holdout"] = False
    rows["holdout_score"] = np.nan
    rows["refit_error"] = None
    for failure in getattr(search, "refit_errors_", []):
        rows.loc[failure["candidate"], "refit_error"] = failure["error"]
    best_index = getattr(search, "best_index_", None)
    if refitted and best_index is not None:
        rows.loc[best_index, "selected_for_holdout"] = True
        rows.loc[best_index, "holdout_score"] = holdout_score
    return rows


def _save_results(experiments, summaries, *, root, experiment_name, timestamp, target_col):
    """Overwrite the same two result files as each validation protocol completes."""
    rows = (pd.concat(experiments, ignore_index=True, sort=False) if experiments
            else pd.DataFrame(columns=["candidate_id", "validation_mode", "group_column", "mean_test_score", "params"]))
    rows = rows.sort_values("mean_test_score", ascending=False, kind="stable", na_position="last").reset_index(drop=True)
    rows.insert(0, "rank", np.arange(1, len(rows) + 1))
    leading = [name for name in ("rank", "candidate_id", "validation_mode", "group_column",
               "mean_test_score", "std_test_score", "mean_train_score", "selected_for_holdout",
               "holdout_score", "refit_error", "params") if name in rows]
    rows = rows[leading + [name for name in rows if name not in leading]]
    experiments_path = root / f"{experiment_name}_experiments.csv"
    summary_path = root / f"{experiment_name}_summary.json"
    rows.to_csv(experiments_path, index=False)
    completed = [summary for summary in summaries if summary["status"] == "complete"]
    refitted = [summary for summary in summaries if summary.get("best_candidate_id") is not None]
    best = max(refitted, key=lambda item: item["best_cv_score"], default={})
    candidate_id = best.get("best_candidate_id")
    summary = {
        "experiment_name": experiment_name, "timestamp": str(timestamp), "target": target_col,
        "status": ("complete" if len(completed) == len(summaries) else "partial" if completed else "failed"),
        "ranking_metric": "mean_test_score", "ranking_order": "descending",
        "experiment_count": len(rows), "best_candidate_id": candidate_id,
        "top_cv_candidate_id": int(rows.iloc[0]["candidate_id"]) if len(rows) else None,
        "best_cv_score": best.get("best_cv_score"), "holdout_score": best.get("holdout_score"),
        "best_params": best.get("best_params", {}), "training_info": best.get("training_info", {}),
        "validation_mode": best.get("validation_mode"), "group_column": best.get("group_column"),
        "holdout_metrics": best.get("holdout_metrics", {}), "protocols": summaries,
    }
    _write_json(summary_path, summary)
    return experiments_path, summary_path


def _release_model(run):
    """Persist before calling; keep at most one protocol's fitted model in memory."""
    run.clear()
    tensorflow = sys.modules.get("tensorflow")
    if tensorflow is not None:
        tensorflow.keras.backend.clear_session()
    gc.collect()


def run_validation_searches(
    search_template, experiment_df, *, validation_param_space=None, sensor_columns,
    train_size, n_splits, random_state, target_col, results_dir, experiment_name,
    timestamp, quick_save=True,
):
    """Fit a separate full search and final holdout evaluation per protocol.

    ``validation_param_space`` accepts lists, ECat or skopt Categorical choices.
    ``last_run`` retains the last protocol's fitted search for notebook plotting;
    ``protocol_results`` contains only lightweight metadata and file locations.
    Only a ranked all-candidate CSV and JSON summary are saved for the entire run.
    ``quick_save`` remains accepted for old callers; model weights are never saved.
    """
    protocols = validation_protocols(validation_param_space)
    root = Path(results_dir) / str(timestamp)
    root.mkdir(parents=True, exist_ok=True)
    summaries, experiments, last_run = [], [], {}
    candidate_count = 0
    for protocol in protocols:
        _release_model(last_run)
        mode, column = protocol["validation_mode"], protocol["group_column"]
        summary = {**protocol, "status": "failed", "best_cv_score": None,
                   "holdout_score": None, "results_dir": str(root)}
        run = {**protocol, "search": None, "experiment_succeeded": False,
               "eval_results": {}, "results_dir": root}
        search = None
        started = perf_counter()
        print(f"\nValidation protocol: {mode} | group: {column or 'sequence_id'}")
        try:
            X_train, X_test, groups, cv = split_validation_protocol(
                experiment_df, protocol, train_size=train_size,
                n_splits=n_splits, random_state=random_state,
            )
            labels = [key for key in experiment_df.columns if key not in sensor_columns
                      and key not in {"row_id", "sequence_counter"}]
            y_train, y_test = X_train[labels].copy(), X_test[labels].copy()
            fixed = {key: column for key in _GROUP_PARAMS}
            search = clone(search_template)
            search.set_params(cv=cv, refit=False)
            search.estimator.set_params(**fixed)
            estimator_params = search.estimator.get_params(deep=True)
            failure_key = "estimator__candidate_failure_log"
            if failure_key in estimator_params:
                fixed[failure_key] = None
                search.estimator.set_params(**{failure_key: None})
            space_name = "search_spaces" if hasattr(search, "search_spaces") else "param_grid"
            search.set_params(**{space_name: _fixed_protocol_spaces(getattr(search, space_name), fixed)})
            search.candidate_checkpoint_path = None
            run.update(search=search, X_train=X_train, X_test=X_test, y_train=y_train,
                       y_test=y_test, groups=groups, cv_object=cv)
            summary["configuration"] = {
                **protocol, "train_size": train_size, "n_splits": n_splits,
                "random_state": random_state, "target": target_col,
                "search_type": type(search).__name__,
                "train_sequences": X_train["sequence_id"].nunique(),
                "holdout_sequences": X_test["sequence_id"].nunique(),
                "train_groups": groups.nunique(),
                "holdout_groups": X_test["sequence_id" if column is None else column].nunique(),
            }
            search_started = perf_counter()
            search.fit(X_train, y_train, groups=groups)
            summary["search_seconds"] = perf_counter() - search_started
            diagnostics = {
                name: getattr(search, f"{name}_", default) for name, default in (
                    ("unique_candidate_count", len(search.cv_results_["params"])),
                    ("duplicate_proposal_count", 0), ("reused_candidate_count", 0),
                    ("rejected_candidate_count", 0), ("rejected_candidate_reasons", {}),
                    ("search_exhausted", False), ("proposal_limit_reached", False),
                )
            }
            summary["search_diagnostics"] = diagnostics
            succeeded = _refit_candidate(search, X_train, y_train)
            summary["refit_failures"] = search.refit_errors_
            if succeeded:
                estimator = search.best_estimator_.named_steps["estimator"]
                summary.update(best_cv_score=float(search.best_score_), best_params=search.best_params_,
                               best_candidate_id=candidate_count + int(search.best_index_) + 1)
                summary["training_info"] = {
                    **summary["configuration"], **_training_info(estimator),
                    "search_seconds": summary["search_seconds"],
                    "refit_seconds": getattr(search, "refit_time_", None),
                }
                del estimator
                evaluation = _holdout_results(search, X_test, y_test, target_col)
                if not np.isfinite(search.best_score_) or not np.isfinite(evaluation["competition_score"]):
                    raise ValueError("Completed validation requires finite CV and holdout scores.")
                run.update(experiment_succeeded=True, eval_results=evaluation)
                summary.update(status="complete", best_cv_score=float(search.best_score_),
                               holdout_score=float(evaluation["competition_score"]),
                               holdout_metrics={key: value for key, value in evaluation.items() if key != "results_df"})
                print(f"{mode}: best CV {search.best_score_:.6f}; holdout {evaluation['competition_score']:.6f}")
            else:
                summary["error"] = "No valid candidate could be refitted; holdout skipped."
        except Exception as exc:
            run["experiment_succeeded"] = False
            summary["error"] = f"{type(exc).__name__}: {exc}"
            print(f"{mode} validation failed: {summary['error']}")
        if search is not None and hasattr(search, "cv_results_"):
            rows = _experiment_rows(search, protocol, summary["holdout_score"],
                                    summary.get("best_candidate_id") is not None, candidate_count + 1)
            experiments.append(rows)
            candidate_count += len(rows)
        summary["elapsed_seconds"] = perf_counter() - started
        summaries.append(summary)
        experiments_path, summary_path = _save_results(
            experiments, summaries, root=root, experiment_name=experiment_name,
            timestamp=timestamp, target_col=target_col)
        last_run.update(run)
        run.clear()
        search = None
    return {"last_run": last_run, "protocol_results": summaries, "results_dir": root,
            "experiments_path": experiments_path, "summary_path": summary_path}
