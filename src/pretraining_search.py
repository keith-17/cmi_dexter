"""Run comparable candidate searches separately for each validation protocol.

Validation changes which recordings may be learned from, so it is an outer
experiment choice rather than a candidate scored against a different test set.
Each protocol gets its own holdout, CV splitter, stage validation grouping and
result files. Scores from different protocols are never ranked together.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
import re
import sys
import traceback

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


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=_json_default), encoding="utf-8")


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


def _holdout_results(search, X_test, y_test, target_col, destination):
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
    _write_json(destination / "holdout_exclusions.json", {
        "reason": "excluded by fitted extractor during prediction", "sequence_ids": excluded,
        "requested_sequences": len(expected), "evaluated_sequences": len(predictions),
        "evaluation_scope": "configured fine-tuning domain accepted by fitted extractor",
    })
    return evaluate_holdout(y_final, predictions, target_col=target_col, verbose=True)


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
    All successful protocols save CV, best and holdout CSVs, even in quick mode.
    """
    if __package__:
        from .base_utils import build_quick_results_df
    else:
        from base_utils import build_quick_results_df
    protocols = validation_protocols(validation_param_space)
    root = Path(results_dir) / str(timestamp) / "validation_protocols"
    root.mkdir(parents=True, exist_ok=True)
    summaries, last_run = [], {}
    for protocol in protocols:
        _release_model(last_run)
        mode, column = protocol["validation_mode"], protocol["group_column"]
        slug = mode if mode == "sequence" else f"{mode}__{re.sub(r'[^A-Za-z0-9_.-]', '_', column)}"
        destination = root / slug
        destination.mkdir(parents=True, exist_ok=True)
        summary = {**protocol, "status": "failed", "best_cv_score": None,
                   "holdout_score": None, "results_dir": str(destination)}
        run = {**protocol, "search": None, "experiment_succeeded": False,
               "eval_results": {}, "results_dir": destination}
        search = None
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
                fixed[failure_key] = str(destination / "candidate_failures.jsonl")
                search.estimator.set_params(**{failure_key: fixed[failure_key]})
            space_name = "search_spaces" if hasattr(search, "search_spaces") else "param_grid"
            search.set_params(**{space_name: _fixed_protocol_spaces(getattr(search, space_name), fixed)})
            search.candidate_checkpoint_path = destination / "partial_cv.csv"
            run.update(search=search, X_train=X_train, X_test=X_test, y_train=y_train,
                       y_test=y_test, groups=groups, cv_object=cv)
            _write_json(destination / "configuration.json", {
                **protocol, "train_size": train_size, "n_splits": n_splits,
                "random_state": random_state, "target": target_col,
                "search_type": type(search).__name__, "search_space": getattr(search, space_name),
                "train_sequence_ids": X_train["sequence_id"].drop_duplicates().tolist(),
                "holdout_sequence_ids": X_test["sequence_id"].drop_duplicates().tolist(),
                "train_groups": groups.drop_duplicates().tolist(),
                "holdout_groups": X_test["sequence_id" if column is None else column].drop_duplicates().tolist(),
            })
            search.fit(X_train, y_train, groups=groups)
            diagnostics = {
                name: getattr(search, f"{name}_", default) for name, default in (
                    ("unique_candidate_count", len(search.cv_results_["params"])),
                    ("duplicate_proposal_count", 0), ("reused_candidate_count", 0),
                    ("rejected_candidate_count", 0), ("rejected_candidate_reasons", {}),
                    ("search_exhausted", False), ("proposal_limit_reached", False),
                )
            }
            _write_json(destination / "candidate_rules.json", diagnostics)
            succeeded = _refit_candidate(search, X_train, y_train)
            _write_json(destination / "refit_failures.json", search.refit_errors_)
            if succeeded:
                evaluation = _holdout_results(search, X_test, y_test, target_col, destination)
                if not np.isfinite(search.best_score_) or not np.isfinite(evaluation["competition_score"]):
                    raise ValueError("Completed validation requires finite CV and holdout scores.")
                run.update(experiment_succeeded=True, eval_results=evaluation)
                estimator = search.best_estimator_.named_steps["estimator"]
                _write_json(destination / "finetuning_history.json", estimator.get_history_dict())
                _write_json(destination / "pretraining_history.json", getattr(estimator, "pretraining_history_", {}))
                if not quick_save:
                    estimator.model_.save(destination / "best_model.keras")
                del estimator
                summary.update(status="complete", best_cv_score=float(search.best_score_),
                               holdout_score=float(evaluation["competition_score"]))
                evaluation["results_df"].to_csv(destination / f"{experiment_name}_holdout.csv", index=False)
                pd.DataFrame([{
                    **protocol, "target": target_col, "best_score": search.best_score_,
                    "best_params": json.dumps(search.best_params_, default=_json_default),
                    "holdout_score": evaluation["competition_score"],
                }]).to_csv(destination / f"{experiment_name}_best.csv", index=False)
                print(f"{mode}: best CV {search.best_score_:.6f}; holdout {evaluation['competition_score']:.6f}")
            else:
                summary["error"] = "No valid candidate could be refitted; holdout skipped."
        except Exception as exc:
            run["experiment_succeeded"] = False
            summary["error"] = f"{type(exc).__name__}: {exc}"
            _write_json(destination / "protocol_error.json", {"error": summary["error"], "traceback": traceback.format_exc()})
            print(f"{mode} validation failed: {summary['error']}")
        if search is not None and hasattr(search, "cv_results_"):
            raw = pd.DataFrame(search.cv_results_)
            raw.to_csv(destination / "raw_cv.csv", index=False)
            quick = (build_quick_results_df(search, holdout_score=summary["holdout_score"])
                     if run["experiment_succeeded"] else raw)
            quick.to_csv(destination / f"{experiment_name}_cv.csv", index=False)
        _write_json(destination / "results.json", {**summary, "best_params": getattr(search, "best_params_", {})})
        summaries.append(summary)
        pd.DataFrame(summaries).to_csv(root / "validation_summary.csv", index=False)
        _write_json(root / "validation_summary.json", summaries)
        last_run.update(run)
        run.clear()
        search = None
    return {"last_run": last_run, "protocol_results": summaries, "results_dir": root}
