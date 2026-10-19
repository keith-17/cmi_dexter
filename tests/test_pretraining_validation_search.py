"""Protocol orchestration checks without TensorFlow or expensive training."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from base_utils import ECat, EFloat
import pretraining_search as workflow


def recordings():
    rows = []
    for person in range(12):
        for category in range(3):
            for sample in range(2):
                rows.append({
                    "sequence_id": f"p{person:02}-c{category}", "subject": f"p{person:02}",
                    "gesture": f"g{category}", "bfrb": f"g{category}" if category < 2 else "non_bfrb",
                    "is_target": int(category < 2), "sequence_counter": sample,
                    "acc_x": float(sample),
                })
    return pd.DataFrame(rows)


class ToyEstimator(BaseEstimator):
    def __init__(self, validation_group_column="old", pretraining_validation_group_column="old",
                 candidate_failure_log=None, learning_rate=0.001):
        self.validation_group_column = validation_group_column
        self.pretraining_validation_group_column = pretraining_validation_group_column
        self.candidate_failure_log = candidate_failure_log
        self.learning_rate = learning_rate

    def fit(self, X, y):
        self.extractor_ = SimpleNamespace(select_finetuning_sequences=lambda frame, labels: frame.loc[frame.is_target == 1])
        return self

    def predict(self, X):
        # Deliberately reverse IDs: holdout evaluation must explicitly align them.
        result = X.drop_duplicates("sequence_id").set_index("sequence_id")["bfrb"].iloc[::-1].copy()
        return result

    def get_history_dict(self):
        return {"loss": [0.8, 0.5], "val_loss": [0.9, 0.6]}


class ToySearch(BaseEstimator):
    observed = []

    def __init__(self, estimator, param_grid, cv=None, refit=False):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.refit = refit

    def fit(self, X, y, groups):
        space = self.param_grid
        for train, validation in self.cv.split(X, y, groups):
            assert not set(groups.iloc[train]) & set(groups.iloc[validation])
        params = {}
        for key, values in space.items():
            if isinstance(values, EFloat):
                params[key] = values.lower
            else:
                params[key] = workflow._choices(values)[0]
        self.cv_results_ = {"params": [params], "mean_test_score": [0.75],
                            "split0_test_score": [0.75], "mean_train_score": [0.9]}
        type(self).observed.append({"params": params, "space": space, "groups": groups.copy(),
                                   "path": self.candidate_checkpoint_path})
        return self


def refit_toy(search, X, y):
    search.refit_errors_ = []
    search.best_params_ = search.cv_results_["params"][0]
    search.best_estimator_ = clone(search.estimator).set_params(**search.best_params_).fit(X, y)
    search.best_score_ = 0.75
    search.best_index_ = 0
    return True


class ValidationSearchTests(unittest.TestCase):
    def setUp(self):
        self.frame = recordings()
        ToySearch.observed = []

    def test_protocols_accept_search_categories_and_deduplicate_sequence_columns(self):
        protocols = workflow.validation_protocols({
            "validation_mode": ECat(["sequence", "subject"]),
            "group_column": ECat(["subject", "household"]),
        })
        self.assertEqual(protocols, [
            {"validation_mode": "sequence", "group_column": None},
            {"validation_mode": "subject", "group_column": "subject"},
            {"validation_mode": "subject", "group_column": "household"},
        ])

    def test_both_outer_and_cv_partitions_obey_selected_grouping(self):
        for mode in ("sequence", "subject"):
            with self.subTest(mode=mode):
                protocol = {"validation_mode": mode, "group_column": "subject" if mode == "subject" else None}
                train, holdout, groups, cv = workflow.split_validation_protocol(
                    self.frame, protocol, train_size=0.7, n_splits=2, random_state=42)
                key = "subject" if mode == "subject" else "sequence_id"
                self.assertFalse(set(train[key]) & set(holdout[key]))
                self.assertEqual(len(train) + len(holdout), len(self.frame))
                for fit, validation in cv.split(train, groups=groups):
                    self.assertFalse(set(train.iloc[fit][key]) & set(train.iloc[validation][key]))

    def test_missing_subject_is_allowed_only_for_sequence_validation(self):
        frame = self.frame.drop(columns="subject")
        train, holdout, _, _ = workflow.split_validation_protocol(
            frame, {"validation_mode": "sequence", "group_column": None},
            train_size=0.7, n_splits=1, random_state=42)
        self.assertFalse(set(train.sequence_id) & set(holdout.sequence_id))
        with self.assertRaisesRegex(ValueError, "Missing group column"):
            workflow.split_validation_protocol(frame, {"validation_mode": "subject", "group_column": "subject"},
                                               train_size=0.7, n_splits=1, random_state=42)

    def test_nonconstant_subjects_are_rejected_before_fitting(self):
        self.frame.loc[0, "subject"] = "different"
        with self.assertRaisesRegex(ValueError, "constant per sequence"):
            workflow.split_validation_protocol(self.frame, {"validation_mode": "subject", "group_column": "subject"},
                                               train_size=0.7, n_splits=1, random_state=42)

    def test_protocol_space_preserves_bayes_subspace_budgets_and_ranges(self):
        try:
            from skopt.space import Categorical, Real
        except ImportError:
            self.skipTest("scikit-optimize is optional in the packaging-only environment")
        learning_rate = Real(1e-5, 1e-3, prior="log-uniform")
        original = [({"estimator__learning_rate": learning_rate,
                      "estimator__validation_group_column": Categorical(["wrong"])}, 3)]
        fixed = workflow._fixed_protocol_spaces(original, {"estimator__validation_group_column": None})
        self.assertEqual(fixed[0][1], 3)
        self.assertIs(fixed[0][0]["estimator__learning_rate"], learning_rate)
        self.assertEqual(fixed[0][0]["estimator__validation_group_column"].categories, (None,))
        self.assertEqual(original[0][0]["estimator__validation_group_column"].categories, ("wrong",))

    def test_separate_searches_save_complete_outputs_even_in_quick_mode(self):
        learning_rate = EFloat(1e-5, 1e-3, distribution="log-uniform")
        template = ToySearch(Pipeline([("estimator", ToyEstimator())]), {
            "estimator__validation_group_column": ECat(["wrong"]),
            "estimator__pretraining_validation_group_column": ECat(["wrong"]),
            "estimator__candidate_failure_log": ECat(["old-path"]),
            "estimator__learning_rate": learning_rate,
        })
        with TemporaryDirectory() as directory, patch.object(workflow, "_refit_candidate", side_effect=refit_toy):
            outcome = workflow.run_validation_searches(
                template, self.frame, validation_param_space={"validation_mode": ["sequence", "subject"]},
                sensor_columns=["acc_x"], train_size=0.7, n_splits=1,
                random_state=42, target_col="bfrb", results_dir=directory,
                experiment_name="test", timestamp="run", quick_save=True)
            self.assertEqual(len(ToySearch.observed), 2)
            self.assertTrue(outcome["last_run"]["experiment_succeeded"])
            for record, protocol in zip(ToySearch.observed, outcome["protocol_results"]):
                column = protocol["group_column"]
                self.assertEqual(record["params"]["estimator__validation_group_column"], column)
                self.assertEqual(record["params"]["estimator__pretraining_validation_group_column"], column)
                self.assertEqual(record["space"]["estimator__learning_rate"].lower, learning_rate.lower)
                self.assertEqual(record["space"]["estimator__learning_rate"].upper, learning_rate.upper)
                destination = Path(protocol["results_dir"])
                self.assertEqual(record["path"].parent, destination)
                self.assertEqual(Path(record["params"]["estimator__candidate_failure_log"]).parent, destination)
                for filename in ("test_cv.csv", "test_best.csv", "test_holdout.csv",
                                 "pretraining_history.json", "finetuning_history.json"):
                    self.assertTrue((destination / filename).is_file(), filename)
                self.assertEqual(protocol["holdout_score"], 1.0)
                best = pd.read_csv(destination / "test_best.csv")
                self.assertEqual(len(best), 1)
                metadata = json.loads((destination / "configuration.json").read_text())
                self.assertFalse(set(metadata["train_groups"]) & set(metadata["holdout_groups"]))
            self.assertEqual(template.estimator.named_steps["estimator"].validation_group_column, "old")
            self.assertEqual(template.param_grid["estimator__validation_group_column"].choices, ["wrong"])
            summary = pd.read_csv(Path(outcome["results_dir"]) / "validation_summary.csv")
            self.assertEqual(summary["status"].tolist(), ["complete", "complete"])

    def test_failed_protocol_is_recorded_without_claiming_success(self):
        template = ToySearch(Pipeline([("estimator", ToyEstimator())]), {"estimator__learning_rate": [0.001]})
        with TemporaryDirectory() as directory, patch.object(workflow, "_refit_candidate", side_effect=refit_toy):
            outcome = workflow.run_validation_searches(
                template, self.frame.drop(columns="subject"),
                validation_param_space={"validation_mode": ["subject", "sequence"]},
                sensor_columns=["acc_x"], train_size=0.7, n_splits=1,
                random_state=42, target_col="bfrb", results_dir=directory,
                experiment_name="test", timestamp="run")
            self.assertEqual([result["status"] for result in outcome["protocol_results"]], ["failed", "complete"])
            failed = Path(outcome["protocol_results"][0]["results_dir"])
            self.assertTrue((failed / "protocol_error.json").is_file())
            self.assertFalse((failed / "test_best.csv").exists())


if __name__ == "__main__":
    unittest.main()
