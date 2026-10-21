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

    def test_separate_searches_save_only_csv_and_json_even_with_full_save_requested(self):
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
                experiment_name="test", timestamp="run", quick_save=False)
            self.assertEqual(len(ToySearch.observed), 2)
            self.assertTrue(outcome["last_run"]["experiment_succeeded"])
            for record, protocol in zip(ToySearch.observed, outcome["protocol_results"]):
                column = protocol["group_column"]
                self.assertEqual(record["params"]["estimator__validation_group_column"], column)
                self.assertEqual(record["params"]["estimator__pretraining_validation_group_column"], column)
                self.assertEqual(record["space"]["estimator__learning_rate"].lower, learning_rate.lower)
                self.assertEqual(record["space"]["estimator__learning_rate"].upper, learning_rate.upper)
                self.assertIsNone(record["path"])
                self.assertIsNone(record["params"]["estimator__candidate_failure_log"])
                self.assertEqual(protocol["holdout_score"], 1.0)
                self.assertGreater(protocol["configuration"]["train_sequences"], 0)
            self.assertEqual(template.estimator.named_steps["estimator"].validation_group_column, "old")
            self.assertEqual(template.param_grid["estimator__validation_group_column"].choices, ["wrong"])
            self.assertEqual({path.name for path in Path(directory).rglob("*") if path.is_file()},
                             {"test_experiments.csv", "test_summary.json"})
            experiments = pd.read_csv(outcome["experiments_path"])
            self.assertEqual(len(experiments), 2)
            self.assertEqual(experiments["rank"].tolist(), [1, 2])
            self.assertEqual(experiments["validation_mode"].tolist(), ["sequence", "subject"])
            self.assertTrue(experiments["selected_for_holdout"].all())
            summary = json.loads(outcome["summary_path"].read_text())
            self.assertEqual([result["status"] for result in summary["protocols"]], ["complete", "complete"])
            self.assertEqual(summary["best_cv_score"], 0.75)
            self.assertEqual(summary["holdout_score"], 1.0)
            self.assertEqual(summary["best_params"], json.loads(experiments.iloc[0]["params"]))
            self.assertEqual(summary["training_info"]["epochs_ran"]["finetuning"], 2)
            self.assertEqual(summary["training_info"]["history"]["finetuning"]["loss"], [0.8, 0.5])

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
            summary = json.loads(outcome["summary_path"].read_text())
            self.assertEqual(summary["status"], "partial")
            self.assertIn("Missing group column", summary["protocols"][0]["error"])
            self.assertEqual(summary["validation_mode"], "sequence")
            self.assertEqual(len(list(Path(directory).rglob("*.*"))), 2)

    def test_all_failed_runs_still_produce_two_readable_results(self):
        template = ToySearch(Pipeline([("estimator", ToyEstimator())]), {"estimator__learning_rate": [0.001]})
        with TemporaryDirectory() as directory:
            outcome = workflow.run_validation_searches(
                template, self.frame.drop(columns="subject"),
                validation_param_space={"validation_mode": ["subject"]},
                sensor_columns=["acc_x"], train_size=0.7, n_splits=1,
                random_state=42, target_col="bfrb", results_dir=directory,
                experiment_name="test", timestamp="run")
            self.assertTrue(pd.read_csv(outcome["experiments_path"]).empty)
            summary = json.loads(outcome["summary_path"].read_text())
            self.assertEqual(summary["status"], "failed")
            self.assertIsNone(summary["holdout_score"])
            self.assertEqual(summary["best_params"], {})

    def test_holdout_failure_keeps_refitted_parameters_and_training_info(self):
        template = ToySearch(Pipeline([("estimator", ToyEstimator())]), {"estimator__learning_rate": [0.001]})
        with TemporaryDirectory() as directory, \
                patch.object(workflow, "_refit_candidate", side_effect=refit_toy), \
                patch.object(workflow, "_holdout_results", side_effect=ValueError("Invalid predictions")):
            outcome = workflow.run_validation_searches(
                template, self.frame, sensor_columns=["acc_x"], train_size=0.7, n_splits=1,
                random_state=42, target_col="bfrb", results_dir=directory,
                experiment_name="test", timestamp="run")
            summary = json.loads(outcome["summary_path"].read_text())
            self.assertEqual(summary["status"], "failed")
            self.assertIsNone(summary["holdout_score"])
            self.assertEqual(summary["best_params"]["estimator__learning_rate"], 0.001)
            self.assertEqual(summary["training_info"]["epochs_ran"]["finetuning"], 2)
            self.assertEqual(summary["protocols"][0]["error"], "ValueError: Invalid predictions")
            experiments = pd.read_csv(outcome["experiments_path"])
            self.assertTrue(experiments.iloc[0]["selected_for_holdout"])
            self.assertTrue(pd.isna(experiments.iloc[0]["holdout_score"]))

    def test_ranked_csv_keeps_every_candidate_and_nested_parameter_and_refit_failure(self):
        params = [{"estimator__learning_rate": rate, "layers": {"acc": [np.int64(32), 64]}}
                  for rate in (0.001, 0.002, 0.003, 0.004, 0.005)]
        search = SimpleNamespace(
            cv_results_={"params": params, "mean_test_score": [0.7, np.nan, 0.9, 0.8, np.inf]},
            best_index_=3, refit_errors_=[{"candidate": 2, "error": "Cannot refit"}])
        rows = workflow._experiment_rows(search, {"validation_mode": "sequence", "group_column": None},
                                         0.85, True, 1)
        with TemporaryDirectory() as directory:
            csv_path, json_path = workflow._save_results(
                [rows], [{"status": "complete", "best_cv_score": np.float64(0.8),
                          "holdout_score": 0.85, "best_params": params[3], "best_candidate_id": 4}],
                root=Path(directory), experiment_name="test", timestamp="run", target_col="bfrb")
            csv = pd.read_csv(csv_path)
            self.assertEqual(csv["candidate_id"].tolist(), [3, 4, 1, 2, 5])
            self.assertEqual(csv["rank"].tolist(), [1, 2, 3, 4, 5])
            self.assertEqual(json.loads(csv.iloc[0]["param_layers"]), {"acc": [32, 64]})
            self.assertEqual(csv.iloc[0]["refit_error"], "Cannot refit")
            self.assertFalse(csv.iloc[0]["selected_for_holdout"])
            self.assertTrue(csv.iloc[1]["selected_for_holdout"])
            self.assertEqual(csv.iloc[1]["holdout_score"], 0.85)
            self.assertTrue(csv.loc[~csv["selected_for_holdout"], "holdout_score"].isna().all())
            summary = json.loads(json_path.read_text(), parse_constant=lambda value: self.fail(value))
            self.assertEqual(summary["best_params"]["layers"], {"acc": [32, 64]})
            self.assertEqual(summary["best_candidate_id"], 4)
            self.assertEqual(summary["top_cv_candidate_id"], 3)

    def test_json_normalizes_nonfinite_training_values_without_losing_types(self):
        value = {"loss": np.array([0.5, np.nan, np.inf]), "epochs": np.int64(3),
                 "missing": np.ma.masked, "scalar": np.array(0.5), "enabled": np.bool_(True)}
        self.assertEqual(workflow._json_value(value),
                         {"loss": [0.5, None, None], "epochs": 3, "missing": None,
                          "scalar": 0.5, "enabled": True})


if __name__ == "__main__":
    unittest.main()
