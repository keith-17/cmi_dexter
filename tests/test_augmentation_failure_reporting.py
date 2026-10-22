"""Failure provenance and score reporting without a model-training run."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import base_utils as base
import multibranch_utils as mb
import pretraining_search as workflow
from pretraining_utils import AdvancedMultiBranchSequenceClassifier


def recordings():
    return pd.DataFrame({
        "sequence_id": ["a"] * 6, "sequence_counter": np.arange(6),
        "acc_x": 0.0, "acc_y": 0.0, "acc_z": 9.81,
        "rot_w": 1.0, "rot_x": 0.0, "rot_y": 0.0, "rot_z": 0.0,
    })


def estimator(**overrides):
    params = dict(
        extractor=base.SequenceExtractor(
            acc_modes="raw", linear_acc_modes="raw", linear_acc_mode="baseline",
            rotation_modes="quaternion", tof_modes="", thm_modes="",
            stft_configs=[], cwt_configs=[], chunk_window_size=6, chunk_stride=6,
        ),
        validation_split=0, invalid_candidate_policy="penalize", random_state=17, early_stopping_patience=0,
    )
    params.update(overrides)
    return mb.MultiBranchSequenceClassifier(**params)


class NoTrainingCandidate(AdvancedMultiBranchSequenceClassifier):
    """Exercise real fit/penalty/logging/search wrappers with no training work."""

    def _fit_impl(self, X, y=None, **fit_params):
        self._fit_stage_ = "augmentation"
        if self.dropout > 0:
            raise ValueError("synthetic augmentation failure")
        self.history_ = {}
        return self

    def score(self, X, y, sample_weight=None):
        return float(self.invalid_candidate_score) if self.fit_failed_ else 0.7


class AugmentationFailureReportingTests(unittest.TestCase):
    def fit_without_tensorflow(self, model, frame):
        labels = pd.DataFrame({"sequence_id": ["a"], "bfrb": ["target"]})
        # Only bypass the availability gate. Failures occur before an epoch can run.
        with patch.object(mb, "TENSORFLOW_AVAILABLE", True), patch.object(mb, "tf", None):
            with self.assertWarns(RuntimeWarning):
                model.fit(frame, labels)
        self.assertTrue(model.fit_failed_)
        self.assertEqual(model.score(frame, labels), -1.0)
        return model.fit_failure_

    def test_original_cleaner_failure_retains_cause_and_extraction_stage(self):
        frame = recordings()
        frame.loc[2, ["rot_w", "rot_x", "rot_y", "rot_z"]] = 0.0
        failure = self.fit_without_tensorflow(estimator(), frame)
        self.assertEqual(failure["stage"], "extraction")
        self.assertIn("quaternions nonzero", failure["original_exception"])
        self.assertIn("ValueError", failure["traceback"])
        self.assertIn("_add_linear_acceleration", failure["traceback"])
        self.assertIn("InvalidExtractorParams", failure["traceback"])
        self.assertLessEqual(len(failure["traceback"]), 16500)

    def test_first_invalid_operation_has_candidate_seed_and_bounded_locations(self):
        augmentor = base.SensorAugmentor(
            augmentations=["scaling", "jitter", "time_shift"], prob=1, per_aug_prob=1,
            scaling_sigma=0.0, seed=123, diagnostics=True, diagnostic_max_locations=2,
        )
        model = estimator(augmentor=augmentor)
        before = recordings()

        def invalid_jitter(values, **kwargs):
            return np.full_like(values, np.inf)

        with patch.dict(base._AUGMENTATION_REGISTRY, {"jitter": invalid_jitter}):
            failure = self.fit_without_tensorflow(model, before)
        report = failure["augmentation_diagnostics"]
        self.assertEqual(failure["stage"], "augmentation")
        self.assertEqual(failure["augmentation_seed"], 123)
        self.assertEqual(failure["candidate_id"], clone(model)._candidate_key())
        self.assertEqual(report["candidate_id"], failure["candidate_id"])
        self.assertEqual(report["sequence_id"], "a")
        self.assertEqual(report["seed"], 123)
        self.assertFalse(report["input_invalid"])
        self.assertEqual(report["first_invalid_operation"], "jitter")
        self.assertEqual([op["name"] for op in report["operations"]], ["scaling", "jitter"])
        self.assertLessEqual(len(report["validity"]["nonfinite_locations"]), 2)
        self.assertIn("counter", report["validity"]["nonfinite_locations"][0])
        self.assertIn("traceback", report)
        self.assertFalse(hasattr(model.extractor_, "_training_feature_augmentor"))
        pd.testing.assert_frame_equal(before, recordings())

    def test_invalid_original_orientation_is_identified_before_any_operation(self):
        frame = recordings()
        frame.loc[3, ["rot_w", "rot_x", "rot_y", "rot_z"]] = 0.0
        with TemporaryDirectory() as directory:
            model = estimator(
                augmentor=base.SensorAugmentor(prob=1, per_aug_prob=1, seed=9, diagnostics=True),
                candidate_failure_log=str(Path(directory) / "failures.jsonl"),
            )
            failure = self.fit_without_tensorflow(model, frame)
            saved = json.loads(Path(model.candidate_failure_log).read_text())
        self.assertEqual(saved["candidate_id"], failure["candidate_id"])
        report = saved["augmentation_diagnostics"]
        self.assertTrue(report["input_invalid"])
        self.assertEqual(report["first_invalid_operation"], "input")
        self.assertEqual(report["operations"], [])
        self.assertEqual(report["validity"]["zero_quaternion_count"], 1)
        self.assertEqual(report["validity"]["quaternion_locations"][0]["counter"], "3")

    def test_model_fit_exception_is_training_failure_and_raise_policy_keeps_traceback(self):
        def fail_fit(*args, **kwargs):
            raise RuntimeError("synthetic training failure")

        model = estimator()
        fake = SimpleNamespace(fit=fail_fit)
        with patch.object(mb, "build_multibranch_model", return_value=fake):
            failure = self.fit_without_tensorflow(model, recordings())
        self.assertEqual(failure["stage"], "training")
        self.assertIn("fail_fit", failure["traceback"])
        model.set_params(invalid_candidate_policy="raise")
        with patch.object(model, "_fit_impl", side_effect=ValueError("original exception")):
            with self.assertRaisesRegex(ValueError, "original exception"):
                model.fit(recordings(), None)
        self.assertFalse(model.fit_failed_)
        self.assertIn("original exception", model.fit_failure_["traceback"])

    def test_failed_folds_are_separate_from_measured_scores_without_changing_penalties(self):
        search = SimpleNamespace(
            estimator=Pipeline([("estimator", estimator())]), refit_errors_=[], best_index_=0,
            cv_results_={
                "params": [{}, {}, {}, {}], "mean_test_score": [0.6, -1.0, 0.2, np.nan],
                "split0_test_score": [0.6, -1.0, -1.0, np.nan],
                "split1_test_score": [0.6, -1.0, 1.4, np.nan],
            },
        )
        rows = workflow._experiment_rows(search, {"validation_mode": "sequence", "group_column": None},
                                         None, False, 1)
        self.assertEqual(rows["score_status"].tolist(), ["valid", "failed", "failed", "failed"])
        self.assertEqual(rows["failed_fold_count"].tolist(), [0, 2, 1, 2])
        self.assertEqual(rows.iloc[1]["mean_test_score"], -1.0)
        self.assertEqual(rows.iloc[2]["mean_test_score"], 0.2)
        self.assertTrue(rows.loc[1:, "valid_score"].isna().all())
        self.assertEqual(rows.loc[1:, "failure_penalty"].tolist(), [-1.0] * 3)
        with TemporaryDirectory() as directory:
            _, summary_path = workflow._save_results(
                [rows], [{"status": "failed"}], root=Path(directory),
                experiment_name="test", timestamp="test", target_col="bfrb",
            )
            summary = json.loads(summary_path.read_text())
        self.assertEqual(summary["valid_candidate_count"], 1)
        self.assertEqual(summary["failed_candidate_count"], 3)
        self.assertEqual(summary["top_cv_candidate_id"], 1)

    def test_failure_records_join_candidate_ids_and_remain_bounded(self):
        model = estimator(augmentor=base.SensorAugmentor(seed=31))
        search = SimpleNamespace(
            estimator=Pipeline([("estimator", model)]),
            cv_results_={"params": [{"estimator__dropout": 0.2}]},
        )
        key = clone(model).set_params(dropout=0.2)._candidate_key()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "failures.jsonl"
            record = {"candidate_id": key, "stage": "augmentation", "error": "bad", "params": {}}
            path.write_text("\n".join([json.dumps(record)] * 3))
            records, omitted = workflow._collect_candidate_failures(search, path, 26, limit=2)
        self.assertEqual(len(records), 2)
        self.assertEqual(omitted, 1)
        self.assertEqual(records[0]["candidate_id"], 26)
        self.assertEqual(records[0]["candidate_key"], key)
        self.assertNotIn("params", records[0])

    def test_worker_failure_logging_does_not_enable_expensive_diagnostics(self):
        with TemporaryDirectory() as directory:
            model = estimator(
                augmentor=base.SensorAugmentor(augmentations=["sensor_dropout"], prob=1, per_aug_prob=1),
                candidate_failure_log=str(Path(directory) / "failures.jsonl"),
            )
            with patch.object(base.SensorAugmentor, "_validity", side_effect=AssertionError("unrequested scans")), \
                    patch.object(model, "_build_model", side_effect=RuntimeError("synthetic training failure")):
                failure = self.fit_without_tensorflow(model, recordings())
            self.assertEqual(failure["stage"], "training")
            self.assertIsNone(failure["augmentation_diagnostics"])

    def test_failure_log_write_error_preserves_original_failure_and_penalty(self):
        with TemporaryDirectory() as directory:
            model = estimator(candidate_failure_log=directory)  # Opening a directory as a file must fail.
            with patch.object(model, "_fit_impl", side_effect=RuntimeError("original failure")):
                with self.assertWarns(RuntimeWarning):
                    model.fit(None, None)
            self.assertEqual(model.score(None, None), -1.0)
            self.assertIn("original failure", model.fit_failure_["error"])
            self.assertIn("logging_error", model.fit_failure_)

    def test_runner_embeds_failures_and_cleans_temporary_logs_without_training(self):
        frame = pd.concat([recordings().assign(
            sequence_id=f"s{index}", subject=f"p{index}", gesture="g", bfrb="g", is_target=1,
        ) for index in range(12)], ignore_index=True)
        candidate = NoTrainingCandidate(
            dropout=0, invalid_candidate_policy="penalize", augmentor=base.SensorAugmentor(seed=61),
        )
        search = GridSearchCV(
            Pipeline([("estimator", candidate)]), {"estimator__dropout": [0.0, 0.2]},
            scoring=mb.candidate_score, refit=False, n_jobs=1, error_score="raise",
        )
        with TemporaryDirectory() as directory, \
                patch.object(workflow, "_holdout_results", return_value={"competition_score": 0.8}):
            with self.assertWarns(RuntimeWarning):
                result = workflow.run_validation_searches(
                    search, frame, sensor_columns=[c for c in frame if c.startswith(("acc_", "rot_"))],
                    train_size=0.7, n_splits=1, random_state=42, target_col="bfrb", results_dir=directory,
                    experiment_name="diagnostics", timestamp="test",
                )
            summary = json.loads(result["summary_path"].read_text())
            csv = pd.read_csv(result["experiments_path"])
            self.assertEqual(len(list(Path(directory).rglob("*.*"))), 2)
            self.assertEqual(summary["valid_candidate_count"], 1)
            self.assertEqual(summary["failed_candidate_count"], 1)
            failure = summary["protocols"][0]["candidate_failures"][0]
            self.assertEqual(failure["candidate_id"], 2)
            self.assertEqual(failure["stage"], "augmentation")
            self.assertEqual(failure["augmentation_seed"], 61)
            self.assertIn("synthetic augmentation failure", failure["traceback"])
            self.assertEqual(csv["score_status"].tolist(), ["valid", "failed"])
            self.assertEqual(csv.iloc[1]["failure_stage"], "augmentation")
            returned_search = result["last_run"]["search"]
            self.assertIsNone(returned_search.estimator.named_steps["estimator"].candidate_failure_log)
            self.assertIsNone(returned_search.best_estimator_.named_steps["estimator"].candidate_failure_log)


if __name__ == "__main__":
    unittest.main()
