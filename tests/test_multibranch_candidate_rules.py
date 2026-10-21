"""Dependency and cheap preflight checks for unattended architecture search."""
import json
from copy import deepcopy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from multibranch_utils import candidate_error


class CandidateRulesTests(unittest.TestCase):
    def test_bert_validates_active_controls_without_rewriting_candidate(self):
        params = {"estimator__temporal_architecture": "bert", "estimator__branch_cnn_filters": 48,
                  "estimator__cnn_depth": -5, "estimator__gru_units": -5,
                  "estimator__branch_num_conv_layers": -5, "estimator__use_bigru": True,
                  "estimator__phase_attention_temperature": -1, "estimator__branch_encoder_type": "cnn"}
        before = deepcopy(params)
        self.assertIsNone(candidate_error(params))
        self.assertEqual(params, before)
        self.assertEqual(params["estimator__branch_encoder_type"], "cnn")
        self.assertTrue(params["estimator__use_bigru"])

    def test_candidate_and_base_inactive_values_are_preserved_and_ignored(self):
        base = {"estimator__temporal_architecture": "bert", "estimator__cnn_depth": -1,
                "estimator__gru_units": -1, "estimator__branch_filter_mode": "unused"}
        params = {"estimator__branch_cnn_depth": 0}
        before_base, before_params = deepcopy(base), deepcopy(params)
        self.assertIsNone(candidate_error({}, base))
        self.assertIn("branch_cnn_depth", candidate_error(params, base))
        self.assertEqual((base, params), (before_base, before_params))

    def test_bert_attention_validates_width_only_when_consumed(self):
        params = {"temporal_architecture": "bert", "attention_type": "additive", "attention_units": 0}
        for extra, invalid in (({}, True), ({"bert_pooling": "avg"}, False), ({"attention_type": "linear"}, False)):
            candidate = dict(params, **extra)
            before = deepcopy(candidate)
            error = candidate_error(candidate)
            self.assertIn("attention_units", error) if invalid else self.assertIsNone(error)
            self.assertEqual(candidate, before)

    def test_label_cache_reuses_column_check_only_for_same_table(self):
        y = pd.DataFrame({"sequence_id": [1, 2], "orientation": ["a", "b"]})
        cache = {}
        params = {"use_orientation_aux_head": True}
        self.assertIsNone(candidate_error(params, y=y, label_cache=cache))
        with patch.object(y, "groupby", side_effect=AssertionError("duplicate validation")):
            self.assertIsNone(candidate_error(params, y=y, label_cache=cache))
        changed = y.drop(columns="orientation")
        self.assertIn("orientation", candidate_error(params, y=changed, label_cache=cache))

    def test_aux_toggle_changes_validation_without_removing_branch_parameters(self):
        legacy = {"branch_num_conv_layers": 0, "branch_cnn_depth": 4, "branch_dropout": .2,
                  "pooling_mode": "mean", "attention_units": 32}
        before = deepcopy(legacy)
        self.assertIn("branch_num_conv_layers", candidate_error(legacy))
        self.assertIsNone(candidate_error(legacy, {"use_orientation_aux_head": True}))
        self.assertEqual(legacy, before)
        temporal = dict(legacy, branch_num_conv_layers=3, branch_cnn_depth=0)
        self.assertIsNone(candidate_error(temporal))
        self.assertIn("branch_cnn_depth", candidate_error(temporal, {"use_orientation_aux_head": True}))

    def test_phase_aux_validates_supervision_without_rewriting_phase_controls(self):
        params = {"temporal_architecture": "bert", "use_phase_aux_head": True, "n_phases": 4,
                  "phase_attention_temperature": -1, "phase_attention_units": -1, "phase_fusion_mode": "unused"}
        before = deepcopy(params)
        self.assertIsNone(candidate_error(params, phase_labels_available=True))
        self.assertIn("Phase supervision", candidate_error(params))
        self.assertEqual(params, before)

    def test_bert_constraints_are_active_only_for_bert(self):
        for params, reason in [({"bert_d_model": 10, "bert_num_heads": 4}, "divisible"),
                               ({"bert_pooling": "cls", "bert_use_cls_token": False}, "requires"),
                               ({"bert_num_layers": 0}, "positive integer")]:
            with self.subTest(params=params):
                self.assertIn(reason, candidate_error(params, {"temporal_architecture": "bert"}))
                self.assertIsNone(candidate_error(params, {"temporal_architecture": "phase_attention"}))

    def test_spatial_tof_raw_requirement_and_disabled_modality(self):
        params = {"temporal_architecture": "bert", "tof_encoder_type": "spatial_cnn_gru"}
        before = deepcopy(params)
        self.assertIn("tof_modes='raw'", candidate_error(params))
        self.assertIsNone(candidate_error(params, {"extractor__tof_modes": "raw"}))
        self.assertIn("tof_modes='raw'", candidate_error(params, {"extractor__tof_modes": "raw|sensor_stats"}))
        self.assertIsNone(candidate_error(params, {"extractor__tof_modes": "", "tof_gru_units": -1}))
        self.assertEqual(params, before)

    def test_head_targets_windows_and_regression_are_checked(self):
        heads = [{"name": "short_head", "target": "numeric", "task": "regression", "window_size": 50}]
        y = pd.DataFrame({"sequence_id": [1, 2], "numeric": [2., 3.]})
        params = {"custom_aux_heads": json.dumps(heads), "extractor__chunk_window_size": 40}
        self.assertIn("exceeds chunk length", candidate_error(params, y=y))
        params["extractor__chunk_window_size"] = 50
        self.assertIsNone(candidate_error(params, y=y))
        self.assertIn("numeric", candidate_error(params, y=y.drop(columns="numeric")))
        self.assertIn("must be numeric", candidate_error(params, y=y.assign(numeric=["a", "b"])))
        self.assertEqual(heads[0]["window_size"], 50)
        self.assertNotIn("dropout", heads[0])

    def test_orientation_requires_column_and_sequence_constant_labels(self):
        params = {"use_orientation_aux_head": True}
        y = pd.DataFrame({"sequence_id": [1, 1], "orientation": ["a", "b"]})
        self.assertIn("constant within", candidate_error(params, y=y))
        self.assertIn("orientation", candidate_error(params, y=y.drop(columns="orientation")))
        self.assertIsNone(candidate_error(params, y=y.assign(orientation="a").set_index("sequence_id")))

    def test_active_dropout_and_dimensions_and_disabled_losses(self):
        self.assertIn("branch_dropout", candidate_error({"temporal_architecture": "bert", "branch_dropout": 1.0}))
        self.assertIn("cnn_depth", candidate_error({"temporal_architecture": "phase_attention", "cnn_depth": 0}))
        self.assertIsNone(candidate_error({"orientation_loss_weight": -1, "phase_loss_weight": -1}))
        self.assertIn("orientation_loss_weight", candidate_error({"use_orientation_aux_head": True, "orientation_loss_weight": -1}))

    def test_inferred_orientation_and_linear_attention_widths_are_not_validated(self):
        params = {"use_orientation_aux_head": True, "orientation_num_classes": 0}
        before = deepcopy(params)
        self.assertIsNone(candidate_error(params))
        self.assertEqual(params, before)
        params = {"temporal_architecture": "cnn_attention_pool", "post_fusion_attention_type": "linear",
                  "post_fusion_attention_units": 0}
        before = deepcopy(params)
        self.assertIsNone(candidate_error(params))
        self.assertEqual(params, before)

    def test_incompatible_candidate_is_penalized_before_extractor_or_tensorflow(self):
        import multibranch_utils as mba
        estimator = mba.MultiBranchSequenceClassifier(temporal_architecture="bert", bert_d_model=10,
            bert_num_heads=4, invalid_candidate_policy="penalize")
        y = pd.DataFrame({"sequence_id": [1], "bfrb": ["a"]})
        with patch.object(mba, "TENSORFLOW_AVAILABLE", False), patch.object(mba, "clone") as clone:
            estimator.fit(None, y)
        clone.assert_not_called()
        self.assertTrue(estimator.fit_failed_)
        self.assertIn("divisible", estimator.fit_error_)
        self.assertEqual(estimator.score(None, y), -1.)


if __name__ == "__main__":
    unittest.main()
