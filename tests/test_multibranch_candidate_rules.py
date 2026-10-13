"""Dependency and cheap preflight checks for unattended architecture search."""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from multibranch_candidate_rules import canonical_candidate, candidate_error


class CandidateRulesTests(unittest.TestCase):
    def test_bert_keeps_branch_cnn_but_drops_other_backbone_parameters(self):
        params = {"estimator__temporal_architecture": "bert", "estimator__branch_cnn_filters": 48,
                  "estimator__cnn_depth": -5, "estimator__gru_units": -5,
                  "estimator__branch_num_conv_layers": -5, "estimator__use_bigru": True,
                  "estimator__phase_attention_temperature": -1, "estimator__branch_encoder_type": "cnn"}
        before = params.copy()
        canonical = canonical_candidate(params)
        self.assertEqual(canonical["estimator__branch_cnn_filters"], 48)
        self.assertEqual(canonical["estimator__branch_encoder_type"], "conv1d")
        self.assertFalse(canonical["estimator__use_bigru"])
        for name in ("cnn_depth", "gru_units", "branch_num_conv_layers", "phase_attention_temperature"):
            self.assertNotIn("estimator__" + name, canonical)
        self.assertEqual(params, before)
        self.assertIsNone(candidate_error(params))

    def test_both_candidate_and_base_inactive_bad_values_are_ignored(self):
        base = {"estimator__temporal_architecture": "bert", "estimator__cnn_depth": -1,
                "estimator__gru_units": -1, "estimator__branch_filter_mode": "unused"}
        self.assertIsNone(candidate_error({}, base))
        self.assertIn("branch_cnn_depth", candidate_error({"estimator__branch_cnn_depth": 0}, base))

    def test_bert_attention_uses_general_attention_controls(self):
        params = {"temporal_architecture": "bert", "attention_type": "additive", "attention_units": 32}
        self.assertIn("attention_units", canonical_candidate(params))
        self.assertIn("attention_units", candidate_error(dict(params, attention_units=0)))
        self.assertNotIn("attention_units", canonical_candidate(dict(params, bert_pooling="avg")))
        self.assertIsNone(candidate_error(dict(params, bert_pooling="avg", attention_units=0)))
        self.assertNotIn("attention_units", canonical_candidate(dict(params, attention_type="linear")))

    def test_label_cache_reuses_column_check_only_for_same_table(self):
        y = pd.DataFrame({"sequence_id": [1, 2], "orientation": ["a", "b"]})
        cache = {}
        params = {"use_orientation_aux_head": True}
        self.assertIsNone(candidate_error(params, y=y, label_cache=cache))
        with patch.object(y, "groupby", side_effect=AssertionError("duplicate validation")):
            self.assertIsNone(candidate_error(params, y=y, label_cache=cache))
        changed = y.drop(columns="orientation")
        self.assertIn("orientation", candidate_error(params, y=changed, label_cache=cache))

    def test_aux_toggle_changes_legacy_branch_parameter_dependencies(self):
        params = {"branch_num_conv_layers": 3, "branch_cnn_depth": 4, "branch_dropout": .2,
                  "pooling_mode": "mean", "attention_units": 32}
        legacy = canonical_candidate(params)
        self.assertIn("branch_num_conv_layers", legacy)
        self.assertNotIn("branch_cnn_depth", legacy)
        temporal = canonical_candidate(params, {"use_orientation_aux_head": True})
        self.assertNotIn("branch_num_conv_layers", temporal)
        self.assertNotIn("pooling_mode", temporal)
        self.assertIn("branch_cnn_depth", temporal)
        self.assertIn("attention_units", temporal)

    def test_phase_aux_keeps_class_count_without_phase_attention_internals(self):
        params = {"temporal_architecture": "bert", "use_phase_aux_head": True, "n_phases": 4,
                  "phase_attention_temperature": -1, "phase_attention_units": -1, "phase_fusion_mode": "unused"}
        canonical = canonical_candidate(params)
        self.assertEqual(canonical["n_phases"], 4)
        self.assertNotIn("phase_attention_temperature", canonical)
        self.assertIsNone(candidate_error(params, phase_labels_available=True))
        self.assertIn("Phase supervision", candidate_error(params))

    def test_bert_constraints_are_active_only_for_bert(self):
        for params, reason in [({"bert_d_model": 10, "bert_num_heads": 4}, "divisible"),
                               ({"bert_pooling": "cls", "bert_use_cls_token": False}, "requires"),
                               ({"bert_num_layers": 0}, "positive integer")]:
            with self.subTest(params=params):
                self.assertIn(reason, candidate_error(params, {"temporal_architecture": "bert"}))
                self.assertIsNone(candidate_error(params, {"temporal_architecture": "phase_attention"}))

    def test_spatial_tof_raw_requirement_and_disabled_modality(self):
        params = {"temporal_architecture": "bert", "tof_encoder_type": "spatial_cnn_gru"}
        self.assertIn("tof_modes='raw'", candidate_error(params))
        self.assertIsNone(candidate_error(params, {"extractor__tof_modes": "raw"}))
        self.assertIn("tof_modes='raw'", candidate_error(params, {"extractor__tof_modes": "raw|sensor_stats"}))
        disabled = canonical_candidate(params, {"extractor__tof_modes": ""})
        self.assertNotIn("tof_encoder_type", disabled)
        self.assertIsNone(candidate_error(params, {"extractor__tof_modes": "", "tof_gru_units": -1}))

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

    def test_inferred_orientation_width_and_linear_attention_width_do_not_search(self):
        params = {"use_orientation_aux_head": True, "orientation_num_classes": 12}
        self.assertNotIn("orientation_num_classes", canonical_candidate(params))
        params = {"temporal_architecture": "cnn_attention_pool", "post_fusion_attention_type": "linear",
                  "post_fusion_attention_units": 0}
        self.assertNotIn("post_fusion_attention_units", canonical_candidate(params))
        self.assertIsNone(candidate_error(params))

    def test_incompatible_candidate_is_penalized_before_extractor_or_tensorflow(self):
        import multibranch_attention_sol as mba
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
