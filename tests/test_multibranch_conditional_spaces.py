"""Conditional-space compilation retains every potentially active model knob."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from skopt.space import Categorical, Integer, Real
except ImportError:
    Categorical = Integer = Real = None

from multibranch_utils import conditional_bayesian_spaces


@unittest.skipUnless(Categorical is not None, "scikit-optimize unavailable")
class ConditionalBayesianSpaceTests(unittest.TestCase):
    def test_bert_removes_shared_gru_and_keeps_modality_encoders(self):
        filters = Integer(16, 64)
        spaces = conditional_bayesian_spaces({
            "estimator__temporal_architecture": ["bert"],
            "estimator__use_bigru": [False, True],
            "estimator__gru_units": Integer(8, 64),
            "estimator__cnn_depth": Integer(1, 4),
            "estimator__branch_cnn_filters": filters,
            "estimator__tof_encoder_type": ["spatial_cnn_gru"],
            "estimator__tof_gru_units": Integer(8, 32),
            "estimator__extractor__tof_modes": ["raw"],
        })
        self.assertEqual(len(spaces), 1)
        space = spaces[0]
        self.assertNotIn("estimator__gru_units", space)
        self.assertNotIn("estimator__cnn_depth", space)
        self.assertIs(space["estimator__branch_cnn_filters"], filters)
        self.assertIn("estimator__tof_gru_units", space)
        self.assertEqual(space["estimator__use_bigru"].categories, (False,))

    def test_architecture_and_gru_are_split_before_dependencies(self):
        spaces = conditional_bayesian_spaces({
            "temporal_architecture": Categorical(["bert", "bigru_attention"]),
            "use_bigru": [False, True],
            "gru_units": [16, 32], "bert_d_model": [32, 64],
            "branch_num_conv_layers": [1, 2], "branch_cnn_depth": [1, 2],
        })
        self.assertEqual(len(spaces), 3)
        for space in spaces:
            architecture = space["temporal_architecture"].categories[0]
            use_bigru = space["use_bigru"].categories[0]
            with self.subTest(architecture=architecture, use_bigru=use_bigru):
                self.assertEqual("gru_units" in space, architecture == "bigru_attention" and use_bigru)
                self.assertEqual("bert_d_model" in space, architecture == "bert")
                self.assertEqual("branch_num_conv_layers" in space,
                                 architecture == "bigru_attention" and not use_bigru)
                self.assertEqual("branch_cnn_depth" in space,
                                 architecture == "bert" or use_bigru)

    def test_auxiliary_switch_preserves_both_legacy_and_temporal_choices(self):
        spaces = conditional_bayesian_spaces({
            "temporal_architecture": "bigru_attention", "use_bigru": False,
            "use_orientation_aux_head": [False, True],
            "branch_num_conv_layers": [1, 2], "branch_cnn_depth": [2, 3],
            "orientation_loss_weight": [0.1, 0.3],
        })
        space = spaces[0]
        self.assertIn("branch_num_conv_layers", space)
        self.assertIn("branch_cnn_depth", space)
        self.assertEqual(space["use_orientation_aux_head"].categories, (False, True))
        self.assertEqual(space["orientation_loss_weight"].categories, (0.1, 0.3))

    def test_custom_auxiliary_heads_can_activate_temporal_branch(self):
        spaces = conditional_bayesian_spaces({
            "temporal_architecture": "bigru_attention", "use_bigru": False,
            "custom_aux_heads": [[], [{"name": "task", "target": "orientation"}]],
            "branch_num_conv_layers": [1], "branch_cnn_depth": [2],
        })
        self.assertIn("branch_num_conv_layers", spaces[0])
        self.assertIn("branch_cnn_depth", spaces[0])

    def test_pooling_and_linear_attention_preserve_only_possible_widths(self):
        base_space = {"temporal_architecture": "bert", "attention_units": [16, 32]}
        average = conditional_bayesian_spaces(dict(base_space, bert_pooling=["avg"]))[0]
        mixed = conditional_bayesian_spaces(dict(base_space, bert_pooling=["avg", "attention"]))[0]
        linear = conditional_bayesian_spaces(dict(base_space, attention_type=["linear"]))[0]
        self.assertNotIn("attention_units", average)
        self.assertIn("attention_units", mixed)
        self.assertNotIn("attention_units", linear)

    def test_spatial_tof_choice_retains_gru_while_disabled_tof_drops_it(self):
        space = {"temporal_architecture": "bert", "tof_encoder_type": ["flatten_1d", "spatial_cnn_gru"],
                 "tof_gru_units": [16, 32], "extractor__tof_modes": ["", "raw"]}
        self.assertIn("tof_gru_units", conditional_bayesian_spaces(space)[0])
        disabled = conditional_bayesian_spaces(dict(space, extractor__tof_modes=[""]))[0]
        self.assertNotIn("tof_gru_units", disabled)
        self.assertNotIn("tof_encoder_type", disabled)

    def test_base_pipeline_settings_control_absent_architecture_and_auxiliary_switch(self):
        spaces = conditional_bayesian_spaces({
            "estimator__branch_cnn_depth": [1, 2],
            "estimator__branch_num_conv_layers": [1, 2],
        }, {"estimator__temporal_architecture": "bigru_attention",
            "estimator__use_bigru": False, "estimator__use_orientation_aux_head": True})
        self.assertIn("estimator__branch_cnn_depth", spaces[0])
        self.assertNotIn("estimator__branch_num_conv_layers", spaces[0])

    def test_all_inactive_space_keeps_a_fixed_dimension_with_correct_prefix(self):
        space = conditional_bayesian_spaces({"estimator__gru_units": [16, 32]},
                                           {"estimator__temporal_architecture": "bert"})[0]
        self.assertEqual(set(space), {"estimator__temporal_architecture"})
        self.assertEqual(space["estimator__temporal_architecture"].categories, ("bert",))
        direct = conditional_bayesian_spaces({}, {"temporal_architecture": "bert"})[0]
        self.assertEqual(set(direct), {"temporal_architecture"})

    def test_input_choices_and_dimension_metadata_are_not_mutated(self):
        aliases = ["BERT", "bert", "cnn_attention_pool"]
        encoders = Categorical(["cnn", "conv1d"], prior=[0.25, 0.75])
        space = {"temporal_architecture": aliases, "branch_encoder_type": encoders,
                 "branch_cnn_depth": (1, 4), "learning_rate": Real(1e-4, 1e-2, prior="log-uniform")}
        compiled = conditional_bayesian_spaces(space)
        self.assertEqual(aliases, ["BERT", "bert", "cnn_attention_pool"])
        self.assertEqual(encoders.categories, ("cnn", "conv1d"))
        self.assertEqual(encoders.prior, [0.25, 0.75])
        self.assertEqual(len(compiled), 2)
        self.assertIsInstance(compiled[0]["branch_cnn_depth"], Integer)
        self.assertIs(compiled[0]["branch_encoder_type"], encoders)

    def test_large_controller_products_use_conservative_bounded_pruning(self):
        heads = [[{"name": f"task_{index}", "target": "orientation"}] for index in range(130)]
        space = {"temporal_architecture": "bert", "custom_aux_heads": heads,
                 "bert_pooling": ["attention", "avg", "avg_max", "cls"],
                 "attention_type": ["additive", "linear"], "use_se": [False, True],
                 "use_orientation_aux_head": [False, True], "use_phase_aux_head": [False, True],
                 "branch_cnn_depth": [1, 2], "gru_units": [16, 32], "se_ratio": [4, 8]}
        compiled = conditional_bayesian_spaces(space)[0]
        self.assertIn("branch_cnn_depth", compiled)
        self.assertIn("se_ratio", compiled)
        self.assertNotIn("gru_units", compiled)
        self.assertEqual(len(compiled["custom_aux_heads"].categories), 130)
        self.assertEqual(json.loads(compiled["custom_aux_heads"].categories[-1]), heads[-1])

    def test_nonfinite_controller_is_conservatively_retained(self):
        space = conditional_bayesian_spaces({"temporal_architecture": "bert", "use_se": Real(0, 1),
                                            "se_ratio": [4, 8], "gru_units": [16, 32]})[0]
        self.assertIn("se_ratio", space)
        self.assertNotIn("gru_units", space)


if __name__ == "__main__":
    unittest.main()
