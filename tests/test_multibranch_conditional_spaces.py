"""Bayesian compilation preserves every configured parameter choice."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from skopt.space import Categorical, Integer, Real
except ImportError:
    Categorical = Integer = Real = None

from multibranch_utils import (
    conditional_bayesian_spaces, HierarchicalBayesSearchCV,
    prepare_multibranch_bayesian_space, candidate_score,
)
from test_multibranch_conditional_ga import (
    PROPOSAL_FIT_LOG, recording_pipeline, proposal_data,
)


@unittest.skipUnless(Categorical is not None, "scikit-optimize unavailable")
class ConditionalBayesianSpaceTests(unittest.TestCase):
    def setUp(self):
        PROPOSAL_FIT_LOG.clear()

    def test_bert_keeps_all_supplied_backbone_and_encoder_choices(self):
        filters = Integer(16, 64)
        gru = Categorical([False, True], prior=[0.25, 0.75], name="requested_gru")
        supplied = {
            "estimator__temporal_architecture": ["bert"],
            "estimator__use_bigru": gru,
            "estimator__gru_units": Integer(8, 64),
            "estimator__cnn_depth": Integer(1, 4),
            "estimator__branch_cnn_filters": filters,
            "estimator__tof_encoder_type": ["spatial_cnn_gru"],
            "estimator__tof_gru_units": Integer(8, 32),
            "estimator__extractor__tof_modes": ["raw"],
        }
        spaces = conditional_bayesian_spaces(supplied)
        self.assertEqual(len(spaces), 1)
        self.assertEqual(set(spaces[0]), set(supplied))
        self.assertIs(spaces[0]["estimator__use_bigru"], gru)
        self.assertEqual(gru.categories, (False, True))
        self.assertIs(spaces[0]["estimator__branch_cnn_filters"], filters)

    def test_only_explicit_architecture_and_pretraining_source_are_split(self):
        gru = Categorical([False, True])
        supplied = {
            "temporal_architecture": Categorical(["bert", "bigru_attention"]),
            "extractor__pretraining_source": ["baseline", "non_target"],
            "use_bigru": gru, "gru_units": [16, 32], "bert_d_model": [32, 64],
            "branch_num_conv_layers": [1, 2], "branch_cnn_depth": [1, 2],
        }
        spaces = conditional_bayesian_spaces(supplied)
        self.assertEqual(len(spaces), 4)
        self.assertEqual({
            (space["temporal_architecture"].categories[0],
             space["extractor__pretraining_source"].categories[0])
            for space in spaces
        }, {("bert", "baseline"), ("bert", "non_target"),
            ("bigru_attention", "baseline"), ("bigru_attention", "non_target")})
        for space in spaces:
            self.assertEqual(set(space), set(supplied))
            self.assertIs(space["use_bigru"], gru)

    def test_auxiliary_and_pooling_choices_do_not_drop_other_parameters(self):
        supplied = {
            "temporal_architecture": ["bigru_attention"], "use_bigru": [False],
            "use_orientation_aux_head": [False, True],
            "custom_aux_heads": [[], [{"name": "task", "target": "orientation"}]],
            "branch_num_conv_layers": [1, 2], "branch_cnn_depth": [2, 3],
            "orientation_loss_weight": [0.1, 0.3],
            "orientation_num_classes": [2, 5],
            "bert_pooling": ["avg"], "attention_type": ["linear"],
            "attention_units": [16, 32], "use_se": [False], "se_ratio": [4, 8],
        }
        space = conditional_bayesian_spaces(supplied)[0]
        self.assertEqual(set(space), set(supplied))
        self.assertEqual(space["attention_units"].categories, (16, 32))
        self.assertEqual(space["orientation_num_classes"].categories, (2, 5))
        self.assertEqual(json.loads(space["custom_aux_heads"].categories[1]),
                         supplied["custom_aux_heads"][1])

    def test_disabled_tof_does_not_drop_configured_encoder_or_spatial_choices(self):
        supplied = {
            "temporal_architecture": ["bert"], "extractor__tof_modes": [""],
            "tof_encoder_type": ["flatten_1d", "spatial_cnn_gru"],
            "tof_gru_units": [16, 32], "tof_spatial_filters": [4, 8],
        }
        space = conditional_bayesian_spaces(supplied)[0]
        self.assertEqual(set(space), set(supplied))
        self.assertEqual(space["tof_encoder_type"].categories,
                         ("flatten_1d", "spatial_cnn_gru"))

    def test_base_settings_do_not_remove_or_inject_supplied_dimensions(self):
        supplied = {
            "estimator__branch_cnn_depth": [1, 2],
            "estimator__branch_num_conv_layers": [1, 4],
        }
        base = {"estimator__temporal_architecture": "bigru_attention",
                "estimator__use_bigru": False,
                "estimator__use_orientation_aux_head": True}
        compiled = conditional_bayesian_spaces(supplied, base)
        self.assertEqual(len(compiled), 1)
        self.assertEqual(set(compiled[0]), set(supplied))
        only_unused = conditional_bayesian_spaces(
            {"estimator__gru_units": [16, 32]},
            {"estimator__temporal_architecture": "bert"})[0]
        self.assertEqual(set(only_unused), {"estimator__gru_units"})
        self.assertEqual(conditional_bayesian_spaces({}, base), [{}])

    def test_case_alias_choices_remain_separate_and_metadata_is_preserved(self):
        aliases = ["BERT", "bert", "cnn_attention_pool"]
        encoders = Categorical(["cnn", "conv1d"], prior=[0.25, 0.75],
                               transform="identity", name="encoder_choice")
        rate = Real(1e-4, 1e-2, prior="log-uniform", name="learning_rate")
        supplied = {"temporal_architecture": aliases,
                    "branch_encoder_type": encoders,
                    "branch_cnn_depth": (1, 4), "learning_rate": rate}
        compiled = conditional_bayesian_spaces(supplied)
        self.assertEqual(aliases, ["BERT", "bert", "cnn_attention_pool"])
        self.assertEqual(len(compiled), 3)
        self.assertEqual([space["temporal_architecture"].categories[0]
                          for space in compiled], aliases)
        for space in compiled:
            self.assertIs(space["branch_encoder_type"], encoders)
            self.assertIs(space["learning_rate"], rate)
            self.assertEqual(encoders.prior, [0.25, 0.75])
            self.assertEqual(encoders.name, "encoder_choice")
            self.assertEqual(encoders.transform_, "identity")
            self.assertIsInstance(space["branch_cnn_depth"], Integer)

    def test_large_controller_products_preserve_all_declared_dimensions(self):
        heads = [[{"name": f"task_{index}", "target": "orientation"}]
                 for index in range(130)]
        supplied = {
            "temporal_architecture": ["bert"], "custom_aux_heads": heads,
            "bert_pooling": ["attention", "avg", "avg_max", "cls"],
            "attention_type": ["additive", "linear"], "use_se": [False, True],
            "use_orientation_aux_head": [False, True],
            "use_phase_aux_head": [False, True], "branch_cnn_depth": [1, 2],
            "gru_units": [16, 32], "se_ratio": [4, 8],
        }
        compiled = conditional_bayesian_spaces(supplied)[0]
        self.assertEqual(set(compiled), set(supplied))
        self.assertEqual(len(compiled["custom_aux_heads"].categories), 130)
        self.assertEqual(json.loads(compiled["custom_aux_heads"].categories[-1]), heads[-1])

    def test_nonfinite_controller_does_not_remove_other_dimensions(self):
        controller = Real(0, 1)
        supplied = {"temporal_architecture": ["bert"], "use_se": controller,
                    "se_ratio": [4, 8], "gru_units": [16, 32]}
        compiled = conditional_bayesian_spaces(supplied)[0]
        self.assertEqual(set(compiled), set(supplied))
        self.assertIs(compiled["use_se"], controller)

    def fit_search(self, supplied):
        X, y = proposal_data()
        return HierarchicalBayesSearchCV(
            recording_pipeline(),
            search_spaces=prepare_multibranch_bayesian_space(supplied),
            n_iter=2, cv=2, n_jobs=1, scoring=candidate_score, refit=False,
            random_state=23, optimizer_kwargs={"n_initial_points": 2},
        ).fit(X, y)

    def assert_candidates_were_fitted(self, search):
        self.assertEqual(len(PROPOSAL_FIT_LOG), 4)
        self.assertEqual(len(search.cv_results_["params"]), 2)
        for candidate in search.cv_results_["params"]:
            matches = [fitted for fitted in PROPOSAL_FIT_LOG if all(
                fitted[key.removeprefix("estimator__")] == value
                for key, value in candidate.items()
            )]
            self.assertEqual(len(matches), 2, candidate)

    def test_static_depth_choices_reach_bayesian_fits_without_collapsing(self):
        depths = [{"acc": 4, "stft": 1, "cwt": 1},
                  {"acc": 4, "stft": 4, "cwt": 4}]
        search = self.fit_search({
            "estimator__temporal_architecture": ["bigru_attention"],
            "estimator__use_bigru": [False],
            "estimator__branch_num_conv_layers": depths,
        })
        self.assertEqual({json.loads(row["branch_num_conv_layers"])["stft"]
                          for row in PROPOSAL_FIT_LOG}, {1, 4})
        self.assert_candidates_were_fitted(search)

    def test_spectral_fields_reach_bayesian_fits_unchanged(self):
        configs = [[{"name": "custom", "window_seconds": 1.0,
                     "nperseg": count, "acc_modes": "raw", "scaling": "density"}]
                   for count in (8, 32)]
        search = self.fit_search({"estimator__extractor__stft_configs": configs})
        self.assertEqual({json.loads(row["extractor__stft_configs"])[0]["nperseg"]
                          for row in PROPOSAL_FIT_LOG}, {8, 32})
        for row in PROPOSAL_FIT_LOG:
            config = json.loads(row["extractor__stft_configs"])[0]
            self.assertEqual(config["acc_modes"], "raw")
            self.assertEqual(config["scaling"], "density")
        self.assert_candidates_were_fitted(search)

    def test_explicit_unused_switch_values_remain_distinct_bayesian_fits(self):
        search = self.fit_search({
            "estimator__temporal_architecture": ["bert"],
            "estimator__use_bigru": [False, True],
            "estimator__gru_units": [16],
        })
        self.assertEqual({row["use_bigru"] for row in PROPOSAL_FIT_LOG}, {False, True})
        self.assert_candidates_were_fitted(search)


if __name__ == "__main__":
    unittest.main()
