"""Non-architecture no-op search knobs and their active counterexamples."""
import ast
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from base_utils import (SequenceExtractor, SensorAugmentor, inactive_sequence_parameters,
                        prepare_random_forest_bayesian_spaces)
from multibranch_utils import canonical_candidate, conditional_bayesian_spaces, HierarchicalGASearchCV
from test_multibranch_conditional_ga import CountingInt
from test_multibranch_hierarchical_search import dry_pipeline, toy_data


class TrainingDependencies(unittest.TestCase):
    def test_optimizer_and_loss_specific_parameters(self):
        for optimizer, active in (("adam", set()), ("adamw", {"weight_decay"}), ("sgd", {"momentum"})):
            result = canonical_candidate(dict(optimizer_name=optimizer, momentum=0.7, weight_decay=0.01))
            self.assertEqual(set(result) & {"momentum", "weight_decay"}, active)
        self.assertNotIn("label_smoothing", canonical_candidate(dict(loss_name="focal", label_smoothing=0.2)))
        self.assertIn("label_smoothing", canonical_candidate(dict(loss_name="sparse_cce", label_smoothing=0.2)))

    def test_scheduler_family_and_validation_dependencies(self):
        knobs = dict(lr_scheduler_type="plateau", lr_factor=0.2, lr_patience=3,
                     min_lr=1e-5, cosine_t_max=10, warmup_epochs=2)
        self.assertFalse(set(knobs) & set(canonical_candidate(knobs)))
        plateau = canonical_candidate(dict(knobs, use_lr_scheduler=True))
        self.assertIn("lr_factor", plateau)
        self.assertNotIn("warmup_epochs", plateau)
        cosine = canonical_candidate(dict(knobs, use_lr_scheduler=True, lr_scheduler_type="cosine"))
        self.assertIn("warmup_epochs", cosine)
        self.assertNotIn("lr_factor", cosine)
        no_validation = canonical_candidate(dict(knobs, use_lr_scheduler=True, validation_split=0))
        self.assertNotIn("min_lr", no_validation)
        self.assertIn("min_lr", canonical_candidate(dict(knobs, use_lr_scheduler=True,
                          lr_scheduler_type="cosine", validation_split=0)))

    def test_disabled_augmentation_and_selected_transform_controls(self):
        aug = SensorAugmentor(prob=0.5, per_aug_prob=0.5)
        base = {"augmentor": aug}
        disabled = canonical_candidate({"augmentor__prob": 0, "augmentor__noise_std": 8,
                                        "augmentor__per_aug_prob": 0.8}, base)
        self.assertNotIn("augmentor__noise_std", disabled)
        self.assertEqual(disabled["augmentor__per_aug_prob"], 0)
        jitter = canonical_candidate({"augmentor__augmentations": '["jitter"]',
                                      "augmentor__noise_std": 8, "augmentor__jitter_sigma": 0.2}, base)
        self.assertNotIn("augmentor__noise_std", jitter)
        self.assertIn("augmentor__jitter_sigma", jitter)
        empty = canonical_candidate({"augmentor__augmentations": []}, base)
        self.assertEqual(empty["augmentor__augmentations"], [])

    def test_temporal_spatial_only_branch_drops_generic_cnn_but_keeps_spatial_filters(self):
        params = dict(temporal_architecture="phase_attention", tof_encoder_type="spatial_cnn",
                      extractor__acc_modes="", extractor__rotation_modes="", extractor__thm_modes="",
                      extractor__tof_modes="raw", extractor__stft_configs=None, extractor__cwt_configs=None,
                      static_branch_units=20, branch_cnn_filters=32, use_se=True, se_ratio=8,
                      use_residual=True, use_batchnorm=True, cnn_filters=64, tof_spatial_filters=16)
        active = canonical_candidate(params)
        for name in ("static_branch_units", "branch_cnn_filters", "use_se", "se_ratio"):
            self.assertNotIn(name, active)
        for name in ("use_residual", "use_batchnorm", "cnn_filters", "tof_spatial_filters"):
            self.assertIn(name, active)
        active = canonical_candidate(dict(params, extractor__acc_modes="raw"))
        self.assertIn("branch_cnn_filters", active)
        self.assertIn("use_se", active)

    def test_unknown_extractor_subclasses_are_not_pruned_as_native(self):
        class Custom(SequenceExtractor):
            pass
        params = {"extractor": Custom(), "extractor__use_dead_reckoning": True}
        self.assertIn("extractor__use_dead_reckoning", canonical_candidate(params))

    def test_legacy_branch_dicts_drop_disabled_sensors_and_unused_filter_tail(self):
        common = dict(extractor__rotation_modes="", extractor__tof_modes="", extractor__thm_modes="",
                      extractor__stft_configs=None, extractor__cwt_configs=None,
                      branch_num_conv_layers=2, branch_filter_mode="custom")
        first = canonical_candidate(dict(common, branch_filters={"acc": [16, 32, 999], "tof": 888}))
        second = canonical_candidate(dict(common, branch_filters={"acc": [16, 32, 1], "tof": -5}))
        self.assertEqual(first, second)
        self.assertEqual(first["branch_filters"], {"acc": [16, 32]})
        scalar = canonical_candidate(dict(common, branch_filters={"acc": 16}))
        repeated = canonical_candidate(dict(common, branch_filters={"acc": [16, 16]}))
        self.assertEqual(scalar, repeated)

    def test_missing_branch_entry_keeps_its_real_builder_fallback(self):
        active = canonical_candidate(dict(branch_filters={"tof": 16}, extractor__tof_modes=""))
        self.assertEqual(active["branch_filters"]["acc"], [32])


class ExtractorDependencies(unittest.TestCase):
    def test_filter_resampling_kalman_and_smoothing_guards(self):
        inactive = inactive_sequence_parameters({"acc_modes": "raw", "linear_acc_mode": None})
        for name in ("kalman_process_noise", "dead_reckoning_detrend", "imu_target_sampling_rate",
                     "problematic_features_threshold", "smooth_alpha", "window_size", "gravity_magnitude"):
            self.assertIn(name, inactive)
        active = inactive_sequence_parameters({"motion_filter_mode": "kalman", "resample_modalities": True,
                                              "filter_problematic_sequences": True, "acc_modes": "smoothed"})
        for name in ("kalman_process_noise", "imu_target_sampling_rate", "problematic_features_threshold", "smooth_alpha", "window_size"):
            self.assertNotIn(name, active)
        self.assertIn("window_size", inactive_sequence_parameters({"acc_modes": "smoothed", "smooth_alpha": 0.3}))

    def test_chunk_frame_and_stride_guards(self):
        frame = inactive_sequence_parameters({"output_format": "frame"})
        for name in ("maxlen", "chunk_window_size", "chunk_stride", "padding_value", "add_global_context"):
            self.assertIn(name, frame)
        self.assertNotIn("frame_stats", frame)
        chunks = inactive_sequence_parameters({"use_chunk_stride_ratio": True, "sequence_crop_mode": "none"})
        for name in ("frame_stats", "chunk_stride", "window_cap_policy", "maxlen"):
            self.assertIn(name, chunks)
        self.assertNotIn("chunk_stride_ratio", chunks)
        self.assertNotIn("window_cap_policy", inactive_sequence_parameters({"max_windows_per_sequence": 2}))

    def test_spectral_disabled_overridden_and_partially_overridden_defaults(self):
        disabled = inactive_sequence_parameters({"stft_configs": None, "cwt_configs": []})
        self.assertTrue({"stft_nperseg", "cwt_wavelet", "cwt_n_scales"} <= disabled)
        overridden = inactive_sequence_parameters({"stft_configs": [{"nperseg": 8, "window_type": "hann"}],
                                                   "cwt_configs": [{"widths": [2, 3, 4]}]})
        self.assertTrue({"stft_nperseg", "stft_window_type", "cwt_n_scales"} <= overridden)
        partial = inactive_sequence_parameters({"stft_configs": [{"nperseg": 8}, {}]})
        self.assertNotIn("stft_nperseg", partial)
        self.assertIn("stft_nperseg", inactive_sequence_parameters({"stft_window_seconds": 0.5}))

    def test_spectral_unused_internal_fields_and_aliases_share_a_candidate(self):
        first = {"extractor__stft_configs": [{"acc_modes": "raw", "window_seconds": 0.5,
                                              "nperseg": 8, "scaling": "density"}],
                 "extractor__cwt_configs": [{"acc_modes": "raw", "widths": [2, 3, 4],
                                             "n_scales": 20, "min_scale": 3}]}
        second = {"extractor__stft_configs": [{"acc_modes": "jerk", "window_seconds": 0.5,
                                               "nperseg": 32, "scaling": "psd"}],
                  "extractor__cwt_configs": [{"acc_modes": "", "widths": [2, 3, 4],
                                              "n_scales": 40, "min_scale": 4}]}
        before = json.loads(json.dumps(first))
        self.assertEqual(canonical_candidate(first), canonical_candidate(second))
        self.assertEqual(first, before)

    def test_dead_reckoning_unconsumed_vs_explicit_spectral_consumer(self):
        x = pd.DataFrame({"sequence_id": ["a"] * 12, "sequence_counter": range(12),
                          "acc_x": np.linspace(1, 4, 12)})
        options = dict(stft_configs=None, cwt_configs=None, chunk_window_size=4, rotation_modes="")
        off = SequenceExtractor(**options).fit_transform(x)
        on = SequenceExtractor(**options, use_dead_reckoning=True, dead_reckoning_detrend=True)
        np.testing.assert_array_equal(off["X"], on.fit_transform(x)["X"])
        self.assertFalse(on.motion_filter.use_dead_reckoning)
        config = [{"signal_columns": ["acc_x_raw_dead_reckoning_velocity"], "nperseg": 4}]
        explicit = SequenceExtractor(stft_configs=config, cwt_configs=None, use_dead_reckoning=True, rotation_modes="")
        self.assertTrue(explicit.motion_filter.use_dead_reckoning)
        self.assertNotIn("use_dead_reckoning", inactive_sequence_parameters(explicit.get_params()))
        self.assertTrue(any("dead_reckoning_velocity" in name for name in explicit.fit_transform(x)["feature_names"]))

    def test_frame_transform_does_not_build_unused_global_context(self):
        x = pd.DataFrame({"sequence_id": [1, 1], "sequence_counter": [0, 1], "acc_x": [1., 2.]})
        model = SequenceExtractor(output_format="frame", add_global_context=True, stft_configs=None,
                                  cwt_configs=None, rotation_modes="").fit(x)
        with patch.object(model, "_append_global_context", side_effect=AssertionError("unused context")):
            self.assertEqual(len(model.transform_frame(x)), 1)

    def test_rf_notebook_no_longer_searches_frame_ignored_chunk_settings(self):
        path = Path(__file__).resolve().parents[1] / "notebooks/random_forest.ipynb"
        ignored = inactive_sequence_parameters({"output_format": "frame"})
        for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]:
            source = "".join(cell.get("source", []))
            if "GRID_PARAM_SPACE = {" not in source:
                continue
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("estimator__extractor__"):
                    self.assertFalse(node.value.split("__")[-1] in ignored & {
                        "maxlen", "chunk_window_size", "chunk_stride", "padding_value", "add_global_context"})
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    self.assertFalse(node.value.startswith("augmentor__"))


class SearchIntegration(unittest.TestCase):
    def test_switching_optimizer_and_scheduler_initializes_new_dependent_genes(self):
        momentum, warmup = CountingInt(1, 1), CountingInt(2, 2)
        search = HierarchicalGASearchCV(dry_pipeline(), param_grid={}, mutation_probability=1.0)
        _, y = toy_data()
        search._initialise_rules(y, {})
        space = {"estimator__optimizer_name": ["sgd"], "estimator__momentum": momentum,
                 "estimator__use_lr_scheduler": [True], "estimator__lr_scheduler_type": ["cosine"],
                 "estimator__warmup_epochs": warmup}
        active = search._mutate({"estimator__optimizer_name": "adam", "estimator__use_lr_scheduler": False},
                                space, np.random.default_rng(5))
        self.assertEqual((momentum.draws, warmup.draws), (1, 1))
        self.assertEqual(active["estimator__warmup_epochs"], 2)

    def test_tof_encoder_is_drawn_before_squeeze_excitation_eligibility(self):
        pipeline = dry_pipeline().set_params(estimator__temporal_architecture="bert",
            estimator__tof_encoder_type="spatial_cnn", estimator__extractor__tof_modes="raw",
            estimator__extractor__acc_modes="", estimator__extractor__rotation_modes="",
            estimator__extractor__thm_modes="", estimator__extractor__stft_configs=None,
            estimator__extractor__cwt_configs=None)
        search = HierarchicalGASearchCV(pipeline, param_grid={})
        _, y = toy_data()
        search._initialise_rules(y, {})
        proposal = search._sample_individual({"estimator__tof_encoder_type": ["flatten_1d"],
                                              "estimator__use_se": [True]}, np.random.default_rng(5))
        self.assertTrue(proposal["estimator__use_se"])

    def test_ga_does_not_draw_non_sgd_momentum_or_disabled_scheduler_rates(self):
        momentum, warmup = CountingInt(1, 8), CountingInt(1, 8)
        search = HierarchicalGASearchCV(dry_pipeline(), param_grid={
            "estimator__optimizer_name": ["adam"], "estimator__momentum": momentum,
            "estimator__use_lr_scheduler": [False], "estimator__warmup_epochs": warmup,
        }, population_size=2, tournament_size=2, generations=0, refit=False, cv=2, verbose=0)
        x, y = toy_data()
        search.fit(x, y)
        self.assertEqual((momentum.draws, warmup.draws), (0, 0))

    def test_bayes_keeps_dependencies_when_their_switch_is_searched(self):
        try:
            from skopt.space import Categorical
        except ImportError:
            self.skipTest("scikit-optimize unavailable")
        space = dict(temporal_architecture=Categorical(["bert"]),
                     optimizer_name=Categorical(["adam", "sgd"]), momentum=Categorical([0.1, 0.9]),
                     use_lr_scheduler=Categorical([False, True]), warmup_epochs=Categorical([0, 2]),
                     lr_scheduler_type=Categorical(["cosine"]))
        compiled = conditional_bayesian_spaces(space)[0]
        self.assertIn("momentum", compiled)
        self.assertIn("warmup_epochs", compiled)
        compiled = conditional_bayesian_spaces(dict(space, optimizer_name=["adam"], use_lr_scheduler=[False]))[0]
        self.assertNotIn("momentum", compiled)
        self.assertNotIn("warmup_epochs", compiled)


class ForestDependencies(unittest.TestCase):
    def test_integer_split_leaf_pairs_collapse_and_share_the_original_budget(self):
        try:
            from skopt.space import Categorical
        except ImportError:
            self.skipTest("scikit-optimize unavailable")
        space = {"estimator__estimator__min_samples_leaf": Categorical([10, 30, 50]),
                 "estimator__estimator__min_samples_split": Categorical([10, 30, 50])}
        compiled = prepare_random_forest_bayesian_spaces(space, 37)
        self.assertEqual(sum(budget for _, budget in compiled), 37)
        self.assertEqual([len(s["estimator__estimator__min_samples_split"].categories) for s, _ in compiled], [3, 1, 1])
        self.assertEqual(space["estimator__estimator__min_samples_split"].categories, (10, 30, 50))
        self.assertEqual(sum(len(s["estimator__estimator__min_samples_split"].categories) for s, _ in compiled), 5)

    def test_equivalent_split_thresholds_build_identical_forest_trees(self):
        from sklearn.ensemble import RandomForestClassifier
        rng = np.random.default_rng(31)
        x = rng.normal(size=(240, 8))
        y = (x[:, 0] + x[:, 1] > 0).astype(int)
        forests = [RandomForestClassifier(n_estimators=3, min_samples_leaf=30, min_samples_split=split,
                                         random_state=3).fit(x, y) for split in (10, 30, 50)]
        for other in forests[1:]:
            np.testing.assert_array_equal(forests[0].predict_proba(x), other.predict_proba(x))
            for first, second in zip(forests[0].estimators_, other.estimators_):
                for attribute in ("feature", "threshold", "value"):
                    np.testing.assert_array_equal(getattr(first.tree_, attribute), getattr(second.tree_, attribute))

    def test_fractional_counts_and_tiny_budgets_remain_unmodified(self):
        fractional = {"min_samples_leaf": [0.1, 0.2], "min_samples_split": [0.2, 0.5]}
        self.assertEqual(prepare_random_forest_bayesian_spaces(fractional, 20), [(fractional, 20)])
        integers = {"min_samples_leaf": [10, 30, 50], "min_samples_split": [10, 30, 50]}
        self.assertEqual(prepare_random_forest_bayesian_spaces(integers, 1), [(integers, 1)])

    def test_notebook_uses_conditional_pairs_without_multiplying_its_budget(self):
        path = Path(__file__).resolve().parents[1] / "notebooks/random_forest.ipynb"
        sources = ["".join(cell.get("source", [])) for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]]
        tree = ast.parse(next(source for source in sources if "search = BayesSearchCV(" in source))
        constructor = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                           and isinstance(node.func, ast.Name) and node.func.id == "BayesSearchCV")
        space = next(keyword.value for keyword in constructor.keywords if keyword.arg == "search_spaces")
        self.assertIsInstance(space, ast.Call)
        self.assertEqual(space.func.id, "prepare_random_forest_bayesian_spaces")
        self.assertEqual([argument.id for argument in space.args], ["param_space", "n_iter"])


if __name__ == "__main__":
    unittest.main()
