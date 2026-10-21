"""Non-architecture no-op search knobs and their active counterexamples."""
import ast
from copy import deepcopy
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
from multibranch_utils import candidate_error, conditional_bayesian_spaces, HierarchicalGASearchCV
from test_multibranch_conditional_ga import CountingInt
from test_multibranch_hierarchical_search import FIT_LOG, dry_pipeline, toy_data


class TrainingDependencies(unittest.TestCase):
    def assert_retained(self, params, base=None):
        FIT_LOG.clear()
        pipeline = dry_pipeline()
        if base:
            pipeline.set_params(**{'estimator__' + key: value for key, value in base.items()})
        supplied = {'estimator__' + key: value for key, value in params.items()}
        before = deepcopy(params)
        search = HierarchicalGASearchCV(pipeline, param_grid={key: [value] for key, value in supplied.items()},
            population_size=2, tournament_size=2, generations=0, refit=False, cv=2, verbose=0)
        x, y = toy_data()
        search.fit(x, y)
        self.assertEqual(len(search.cv_results_['params']), 1)
        reported = search.cv_results_['params'][0]
        self.assertEqual(set(reported), set(supplied))
        for key, value in supplied.items():
            actual = reported[key]
            if isinstance(value, (dict, list, tuple)) and isinstance(actual, str):
                actual = json.loads(actual)
            self.assertEqual(actual, value)
        self.assertEqual(len(FIT_LOG), 2)
        for fitted in FIT_LOG:
            for key in params:
                self.assertEqual(fitted[key], reported['estimator__' + key])
        self.assertEqual(params, before)

    def test_optimizer_and_loss_settings_reach_fit_unchanged(self):
        for optimizer in ('adam', 'adamw', 'sgd'):
            self.assert_retained(dict(optimizer_name=optimizer, momentum=0.7, weight_decay=0.01))
        for loss in ('focal', 'sparse_cce'):
            self.assert_retained(dict(loss_name=loss, label_smoothing=0.2))

    def test_scheduler_settings_are_kept_for_disabled_and_unused_families(self):
        knobs = dict(lr_scheduler_type='plateau', lr_factor=0.2, lr_patience=3,
                     min_lr=1e-5, cosine_t_max=10, warmup_epochs=2)
        self.assert_retained(knobs)
        self.assert_retained(dict(knobs, use_lr_scheduler=True))
        self.assert_retained(dict(knobs, use_lr_scheduler=True, lr_scheduler_type='cosine'))
        self.assert_retained(dict(knobs, use_lr_scheduler=True, validation_split=0))

    def test_disabled_augmentation_keeps_every_requested_value(self):
        aug = SensorAugmentor(prob=0.5, per_aug_prob=0.5)
        self.assert_retained({'augmentor__prob': 0, 'augmentor__noise_std': 8,
                              'augmentor__per_aug_prob': 0.8}, {'augmentor': aug})
        self.assert_retained({'augmentor__augmentations': '["jitter"]',
                              'augmentor__noise_std': 8, 'augmentor__jitter_sigma': 0.2}, {'augmentor': aug})
        self.assert_retained({'augmentor__augmentations': []}, {'augmentor': aug})

    def test_spatial_only_candidate_keeps_other_branch_settings(self):
        params = dict(temporal_architecture='phase_attention', tof_encoder_type='spatial_cnn',
                      extractor__acc_modes='', extractor__rotation_modes='', extractor__thm_modes='',
                      extractor__tof_modes='raw', extractor__stft_configs=None, extractor__cwt_configs=None,
                      static_branch_units=20, branch_cnn_filters=32, use_se=True, se_ratio=8,
                      use_residual=True, use_batchnorm=True, cnn_filters=64, tof_spatial_filters=16)
        self.assert_retained(params)
        self.assert_retained(dict(params, extractor__acc_modes='raw'))

    def test_unknown_extractor_subclass_is_not_mutated(self):
        class Custom(SequenceExtractor):
            pass
        extractor = Custom()
        params = {'extractor': extractor, 'extractor__use_dead_reckoning': True}
        self.assertIsNone(candidate_error(params))
        self.assertIs(params['extractor'], extractor)
        self.assertTrue(params['extractor__use_dead_reckoning'])
        self.assertFalse(extractor.use_dead_reckoning)

    def test_branch_dicts_keep_disabled_sensors_and_complete_filter_lists(self):
        common = dict(extractor__rotation_modes='', extractor__tof_modes='', extractor__thm_modes='',
                      extractor__stft_configs=None, extractor__cwt_configs=None,
                      branch_num_conv_layers=2, branch_filter_mode='custom')
        self.assert_retained(dict(common, branch_filters={'acc': [16, 32, 999], 'tof': 888}))
        self.assert_retained(dict(common, branch_filters={'acc': [16, 32, 1], 'tof': 5}))
        self.assert_retained(dict(common, branch_filters={'acc': 16}))
        self.assert_retained(dict(common, branch_filters={'acc': [16, 16]}))

    def test_missing_branch_entries_are_not_inserted(self):
        self.assert_retained(dict(branch_filters={'tof': 16}, extractor__tof_modes=''))


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

    def test_spectral_configuration_fields_are_retained_without_rewriting(self):
        configs = [{'acc_modes': 'raw', 'window_seconds': 0.5, 'nperseg': 8, 'scaling': 'density'},
                   {'acc_modes': 'jerk', 'window_seconds': 0.5, 'nperseg': 32, 'scaling': 'psd'}]
        params = {'extractor__stft_configs': [[config] for config in configs]}
        before = deepcopy(params)
        compiled = conditional_bayesian_spaces(params)[0]
        actual = [json.loads(value) if isinstance(value, str) else value
                  for value in compiled['extractor__stft_configs'].categories]
        self.assertEqual(actual, [[config] for config in configs])
        self.assertEqual(params, before)

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

    def test_ga_samples_and_retains_every_requested_training_setting(self):
        momentum, warmup = CountingInt(1, 8), CountingInt(1, 8)
        search = HierarchicalGASearchCV(dry_pipeline(), param_grid={
            'estimator__optimizer_name': ['adam'], 'estimator__momentum': momentum,
            'estimator__use_lr_scheduler': [False], 'estimator__warmup_epochs': warmup,
        }, population_size=2, tournament_size=2, generations=0, refit=False, cv=2, verbose=0)
        x, y = toy_data()
        search.fit(x, y)
        self.assertGreaterEqual(momentum.draws, 2)
        self.assertGreaterEqual(warmup.draws, 2)
        for params in search.cv_results_['params']:
            self.assertIn('estimator__momentum', params)
            self.assertIn('estimator__warmup_epochs', params)

    def test_bayes_keeps_requested_dimensions_for_fixed_and_searched_switches(self):
        try:
            from skopt.space import Categorical
        except ImportError:
            self.skipTest('scikit-optimize unavailable')
        space = dict(temporal_architecture=Categorical(['bert']),
                     optimizer_name=Categorical(['adam', 'sgd']), momentum=Categorical([0.1, 0.9]),
                     use_lr_scheduler=Categorical([False, True]), warmup_epochs=Categorical([0, 2]),
                     lr_scheduler_type=Categorical(['cosine']))
        for requested in (space, dict(space, optimizer_name=['adam'], use_lr_scheduler=[False])):
            compiled = conditional_bayesian_spaces(requested)[0]
            self.assertEqual(set(compiled), set(requested))
            self.assertEqual(tuple(compiled['momentum'].categories), (0.1, 0.9))
            self.assertEqual(tuple(compiled['warmup_epochs'].categories), (0, 2))


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
