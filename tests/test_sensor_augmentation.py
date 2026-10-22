"""Sensor augmentation contracts without model fitting or external data."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import clone


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import base_utils as base


def recordings(length=8, sequences=2, duplicate_index=False):
    """Interleave sequences so assignments must respect row positions."""
    count = length * sequences
    time = np.repeat(np.arange(length, dtype=float), sequences)
    sequence = np.tile(np.arange(sequences), length)
    frame = pd.DataFrame({
        "sequence_id": [f"s{value}" for value in sequence],
        "sequence_counter": time.astype(int),
        "subject": [f"person{value}" for value in sequence],
        "bfrb": ["target"] * count,
        "is_target": np.ones(count, dtype=int),
        "auxiliary_score": np.full(count, 0.75),
        "acc_x": 1.0 + time,
        "acc_jerk_x": 3.0 + time ** 2,
        "rot_w": np.ones(count),
        "rot_angular_velocity_x": time / 10.0,
        "lin_acc_x": 2.0 + time + sequence,
        "tof_1_v0": 4.0 + time ** 2 + sequence,
        "thm_1": 20.0 + time + sequence,
        "tof_note": ["kept as text"] * count,
    })
    if duplicate_index:
        frame.index = np.repeat(np.arange((count + 1) // 2), 2)[:count]
    else:
        frame.index = np.arange(count) * 3 + 7
    return frame


def augmentor(names, **options):
    settings = dict(
        augmentations=names, prob=1.0, per_aug_prob=1.0, seed=17,
        jitter_sigma=0.4, noise_std=0.4, scaling_sigma=0.4,
        time_shift_frac=0.75, crop_frac_range=(0.5, 0.5),
        temporal_mask_frac=0.5, temporal_num_masks=1,
        channel_drop_prob=1.0, sensor_drop_prob=1.0,
        warp_sigma=0.4, warp_num_knots=4,
    )
    settings.update(options)
    return base.SensorAugmentor(**settings)


RAW_GEOMETRY = ["acc_x", "acc_jerk_x", "rot_w", "rot_angular_velocity_x"]
ELIGIBLE = ["lin_acc_x", "tof_1_v0", "thm_1"]
METADATA = ["sequence_id", "sequence_counter", "subject", "bfrb",
            "is_target", "auxiliary_score", "tof_note"]


class StandaloneAugmentationTests(unittest.TestCase):
    def assert_noop_without_random_draws(self, function, **kwargs):
        values = np.arange(24, dtype=float).reshape(8, 3)
        original = values.copy()
        rng = np.random.default_rng(42)
        state = deepcopy(rng.bit_generator.state)
        result = function(values, rng=rng, **kwargs)
        np.testing.assert_array_equal(result, original)
        np.testing.assert_array_equal(values, original)
        self.assertEqual(rng.bit_generator.state, state)

    def test_zero_strengths_are_exact_noops_without_consuming_rng(self):
        cases = [
            (base.augment_jitter, {"sigma": 0.0}),
            (base.augment_gaussian_noise, {"std": 0.0}),
            (base.augment_scaling, {"sigma": 0.0}),
            (base.augment_time_shift, {"max_shift_frac": 0.0}),
            (base.augment_crop_resize, {"crop_frac_range": (1.0, 1.0)}),
            (base.augment_temporal_mask, {"mask_frac": 0.0, "num_masks": 2}),
            (base.augment_temporal_mask, {"mask_frac": 0.5, "num_masks": 0}),
            (base.augment_channel_dropout, {"drop_prob": 0.0}),
            (base.augment_sensor_dropout, {"sensor_groups": {"one": [0, 1]}, "drop_prob": 0.0}),
            (base.augment_magnitude_warp, {"sigma": 0.0, "num_knots": 4}),
            (base.augment_magnitude_warp, {"sigma": 0.5, "num_knots": 0}),
        ]
        for function, kwargs in cases:
            with self.subTest(function=function.__name__, kwargs=kwargs):
                self.assert_noop_without_random_draws(function, **kwargs)

    def test_interpolation_and_shift_handle_empty_and_single_sample_inputs(self):
        for function in (base.augment_time_shift, base.augment_crop_resize,
                         base.augment_magnitude_warp):
            for length in (0, 1):
                with self.subTest(function=function.__name__, length=length):
                    values = np.arange(length * 3, dtype=float).reshape(length, 3)
                    original = values.copy()
                    result = function(values, rng=np.random.default_rng(3))
                    self.assertEqual(result.shape, values.shape)
                    self.assertTrue(np.isfinite(result).all())
                    np.testing.assert_array_equal(values, original)

    def test_time_shift_keeps_shape_even_when_fraction_exceeds_sequence_length(self):
        values = np.arange(6, dtype=float).reshape(2, 3)
        original = values.copy()
        for seed in range(6):
            result = base.augment_time_shift(
                values, max_shift_frac=4.0, rng=np.random.default_rng(seed))
            self.assertEqual(result.shape, values.shape)
            self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(values, original)


class SensorAugmentorPolicyTests(unittest.TestCase):
    def test_every_raw_operation_preserves_all_acceleration_and_rotation_prefixes(self):
        source = recordings()
        original = source.copy(deep=True)
        for name in base.ALL_AUGMENTATION_NAMES:
            with self.subTest(operation=name):
                result = augmentor([name]).fit_transform(source)
                pd.testing.assert_frame_equal(result[RAW_GEOMETRY], source[RAW_GEOMETRY])
                pd.testing.assert_frame_equal(result[METADATA], source[METADATA])
                pd.testing.assert_index_equal(result.index, source.index)
                self.assertEqual(result.columns.tolist(), source.columns.tolist())
                self.assertTrue(np.isfinite(result[ELIGIBLE].to_numpy()).all())
                pd.testing.assert_frame_equal(source, original)

    def test_raw_scaling_augments_eligible_sensors_and_keeps_geometry_and_labels(self):
        source = recordings()
        original = source.copy(deep=True)
        result = augmentor(["scaling"]).fit_transform(source)
        for column in ELIGIBLE:
            self.assertFalse(result[column].equals(source[column]), column)
        pd.testing.assert_frame_equal(result[RAW_GEOMETRY + METADATA],
                                      source[RAW_GEOMETRY + METADATA])
        pd.testing.assert_frame_equal(source, original)

    def test_feature_mode_can_augment_processed_acceleration_and_rotation(self):
        source = recordings()
        original = source.copy(deep=True)
        model = augmentor(["scaling"]).fit(source)
        result = model.transform_features(source)
        for column in RAW_GEOMETRY + ELIGIBLE:
            self.assertFalse(result[column].equals(source[column]), column)
        pd.testing.assert_frame_equal(result[METADATA], source[METADATA])
        pd.testing.assert_frame_equal(source, original)

    def test_explicit_feature_columns_limit_feature_augmentation(self):
        source = recordings()
        model = augmentor(["channel_dropout"]).fit(source)
        result = model.transform_features(source, feature_columns=["acc_x", "rot_w"])
        np.testing.assert_array_equal(result[["acc_x", "rot_w"]], 0.0)
        retained = source.columns.difference(["acc_x", "rot_w"], sort=False)
        pd.testing.assert_frame_equal(result[retained], source[retained])

    def test_duplicate_indices_and_interleaved_sequences_use_positional_assignment(self):
        source = recordings(duplicate_index=True)
        original = source.copy(deep=True)
        result = augmentor(["scaling"]).fit_transform(source)
        pd.testing.assert_index_equal(result.index, source.index)
        pd.testing.assert_frame_equal(result[RAW_GEOMETRY + METADATA],
                                      source[RAW_GEOMETRY + METADATA])
        for sequence_id in source.sequence_id.unique():
            positions = np.flatnonzero(source.sequence_id.eq(sequence_id).to_numpy())
            ratios = (result.iloc[positions][ELIGIBLE].to_numpy()
                      / source.iloc[positions][ELIGIBLE].to_numpy())
            np.testing.assert_allclose(ratios, ratios[0, 0])
        pd.testing.assert_frame_equal(source, original)

    def test_seed_reproducibility_and_cloning(self):
        source = recordings()
        model = augmentor(["jitter", "crop_resize", "magnitude_warp"])
        first = clone(model).fit_transform(source)
        second = clone(model).fit_transform(source)
        different = clone(model).set_params(seed=18).fit_transform(source)
        pd.testing.assert_frame_equal(first, second)
        self.assertFalse(first[ELIGIBLE].equals(different[ELIGIBLE]))

    def test_all_operations_accept_single_sample_sequences_and_empty_frames(self):
        for length in (0, 1):
            with self.subTest(length=length):
                source = recordings(length=length)
                original = source.copy(deep=True)
                result = augmentor(list(base.ALL_AUGMENTATION_NAMES)).fit_transform(source)
                self.assertEqual(result.shape, source.shape)
                pd.testing.assert_index_equal(result.index, source.index)
                pd.testing.assert_frame_equal(result[RAW_GEOMETRY + METADATA],
                                              source[RAW_GEOMETRY + METADATA])
                self.assertTrue(np.isfinite(result[ELIGIBLE].to_numpy()).all())
                pd.testing.assert_frame_equal(source, original)

    def test_disabled_probabilities_and_empty_operation_lists_do_not_change_input(self):
        source = recordings()
        for settings in ({"prob": 0.0}, {"per_aug_prob": 0.0}, {"augmentations": []}):
            with self.subTest(settings=settings):
                model = augmentor(["jitter"]).set_params(**settings)
                pd.testing.assert_frame_equal(model.fit_transform(source), source)

    def test_zero_strength_transformer_preserves_raw_and_processed_values(self):
        source = recordings()
        model = augmentor(
            list(base.ALL_AUGMENTATION_NAMES), jitter_sigma=0.0, noise_std=0.0,
            scaling_sigma=0.0, time_shift_frac=0.0, crop_frac_range=(1.0, 1.0),
            temporal_mask_frac=0.0, temporal_num_masks=0,
            channel_drop_prob=0.0, sensor_drop_prob=0.0,
            warp_sigma=0.0, warp_num_knots=0,
        )
        pd.testing.assert_frame_equal(model.fit_transform(source), source)
        pd.testing.assert_frame_equal(model.transform_features(source), source)


class SensorAugmentorDiagnosticTests(unittest.TestCase):
    def test_first_corrupting_operation_reports_context_and_bounded_locations(self):
        source = recordings(length=8, sequences=1)
        original = source.copy(deep=True)
        called = []

        def bad_operation(values, rng=None):
            called.append("bad_operation")
            return np.full_like(values, np.nan)

        def should_not_run(values, rng=None, **kwargs):
            self.fail("Operations after the first corruption must not run")

        with patch.dict(base._AUGMENTATION_REGISTRY,
                        {"bad_operation": bad_operation, "scaling": should_not_run}):
            model = augmentor(["jitter", "bad_operation", "scaling"],
                              diagnostics=True, diagnostic_max_locations=2).fit(source)
            model.diagnostic_context_ = {"candidate_id": "candidate-7", "stage_name": "finetuning"}
            with self.assertRaises(ValueError) as caught:
                model.transform_features(source)

        report = caught.exception.augmentation_diagnostics
        self.assertEqual(called, ["bad_operation"])
        self.assertEqual(report["candidate_id"], "candidate-7")
        self.assertEqual(report["seed"], 17)
        self.assertEqual(report["sequence_id"], "s0")
        self.assertEqual(report["stage"], "features")
        self.assertFalse(report["input_invalid"])
        self.assertEqual(report["input_validity"]["nonfinite_count"], 0)
        self.assertEqual(report["first_invalid_operation"], "bad_operation")
        self.assertEqual([step["name"] for step in report["operations"]],
                         ["jitter", "bad_operation"])
        self.assertEqual(report["operations"][0]["parameters"]["sigma"], 0.4)
        self.assertEqual(report["validity"]["nonfinite_count"],
                         len(source) * len(RAW_GEOMETRY + ELIGIBLE))
        locations = report["validity"]["nonfinite_locations"]
        self.assertEqual(len(locations), 2)
        self.assertTrue({"row_position", "index", "column", "counter"} <= set(locations[0]))
        self.assertIn("exception", report)
        self.assertIn("traceback", report)
        self.assertIn("rng_state", report)
        self.assertLessEqual(len(model.diagnostics_), model.diagnostic_max_sequences)
        self.assertEqual(len(model.diagnostics_), 1,
                         "A failed sequence must consume only one report slot")
        pd.testing.assert_frame_equal(source, original)

    def test_preexisting_nonfinite_feature_input_is_distinguished_from_corruption(self):
        source = recordings(sequences=1)
        source.iloc[0, source.columns.get_loc("tof_1_v0")] = np.nan
        original = source.copy(deep=True)
        model = augmentor(["scaling"], diagnostics=True).fit(source)
        with self.assertRaises(ValueError) as caught:
            model.transform_features(source)
        report = caught.exception.augmentation_diagnostics
        self.assertTrue(report["input_invalid"])
        self.assertEqual(report["first_invalid_operation"], "input")
        self.assertEqual(report["input_validity"]["nonfinite_count"], 1)
        self.assertEqual(report["operations"], [])
        self.assertEqual(report["input_validity"]["nonfinite_locations"][0]["column"], "tof_1_v0")
        pd.testing.assert_frame_equal(source, original)

    def test_zero_raw_quaternions_are_rejected_before_augmentation(self):
        source = recordings(sequences=1).assign(rot_w=0.0, rot_x=0.0, rot_y=0.0, rot_z=0.0)
        original = source.copy(deep=True)
        model = augmentor(["jitter"], diagnostics=True, diagnostic_max_locations=2)
        with self.assertRaises(ValueError) as caught:
            model.fit_transform(source)
        report = caught.exception.augmentation_diagnostics
        self.assertTrue(report["input_invalid"])
        self.assertEqual(report["first_invalid_operation"], "input")
        self.assertEqual(report["input_validity"]["zero_quaternion_count"], len(source))
        self.assertEqual(len(report["input_validity"]["quaternion_locations"]), 2)
        self.assertEqual(report["operations"], [])
        pd.testing.assert_frame_equal(source, original)

    def test_near_zero_quaternions_are_reported_without_modifying_geometry(self):
        source = recordings(sequences=1).assign(rot_w=1e-14, rot_x=0.0, rot_y=0.0, rot_z=0.0)
        model = augmentor(["scaling"], diagnostics=True, diagnostic_max_locations=2)
        result = model.fit_transform(source)
        self.assertEqual(model.diagnostics_[0]["input_validity"]["near_zero_quaternion_count"],
                         len(source))
        self.assertEqual(model.diagnostics_[0]["input_validity"]["zero_quaternion_count"], 0)
        self.assertEqual(len(model.diagnostics_[0]["input_validity"]["quaternion_locations"]), 2)
        pd.testing.assert_frame_equal(result[["rot_w", "rot_x", "rot_y", "rot_z"]],
                                      source[["rot_w", "rot_x", "rot_y", "rot_z"]])

    def test_processed_quaternion_features_can_be_masked_without_geometry_rejection(self):
        source = recordings(sequences=1).assign(rot_x=0.0, rot_y=0.0, rot_z=0.0)
        model = augmentor(["sensor_dropout"], diagnostics=True).fit(source)
        result = model.transform_features(source)
        np.testing.assert_array_equal(result[["rot_w", "rot_x", "rot_y", "rot_z"]], 0.0)
        self.assertEqual(model.diagnostics_[0]["stage"], "features")
        self.assertFalse(model.diagnostics_[0]["validity"]["invalid"])
        self.assertIsNone(model.diagnostics_[0]["first_invalid_operation"])

    def test_input_probe_records_missing_raw_data_for_the_cleaner_without_mutation(self):
        source = recordings(sequences=1)
        source.iloc[0, source.columns.get_loc("acc_x")] = np.nan
        original = source.copy(deep=True)
        model = augmentor(["jitter"], diagnostics=True).fit(source)
        model.diagnose_input(source)
        self.assertEqual(len(model.diagnostics_), 1)
        self.assertTrue(model.diagnostics_[0]["input_invalid"])
        self.assertEqual(model.diagnostics_[0]["first_invalid_operation"], "input")
        self.assertEqual(model.diagnostics_[0]["input_validity"]["nonfinite_count"], 1)
        self.assertEqual(model.diagnostics_[0]["operations"], [])
        pd.testing.assert_frame_equal(source, original)

    def test_input_probe_retains_near_zero_quaternion_counts_for_training_hook(self):
        source = recordings(sequences=1).assign(rot_w=1e-14, rot_x=0.0, rot_y=0.0, rot_z=0.0)
        model = augmentor(["jitter"], diagnostics=True).fit(source)
        model.diagnose_input(source)
        self.assertEqual(len(model.diagnostics_), 1)
        report = model.diagnostics_[0]
        self.assertEqual(report["input_validity"]["near_zero_quaternion_count"], len(source))
        self.assertFalse(report["input_invalid"])
        self.assertIsNone(report["first_invalid_operation"])
        self.assertEqual(report["operations"], [])

    def test_report_count_is_capped_and_diagnostics_do_not_change_random_results(self):
        source = recordings(sequences=12)
        plain = augmentor(["jitter", "scaling"])
        diagnostic = clone(plain).set_params(diagnostics=True, diagnostic_max_sequences=2)
        expected = plain.fit_transform(source)
        actual = diagnostic.fit_transform(source)
        pd.testing.assert_frame_equal(actual, expected)
        self.assertEqual(len(diagnostic.diagnostics_), 2)
        self.assertTrue(all(report["first_invalid_operation"] is None
                            for report in diagnostic.diagnostics_))

    def test_disabled_diagnostics_do_not_run_validity_scans(self):
        source = recordings()
        model = augmentor(["jitter"], diagnostics=False).fit(source)
        with patch.object(model, "_validity", side_effect=AssertionError("Unexpected diagnostic scan")):
            model.diagnose_input(source)
            model.transform(source)
            model.transform_features(source)
        self.assertEqual(model.diagnostics_, [])


if __name__ == "__main__":
    unittest.main()
