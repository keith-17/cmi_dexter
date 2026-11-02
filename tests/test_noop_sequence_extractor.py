"""Numerical and compatibility contracts for optional NOOP-inspired summaries."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from base_utils import SequenceExtractor, SensorAugmentor, InvalidExtractorParams
from noop_sequence_extractor import (
    NoopOptions, NoopSequenceExtractor, _window_psd,
    inactive_noop_parameters, validate_noop_settings,
)


def frame(n=600, fs=100.0, frequency=2.0, seq="s"):
    t = np.arange(n) / fs
    return pd.DataFrame({
        "sequence_id": seq, "sequence_counter": np.arange(n),
        "acc_x": 0.2 * np.sin(2 * np.pi * frequency * t),
        "acc_y": np.zeros(n), "acc_z": np.ones(n) * 9.80665,
        "rot_w": np.ones(n), "rot_x": np.zeros(n),
        "rot_y": np.zeros(n), "rot_z": np.zeros(n),
    })


def settings(**overrides):
    result = dict(
        acc_modes="raw", rotation_modes="quaternion|angular_velocity",
        tof_modes="", thm_modes="", stft_configs=None, cwt_configs=None,
        linear_acc_mode="baseline", compute_dt=True,
        imu_native_sampling_rate=100, rot_native_sampling_rate=100,
        chunk_window_size=250, chunk_stride_ratio=0.75,
        use_chunk_stride_ratio=True, final_window_mode="overlap",
        sequence_crop_mode="none", maxlen=None, padding_value=0.0,
    )
    result.update(overrides)
    return result


def feature(chunks, name):
    return chunks["noop_features"][:, chunks["noop_feature_names"].index("noop_" + name)]


class NoopCompatibilityTests(unittest.TestCase):
    def assertLegacyEqual(self, left, right):
        self.assertEqual(set(left), set(right))
        for key in left:
            if isinstance(left[key], np.ndarray):
                np.testing.assert_array_equal(left[key], right[key])
            else:
                self.assertEqual(left[key], right[key])

    def test_none_and_all_off_preserve_exact_outputs_in_both_paths(self):
        x = frame(601)
        for options in (None, NoopOptions()):
            legacy = SequenceExtractor(**settings())
            child = NoopSequenceExtractor(**settings(), noop=options)
            self.assertLegacyEqual(legacy.fit_transform(x), child.fit_transform(x))
            self.assertLegacyEqual(legacy.transform_chunks(x), child.transform_chunks(x))
            self.assertFalse(hasattr(child, "_noop_source"))

    def test_parent_and_nested_parameters_survive_clone(self):
        original = NoopSequenceExtractor(**settings(), noop=NoopOptions(motion_stats="acc_ac_rms"))
        copied = clone(original)
        self.assertEqual(copied.get_params()["chunk_window_size"], 250)
        self.assertEqual(copied.get_params()["noop__motion_stats"], "acc_ac_rms")
        self.assertIsNot(copied.noop, original.noop)
        copied.set_params(noop__motion_stats="axis_rms", chunk_window_size=120)
        self.assertEqual(original.noop.motion_stats, "acc_ac_rms")
        self.assertEqual(copied.chunk_window_size, 120)
        self.assertEqual(set(original.get_params(deep=False)) - {"noop"}, set(SequenceExtractor().get_params(deep=False)))

    def test_nested_parameters_can_be_enabled_directly_from_default_off(self):
        original = NoopSequenceExtractor(**settings())
        self.assertIsNone(original.get_params(deep=False)["noop"])
        self.assertIn("noop__motion_stats", original.get_params(deep=True))
        self.assertIsNone(clone(original).noop)
        original.set_params(noop__motion_stats="axis_rms")
        result = original.fit_transform(frame(50))
        self.assertEqual(feature(result, "acc_x_rms_supported")[0], 1)
        self.assertEqual(clone(original).get_params()["noop__motion_stats"], "axis_rms")

    def test_enabled_features_use_identical_parent_tail_crop_and_cap(self):
        x = frame(601)
        for controls in ({}, {"max_windows_per_sequence": 2, "window_cap_policy": "uniform"},
                         {"maxlen": 400, "sequence_crop_mode": "tail"}):
            options = settings(**controls)
            legacy = SequenceExtractor(**options).fit_transform(x)
            extractor = NoopSequenceExtractor(**options, noop=NoopOptions(quality="report"))
            actual = extractor.fit_transform(x)
            self.assertLegacyEqual(legacy, {key: actual[key] for key in legacy})
            self.assertEqual(len(actual["noop_quality"]), len(actual["X"]))
            self.assertEqual(actual["noop_features"].shape, (len(actual["X"]), 0))
        actual = NoopSequenceExtractor(**settings(), noop=NoopOptions(quality="report")).fit_transform(frame(600))
        self.assertEqual([r["start_counter"] for r in actual["noop_quality"]], [0, 188, 350])

    def test_fit_then_transform_and_fit_transform_share_summary_schema(self):
        x = frame()
        first = NoopSequenceExtractor(**settings(), noop=NoopOptions(motion_stats="full", periodicity="acf", spectral_stats="all", quality="features"))
        a = first.fit_transform(x)
        b = clone(first).fit(x).transform_chunks(x)
        self.assertLegacyEqual(a, b)

    def test_all_off_does_not_consume_extra_augmentation_random_numbers(self):
        x = frame()
        augmentor = SensorAugmentor(augmentations=["scaling"], prob=1, per_aug_prob=1, scaling_sigma=0.2, seed=81)
        a, b = SequenceExtractor(**settings()), NoopSequenceExtractor(**settings(), noop=NoopOptions())
        for extractor in (a, b):
            extractor._training_feature_augmentor = clone(augmentor).fit(x)
        self.assertLegacyEqual(a.fit_transform(x), b.fit_transform(x))
        self.assertEqual(a._training_feature_augmentor._rng.bit_generator.state,
                         b._training_feature_augmentor._rng.bit_generator.state)


class NoopNumericalTests(unittest.TestCase):
    def test_statistics_are_chunk_local_and_do_not_include_padding(self):
        x = frame(400)
        x["acc_x"] = np.r_[np.ones(200), np.ones(200) * 5]
        result = NoopSequenceExtractor(**settings(chunk_window_size=200, use_chunk_stride_ratio=False, chunk_stride=200),
                                      noop=NoopOptions(motion_stats="axis_rms", quality="features")).fit_transform(x)
        np.testing.assert_allclose(feature(result, "acc_x_rms"), [1, 5])
        short = NoopSequenceExtractor(**settings(), noop=NoopOptions(motion_stats="full", quality="features")).fit_transform(frame(10))
        self.assertTrue(np.isfinite(short["noop_features"]).all())
        self.assertEqual(short["noop_quality"][0]["real_samples"], 10)
        self.assertEqual(feature(short, "acc_x_rms_supported")[0], 0)
        self.assertEqual(feature(short, "observed_fraction")[0], 1)

    def test_jerk_has_physical_time_units_at_two_sampling_rates(self):
        for fs in (50, 100):
            x = frame(200, fs=fs)
            x["acc_x"] = np.arange(len(x)) / fs
            result = NoopSequenceExtractor(**settings(imu_native_sampling_rate=fs, rot_native_sampling_rate=fs),
                                          noop=NoopOptions(motion_stats="jerk_rms")).fit_transform(x)
            np.testing.assert_allclose(feature(result, "jerk_rms"), 1, atol=1e-6)

    def test_rotation_uses_quaternion_rad_per_second_and_sign_invariance(self):
        x = frame(250)
        angle = 0.7 * np.arange(len(x)) / 100
        x["rot_w"], x["rot_z"] = np.cos(angle / 2), np.sin(angle / 2)
        x.loc[::2, ["rot_w", "rot_x", "rot_y", "rot_z"]] *= -1
        result = NoopSequenceExtractor(**settings(), noop=NoopOptions(motion_stats="rotation_mean|rotation_rms")).fit_transform(x)
        np.testing.assert_allclose(feature(result, "rotation_mean_rad_s"), 0.7, atol=1e-6)
        np.testing.assert_allclose(feature(result, "rotation_rms_rad_s"), 0.7, atol=1e-6)

    def test_signed_axis_acf_avoids_magnitude_double_frequency(self):
        x = frame(600, frequency=1.4)
        x["acc_z"] = 1.0
        result = NoopSequenceExtractor(**settings(chunk_window_size=600), noop=NoopOptions(periodicity="acf_harmonic_checked")).fit_transform(x)
        self.assertAlmostEqual(feature(result, "acf_frequency_hz")[0], 1.4, delta=0.03)
        self.assertGreater(feature(result, "acf_strength")[0], 0.8)
        self.assertEqual(feature(result, "acf_magnitude_harmonic")[0], 1)

    def test_counter_gaps_do_not_become_a_compressed_time_series(self):
        x = frame(250)
        x["sequence_counter"] = np.arange(250) * 2
        result = NoopSequenceExtractor(**settings(compute_dt=False), noop=NoopOptions(periodicity="acf", spectral_stats="all", quality="features")).fit_transform(x)
        self.assertEqual(feature(result, "acf_strength_supported")[0], 0)
        self.assertEqual(feature(result, "spectral_peak_fraction_supported")[0], 0)
        self.assertEqual(result["noop_quality"][0]["longest_contiguous_samples"], 1)

    def test_short_gap_fill_is_bounded_and_traceable(self):
        x = frame(100)
        x.loc[49:50, "acc_x"] = np.nan
        for max_gap, expected_imputed, longest in ((0, 0, 49), (1, 0, 49), (2, 2, 100)):
            result = NoopSequenceExtractor(**settings(), noop=NoopOptions(quality="features", max_gap_samples=max_gap)).fit_transform(x)
            report = result["noop_quality"][0]
            self.assertEqual(report["observed_samples"], 98)
            self.assertEqual(report["imputed_samples"], expected_imputed)
            self.assertEqual(report["longest_contiguous_samples"], longest)

    def test_psd_integrates_to_mean_square_and_scales_quadratically(self):
        fs, n = 100, 1000
        signal = np.sin(2 * np.pi * 5 * np.arange(n) / fs)[:, None]
        f, one = _window_psd(signal, fs, "constant")
        _, two = _window_psd(signal * 2, fs, "constant")
        self.assertAlmostEqual(one.sum() * (f[1] - f[0]), 0.5, places=10)
        np.testing.assert_allclose(two, one * 4, rtol=1e-12, atol=1e-20)

    def test_local_psd_band_fractions_change_between_two_windows(self):
        x = frame(400)
        t = np.arange(200) / 100
        x["acc_x"] = np.r_[np.sin(2 * np.pi * 2 * t), np.sin(2 * np.pi * 7 * t)]
        result = NoopSequenceExtractor(**settings(chunk_window_size=200, use_chunk_stride_ratio=False, chunk_stride=200),
                                      noop=NoopOptions(spectral_stats="band_fractions", bands=((0.5, 4), (4, 8)))).fit_transform(x)
        np.testing.assert_allclose(feature(result, "spectral_band_0_fraction"), [1, 0], atol=1e-6)
        np.testing.assert_allclose(feature(result, "spectral_band_1_fraction"), [0, 1], atol=1e-6)

    def test_summaries_read_same_augmented_raw_channels_as_model(self):
        x = frame(250)
        extractor = NoopSequenceExtractor(**settings(), noop=NoopOptions(motion_stats="axis_rms"))
        extractor._training_feature_augmentor = SensorAugmentor(augmentations=["scaling"], prob=1, per_aug_prob=1, scaling_sigma=0.2, seed=12).fit(x)
        result = extractor.fit_transform(x)
        column = result["feature_names"].index("acc_x_raw")
        expected = np.sqrt(np.mean(result["X"][0, :, column] ** 2))
        self.assertAlmostEqual(feature(result, "acc_x_rms")[0], expected, places=6)
        # Parent feature selection is never overridden to manufacture a source.
        extractor = NoopSequenceExtractor(**settings(acc_modes="smoothed"), noop=NoopOptions(motion_stats="axis_rms"))
        extractor._training_feature_augmentor = SensorAugmentor(augmentations=["scaling"], prob=1, per_aug_prob=1, seed=12).fit(x)
        result = extractor.fit_transform(x)
        self.assertEqual(feature(result, "acc_x_rms_supported")[0], 0)

    def test_invalid_quaternions_preserve_parent_validation(self):
        x = frame(50)
        x.loc[20, ["rot_w", "rot_x", "rot_y", "rot_z"]] = 0
        with self.assertRaisesRegex(ValueError, "Quaternion|quaternion"):
            NoopSequenceExtractor(**settings(), noop=NoopOptions(motion_stats="rotation_mean")).fit_transform(x)

    def test_nested_validation_inactivity_and_json_frequency_choices(self):
        options = NoopOptions(spectral_stats="band_fractions", band="[0.5, 8]", bands="[[0.5, 4], [4, 8]]")
        validate_noop_settings({"noop": options, "imu_native_sampling_rate": 100})
        self.assertNotIn("noop__bands", inactive_noop_parameters({"noop": options}))
        self.assertIn("noop__acf_harmonic_tolerance", inactive_noop_parameters({"noop": options}))
        with self.assertRaisesRegex(InvalidExtractorParams, "Nyquist"):
            validate_noop_settings({"noop": options, "imu_native_sampling_rate": 10})
        with self.assertRaises(InvalidExtractorParams):
            NoopOptions(motion_stats=False).validate(100)

    def test_inactive_tuning_cannot_reject_an_unrelated_enabled_family(self):
        x = frame(50)
        # Those controls do not participate in motion statistics.
        options = NoopOptions(motion_stats="axis_rms", signal="unused", detrend="unused", max_gap_samples="unused")
        result = NoopSequenceExtractor(**settings(), noop=options).fit_transform(x)
        self.assertEqual(feature(result, "acc_x_rms_supported")[0], 1)
        # A report does not require a minimum sample count for an estimate.
        result = NoopSequenceExtractor(**settings(), noop=NoopOptions(quality="report", min_samples="unused")).fit_transform(x)
        self.assertEqual(result["noop_quality"][0]["real_samples"], 50)


if __name__ == "__main__":
    unittest.main()
