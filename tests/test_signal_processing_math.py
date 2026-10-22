"""Numerical regression tests for signal-processing feature definitions."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from base_utils_sol import (  # noqa: E402
    CWTExtractor,
    IMUExtractor,
    InvalidExtractorParams,
    RotationExtractor,
    STFTExtractor,
    SequenceExtractor,
    SignalCleaner,
    TOFExtractor,
)


def sequence_frame(signal: np.ndarray, *, fs: float, sequence_id: str = "s") -> pd.DataFrame:
    return pd.DataFrame({
        "sequence_id": sequence_id,
        "sequence_counter": np.arange(len(signal)),
        "acc_x": signal,
    })


class STFTMathTests(unittest.TestCase):
    def test_linear_psd_matches_mean_square_and_scales_quadratically(self):
        fs = 100.0
        time = np.arange(1000) / fs
        sine = np.sin(2 * np.pi * 5 * time)
        extractor = STFTExtractor(
            sampling_rate=fs,
            nperseg=200,
            noverlap=100,
            scaling="density",
            use_log_scale=True,
            frequency_bands={"five_hz": (4.0, 6.0)},
        ).fit(sequence_frame(sine, fs=fs))

        one = extractor._compute_stft_features(sine)
        two = extractor._compute_stft_features(2 * sine)

        self.assertAlmostEqual(one["stft_mean_power"], np.mean(sine ** 2), delta=0.025)
        self.assertAlmostEqual(two["stft_mean_power"] / one["stft_mean_power"], 4.0, delta=0.08)
        self.assertAlmostEqual(one["stft_peak_frequency_hz"], 5.0, delta=0.5)
        self.assertGreater(one["stft_band_five_hz_mean_power"], 0.35)
        self.assertIn("stft_log1p_psd_mean", one)
        self.assertNotIn("stft_total_power", one)

    def test_short_signals_keep_schema_and_unresolved_bands(self):
        extractor = STFTExtractor(
            sampling_rate=20,
            nperseg=32,
            noverlap=16,
            frequency_bands={"empty": (2.1, 2.4)},
        ).fit(sequence_frame(np.ones(2), fs=20))
        short = extractor._compute_stft_features(np.ones(2))
        long = extractor._compute_stft_features(np.ones(50))
        self.assertEqual(set(short), set(long))
        self.assertEqual(short["stft_band_empty_resolved"], 0.0)
        self.assertEqual(short["stft_band_empty_mean_power"], 0.0)

    def test_unresolved_counter_gaps_are_rejected(self):
        frame = sequence_frame(np.ones(4), fs=20)
        frame["sequence_counter"] = [0, 1, 3, 4]
        extractor = STFTExtractor(
            sampling_rate=20, nperseg=4, noverlap=2, signal_columns=["acc_x"]
        ).fit(frame)
        with self.assertRaisesRegex(ValueError, "uniformly spaced"):
            extractor.transform(frame)


class CWTMathTests(unittest.TestCase):
    def test_nonzero_response_scale_frequency_mapping_and_empty_behavior(self):
        fs = 20.0
        time = np.arange(400) / fs
        sine = np.sin(2 * np.pi * 2 * time)
        frame = sequence_frame(sine, fs=fs)
        extractor = CWTExtractor(
            wavelet="morl",
            widths=np.array([2.5, 3.0, 4.0, 6.0, 8.0, 12.0]),
            sampling_rate=fs,
            signal_columns=["acc_x"],
        ).fit(frame)
        features = extractor._compute_cwt_features(sine)
        empty = extractor._compute_cwt_features(np.array([np.nan, np.nan]))

        self.assertGreater(features["cwt_mean_magnitude"], 0.05)
        self.assertGreater(features["cwt_coefficient_power_mean"], 0.0)
        self.assertAlmostEqual(
            extractor.scale_frequencies_hz_[0] / extractor.scale_frequencies_hz_[-1],
            12.0 / 2.5,
            delta=1e-10,
        )
        self.assertGreater(features["cwt_dominant_frequency_hz"], 0.0)
        self.assertEqual(empty["cwt_mean_magnitude"], 0.0)
        self.assertEqual(empty["cwt_valid_coefficient_fraction"], 0.0)
        self.assertGreater(features["cwt_boundary_fraction"], 0.0)

    def test_aliasing_and_invalid_cwt_configuration_raise(self):
        frame = sequence_frame(np.ones(32), fs=20)
        with self.assertRaisesRegex(InvalidExtractorParams, "above Nyquist"):
            CWTExtractor(
                widths=np.array([1.0]), sampling_rate=20, signal_columns=["acc_x"]
            ).fit(frame)
        with self.assertRaisesRegex(InvalidExtractorParams, "sampling_rate"):
            CWTExtractor(
                widths=np.array([2.0]), sampling_rate=0, signal_columns=["acc_x"]
            ).fit(frame)


class RotationMathTests(unittest.TestCase):
    @staticmethod
    def _frame_from_wxyz(quaternions: np.ndarray, sequence_ids=None) -> pd.DataFrame:
        if sequence_ids is None:
            sequence_ids = ["s"] * len(quaternions)
        return pd.DataFrame({
            "sequence_id": sequence_ids,
            "sequence_counter": np.arange(len(quaternions)),
            "rot_w": quaternions[:, 0],
            "rot_x": quaternions[:, 1],
            "rot_y": quaternions[:, 2],
            "rot_z": quaternions[:, 3],
            "rot_dt": 0.05,
        })

    def test_rot6d_matches_independent_complete_rotation_matrices(self):
        rotations = Rotation.from_rotvec(np.array([
            [0.0, 0.0, 0.0],
            [0.4, -0.6, 0.2],
            [0.8, 0.1, -0.5],
        ]))
        xyzw = rotations.as_quat()
        wxyz = xyzw[:, [3, 0, 1, 2]]
        frame = self._frame_from_wxyz(wxyz, ["a", "b", "c"])
        output = RotationExtractor(rotation_modes="rot6d").fit_transform(frame)

        for index, matrix in enumerate(rotations.as_matrix()):
            expected = np.concatenate([matrix[:, 0], matrix[:, 1]])
            actual = output.iloc[index].to_numpy(dtype=float)
            np.testing.assert_allclose(actual, expected, atol=1e-12)
            np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(matrix), 1.0, places=12)
        self.assertNotAlmostEqual(output.iloc[1]["rot6d_c1_z"], 0.0, places=6)
        self.assertNotAlmostEqual(output.iloc[1]["rot6d_c2_z"], 0.0, places=6)

    def test_angular_velocity_is_sign_invariant_and_rad_per_second(self):
        fs = 20.0
        speed = 1.2
        time = np.arange(21) / fs
        theta = speed * time
        q = np.column_stack([
            np.cos(theta / 2), np.zeros(len(theta)), np.zeros(len(theta)),
            np.sin(theta / 2),
        ])
        q[1::2] *= -1.0
        frame = self._frame_from_wxyz(q)
        frame["rot_dt"] = 1.0 / fs
        frame.loc[0, "rot_dt"] = 1.0 / fs
        output = RotationExtractor(rotation_modes="angular_velocity").fit_transform(frame)
        self.assertAlmostEqual(output["rot_angular_velocity_body_z_rad_s"].iloc[0], 0.0)
        np.testing.assert_allclose(
            output["rot_angular_velocity_body_z_rad_s"].iloc[1:],
            speed,
            atol=1e-10,
        )

        stationary = np.tile([1.0, 0.0, 0.0, 0.0], (10, 1))
        stationary[1::2] *= -1.0
        still = self._frame_from_wxyz(stationary)
        still["rot_dt"] = 1.0 / fs
        np.testing.assert_allclose(
            RotationExtractor(rotation_modes="angular_velocity")
            .fit_transform(still).to_numpy(),
            0.0,
            atol=1e-12,
        )

    def test_invalid_quaternion_is_rejected(self):
        frame = self._frame_from_wxyz(np.array([[0.0, 0.0, 0.0, 0.0]]))
        extractor = RotationExtractor().fit(frame)
        with self.assertRaisesRegex(ValueError, "finite and nonzero"):
            extractor.transform(frame)


class TimingAndIntegrationTests(unittest.TestCase):
    def test_gravity_removed_in_world_and_body_frames(self):
        angle = np.pi / 2
        rotation = Rotation.from_rotvec([angle, 0.0, 0.0])
        q_xyzw = rotation.as_quat()
        q_wxyz = q_xyzw[[3, 0, 1, 2]]
        gravity_world = np.array([0.0, 0.0, 9.80665])
        acc_body = rotation.inv().apply(gravity_world)
        frame = pd.DataFrame({
            "sequence_id": ["s"],
            "sequence_counter": [0],
            "acc_x": [acc_body[0]],
            "acc_y": [acc_body[1]],
            "acc_z": [acc_body[2]],
            "rot_w": [q_wxyz[0]],
            "rot_x": [q_wxyz[1]],
            "rot_y": [q_wxyz[2]],
            "rot_z": [q_wxyz[3]],
        })
        world = SignalCleaner(
            native_sampling_rate=20, linear_acc_mode="baseline",
            use_highpass_fallback=False, linear_acc_frame="world",
        ).fit_transform(frame)
        body = SignalCleaner(
            native_sampling_rate=20, linear_acc_mode="baseline",
            use_highpass_fallback=False, linear_acc_frame="body",
        ).fit_transform(frame)
        np.testing.assert_allclose(world[["lin_acc_x", "lin_acc_y", "lin_acc_z"]], 0.0, atol=1e-12)
        np.testing.assert_allclose(body[["lin_acc_x", "lin_acc_y", "lin_acc_z"]], 0.0, atol=1e-12)

    def test_counter_intervals_drive_jerk_without_clipping(self):
        frame = pd.DataFrame({
            "sequence_id": ["s"] * 4,
            "sequence_counter": [10, 11, 13, 16],
            "acc_x": [0.0, 2.0, 6.0, 12.0],
        })
        cleaner = SignalCleaner(native_sampling_rate=20).fit(frame)
        cleaned = cleaner.transform(frame)
        np.testing.assert_allclose(cleaned["dt"], [0.05, 0.05, 0.1, 0.15])
        jerk = IMUExtractor(acc_modes="jerk").fit_transform(cleaned)
        np.testing.assert_allclose(jerk["acc_x_jerk"], [0.0, 40.0, 40.0, 40.0])

    def test_duplicate_counters_and_cross_sequence_imputation_are_handled(self):
        frame = pd.DataFrame({
            "sequence_id": ["a", "a", "b", "b"],
            "sequence_counter": [0, 1, 0, 1],
            "acc_x": [1.0, np.nan, np.nan, 9.0],
        })
        cleaned = SignalCleaner(
            native_sampling_rate=20, interp_mode="ffill"
        ).fit_transform(frame)
        np.testing.assert_allclose(cleaned["acc_x"], [1.0, 1.0, 9.0, 9.0])
        invalid = frame.copy()
        invalid.loc[1, "sequence_counter"] = 0
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            SignalCleaner().fit_transform(invalid)

    def test_resampling_preserves_source_points_and_source_intervals(self):
        fs = 20.0
        time = np.arange(20) / fs
        sine = np.sin(2 * np.pi * 2 * time)
        frame = sequence_frame(sine, fs=fs)
        frame["dt"] = 1.0 / fs
        frame["rot_dt"] = 1.0 / fs
        extractor = SequenceExtractor(
            output_format="frame",
            resample_modalities=True,
            imu_native_sampling_rate=20,
            imu_target_sampling_rate=10,
            rot_native_sampling_rate=20,
            rot_target_sampling_rate=10,
            tof_native_sampling_rate=5,
            tof_target_sampling_rate=5,
            thm_native_sampling_rate=5,
            thm_target_sampling_rate=5,
        )
        resampled = extractor._maybe_resample(frame)
        self.assertEqual(len(resampled), len(frame))
        self.assertGreater(float(resampled["acc_x"].std()), 0.25)
        self.assertTrue(np.allclose(resampled["dt"], frame["dt"]))
        self.assertTrue(np.allclose(resampled["sequence_counter"], frame["sequence_counter"]))

    def test_quaternion_resampling_uses_slerp_on_rotation_path(self):
        fs = 20.0
        time = np.arange(20) / fs
        angle = 1.2 * time
        frame = pd.DataFrame({
            "sequence_id": ["s"] * len(time),
            "sequence_counter": np.arange(len(time)),
            "rot_w": np.cos(angle / 2),
            "rot_x": np.zeros(len(time)),
            "rot_y": np.zeros(len(time)),
            "rot_z": np.sin(angle / 2),
        })
        extractor = SequenceExtractor(
            output_format="frame",
            resample_modalities=True,
            rot_native_sampling_rate=20,
            rot_target_sampling_rate=10,
        )

        resampled = extractor._maybe_resample(frame)
        actual = resampled[["rot_w", "rot_x", "rot_y", "rot_z"]].to_numpy()
        expected = frame[["rot_w", "rot_x", "rot_y", "rot_z"]].to_numpy()
        np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(np.abs(np.sum(actual * expected, axis=1)), 1.0, atol=1e-12)

        single = frame.iloc[:1].copy()
        one_sample = SequenceExtractor(
            output_format="frame",
            resample_modalities=True,
            rot_native_sampling_rate=20,
            rot_target_sampling_rate=40,
        )._maybe_resample(single)
        self.assertEqual(len(one_sample), 1)
        self.assertAlmostEqual(
            np.linalg.norm(one_sample[["rot_w", "rot_x", "rot_y", "rot_z"]].to_numpy()),
            1.0,
            places=12,
        )

    def test_tof_resampling_preserves_invalidity_without_filter_leakage(self):
        frame = pd.DataFrame({
            "sequence_id": ["s"] * 20,
            "sequence_counter": np.arange(20),
            "tof_1_v0": np.full(20, 10.0),
        })
        frame.loc[5, "tof_1_v0"] = -1.0
        extractor = SequenceExtractor(
            output_format="frame",
            resample_modalities=True,
            tof_native_sampling_rate=20,
            tof_target_sampling_rate=10,
        )

        resampled = extractor._maybe_resample(frame)
        self.assertEqual(resampled.loc[5, "tof_1_v0"], -1.0)
        valid_values = resampled.loc[resampled["tof_1_v0"] != -1, "tof_1_v0"]
        self.assertTrue(np.isfinite(valid_values).all())
        self.assertLess(float(valid_values.max()), 20.0)


class TOFValidityTests(unittest.TestCase):
    def test_valid_statistics_are_separate_from_imputed_statistics(self):
        frame = pd.DataFrame({
            "sequence_id": ["s"],
            "tof_1_v0": [10.0],
            "tof_1_v1": [-1.0],
        })
        out = TOFExtractor(
            tof_modes="sensor_stats", tof_fill_mode="far_500"
        ).fit_transform(frame)
        self.assertAlmostEqual(out["tof_1_imputed_mean"].iloc[0], 255.0)
        self.assertAlmostEqual(out["tof_1_valid_mean"].iloc[0], 10.0)
        self.assertAlmostEqual(out["tof_1_valid_fraction"].iloc[0], 0.5)

    def test_spatial_pooling_uses_pixel_order_and_keeps_valid_fraction(self):
        names = [f"tof_1_v{i}" for i in range(64)]
        frame = pd.DataFrame(
            {"sequence_id": ["s"], **{name: [float(i)] for i, name in enumerate(names)}}
        )
        frame["tof_1_v0"] = -1.0
        frame = frame[["sequence_id"] + names[::-1]]
        out = TOFExtractor(tof_modes="pooled", tof_fill_mode="far_500").fit_transform(frame)
        self.assertAlmostEqual(out["tof_1_pool_imputed_0"].iloc[0], 129.5)
        self.assertAlmostEqual(out["tof_1_pool_valid_fraction_0"].iloc[0], 0.75)


if __name__ == "__main__":
    unittest.main()
