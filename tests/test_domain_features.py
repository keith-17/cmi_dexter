"""Comprehensive tests for M4/M5 domain features: handedness, filtering, presets, ToF, thermal, cross-modal, events & seq_agg."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from base_utils import (
    apply_filter_1d_or_2d,
    canonicalize_handedness_frame,
    detect_peak_anchor_starts,
    reject_tof_spikes,
)
from utils import MultiModalSequenceExtractor, make_synthetic_cmi_df


class DomainFeatureFamiliesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = make_synthetic_cmi_df(n_subjects=2, sequences_per_subject=3, n_timesteps=80, random_state=31)

    def test_handedness_reflection_swap_lr_flip_and_idempotence(self):
        df = self.df.iloc[:20].copy()
        df["handedness"] = 0  # left-handed
        # Place a distinct marker in tof_3 pixel (row 1, col 2 -> index 10)
        df["tof_3_v10"] = 33.0
        df["tof_5_v10"] = 180.0
        df["thm_3"] = 32.5
        df["thm_5"] = 30.1

        canon = canonicalize_handedness_frame(df, handedness_col="handedness")
        # 1. X acceleration negated
        np.testing.assert_allclose(canon["acc_x"].to_numpy(), -df["acc_x"].to_numpy())
        # 2. Quaternion conjugated (w, x, -y, -z) preserving unit norm and det(R)=+1
        np.testing.assert_allclose(canon["rot_w"].to_numpy(), df["rot_w"].to_numpy())
        np.testing.assert_allclose(canon["rot_x"].to_numpy(), df["rot_x"].to_numpy())
        np.testing.assert_allclose(canon["rot_y"].to_numpy(), -df["rot_y"].to_numpy())
        np.testing.assert_allclose(canon["rot_z"].to_numpy(), -df["rot_z"].to_numpy())
        q_c = canon[["rot_x", "rot_y", "rot_z", "rot_w"]].to_numpy()
        dets = np.linalg.det(Rotation.from_quat(q_c).as_matrix())
        np.testing.assert_allclose(dets, np.ones_like(dets), atol=1e-6)
        # 3. Thermopile 3 <-> 5 swapped
        np.testing.assert_allclose(canon["thm_3"].to_numpy(), df["thm_5"].to_numpy())
        np.testing.assert_allclose(canon["thm_5"].to_numpy(), df["thm_3"].to_numpy())
        # 4. ToF 3 <-> 5 swapped AND horizontally mirrored: (row 1, col 2) -> (row 1, col 5 -> index 13) on sensor 5!
        np.testing.assert_allclose(canon["tof_5_v13"].to_numpy(), 33.0)
        # 5. Applying reflection twice is exact identity (involutory)
        twice = canonicalize_handedness_frame(canon, handedness_col="handedness")
        np.testing.assert_allclose(twice["acc_x"].to_numpy(), df["acc_x"].to_numpy(), atol=1e-6)
        np.testing.assert_allclose(twice["tof_3_v10"].to_numpy(), df["tof_3_v10"].to_numpy(), atol=1e-6)

        # 6. Optional 180-deg Z mount flip rotates (acc_x, acc_y) -> (-acc_x, -acc_y)
        df_r = self.df.iloc[:10].copy()
        df_r["handedness"] = 1
        subj0 = str(df_r["subject"].iloc[0])
        flipped = canonicalize_handedness_frame(df_r, mount_flip_subjects=[subj0])
        np.testing.assert_allclose(flipped["acc_x"].to_numpy(), -df_r["acc_x"].to_numpy())
        np.testing.assert_allclose(flipped["acc_y"].to_numpy(), -df_r["acc_y"].to_numpy())

    def test_modalities_imu_only_emits_zero_thm_or_tof_columns(self):
        ext = MultiModalSequenceExtractor(
            window_seconds=0.5,
            modalities="imu",
            static_backend="catch22",
            static_channels="all",
            orientation_features=True,
            vibration_features=True,
            tof_spatial_features=True,
            thermal_features=True,
            cross_modal_features=True,
        )
        chunks = ext.fit(self.df).transform_chunks(self.df)
        self.assertEqual(list(chunks["branches"].keys()), ["imu"])
        for col in chunks["static_names"]:
            self.assertNotIn("thm", col.lower(), f"THM leaked into IMU-only static_names: {col}")
            self.assertNotIn("tof", col.lower(), f"ToF leaked into IMU-only static_names: {col}")

    def test_public_core_preset_lag_diffs_rot6d_and_tof_blocks(self):
        ext = MultiModalSequenceExtractor(
            window_seconds=0.5,
            imu_feature_preset="public_core_jerk",
            rotation_modes="quaternion|angular_velocity|rot6d|angular_distance|angular_accel|angular_jerk",
            lag_diff_scales=(1, 2, 5),
            tof_modes="raw|pooled_stats|sensor_stats|block_2x2|block_4x4",
            static_backend="none",
        )
        chunks = ext.fit(self.df).transform_chunks(self.df)
        imu_ch = chunks["channel_names"]["imu"]
        for expected in ("acc_mag", "lin_mag", "ang_speed", "angular_distance", "rot6d_0", "rot6d_5",
                         "ang_acc_x", "ang_jerk_x", "acc_x_diff1", "acc_x_diff5", "lin_z_diff2"):
            self.assertIn(expected, imu_ch)
        tof_ch = chunks["channel_names"]["tof"]
        self.assertEqual(tof_ch[0], "tof_raw_0")
        self.assertIn("tof_b2_s1_r0_c0", tof_ch)
        self.assertIn("tof_b2_s1_r0_c0_diff", tof_ch)
        self.assertIn("tof_b4_s5_r1_c1", tof_ch)
        self.assertTrue(np.isfinite(chunks["branches"]["imu"]).all())
        self.assertTrue(np.isfinite(chunks["branches"]["tof"]).all())

    def test_all_zero_phase_filters_and_tof_spike_rejection(self):
        fs = 100.0
        t = np.arange(200) / fs
        sig = np.column_stack([np.sin(2 * np.pi * 2.0 * t) + 0.2 * np.sin(2 * np.pi * 35.0 * t), np.cos(2 * np.pi * 2.0 * t)])
        for method in ("none", "lowpass", "highpass", "bandpass", "median", "savgol", "median_lowpass", "wavelet"):
            with self.subTest(method=method):
                out = apply_filter_1d_or_2d(sig, fs, method=method)
                self.assertEqual(out.shape, sig.shape)
                self.assertTrue(np.isfinite(out).all())

        # Spike rejection: single-frame 240 mm glitch on a 50 mm surface
        tof = np.full((30, 5), 50.0)
        tof[12, 2] = 240.0
        cleaned, frac = reject_tof_spikes(tof, spike_threshold=80.0, kernel_size=3)
        self.assertTrue(np.isnan(cleaned[12, 2]))
        self.assertAlmostEqual(cleaned[11, 2], 50.0)
        self.assertGreater(frac, 0.0)

    def test_all_seven_domain_families_multiscale_and_seq_agg_end_to_end(self):
        ext = MultiModalSequenceExtractor(
            window_seconds=0.6,
            stride_ratio=0.5,
            tof_modes="raw|pooled_stats|sensor_stats",
            thm_modes="raw|centered_diff",
            rotation_modes="quaternion|angular_velocity",
            static_backend="none",
            filter_mode="savgol",
            tof_spike_threshold=120.0,
            baseline_mode="first_n_seconds",
            window_seconds_list=(0.2, 0.4),
            sequence_aggregation_features=True,
            orientation_features=True,
            vibration_features=True,
            tof_spatial_features=True,
            tof_temporal_features=True,
            thermal_features=True,
            cross_modal_features=True,
            events_complexity_features=True,
        )
        chunks = ext.fit(self.df).transform_chunks(self.df)
        names = set(chunks["static_names"])
        expected_prefixes = (
            "orientation__", "vib__", "tof_spatial__", "tof_temp__",
            "thm_dyn__", "xmodal__", "evcomp__", "mscale__", "seq__", "seq_ctx__",
        )
        for pref in expected_prefixes:
            self.assertTrue(any(n.startswith(pref) for n in names), f"Missing prefix {pref!r}")
        self.assertTrue(np.isfinite(chunks["static"]).all())

        # Verify peak-anchored window starts helper
        pulse = np.zeros(200)
        pulse[60] = 5.0
        pulse[140] = 6.0
        anchors = detect_peak_anchor_starts(pulse, fs=100.0, window_samples=40, max_anchors=4)
        self.assertGreaterEqual(len(anchors), 2)

    def test_synthetic_math_precision_tof_blob_approach_thermal_and_mixed_rate_lag(self):
        """Verify quantitative physical accuracy of ToF spatial blob, ToF velocity, thermal flux & lagged corr."""
        fs_row = 100.0
        n = 200  # 2.0 seconds -> 200 IMU samples, 40 THM/ToF samples at 20 Hz
        t = np.arange(n) / fs_row
        # IMU burst peaks around t = 0.8 s; ToF closest approach peaks at t = 1.0 s (+0.2 s lag after IMU)
        acc_env = np.exp(-0.5 * ((t - 0.8) / 0.15) ** 2)
        t_hold = np.floor(np.arange(n) / 5.0) * (5.0 / fs_row)
        tof_dist_t = 220.0 - 170.0 * np.exp(-0.5 * ((t_hold - 1.0) / 0.15) ** 2)  # drops from 220 mm to 50 mm!
        thm_ramp = 30.0 + 2.0 * (t_hold / 2.0)  # +1.0 degC/s linear warming

        data = {
            "sequence_id": ["S_MATH"] * n,
            "subject": ["SUBJ_M"] * n,
            "sequence_counter": np.arange(n),
            "acc_x": 3.0 * acc_env,
            "acc_y": np.zeros(n),
            "acc_z": np.full(n, 9.80665),
            "rot_w": np.ones(n),
            "rot_x": np.zeros(n),
            "rot_y": np.zeros(n),
            "rot_z": np.zeros(n),
        }
        for s in range(1, 6):
            data[f"thm_{s}"] = thm_ramp + 0.1 * (s - 1)
        # Put a single tight contact blob centered at (row=2, col=3 -> pixel 19) on all 5 sensors
        for s in range(1, 6):
            for p in range(64):
                r, c = divmod(p, 8)
                if (r, c) in {(2, 3), (2, 4), (3, 3)}:
                    data[f"tof_{s}_v{p}"] = tof_dist_t
                else:
                    data[f"tof_{s}_v{p}"] = np.full(n, 240.0)
        df_math = pd.DataFrame(data)

        ext = MultiModalSequenceExtractor(
            window_seconds=2.0,
            tof_modes="raw|pooled_stats|sensor_stats",
            thm_modes="raw",
            static_backend="none",
            tof_spatial_features=True,
            tof_temporal_features=True,
            thermal_features=True,
            cross_modal_features=True,
        )
        raw = ext.raw_chunks(df_math)
        feat = ext.featurize(raw)
        vals = {k: float(feat["static"][0, j]) for j, k in enumerate(feat["static_names"])}

        # 1. ToF spatial blob located at row 2, col 3 with 1 connected component
        self.assertEqual(vals["tof_spatial__supported"], 1.0)
        self.assertIn(int(round(vals["tof_spatial__closest_row"])), (2, 3))
        self.assertIn(int(round(vals["tof_spatial__closest_col"])), (3, 4))
        self.assertGreater(vals["tof_spatial__n_components_mean"], 0.0)

        # 2. ToF temporal approach speed strictly positive (3/64 pixels drop 170 mm over 0.15 s -> ~31.6 mm/s pooled)
        self.assertEqual(vals["tof_temp__supported"], 1.0)
        self.assertGreater(vals["tof_temp__approach_vel_max_mm_s"], 20.0)
        self.assertAlmostEqual(vals["tof_temp__time_to_closest_s"], 1.0, delta=0.25)

        # 3. Thermal positive flux and positive mean dT/dt (~1.0 degC/s)
        self.assertEqual(vals["thm_dyn__supported"], 1.0)
        self.assertGreater(vals["thm_dyn__pos_flux_c_s"], 1.0)
        self.assertAlmostEqual(vals["thm_dyn__s1_dt_mean_c_s"], 1.0, delta=0.2)

        # 4. Mixed-rate cross-modal lagged correlation detects IMU leading ToF by +0.2 s!
        self.assertEqual(vals["xmodal__imu_tof_supported"], 1.0)
        self.assertGreater(vals["xmodal__imu_tof__max_abs_lag_corr"], 0.85)
        self.assertAlmostEqual(vals["xmodal__imu_tof__best_lag_s"], 0.2, delta=0.05)


if __name__ == "__main__":
    unittest.main()
