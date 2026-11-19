"""Unit tests for unified physics, base_utils hardening, and ProblematicSequenceFilter preservation."""
from __future__ import annotations

import inspect
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import base_utils as bu
from base_utils import (
    ALGO_VERSION,
    FEATURE_VERSION,
    GRAVITY,
    clean_tof_array,
    quaternions_to_euler,
    quaternions_to_rot6d,
    remove_gravity_quaternion,
    stft_frame_coverage,
)
from utils import MultiModalSequenceExtractor, make_synthetic_cmi_df


class BaseUtilsPhysicsHardeningTests(unittest.TestCase):
    def test_stft_and_cwt_use_column_sampling_rate_for_10hz_sine_at_100hz(self):
        fs = 100.0
        t = np.arange(300) / fs
        sig = np.sin(2.0 * np.pi * 10.0 * t)
        df = pd.DataFrame({
            "sequence_id": ["s1"] * len(t),
            "sequence_counter": np.arange(len(t)),
            "dt": np.full(len(t), 1.0 / fs),
            "acc_x": sig,
        })
        stft_ext = bu.STFTExtractor(
            acc_modes="psd",
            sampling_rate=20.0,
            column_sampling_rates={"acc_x": 100.0},
            nperseg=64,
            noverlap=32,
            extended_spectral=True,
            return_spectrogram=True,
            frequency_bands={"low": (0.5, 4.0), "mid": (4.0, 15.0)},
        )
        out = stft_ext.fit_transform(df)
        dom_col = [c for c in out.columns if "peak_frequency_hz" in c][0]
        dom_freqs = out[dom_col].to_numpy()
        # Peak frequency must land near 10 Hz (NOT 2 Hz from the legacy 20 Hz default!)
        self.assertAlmostEqual(float(np.median(dom_freqs)), 10.0, delta=2.0)
        self.assertTrue(any("rolloff_85" in c for c in out.columns))
        self.assertTrue(any("flatness" in c for c in out.columns))
        self.assertTrue(any("crest" in c for c in out.columns))
        self.assertGreater(len(stft_ext.last_spectrograms_), 0)

        cwt_ext = bu.CWTExtractor(
            acc_modes="raw",
            sampling_rate="auto",
            n_scales=16,
        )
        cwt_out = cwt_ext.fit_transform(df)
        self.assertGreater(cwt_out.shape[1], 0)
        self.assertTrue(np.isfinite(cwt_out.to_numpy()).all())

    def test_stft_edge_coverage_corrects_power_and_magnitude(self):
        cov = stft_frame_coverage(np.array([0.0, 0.5, 1.0]), signal_length=100, sampling_rate=100.0, nperseg=32, window_type="hann")
        self.assertLess(cov[0], 0.99)
        self.assertAlmostEqual(cov[1], 1.0, places=5)
        self.assertLess(cov[2], 0.99)

    def test_imu_extractor_gravity_gated_integration_and_smoothing_split(self):
        fs = 100.0
        n = 200
        t = np.arange(n) / fs
        df = pd.DataFrame({
            "sequence_id": ["s1"] * n,
            "sequence_counter": np.arange(n),
            "dt": np.full(n, 1.0 / fs),
            "acc_x": np.sin(2.0 * np.pi * 2.0 * t),
            "acc_y": np.zeros(n),
            "acc_z": np.full(n, GRAVITY),
            "rot_w": np.ones(n),
            "rot_x": np.zeros(n),
            "rot_y": np.zeros(n),
            "rot_z": np.zeros(n),
        })
        # Raw gravity integration must reject with InvalidExtractorParams unless explicitly allowed
        imu_raw_gate = bu.IMUExtractor(
            acc_modes="raw|velocity",
            allow_raw_gravity_integration=False,
            integration_highpass_hz=None,
            smoothing_window_size=5,
        )
        with self.assertRaises(bu.InvalidExtractorParams):
            imu_raw_gate.fit_transform(df)

        # With linear_acc + zero-phase integration highpass, no quadratic gravity drift!
        cleaned = bu.SignalCleaner(
            linear_acc_mode="baseline",
            highpass_window_size=15,
            native_sampling_rate=fs,
        ).fit_transform(df)
        imu_lin = bu.IMUExtractor(
            acc_modes="raw",
            linear_acc_modes="raw|velocity|displacement",
            integration_highpass_hz=0.3,
            smoothing_window_size=5,
        )
        res_lin = imu_lin.fit_transform(cleaned)
        disp_cols = [c for c in res_lin.columns if "disp" in c]
        self.assertGreater(len(disp_cols), 0)
        self.assertTrue(np.isfinite(res_lin[disp_cols].to_numpy()).all())
        self.assertLess(float(np.abs(res_lin[disp_cols].to_numpy()).max()), 2.0)

    def test_motion_filter_kalman_1d_and_deprecated_extended_kalman(self):
        df = pd.DataFrame({
            "sequence_id": ["s1"] * 40,
            "sequence_counter": np.arange(40),
            "acc_x": np.sin(np.linspace(0, 4, 40)) + 0.1 * np.cos(np.linspace(0, 20, 40)),
        })
        k1 = bu.MotionFilter(motion_filter_mode="kalman_1d").fit_transform(df)
        k_legacy = bu.MotionFilter(motion_filter_mode="kalman").fit_transform(df)
        np.testing.assert_allclose(k1["acc_x"].to_numpy(), k_legacy["acc_x"].to_numpy(), rtol=1e-12, atol=1e-12)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            k_ekf = bu.MotionFilter(motion_filter_mode="extended_kalman").fit_transform(df)
        self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))
        np.testing.assert_allclose(k_ekf["acc_x"].to_numpy(), k1["acc_x"].to_numpy(), rtol=1e-12, atol=1e-12)

    def test_scipy_rotation_rot6d_and_euler_and_partial_nan_gravity_removal(self):
        rot = Rotation.from_euler("xyz", [[0.0, 0.0, 0.0], [np.pi / 2, 0.0, 0.0], [0.0, np.pi / 4, 0.3]])
        xyzw = rot.as_quat()
        wxyz = xyzw[:, [3, 0, 1, 2]]
        r6 = quaternions_to_rot6d(wxyz)
        mats = rot.as_matrix()
        np.testing.assert_allclose(r6[:, :3], mats[:, :, 0], atol=1e-7)
        np.testing.assert_allclose(r6[:, 3:], mats[:, :, 1], atol=1e-7)
        eul = quaternions_to_euler(wxyz, seq="xyz", unwrap=False)
        np.testing.assert_allclose(eul, rot.as_euler("xyz"), atol=1e-7)

        # Partial NaN quaternion row: valid rows MUST still get exact quaternion gravity removal!
        n = 30
        q_seq = np.tile(wxyz[1], (n, 1))
        q_seq[15] = np.nan
        # Gravity in body frame for rot[1] (90 deg about X): R.inv().apply([0,0,g])
        g_body = rot[1].inv().apply(np.array([0.0, 0.0, GRAVITY]))
        true_dyn = np.column_stack([np.sin(np.linspace(0, 6, n)), np.zeros(n), np.zeros(n)])
        acc = g_body[None, :] + true_dyn

        lin_body = remove_gravity_quaternion(acc, q_seq, frame="body", fs=100.0)
        lin_world = remove_gravity_quaternion(acc, q_seq, frame="world", fs=100.0)
        valid_idx = [i for i in range(n) if i != 15]
        np.testing.assert_allclose(lin_body[valid_idx], true_dyn[valid_idx], atol=1e-5)
        expected_world = rot[1].apply(true_dyn[valid_idx])
        np.testing.assert_allclose(lin_world[valid_idx], expected_world, atol=1e-5)

    def test_clean_tof_array_preserves_zero_and_filters_configurable_sentinels(self):
        raw = np.array([[-1.0, 0.0, 42.0, 255.0, np.nan]])
        obs_default, filled_default, mask_default = clean_tof_array(raw)
        self.assertFalse(mask_default[0, 0])  # -1 is invalid
        self.assertTrue(mask_default[0, 1])   # 0 is VALID near-contact reading
        self.assertTrue(mask_default[0, 2])
        self.assertTrue(mask_default[0, 3])   # 255 valid under default (-1.0,)
        self.assertFalse(mask_default[0, 4])  # NaN invalid
        self.assertEqual(filled_default[0, 0], 500.0)

        _, _, mask_custom = clean_tof_array(raw, invalid_sentinels=(-1.0, 255.0))
        self.assertFalse(mask_custom[0, 3])

    def test_half_padded_window_invariance_and_scale_observed_only(self):
        df = make_synthetic_cmi_df(n_subjects=2, sequences_per_subject=2, n_timesteps=40, random_state=3)
        # 1.0 s window at 100 Hz = 100 samples, so 40-step sequences are 60% padded!
        ext_obs = MultiModalSequenceExtractor(
            window_seconds=1.0,
            static_backend="none",
            scale_observed_only=True,
            orientation_features=True,
            vibration_features=True,
        )
        chunks = ext_obs.fit(df).transform_chunks(df)
        imu = chunks["branches"]["imu"]
        mask = chunks["masks"]["imu"]
        # Padded region must remain strictly 0.0 after scaling
        self.assertEqual(float(np.abs(imu[~mask]).max()), 0.0)
        self.assertTrue(np.isfinite(chunks["static"]).all())
        self.assertEqual(ALGO_VERSION, "4.0.0")
        self.assertEqual(FEATURE_VERSION, "4.0.0")

    def test_problematic_sequence_filter_remains_untouched(self):
        self.assertTrue(hasattr(bu, "ProblematicSequenceFilter"))
        src = inspect.getsource(bu.ProblematicSequenceFilter)
        self.assertIn("class ProblematicSequenceFilter", src)
        self.assertIn("def problematic_mask", src)


if __name__ == "__main__":
    unittest.main()
