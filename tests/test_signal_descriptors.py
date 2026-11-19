"""Unit tests for modular src/signal_descriptors/ atoms, edge cases, and naming parity."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from base_utils import (
    DESCRIPTOR_CATALOG,
    NoopOptions,
    SignalDescriptorOptions,
    acf_cadence_estimate,
    acf_parabolic_fundamental,
    baevsky_stress_metrics,
    local_baseline_dynamic_amplitude,
    summarize_branch_window,
    summarize_imu_window,
    walch_dog_variability,
)
from utils import MultiModalSequenceExtractor, make_synthetic_cmi_df


class SignalDescriptorAtomTests(unittest.TestCase):
    def _make_src(self, acc: np.ndarray, fs: float = 100.0, rot: np.ndarray | None = None):
        n = len(acc)
        if rot is None:
            rot = np.zeros((n, 3), dtype=float)
        return {
            "raw": acc,
            "observed": np.isfinite(acc),
            "linear": acc - np.array([0.0, 0.0, 9.80665]),
            "rotation": rot,
            "rotation_observed": np.isfinite(rot).all(axis=1),
            "counter": np.arange(n, dtype=float),
            "dt": np.full(n, 1.0 / fs, dtype=float),
            "row_present": np.ones(n, dtype=bool),
        }

    def test_edge_cases_constant_short_single_sample_and_all_nan(self):
        opts = SignalDescriptorOptions(
            motion_stats="all",
            periodicity="acf_kimmitch",
            spectral_stats="all",
            quality="features",
            power_branches="all",
            min_samples=8,
        )
        # 1. Constant signal
        const_acc = np.tile([0.0, 0.0, 9.81], (100, 1))
        feats_c, _ = summarize_imu_window(self._make_src(const_acc), 100.0, opts)
        self.assertTrue(all(np.isfinite(v) for v in feats_c.values()))
        self.assertAlmostEqual(feats_c["noop_acc_ac_rms"], 0.0, places=6)
        self.assertAlmostEqual(feats_c["noop_dyn_amp"], 0.0, places=6)

        # 2. Short window (L=3) & single sample (L=1) & all-NaN
        for arr in (np.ones((3, 3)), np.ones((1, 3)), np.full((20, 3), np.nan)):
            feats, _ = summarize_imu_window(self._make_src(arr), 100.0, opts)
            self.assertTrue(all(np.isfinite(v) for v in feats.values()))
            self.assertEqual(feats["noop_acc_ac_rms_supported"], 0.0)
            self.assertEqual(feats["noop_cadence_hz_supported"], 0.0)

    def test_dyn_amp_is_calibration_offset_invariant(self):
        fs = 100.0
        t = np.arange(250) / fs
        acc = np.column_stack([
            1.2 * np.sin(2 * np.pi * 2.5 * t),
            0.7 * np.cos(2 * np.pi * 2.5 * t),
            9.81 + 0.4 * np.sin(2 * np.pi * 5.0 * t),
        ])
        d1 = local_baseline_dynamic_amplitude(acc, fs)
        d2 = local_baseline_dynamic_amplitude(acc + np.array([15.0, -23.0, 42.0]), fs)
        self.assertGreater(d1, 0.5)
        self.assertAlmostEqual(d1, d2, places=6)

    def test_l2_consistent_acc_gyro_coupling_ratio(self):
        fs = 100.0
        t = np.arange(200) / fs
        acc = np.column_stack([2.0 * np.sin(2 * np.pi * 2.0 * t), np.zeros(200), np.full(200, 9.81)])
        rot = np.column_stack([4.0 * np.abs(np.sin(2 * np.pi * 2.0 * t)), np.zeros(200), np.zeros(200)])
        opts = SignalDescriptorOptions(motion_stats="acc_gyro_coupling", min_samples=16)
        feats, _ = summarize_imu_window(self._make_src(acc, fs=fs, rot=rot), fs, opts)
        self.assertEqual(feats["noop_gyro_to_acc_ratio_supported"], 1.0)
        # Expected L2 RMS(omega) / (RMS(acc_mag - mean) + 1e-4)
        acc_m = np.linalg.norm(acc[1:], axis=1)
        rot_m = np.linalg.norm(rot[1:], axis=1)
        expected = float(np.sqrt(np.mean(rot_m ** 2)) / (float(np.std(acc_m)) + 1e-4))
        self.assertAlmostEqual(feats["noop_gyro_to_acc_ratio"], expected, places=5)

    def test_walch_dog_baevsky_and_acf_fundamental_precision(self):
        fs = 100.0
        t = np.arange(300) / fs
        sig = 10.0 + 1.5 * np.sin(2 * np.pi * 2.35 * t)
        dog = walch_dog_variability(sig, fs)
        self.assertGreater(dog, 0.05)

        amo, mxdmn, si = baevsky_stress_metrics(sig)
        self.assertGreater(amo, 0.0)
        self.assertGreater(mxdmn, 2.0)
        self.assertGreater(si, 0.0)

        hz, strength, ok = acf_cadence_estimate(sig, fs, band=(1.0, 4.0), min_strength=0.2)
        self.assertTrue(ok)
        self.assertAlmostEqual(hz, 2.35, delta=0.1)
        self.assertGreater(strength, 0.7)

        k_hz, sub_hz, k_str, conc, k_ok = acf_parabolic_fundamental(sig, fs, band=(1.0, 4.0), min_strength=0.2)
        self.assertTrue(k_ok)
        self.assertAlmostEqual(sub_hz, 2.35, delta=0.04)
        self.assertGreater(conc, 0.0)

    def test_canonical_and_legacy_naming_numerical_parity(self):
        df = make_synthetic_cmi_df(n_subjects=2, sequences_per_subject=2, n_timesteps=60, random_state=17)
        opts = NoopOptions(motion_stats="all", periodicity="acf_kimmitch", spectral_stats="all", quality="features", power_branches="all")
        ext_leg = MultiModalSequenceExtractor(window_seconds=0.6, static_backend="none", noop=opts, descriptor_naming="legacy")
        ext_can = MultiModalSequenceExtractor(window_seconds=0.6, static_backend="none", noop=opts, descriptor_naming="canonical")
        c_leg = ext_leg.fit(df).transform_chunks(df)
        c_can = ext_can.fit(df).transform_chunks(df)

        self.assertEqual(len(c_leg["noop_feature_names"]), len(c_can["noop_feature_names"]))
        self.assertTrue(all(n.startswith("noop_") for n in c_leg["noop_feature_names"]))
        self.assertFalse(any(n.startswith("noop_") for n in c_can["noop_feature_names"]))
        self.assertIn("walch_dog_variability", c_can["noop_feature_names"])
        self.assertIn("baevsky_stress_index", c_can["noop_feature_names"])
        self.assertIn("dyn_local_baseline_amplitude", c_can["noop_feature_names"])
        np.testing.assert_allclose(c_leg["static"], c_can["static"], rtol=1e-6, atol=1e-6)

        # Catalog provenance must never contain chatbot names
        forbidden = ("claude", "mistral", "perplexity", "openai", "grok", "deepseek", "qwen", "meta ai")
        for entry in DESCRIPTOR_CATALOG:
            blob = f"{entry.canonical_name} {entry.provenance} {entry.paper_or_ref}".lower()
            for word in forbidden:
                self.assertNotIn(word, blob)


if __name__ == "__main__":
    unittest.main()
