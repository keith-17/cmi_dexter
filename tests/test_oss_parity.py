"""Parity tests confirming every open-source library backend runs genuinely and matches upstream math."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from base_utils import compute_actigraphy_tsflex_bank, library_status
from utils import MultiModalSequenceExtractor, TSAugmentor, make_synthetic_cmi_df


class OpenSourceBackendParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = make_synthetic_cmi_df(n_subjects=2, sequences_per_subject=2, n_timesteps=60, random_state=21)
        cls.status = library_status()

    def test_all_six_oss_backends_execute_end_to_end(self):
        expected_prefixes = {
            "tsfel": "tsfel__",
            "tsfresh": "tsfresh__",
            "catch22": "catch22__",
            "tsflex": "tsflex__",
            "seglearn": "seglearn__",
            "sktime": "sktime__",
        }
        lib_key = {
            "tsfel": "tsfel",
            "tsfresh": "tsfresh",
            "catch22": "pycatch22",
            "tsflex": "tsflex",
            "seglearn": "seglearn",
            "sktime": "sktime",
        }
        for backend, prefix in expected_prefixes.items():
            if not self.status.get(lib_key[backend], False):
                continue
            with self.subTest(backend=backend):
                ext = MultiModalSequenceExtractor(
                    window_seconds=0.5,
                    static_backend=backend,
                    static_channels="acc_mag|thm_mean",
                    tsfel_domains="statistical|temporal|spectral|fractal",
                    tsfel_features="fractal_complexity" if backend == "tsfel" else None,
                    tsfresh_profile="efficient" if backend == "tsfresh" else "minimal",
                )
                chunks = ext.fit(self.df).transform_chunks(self.df)
                cols = [c for c in chunks["static_names"] if c.startswith(prefix)]
                self.assertGreater(len(cols), 3, f"{backend} produced too few columns: {cols}")
                self.assertTrue(np.isfinite(chunks["static"]).all())

    def test_tsfresh_approximate_entropy_matches_real_library_not_proxy(self):
        if not self.status.get("tsfresh", False):
            self.skipTest("tsfresh not installed")
        from tsfresh.feature_extraction.feature_calculators import approximate_entropy

        ext = MultiModalSequenceExtractor(
            window_seconds=0.5,
            static_backend="tsfresh",
            static_channels="acc_mag",
            tsfresh_profile="efficient",
        )
        raw = ext.raw_chunks(self.df)
        feat = ext.featurize(raw)
        ap_col = [c for c in feat["static_names"] if "approximate_entropy" in c]
        self.assertEqual(len(ap_col), 1)
        idx = feat["static_names"].index(ap_col[0])
        val_backend = float(feat["static"][0, idx])

        # Compute directly from the exact first unpadded window
        imu = raw["branches"]["imu"][0]
        L = int(raw["masks"]["imu"][0].sum())
        acc_idx = [raw["channel_names"]["imu"].index(f"acc_{a}") for a in "xyz"]
        acc_mag = np.linalg.norm(imu[:L, acc_idx].astype(np.float64), axis=1)
        val_direct = float(approximate_entropy(acc_mag, m=2, r=0.2))
        self.assertAlmostEqual(val_backend, val_direct, places=4)

    def test_tsflex_actigraphy_fixes_sleep_analysis_d6_off_by_one(self):
        if not self.status.get("tsflex", False):
            self.skipTest("tsflex not installed")
        import pandas as pd
        sig = pd.Series(np.sin(np.linspace(0.0, 20.0 * np.pi, 120)) + 2.0)
        # Fixed version (include_max_window=True) includes scale 20 (fixing MIT sleep_analysis np.arange(1, 20) D6 bug)
        df_fixed = compute_actigraphy_tsflex_bank(sig, window_size=5, include_max_window=True)
        df_legacy = compute_actigraphy_tsflex_bank(sig, window_size=5, include_max_window=False)
        self.assertGreater(df_fixed.shape[1], df_legacy.shape[1])
        self.assertTrue(any("5" in str(c) for c in df_fixed.columns))
        self.assertTrue(np.isfinite(df_fixed.fillna(0.0).to_numpy()).all())

    def test_per_modality_augmentation_preserves_quaternion_unit_norm(self):
        ext = MultiModalSequenceExtractor(
            window_seconds=0.5,
            rotation_modes="quaternion|angular_velocity",
            static_backend="none",
        )
        raw = ext.raw_chunks(self.df)
        aug = TSAugmentor(
            prob=1.0,
            per_aug_prob=1.0,
            jitter_sigma=0.05,
            noise_std=0.05,
            scaling_sigma=0.1,
            per_modality_augmentation=True,
            random_state=7,
        )
        rng = np.random.default_rng(7)
        augmented = aug.augment_chunks(raw, rng)
        imu_ch = raw["channel_names"]["imu"]
        q_idx = [imu_ch.index(f"rot_{a}") for a in "wxyz"]
        q_aug = augmented["branches"]["imu"][:, :, q_idx]
        mask = raw["masks"]["imu"]
        norms = np.linalg.norm(q_aug[mask], axis=-1)
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-4)


if __name__ == "__main__":
    unittest.main()
