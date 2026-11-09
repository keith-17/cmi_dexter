"""Comprehensive test suite for src/utils_v3.py and notebooks/multi-branch_version_3.ipynb."""
from __future__ import annotations

import ast
from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import utils_v3 as u3
from utils_v3 import (
    AUGMENTATIONS,
    ECat,
    EFloat,
    EInt,
    GASearchCV,
    HierarchicalBayesSearchCV,
    HierarchicalGASearchCV,
    InvalidParams,
    MultiBranchTSAIClassifier,
    MultiBranchTSAINet,
    MultiModalSequenceExtractor,
    NOOP_MOTION_ATOMS,
    NOOP_MOTION_MODES,
    NOOP_PERIODICITY_MODES,
    NOOP_QUALITY_MODES,
    NOOP_SPECTRAL_ATOMS,
    NOOP_SPECTRAL_MODES,
    NoopOptions,
    ParamSpec,
    SKOPT_AVAILABLE,
    TSAI_ARCHS,
    TSAI_AVAILABLE,
    TSAUG_AVAILABLE,
    TSAugmentor,
    TSFEL_AVAILABLE,
    TSFRESH_AVAILABLE,
    ToFSpatialEncoder,
    build_tsai_backbone,
    cadence_estimate,
    candidate_score,
    compile_unified_search_space,
    competition_score,
    compute_permutation_importance,
    deduplicate_param_grid,
    evaluate_holdout,
    find_data_root,
    inactive_noop_parameters,
    inactive_parameters_for_candidate,
    kimmitch_acf_estimate,
    library_status,
    make_competition_scorer,
    make_synthetic_cmi_df,
    noop_imu_window,
    noop_power_window,
    refit_valid_candidate,
    run_validation_searches,
    simulate_hidden_test_nullification,
    split_validation_protocol,
    validation_protocols,
)


class OpenSourceStackAndExtractorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = make_synthetic_cmi_df(n_subjects=4, sequences_per_subject=4, n_timesteps=60, random_state=42)

    def test_library_status_reports_expected_keys(self):
        status = library_status()
        for key in ("torch", "tsai", "tsaug", "tsfel", "tsfresh", "scikit-optimize"):
            self.assertIn(key, status)
            self.assertIsInstance(status[key], bool)

    def test_dynamic_sample_rate_adaptation_across_sensors(self):
        """IMU at 100 Hz vs ToF/Thermopile at 20 Hz produce proportional window sample counts."""
        ext = MultiModalSequenceExtractor(
            window_seconds=2.0,
            imu_native_sampling_rate=100.0,
            thm_native_sampling_rate=20.0,
            tof_native_sampling_rate=20.0,
            static_backend="none",
        )
        lengths = ext.window_lengths()
        self.assertEqual(lengths["imu"], 200)
        self.assertEqual(lengths["thm"], 40)
        self.assertEqual(lengths["tof"], 40)

        chunks = ext.fit(self.df).transform_chunks(self.df)
        self.assertEqual(chunks["branches"]["imu"].shape[1], 200)
        self.assertEqual(chunks["branches"]["thm"].shape[1], 40)
        self.assertEqual(chunks["branches"]["tof"].shape[1], 40)
        self.assertEqual(chunks["rates"], {"imu": 100.0, "thm": 20.0, "tof": 20.0})

        # Custom target rates (e.g. IMU 50 Hz, Thm/ToF 25 Hz) dynamically adapt window lengths and PSD bands
        ext_custom = MultiModalSequenceExtractor(
            window_seconds=2.0,
            imu_target_sampling_rate=50.0,
            thm_target_sampling_rate=25.0,
            tof_target_sampling_rate=25.0,
            static_backend="none",
            noop=NoopOptions(
                motion_stats="full",
                periodicity="acf_cadence",
                spectral_stats="all",
                power_branches="imu|thm|tof",
                band=(0.5, 18.0),
                nyquist_policy="clip",
            ),
        )
        chunks_custom = ext_custom.fit(self.df).transform_chunks(self.df)
        self.assertEqual(chunks_custom["branches"]["imu"].shape[1], 100)
        self.assertEqual(chunks_custom["branches"]["thm"].shape[1], 50)
        self.assertEqual(chunks_custom["branches"]["tof"].shape[1], 50)
        self.assertTrue(np.isfinite(chunks_custom["static"]).all())

    def test_retained_noop_and_cr7_kimmitch_features(self):
        """Verify all legacy NOOP features + full cr7_kimmitch descriptors across IMU, Thm, and ToF."""
        opts = NoopOptions(
            motion_stats="full",
            periodicity="acf_kimmitch",
            spectral_stats="all",
            quality="features",
            power_branches="imu|thm|tof",
            cadence_band=(1.2, 3.5),
            min_cadence_strength=0.15,
        )
        ext = MultiModalSequenceExtractor(
            noop=opts,
            window_seconds=1.5,
            static_backend="none",
        )
        chunks = ext.fit(self.df).transform_chunks(self.df)
        noop_names = set(chunks["noop_feature_names"])

        expected_features = {
            # Legacy motion stats
            "noop_acc_ac_rms",
            "noop_acc_x_rms",
            "noop_acc_y_rms",
            "noop_acc_z_rms",
            "noop_jerk_rms",
            "noop_rotation_mean_rad_s",
            "noop_rotation_rms_rad_s",
            # cr7_kimmitch motion, posture, jerk floor, cross-modal coupling, DoG, and Baevsky SI
            "noop_gyro_energy_dps",
            "noop_posture_var",
            "noop_motion_intensity_mean",
            "noop_motion_variance",
            "noop_motion_cv",
            "noop_jerk_floor",
            "noop_move_fraction",
            "noop_still_fraction",
            "noop_jerk_gate_ratio",
            "noop_acc_gyro_corr",
            "noop_gyro_to_acc_ratio",
            "noop_dog_variability",
            "noop_baevsky_amo_pct",
            "noop_baevsky_mxdmn",
            "noop_baevsky_si",
            # cr7_kimmitch fundamental-preferred sub-lag ACF periodicity
            "noop_cadence_hz",
            "noop_cadence_sublag_hz",
            "noop_cadence_strength",
            "noop_acf_sublag_concavity",
            # Legacy + cr7_kimmitch spectral stats on IMU
            "noop_spectral_peak_fraction",
            "noop_spectral_peak_neighbourhood_fraction",
            "noop_spectral_band_0_fraction",
            "noop_spectral_band_1_fraction",
            "noop_spectral_band_2_fraction",
            "noop_spectral_entropy_normalized",
            "noop_spectral_peakedness",
            # Multi-rate power branches (thm and tof at 20 Hz: drift, deviation, CV, and spectral)
            "noop_thm_drift_slope",
            "noop_thm_dev_rms",
            "noop_thm_motion_cv",
            "noop_thm_spectral_peak_fraction",
            "noop_thm_spectral_entropy_normalized",
            "noop_thm_spectral_peakedness",
            "noop_tof_drift_slope",
            "noop_tof_dev_rms",
            "noop_tof_motion_cv",
            "noop_tof_spectral_peak_fraction",
            "noop_tof_spectral_entropy_normalized",
            "noop_tof_spectral_peakedness",
            # Quality features
            "noop_observed_fraction",
            "noop_imputed_fraction",
            "noop_longest_contiguous_fraction",
            "noop_counter_gap_fraction",
        }
        self.assertTrue(expected_features.issubset(noop_names), expected_features - noop_names)

        # Also test 'kimmitch' motion alias and acf_harmonic_checked periodicity
        ext_kim = MultiModalSequenceExtractor(
            noop=NoopOptions(motion_stats="kimmitch", periodicity="acf_harmonic_checked"),
            window_seconds=1.5,
            static_backend="none",
        )
        chunks_kim = ext_kim.fit(self.df).transform_chunks(self.df)
        self.assertIn("noop_posture_var", chunks_kim["noop_feature_names"])
        self.assertIn("noop_baevsky_si", chunks_kim["noop_feature_names"])
        self.assertIn("noop_acf_frequency_hz", chunks_kim["noop_feature_names"])
        self.assertIn("noop_acf_strength", chunks_kim["noop_feature_names"])
        self.assertIn("noop_acf_magnitude_harmonic", chunks_kim["noop_feature_names"])

    def test_cadence_and_kimmitch_acf_detect_fundamental_sinusoid(self):
        fs = 100.0
        t = np.arange(300) / fs
        sig = 1.0 + 0.05 * t + 0.8 * np.sin(2 * np.pi * 2.15 * t)
        hz, strength, usable = cadence_estimate(sig, fs, band=(1.0, 3.5), min_strength=0.2)
        self.assertTrue(usable)
        self.assertAlmostEqual(hz, 2.15, delta=0.15)
        self.assertGreater(strength, 0.6)

        # kimmitch_acf_estimate prefers the 2.15 Hz fundamental over the ~1.075 Hz subharmonic
        # and refines sub-lag precision via parabolic interpolation
        k_hz, k_sub_hz, k_str, k_conc, k_ok = kimmitch_acf_estimate(sig, fs, band=(1.0, 3.5), min_strength=0.2)
        self.assertTrue(k_ok)
        self.assertAlmostEqual(k_sub_hz, 2.15, delta=0.05)
        self.assertGreater(k_str, 0.7)
        self.assertGreater(k_conc, 0.0)

    @unittest.skipUnless(TSFEL_AVAILABLE, "tsfel not installed")
    def test_tsfel_static_backend_extracts_rate_adapted_features(self):
        subset = self.df[self.df["sequence_id"].isin(self.df["sequence_id"].unique()[:3])].copy()
        ext = MultiModalSequenceExtractor(
            window_seconds=1.0,
            static_backend="tsfel",
            static_channels="acc_mag|thm_mean",
            tsfel_domains="statistical|temporal|spectral",
        )
        chunks = ext.fit(subset).transform_chunks(subset)
        tsfel_cols = [n for n in chunks["static_names"] if n.startswith("tsfel__")]
        self.assertGreater(len(tsfel_cols), 5)
        self.assertTrue(np.isfinite(chunks["static"]).all())

    @unittest.skipUnless(TSFRESH_AVAILABLE, "tsfresh not installed")
    def test_tsfresh_static_backend_extracts_rate_adapted_features(self):
        subset = self.df[self.df["sequence_id"].isin(self.df["sequence_id"].unique()[:3])].copy()
        ext = MultiModalSequenceExtractor(
            window_seconds=1.0,
            static_backend="tsfresh",
            static_channels="acc_mag|tof_mean",
            tsfresh_profile="minimal",
        )
        chunks = ext.fit(subset).transform_chunks(subset)
        tsfresh_cols = [n for n in chunks["static_names"] if n.startswith("tsfresh__")]
        self.assertGreater(len(tsfresh_cols), 5)
        self.assertTrue(np.isfinite(chunks["static"]).all())


class TSAugmentorTests(unittest.TestCase):
    @unittest.skipUnless(TSAUG_AVAILABLE, "tsaug not installed")
    def test_all_tsaug_operations_and_modality_dropout(self):
        df = make_synthetic_cmi_df(n_subjects=2, sequences_per_subject=2, n_timesteps=50, random_state=7)
        ext = MultiModalSequenceExtractor(window_seconds=1.0, static_backend="none")
        raw = ext.raw_chunks(df)

        aug = TSAugmentor(
            prob=1.0,
            per_aug_prob=1.0,
            augmentations=None,  # enables all AUGMENTATIONS
            n_copies=1,
            jitter_sigma=0.02,
            noise_std=0.03,
            scaling_sigma=0.05,
            warp_sigma=0.1,
            warp_num_knots=4,
            time_warp_max_speed_ratio=1.2,
            time_warp_n_speed_change=3,
            time_shift_frac=0.1,
            crop_frac_range=(0.75, 0.95),
            temporal_num_masks=2,
            temporal_mask_frac=0.1,
            temporal_mask_fill="ffill",
            channel_drop_prob=0.1,
            sensor_drop_prob=0.25,
            sensor_drop_joint_prob=0.25,
            random_state=42,
        )
        self.assertEqual(set(aug.active_ops()), set(AUGMENTATIONS))
        rng = np.random.default_rng(42)
        augmented = aug.augment_chunks(raw, rng)
        for name in ("imu", "thm", "tof"):
            self.assertEqual(augmented["branches"][name].shape, raw["branches"][name].shape)
            self.assertTrue(np.isfinite(augmented["branches"][name]).all())
            self.assertFalse(np.allclose(augmented["branches"][name], raw["branches"][name]))

        keep = aug.sample_modality_keep(200, rng)
        self.assertEqual(keep.shape, (200, 3))
        self.assertTrue(keep[:, 0].all())  # IMU is never dropped
        self.assertTrue((~keep[:, 1]).any())
        self.assertTrue((~keep[:, 2]).any())


class TSAINetworkAndEstimatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = make_synthetic_cmi_df(n_subjects=4, sequences_per_subject=4, n_timesteps=50, random_state=11)

    def test_all_tsai_backbones_and_spatial_tof_forward(self):
        for arch in TSAI_ARCHS:
            if arch != "Conv1d" and not TSAI_AVAILABLE:
                continue
            with self.subTest(arch=arch):
                backbone = build_tsai_backbone(
                    arch, c_in=6, c_out=16, seq_len=30,
                    filters=16, depth=2, kernel_size=5, dropout=0.1, conv_dropout=0.05,
                )
                x = torch.randn(4, 6, 30)
                out = backbone(x)
                self.assertEqual(tuple(out.shape), (4, 16))
                self.assertTrue(torch.isfinite(out).all())

        # Spatial 5x8x8 ToF encoder + MultiBranchTSAINet with raw ToF (320 + 15 = 335 channels)
        for fusion_type in ("attention", "gated", "concat"):
            for pooling_mode in ("default", "concat_pool", "attention"):
                net = MultiBranchTSAINet(
                    branch_specs={"imu": (50, 8), "thm": (10, 11), "tof": (10, 335)},
                    static_dim=12,
                    n_classes=3,
                    arch="InceptionTimePlus" if TSAI_AVAILABLE else "Conv1d",
                    branch_filters=16,
                    branch_depth=2,
                    branch_units=24,
                    pooling_mode=pooling_mode,
                    tof_spatial_mode="conv2d",
                    tof_spatial_filters=12,
                    fusion_type=fusion_type,
                    fusion_units=24,
                    dense_units=24,
                )
                branches = {
                    "imu": torch.randn(4, 50, 8),
                    "thm": torch.randn(4, 10, 11),
                    "tof": torch.randn(4, 10, 335),
                }
                masks = {
                    "imu": torch.ones(4, 50, dtype=torch.bool),
                    "thm": torch.ones(4, 10, dtype=torch.bool),
                    "tof": torch.ones(4, 10, dtype=torch.bool),
                }
                # Test with 50% of batch having thm & tof absent
                present = torch.tensor([
                    [True, True, True],
                    [True, False, False],
                    [True, True, False],
                    [True, False, True],
                ])
                static = torch.randn(4, 12)
                logits = net(branches, masks, present, static)
                self.assertEqual(tuple(logits.shape), (4, 3))
                self.assertTrue(torch.isfinite(logits).all())

    def test_estimator_fit_predict_history_and_missing_modality_resilience(self):
        ext = MultiModalSequenceExtractor(
            window_seconds=0.5,
            static_backend="none",
            noop=NoopOptions(motion_stats="basic", periodicity="acf_cadence", min_samples=8),
            tof_modes="raw|pooled_stats",
        )
        aug = TSAugmentor(prob=0.5, jitter_sigma=0.01, sensor_drop_joint_prob=0.3, random_state=42)
        clf = MultiBranchTSAIClassifier(
            extractor=ext,
            augmentor=aug,
            arch="InceptionTimePlus" if TSAI_AVAILABLE else "Conv1d",
            branch_filters=16,
            branch_depth=2,
            branch_units=24,
            fusion_type="attention",
            fusion_units=24,
            dense_units=24,
            epochs=3,
            batch_size=8,
            learning_rate=2e-3,
            validation_split=0.25,
            random_state=42,
        )
        labels = self.df[["sequence_id", "subject", "is_target", "gesture", "bfrb"]].copy()
        clf.fit(self.df, labels)

        self.assertFalse(clf.fit_failed_)
        history = clf.get_history_dict()
        for key in ("loss", "val_loss", "accuracy", "val_accuracy"):
            self.assertEqual(len(history[key]), 3)
            self.assertTrue(all(np.isfinite(v) for v in history[key]))

        preds = clf.predict(self.df)
        self.assertEqual(len(preds), self.df["sequence_id"].nunique())
        self.assertTrue(set(preds.unique()).issubset(set(labels["bfrb"].unique())))
        score = clf.score(self.df, labels)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class CompetitionScorerAndHiddenTestSimulationTests(unittest.TestCase):
    def test_competition_score_exact_binary_and_macro_average(self):
        y_true = pd.Series(["g1", "g1", "g2", "g2", "non_bfrb", "non_bfrb"], index=[f"s{i}" for i in range(6)])
        y_pred_perfect = y_true.copy()
        res = competition_score(y_true, y_pred_perfect)
        self.assertAlmostEqual(res["binary_f1"], 1.0)
        self.assertAlmostEqual(res["gesture_f1"], 1.0)
        self.assertAlmostEqual(res["competition_score"], 1.0)

        # Partial prediction: binary is perfect (all target predicted as g1, non_bfrb predicted as non_bfrb)
        y_pred_half = pd.Series(["g1", "g1", "g1", "g1", "non_bfrb", "non_bfrb"], index=y_true.index)
        res_half = competition_score(y_true, y_pred_half)
        self.assertAlmostEqual(res_half["binary_f1"], 1.0)
        # g1 F1 = 2*2/(4+2) = 2/3, g2 F1 = 0 -> macro = 1/3
        self.assertAlmostEqual(res_half["gesture_f1"], 1.0 / 3.0, places=5)
        self.assertAlmostEqual(res_half["competition_score"], (1.0 + 1.0 / 3.0) / 2.0, places=5)

    def test_simulate_hidden_test_nullifies_50_percent_of_sequences(self):
        df = make_synthetic_cmi_df(n_subjects=3, sequences_per_subject=4, n_timesteps=20, random_state=42)
        total_seqs = df["sequence_id"].nunique()
        self.assertEqual(total_seqs, 12)

        null_df = simulate_hidden_test_nullification(df, null_fraction=0.5, random_state=42)
        null_ids = set(null_df.attrs["nullified_sequence_ids"])
        self.assertEqual(len(null_ids), 6)

        thm_tof_cols = [c for c in df.columns if c.startswith("thm_") or c.startswith("tof_")]
        imu_cols = ["acc_x", "acc_y", "acc_z", "rot_w", "rot_x", "rot_y", "rot_z"]

        null_rows = null_df[null_df["sequence_id"].isin(null_ids)]
        kept_rows = null_df[~null_df["sequence_id"].isin(null_ids)]

        self.assertTrue(null_rows[thm_tof_cols].isna().all().all())
        self.assertTrue(kept_rows[thm_tof_cols].notna().any().any())
        self.assertTrue(null_rows[imu_cols].notna().all().all())

        # Verify extractor marks present[:, 1:] == False for nullified sequences
        ext = MultiModalSequenceExtractor(window_seconds=0.2, static_backend="none")
        chunks = ext.fit(df).transform_chunks(null_df)
        for i, seq_id in enumerate(chunks["sequence_ids"]):
            if str(seq_id) in null_ids:
                self.assertTrue(chunks["present"][i, 0])
                self.assertFalse(chunks["present"][i, 1])
                self.assertFalse(chunks["present"][i, 2])
                self.assertEqual(float(np.abs(chunks["branches"]["thm"][i]).sum()), 0.0)
                self.assertEqual(float(np.abs(chunks["branches"]["tof"][i]).sum()), 0.0)


class NotebookAndUnifiedSearchSpaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb_path = ROOT / "notebooks" / "multi-branch_version_3.ipynb"
        cls.notebook = json.loads(cls.nb_path.read_text(encoding="utf-8"))

    def test_notebook_parses_cleanly_has_required_sections_and_zero_updates(self):
        markdown_headers = []
        code_cells = []
        for idx, cell in enumerate(self.notebook["cells"]):
            src = "".join(cell["source"])
            if cell["cell_type"] == "markdown":
                for line in src.splitlines():
                    if line.startswith("# "):
                        markdown_headers.append(line.strip())
            elif cell["cell_type"] == "code":
                code_cells.append((idx, src))
                tree = ast.parse(src)
                # No function or lambda definitions inside notebook cells (excluding the standard pandas .apply row mapping in Prep if any, let's check FunctionDef)
                self.assertFalse(
                    any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(tree)),
                    f"FunctionDef found in cell {idx}",
                )
                # Zero .update() calls anywhere in the notebook (Requirement 5)
                self.assertFalse(
                    any(
                        isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Attribute)
                        and n.func.attr == "update"
                        for n in ast.walk(tree)
                    ),
                    f"Fragmented .update() call found in cell {idx}",
                )
                self.assertEqual(cell.get("outputs", []), [])
                self.assertIsNone(cell.get("execution_count"))

        for required in ("# Params", "# Prep", "# Train", "# Analysis", "# Trained-model permutation importance"):
            self.assertIn(required, markdown_headers, f"Missing section {required!r}")

    def test_separate_grid_bayes_and_evo_spaces_cover_all_leaves_and_constraints(self):
        namespace = {"__name__": "__main__"}
        space_cells = {"BEST_PARAMS": 0, "GRID_PARAM_SPACE": 0, "BAYESIAN_PARAM_SPACE": 0, "EVOLUTIONARY_PARAM_SPACE": 0}
        with TemporaryDirectory() as tmpdir, redirect_stdout(StringIO()):
            # Execute Imports, Config, Best Params, Pipeline Setup, and the separate Grid/Bayes/Evo cells
            for cell in self.notebook["cells"]:
                if cell["cell_type"] != "code":
                    continue
                src = "".join(cell["source"])
                for key in space_cells:
                    if f"{key} =" in src:
                        space_cells[key] += 1
                if any(
                    marker in src
                    for marker in (
                        "from utils_v3 import",
                        "experiment_name =",
                        "BEST_PARAMS =",
                        "pipeline = Pipeline(",
                        "GRID_PARAM_SPACE =",
                        "BAYESIAN_PARAM_SPACE =",
                        "EVOLUTIONARY_PARAM_SPACE =",
                    )
                ):
                    src = src.replace("workspace_root = current_dir", f"workspace_root = {tmpdir!r}")
                    exec(compile(src, "notebook-cell", "exec"), namespace)

        # Verify BEST_PARAMS, GRID_PARAM_SPACE, BAYESIAN_PARAM_SPACE, and EVOLUTIONARY_PARAM_SPACE are in separate cells
        self.assertEqual(space_cells["BEST_PARAMS"], 1)
        self.assertEqual(space_cells["GRID_PARAM_SPACE"], 1)
        self.assertEqual(space_cells["BAYESIAN_PARAM_SPACE"], 1)
        self.assertEqual(space_cells["EVOLUTIONARY_PARAM_SPACE"], 1)

        best_params = namespace["BEST_PARAMS"]
        grid_space = namespace["GRID_PARAM_SPACE"]
        bayes_space = namespace["BAYESIAN_PARAM_SPACE"]
        evo_space = namespace["EVOLUTIONARY_PARAM_SPACE"]
        pipeline = namespace["pipeline"]

        all_pipe_params = pipeline.get_params(deep=True)
        containers = {
            "steps", "memory", "verbose", "estimator",
            "estimator__extractor", "estimator__extractor__noop",
            "estimator__augmentor", "estimator__candidate_failure_log",
        }
        expected_leaves = {k for k in all_pipe_params if k not in containers}

        self.assertEqual(set(best_params.keys()), expected_leaves)
        self.assertEqual(set(grid_space.keys()), expected_leaves)
        self.assertEqual(set(evo_space.keys()), expected_leaves)
        if SKOPT_AVAILABLE:
            self.assertEqual(set(bayes_space.keys()), expected_leaves)

        # Verify feature extraction modes have 3 options each
        feature_extraction_keys = (
            "estimator__extractor__acc_modes",
            "estimator__extractor__rotation_modes",
            "estimator__extractor__thm_modes",
            "estimator__extractor__tof_modes",
            "estimator__extractor__static_backend",
            "estimator__extractor__static_channels",
            "estimator__extractor__tsfel_domains",
            "estimator__extractor__noop__motion_stats",
            "estimator__extractor__noop__periodicity",
            "estimator__extractor__noop__spectral_stats",
            "estimator__extractor__noop__power_branches",
            "estimator__extractor__noop__signal",
            "estimator__extractor__noop__quality",
        )
        for key in feature_extraction_keys:
            self.assertEqual(len(grid_space[key]), 3, f"{key} should have 3 options in GRID_PARAM_SPACE")
            if SKOPT_AVAILABLE:
                self.assertEqual(len(bayes_space[key].categories), 3, f"{key} should have 3 options in BAYESIAN_PARAM_SPACE")

        # Verify simplest single architecture ('Conv1d') and small size range
        self.assertEqual(grid_space["estimator__arch"], ["Conv1d"])
        if SKOPT_AVAILABLE:
            self.assertEqual(list(bayes_space["estimator__arch"].categories), ["Conv1d"])
        for size_key in ("estimator__branch_filters", "estimator__branch_units", "estimator__fusion_units", "estimator__dense_units"):
            self.assertEqual(grid_space[size_key], [64, 96, 128])

        # Verify regularization and dropouts have 2 options starting at 0.0
        for reg_key in (
            "estimator__dropout",
            "estimator__branch_dropout",
            "estimator__conv_dropout",
            "estimator__static_dropout",
            "estimator__fusion_dropout",
            "estimator__weight_decay",
            "estimator__label_smoothing",
        ):
            self.assertEqual(len(grid_space[reg_key]), 2, f"{reg_key} should have 2 options")
            self.assertEqual(grid_space[reg_key][0], 0.0, f"{reg_key} must start at 0.0")
            if SKOPT_AVAILABLE:
                cats = list(bayes_space[reg_key].categories)
                self.assertEqual(len(cats), 2)
                self.assertEqual(cats[0], 0.0)

        # Verify clone(pipeline).set_params(**BEST_PARAMS) succeeds and validates
        cloned = clone(pipeline).set_params(**best_params)
        cloned.named_steps["estimator"].validate()

    def test_importance_cell_uses_internal_validation_and_never_holdout(self):
        imp_cell = next(
            "".join(c["source"])
            for c in self.notebook["cells"]
            if c["cell_type"] == "code" and "compute_permutation_importance(" in "".join(c["source"])
        )
        tree = ast.parse(imp_cell)
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        self.assertNotIn("X_test", names)
        self.assertNotIn("y_test", names)
        self.assertIn("validation_sequence_ids_", imp_cell)
        self.assertIn("require_internal_validation=True", imp_cell)


class EndToEndSearchAndImportanceIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = make_synthetic_cmi_df(n_subjects=4, sequences_per_subject=4, n_timesteps=40, random_state=99)

    def _fast_pipeline(self):
        ext = MultiModalSequenceExtractor(
            window_seconds=0.4,
            static_backend="none",
            noop=NoopOptions(motion_stats="basic", periodicity="acf_cadence", min_samples=8),
            tof_modes="pooled_stats",
        )
        aug = TSAugmentor(prob=0.0, random_state=42)
        clf = MultiBranchTSAIClassifier(
            extractor=ext,
            augmentor=aug,
            arch="Conv1d",
            branch_filters=12,
            branch_depth=1,
            branch_units=16,
            static_units=16,
            static_layers=1,
            fusion_type="concat",
            fusion_units=16,
            dense_units=16,
            dense_layers=1,
            epochs=2,
            batch_size=8,
            validation_split=0.3,
            random_state=42,
        )
        return Pipeline([("estimator", clf)])

    def test_run_validation_searches_and_permutation_importance_end_to_end(self):
        pipe = self._fast_pipeline()
        evo_space = {
            "estimator__learning_rate": EFloat(5e-4, 2e-3, "log-uniform"),
            "estimator__dense_units": ECat([16, 24]),
        }
        search = HierarchicalGASearchCV(
            estimator=pipe,
            param_space=evo_space,
            population_size=2,
            generations=1,
            refit=refit_valid_candidate,
            random_state=42,
        )
        with TemporaryDirectory() as tmpdir:
            runs = run_validation_searches(
                search=search,
                experiment_df=self.df,
                target_col="bfrb",
                validation_space={"validation_mode": ["sequence"]},
                train_size=0.75,
                n_splits=1,
                random_state=42,
                results_dir=tmpdir,
                experiment_name="test_v3",
                search_mode="evo",
                simulate_hidden_test=True,
                hidden_test_null_fraction=0.5,
            )
            self.assertTrue(Path(runs["experiments_path"]).is_file())
            self.assertTrue(Path(runs["summary_path"]).is_file())
            primary = runs["runs"][0]
            self.assertTrue(primary["experiment_succeeded"])
            self.assertIn("competition_score", primary["holdout_results"])
            self.assertIn("hidden_test_competition_score", primary["holdout_results"])

            # Verify permutation importance on internal validation sequences
            best_est = primary["search"].best_estimator_.named_steps["estimator"]
            val_ids = best_est.validation_sequence_ids_
            X_train = primary["X_train"]
            y_train = primary["y_train"]
            X_internal = X_train[X_train["sequence_id"].astype(str).isin(val_ids.astype(str))].copy()
            y_internal = y_train.loc[X_internal.index]

            imp_df = compute_permutation_importance(
                best_est,
                X_internal,
                y_internal,
                repeats=2,
                random_state=42,
                require_internal_validation=True,
                output_dir=tmpdir,
                prefix="test_imp",
            )
            self.assertGreater(len(imp_df), 0)
            self.assertIn("noop_cadence_hz", set(imp_df["name"]))
            for p in imp_df.attrs["paths"].values():
                self.assertTrue(Path(p).is_file())

            # Verify holdout or training sequences are rejected by compute_permutation_importance
            with self.assertRaisesRegex(ValueError, "outside that partition"):
                compute_permutation_importance(best_est, primary["X_test"], primary["y_test"])

    def test_grid_deduplication_and_inactive_parameter_pruning(self):
        pipe = self._fast_pipeline()
        base = pipe.get_params(deep=True)
        # When noop__motion_stats, periodicity, spectral_stats, quality are all None,
        # varying noop__min_cycles or noop__acf_harmonic_tolerance is inactive and deduplicates to 1 candidate.
        grid = {
            "estimator__extractor__noop__motion_stats": [None],
            "estimator__extractor__noop__periodicity": [None],
            "estimator__extractor__noop__spectral_stats": [None],
            "estimator__extractor__noop__quality": [None],
            "estimator__extractor__noop__min_cycles": [1.5, 2.0, 3.0],
            "estimator__extractor__noop__acf_harmonic_tolerance": [0.05, 0.1, 0.2],
            "estimator__augmentor__prob": [0.0],
            "estimator__augmentor__jitter_sigma": [0.01, 0.05, 0.1],
        }
        deduped = deduplicate_param_grid(grid, base_params=base)
        self.assertEqual(len(deduped), 1)

    @unittest.skipUnless(SKOPT_AVAILABLE, "scikit-optimize not installed")
    def test_hierarchical_bayes_search_end_to_end_and_clean_starting_combination(self):
        from skopt.space import Categorical, Real

        pipe = self._fast_pipeline()
        bayes_space = {
            "estimator__augmentor__prob": Categorical([0.0, 0.4]),
            "estimator__augmentor__per_aug_prob": Categorical([0.0, 0.5]),
            "estimator__dropout": Categorical([0.0, 0.1]),
            "estimator__branch_dropout": Categorical([0.0, 0.1]),
            "estimator__weight_decay": Categorical([0.0, 1e-5]),
            "estimator__label_smoothing": Categorical([0.0, 0.05]),
            "estimator__learning_rate": Real(5e-4, 2e-3, prior="log-uniform"),
            "estimator__extractor__noop__cadence_band": Categorical([
                json.dumps((1.0, 3.5), sort_keys=True),
                json.dumps((1.2, 3.5), sort_keys=True),
            ]),
        }
        search = HierarchicalBayesSearchCV(
            estimator=pipe,
            search_spaces=bayes_space,
            n_iter=2,
            cv=GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42),
            refit=refit_valid_candidate,
            random_state=42,
        )
        labels = self.df[["sequence_id", "subject", "is_target", "gesture", "bfrb"]].copy()
        search.fit(self.df, labels, groups=self.df["sequence_id"])
        self.assertEqual(len(search.cv_results_["params"]), 2)
        first_cand = search.cv_results_["params"][0]
        self.assertEqual(first_cand["estimator__augmentor__prob"], 0.0)
        self.assertEqual(first_cand["estimator__augmentor__per_aug_prob"], 0.0)
        self.assertEqual(first_cand["estimator__dropout"], 0.0)
        self.assertEqual(first_cand["estimator__branch_dropout"], 0.0)
        self.assertEqual(first_cand["estimator__weight_decay"], 0.0)
        self.assertEqual(first_cand["estimator__label_smoothing"], 0.0)
        self.assertIsInstance(search.best_params_["estimator__extractor__noop__cadence_band"], (tuple, list))


if __name__ == "__main__":
    unittest.main()
