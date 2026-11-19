"""Comprehensive unit & integration tests for pretraining_ssl_utils, bundles, and MultiBranchEnsemble."""
from __future__ import annotations

import ast
import json
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from base_utils import ROT_COLUMNS, find_data_root  # noqa: E402
from pretraining_ssl_utils import (  # noqa: E402
    MaskedSSLReconstructionHead,
    corrupt_raw_dataframe_spans,
    select_pretraining_sequences,
    transfer_torch_backbone_weights,
)
from utils import (  # noqa: E402
    Ensemble,
    MultiBranchEnsemble,
    MultiBranchTSAIClassifier,
    MultiModalSequenceExtractor,
    NoopOptions,
    SKOPT_AVAILABLE,
    TSAugmentor,
    make_synthetic_cmi_df,
    run_validation_searches,
)


def _fast_clf(**overrides):
    ext = MultiModalSequenceExtractor(
        window_seconds=0.4,
        static_backend="none",
        noop=NoopOptions(motion_stats="basic", periodicity="acf_cadence", min_samples=8),
        tof_modes="pooled_stats",
    )
    aug = TSAugmentor(prob=0.0, random_state=42)
    params = dict(
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
        validation_split=0.25,
        early_stopping_patience=2,
        random_state=42,
    )
    params.update(overrides)
    return MultiBranchTSAIClassifier(**params)


class PretrainingAndSSLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = make_synthetic_cmi_df(n_subjects=4, sequences_per_subject=4, n_timesteps=40, random_state=17)

    def test_zero_pretraining_has_no_overhead_or_history(self):
        clf = _fast_clf()
        self.assertEqual(clf.pretraining_epochs, 0)
        self.assertEqual(clf.finetuning_head_epochs, 0)
        self.assertEqual(clf.ssl_mask_ratio, 0.0)
        self.assertEqual(clf.ssl_span_length, 0)
        clf.fit(self.df, self.df)
        self.assertFalse(clf.pretraining_completed_)
        self.assertEqual(clf.pretraining_history_, {})
        self.assertEqual(clf.finetuning_head_history_, {})
        self.assertEqual(clf.transferred_layer_names_, [])

    def test_supervised_pretraining_and_head_warmup_transfer_weights(self):
        clf = _fast_clf(
            pretraining_mode="supervised",
            pretraining_source="all",
            pretraining_label_column="gesture",
            pretraining_epochs=2,
            pretraining_learning_rate=1e-3,
            finetuning_head_epochs=1,
            finetuning_learning_rate=5e-4,
        )
        clf.fit(self.df, self.df)
        self.assertTrue(clf.pretraining_completed_)
        self.assertEqual(len(clf.pretraining_history_["loss"]), 2)
        self.assertEqual(len(clf.finetuning_head_history_["loss"]), 1)
        self.assertGreater(len(clf.transferred_layer_names_), 0)
        proba = clf.predict_proba(self.df)
        self.assertEqual(proba.shape, (self.df["sequence_id"].nunique(), len(clf.classes_)))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_corrupt_raw_dataframe_preserves_quaternion_unit_norm(self):
        corrupted = corrupt_raw_dataframe_spans(
            self.df, sequence_col="sequence_id", mask_ratio=0.35, span_length=8, seed=99,
        )
        norms = np.linalg.norm(corrupted[list(ROT_COLUMNS)].to_numpy(dtype=float), axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_masked_ssl_pretraining_runs_reconstructs_and_transfers(self):
        clf = _fast_clf(
            pretraining_mode="masked_ssl",
            pretraining_source="all",
            pretraining_epochs=2,
            ssl_mask_ratio=0.3,
            ssl_span_length=8,
            ssl_decoder_units=24,
            ssl_decoder_dropout=0.1,
            ssl_loss_name="huber",
            ssl_resample_mask_each_epoch=True,
        )
        clf.fit(self.df, self.df)
        self.assertTrue(clf.pretraining_completed_)
        self.assertEqual(len(clf.pretraining_history_["loss"]), 2)
        self.assertTrue(all(np.isfinite(v) for v in clf.pretraining_history_["loss"]))
        self.assertGreater(len(clf.transferred_layer_names_), 0)
        preds = clf.predict(self.df)
        self.assertEqual(len(preds), self.df["sequence_id"].nunique())

    def test_select_pretraining_sequences_filters_baseline_vs_target(self):
        base_df = select_pretraining_sequences(self.df, self.df, source="baseline")
        self.assertTrue((base_df["is_target"] == 0).all())
        tgt_df = select_pretraining_sequences(self.df, self.df, source="target")
        self.assertTrue((tgt_df["is_target"] == 1).all())


class BundleAndEnsembleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = make_synthetic_cmi_df(n_subjects=6, sequences_per_subject=4, n_timesteps=40, random_state=77)
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        tr_idx, te_idx = next(splitter.split(cls.df, groups=cls.df["subject"]))
        cls.X_tr = cls.df.iloc[tr_idx].copy()
        cls.X_te = cls.df.iloc[te_idx].copy()

    def test_save_and_load_bundle_reproduces_identical_probabilities(self):
        clf = _fast_clf()
        clf.fit(self.X_tr, self.X_tr)
        p_orig = clf.predict_proba(self.X_te)
        ho_ids = pd.Index(self.X_te["sequence_id"].astype(str).unique())

        with TemporaryDirectory() as tmpdir:
            b_dir = clf.save_bundle(
                Path(tmpdir) / "model_a",
                oof_proba=clf.predict_proba(self.X_tr),
                oof_sequence_ids=pd.Index(self.X_tr["sequence_id"].astype(str).unique()),
                holdout_proba=p_orig,
                holdout_sequence_ids=ho_ids,
                cv_score=0.75,
                holdout_score=0.72,
            )
            for fname in ("weights.pt", "config.json", "oof_proba.csv", "holdout_proba.csv", "manifest.json"):
                self.assertTrue((b_dir / fname).exists(), f"Missing {fname}")
            loaded = MultiBranchTSAIClassifier.load_bundle(b_dir, device="cpu")
            p_loaded = loaded.predict_proba(self.X_te)
            np.testing.assert_allclose(p_orig, p_loaded, atol=1e-5)

    def test_ensemble_scales_2_to_5_models_e2e_and_from_saved_bundles(self):
        self.assertIs(Ensemble, MultiBranchEnsemble)
        base_params = {
            "branch_filters": 12,
            "branch_depth": 1,
            "branch_units": 16,
            "static_units": 16,
            "static_layers": 1,
            "fusion_units": 16,
            "dense_units": 16,
            "dense_layers": 1,
            "epochs": 2,
            "batch_size": 8,
            "early_stopping_patience": 2,
            "extractor__window_seconds": 0.4,
            "extractor__static_backend": "none",
            "extractor__tof_modes": "pooled_stats",
        }
        specs_e2e = [
            {"name": "m1_all", "arch": "Conv1d", "extractor__modalities": "all", "fusion_type": "concat"},
            {"name": "m2_imu", "arch": "Conv1d", "extractor__modalities": "imu", "fusion_type": "concat"},
            {"name": "m3_all", "arch": "Conv1d", "extractor__modalities": "all", "fusion_type": "gated"},
            {"name": "m4_imu", "arch": "Conv1d", "extractor__modalities": "imu", "fusion_type": "concat"},
            {"name": "m5_all", "arch": "Conv1d", "extractor__modalities": "all", "fusion_type": "attention"},
        ]

        # 1. End-to-end training for 2 models (soft_vote & stacking) and 5 models (hierarchical)
        for n_m, mode in ((2, "soft_vote"), (2, "stacking"), (5, "hierarchical")):
            ens = MultiBranchEnsemble(
                model_specs=specs_e2e,
                base_estimator_params=base_params,
                n_models=n_m,
                use_saved_bundles=False,
                ensemble_mode=mode,
                cv_splits=2,
                imu_specialist_boost=0.5,
                random_state=42,
            )
            ens.fit(self.X_tr, self.X_tr)
            self.assertFalse(ens.fit_failed_, ens.fit_error_)
            self.assertEqual(len(ens.members_), n_m)
            proba = ens.predict_proba(self.X_te)
            self.assertEqual(proba.shape, (self.X_te["sequence_id"].nunique(), len(ens.classes_)))
            np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-4)
            summary = ens.summarize_members()
            self.assertEqual(len(summary), n_m + 1)

        # 2. Bundle-based hierarchical ensembling
        with TemporaryDirectory() as tmpdir:
            tr_ids = pd.Index(self.X_tr["sequence_id"].astype(str).unique())
            bundle_paths = []
            for idx_b, mod_mode in enumerate(("all", "imu", "all")):
                clf = _fast_clf(fusion_type="concat")
                clf.extractor.modalities = mod_mode
                clf.fit(self.X_tr, self.X_tr)
                b_dir = clf.save_bundle(
                    Path(tmpdir) / f"bundle_{idx_b}",
                    oof_proba=clf.predict_proba(self.X_tr),
                    oof_sequence_ids=tr_ids,
                )
                bundle_paths.append(str(b_dir))

            ens_b = MultiBranchEnsemble(
                bundle_dirs=bundle_paths,
                n_models=3,
                use_saved_bundles=True,
                ensemble_mode="hierarchical",
                imu_specialist_boost=0.5,
            )
            ens_b.fit(self.X_tr, self.X_tr)
            self.assertFalse(ens_b.fit_failed_, ens_b.fit_error_)
            ho_preds = ens_b.predict(self.X_te)
            self.assertEqual(len(ho_preds), self.X_te["sequence_id"].nunique())
            ens_dir = ens_b.save_bundle(Path(tmpdir) / "ensemble_out")
            self.assertTrue((ens_dir / "manifest.json").exists())
            self.assertTrue((ens_dir / "member_summary.csv").exists())

    def test_sample_csv_37_sequences_runs_fast_for_single_and_ensemble(self):
        data_root = find_data_root()
        sample_path = data_root / "sample.csv"
        if not sample_path.exists():
            self.skipTest("data/sample.csv not present")
        sample_df = pd.read_csv(sample_path)
        dem_path = data_root / "train_demographics.csv"
        if dem_path.exists():
            dem_df = pd.read_csv(dem_path)
            cols = ["subject"] + [c for c in dem_df.columns if c not in sample_df.columns]
            sample_df = sample_df.merge(dem_df[cols], on="subject", how="left")
        if "bfrb" not in sample_df.columns:
            sample_df["bfrb"] = np.where(sample_df["sequence_type"] == "Target", sample_df["gesture"], "non_bfrb")
        self.assertEqual(sample_df["sequence_id"].nunique(), 37)

        clf = _fast_clf(epochs=1, batch_size=16, validation_split=0.2)
        ens = MultiBranchEnsemble(
            model_specs=[
                {"name": "s1", "arch": "Conv1d", "extractor__modalities": "all", "fusion_type": "concat"},
                {"name": "s2", "arch": "Conv1d", "extractor__modalities": "imu", "fusion_type": "concat"},
            ],
            base_estimator_params={
                "branch_filters": 8, "branch_depth": 1, "branch_units": 12,
                "static_units": 12, "static_layers": 1, "fusion_units": 12,
                "dense_units": 12, "dense_layers": 1, "epochs": 1, "batch_size": 16,
                "extractor__window_seconds": 0.4, "extractor__static_backend": "none",
                "extractor__tof_modes": "pooled_stats",
            },
            n_models=2,
            use_saved_bundles=False,
            ensemble_mode="hierarchical",
            cv_splits=2,
        )
        clf.fit(sample_df, sample_df)
        ens.fit(sample_df, sample_df)
        self.assertFalse(clf.fit_failed_)
        self.assertFalse(ens.fit_failed_, ens.fit_error_)
        self.assertEqual(len(clf.predict(sample_df)), 37)
        self.assertEqual(len(ens.predict(sample_df)), 37)

    def test_ensemble_notebook_structure_and_leaf_coverage(self):
        nb_path = ROOT / "notebooks" / "multibranch_ensemble.ipynb"
        self.assertTrue(nb_path.exists())
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        ns = {"__name__": "__main__"}
        with TemporaryDirectory() as tmpdir, redirect_stdout(StringIO()):
            for idx, cell in enumerate(nb["cells"]):
                if cell["cell_type"] != "code":
                    continue
                self.assertEqual(cell.get("outputs", []), [])
                self.assertIsNone(cell.get("execution_count"))
                src = "".join(cell["source"])
                tree = ast.parse(src)
                self.assertFalse(
                    any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(tree)),
                    f"FunctionDef found in cell {idx}",
                )
                self.assertFalse(
                    any(
                        isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Attribute)
                        and n.func.attr == "update"
                        for n in ast.walk(tree)
                    ),
                    f".update() found in cell {idx}",
                )
                if any(
                    m in src
                    for m in ("from utils import", "experiment_name =", "pipeline = Pipeline(",
                              "GRID_PARAM_SPACE =", "BAYESIAN_PARAM_SPACE =", "EVOLUTIONARY_PARAM_SPACE =")
                ):
                    src = src.replace("workspace_root = current_dir", f"workspace_root = {tmpdir!r}")
                    exec(compile(src, "multibranch_ensemble.ipynb", "exec"), ns)

        pipe = ns["pipeline"]
        containers = {
            "steps", "memory", "verbose", "transform_input", "estimator",
            "estimator__bundle_dirs", "estimator__estimators",
            "estimator__model_specs", "estimator__base_estimator_params",
        }
        expected_leaves = {k for k in pipe.get_params(deep=True) if k not in containers}
        self.assertEqual(set(ns["GRID_PARAM_SPACE"].keys()), expected_leaves)
        self.assertEqual(set(ns["EVOLUTIONARY_PARAM_SPACE"].keys()), expected_leaves)
        if SKOPT_AVAILABLE:
            self.assertEqual(set(ns["BAYESIAN_PARAM_SPACE"].keys()), expected_leaves)

    def test_multibranch_notebook_keeps_pretraining_ssl_params_at_zero(self):
        nb = json.loads((ROOT / "notebooks" / "multibranch.ipynb").read_text(encoding="utf-8"))
        ns = {"__name__": "__main__"}
        with TemporaryDirectory() as tmpdir, redirect_stdout(StringIO()):
            for cell in nb["cells"]:
                if cell["cell_type"] != "code":
                    continue
                src = "".join(cell["source"])
                if any(m in src for m in ("from utils import", "experiment_name =", "GRID_PARAM_SPACE =", "BAYESIAN_PARAM_SPACE =", "EVOLUTIONARY_PARAM_SPACE =")):
                    src = src.replace("workspace_root = current_dir", f"workspace_root = {tmpdir!r}")
                    exec(compile(src, "multibranch.ipynb", "exec"), ns)

        for key in (
            "estimator__pretraining_epochs", "estimator__pretraining_learning_rate",
            "estimator__pretraining_batch_size", "estimator__pretraining_validation_split",
            "estimator__pretraining_early_stopping_patience",
            "estimator__finetuning_head_epochs", "estimator__finetuning_head_learning_rate",
            "estimator__finetuning_learning_rate",
            "estimator__ssl_mask_ratio", "estimator__ssl_span_length",
            "estimator__ssl_decoder_units", "estimator__ssl_decoder_dropout",
        ):
            self.assertEqual(ns["GRID_PARAM_SPACE"][key], [0] if isinstance(ns["GRID_PARAM_SPACE"][key][0], int) else [0.0])
            self.assertEqual(list(ns["EVOLUTIONARY_PARAM_SPACE"][key].choices), [0] if isinstance(list(ns["EVOLUTIONARY_PARAM_SPACE"][key].choices)[0], int) else [0.0])
            if SKOPT_AVAILABLE:
                self.assertEqual(list(ns["BAYESIAN_PARAM_SPACE"][key].categories), [0] if isinstance(list(ns["BAYESIAN_PARAM_SPACE"][key].categories)[0], int) else [0.0])
        self.assertEqual(ns["GRID_PARAM_SPACE"]["estimator__pretraining_mode"], ["none"])


if __name__ == "__main__":
    unittest.main()
