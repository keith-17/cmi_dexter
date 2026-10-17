"""Execute the large notebook configurations and verify both actual objectives."""
import ast
from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
import warnings


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
# An unactivated Windows Conda interpreter needs its own LAPACK DLL directory.
# Keep the handle alive for the entire process; no environment files are changed.
_conda_dll_directory = Path(sys.prefix) / "Library" / "bin"
_conda_dll_handle = (os.add_dll_directory(str(_conda_dll_directory))
                     if os.name == "nt" and _conda_dll_directory.is_dir() else None)

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import base_utils as data_utils
import multibranch_utils as mb
import pretraining_utils as pretraining


def full_modality_recordings():
    """Independent subjects and all four sensor families, with both domains."""
    frames = []
    for subject in range(8):
        for category in range(4):
            time = np.arange(32, dtype=float)
            phase = time / 4.0 + category * 0.4 + subject * 0.01
            angle = 0.12 * np.sin(phase)
            data = {
                "sequence_id": f"p{subject}-c{category}", "subject": f"p{subject}",
                "sequence_counter": time.astype(int), "is_target": int(category < 2),
                "gesture": f"gesture_{category}",
                "bfrb": f"target_{category}" if category < 2 else "non_bfrb",
                "orientation": "upright", "acc_x": np.sin(phase),
                "acc_y": np.cos(phase), "acc_z": 9.8 + 0.1 * np.sin(phase),
                "rot_w": np.cos(angle), "rot_x": np.sin(angle) / np.sqrt(3),
                "rot_y": np.sin(angle) / np.sqrt(3), "rot_z": np.sin(angle) / np.sqrt(3),
            }
            for sensor in range(1, 6):
                data[f"thm_{sensor}"] = 30 + sensor / 10 + 0.1 * np.sin(phase)
                for pixel in range(64):
                    data[f"tof_{sensor}_v{pixel}"] = 200 + sensor * 3 + pixel / 10 + np.cos(phase)
            frames.append(pd.DataFrame(data))
    return pd.concat(frames, ignore_index=True)


class LargePretrainingNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebooks = {
            variant: json.loads((ROOT / f"notebooks/multibranch_pretraining_{variant}_large.ipynb").read_text(encoding="utf-8"))
            for variant in ("model1", "model2")
        }
        cls.original = json.loads((ROOT / "notebooks/multibranch_v1.ipynb").read_text(encoding="utf-8"))

    def source(self, variant, index):
        return "".join(self.notebooks[variant]["cells"][index]["source"])

    def namespace(self, variant, directory):
        namespace = {
            "workspace_root": directory, "Path": Path, "datetime": datetime,
            "np": np, "pd": pd, "json": json, "GroupShuffleSplit": GroupShuffleSplit,
            "GroupKFold": GroupKFold, "Pipeline": Pipeline, "ECat": data_utils.ECat,
            "EInt": data_utils.EInt, "EFloat": data_utils.EFloat,
            "SequenceExtractor": data_utils.SequenceExtractor,
            "SensorAugmentor": data_utils.SensorAugmentor,
            "pretraining_keras": pretraining, "multibranch_keras": mb,
            "SKOPT_AVAILABLE": mb.HierarchicalBayesSearchCV is not None,
        }
        with redirect_stdout(StringIO()):
            for index in (3, 4, 11, 14):
                exec(compile(self.source(variant, index), f"{variant}-cell-{index}", "exec"), namespace)
        return namespace

    def test_preserves_layout_without_notebook_functions_or_stale_outputs(self):
        for variant, notebook in self.notebooks.items():
            self.assertEqual(len(notebook["cells"]), 26)
            self.assertEqual([cell["cell_type"] for cell in notebook["cells"]],
                             [cell["cell_type"] for cell in self.original["cells"]])
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "code":
                    tree = ast.parse("".join(cell["source"]))
                    self.assertFalse(any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
                                         for node in ast.walk(tree)), (variant, index))
                    self.assertEqual(cell["outputs"], [])

    def test_full_budgets_large_active_backbone_and_cloning(self):
        for variant in self.notebooks:
            with self.subTest(variant=variant), TemporaryDirectory() as directory:
                namespace = self.namespace(variant, directory)
                self.assertEqual(namespace["n_iter"], 37)
                self.assertEqual(namespace["train_size"], 0.7)
                self.assertEqual(namespace["GROUP_COLUMN"], "subject")
                self.assertFalse(namespace["FIRST_RUN"])
                estimator = clone(namespace["pipeline"]).named_steps["estimator"]
                self.assertEqual(estimator.epochs, 150)
                self.assertEqual(estimator.pretraining_epochs, 30)
                self.assertEqual(estimator.temporal_architecture, "bert")
                self.assertEqual((estimator.branch_cnn_filters, estimator.branch_cnn_depth,
                                  estimator.branch_cnn_kernel_size), (128, 4, 7))
                self.assertEqual(estimator.static_branch_units, 100)
                grid = namespace["GRID_PARAM_SPACE"]
                self.assertEqual(grid["estimator__bert_d_model"], [128, 192, 256])
                self.assertEqual(grid["estimator__bert_num_layers"], [2, 3])
                self.assertEqual(grid["estimator__bert_ff_dim"], [256, 512])
                for width in grid["estimator__bert_d_model"]:
                    for heads in grid["estimator__bert_num_heads"]:
                        self.assertEqual(width % heads, 0)
                active = mb.canonical_candidate(namespace["BEST_PARAMS"], namespace["pipeline"].get_params())
                for name in ("branch_cnn_filters", "branch_cnn_depth", "branch_cnn_kernel_size",
                             "bert_d_model", "static_branch_units", "fusion_units"):
                    self.assertIn("estimator__" + name, active)

    def test_reference_features_applied_to_the_actual_estimator(self):
        for variant in self.notebooks:
            with self.subTest(variant=variant), TemporaryDirectory() as directory:
                namespace = self.namespace(variant, directory)
                params = namespace["pipeline"].get_params(deep=True)
                reference = namespace["REFERENCE_BEST_PARAMS"]
                for key, expected in reference.items():
                    if key.startswith("estimator__extractor__"):
                        self.assertEqual(params[key], expected, key)
                self.assertFalse(namespace["LARGE_UNSUPPORTED_REFERENCE_PARAMS"])
                self.assertEqual(params["estimator__extractor__chunk_window_size"], 200)
                self.assertIsNone(params["estimator__extractor__maxlen"])
                self.assertEqual(params["estimator__extractor__sequence_crop_mode"], "none")

    def test_all_searches_expose_every_parameter_and_four_source_conditions(self):
        sources = ["non_target", "all", "target", "baseline"]
        for variant in self.notebooks:
            with self.subTest(variant=variant), TemporaryDirectory() as directory:
                namespace = self.namespace(variant, directory)
                leaves = {key for key in namespace["pipeline"].get_params(deep=True)
                          if key.startswith("estimator__")
                          and key not in {"estimator__extractor", "estimator__augmentor"}}
                for name in ("GRID_PARAM_SPACE", "BAYESIAN_PARAM_SPACE", "EVOLUTIONARY_SEARCH_SPACE"):
                    self.assertEqual(set(namespace[name]), leaves, name)
                key = "estimator__extractor__pretraining_source"
                self.assertEqual(namespace["GRID_PARAM_SPACE"][key], sources)
                self.assertEqual(list(namespace["EVOLUTIONARY_PARAM_SPACE"][key].choices), sources)
                if namespace["SKOPT_AVAILABLE"]:
                    self.assertEqual(list(namespace["BAYESIAN_PARAM_SPACE"][key].categories), sources)

    def test_training_controls_restore_reference_optimisation_with_zero_dropout(self):
        for variant in self.notebooks:
            with self.subTest(variant=variant), TemporaryDirectory() as directory:
                namespace = self.namespace(variant, directory)
                grid = namespace["GRID_PARAM_SPACE"]
                for key, values in grid.items():
                    if "dropout" in key or "weight_decay" in key or "label_smoothing" in key:
                        self.assertEqual(values, [0.0], key)
                    if "gradient_clip" in key:
                        self.assertEqual(values, [None], key)
                for prefix in ("", "pretraining_"):
                    self.assertEqual(grid[f"estimator__{prefix}optimizer_name"], ["adamw"])
                    self.assertEqual(grid[f"estimator__{prefix}use_lr_scheduler"], [True])
                    self.assertEqual(grid[f"estimator__{prefix}lr_scheduler_type"], ["cosine"])
                    self.assertEqual(grid[f"estimator__{prefix}learning_rate"], [1e-4])
                self.assertEqual(grid["estimator__use_batchnorm"], [True])
                self.assertEqual(grid["estimator__augmentor__prob"], [0.0])
                self.assertEqual(grid["estimator__augmentor__per_aug_prob"], [0.0])
                self.assertEqual(grid["estimator__augmentor__augmentations"], [None])
                self.assertEqual(grid["estimator__pretraining_augmentor"], [None])

    def test_matching_search_spaces_differ_only_by_objective(self):
        with TemporaryDirectory() as directory:
            first = self.namespace("model1", directory)
            second = self.namespace("model2", directory)
        grid1, grid2 = first["GRID_PARAM_SPACE"], second["GRID_PARAM_SPACE"]
        differences = {key for key in grid1 if grid1[key] != grid2[key]}
        # Each experiment writes diagnostics to its own results directory.
        differences.discard("estimator__candidate_failure_log")
        self.assertEqual(differences, {"estimator__pretraining_mode"})
        self.assertEqual(grid1["estimator__pretraining_mode"], ["supervised"])
        self.assertEqual(grid2["estimator__pretraining_mode"], ["masked_ssl"])

    def test_holdout_and_cv_separate_subjects_and_keep_both_domains(self):
        frame = full_modality_recordings()
        for variant in self.notebooks:
            with self.subTest(variant=variant), TemporaryDirectory() as directory:
                namespace = self.namespace(variant, directory)
                namespace.update(train_df=frame.copy(), sensor_columns=[])
                with redirect_stdout(StringIO()):
                    exec(compile(self.source(variant, 9), "subject-split", "exec"), namespace)
                train, test = namespace["X_train"], namespace["X_test"]
                self.assertTrue(set(train.subject).isdisjoint(test.subject))
                self.assertEqual(set(train.is_target), {0, 1})
                self.assertEqual(set(test.is_target), {0, 1})
                for fitting, validation in namespace["cv_object"].split(train, namespace["y_train"], namespace["groups"]):
                    self.assertTrue(set(train.iloc[fitting].subject).isdisjoint(train.iloc[validation].subject))

    def assert_real_one_epoch(self, variant):
        frame = full_modality_recordings()
        labels = frame[["sequence_id", "subject", "is_target", "gesture", "bfrb", "orientation"]].copy()
        with TemporaryDirectory() as directory:
            namespace = self.namespace(variant, directory)
            estimator = clone(namespace["pipeline"]).named_steps["estimator"]
            # Runtime-only epoch overrides; every feature and backbone size
            # remains the large experiment's actual first candidate. "all"
            # is one of the search's four legitimate source conditions.
            estimator.set_params(epochs=1, pretraining_epochs=1, verbose=0,
                                 pretraining_verbose=0, extractor__pretraining_source="all")
            with warnings.catch_warnings():
                # Constructing the full ToF validity/masking matrices must not
                # fragment the frame or emit thousands of warnings per batch.
                warnings.simplefilter("error", pd.errors.PerformanceWarning)
                estimator.fit(frame, labels)
            self.assertTrue(estimator.pretraining_completed_)
            self.assertTrue(estimator.transferred_changed_layer_names_)
            self.assertEqual(estimator.model_.get_layer("acc_temporal_conv4").filters, 128)
            self.assertEqual(estimator.model_.get_layer("rotation_temporal_conv4").kernel_size, (7,))
            self.assertEqual(estimator.model_.get_layer("tof_temporal_conv4").filters, 128)
            self.assertEqual(estimator.model_.get_layer("thermo_temporal_conv4").filters, 128)
            self.assertEqual(estimator.model_.get_layer("bert_input_projection").units, 128)
            self.assertEqual(estimator.model_.get_layer("fusion_projection").units, 256)
            self.assertEqual(estimator.model_.get_layer("stft_static_dense").units, 100)
            self.assertTrue(set(estimator.preprocessing_sequence_ids_).isdisjoint(estimator.validation_sequence_ids_))
            if variant == "model2":
                self.assertGreater(estimator.ssl_first_batch_diagnostics_["hidden_valid_measurements"], 0)
            probabilities = np.asarray(estimator.predict_proba(frame.loc[frame.is_target == 1]))
            self.assertEqual(probabilities.shape[0], 16)
            self.assertTrue(np.isfinite(probabilities).all())
            np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
            self.assertTrue(np.isfinite(estimator.score(frame, labels)))
            mb.tf.keras.backend.clear_session()

    @unittest.skipUnless(mb.TENSORFLOW_AVAILABLE, "TensorFlow unavailable")
    def test_real_one_epoch_model1_keeps_the_large_architecture(self):
        self.assert_real_one_epoch("model1")

    @unittest.skipUnless(mb.TENSORFLOW_AVAILABLE, "TensorFlow unavailable")
    def test_real_one_epoch_model2_keeps_the_large_architecture(self):
        self.assert_real_one_epoch("model2")


if __name__ == "__main__":
    unittest.main()
