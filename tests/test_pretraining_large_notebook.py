"""Check active notebook configuration without changing its editable search spaces."""
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
from unittest.mock import patch


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

try:
    from skopt.space import Categorical, Integer, Real
except ImportError:
    Categorical = Integer = Real = None


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import base_utils as data_utils
import multibranch_utils as mb
import pretraining_utils as pretraining
import pretraining_search


def decoded_choice(value):
    value = mb._json_maybe(value)
    return list(value) if isinstance(value, tuple) else value


def literal_assignments(source, name):
    return [node for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
            and any(isinstance(target, ast.Name) and target.id == name
                    for target in node.targets)]


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
            variant: json.loads((ROOT / "notebooks" / ("multibranch_pretraining.ipynb" if variant == "model1" else "multibranch_ssl.ipynb")).read_text(encoding="utf-8"))
            for variant in ("model1", "model2")
        }
        cls.original = json.loads((ROOT / "notebooks/multibranch_v1.ipynb").read_text(encoding="utf-8"))

    def source(self, variant, index):
        return "".join(self.notebooks[variant]["cells"][index]["source"])

    def namespace(self, variant, directory, validation_mode=None, source_edits=None):
        namespace = {
            "workspace_root": directory, "Path": Path, "datetime": datetime,
            "np": np, "pd": pd, "json": json, "GroupShuffleSplit": GroupShuffleSplit,
            "GroupKFold": GroupKFold, "Pipeline": Pipeline, "ECat": data_utils.ECat,
            "data_utils": data_utils,
            "EInt": data_utils.EInt, "EFloat": data_utils.EFloat,
            "SequenceExtractor": data_utils.SequenceExtractor,
            "SensorAugmentor": data_utils.SensorAugmentor,
            "pretraining_keras": pretraining, "multibranch_keras": mb,
            "pretraining_search": pretraining_search,
            "SKOPT_AVAILABLE": mb.HierarchicalBayesSearchCV is not None,
            "Categorical": Categorical, "Integer": Integer, "Real": Real,
        }
        with redirect_stdout(StringIO()):
            indices = (3, 4, 11, 13, 14, 15) if variant == "model1" else (3, 4, 11, 14)
            for index in indices:
                source = self.source(variant, index)
                if validation_mode is not None:
                    tree = ast.parse(source)
                    for node in tree.body:
                        if isinstance(node, ast.Assign) and any(
                                isinstance(target, ast.Name) and target.id == "VALIDATION_PARAM_SPACE"
                                for target in node.targets):
                            lines = source.splitlines(keepends=True)
                            lines.insert(node.end_lineno,
                                         f"\nVALIDATION_PARAM_SPACE['validation_mode'] = [{validation_mode!r}]\n")
                            source = "".join(lines)
                            break
                if source_edits is not None:
                    source = source_edits(index, source)
                exec(compile(source, f"{variant}-cell-{index}", "exec"), namespace)
        return namespace

    def test_preserves_layout_without_notebook_functions_or_false_curve_save_claims(self):
        for variant, notebook in self.notebooks.items():
            self.assertEqual(len(notebook["cells"]), 26)
            expected_types = [cell["cell_type"] for cell in self.original["cells"]]
            if variant == "model1":
                expected_types[12:14] = ["markdown", "code"]
            self.assertEqual([cell["cell_type"] for cell in notebook["cells"]],
                             expected_types)
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "code":
                    tree = ast.parse("".join(cell["source"]))
                    self.assertFalse(any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
                                         for node in ast.walk(tree)), (variant, index))
                    # Active notebooks may retain the user's training logs and
                    # inline figures from previous runs.
                    self.assertNotIn('Saved curves to', str(cell.get('outputs', [])))

    def declared_space(self, variant, name, namespace):
        assignments = [node for cell in self.notebooks[variant]["cells"]
                       if cell["cell_type"] == "code"
                       for node in literal_assignments("".join(cell["source"]), name)]
        self.assertEqual(len(assignments), 1, name)
        return eval(compile(ast.Expression(assignments[0].value), "declared-space", "eval"), namespace)

    def assert_same_evolutionary_values(self, actual, expected):
        self.assertEqual(type(actual), type(expected))
        if hasattr(expected, "choices"):
            self.assertEqual([decoded_choice(value) for value in actual.choices],
                             [decoded_choice(value) for value in expected.choices])
        else:
            for attribute in ("low", "high", "distribution"):
                if hasattr(expected, attribute):
                    self.assertEqual(getattr(actual, attribute), getattr(expected, attribute))

    def assert_same_bayesian_values(self, actual, expected):
        if isinstance(expected, (list, tuple)):
            values = getattr(actual, "categories", actual)
            self.assertEqual([decoded_choice(value) for value in values],
                             [decoded_choice(value) for value in expected])
            return
        self.assertEqual(type(actual), type(expected))
        if hasattr(expected, "categories"):
            self.assertEqual([decoded_choice(value) for value in actual.categories],
                             [decoded_choice(value) for value in expected.categories])
            np.testing.assert_equal(actual.prior, expected.prior)
        else:
            for attribute in ("low", "high", "prior", "base"):
                self.assertEqual(getattr(actual, attribute), getattr(expected, attribute))

    def test_full_budgets_remain_explicit_and_best_mode_is_cloneable(self):
        for variant in self.notebooks:
            with self.subTest(variant=variant), TemporaryDirectory() as directory:
                namespace = self.namespace(variant, directory)
                self.assertGreater(namespace["n_iter"], 0)
                self.assertGreater(namespace["train_size"], 0)
                self.assertLess(namespace["train_size"], 1)
                self.assertTrue(pretraining_search.validation_protocols(
                    namespace["VALIDATION_PARAM_SPACE"]))
                grid = namespace["GRID_PARAM_SPACE"]
                estimator = clone(namespace["pipeline"]).set_params(
                    **namespace["BEST_PARAMS"]).named_steps["estimator"]
                self.assertEqual(estimator.epochs, grid["estimator__epochs"][0])
                self.assertEqual(estimator.pretraining_epochs, grid["estimator__pretraining_epochs"][0])
                self.assertEqual(estimator.temporal_architecture, grid["estimator__temporal_architecture"][0])

    def test_primary_grid_candidates_and_evolutionary_ranges_survive_compilation(self):
        for variant in self.notebooks:
            with self.subTest(variant=variant), TemporaryDirectory() as directory:
                namespace = self.namespace(variant, directory)
                declared_grid = self.declared_space(variant, "GRID_PARAM_SPACE", namespace)
                declared_bayesian = (self.declared_space(variant, "BAYESIAN_PARAM_SPACE", namespace)
                                     if variant == "model1" else declared_grid)
                declared_evolutionary = self.declared_space(variant, "EVOLUTIONARY_PARAM_SPACE", namespace)
                for key, choices in declared_grid.items():
                    self.assertEqual(namespace["GRID_PARAM_SPACE"][key], choices, key)
                for key, choices in declared_bayesian.items():
                    with self.subTest(parameter=key):
                        self.assert_same_bayesian_values(namespace["BAYESIAN_PARAM_SPACE"][key], choices)
                for key, choices in declared_evolutionary.items():
                    self.assert_same_evolutionary_values(
                        namespace["EVOLUTIONARY_PARAM_SPACE"][key], choices)
                    self.assert_same_evolutionary_values(
                        namespace["EVOLUTIONARY_SEARCH_SPACE"][key], choices)

    @unittest.skipIf(Integer is None, "scikit-optimize unavailable")
    def test_model1_bayesian_ranges_keep_integer_real_and_off_choice_semantics(self):
        with TemporaryDirectory() as directory:
            namespace = self.namespace("model1", directory)
        space = namespace["BAYESIAN_PARAM_SPACE"]
        integer_bounds = {
            "augmentor__warp_num_knots": (2, 8),
            "augmentor__temporal_num_masks": (0, 7),
            "extractor__chunk_window_size": (250, 300),
            "branch_dense_units": (512, 1024), "dense_units": (400, 1024),
        }
        for name, bounds in integer_bounds.items():
            dimension = space["estimator__" + name]
            self.assertIsInstance(dimension, Integer, name)
            self.assertEqual((dimension.low, dimension.high), bounds, name)
        for name in ("conv_dropout", "dropout", "learning_rate"):
            dimension = space["estimator__" + name]
            self.assertIsInstance(dimension, Real, name)
            bounds = (3e-5, 1e-4) if name == "learning_rate" else (0.1, 0.5)
            self.assertEqual((dimension.low, dimension.high), bounds, name)
            self.assertEqual(dimension.prior, "log-uniform" if name == "learning_rate" else "uniform")
        for name in ("sensor_drop_prob", "channel_drop_prob", "noise_std", "jitter_sigma",
                     "scaling_sigma", "warp_sigma", "time_shift_frac", "temporal_mask_frac"):
            dimension = space["estimator__augmentor__" + name]
            self.assertIsInstance(dimension, Categorical, name)
            self.assertIn(0.0, dimension.categories, name)
        for name, choices in {
            "batch_size": [128, 256], "pretraining_epochs": [30],
            "augmentor__augmentations": [None], "pretraining_shuffle": [True],
            "pretraining_mode": ["supervised"],
        }.items():
            self.assertIsInstance(space["estimator__" + name], Categorical, name)
            self.assertEqual(list(space["estimator__" + name].categories), choices, name)

    def test_model1_grid_and_evolutionary_setup_do_not_require_skopt(self):
        for mode, name in (("grid", "GRID_PARAM_SPACE"), ("evo", "EVOLUTIONARY_SEARCH_SPACE")):
            def disable_skopt(index, source):
                if index == 3:
                    return source + ("\nSKOPT_AVAILABLE = False\nCategorical = Integer = Real = None\n"
                                     f"search_mode = {mode!r}\n")
                return source

            with self.subTest(mode=mode), TemporaryDirectory() as directory:
                namespace = self.namespace("model1", directory, source_edits=disable_skopt)
                self.assertEqual(namespace["BAYESIAN_PARAM_SPACE"], {})
                self.assertIs(namespace["param_space"], namespace[name])

    @unittest.skipIf(mb.HierarchicalBayesSearchCV is None, "scikit-optimize unavailable")
    def test_model1_full_bayesian_space_fits_search_without_neural_training(self):
        frame = full_modality_recordings()
        labels = frame[["sequence_id", "subject", "is_target", "gesture", "bfrb"]]
        fitted = []

        def fit(estimator, X, y, **kwargs):
            estimator.extractor_ = clone(estimator.extractor)
            estimator._validate_extractor()
            fitted.append((estimator.extractor.chunk_window_size, estimator.learning_rate))
            return estimator

        def score(estimator, X, y, sample_weight=None):
            return 0.5 + float(estimator.learning_rate) / 1e-3

        with TemporaryDirectory() as directory:
            namespace = self.namespace("model1", directory)
            self.assertEqual(len(namespace["BAYESIAN_PARAM_SPACE"]), 213)
            search = mb.HierarchicalBayesSearchCV(
                estimator=namespace["pipeline"], search_spaces=namespace["BAYESIAN_PARAM_SPACE"],
                n_iter=2, cv=GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42),
                random_state=42, n_jobs=1, scoring=mb.candidate_score, refit=False,
                error_score="raise",
            )
            with patch.object(pretraining.AdvancedMultiBranchSequenceClassifier, "_fit_impl", fit), \
                    patch.object(pretraining.AdvancedMultiBranchSequenceClassifier, "score", score), \
                    redirect_stdout(StringIO()):
                search.fit(frame, labels, groups=frame.subject)
        self.assertEqual(len(search.cv_results_["params"]), 2)
        self.assertEqual(len(fitted), 2)
        self.assertTrue(np.all(np.isfinite(search.cv_results_["mean_test_score"])))
        self.assertTrue(all(250 <= window <= 300 and 3e-5 <= rate <= 1e-4 for window, rate in fitted))

    def test_searches_contain_only_declared_choices_and_stage_defaults(self):
        for variant in self.notebooks:
            with self.subTest(variant=variant), TemporaryDirectory() as directory:
                namespace = self.namespace(variant, directory)
                stage_keys = set(namespace.get("PRETRAINING_PARAM_SPACE", {}))
                names = ["GRID_PARAM_SPACE", "EVOLUTIONARY_PARAM_SPACE"]
                if variant == "model1":
                    self.assertNotIn("PRETRAINING_PARAM_SPACE", namespace)
                    names.append("BAYESIAN_PARAM_SPACE")
                for name in names:
                    declared = self.declared_space(variant, name, namespace)
                    self.assertEqual(set(namespace[name]), set(declared) | stage_keys)
                self.assertNotIn("estimator__candidate_failure_log", namespace["GRID_PARAM_SPACE"])
                key = "estimator__extractor__pretraining_source"
                sources = (["baseline"] if variant == "model1" else
                           namespace["PRETRAINING_PARAM_SPACE"][key])
                self.assertEqual(namespace["GRID_PARAM_SPACE"][key], sources)
                self.assertEqual(list(namespace["EVOLUTIONARY_PARAM_SPACE"][key].choices), sources)

    def test_configuration_has_no_reference_overrides_or_file_writes(self):
        def capture_params(index, source):
            if literal_assignments(source, "GRID_PARAM_SPACE"):
                return "_params_before_search_setup = pipeline.get_params(deep=True)\n" + source
            return source

        for variant in self.notebooks:
            with self.subTest(variant=variant), TemporaryDirectory() as directory:
                namespace = self.namespace(variant, directory, source_edits=capture_params)
                self.assertEqual(namespace["pipeline"].get_params(deep=True),
                                 namespace["_params_before_search_setup"])
                self.assertFalse(list(Path(directory).rglob("*_configuration.json")))
                for name in namespace:
                    self.assertFalse(name.startswith(("LARGE_", "REFERENCE_", "FIRST_RUN")), name)
                self.assertNotIn("ACTIVE_STAGE_PARAM_SPACE", namespace)
                self.assertNotIn("EVOLUTIONARY_STAGE_OVERRIDES", namespace)

    def test_models_keep_their_own_explicit_search_choices(self):
        with TemporaryDirectory() as directory:
            first = self.namespace("model1", directory)
            second = self.namespace("model2", directory)
        self.assertEqual(first["GRID_PARAM_SPACE"]["estimator__pretraining_mode"], ["supervised"])
        self.assertEqual(second["GRID_PARAM_SPACE"]["estimator__pretraining_mode"], ["masked_ssl"])
        self.assertEqual(first["GRID_PARAM_SPACE"]["estimator__extractor__chunk_window_size"],
                         [250, 300])
        self.assertEqual(second["GRID_PARAM_SPACE"]["estimator__extractor__chunk_window_size"],
                         [250, 300, 350, 400])

    def test_default_holdout_and_cv_separate_sequences_and_keep_both_domains(self):
        frame = full_modality_recordings()
        for variant in self.notebooks:
            with self.subTest(variant=variant), TemporaryDirectory() as directory:
                namespace = self.namespace(variant, directory)
                protocol, = pretraining_search.validation_protocols(namespace["VALIDATION_PARAM_SPACE"])
                with redirect_stdout(StringIO()):
                    train, test, groups, cv = pretraining_search.split_validation_protocol(
                        frame, protocol, train_size=namespace["train_size"],
                        n_splits=namespace["n_splits"], random_state=namespace["random_state"])
                self.assertTrue(set(train.sequence_id).isdisjoint(test.sequence_id))
                self.assertTrue(set(train.subject).intersection(test.subject))
                self.assertEqual(set(train.is_target), {0, 1})
                self.assertEqual(set(test.is_target), {0, 1})
                pd.testing.assert_series_equal(groups, train.sequence_id)
                for fitting, validation in cv.split(train, groups=groups):
                    self.assertTrue(set(train.iloc[fitting].sequence_id).isdisjoint(train.iloc[validation].sequence_id))

    def test_subject_option_separates_subjects_in_holdout_and_cv(self):
        frame = full_modality_recordings()
        for variant in self.notebooks:
            with self.subTest(variant=variant), TemporaryDirectory() as directory:
                namespace = self.namespace(variant, directory, validation_mode="subject")
                protocol, = pretraining_search.validation_protocols(namespace["VALIDATION_PARAM_SPACE"])
                self.assertEqual(protocol, {"validation_mode": "subject", "group_column": "subject"})
                with redirect_stdout(StringIO()):
                    train, test, groups, cv = pretraining_search.split_validation_protocol(
                        frame, protocol, train_size=namespace["train_size"],
                        n_splits=namespace["n_splits"], random_state=namespace["random_state"])
                self.assertTrue(set(train.subject).isdisjoint(test.subject))
                self.assertEqual(set(train.is_target), {0, 1})
                self.assertEqual(set(test.is_target), {0, 1})
                pd.testing.assert_series_equal(groups, train.subject)
                for fitting, validation in cv.split(train, groups=groups):
                    self.assertTrue(set(train.iloc[fitting].subject).isdisjoint(train.iloc[validation].subject))

    def test_selected_mode_reaches_both_internal_validation_stages(self):
        frame = full_modality_recordings()
        labels = frame[["sequence_id", "subject", "is_target", "gesture", "bfrb"]].copy()
        for variant in self.notebooks:
            for mode, expected in (("sequence", None), ("subject", "subject")):
                with self.subTest(variant=variant, mode=mode), TemporaryDirectory() as directory:
                    namespace = self.namespace(variant, directory, validation_mode=mode)
                    estimator = clone(namespace["pipeline"]).named_steps["estimator"]
                    # The protocol runner applies these coupled fields to each
                    # clone before candidate fitting, never as competing scores.
                    protocol, = pretraining_search.validation_protocols(namespace["VALIDATION_PARAM_SPACE"])
                    for name in ("validation_group_column", "pretraining_validation_group_column"):
                        estimator.set_params(**{name: protocol["group_column"]})
                        self.assertEqual(getattr(estimator, name), expected)
                    estimator.set_params(extractor__pretraining_source="all")
                    estimator.extractor_ = clone(estimator.extractor)
                    fitting, validation = estimator._split_validation_sequences(frame, labels)
                    self.assertTrue(set(fitting.sequence_id).isdisjoint(validation.sequence_id))
                    self.assertTrue(set(estimator.preprocessing_sequence_ids_).isdisjoint(estimator.validation_sequence_ids_))
                    self.assertTrue(set(estimator.preprocessing_sequence_ids_).isdisjoint(estimator.pretraining_validation_sequence_ids_))
                    self.assertTrue(set(estimator._pretraining_X.sequence_id).isdisjoint(estimator.pretraining_validation_sequence_ids_))
                    if mode == "subject":
                        fitting_subjects = set(frame.loc[frame.sequence_id.isin(estimator.preprocessing_sequence_ids_), "subject"])
                        validation_subjects = set(frame.loc[frame.sequence_id.isin(estimator.validation_sequence_ids_), "subject"])
                        self.assertTrue(fitting_subjects.isdisjoint(validation_subjects))

    def test_sequence_mode_allows_missing_subject_and_subject_mode_rejects_it(self):
        frame = full_modality_recordings().drop(columns="subject")
        for variant in self.notebooks:
            for mode in ("sequence", "subject"):
                with self.subTest(variant=variant, mode=mode), TemporaryDirectory() as directory:
                    namespace = self.namespace(variant, directory, validation_mode=mode)
                    protocol, = pretraining_search.validation_protocols(namespace["VALIDATION_PARAM_SPACE"])
                    with redirect_stdout(StringIO()):
                        if mode == "subject":
                            with self.assertRaisesRegex(ValueError, "subject"):
                                pretraining_search.split_validation_protocol(
                                    frame, protocol, train_size=namespace["train_size"],
                                    n_splits=namespace["n_splits"], random_state=namespace["random_state"])
                        else:
                            train, test, _, _ = pretraining_search.split_validation_protocol(
                                frame, protocol, train_size=namespace["train_size"],
                                n_splits=namespace["n_splits"], random_state=namespace["random_state"])
                            self.assertTrue(set(train.sequence_id).isdisjoint(test.sequence_id))

    def test_invalid_validation_mode_is_rejected(self):
        for variant in self.notebooks:
            with self.subTest(variant=variant), TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, "validation_mode"):
                    namespace = self.namespace(variant, directory, validation_mode="rows")
                    pretraining_search.validation_protocols(namespace["VALIDATION_PARAM_SPACE"])



if __name__ == "__main__":
    unittest.main()
