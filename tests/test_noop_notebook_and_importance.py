"""Check the existing notebook's NOOP wiring and sequence permutation semantics."""

import ast
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from multibranch_utils import compute_permutation_importance, _donor_rows, _sequence_layout


class _CachedExtractor:
    sequence_col = "sequence_id"

    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = 0

    def transform_chunks(self, frame):
        self.calls += 1
        return self.chunks


class _Predictor:
    """A known input-dependent fixture, not an alternate importance model."""

    def __init__(self, mask, constant=False, fail_at=None):
        self.mask = mask.copy()
        self.constant = constant
        self.fail_at = fail_at
        self.calls = 0

    def predict(self, inputs, verbose=0):
        self.calls += 1
        if self.fail_at == self.calls:
            raise RuntimeError("fixture prediction failure")
        np.testing.assert_array_equal(inputs[-1], self.mask)
        # Read the fitted network input column directly; all other columns ignored.
        p = np.zeros(len(self.mask)) if self.constant else inputs[2][:, 0]
        return np.column_stack([1 - p, p])


def _fixture(constant=False, fail_at=None):
    ids = np.array(["a", "a", "b", "b"])
    mask = np.array([[True, True, False], [True, True, True]] * 2)
    tensor = np.zeros((4, 3, 2), dtype=np.float32)
    tensor[0, :2, :] = [10, 100]
    tensor[1, :, :] = [11, 101]
    tensor[2, :2, :] = [20, 200]
    tensor[3, :, :] = [21, 201]
    noop = np.array([[0, 4], [0, 5], [1, 6], [1, 7]], dtype=np.float32)
    chunks = {"X": tensor, "mask": mask, "sequence_ids": ids,
              "feature_names": ["acc_x_raw", "tof_value"],
              "noop_features": noop, "noop_feature_names": ["noop_signal", "noop_ignored"]}
    extractor = _CachedExtractor(chunks)
    network = _Predictor(mask, constant=constant, fail_at=fail_at)
    estimator = SimpleNamespace(
        extractor_=extractor, model_=network,
        le_=SimpleNamespace(classes_=np.array(["left", "right"])),
        input_order_=["acc", "tof", "noop"],
        primary_target="bfrb", validation_sequence_ids_=pd.Index(["a", "b"]),
        train_sequence_ids_=pd.Index(["train"]),
        noop_feature_names_=["noop_signal", "noop_ignored"],
        _branches_from_chunks=lambda c: {
            "acc": c["X"][:, :, :1], "tof": c["X"][:, :, 1:], "noop": c["noop_features"]},
        _model_inputs=lambda branches, m: [branches["acc"], branches["tof"], branches["noop"], m],
    )
    frame = pd.DataFrame({"sequence_id": ["a", "b"], "unused": [1, 2]})
    labels = pd.DataFrame({"sequence_id": ["a", "b"], "bfrb": ["left", "right"]})
    return estimator, frame, labels, chunks


class PermutationImportanceTests(unittest.TestCase):
    def test_actual_predictor_inputs_per_feature_and_single_extraction(self):
        estimator, frame, labels, chunks = _fixture()
        original = deepcopy(chunks)
        frame_original = frame.copy(deep=True)
        result = compute_permutation_importance(estimator, frame, labels, repeats=3)
        rows = result.set_index(["kind", "name"])
        self.assertEqual(rows.loc[("noop_feature", "noop_signal"), "importance_mean"], 1.)
        self.assertEqual(rows.loc[("noop_feature", "noop_ignored"), "importance_mean"], 0.)
        self.assertEqual(rows.loc[("group", "tof"), "importance_mean"], 0.)
        self.assertEqual(rows.loc[("group", "motion_coherent"), "importance_mean"], 1.)
        self.assertEqual(result.attrs["baseline_macro_f1"], 1.)
        self.assertEqual(estimator.extractor_.calls, 1)
        self.assertEqual(estimator.model_.calls, 1 + len(result) * 3)
        self.assertEqual(result.attrs["permutable_fraction"], 1.)
        for key in ("X", "mask", "sequence_ids", "noop_features"):
            np.testing.assert_array_equal(chunks[key], original[key])
        pd.testing.assert_frame_equal(frame, frame_original)

    def test_predictor_with_no_feature_dependence_has_zero_drop(self):
        estimator, frame, labels, _ = _fixture(constant=True)
        result = compute_permutation_importance(estimator, frame, labels, repeats=2)
        np.testing.assert_array_equal(result["importance_mean"], 0.)
        np.testing.assert_array_equal(result["importance_std"], 0.)

    def test_original_arrays_restored_even_when_prediction_raises(self):
        estimator, frame, labels, chunks = _fixture(fail_at=2)
        original = {key: chunks[key].copy() for key in ("X", "mask", "noop_features")}
        with self.assertRaisesRegex(RuntimeError, "fixture prediction failure"):
            compute_permutation_importance(estimator, frame, labels)
        for key, value in original.items():
            np.testing.assert_array_equal(chunks[key], value)

    def test_sequence_donors_preserve_window_order_and_padding_layout(self):
        ids = np.array(["a", "a", "b", "b", "singleton"])
        mask = np.array([[1, 1, 0], [1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0]], dtype=bool)
        unique, rows, strata = _sequence_layout(ids, mask)
        donor, moved = _donor_rows(rows, strata, np.random.default_rng(42), len(ids))
        np.testing.assert_array_equal(donor, [2, 3, 0, 1, 4])
        np.testing.assert_array_equal(mask[donor], mask)
        self.assertEqual(moved, 2)
        self.assertEqual(len(unique), 3)

    def test_rejects_final_holdout_or_training_sequences(self):
        estimator, frame, labels, _ = _fixture()
        with self.assertRaisesRegex(ValueError, "outside that partition"):
            compute_permutation_importance(estimator, frame.assign(sequence_id=["a", "holdout"]), labels)
        estimator.train_sequence_ids_ = pd.Index(["a"])
        with self.assertRaisesRegex(ValueError, "training sequences"):
            compute_permutation_importance(estimator, frame, labels)

    def test_rejects_misaligned_labels_and_invalid_repeat_count(self):
        estimator, frame, labels, _ = _fixture()
        with self.assertRaisesRegex(ValueError, "Missing labels"):
            compute_permutation_importance(estimator, frame, labels.iloc[:1])
        conflict = pd.concat([labels, pd.DataFrame({"sequence_id": ["a"], "bfrb": ["right"]})])
        with self.assertRaisesRegex(ValueError, "constant within"):
            compute_permutation_importance(estimator, frame, conflict)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            compute_permutation_importance(estimator, frame, labels, repeats=0)

    def test_saves_actual_scores_scope_and_plot(self):
        estimator, frame, labels, _ = _fixture()
        with TemporaryDirectory() as directory:
            result = compute_permutation_importance(estimator, frame, labels, repeats=2, output_dir=directory)
            paths = result.attrs["paths"]
            for path in paths.values():
                self.assertTrue(Path(path).is_file())
            payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["metadata"]["scope"], "saved internal validation sequences")
            self.assertEqual(payload["metadata"]["baseline_macro_f1"], 1.)
            self.assertEqual(len(payload["importance"]), len(result))
            self.assertEqual(len(pd.read_csv(paths["csv"])), len(result))


class NoopNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads((ROOT / "notebooks/multibranch_pretraining.ipynb").read_text(encoding="utf-8"))

    @staticmethod
    def source(notebook, index):
        return "".join(notebook["cells"][index]["source"])

    def test_all_cells_parse_and_saved_outputs_are_stripped(self):
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])

    def test_noop_options_are_exposed_in_each_editable_search_space(self):
        from noop_sequence_extractor import NoopOptions
        from test_pretraining_large_notebook import LargePretrainingNotebookTests

        loader = LargePretrainingNotebookTests()
        loader.notebooks = {"model1": self.notebook}
        with TemporaryDirectory() as directory:
            namespace = loader.namespace("model1", directory)
        prefix = "estimator__extractor__noop__"
        expected_keys = set(NoopOptions().get_params(deep=False))
        for name in ("GRID_PARAM_SPACE", "BAYESIAN_PARAM_SPACE", "EVOLUTIONARY_PARAM_SPACE"):
            with self.subTest(space=name):
                space = namespace[name]
                self.assertEqual({key[len(prefix):] for key in space if key.startswith(prefix)},
                                 expected_keys)
                for mode in ("motion_stats", "periodicity", "spectral_stats", "quality"):
                    dimension = space[prefix + mode]
                    choices = (dimension.categories if hasattr(dimension, "categories") else
                               dimension.choices if hasattr(dimension, "choices") else dimension)
                    self.assertIn(None, choices, (name, mode))

    def test_extractors_and_grid_deduplication_are_connected(self):
        setup = self.source(self.notebook, 12)
        self.assertIn("extractor = NoopSequenceExtractor(", setup)
        self.assertIn("noop=NoopOptions()", setup)
        train = self.source(self.notebook, 20)
        self.assertEqual(train.count("deduplicate_param_grid("), 2)
        self.assertIn("base_params=pipeline.get_params(deep=True)", train)

    def test_saved_best_params_are_the_winning_reference_and_not_overwritten(self):
        reference = json.loads((ROOT / "experiments/noop_candidate21/reference_params.json").read_text(encoding="utf-8"))
        tree = ast.parse(self.source(self.notebook, 5))
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "BEST_PARAMS" for t in node.targets))
        saved = ast.literal_eval(assignment.value)
        for key, value in reference.items():
            self.assertEqual(saved[key], value, key)
        self.assertEqual(saved["estimator__epochs"], 2000)
        self.assertEqual(saved["estimator__early_stopping_patience"], 0)
        self.assertFalse(saved["estimator__restore_best_weights"])
        for mode in ("motion_stats", "periodicity", "spectral_stats", "quality"):
            self.assertIsNone(saved["estimator__extractor__noop__" + mode])
        space_tree = ast.parse(self.source(self.notebook, 18))
        self.assertFalse(any(isinstance(node, ast.Assign)
                             and any(isinstance(t, ast.Name) and t.id == "BEST_PARAMS" for t in node.targets)
                             for node in ast.walk(space_tree)))
        self.assertFalse((ROOT / "notebooks/multibranch_v2.ipynb").exists())

    def test_search_mode_selects_declared_spaces_without_fixed_experiment_overrides(self):
        tree = ast.parse(self.source(self.notebook, 18))
        selection = next(node for node in tree.body if isinstance(node, ast.If)
                         and ast.unparse(node.test) == "search_mode == 'bayes'")
        spaces = {"GRID_PARAM_SPACE": object(), "BAYESIAN_PARAM_SPACE": object(),
                  "EVOLUTIONARY_SEARCH_SPACE": object()}
        for mode, expected in (("grid", "GRID_PARAM_SPACE"), ("bayes", "BAYESIAN_PARAM_SPACE"),
                               ("evo", "EVOLUTIONARY_SEARCH_SPACE")):
            namespace = dict(spaces, search_mode=mode)
            exec(compile(ast.Module(body=[selection], type_ignores=[]), "space-selection", "exec"), namespace)
            self.assertIs(namespace["param_space"], spaces[expected])

    def test_importance_is_provided_by_the_shared_multibranch_utilities(self):
        from multibranch_utils import save_permutation_importance

        self.assertEqual(compute_permutation_importance.__module__, "multibranch_utils")
        self.assertEqual(save_permutation_importance.__module__, "multibranch_utils")
        self.assertFalse((ROOT / "src/noop_importance.py").exists())
        imports = [node for node in ast.walk(ast.parse(self.source(self.notebook, 2)))
                   if isinstance(node, ast.ImportFrom)]
        self.assertTrue(any(node.module == "multibranch_utils" and
                            any(name.name == "compute_permutation_importance" for name in node.names)
                            for node in imports))
        self.assertFalse(any(node.module == "noop_importance" for node in imports))

    def test_importance_uses_internal_train_partition_not_final_holdout(self):
        source = self.source(self.notebook, len(self.notebook["cells"]) - 1)
        tree = ast.parse(source)
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        self.assertTrue({"X_train", "y_train", "X_internal", "y_internal"}.issubset(names))
        self.assertNotIn("X_test", names)
        self.assertNotIn("y_test", names)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "compute_permutation_importance"]
        self.assertEqual(len(calls), 1)
        kwargs = {kw.arg: kw.value for kw in calls[0].keywords}
        self.assertTrue(ast.literal_eval(kwargs["require_internal_validation"]))
        self.assertIn("validation_sequence_ids_", source)
        self.assertIn("select_finetuning_sequences(raw_internal, y_train)", source)
        self.assertIn("y_internal = y_train.loc[X_internal.index]", source)


@unittest.skipUnless(importlib.util.find_spec("tensorflow") is not None, "TensorFlow runtime not available")
class ActualNetworkImportanceTests(unittest.TestCase):
    def test_fitted_advanced_network_noop_branch_and_importance(self):
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        # Some unrelated discovery tests install a tensorflow module shim. Its
        # import spec alone does not mean the multibranch runtime can train.
        import multibranch_utils as mb
        if not mb.TENSORFLOW_AVAILABLE:
            self.skipTest("The multibranch TensorFlow runtime is not available")
        from noop_sequence_extractor import NoopOptions, NoopSequenceExtractor
        from pretraining_utils import AdvancedMultiBranchSequenceClassifier
        rows = []
        for subject in range(4):
            for label in range(2):
                for counter in range(16):
                    rows.append({
                        "sequence_id": f"s{subject}-c{label}", "subject": f"s{subject}",
                        "sequence_counter": counter, "is_target": 1,
                        "gesture": f"gesture{label}", "bfrb": f"target{label}",
                        "orientation": "upright", "acc_x": np.sin(counter / 3 + label),
                        "acc_y": np.cos(counter / 3), "acc_z": 9.8 + .1 * np.sin(counter),
                        "rot_w": 1., "rot_x": 0., "rot_y": 0., "rot_z": 0.,
                    })
        frame = pd.DataFrame(rows)
        labels = frame[["sequence_id", "subject", "gesture", "bfrb", "orientation", "is_target"]]
        extractor = NoopSequenceExtractor(
            noop=NoopOptions(motion_stats="acc_ac_rms", min_samples=4),
            acc_modes="raw", rotation_modes="quaternion", tof_modes="", thm_modes="",
            stft_configs=None, cwt_configs=None, chunk_window_size=16, chunk_stride=16,
            pretraining_source="baseline", finetuning_source="target",
        )
        model = AdvancedMultiBranchSequenceClassifier(
            extractor=extractor, temporal_architecture="bert", bert_num_layers=1,
            bert_d_model=8, bert_num_heads=2, bert_ff_dim=16, bert_dropout=0.,
            branch_cnn_filters=4, branch_cnn_depth=1, branch_cnn_kernel_size=3,
            branch_dropout=0., fusion_units=8, fusion_dropout=0., dense_units=8,
            dropout=0., conv_dropout=0., static_dropout=0., static_branch_units=4,
            use_batchnorm=False, epochs=1, batch_size=4, validation_split=.25,
            validation_group_column="subject", early_stopping_patience=0,
            use_lr_scheduler=False, invalid_candidate_policy="raise", verbose=0,
            random_state=7,
        ).fit(frame, labels)
        raw_internal = frame.loc[frame.sequence_id.isin(model.validation_sequence_ids_)]
        internal = model.extractor_.select_finetuning_sequences(raw_internal, labels)
        y_internal = labels.loc[internal.index]
        before = model.predict_proba(internal)
        result = compute_permutation_importance(model, internal, y_internal, repeats=1)
        pd.testing.assert_frame_equal(before, model.predict_proba(internal))
        self.assertIn("noop", model.input_order_)
        self.assertTrue(result["importance_mean"].map(np.isfinite).all())
        self.assertTrue((result.kind == "noop_feature").any())


if __name__ == "__main__":
    unittest.main()
