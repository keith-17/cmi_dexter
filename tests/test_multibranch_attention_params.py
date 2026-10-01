"""Parameter-contract tests for the multi-branch notebook and estimator.

These tests deliberately do not run a hyperparameter search: they exercise every
candidate independently, which keeps the suite fast while catching renamed,
removed, or incorrectly serialised parameters before a long grid/Bayesian run.
"""

from __future__ import annotations

import ast
import inspect
import json
import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import multibranch_attention as mba  # noqa: E402
from base_utils_qwen import SensorAugmentor, SequenceExtractor  # noqa: E402


NOTEBOOK = ROOT / "notebooks" / "multibranch_attention.ipynb"


def notebook_grid_param_space():
    """Read the literal ``GRID_PARAM_SPACE`` without executing the notebook."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = next(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if "GRID_PARAM_SPACE = {" in "".join(cell.get("source", []))
    )
    tree = ast.parse(source)
    assignment_index = next(
        index
        for index, node in enumerate(tree.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "GRID_PARAM_SPACE" for target in node.targets)
    )
    # The notebook uses named candidate lists (for example ``STFT_CONFIGS``),
    # so evaluate only the declarations leading up to the grid definition.
    namespace = {"problematic_sequence_bool": True}
    prefix = ast.Module(body=tree.body[: assignment_index + 1], type_ignores=[])
    exec(compile(prefix, str(NOTEBOOK), "exec"), namespace)
    return namespace["GRID_PARAM_SPACE"]


def make_pipeline():
    return Pipeline(
        [
            ("augmentor", SensorAugmentor(sequence_col="sequence_id", counter_col="sequence_counter")),
            (
                "estimator",
                mba.MultiBranchSequenceClassifier(
                    extractor=SequenceExtractor(output_format="chunks"),
                    primary_target="bfrb",
                ),
            ),
        ]
    )


class NotebookParameterContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grid_space = notebook_grid_param_space()

    def test_every_grid_key_is_reachable_from_the_pipeline(self):
        available = set(make_pipeline().get_params(deep=True))
        self.assertFalse(set(self.grid_space) - available)

    def test_every_grid_candidate_can_be_set_independently(self):
        """Catch bad names and incompatible candidate values before GridSearchCV."""
        pipeline = make_pipeline()
        for key, candidates in self.grid_space.items():
            for candidate in candidates:
                with self.subTest(parameter=key, candidate=repr(candidate)):
                    clone(pipeline).set_params(**{key: candidate})

    def test_grid_never_combines_incompatible_gradient_clipping_modes(self):
        """Keras accepts norm clipping or value clipping, but never both."""
        norm_values = self.grid_space["estimator__gradient_clip_norm"]
        value_values = self.grid_space["estimator__gradient_clip_value"]
        self.assertFalse(
            any(norm is not None and value is not None for norm in norm_values for value in value_values)
        )

    def test_new_model_parameters_are_constructor_parameters(self):
        constructor_params = set(inspect.signature(mba.MultiBranchSequenceClassifier.__init__).parameters)
        model_params = {
            key.removeprefix("estimator__").split("__", 1)[0]
            for key in self.grid_space
            if key.startswith("estimator__")
        }
        self.assertFalse(model_params - constructor_params)

    def test_bayesian_space_preserves_all_keys_and_candidates(self):
        bayes_space = mba.prepare_multibranch_bayesian_space(self.grid_space)
        self.assertEqual(set(bayes_space), set(self.grid_space))
        pipeline = make_pipeline()
        for key, space in bayes_space.items():
            candidates = list(getattr(space, "categories", space))
            self.assertTrue(candidates, key)
            for candidate in candidates:
                with self.subTest(parameter=key, candidate=repr(candidate)):
                    clone(pipeline).set_params(**{key: candidate})

    def test_bayesian_branch_dicts_are_json_and_are_decoded_by_the_estimator(self):
        bayes_space = mba.prepare_multibranch_bayesian_space(self.grid_space)
        for key in (
            "estimator__branch_filters",
            "estimator__branch_kernel_sizes",
            "estimator__branch_pool_sizes",
        ):
            candidate = next(iter(getattr(bayes_space[key], "categories", bayes_space[key])))
            self.assertIsInstance(candidate, str)
            decoded = mba._json_maybe(candidate)
            self.assertIsInstance(decoded, dict)
            clone(make_pipeline()).set_params(**{key: candidate})


@unittest.skipUnless(mba.TENSORFLOW_AVAILABLE, "TensorFlow is required for Keras model construction tests")
class KerasParameterConstructionTests(unittest.TestCase):
    def build(self, **overrides):
        options = {
            "time_series_shapes": {"acc": (8, 3), "rotation": (8, 4)},
            "static_shapes": {"stft": 5, "cwt": 6},
            "n_classes": 3,
            "branch_filters": {"acc": 8, "rotation": 8},
            "branch_kernel_sizes": {"acc": 3, "rotation": 3},
            "branch_pool_sizes": {"acc": 1, "rotation": 1},
            "branch_num_conv_layers": {"acc": 2, "rotation": 1},
            "conv_dropout": 0.1,
        }
        options.update(overrides)
        return mba.build_multibranch_model(**options)

    def test_new_architecture_optimizer_and_loss_choices_construct_and_predict(self):
        for activation in ("relu", "gelu", "swish", "elu", "leaky"):
            for pooling_mode in ("attention", "mean", "max", "last"):
                with self.subTest(activation=activation, pooling_mode=pooling_mode):
                    model = self.build(activation=activation, pooling_mode=pooling_mode)
                    output = model([np.zeros((2, 8, 3)), np.zeros((2, 8, 4)), np.zeros((2, 5)), np.zeros((2, 6))])
                    self.assertEqual(tuple(output.shape), (2, 3))

        for optimizer_name in ("adam", "adamw", "rmsprop", "sgd", "nadam", "adagrad", "adadelta"):
            with self.subTest(optimizer=optimizer_name):
                self.assertEqual(self.build(optimizer_name=optimizer_name).output_shape, (None, 3))

        for loss_name in ("sparse_cce", "focal"):
            with self.subTest(loss=loss_name):
                self.assertEqual(
                    self.build(loss_name=loss_name, label_smoothing=0.1).output_shape,
                    (None, 3),
                )

    def test_all_branch_filter_modes_construct(self):
        configs = {
            "custom": {"acc": [8, 16], "rotation": [8]},
            "double": {"acc": 8, "rotation": 8},
            "constant": {"acc": 8, "rotation": 8},
        }
        for mode, filters in configs.items():
            with self.subTest(branch_filter_mode=mode):
                self.assertEqual(
                    self.build(branch_filter_mode=mode, branch_filters=filters).output_shape,
                    (None, 3),
                )

    def test_gradient_clipping_modes_construct_and_conflict_is_clear(self):
        for options in (
            {"gradient_clip_norm": 1.0},
            {"gradient_clip_value": 1.0},
            {},
        ):
            with self.subTest(options=options):
                self.assertEqual(self.build(**options).output_shape, (None, 3))

        with self.assertRaisesRegex(ValueError, "at most one"):
            self.build(gradient_clip_norm=1.0, gradient_clip_value=1.0)

    def test_sparse_label_smoothing_runs_a_training_step(self):
        """Exercise the Keras 3-compatible sparse smoothing loss at fit time."""
        model = self.build(loss_name="sparse_cce", label_smoothing=0.1)
        scheduler_owner = mba.MultiBranchSequenceClassifier(
            use_lr_scheduler=True,
            lr_scheduler_type="cosine",
            validation_split=0.2,
            early_stopping_patience=0,
            learning_rate=1e-3,
            min_lr=1e-6,
            warmup_epochs=1,
            cosine_t_max=2,
        )
        history = model.fit(
            [
                np.zeros((3, 8, 3), dtype=np.float32),
                np.zeros((3, 8, 4), dtype=np.float32),
                np.zeros((3, 5), dtype=np.float32),
                np.zeros((3, 6), dtype=np.float32),
            ],
            np.array([0, 1, 2]),
            batch_size=3,
            epochs=1,
            callbacks=scheduler_owner._build_callbacks(),
            verbose=0,
        )
        self.assertIn("loss", history.history)

    def test_plateau_and_cosine_scheduler_choices_are_keras_callbacks(self):
        for scheduler in ("plateau", "cosine"):
            with self.subTest(scheduler=scheduler):
                classifier = mba.MultiBranchSequenceClassifier(
                    use_lr_scheduler=True,
                    lr_scheduler_type=scheduler,
                    validation_split=0.2,
                    learning_rate=1e-3,
                    min_lr=1e-6,
                    warmup_epochs=1,
                    cosine_t_max=2,
                )
                callbacks = classifier._build_callbacks()
                self.assertGreaterEqual(len(callbacks), 2)
                self.assertTrue(all(isinstance(callback, mba.callbacks.Callback) for callback in callbacks))
