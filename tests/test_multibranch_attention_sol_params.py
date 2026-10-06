"""Parameter-contract tests for the multi-branch notebook and estimator.

These tests deliberately do not run a hyperparameter search: they exercise every
candidate independently, which keeps the suite fast while catching renamed,
removed, or incorrectly serialised parameters before a long grid/Bayesian run.
"""

from __future__ import annotations

import ast
from contextlib import redirect_stdout
from io import StringIO
import inspect
import json
import sys
import unittest
from types import MethodType, SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import multibranch_attention_sol as mba  # noqa: E402
from base_utils_sol import InvalidExtractorParams, SensorAugmentor, SequenceExtractor  # noqa: E402


NOTEBOOK = ROOT / "notebooks" / "multibranch_v2.ipynb"


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
            (
                "estimator",
                mba.MultiBranchSequenceClassifier(
                    extractor=SequenceExtractor(output_format="chunks"),
                    augmentor=SensorAugmentor(
                        sequence_col="sequence_id",
                        counter_col="sequence_counter",
                        prob=0.0,
                        per_aug_prob=0.0,
                    ),
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

    def test_new_architecture_parameters_are_cloneable_and_searchable(self):
        pipeline = make_pipeline()
        params = pipeline.get_params(deep=True)
        self.assertEqual(
            set(self.grid_space["estimator__temporal_architecture"]),
            {"bigru_attention", "phase_attention", "cnn_attention_pool", "bert"},
        )
        for name, value in (
            ("temporal_architecture", "bert"),
            ("bert_num_layers", 2),
            ("bert_d_model", 64),
            ("phase_attention_units", 32),
            ("tof_encoder_type", "spatial_cnn"),
        ):
            key = f"estimator__{name}"
            self.assertIn(key, params)
            clone(pipeline).set_params(**{key: value})

        for width in self.grid_space["estimator__bert_d_model"]:
            for heads in self.grid_space["estimator__bert_num_heads"]:
                self.assertEqual(width % heads, 0)

    def test_grid_never_combines_incompatible_gradient_clipping_modes(self):
        """Keras accepts norm clipping or value clipping, but never both."""
        norm_values = self.grid_space["estimator__gradient_clip_norm"]
        value_values = self.grid_space["estimator__gradient_clip_value"]
        self.assertFalse(
            any(norm is not None and value is not None for norm in norm_values for value in value_values)
        )

    def _make_chunk_extractor(self, **params):
        extractor = SequenceExtractor(
            output_format="chunks",
            **params,
        )
        extractor.feature_names_in_ = np.array(["feature"])
        extractor._preprocess_features = lambda frame: frame
        return extractor

    def _sequence_frame(self, sequence_lengths):
        sequence_ids = []
        values = []
        for sequence_id, length in sequence_lengths.items():
            sequence_ids.extend([sequence_id] * length)
            values.extend(range(length))
        return pd.DataFrame({"sequence_id": sequence_ids, "feature": values})

    def test_per_sequence_cap_is_applied_independently(self):
        extractor = self._make_chunk_extractor(
            chunk_window_size=100,
            chunk_stride=50,
            max_windows_per_sequence=2,
        )
        frame = self._sequence_frame({"a": 150, "b": 150})
        output = StringIO()
        with redirect_stdout(output):
            chunks = extractor.transform_chunks(frame)
        self.assertEqual(chunks["X"].shape, (4, 100, 1))
        self.assertEqual(chunks["sequence_ids"].tolist(), ["a", "a", "b", "b"])
        self.assertIn("sequences=2, capped_sequences=2", output.getvalue())
        self.assertIn("max_windows_before_cap=3, max_windows_after_cap=2", output.getvalue())

    def test_internal_validation_chunk_ids_are_disjoint(self):
        frame = self._sequence_frame({"a": 250, "b": 250, "c": 250, "d": 250})
        extractor = self._make_chunk_extractor(
            chunk_window_size=100,
            chunk_stride=50,
            maxlen=0,
        )
        classifier = mba.MultiBranchSequenceClassifier(
            extractor=extractor,
            validation_split=0.25,
            random_state=13,
        )
        classifier.extractor_ = extractor
        targets = pd.DataFrame({
            "sequence_id": ["a", "b", "c", "d"],
            "bfrb": ["A", "A", "B", "B"],
        })
        train_rows, validation_rows = classifier._split_validation_sequences(frame, targets)
        train_chunks = extractor.transform_chunks(train_rows)
        validation_chunks = extractor.transform_chunks(validation_rows)

        train_ids = set(train_chunks["sequence_ids"])
        validation_ids = set(validation_chunks["sequence_ids"])
        self.assertTrue(train_ids)
        self.assertTrue(validation_ids)
        self.assertTrue(train_ids.isdisjoint(validation_ids))
        self.assertEqual(train_ids, set(classifier.train_sequence_ids_))
        self.assertEqual(validation_ids, set(classifier.validation_sequence_ids_))

    def test_validation_split_stratifies_and_falls_back_for_rare_classes(self):
        classifier = mba.MultiBranchSequenceClassifier(validation_split=0.4, random_state=13)
        classifier.extractor_ = SequenceExtractor(output_format="chunks")
        sequence_ids = [f"s{index}" for index in range(10)]
        frame = self._sequence_frame({sequence_id: 2 for sequence_id in sequence_ids})
        labels = pd.DataFrame({
            "sequence_id": sequence_ids,
            "bfrb": ["A"] * 5 + ["B"] * 5,
        })

        _, validation = classifier._split_validation_sequences(frame, labels)

        validation_labels = validation[["sequence_id"]].drop_duplicates().merge(
            labels, on="sequence_id"
        )["bfrb"]
        self.assertEqual(validation_labels.value_counts().to_dict(), {"A": 2, "B": 2})

        rare_ids = [f"r{index}" for index in range(6)]
        rare_frame = self._sequence_frame({sequence_id: 2 for sequence_id in rare_ids})
        rare_labels = pd.DataFrame({"sequence_id": rare_ids, "bfrb": rare_ids})
        train, validation = classifier._split_validation_sequences(rare_frame, rare_labels)
        self.assertTrue(
            set(train["sequence_id"].unique()).isdisjoint(validation["sequence_id"].unique())
        )

    def test_fit_uses_explicit_validation_and_augments_training_only(self):
        class SpyAugmentor:
            def __init__(self):
                self.calls = 0

            def fit_transform(self, frame):
                self.calls += 1
                self.seen_sequence_ids = set(frame["sequence_id"].unique())
                return frame.copy()

        class RecordingModel:
            def fit(self, *args, **kwargs):
                self.fit_kwargs = kwargs
                return SimpleNamespace(history={
                    "loss": [1.0],
                    "val_loss": [1.1],
                    "val_accuracy": [0.5],
                })

        extractor = SequenceExtractor(
            output_format="chunks",
            chunk_window_size=2,
            chunk_stride=2,
            maxlen=0,
        )

        def fake_fit(extractor_self, frame, y=None):
            extractor_self.feature_names_in_ = np.array(["acc_x"])
            return extractor_self

        def fake_transform_chunks(extractor_self, frame):
            chunk_ids = np.repeat(frame["sequence_id"].drop_duplicates().to_numpy(), 2)
            return {
                "X": np.zeros((len(chunk_ids), 2, 1), dtype=np.float32),
                "mask": np.ones((len(chunk_ids), 2), dtype=bool),
                "sequence_ids": chunk_ids,
                "feature_names": ["acc_x"],
            }

        extractor.fit = MethodType(fake_fit, extractor)
        extractor.transform_chunks = MethodType(fake_transform_chunks, extractor)
        augmentor = SpyAugmentor()
        classifier = mba.MultiBranchSequenceClassifier(
            extractor=extractor,
            augmentor=augmentor,
            validation_split=0.25,
            class_weight_mode=None,
            early_stopping_patience=0,
            random_state=13,
        )
        frame = self._sequence_frame({"a": 4, "b": 4, "c": 4, "d": 4})
        labels = pd.DataFrame({"sequence_id": ["a", "b", "c", "d"], "bfrb": ["target"] * 4})
        model = RecordingModel()

        with patch.object(mba, "TENSORFLOW_AVAILABLE", True), \
                patch.object(mba, "clone", side_effect=lambda estimator: estimator), \
                patch.object(mba, "build_multibranch_model", return_value=model):
            classifier.fit(frame, labels)

        self.assertTrue(set(classifier.train_sequence_ids_).isdisjoint(classifier.validation_sequence_ids_))
        self.assertEqual(augmentor.calls, 1)
        self.assertEqual(augmentor.seen_sequence_ids, set(classifier.train_sequence_ids_))
        self.assertIsNotNone(model.fit_kwargs["validation_data"])
        self.assertEqual(model.fit_kwargs["validation_split"], 0.0)
        self.assertIn("val_loss", classifier.history_)
        self.assertIn("val_accuracy", classifier.history_)

        validation_rows = frame[frame["sequence_id"].isin(classifier.validation_sequence_ids_)]
        classifier._transform_to_branches(validation_rows)
        self.assertEqual(augmentor.calls, 1)

    def test_none_stft_cwt_configs_create_no_branches(self):
        extractor = SequenceExtractor(
            output_format="chunks",
            stft_configs=None,
            cwt_configs=None,
        )
        classifier = mba.MultiBranchSequenceClassifier(extractor=extractor)
        classifier.extractor_ = extractor

        self.assertEqual(extractor.stft_extractors, [])
        self.assertEqual(extractor.cwt_extractors, [])
        self.assertEqual(set(classifier._branch_groups(["acc_x"])), {"acc"})

    def test_derived_imu_branch_is_separate_only_for_new_temporal_paths(self):
        classifier = mba.MultiBranchSequenceClassifier()
        classifier.extractor_ = SequenceExtractor(stft_configs=None, cwt_configs=None)
        feature_names = [
            "acc_x_raw",
            "acc_x_raw_integrated_velocity",
            "lin_acc_x",
            "rot_x",
            "tof_sensor0_v0",
        ]

        legacy_groups = classifier._branch_groups(feature_names)
        self.assertEqual(
            legacy_groups,
            {"acc": [0, 1, 2], "rotation": [3], "tof": [4]},
        )

        temporal_groups = classifier._branch_groups(
            feature_names,
            separate_derived_imu=True,
        )
        self.assertEqual(
            temporal_groups,
            {
                "acc": [0],
                "derived_imu": [1, 2],
                "rotation": [3],
                "tof": [4],
            },
        )

    def test_callbacks_monitor_validation_loss_only_when_validation_exists(self):
        class Callback:
            def __init__(self, monitor=None, **kwargs):
                self.monitor = monitor

        callback_module = SimpleNamespace(
            EarlyStopping=Callback,
            ReduceLROnPlateau=Callback,
            LearningRateScheduler=Callback,
        )
        classifier = mba.MultiBranchSequenceClassifier(
            early_stopping_patience=2,
            use_lr_scheduler=True,
            lr_scheduler_type="plateau",
        )
        with patch.object(mba, "callbacks", callback_module):
            validation_callbacks = classifier._build_callbacks(has_validation_data=True)
            no_validation_callbacks = classifier._build_callbacks(has_validation_data=False)

        self.assertEqual(validation_callbacks[0].monitor, "val_loss")
        self.assertEqual(no_validation_callbacks[0].monitor, "loss")
        self.assertEqual(len(validation_callbacks), 2)
        self.assertEqual(len(no_validation_callbacks), 1)

    def test_crop_modes_choose_expected_timesteps(self):
        frame = self._sequence_frame({"long": 500})
        expected_first_values = {"head": 0.0, "tail": 300.0, "center": 150.0, "none": 0.0}
        expected_window_counts = {"head": 1, "tail": 1, "center": 1, "none": 3}

        for crop_mode, expected_first in expected_first_values.items():
            with self.subTest(crop_mode=crop_mode):
                extractor = self._make_chunk_extractor(
                    maxlen=200,
                    sequence_crop_mode=crop_mode,
                    chunk_window_size=200,
                    use_chunk_stride_ratio=True,
                    chunk_stride_ratio=1.0,
                )
                chunks = extractor.transform_chunks(frame)
                self.assertEqual(chunks["X"].shape[0], expected_window_counts[crop_mode])
                self.assertEqual(chunks["X"][0, 0, 0], expected_first)

    def test_frame_output_uses_crop_mode_before_sequence_statistics(self):
        extractor = SequenceExtractor(
            output_format="frame",
            maxlen=200,
            sequence_crop_mode="tail",
            frame_stats="mean",
        )
        extractor.base_feature_names_ = ["feature"]
        extractor.frame_feature_names_ = ["feature_mean"]
        extractor.frame_stats_ = ["mean"]
        extractor._preprocess_features = lambda frame: frame
        frame = self._sequence_frame({"long": 500})

        result = extractor.transform_frame(frame)

        self.assertEqual(result.loc["long", "feature_mean"], np.mean(np.arange(300, 500)))

    def test_tail_crop_with_shorter_window_creates_expected_pad_mode_windows(self):
        extractor = self._make_chunk_extractor(
            maxlen=200,
            sequence_crop_mode="tail",
            chunk_window_size=100,
            use_chunk_stride_ratio=True,
            chunk_stride_ratio=1.0,
            final_window_mode="pad",
        )
        chunks = extractor.transform_chunks(self._sequence_frame({"long": 500}))
        self.assertEqual(chunks["X"][:, 0, 0].tolist(), [300.0, 400.0])

    def test_overlap_mode_and_last_cap_keep_the_final_windows(self):
        extractor = self._make_chunk_extractor(
            maxlen=200,
            sequence_crop_mode="none",
            chunk_window_size=100,
            use_chunk_stride_ratio=True,
            chunk_stride_ratio=0.2,
            final_window_mode="overlap",
            max_windows_per_sequence=4,
            window_cap_policy="last",
        )
        chunks = extractor.transform_chunks(self._sequence_frame({"long": 500}))
        self.assertEqual(chunks["X"][:, 0, 0].tolist(), [340.0, 360.0, 380.0, 400.0])

    def test_uniform_cap_includes_first_and_final_windows(self):
        extractor = self._make_chunk_extractor(
            sequence_crop_mode="none",
            chunk_window_size=100,
            use_chunk_stride_ratio=True,
            chunk_stride_ratio=0.2,
            final_window_mode="overlap",
            max_windows_per_sequence=4,
            window_cap_policy="uniform",
        )
        chunks = extractor.transform_chunks(self._sequence_frame({"long": 500}))
        self.assertEqual(chunks["X"][:, 0, 0].tolist(), [0.0, 140.0, 260.0, 400.0])

    def test_first_cap_policy_keeps_initial_windows(self):
        extractor = self._make_chunk_extractor(
            sequence_crop_mode="none",
            chunk_window_size=100,
            use_chunk_stride_ratio=True,
            chunk_stride_ratio=0.2,
            final_window_mode="overlap",
            max_windows_per_sequence=4,
            window_cap_policy="first",
        )
        chunks = extractor.transform_chunks(self._sequence_frame({"long": 500}))
        self.assertEqual(chunks["X"][:, 0, 0].tolist(), [0.0, 20.0, 40.0, 60.0])

    def test_absolute_stride_is_clamped_to_window_size(self):
        extractor = self._make_chunk_extractor(
            sequence_crop_mode="none",
            chunk_window_size=100,
            chunk_stride=200,
        )
        chunks = extractor.transform_chunks(self._sequence_frame({"sequence": 250}))
        self.assertEqual(chunks["X"][:, 0, 0].tolist(), [0.0, 100.0, 200.0])

    def test_short_sequence_is_padded_and_masked(self):
        extractor = self._make_chunk_extractor(
            maxlen=0,
            chunk_window_size=100,
            chunk_stride=100,
        )
        chunks = extractor.transform_chunks(self._sequence_frame({"short": 80}))
        self.assertEqual(chunks["X"].shape, (1, 100, 1))
        self.assertTrue(chunks["mask"][0, :80].all())
        self.assertFalse(chunks["mask"][0, 80:].any())

    def test_pad_and_overlap_final_window_modes(self):
        frame = self._sequence_frame({"sequence": 250})
        pad_extractor = self._make_chunk_extractor(
            maxlen=0,
            chunk_window_size=100,
            chunk_stride=100,
            final_window_mode="pad",
        )
        overlap_extractor = self._make_chunk_extractor(
            maxlen=0,
            chunk_window_size=100,
            chunk_stride=100,
            final_window_mode="overlap",
        )
        pad_chunks = pad_extractor.transform_chunks(frame)
        overlap_chunks = overlap_extractor.transform_chunks(frame)
        self.assertEqual(pad_chunks["X"][:, 0, 0].tolist(), [0.0, 100.0, 200.0])
        self.assertFalse(pad_chunks["mask"][-1, 50:].any())
        self.assertEqual(overlap_chunks["X"][:, 0, 0].tolist(), [0.0, 100.0, 150.0])

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

    def test_all_temporal_architectures_construct_and_predict(self):
        cases = (
            ("bigru_attention", {"use_bigru": True}),
            ("phase_attention", {"phase_fusion_mode": "concat"}),
            ("phase_attention", {"phase_fusion_mode": "sum"}),
            ("phase_attention", {"phase_fusion_mode": "learned"}),
            ("cnn_attention_pool", {"fusion_type": "sum"}),
            ("cnn_attention_pool", {"fusion_type": "attention"}),
            ("cnn_attention_pool", {"post_fusion_pooling": "attention"}),
            ("cnn_attention_pool", {"post_fusion_pooling": "avg"}),
            ("cnn_attention_pool", {"post_fusion_pooling": "max"}),
            ("cnn_attention_pool", {"post_fusion_pooling": "avg_max"}),
            ("cnn_attention_pool", {"post_fusion_pooling": "attention_avg_max"}),
            ("bert", {"bert_positional_encoding": "learned"}),
            ("bert", {"bert_positional_encoding": "sinusoidal"}),
            ("bert", {"bert_pooling": "cls", "bert_use_cls_token": True}),
            ("bert", {"bert_pooling": "avg"}),
            ("bert", {"bert_pooling": "avg_max"}),
        )
        for architecture, overrides in cases:
            with self.subTest(architecture=architecture, options=overrides):
                model = self.build(temporal_architecture=architecture, **overrides)
                inputs = [
                    np.zeros((2, 8, 3), dtype=np.float32),
                    np.zeros((2, 8, 4), dtype=np.float32),
                    np.zeros((2, 5), dtype=np.float32),
                    np.zeros((2, 6), dtype=np.float32),
                    np.ones((2, 8), dtype=np.float32),
                ]
                output = model(inputs, training=False)
                if isinstance(output, dict):
                    output = output["predictions"]
                self.assertEqual(tuple(output.shape), (2, 3))

    def test_temporal_model_masks_padding_and_handles_spatial_tof(self):
        model = self.build(
            temporal_architecture="cnn_attention_pool",
            time_series_shapes={"acc": (8, 3), "tof": (8, 64)},
            static_shapes={},
            tof_encoder_type="spatial_cnn_gru",
        )
        output = model(
            [
                np.zeros((2, 8, 3), dtype=np.float32),
                np.zeros((2, 8, 64), dtype=np.float32),
                np.array([[1] * 8, [1, 1, 1, 0, 0, 0, 0, 0]], dtype=np.float32),
            ],
            training=False,
        )
        self.assertEqual(tuple(output.shape), (2, 3))

        with self.assertRaisesRegex(ValueError, "multiple of 64"):
            self.build(
                temporal_architecture="bert",
                time_series_shapes={"tof": (8, 63)},
                static_shapes={},
                tof_encoder_type="spatial_cnn",
            )

    def test_optional_auxiliary_heads_accept_explicit_labels(self):
        model = self.build(
            temporal_architecture="phase_attention",
            use_orientation_aux_head=True,
            use_phase_aux_head=True,
            orientation_num_classes=4,
        )
        inputs = [
            np.zeros((2, 8, 3), dtype=np.float32),
            np.zeros((2, 8, 4), dtype=np.float32),
            np.zeros((2, 5), dtype=np.float32),
            np.zeros((2, 6), dtype=np.float32),
            np.array([[1] * 8, [1, 1, 1, 0, 0, 0, 0, 0]], dtype=np.float32),
        ]
        labels = {
            "predictions": np.array([0, 1], dtype=np.int32),
            "orientation_aux": np.array([1, 2], dtype=np.int32),
            "phase_aux": np.zeros((2, 8), dtype=np.int32),
        }
        sample_weights = {
            "predictions": np.ones(2, dtype=np.float32),
            "orientation_aux": np.ones(2, dtype=np.float32),
            "phase_aux": inputs[-1],
        }
        history = model.train_on_batch(
            inputs,
            labels,
            sample_weight=sample_weights,
            return_dict=True,
        )
        self.assertIn("loss", history)

    def test_invalid_bert_configuration_is_rejected_before_fit(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            self.build(temporal_architecture="bert", bert_d_model=30, bert_num_heads=4)
        with self.assertRaisesRegex(ValueError, "requires bert_use_cls_token"):
            self.build(temporal_architecture="bert", bert_pooling="cls")
        self.assertEqual(
            self.build(
                temporal_architecture="phase_attention",
                bert_d_model=0,
                bert_num_heads=0,
            ).output_shape,
            (None, 3),
        )
        self.assertEqual(
            self.build(
                temporal_architecture="bert",
                n_phases=0,
                phase_attention_temperature=0.0,
            ).output_shape,
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
