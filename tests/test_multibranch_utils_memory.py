"""Regressions for avoiding duplicate multibranch feature tensors."""
import sys
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import multibranch_utils as mba
from base_utils import RandomForestSequenceClassifier


class MultibranchMemoryTests(unittest.TestCase):
    def test_score_handles_is_target_as_primary_label_without_duplicate_columns(self):
        labels = pd.DataFrame({"sequence_id": ["a", "b"], "is_target": [1, 0]})
        predictions = pd.Series([1, 0], index=pd.Index(["a", "b"], name="sequence_id"))
        for model in (mba.MultiBranchSequenceClassifier(primary_target="is_target"),
                      RandomForestSequenceClassifier(primary_target="is_target")):
            model.extractor_ = SimpleNamespace(sequence_col="sequence_id")
            with self.subTest(model=type(model).__name__), patch.object(model, "predict", return_value=predictions):
                self.assertEqual(model.score(None, labels), 1.0)

    def test_static_reduction_masks_padding_and_preserves_float32(self):
        values = np.random.default_rng(42).normal(size=(8, 31, 12)).astype(np.float32)
        original = values.copy()
        mask = np.arange(31)[None, :] < np.array([0, 1, 3, 7, 15, 17, 29, 31])[:, None]
        estimator = mba.MultiBranchSequenceClassifier()
        groups = {"acc": [0], "stft": list(range(1, 8)), "cwt": list(range(8, 12))}
        branches = estimator._split_branches(values, mask, groups)
        np.testing.assert_array_equal(branches["acc"], values[:, :, [0]])
        for branch in ("stft", "cwt"):
            weights = mask[:, :, None].astype(np.float32)
            expected = (values[:, :, groups[branch]] * weights).sum(axis=1)
            expected /= np.maximum(weights.sum(axis=1), 1.0)
            np.testing.assert_array_equal(branches[branch].view(np.uint32), expected.view(np.uint32))
            self.assertEqual(branches[branch].dtype, np.float32)
        np.testing.assert_array_equal(values, original)

    def test_no_validation_or_augmentation_reuses_read_only_input(self):
        frame = pd.DataFrame({"sequence_id": ["a", "a"], "acc_x": [1.0, 2.0]})
        original = frame.copy(deep=True)
        estimator = mba.MultiBranchSequenceClassifier(validation_split=0, augmentor=None)
        estimator.extractor_ = SimpleNamespace(sequence_col="sequence_id")
        train, validation = estimator._split_validation_sequences(frame, None)
        self.assertIs(train, frame)
        self.assertIsNone(validation)
        self.assertIs(estimator._augment_training_sequences(train), frame)
        pd.testing.assert_frame_equal(frame, original)

    def test_predict_releases_combined_tensor_before_runtime_allocation(self):
        refs = []
        mask = np.array([[True, True], [True, False], [True, True]])

        def transform(frame):
            combined = np.arange(12, dtype=np.float32).reshape(3, 2, 2).copy()
            refs.append(weakref.ref(combined))
            return {"X": combined, "mask": mask, "sequence_ids": np.array(["a", "a", "b"])}

        def predict(inputs, verbose):
            self.assertIsNone(refs[0]())
            self.assertEqual(inputs[-1].dtype, np.float32)
            np.testing.assert_array_equal(inputs[-1], mask)
            return np.array([[0.2, 0.8], [0.4, 0.6], [0.9, 0.1]], dtype=np.float32)

        estimator = mba.MultiBranchSequenceClassifier()
        estimator.extractor_ = SimpleNamespace(transform_chunks=transform, sequence_col="sequence_id")
        estimator.branch_groups_ = {"acc": [0], "rotation": [1]}
        estimator.input_order_ = ["acc", "rotation"]
        estimator.model_requires_mask_ = True
        estimator.le_ = LabelEncoder().fit(["left", "right"])
        estimator.model_ = SimpleNamespace(predict=predict)
        probabilities = estimator.predict_proba(pd.DataFrame())
        np.testing.assert_allclose(probabilities.to_numpy(), [[0.3, 0.7], [0.9, 0.1]])
        self.assertEqual(probabilities.index.tolist(), ["a", "b"])

    def test_fit_uses_single_extraction_pass_for_training(self):
        calls = []

        class TrainingExtractor(mba.SequenceExtractor):
            def fit_transform(self, frame, y=None, **fit_params):
                calls.append("fit_transform")
                return {"X": np.ones((1, 2, 1), dtype=np.float32),
                        "mask": np.ones((1, 2), dtype=bool),
                        "sequence_ids": np.array(["a"]), "feature_names": ["acc_x"]}

            def fit(self, frame, y=None):
                raise AssertionError("Training data should only be extracted once")

            def transform_chunks(self, frame):
                raise AssertionError("Training data should only be extracted once")

        estimator = mba.MultiBranchSequenceClassifier(
            extractor=TrainingExtractor(output_format="chunks", chunk_window_size=2),
            validation_split=0, early_stopping_patience=0,
        )
        model = SimpleNamespace(fit=lambda *args, **kwargs: SimpleNamespace(history={"loss": [1.0]}))
        frame = pd.DataFrame({"sequence_id": ["a", "a"], "acc_x": [1.0, 2.0]})
        labels = pd.DataFrame({"sequence_id": ["a"], "bfrb": ["target"]})
        with patch.object(mba, "TENSORFLOW_AVAILABLE", True), \
                patch.object(mba, "build_multibranch_model", return_value=model):
            estimator.fit(frame, labels)
        self.assertEqual(calls, ["fit_transform"])

    def test_fit_preserves_subclass_fit_and_chunk_transform_customization(self):
        calls = []

        class CustomizedExtractor(mba.SequenceExtractor):
            def fit(self, frame, y=None):
                calls.append("fit")
                return self

            def transform_chunks(self, frame):
                calls.append("transform_chunks")
                return {"X": np.ones((1, 2, 1), dtype=np.float32),
                        "mask": np.ones((1, 2), dtype=bool),
                        "sequence_ids": np.array(["a"]), "feature_names": ["acc_x"]}

        estimator = mba.MultiBranchSequenceClassifier(
            extractor=CustomizedExtractor(output_format="chunks", chunk_window_size=2),
            validation_split=0, early_stopping_patience=0,
        )
        model = SimpleNamespace(fit=lambda *args, **kwargs: SimpleNamespace(history={"loss": [1.0]}))
        frame = pd.DataFrame({"sequence_id": ["a", "a"], "acc_x": [1.0, 2.0]})
        labels = pd.DataFrame({"sequence_id": ["a"], "bfrb": ["target"]})
        with patch.object(mba, "TENSORFLOW_AVAILABLE", True), \
                patch.object(mba, "build_multibranch_model", return_value=model):
            estimator.fit(frame, labels)
        self.assertEqual(calls, ["fit", "transform_chunks"])


if __name__ == "__main__":
    unittest.main()
