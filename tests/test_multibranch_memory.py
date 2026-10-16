"""Keep search memory bounded without changing extracted model inputs."""
import sys
import unittest
import weakref
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import base_utils as base
import multibranch_utils as mba


class ChunkMemoryTests(unittest.TestCase):
    def test_float32_windows_match_previous_float64_then_cast_values(self):
        values = np.array([
            [1.2345678912345, np.nan], [np.inf, -np.inf],
            [0.10000000149, 1.000000059604645], [3.14159265359, -2.71828182846],
            [5.5, -5.5], [6.6, -6.6], [7.7, -7.7], [8.8, -8.8],
        ], dtype=np.float64)
        short = np.array([[0.333333333333, 1.234567890123], [2.2, -2.2]])
        frame = pd.DataFrame(np.concatenate([values, short]), columns=["f1", "f2"])
        frame["sequence_id"] = ["long"] * 8 + ["short"] * 2
        cases = [
            ({"final_window_mode": "pad"}, values, [0, 2, 4, 6]),
            ({"final_window_mode": "overlap"}, values, [0, 2, 4, 5]),
            ({"final_window_mode": "overlap", "max_windows_per_sequence": 2,
              "window_cap_policy": "first"}, values, [0, 2]),
            ({"final_window_mode": "overlap", "max_windows_per_sequence": 2,
              "window_cap_policy": "last"}, values, [4, 5]),
            ({"final_window_mode": "overlap", "max_windows_per_sequence": 2,
              "window_cap_policy": "uniform"}, values, [0, 5]),
            ({"final_window_mode": "overlap", "maxlen": 5,
              "sequence_crop_mode": "tail"}, values[-5:], [0, 2]),
        ]
        for options, long_values, starts in cases:
            with self.subTest(options=options):
                extractor = base.SequenceExtractor(
                    output_format="chunks", chunk_window_size=3, chunk_stride=2,
                    padding_value=0.1234567890123, **options,
                )
                extractor.feature_names_in_ = ["f1", "f2"]
                extractor._preprocess_features = lambda data: data
                expected, masks = [], []
                for sequence, offsets in ((long_values, starts), (short, [0])):
                    clean = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)
                    for start in offsets:
                        window = clean[start:start + 3]
                        valid = len(window)
                        expected.append(np.vstack([
                            window, np.full((3 - valid, 2), extractor.padding_value),
                        ]))
                        masks.append([True] * valid + [False] * (3 - valid))
                expected = np.stack(expected).astype(np.float32)
                original_stack = np.stack
                stacked_dtypes = []
                def record_stack(arrays, *args, **kwargs):
                    stacked_dtypes.append({array.dtype for array in arrays})
                    return original_stack(arrays, *args, **kwargs)
                with patch.object(base.np, "stack", side_effect=record_stack):
                    chunks = extractor.transform_chunks(frame)
                self.assertEqual(stacked_dtypes[0], {np.dtype("float32")})
                self.assertEqual(chunks["X"].dtype, np.float32)
                np.testing.assert_array_equal(chunks["X"].view(np.uint32), expected.view(np.uint32))
                np.testing.assert_array_equal(chunks["mask"], masks)
                self.assertEqual(chunks["mask"].dtype, np.bool_)
                self.assertEqual(chunks["sequence_ids"].tolist(), ["long"] * len(starts) + ["short"])
                self.assertEqual(chunks["feature_names"], ["f1", "f2"])

    def test_phase_provider_reads_combined_features_before_they_are_released(self):
        extractor = base.SequenceExtractor(output_format="chunks", chunk_window_size=2, chunk_stride=2)
        tensor_refs, provider_calls = [], []
        def fit(extractor_self, frame):
            extractor_self.feature_names_in_ = ["acc_x"]
            return extractor_self
        def transform(extractor_self, frame):
            ids = frame["sequence_id"].drop_duplicates().to_numpy()
            tensor = np.ones((len(ids), 2, 1), dtype=np.float32)
            tensor_refs.append(weakref.ref(tensor))
            return {"X": tensor, "mask": np.ones((len(ids), 2), dtype=bool),
                    "sequence_ids": ids, "feature_names": ["acc_x"]}
        extractor.fit = MethodType(fit, extractor)
        extractor.transform_chunks = MethodType(transform, extractor)
        def provider(frame, chunks):
            np.testing.assert_array_equal(chunks["X"], 1.0)
            provider_calls.append(set(frame["sequence_id"]))
            return np.zeros(chunks["X"].shape[:2], dtype=np.int32)
        def build(**kwargs):
            self.assertEqual(len(provider_calls), 2)
            self.assertTrue(all(ref() is None for ref in tensor_refs))
            return SimpleNamespace(fit=lambda *args, **kwargs: SimpleNamespace(history={"loss": [1.0]}))
        estimator = mba.MultiBranchSequenceClassifier(
            extractor=extractor, temporal_architecture="bert", use_phase_aux_head=True,
            phase_label_provider=provider, validation_split=0.25, early_stopping_patience=0,
        )
        frame = pd.DataFrame({"sequence_id": np.repeat(["a", "b", "c", "d"], 4),
                              "sequence_counter": list(range(4)) * 4, "acc_x": np.ones(16)})
        labels = pd.DataFrame({"sequence_id": ["a", "b", "c", "d"], "bfrb": ["target"] * 4})
        with patch.object(mba, "TENSORFLOW_AVAILABLE", True), \
             patch.object(mba, "clone", side_effect=lambda value: value), \
             patch.object(mba, "build_multibranch_model", side_effect=build):
            estimator.fit(frame, labels)
        self.assertFalse(provider_calls[0] & provider_calls[1])
        self.assertEqual(estimator.history_, {"loss": [1.0]})


if __name__ == "__main__":
    unittest.main()
