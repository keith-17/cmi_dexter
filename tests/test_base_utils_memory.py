"""Regression coverage for bounded preprocessing and unchanged feature values."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import weakref

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import base_utils as base


def sensor_frame():
    return pd.DataFrame({
        "sequence_id": ["b", "a"] * 8,
        "sequence_counter": np.repeat(np.arange(8), 2),
        "acc_x": np.linspace(-1, 2, 16),
        "acc_y": np.linspace(2, -1, 16),
        "acc_z": np.ones(16),
        "rot_w": np.ones(16),
        "rot_x": np.zeros(16),
        "rot_y": np.zeros(16),
        "rot_z": np.zeros(16),
        "tof_1_v0": np.arange(16, dtype=float),
        "tof_1_v1": [-1.0, 4.0] * 8,
        "thm_1": np.arange(16, dtype=float),
    }, index=np.arange(16) * 3)


def extractor(**options):
    return base.SequenceExtractor(
        stft_configs=[{"nperseg": 4, "noverlap": 2}],
        cwt_configs=[{"widths": [2.0, 3.0, 4.0]}],
        chunk_window_size=4, chunk_stride=3, maxlen=7,
        **options,
    )


class PreprocessingMemoryTests(unittest.TestCase):
    def test_augmentation_leaves_numeric_labels_untouched_and_skips_disabled_copy(self):
        source = sensor_frame().assign(is_target=1, auxiliary_score=0.5)
        original = source.copy(deep=True)
        disabled = base.SensorAugmentor(prob=0.0).fit(source)
        self.assertIs(disabled.transform(source), source)
        active = base.SensorAugmentor(
            augmentations=["scaling"], prob=1.0, per_aug_prob=1.0, scaling_sigma=0.5,
        ).fit_transform(source)
        pd.testing.assert_frame_equal(active[["is_target", "auxiliary_score"]],
                                      source[["is_target", "auxiliary_score"]])
        self.assertFalse(active["acc_x"].equals(source["acc_x"]))
        pd.testing.assert_frame_equal(source, original)

    def test_fit_transform_reuses_training_features_and_retains_no_frame_cache(self):
        source = sensor_frame()
        for output_format in ("frame", "chunks"):
            for context in (False, True):
                with self.subTest(output_format=output_format, context=context):
                    model = extractor(output_format=output_format, add_global_context=context)
                    reference = clone(model).fit(source).transform(source)
                    feature_refs = []
                    combine = model._combine_feature_outputs

                    def capture(*args, **kwargs):
                        result = combine(*args, **kwargs)
                        feature_refs.append(weakref.ref(result))
                        return result

                    with patch.object(model, "_combine_feature_outputs", side_effect=capture):
                        actual = model.fit_transform(source)
                    self.assertEqual(len(feature_refs), 1)
                    self.assertIsNone(feature_refs[0]())
                    if output_format == "frame":
                        pd.testing.assert_frame_equal(actual, reference)
                    else:
                        for key in ("X", "mask", "sequence_ids"):
                            np.testing.assert_array_equal(actual[key], reference[key])
                        self.assertEqual(actual["feature_names"], reference["feature_names"])

    def test_fit_derives_context_schema_without_building_context_values(self):
        model = extractor(add_global_context=True)
        with patch.object(model, "_append_global_context", side_effect=AssertionError("unnecessary context")):
            model.fit(sensor_frame())
        self.assertIn("acc_x_raw_global_std", model.feature_names_in_)

    def test_read_only_extractors_do_not_copy_the_wide_input(self):
        source = sensor_frame()
        original = source.copy(deep=True)
        transforms = [
            base.IMUExtractor(acc_modes="raw|velocity|displacement|jerk"),
            base.RotationExtractor(rotation_modes="quaternion|angular_velocity"),
            base.STFTExtractor(nperseg=4, noverlap=2),
            base.CWTExtractor(widths=[2.0, 3.0, 4.0]),
        ]
        original_copy = pd.DataFrame.copy

        def guarded_copy(frame, *args, **kwargs):
            if frame is source:
                raise AssertionError("A read-only transform copied every sensor column")
            return original_copy(frame, *args, **kwargs)

        for transform in transforms:
            transform.fit(source)
            with self.subTest(transform=type(transform).__name__):
                with patch.object(pd.DataFrame, "copy", new=guarded_copy):
                    transformed = transform.transform(source)
                self.assertFalse(transformed.empty)
                pd.testing.assert_frame_equal(source, original)

    def test_context_preserves_row_order_sample_std_and_feature_order(self):
        frame = pd.DataFrame({"sequence_id": ["b", "a", "b", "c"],
                              "x": [1.0, 4.0, 3.0, 7.0], "y": [8.0, 3.0, 4.0, 9.0]},
                             index=[10, 10, 40, 80])
        expected = frame.copy()
        for column in ("y", "x"):
            expected[f"{column}_global_mean"] = frame.groupby("sequence_id")[column].transform("mean")
            expected[f"{column}_global_std"] = frame.groupby("sequence_id")[column].transform("std").fillna(0)
        actual = base.SequenceExtractor()._append_global_context(frame, ["y", "x"])
        pd.testing.assert_frame_equal(actual, expected)

    def test_ffill_is_sequence_local_with_interleaved_rows(self):
        frame = pd.DataFrame({"sequence_id": ["a", "b", "a", "b"],
                              "sequence_counter": [0, 0, 1, 1],
                              "acc_x": [np.nan, np.nan, np.nan, 5.0]}, index=[3, 6, 9, 12])
        original = frame.copy()
        clean = base.SignalCleaner(interp_mode="ffill").fit_transform(frame)
        np.testing.assert_array_equal(clean["acc_x"], [np.nan, 5.0, np.nan, 5.0])
        pd.testing.assert_frame_equal(frame, original)

    def test_random_forest_extracts_training_features_once(self):
        source = sensor_frame()
        calls = []
        original_combine = base.SequenceExtractor._combine_feature_outputs

        def capture(model, *args, **kwargs):
            calls.append(1)
            return original_combine(model, *args, **kwargs)

        model = base.RandomForestSequenceClassifier(
            extractor=extractor(output_format="frame"),
            estimator=RandomForestClassifier(n_estimators=2, random_state=42),
        )
        labels = pd.DataFrame({"sequence_id": ["a", "b"], "bfrb": ["target", "non_bfrb"]})
        with patch.object(base.SequenceExtractor, "_combine_feature_outputs", new=capture):
            model.fit(source, labels)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(model.predict(source)), 2)

    def test_random_forest_preserves_custom_extractor_fit_contract(self):
        class CustomExtractor(base.SequenceExtractor):
            def fit_transform(self, X, y=None, **fit_params):
                raise AssertionError("The custom extractor requires separate fit and transform")

        source = sensor_frame()
        model = base.RandomForestSequenceClassifier(
            extractor=CustomExtractor(stft_configs=None, cwt_configs=None, output_format="frame"),
            estimator=RandomForestClassifier(n_estimators=2, random_state=42),
        )
        labels = pd.DataFrame({"sequence_id": ["a", "b"], "bfrb": ["target", "non_bfrb"]})
        model.fit(source, labels)
        self.assertEqual(len(model.predict(source)), 2)


class BalancedSamplingTests(unittest.TestCase):
    def test_split_keeps_whole_sequences_and_exact_disjoint_totals(self):
        metadata = pd.DataFrame({"sequence_id": np.arange(20), "subject": ["s1", "s2"] * 10,
                                 "gesture": ["g1"] * 10 + ["g2"] * 10})
        frame = metadata.loc[metadata.index.repeat(3)].assign(acc_x=np.arange(60))
        with redirect_stdout(StringIO()):
            train, test = base.sample_balanced_split(frame, train_pct=0.6, test_pct=0.4)
        self.assertEqual(train["sequence_id"].nunique(), 12)
        self.assertEqual(test["sequence_id"].nunique(), 8)
        self.assertFalse(set(train["sequence_id"]) & set(test["sequence_id"]))
        self.assertTrue(train.groupby("sequence_id").size().eq(3).all())
        self.assertEqual(len(train) + len(test), len(frame))


if __name__ == "__main__":
    unittest.main()
