"""Exercise augmentation with the real gravity removal and feature extractor.

These checks stop before model construction: no neural network is trained.
"""
from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from sklearn.base import clone
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import base_utils as base
import multibranch_utils as mb
import pretraining_utils as pretraining


RAW_IMU = ["acc_x", "acc_y", "acc_z", "rot_w", "rot_x", "rot_y", "rot_z"]
LABEL_COLUMNS = ["sequence_id", "is_target", "gesture", "bfrb", "subject"]


def recordings(lengths=(1, 2, 8, 11), *, id_prefix="sequence"):
    """Use identity, single-component, and changing signed orientations."""
    rows = []
    for number, length in enumerate(lengths):
        for counter in range(length):
            if number % 3 == 0:
                rotation = Rotation.identity()
            elif number % 3 == 1:
                rotation = Rotation.from_quat([1.0, 0.0, 0.0, 0.0])
            else:
                rotation = Rotation.from_rotvec([0.15 * counter, -0.3, 0.2])
            quaternion = rotation.as_quat()[[3, 0, 1, 2]]
            if counter % 2:
                quaternion *= -1.0  # Equivalent signs must remain a valid pair.
            acceleration = rotation.inv().apply([0.6, -0.2, 9.90665])
            rows.append({
                "sequence_id": f"{id_prefix}_{number}", "sequence_counter": counter,
                "subject": f"person_{number}", "is_target": 1,
                "gesture": f"gesture_{number % 2}", "bfrb": f"class_{number % 2}",
                **dict(zip(RAW_IMU, np.r_[acceleration, quaternion])),
                "tof_1_v0": 100.0 + number * 10 + counter,
                "thm_1": 30.0 + number + counter * 0.1,
            })
    frame = pd.DataFrame(rows)
    frame.index = pd.Index(11 + 3 * np.arange(len(frame)), name="original_row")
    return frame


def extractor(**overrides):
    settings = dict(
        acc_modes="raw", linear_acc_mode="baseline", linear_acc_modes="raw",
        use_highpass_fallback=False, rotation_modes="quaternion|rot6d|angular_velocity",
        tof_modes="raw", thm_modes="raw", stft_configs=None, cwt_configs=None,
        chunk_window_size=4, chunk_stride=4, padding_value=0.0, maxlen=0,
    )
    settings.update(overrides)
    return base.SequenceExtractor(**settings)


class AugmentationExtractionIntegrationTests(unittest.TestCase):
    def test_interpolation_and_quaternion_resampling_finish_before_feature_dropout(self):
        frame = recordings((12, 12, 12))
        frame.loc[frame.index[[3, 16, 29]], "acc_x"] = np.nan
        original = frame.copy(deep=True)
        classifier = mb.MultiBranchSequenceClassifier(
            extractor=extractor(resample_modalities=True, linear_acc_modes="",
                                imu_target_sampling_rate=10, rot_target_sampling_rate=10),
            augmentor=base.SensorAugmentor(
                augmentations=["sensor_dropout"], prob=1.0, per_aug_prob=1.0,
                sensor_drop_prob=1.0, seed=4, diagnostics=True,
            ), validation_split=0,
        )
        classifier.extractor_ = clone(classifier.extractor)
        raw_training = classifier._augment_training_sequences(frame)
        quaternions_seen = []
        real_rotation = base.RotationExtractor.transform

        def record_rotation(rotation, processed):
            quaternion = processed[["rot_w", "rot_x", "rot_y", "rot_z"]].to_numpy()
            quaternions_seen.append(quaternion.copy())
            return real_rotation(rotation, processed)

        with patch.object(base.RotationExtractor, "transform", record_rotation):
            chunks = classifier._extract_training_chunks(raw_training)
        self.assertTrue(quaternions_seen)
        for quaternion in quaternions_seen:
            self.assertTrue(np.isfinite(quaternion).all())
            np.testing.assert_allclose(np.linalg.norm(quaternion, axis=1), 1.0, atol=1e-12)
        np.testing.assert_array_equal(chunks["X"], 0.0)
        self.assertTrue(any(report["input_invalid"] for report in classifier.augmentor_.diagnostics_))
        pd.testing.assert_frame_equal(frame, original)

    def test_processed_operations_extract_short_sequences_after_geometry(self):
        frame = recordings()
        original = frame.copy(deep=True)
        expected = extractor().fit_transform(frame)
        for name in (*base.ALL_AUGMENTATION_NAMES, "all"):
            with self.subTest(operation=name):
                classifier = mb.MultiBranchSequenceClassifier(
                    extractor=extractor(), augmentor=base.SensorAugmentor(
                        augmentations=None if name == "all" else [name],
                        prob=1.0, per_aug_prob=1.0, sensor_drop_prob=1.0,
                        channel_drop_prob=1.0, temporal_mask_frac=0.5,
                        crop_frac_range=(0.5, 0.8), seed=19,
                    ), validation_split=0,
                )
                classifier.extractor_ = clone(classifier.extractor)
                raw_training = classifier._augment_training_sequences(frame)
                pd.testing.assert_frame_equal(raw_training, original)
                chunks = classifier._extract_training_chunks(raw_training)
                self.assertTrue(np.isfinite(chunks["X"]).all())
                self.assertEqual(chunks["X"].shape, expected["X"].shape)
                np.testing.assert_array_equal(chunks["mask"], expected["mask"])
                np.testing.assert_array_equal(chunks["sequence_ids"], expected["sequence_ids"])
                self.assertFalse(hasattr(classifier.extractor_, "_training_feature_augmentor"))
                pd.testing.assert_frame_equal(frame, original)

    def test_every_generic_operation_preserves_gravity_inputs_and_extracts(self):
        frame = recordings()
        original = frame.copy(deep=True)
        baseline = base.SignalCleaner(linear_acc_mode="baseline").fit_transform(frame)
        expected_linear = baseline[["lin_acc_x", "lin_acc_y", "lin_acc_z"]]
        np.testing.assert_allclose(expected_linear, np.tile([0.6, -0.2, 0.1], (len(frame), 1)), atol=1e-12)
        for name in (*base.ALL_AUGMENTATION_NAMES, "all"):
            with self.subTest(operation=name):
                augmentor = base.SensorAugmentor(
                    augmentations=None if name == "all" else [name],
                    prob=1.0, per_aug_prob=1.0, sensor_drop_prob=1.0,
                    channel_drop_prob=1.0, temporal_mask_frac=0.5,
                    temporal_num_masks=1, crop_frac_range=(0.5, 0.8), seed=19,
                )
                augmented = augmentor.fit_transform(frame)
                pd.testing.assert_frame_equal(augmented[RAW_IMU], original[RAW_IMU])
                pd.testing.assert_frame_equal(augmented[LABEL_COLUMNS + ["sequence_counter"]],
                                              original[LABEL_COLUMNS + ["sequence_counter"]])
                self.assertEqual(augmented.shape, frame.shape)
                self.assertTrue(augmented.index.equals(frame.index))
                cleaned = base.SignalCleaner(linear_acc_mode="baseline").fit_transform(augmented)
                pd.testing.assert_frame_equal(cleaned[expected_linear.columns], expected_linear)
                features = extractor().fit_transform(augmented)
                self.assertTrue(np.isfinite(features["X"]).all())
                self.assertEqual(set(features["sequence_ids"]), set(frame.sequence_id))
                self.assertEqual(features["X"].shape[:2], features["mask"].shape)
                pd.testing.assert_frame_equal(frame, original)

    def test_dropout_changes_available_non_imu_modalities(self):
        frame = recordings()
        for operation in ("sensor_dropout", "channel_dropout", "temporal_mask"):
            with self.subTest(operation=operation):
                output = base.SensorAugmentor(
                    augmentations=[operation], prob=1.0, per_aug_prob=1.0,
                    sensor_drop_prob=1.0, channel_drop_prob=1.0,
                    temporal_mask_frac=1.0, temporal_num_masks=1, seed=7,
                ).fit_transform(frame)
                np.testing.assert_array_equal(output[["tof_1_v0", "thm_1"]], 0.0)
                pd.testing.assert_frame_equal(output[RAW_IMU], frame[RAW_IMU])

    def test_same_seed_reproduces_features_and_clones_existing_nested_parameters(self):
        frame = recordings()
        augmentor = base.SensorAugmentor(
            augmentations=["jitter", "crop_resize", "temporal_mask"],
            prob=1.0, per_aug_prob=1.0, crop_frac_range=(0.6, 0.9), seed=31,
        )
        first = clone(augmentor).fit_transform(frame)
        second = clone(augmentor).fit_transform(frame)
        pd.testing.assert_frame_equal(first, second)
        np.testing.assert_array_equal(extractor().fit_transform(first)["X"],
                                      extractor().fit_transform(second)["X"])
        pipeline = Pipeline([("estimator", mb.MultiBranchSequenceClassifier(
            extractor=extractor(), augmentor=augmentor,
        ))])
        parameters = {
            "estimator__augmentor__prob": 0.5,
            "estimator__augmentor__per_aug_prob": 1.0,
            "estimator__augmentor__sensor_drop_prob": 0.2,
            "estimator__augmentor__channel_drop_prob": 0.3,
            "estimator__augmentor__temporal_num_masks": 0,
            "estimator__augmentor__temporal_mask_frac": 0.05,
            "estimator__augmentor__jitter_sigma": 0.02,
            "estimator__augmentor__noise_std": 0.0,
            "estimator__augmentor__scaling_sigma": 0.01,
            "estimator__augmentor__time_shift_frac": 0.05,
            "estimator__augmentor__crop_frac_range": (0.7, 0.95),
            "estimator__augmentor__warp_sigma": 0.0,
        }
        copied = clone(pipeline).set_params(**parameters)
        self.assertEqual({name: copied.get_params()[name] for name in parameters}, parameters)
        self.assertIsNot(copied.named_steps["estimator"].augmentor, augmentor)


class AugmentationPartitionIntegrationTests(unittest.TestCase):
    def test_feature_hook_cleanup_and_restoration_when_augmentation_raises(self):
        frame = recordings()
        classifier = mb.MultiBranchSequenceClassifier(
            extractor=extractor(), augmentor=base.SensorAugmentor(
                augmentations=["jitter"], prob=1.0, per_aug_prob=1.0,
            ), validation_split=0,
        )
        classifier.extractor_ = clone(classifier.extractor)
        raw_training = classifier._augment_training_sequences(frame)
        with patch.object(classifier.augmentor_, "transform_features", side_effect=ValueError("hook sentinel")):
            with self.assertRaisesRegex(base.InvalidExtractorParams, "hook sentinel"):
                classifier._extract_training_chunks(raw_training)
        self.assertFalse(hasattr(classifier.extractor_, "_training_feature_augmentor"))
        # Nested calls must restore a previous hook, including on exceptions.
        previous = object()
        classifier.extractor_._training_feature_augmentor = previous
        with patch.object(classifier.augmentor_, "transform_features", side_effect=ValueError("hook sentinel")):
            with self.assertRaisesRegex(base.InvalidExtractorParams, "hook sentinel"):
                classifier._extract_training_chunks(raw_training)
        self.assertIs(classifier.extractor_._training_feature_augmentor, previous)
        del classifier.extractor_._training_feature_augmentor
        expected = extractor().fit_transform(frame)
        actual = classifier.extractor_.transform_chunks(frame)
        np.testing.assert_array_equal(actual["X"], expected["X"])

    def test_actual_classifier_path_augments_processed_training_features_only(self):
        frame = recordings((6,) * 12)
        original = frame.copy(deep=True)
        labels = frame[LABEL_COLUMNS].copy()
        classifier = mb.MultiBranchSequenceClassifier(
            extractor=extractor(), augmentor=base.SensorAugmentor(
                augmentations=["sensor_dropout"], prob=1.0, per_aug_prob=1.0,
                sensor_drop_prob=1.0, seed=4,
            ), validation_split=0.25, class_weight_mode=None, random_state=5,
        )
        seen = []
        training_features = []
        real_clean = base.SignalCleaner.transform
        real_extract = classifier._extract_training_chunks

        def record_clean(cleaner, partition):
            seen.append(partition.copy(deep=True))
            return real_clean(cleaner, partition)

        def record_extract(partition):
            chunks = real_extract(partition)
            training_features.append((chunks["X"].copy(), list(chunks["feature_names"])))
            return chunks

        # Model construction is the first step after all extraction and target
        # alignment. Stopping here proves this check cannot start training.
        class ExtractionComplete(Exception):
            pass

        with patch.object(mb, "TENSORFLOW_AVAILABLE", True), patch.object(mb, "tf", None), \
                patch.object(base.SignalCleaner, "transform", record_clean), \
                patch.object(classifier, "_extract_training_chunks", side_effect=record_extract), \
                patch.object(classifier, "_build_model", side_effect=ExtractionComplete), \
                redirect_stdout(StringIO()):
            with self.assertRaises(ExtractionComplete):
                classifier._fit_impl(frame, labels)
        self.assertEqual(len(seen), 2)
        training, validation = seen
        self.assertEqual(set(training.sequence_id), set(classifier.train_sequence_ids_))
        self.assertEqual(set(validation.sequence_id), set(classifier.validation_sequence_ids_))
        self.assertTrue(set(training.sequence_id).isdisjoint(validation.sequence_id))
        pd.testing.assert_frame_equal(training, frame.loc[training.index])
        pd.testing.assert_frame_equal(validation, frame.loc[validation.index])
        self.assertEqual(len(training_features), 1)
        tensor, names = training_features[0]
        imu_indices = [index for index, name in enumerate(names)
                       if name.startswith(("acc_", "lin_acc_", "rot_", "rot6d_", "ang_vel_"))]
        self.assertTrue(imu_indices)
        np.testing.assert_array_equal(tensor[:, :, imu_indices], 0.0)
        self.assertFalse(hasattr(classifier.extractor_, "_training_feature_augmentor"))
        holdout = recordings((3, 5), id_prefix="holdout")
        expected = extractor().fit(frame).transform_chunks(holdout)
        actual = classifier.extractor_.transform_chunks(holdout)
        np.testing.assert_array_equal(actual["X"], expected["X"])
        self.assertEqual(set(actual["sequence_ids"]), set(holdout.sequence_id))
        pd.testing.assert_frame_equal(frame, original)

    def _lazy_ssl_owner(self, source_augmentor):
        frame = recordings((8,) * 12)
        classifier = pretraining.AdvancedMultiBranchSequenceClassifier(
            extractor=extractor(pretraining_source="all"),
            augmentor=base.SensorAugmentor(
                augmentations=["sensor_dropout"], prob=1.0, per_aug_prob=1.0,
                sensor_drop_prob=1.0, seed=8,
            ), pretraining_augmentor=source_augmentor, pretraining_mode="masked_ssl",
            ssl_mask_columns=RAW_IMU + ["tof_1_v0", "thm_1"],
            ssl_standardize_targets=False, ssl_span_length=2, ssl_mask_ratio=0.25,
            validation_split=0.25, pretraining_validation_split=0.25,
            class_weight_mode=None, random_state=5, pretraining_shuffle=False,
        )
        classifier.extractor_ = clone(classifier.extractor)
        training, _ = classifier._split_validation_sequences(frame, frame[LABEL_COLUMNS])
        training = classifier._augment_training_sequences(training)
        chunks = classifier._extract_training_chunks(training)
        classifier.branch_groups_ = classifier._branch_groups(chunks["feature_names"])
        classifier.input_order_ = list(classifier.branch_groups_)
        classifier.model_requires_mask_ = False
        classifier._prepare_ssl_targets()
        return classifier, frame

    def test_lazy_ssl_no_source_augmentor_does_not_inherit_finetuning_dropout(self):
        classifier, frame = self._lazy_ssl_owner(None)
        dataset = pretraining.MaskedSensorSequence(classifier, classifier._pretraining_X)
        sequence = next(iter(dataset.positions))
        first_inputs, first_targets = dataset._sequence(sequence)
        self.assertTrue(any(np.any(array != 0) for array in first_inputs))
        # Removing the fitted fine-tuning augmentor cannot change source batches.
        classifier.augmentor_ = None
        repeated = pretraining.MaskedSensorSequence(classifier, classifier._pretraining_X)
        second_inputs, second_targets = repeated._sequence(sequence)
        for first, second in zip(first_inputs, second_inputs):
            np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first_targets, second_targets)
        self.assertFalse(hasattr(classifier.extractor_, "_training_feature_augmentor"))
        pd.testing.assert_frame_equal(classifier._pretraining_X,
                                      frame.loc[classifier._pretraining_X.index])

    def test_lazy_ssl_dropout_changes_inputs_and_preserves_raw_targets_and_validation(self):
        source_augmentor = base.SensorAugmentor(
            augmentations=["sensor_dropout"], prob=1.0, per_aug_prob=1.0,
            sensor_drop_prob=1.0, seed=11,
        )
        classifier, frame = self._lazy_ssl_owner(source_augmentor)
        training = pretraining.MaskedSensorSequence(classifier, classifier._pretraining_X)
        sequence = next(iter(training.positions))
        augmented_inputs, augmented_targets = training._sequence(sequence)
        for array in augmented_inputs:
            np.testing.assert_array_equal(array, 0.0)
        reserved = classifier._pretraining_validation_X
        validation = pretraining.MaskedSensorSequence(classifier, reserved, validation=True)
        validation_sequence = next(iter(validation.positions))
        validation_inputs, validation_targets = validation._sequence(validation_sequence)
        self.assertTrue(any(np.any(array != 0) for array in validation_inputs))
        classifier.pretraining_augmentor_ = None
        clean_training = pretraining.MaskedSensorSequence(classifier, classifier._pretraining_X)
        _, clean_targets = clean_training._sequence(sequence)
        np.testing.assert_array_equal(augmented_targets, clean_targets)
        clean_validation = pretraining.MaskedSensorSequence(classifier, reserved, validation=True)
        clean_inputs, clean_validation_targets = clean_validation._sequence(validation_sequence)
        for first, second in zip(validation_inputs, clean_inputs):
            np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(validation_targets, clean_validation_targets)
        self.assertGreater(np.sum(augmented_targets[..., len(classifier.ssl_columns_):]), 0)
        self.assertFalse(hasattr(classifier.extractor_, "_training_feature_augmentor"))
        pd.testing.assert_frame_equal(classifier._pretraining_X,
                                      frame.loc[classifier._pretraining_X.index])
        pd.testing.assert_frame_equal(reserved, frame.loc[reserved.index])

    def test_lazy_ssl_exception_cleans_hook_and_restores_previous_stage(self):
        classifier, _ = self._lazy_ssl_owner(base.SensorAugmentor(
            augmentations=["jitter"], prob=1.0, per_aug_prob=1.0, seed=2,
        ))
        classifier._fit_stage_ = "training"
        dataset = pretraining.MaskedSensorSequence(classifier, classifier._pretraining_X)
        sequence = next(iter(dataset.positions))
        with patch.object(classifier.pretraining_augmentor_, "transform_features",
                          side_effect=ValueError("lazy augmentation sentinel")):
            with self.assertRaisesRegex(ValueError, "lazy augmentation sentinel") as caught:
                dataset._sequence(sequence)
        self.assertEqual(caught.exception.failure_stage, "augmentation")
        self.assertEqual(classifier._fit_stage_, "training")
        self.assertFalse(hasattr(classifier.extractor_, "_training_feature_augmentor"))
        self.assertFalse(dataset._cache)

    def test_masked_ssl_rejects_active_temporal_reindexing_and_accepts_noops(self):
        for operation, parameters in (
            ("time_shift", {"time_shift_frac": 0.5}),
            ("crop_resize", {"crop_frac_range": (0.5, 0.8)}),
        ):
            with self.subTest(active_operation=operation):
                augmentor = base.SensorAugmentor(
                    augmentations=[operation], prob=1.0, per_aug_prob=1.0, **parameters,
                )
                with self.assertRaisesRegex(base.InvalidExtractorParams, "targets and masks must remain aligned") as caught:
                    self._lazy_ssl_owner(augmentor)
                self.assertEqual(caught.exception.failure_stage, "augmentation")
        for parameters in (
            {"time_shift_frac": 0.0, "crop_frac_range": (1.0, 1.0)},
            {"prob": 0.0},
            {"per_aug_prob": 0.0},
        ):
            with self.subTest(disabled=parameters):
                settings = dict(augmentations=["time_shift", "crop_resize"], prob=1.0, per_aug_prob=1.0)
                settings.update(parameters)
                classifier, _ = self._lazy_ssl_owner(base.SensorAugmentor(**settings))
                dataset = pretraining.MaskedSensorSequence(classifier, classifier._pretraining_X)
                inputs, targets = dataset._sequence(next(iter(dataset.positions)))
                self.assertTrue(all(np.isfinite(array).all() for array in inputs))
                self.assertTrue(np.isfinite(targets).all())

    def test_pretraining_augmentation_keeps_reserved_pool_unchanged(self):
        frame = recordings((6,) * 16)
        original = frame.copy(deep=True)
        classifier = pretraining.AdvancedMultiBranchSequenceClassifier(
            extractor=extractor(pretraining_source="all"),
            augmentor=base.SensorAugmentor(
                augmentations=["channel_dropout"], prob=1.0, per_aug_prob=1.0,
                channel_drop_prob=1.0, seed=2,
            ),
            pretraining_augmentor=base.SensorAugmentor(
                augmentations=["temporal_mask"], prob=1.0, per_aug_prob=1.0,
                temporal_mask_frac=1.0, temporal_num_masks=1, seed=3,
            ), validation_split=0.25, pretraining_validation_split=0.25,
            class_weight_mode=None, random_state=5,
        )
        classifier.extractor_ = clone(classifier.extractor)
        labels = frame[LABEL_COLUMNS].copy()
        training, validation = classifier._split_validation_sequences(frame, labels)
        training = classifier._augment_training_sequences(training)
        chunks = classifier._extract_training_chunks(training)
        reserved = classifier._pretraining_validation_X
        self.assertTrue(set(chunks["sequence_ids"]).isdisjoint(reserved.sequence_id))
        self.assertTrue(set(classifier._pretraining_X.sequence_id).isdisjoint(reserved.sequence_id))
        pd.testing.assert_frame_equal(reserved, frame.loc[reserved.index])
        pd.testing.assert_frame_equal(validation, frame.loc[validation.index])
        pd.testing.assert_frame_equal(classifier._pretraining_X,
                                      frame.loc[classifier._pretraining_X.index])
        feature_indices = [index for index, name in enumerate(chunks["feature_names"])
                           if name.startswith(("acc_", "lin_acc_", "rot_", "rot6d_", "ang_vel_"))]
        np.testing.assert_array_equal(chunks["X"][:, :, feature_indices], 0.0)
        source_chunks = classifier._extract_pretraining_chunks()
        np.testing.assert_array_equal(source_chunks["X"], 0.0)
        validation_chunks = classifier.extractor_.transform_chunks(reserved)
        self.assertTrue(np.isfinite(source_chunks["X"]).all())
        self.assertTrue(np.isfinite(validation_chunks["X"]).all())
        pd.testing.assert_frame_equal(frame, original)


if __name__ == "__main__":
    unittest.main()
