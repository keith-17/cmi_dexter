"""Actual CPU training verifies transfer, masking, baseline and isolation."""
from pathlib import Path
import os
import sys
import unittest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

import numpy as np
import pandas as pd
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from base_utils import SequenceExtractor
from pretraining_utils import AdvancedMultiBranchSequenceClassifier


def toy_recordings(subjects=8):
    rows = []
    for subject in range(subjects):
        for category in range(4):
            target = category < 2
            for counter in range(16):
                phase = counter / 4 + category * 0.4
                rows.append(dict(sequence_id=f"p{subject}-c{category}", subject=f"p{subject}",
                                 sequence_counter=counter, is_target=int(target),
                                 gesture=f"gesture_{category}", bfrb=f"target_{category}" if target else "non_bfrb",
                                 orientation="upright", acc_x=np.sin(phase) + subject * .005,
                                 acc_y=np.cos(phase), acc_z=9.8 + .1 * np.sin(phase),
                                 rot_w=1., rot_x=0., rot_y=0., rot_z=0.))
    return pd.DataFrame(rows)


def tiny_model(mode="supervised", source="non_target", **overrides):
    extractor = SequenceExtractor(
        acc_modes="raw", rotation_modes="quaternion", tof_modes="", thm_modes="",
        stft_configs=None, cwt_configs=None, maxlen=16,
        chunk_window_size=16, chunk_stride=16, padding_value=0.,
        pretraining_source=source, pretraining_label_column="gesture", finetuning_source="target")
    params = dict(extractor=extractor, temporal_architecture="bert", bert_num_layers=1,
                  bert_d_model=8, bert_num_heads=2, bert_ff_dim=16, bert_dropout=0.,
                  bert_attention_dropout=0., bert_pooling="attention", bert_use_cls_token=False,
                  branch_cnn_filters=4, branch_cnn_depth=1, branch_cnn_kernel_size=3,
                  branch_dropout=0., fusion_units=8, fusion_dropout=0.,
                  dense_units=8, dropout=0., conv_dropout=0., static_dropout=0., use_batchnorm=False,
                  learning_rate=.001, epochs=1, batch_size=8, validation_split=.25,
                  validation_group_column="subject", early_stopping_patience=0, use_lr_scheduler=False,
                  pretraining_mode=mode, pretraining_epochs=1, pretraining_batch_size=8,
                  pretraining_validation_split=.25, pretraining_validation_group_column="subject",
                  pretraining_early_stopping_patience=0, pretraining_use_lr_scheduler=False,
                  ssl_span_length=3, ssl_mask_ratio=.3, ssl_decoder_units=8,
                  invalid_candidate_policy="raise", class_weight_mode=None, verbose=0, random_state=7)
    params.update(overrides)
    return AdvancedMultiBranchSequenceClassifier(**params)


class ActualPretrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = toy_recordings()
        cls.labels = cls.frame[["sequence_id", "subject", "is_target", "gesture", "bfrb", "orientation"]].copy()

    def assert_isolated(self, model):
        fitting = set(model.preprocessing_sequence_ids_)
        validation = set(model.validation_sequence_ids_)
        self.assertTrue(fitting.isdisjoint(validation))
        source_validation = set(model.pretraining_validation_sequence_ids_)
        self.assertTrue(fitting.isdisjoint(source_validation))
        self.assertTrue(set(model.pretraining_sequence_ids_).issubset(fitting))

    def test_sklearn_clone_and_nested_parameters(self):
        original = tiny_model()
        copied = clone(original)
        copied.set_params(extractor__pretraining_source="all", pretraining_epochs=2, bert_d_model=16)
        self.assertEqual(copied.extractor.pretraining_source, "all")
        self.assertEqual(copied.bert_d_model, 16)
        self.assertEqual(original.extractor.pretraining_source, "non_target")
        self.assertEqual(copied.get_params()["pretraining_epochs"], 2)

    def test_supervised_pretraining_updates_transferred_backbone(self):
        model = tiny_model().fit(self.frame, self.labels)
        self.assertTrue(model.pretraining_completed_)
        self.assertTrue(model.transferred_changed_layer_names_)
        self.assertEqual(set(model.pretraining_label_encoder_.classes_), {"gesture_2", "gesture_3"})
        self.assertEqual(set(model.classes_), {"target_0", "target_1"})
        self.assert_isolated(model)
        probabilities = model.predict_proba(self.frame.loc[self.frame.is_target == 1])
        self.assertTrue(np.isfinite(np.asarray(probabilities)).all())
        np.testing.assert_allclose(np.asarray(probabilities).sum(axis=1), 1., atol=1e-5)
        self.assertTrue(np.isfinite(model.score(self.frame, self.labels)))

    def test_masked_ssl_learns_without_original_gesture_labels(self):
        frame = self.frame.drop(columns="gesture")
        labels = self.labels.drop(columns="gesture")
        model = tiny_model("masked_ssl", "all").fit(frame, labels)
        self.assertTrue(model.pretraining_completed_)
        self.assertTrue(model.transferred_changed_layer_names_)
        self.assertGreater(model.ssl_first_batch_diagnostics_["hidden_valid_measurements"], 0)
        self.assertFalse(model.ssl_pretrained_pooling_)
        self.assert_isolated(model)
        predictions = model.predict(frame.loc[frame.is_target == 1])
        self.assertEqual(len(predictions), frame.loc[frame.is_target == 1, "sequence_id"].nunique())

    def test_baseline_both_modes_skips_pretraining_and_matches_fine_partition(self):
        fine_ids = []
        for mode in ("supervised", "masked_ssl"):
            model = tiny_model(mode, "baseline").fit(self.frame, self.labels)
            self.assertFalse(model.pretraining_completed_)
            self.assertFalse(len(model.pretraining_sequence_ids_))
            self.assertEqual(model.pretraining_history_, {})
            self.assertEqual(model.transferred_layer_names_, [])
            fine_ids.append(set(model.finetuning_sequence_ids_))
        self.assertEqual(*fine_ids)


if __name__ == "__main__":
    unittest.main()
