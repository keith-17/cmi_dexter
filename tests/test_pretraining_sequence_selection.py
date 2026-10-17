"""Whole-sequence, metadata-only selection for supervised/SSL transfer."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from base_utils import InvalidExtractorParams, SequenceExtractor


def recordings():
    rows = []
    for sequence, target, gesture, subject, length in (
        ("a", 1, "scratch", "p1", 4),
        ("b", 0, "wave", "p1", 5),
        ("c", 0, "wave", "p2", 6),
        ("d", 0, "glasses", "p2", 7),
        ("e", 1, "pull", "p3", 8),
        ("f", 0, "wave", "p3", 9),
    ):
        for counter in range(length):
            rows.append(dict(sequence_id=sequence, sequence_counter=counter,
                             is_target=target, gesture=gesture,
                             bfrb=gesture if target else "non_bfrb", subject=subject,
                             acc_x=float(counter), acc_y=0.2, acc_z=9.8,
                             rot_w=1., rot_x=0., rot_y=0., rot_z=0.))
    return pd.DataFrame(rows)


class StageSelectionTests(unittest.TestCase):
    def setUp(self):
        self.frame = recordings()

    def test_sources_select_complete_sequences_and_leave_original_unchanged(self):
        original = self.frame.copy(deep=True)
        extractor = SequenceExtractor()
        pretrain = extractor.select_pretraining_sequences(self.frame)
        finetune = extractor.select_finetuning_sequences(self.frame)
        self.assertEqual(set(pretrain.sequence_id), {"b", "c", "d", "f"})
        self.assertEqual(set(finetune.sequence_id), {"a", "e"})
        self.assertFalse(set(pretrain.sequence_id) & set(finetune.sequence_id))
        self.assertEqual(pretrain.groupby("sequence_id").size().to_dict(), {"b": 5, "c": 6, "d": 7, "f": 9})
        pd.testing.assert_frame_equal(self.frame, original)

    def test_class_and_metadata_filters_compose_before_sampling(self):
        extractor = SequenceExtractor(
            pretraining_source="all", pretraining_include_classes=["wave", "scratch"],
            pretraining_filters={"subject": ["p1", "p2"]},
            pretraining_exclude_filters={"subject": ["p1"]},
            pretraining_min_length=5, pretraining_max_length=7)
        selected = extractor.select_pretraining_sequences(self.frame)
        self.assertEqual(set(selected.sequence_id), {"c"})

    def test_selection_does_not_resurrect_held_out_recordings(self):
        train = self.frame.loc[self.frame.subject != "p3"]
        selected = SequenceExtractor(pretraining_source="all").select_pretraining_sequences(train)
        self.assertNotIn("p3", selected.subject.values)
        self.assertEqual(set(selected.sequence_id), {"a", "b", "c", "d"})

    def test_label_table_may_supply_metadata_without_sensor_copy(self):
        metadata = self.frame[["sequence_id", "is_target", "gesture", "subject"]].drop_duplicates()
        sensors = self.frame.drop(columns=["is_target", "gesture", "subject"])
        selected = SequenceExtractor(pretraining_include_classes=["glasses"]).select_pretraining_sequences(sensors, metadata)
        self.assertEqual(set(selected.sequence_id), {"d"})

    def test_ssl_all_source_does_not_require_gesture_labels(self):
        sensors = self.frame.drop(columns=["is_target", "gesture", "bfrb"])
        selected = SequenceExtractor(pretraining_source="all").select_pretraining_sequences(sensors)
        self.assertEqual(len(selected), len(sensors))

    def test_baseline_skips_pretraining_without_requiring_source_labels(self):
        sensors = self.frame.drop(columns=["is_target", "gesture", "bfrb"])
        extractor = SequenceExtractor(pretraining_source="baseline")
        self.assertTrue(extractor.select_pretraining_sequences(sensors).empty)
        self.assertEqual(extractor.pretraining_selection_summary_["selected_sequences"], 0)

    def test_missing_or_inconsistent_required_metadata_raises(self):
        extractor = SequenceExtractor(pretraining_include_classes=["wave"])
        with self.assertRaisesRegex(InvalidExtractorParams, "gesture"):
            extractor.select_pretraining_sequences(self.frame.drop(columns="gesture"))
        broken = self.frame.copy()
        broken.loc[broken.index[0], "gesture"] = "other"
        with self.assertRaisesRegex(InvalidExtractorParams, "constant"):
            extractor.select_pretraining_sequences(broken)

    def test_sampling_is_reproducible_whole_sequence_and_balances_originals(self):
        extractor = SequenceExtractor(pretraining_balance_classes=True, pretraining_random_state=9)
        first = extractor.select_pretraining_sequences(self.frame)
        second = extractor.select_pretraining_sequences(self.frame)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first.drop_duplicates("sequence_id").gesture.value_counts().to_dict(), {"wave": 1, "glasses": 1})
        self.assertEqual(first.sequence_counter.groupby(first.sequence_id).min().tolist(), [0, 0])

    def test_json_candidates_clone_and_parameter_updates(self):
        extractor = SequenceExtractor(pretraining_include_classes='["wave"]', pretraining_filters='{"subject":["p2"]}')
        copied = clone(extractor)
        self.assertEqual(set(copied.select_pretraining_sequences(self.frame).sequence_id), {"c"})
        copied.set_params(pretraining_source="target", pretraining_include_classes=None, pretraining_filters=None)
        self.assertEqual(set(copied.select_pretraining_sequences(self.frame).sequence_id), {"a", "e"})

    def test_invalid_sampling_controls_fail_instead_of_silently_changing_data(self):
        for params in ({"pretraining_sample_fraction": 0}, {"pretraining_sample_fraction": np.nan},
                       {"pretraining_max_sequences": 0}, {"pretraining_max_sequences_per_class": 1.5},
                       {"pretraining_min_length": 9, "pretraining_max_length": 2}):
            with self.subTest(params=params), self.assertRaises(InvalidExtractorParams):
                SequenceExtractor(**params).select_pretraining_sequences(self.frame)

    def test_ordinary_extraction_is_not_implicitly_target_filtered(self):
        extractor = SequenceExtractor(acc_modes="raw", rotation_modes="", tof_modes="", thm_modes="",
                                      stft_configs=None, cwt_configs=None, chunk_window_size=4, chunk_stride=4)
        chunks = extractor.fit_transform(self.frame)
        self.assertEqual(set(chunks["sequence_ids"]), set(self.frame.sequence_id))


if __name__ == "__main__":
    unittest.main()
