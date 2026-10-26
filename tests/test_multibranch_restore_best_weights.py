"""Early-stopping options remain configurable through sklearn searches."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sklearn.base import clone


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import multibranch_utils as mb
from pretraining_utils import AdvancedMultiBranchSequenceClassifier


class RecordingEarlyStopping:
    def __init__(self, **kwargs):
        self.options = kwargs


class RestoreBestWeightsTests(unittest.TestCase):
    estimator_classes = (
        mb.MultiBranchSequenceClassifier,
        AdvancedMultiBranchSequenceClassifier,
    )

    def test_default_preserves_best_weight_restoration(self):
        for estimator_class in self.estimator_classes:
            with self.subTest(estimator=estimator_class.__name__):
                self.assertIs(estimator_class().restore_best_weights, True)

    def test_sklearn_clone_and_set_params_preserve_option(self):
        for estimator_class in self.estimator_classes:
            for restore in (True, False):
                with self.subTest(estimator=estimator_class.__name__, restore=restore):
                    original = estimator_class(restore_best_weights=restore)
                    copied = clone(original)
                    self.assertIs(copied.get_params()["restore_best_weights"], restore)
                    copied.set_params(restore_best_weights=not restore)
                    self.assertIs(copied.restore_best_weights, not restore)
                    self.assertIs(original.restore_best_weights, restore)

    def test_callback_receives_requested_restoration_and_monitor(self):
        callback_module = SimpleNamespace(EarlyStopping=RecordingEarlyStopping)
        for estimator_class in self.estimator_classes:
            for restore in (True, False):
                for has_validation in (True, False):
                    with self.subTest(estimator=estimator_class.__name__, restore=restore,
                                      has_validation=has_validation):
                        estimator = estimator_class(
                            early_stopping_patience=3,
                            restore_best_weights=restore,
                        )
                        with patch.object(mb, "callbacks", callback_module):
                            result = estimator._build_callbacks(has_validation)
                        self.assertEqual(len(result), 1)
                        self.assertEqual(result[0].options, {
                            "monitor": "val_loss" if has_validation else "loss",
                            "patience": 3,
                            "restore_best_weights": restore,
                        })

    def test_zero_patience_disables_early_stopping_for_either_option(self):
        for estimator_class in self.estimator_classes:
            for restore in (True, False):
                with self.subTest(estimator=estimator_class.__name__, restore=restore):
                    estimator = estimator_class(
                        early_stopping_patience=0,
                        restore_best_weights=restore,
                    )
                    with patch.object(mb, "callbacks") as callback_module:
                        self.assertEqual(estimator._build_callbacks(True), [])
                        callback_module.EarlyStopping.assert_not_called()


if __name__ == "__main__":
    unittest.main()
