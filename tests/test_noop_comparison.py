"""Native notebook log observations: no model fitting or submission in tests."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from plot_noop_comparison import telemetry_history


class NativeLogTests(unittest.TestCase):
    def test_replayed_epoch_is_deduplicated(self):
        rows = [dict(epoch=1, fit=1, loss=2.), dict(epoch=2, fit=1, loss=1.),
                dict(epoch=2, fit=1, loss=.9)]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "epochs.jsonl"
            path.write_text('\n'.join(map(json.dumps, rows)))
            history = telemetry_history(path)
        self.assertEqual(history["epoch"], [1, 2])
        self.assertEqual(history["loss"], [2., .9])

    def test_missing_epochs_keep_their_actual_numbers(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "epochs.jsonl"
            path.write_text('\n'.join(json.dumps(dict(epoch=n, loss=1./n)) for n in (1, 4, 9)))
            history = telemetry_history(path)
        self.assertEqual(history["epoch"], [1, 4, 9])
        self.assertEqual(history["loss"], [1., .25, 1./9])

    def test_refit_history_does_not_mix_with_cv_history(self):
        rows = [dict(fit=1, epoch=n, loss=10./n) for n in (1, 2, 3)]
        rows += [dict(fit=2, epoch=1, loss=3.), dict(fit=1, epoch=3, loss=.1)]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "epochs.jsonl"
            path.write_text('\n'.join(map(json.dumps, rows)))
            history = telemetry_history(path)
        self.assertEqual(history["epoch"], [1])
        self.assertEqual(history["fit"], [2])
        self.assertEqual(history["loss"], [3.])

    def test_native_keras_log_parser_preserves_cv_refit_boundary(self):
        from noop_comparison import parse_native_training_log
        text = ("Fitting 1 folds for each of 1 candidates, totalling 1 fits\n"
                "Epoch 1/2000\n17/17 - 2s - accuracy: 0.4 - loss: 1.2 - val_accuracy: 0.3 - val_loss: 1.4\n"
                "Epoch 2/2000\n17/17 - 2s - accuracy: 0.5 - loss: 0.9 - val_accuracy: 0.4 - val_loss: 1.1\n"
                "Epoch 1/2000\n17/17 - 2s - accuracy: 0.45 - loss: 1.1 - val_accuracy: 0.35 - val_loss: 1.3\n")
        rows = parse_native_training_log(text, "baseline")
        self.assertEqual([(r["fit"], r["epoch"]) for r in rows], [(1, 1), (1, 2), (2, 1)])
        self.assertAlmostEqual(rows[-1]["val_loss"], 1.3)

    def test_incomplete_epoch_has_no_fabricated_metrics(self):
        from noop_comparison import parse_native_training_log
        rows = parse_native_training_log("Epoch 1/2000\nCompiling graph...\n", "enhanced")
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
