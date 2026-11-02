"""No training/submission side effects; frozen recipe and live curve contracts."""
import ast
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from noop_runtime import prepare_data
from plot_noop_comparison import telemetry_history


class ComparisonTests(unittest.TestCase):
    def test_frozen_reference_is_completed_winner_not_original_search_cap(self):
        p = json.loads((ROOT / "experiments/noop_candidate21/reference_params.json").read_text())
        self.assertEqual(len(p), 215)
        self.assertEqual(p["estimator__epochs"], 2000)
        self.assertEqual(p["estimator__early_stopping_patience"], 0)
        self.assertIs(p["estimator__restore_best_weights"], False)
        self.assertEqual(p["estimator__augmentor__prob"], 0)
        self.assertEqual(p["estimator__extractor__acc_modes"], "raw")
        self.assertEqual(p["estimator__conv_dropout"], .1)
        self.assertEqual(p["estimator__weight_decay"], 0)

    def test_data_preparation_splits_whole_sequences_and_matches_manifest(self):
        import base_utils as bu
        rows = []
        for subject in ("p0", "p1", "p2", "p3"):
            for category in (0, 1):
                for sequence in range(4):
                    for n in range(3):
                        rows.append(dict(sequence_id=f"{subject}-{category}-{sequence}",
                            subject=subject, gesture=f"location - action{category}",
                            sequence_type="Target" if category else "Non-Target", orientation="Upright",
                            sequence_counter=n, acc_x=float(n), acc_y=0., acc_z=9.8,
                            rot_w=1., rot_x=0., rot_y=0., rot_z=0.))
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            train, yt, test, yv, manifest = prepare_data(path, bu)
        self.assertEqual(manifest["filtered_sequences"], 32)
        self.assertEqual(manifest["train_sequences"], 16)
        self.assertEqual(manifest["holdout_sequences"], 16)
        self.assertFalse(set(train.sequence_id) & set(test.sequence_id))
        self.assertEqual(list(yt.index), list(train.index))
        self.assertEqual(list(yv.index), list(test.index))
        self.assertEqual(manifest["target_holdout_sequences"], 8)

    def test_live_telemetry_deduplicates_epochs_without_inventing_future_values(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "epoch_telemetry.jsonl"
            rows = [dict(epoch=1, variant="baseline", loss=2., accuracy=.2),
                    dict(epoch=2, variant="baseline", loss=1., accuracy=.4),
                    dict(epoch=2, variant="baseline", loss=.9, accuracy=.45)]
            p.write_text("\n".join(json.dumps(row) for row in rows))
            history = telemetry_history(p)
        self.assertEqual(history["loss"], [2., .9])
        self.assertEqual(history["accuracy"], [.2, .45])
        self.assertEqual(history["epoch"], [1, 2])

    def test_missing_log_epochs_keep_their_actual_numbers(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "epoch_telemetry.jsonl"
            p.write_text('\n'.join(json.dumps(dict(epoch=n, loss=1. / n)) for n in (1, 4, 9)))
            history = telemetry_history(p)
        self.assertEqual(history["epoch"], [1, 4, 9])
        self.assertEqual(history["loss"], [1., .25, 1. / 9])

    def test_one_fit_and_one_final_holdout_adapter_call_in_runtime(self):
        tree = ast.parse((ROOT / "scripts/noop_runtime.py").read_text())
        fits = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == "fit"
                and isinstance(n.func.value, ast.Name) and n.func.value.id == "pipeline"]
        holdouts = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id == "_holdout_results"]
        self.assertEqual(len(fits), 1)
        self.assertEqual(len(holdouts), 1)

    def test_monitor_collects_other_run_after_one_terminal_failure(self):
        import noop_comparison as comparison
        api = SimpleNamespace(
            kernels_logs_stream=lambda kernel: iter(()),
            kernels_status=lambda kernel: SimpleNamespace(to_dict=lambda _: {
                "status": "ERROR" if kernel == "baseline" else "COMPLETE"}))
        attempted = []
        def collect(_api, run_id, variant):
            attempted.append(variant)
            if variant == "baseline":
                raise RuntimeError("baseline failed before fitting")
        with TemporaryDirectory() as tmp, \
                patch.object(comparison, "api_client", return_value=api), \
                patch.object(comparison, "build_root", return_value=Path(tmp) / "build"), \
                patch.object(comparison, "results_root", return_value=Path(tmp) / "results"), \
                patch.object(comparison, "verify_remote", side_effect=lambda api, path: path.name), \
                patch.object(comparison, "collect", side_effect=collect), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "Finished monitoring both runs"):
                comparison.watch("test", seconds=5)
            self.assertEqual(attempted, ["baseline", "enhanced"])
            self.assertTrue((Path(tmp) / "results/baseline/collection_failure.json").exists())

    def test_generous_recipe_enables_methods_not_boolean_aliases(self):
        from noop_sequence_extractor import NoopOptions
        p = json.loads((ROOT / "experiments/noop_candidate21/enhanced_options.json").read_text())
        opts = NoopOptions(**p).validate(100)
        self.assertTrue(opts.enabled())
        self.assertEqual(opts.periodicity, "acf_harmonic_checked")
        self.assertEqual(opts.quality, "features")
        self.assertIn("jerk_rms", opts.motion_stats)


if __name__ == "__main__":
    unittest.main()
