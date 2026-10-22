import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import dashboard_utils as data_utils


class DashboardChecks(unittest.TestCase):
    def test_separate_candidate_histories(self):
        log = """Fitting 1 folds for each of 1 candidates, totalling 1 fits
Epoch 1/150
1/1 - 1s - accuracy: 0.50 - loss: 1.0 - val_accuracy: 0.4 - val_loss: 1.2
Epoch 2/150
1/1 - 1s - accuracy: 0.70 - loss: 0.8 - val_accuracy: 0.5 - val_loss: 1.1
[CV 1/1] END estimator__learning_rate=0.0001, estimator__extractor__acc_modes=raw;, score=(train=0.8, test=0.7) total time= 1min
Fitting 1 folds for each of 1 candidates, totalling 1 fits
Epoch 1/150
1/1 - 1s - accuracy: 0.2 - loss: 2.0 - val_accuracy: 0.1 - val_loss: 2.2
"""
        trials, current, fits = data_utils.parse_epoch_log(log)
        self.assertEqual(len(trials), 1)
        self.assertEqual(trials[0]["score"], .7)
        self.assertEqual(len(trials[0]["histories"][0]), 2)
        self.assertEqual(current[0][0]["epoch"], 1)
        self.assertEqual(current[0][0]["loss"], 2)
        self.assertEqual(fits, 2)

    def test_event_log_unwraps(self):
        text = json.dumps([{"stream_name": "stdout", "data": "Epoch 1/150\n"},
                           {"stream_name": "stdout", "data": "accuracy: 0.5 - loss: 1.0 - val_loss: 1.2\n"}])
        _, current, _ = data_utils.parse_epoch_log(text)
        self.assertEqual(current[0][0]["accuracy"], .5)

    def test_active_run_is_not_overwritten(self):
        class FakeApi:
            def kernels_status(self, kernel): return {"status": "RUNNING"}
            def kernels_push(self, folder): raise AssertionError("Submission must not happen")
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/"kernel-metadata.json").write_text(json.dumps({"id": "user/notebook"}))
            with patch.object(data_utils, "api_client", return_value=FakeApi()):
                with self.assertRaisesRegex(ValueError, "active run"):
                    data_utils.submit_experiment(folder)

    def test_prepared_notebook_has_applied_settings(self):
        snapshots = data_utils.snapshots()
        if not snapshots:
            self.skipTest("No API source snapshot")
        snapshot = data_utils.load_snapshot(snapshots[0])
        settings = {"name": "dashboard_validation", "search": "bayes", "iterations": 20,
                    "epochs": 1000, "target": "bfrb", "orientations": [],
                    "space": {"estimator__learning_rate": [3e-5], "estimator__epochs": [1000],
                              "estimator__conv_dropout": [.05, .1],
                              "estimator__extractor__acc_modes": ["raw"]}}
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(data_utils, "ROOT", Path(temp)):
                build = data_utils.prepare_experiment(snapshot, settings)
            notebook = data_utils.read_json(build/"experiment.ipynb")
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code": ast.parse("".join(cell["source"]))
            config = data_utils.get_config(notebook)
            self.assertEqual(config["n_iter"], 20)
            self.assertEqual(config["_dashboard_space"]["estimator__learning_rate"], [3e-5])
            self.assertEqual(config["_dashboard_space"]["estimator__conv_dropout"], [.05, .1])
            metadata = data_utils.read_json(build/"kernel-metadata.json")
            self.assertTrue(metadata["enable_gpu"])
            self.assertEqual(metadata["is_private"], snapshot["metadata"]["is_private"])
            self.assertEqual(metadata["dataset_sources"], snapshot["metadata"]["dataset_sources"])
            self.assertIn("GPU required", "".join(notebook["cells"][0]["source"]))


if __name__ == "__main__":
    unittest.main()

