"""Executable notebook configuration checks without fitting expensive models."""
import ast
from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import base_utils as data_utils
import multibranch_utils as multibranch_keras
import pretraining_utils as pretraining_keras
from base_utils import ECat, EFloat, EInt, SensorAugmentor, SequenceExtractor


class PretrainingNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads((ROOT / "archive/notebooks/multibranch_pretraining_v1.ipynb").read_text(encoding="utf-8"))
        cls.original = json.loads((ROOT / "notebooks/multibranch_v1.ipynb").read_text(encoding="utf-8"))

    def source(self, index):
        return "".join(self.notebook["cells"][index]["source"])

    def namespace(self, directory, model="model_1", first_run=True):
        namespace = {
            "workspace_root": directory, "Path": Path, "datetime": datetime,
            "np": np, "pd": pd, "json": json, "GroupShuffleSplit": GroupShuffleSplit,
            "GroupKFold": GroupKFold, "Pipeline": Pipeline, "ECat": ECat,
            "EInt": EInt, "EFloat": EFloat, "SequenceExtractor": SequenceExtractor,
            "SensorAugmentor": SensorAugmentor, "pretraining_keras": pretraining_keras,
            "multibranch_keras": multibranch_keras,
            "SKOPT_AVAILABLE": multibranch_keras.HierarchicalBayesSearchCV is not None,
        }
        source = self.source(3).replace("MODEL_VARIANT = 'model_1'", f"MODEL_VARIANT = {model!r}")
        source = source.replace("FIRST_RUN = True", f"FIRST_RUN = {first_run!r}")
        with redirect_stdout(StringIO()):
            exec(compile(source, "configuration", "exec"), namespace)
            for index in (4, 11, 14):
                exec(compile(self.source(index), f"notebook-cell-{index}", "exec"), namespace)
        return namespace

    def test_layout_and_no_functions(self):
        self.assertEqual(len(self.notebook["cells"]), len(self.original["cells"]))
        self.assertEqual([cell["cell_type"] for cell in self.notebook["cells"]],
                         [cell["cell_type"] for cell in self.original["cells"]])
        for index, cell in enumerate(self.notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            tree = ast.parse("".join(cell["source"]))
            self.assertFalse(any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
                                 for node in ast.walk(tree)), f"Function in cell {index}")
            self.assertEqual(cell["outputs"], [])

    def test_both_objectives_and_all_spaces_expose_every_parameter(self):
        sources = ["non_target", "all", "target", "baseline"]
        for model in ("model_1", "model_2"):
            for first_run in (True, False):
                with self.subTest(model=model, first_run=first_run), TemporaryDirectory() as directory:
                    namespace = self.namespace(directory, model=model, first_run=first_run)
                    pipeline = namespace["pipeline"]
                    estimator = pipeline.named_steps["estimator"]
                    self.assertEqual(estimator.pretraining_mode,
                                     "supervised" if model == "model_1" else "masked_ssl")
                    params = pipeline.get_params(deep=True)
                    leaves = {key for key in params if key.startswith("estimator__")
                              and key not in {"estimator__extractor", "estimator__augmentor"}}
                    for name in ("GRID_PARAM_SPACE", "BAYESIAN_PARAM_SPACE", "EVOLUTIONARY_SEARCH_SPACE"):
                        self.assertEqual(leaves, set(namespace[name]), name)
                    self.assertEqual(namespace["GRID_PARAM_SPACE"]["estimator__extractor__pretraining_source"], sources)
                    evolutionary = namespace["EVOLUTIONARY_PARAM_SPACE"]["estimator__extractor__pretraining_source"]
                    self.assertEqual(list(evolutionary.choices), sources)
                    if namespace["SKOPT_AVAILABLE"]:
                        self.assertEqual(list(namespace["BAYESIAN_PARAM_SPACE"]["estimator__extractor__pretraining_source"].categories), sources)
                    self.assertEqual(clone(pipeline).get_params()["estimator__pretraining_mode"], estimator.pretraining_mode)

    def test_first_run_fixed_learning_and_disabled_augmentation_regularisation(self):
        with TemporaryDirectory() as directory:
            namespace = self.namespace(directory)
            grid = namespace["GRID_PARAM_SPACE"]
            self.assertEqual(namespace["n_iter"], 5)
            self.assertEqual(grid["estimator__epochs"], [5])
            self.assertEqual(grid["estimator__pretraining_epochs"], [5])
            for key in ("learning_rate", "pretraining_learning_rate", "finetuning_learning_rate"):
                self.assertEqual(grid["estimator__" + key], [1e-4])
            for key, value in namespace["FIRST_RUN_FIXED_PARAMS"].items():
                self.assertEqual(grid[key], [value], key)
            for key in ("use_lr_scheduler", "pretraining_use_lr_scheduler", "use_batchnorm"):
                self.assertEqual(grid["estimator__" + key], [False])
            for key in ("early_stopping_patience", "pretraining_early_stopping_patience"):
                self.assertEqual(grid["estimator__" + key], [0])
            # Full ranges survive the pilot override for the next experiment.
            self.assertEqual(namespace["FULL_GRID_PARAM_SPACE"]["estimator__epochs"], [150])
            self.assertEqual(grid["estimator__bert_d_model"], [32, 64, 96, 128, 160])
            widths = grid["estimator__bert_d_model"]
            for width in widths:
                self.assertEqual(width % grid["estimator__bert_num_heads"][0], 0)
            default_features = SequenceExtractor(stft_configs=[{}], cwt_configs=[{}]).get_params()
            for key in ("acc_modes", "rotation_modes", "tof_modes", "thm_modes", "maxlen",
                        "chunk_window_size", "padding_value", "stft_configs", "cwt_configs"):
                self.assertEqual(grid["estimator__extractor__" + key], [default_features[key]])

    def test_holdout_and_cv_keep_both_domains_but_separate_wearers(self):
        records = []
        for wearer in range(20):
            for target in (0, 1):
                for timestep in range(2):
                    records.append({"subject": f"s{wearer}", "sequence_id": f"s{wearer}_{target}",
                                    "is_target": target, "bfrb": "target" if target else "non_bfrb",
                                    "gesture": "target" if target else "everyday", "counter": timestep})
        with TemporaryDirectory() as directory:
            namespace = self.namespace(directory)
            namespace.update(train_df=pd.DataFrame(records), data_utils=data_utils, sensor_columns=[])
            with redirect_stdout(StringIO()):
                exec(compile(self.source(9), "split", "exec"), namespace)
            train, test = namespace["X_train"], namespace["X_test"]
            self.assertTrue(set(train.subject).isdisjoint(test.subject))
            self.assertEqual(set(train.is_target), {0, 1})
            self.assertEqual(set(test.is_target), {0, 1})
            for fitting, validation in namespace["cv_object"].split(train, namespace["y_train"], namespace["groups"]):
                self.assertTrue(set(train.iloc[fitting].subject).isdisjoint(train.iloc[validation].subject))
                self.assertEqual(set(train.iloc[fitting].is_target), {0, 1})

    def test_holdout_aligns_labels_and_records_problematic_exclusions(self):
        frame = pd.DataFrame({"sequence_id": ["keep", "keep", "bad", "bad", "source", "source"],
                              "is_target": [1, 1, 1, 1, 0, 0],
                              "gesture": ["a", "a", "b", "b", "everyday", "everyday"],
                              "bfrb": ["a", "a", "b", "b", "non_bfrb", "non_bfrb"]})
        estimator = SimpleNamespace(extractor_=SequenceExtractor())
        model = SimpleNamespace(named_steps={"estimator": estimator},
                                predict=Mock(return_value=pd.Series(["a"], index=pd.Index(["keep"], name="sequence_id"))))
        with TemporaryDirectory() as directory:
            namespace = dict(experiment_succeeded=True, search=SimpleNamespace(best_estimator_=model),
                             X_test=frame, y_test=frame.copy(), TARGET_COL="bfrb", pd=pd, json=json,
                             results_dir=Path(directory), timestamp="run", evaluate_holdout=data_utils.evaluate_holdout)
            with redirect_stdout(StringIO()):
                exec(compile(self.source(22), "holdout", "exec"), namespace)
            self.assertEqual(set(namespace["y_test_final"].sequence_id), {"keep"})
            exclusions = json.loads((Path(directory) / "run_holdout_exclusions.json").read_text())
            self.assertEqual(exclusions["sequence_ids"], ["bad"])
            self.assertEqual(exclusions["evaluated_sequences"], 1)
            self.assertEqual(exclusions["requested_sequences"], 2)


if __name__ == "__main__":
    unittest.main()
