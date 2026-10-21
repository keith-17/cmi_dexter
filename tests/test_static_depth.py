"""Regression checks for configured static branch depth in the legacy CNN."""

import json
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import tensorflow as tf

from multibranch_utils import build_multibranch_model


class StaticDepthTests(unittest.TestCase):
    def tearDown(self):
        tf.keras.backend.clear_session()

    def build(self, depths):
        return build_multibranch_model(
            time_series_shapes={},
            static_shapes={"stft": 8, "cwt": 6},
            n_classes=3,
            branch_num_conv_layers=depths,
            static_branch_units=10,
            static_dropout=0.1,
            use_batchnorm=True,
            dropout=0.0,
            optimizer_name="adam",
            loss_name="sparse_cce",
        )

    @staticmethod
    def dense_names(model, branch):
        return [
            layer.name
            for layer in model.layers
            if isinstance(layer, tf.keras.layers.Dense)
            and layer.name.startswith(branch + "_")
        ]

    def test_independent_static_depths_and_forward_pass(self):
        model = self.build(json.dumps({"stft": 1, "cwt": 4}))
        self.assertEqual(self.dense_names(model, "stft"), ["stft_embed"])
        self.assertEqual(
            self.dense_names(model, "cwt"),
            ["cwt_dense1", "cwt_dense2", "cwt_dense3", "cwt_embed"],
        )
        for branch in ("stft", "cwt"):
            for name in self.dense_names(model, branch):
                self.assertEqual(model.get_layer(name).units, 10)
        output = model(
            {"stft_input": np.zeros((2, 8)), "cwt_input": np.zeros((2, 6))},
            training=False,
        ).numpy()
        self.assertEqual(output.shape, (2, 3))
        np.testing.assert_allclose(output.sum(axis=1), np.ones(2), atol=1e-6)

    def test_two_layer_names_remain_compatible(self):
        model = self.build({"stft": 2, "cwt": 2})
        for branch in ("stft", "cwt"):
            self.assertEqual(
                self.dense_names(model, branch),
                [f"{branch}_dense1", f"{branch}_embed"],
            )
            self.assertIsInstance(model.get_layer(f"{branch}_bn"), tf.keras.layers.BatchNormalization)
            self.assertIsInstance(model.get_layer(f"{branch}_dropout"), tf.keras.layers.Dropout)

    def test_missing_static_entries_preserve_original_depth(self):
        for depths in ({"acc": 4}, 4):
            with self.subTest(depths=depths):
                model = self.build(depths)
                self.assertEqual(len(self.dense_names(model, "stft")), 2)
                self.assertEqual(len(self.dense_names(model, "cwt")), 2)

    def test_invalid_static_depths_raise(self):
        for depth in (0, -1, 1.5, True, "4"):
            with self.subTest(depth=depth):
                with self.assertRaisesRegex(ValueError, "Static branch depth.*positive integer"):
                    self.build({"stft": depth, "cwt": 2})

    def build_bert(self, depths):
        return build_multibranch_model(
            time_series_shapes={},
            static_shapes={"stft": 8, "cwt": 6},
            sequence_length=4,
            n_classes=3,
            temporal_architecture="bert",
            branch_num_conv_layers=depths,
            static_branch_units=10,
            static_dropout=0.1,
            dense_units=8,
            dropout=0.0,
            fusion_units=8,
            bert_d_model=8,
            bert_num_heads=2,
            bert_num_layers=1,
            bert_ff_dim=8,
            bert_pooling="avg",
            optimizer_name="adam",
            loss_name="sparse_cce",
        )

    def test_bert_independent_static_depths_and_forward_pass(self):
        model = self.build_bert(json.dumps({"stft": 1, "cwt": 4}))
        self.assertEqual(self.dense_names(model, "stft"), ["stft_static_dense"])
        self.assertEqual(
            self.dense_names(model, "cwt"),
            ["cwt_static_hidden1", "cwt_static_hidden2", "cwt_static_hidden3", "cwt_static_dense"],
        )
        for branch in ("stft", "cwt"):
            self.assertEqual(model.get_layer(f"{branch}_static_repeat").output.shape, (None, 4, 10))
            for name in self.dense_names(model, branch):
                self.assertEqual(model.get_layer(name).units, 10)
        output = model(
            {
                "stft_input": np.zeros((2, 8)),
                "cwt_input": np.zeros((2, 6)),
                "valid_timestep_mask": np.ones((2, 4)),
            },
            training=False,
        ).numpy()
        self.assertEqual(output.shape, (2, 3))
        np.testing.assert_allclose(output.sum(axis=1), np.ones(2), atol=1e-6)

    def test_bert_default_static_projection_remains_compatible(self):
        for depths in ({"acc": 4}, 4, {"stft": 1, "cwt": 1}):
            with self.subTest(depths=depths):
                model = self.build_bert(depths)
                for branch in ("stft", "cwt"):
                    self.assertEqual(self.dense_names(model, branch), [f"{branch}_static_dense"])
                    self.assertIsInstance(model.get_layer(f"{branch}_static_dropout"), tf.keras.layers.Dropout)

    def test_bert_invalid_static_depths_raise(self):
        for depth in (0, -1, 1.5, True, "4"):
            with self.subTest(depth=depth):
                with self.assertRaisesRegex(ValueError, "Static branch depth.*positive integer"):
                    self.build_bert({"stft": depth, "cwt": 1})


if __name__ == "__main__":
    unittest.main()
