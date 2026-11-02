"""NOOP's optional static branch and parameter-preserving search contract."""
import copy
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from base_utils import (ECat, InvalidExtractorParams, SequenceExtractor,
                        inactive_sequence_parameters, sequence_feature_families)
import multibranch_utils as mb
from noop_sequence_extractor import NoopOptions, NoopSequenceExtractor
from pretraining_utils import AdvancedMultiBranchSequenceClassifier


def extractor(child=True, options=None, **overrides):
    settings = dict(acc_modes="raw", rotation_modes="", tof_modes="", thm_modes="",
                    stft_configs=None, cwt_configs=None, maxlen=32,
                    chunk_window_size=32, chunk_stride=32, padding_value=0.,
                    pretraining_source="baseline", finetuning_source="all")
    settings.update(overrides)
    return (NoopSequenceExtractor(noop=NoopOptions() if options is None else options, **settings)
            if child else SequenceExtractor(**settings))


def recordings():
    rows = []
    for subject in range(6):
        for category in range(2):
            for counter in range(32):
                t = counter / 20.
                rows.append(dict(sequence_id=f"s{subject}-{category}", subject=f"s{subject}",
                                 sequence_counter=counter, target=f"c{category}",
                                 gesture=f"g{category}", is_target=1,
                                 acc_x=np.sin(2 * np.pi * (1 + category) * t),
                                 acc_y=np.cos(2 * np.pi * (1 + category) * t),
                                 acc_z=9.8 + .1 * np.sin(2 * np.pi * t)))
    frame = pd.DataFrame(rows)
    return frame, frame[["sequence_id", "subject", "target", "gesture", "is_target"]].copy()


def classifier(ext, **overrides):
    settings = dict(primary_target="target", extractor=ext, epochs=1, batch_size=4,
                    validation_split=.25, pretraining_validation_split=.25,
                    validation_group_column="subject", pretraining_validation_group_column="subject",
                    class_weight_mode=None, early_stopping_patience=0,
                    use_lr_scheduler=False, use_batchnorm=False, dropout=0.,
                    conv_dropout=0., static_dropout=0., branch_num_conv_layers=1,
                    branch_filters={"acc": 4}, branch_pool_sizes={"acc": 2},
                    branch_dense_units=4, static_branch_units=4, dense_units=4,
                    invalid_candidate_policy="raise", random_state=7, verbose=0)
    settings.update(overrides)
    return AdvancedMultiBranchSequenceClassifier(**settings)


FIT_LOG = []


class DryNoopClassifier(AdvancedMultiBranchSequenceClassifier):
    def fit(self, X, y=None, **kwargs):
        error = mb.candidate_error({}, self.get_params(deep=True), y=y)
        if error is not None:
            raise ValueError(error)
        FIT_LOG.append(self.get_params(deep=True))
        return self

    def score(self, X, y=None):
        return .75


class NoopSearchIntegrationTests(unittest.TestCase):
    def pipeline(self):
        return Pipeline([("estimator", DryNoopClassifier(extractor=extractor()))])

    def test_nested_clone_and_conditional_parameter_visibility(self):
        original = classifier(extractor())
        copied = clone(original).set_params(extractor__noop__periodicity="acf",
                                            extractor__noop__min_samples=24)
        self.assertEqual(copied.get_params()["extractor__noop__min_samples"], 24)
        self.assertIsNone(original.extractor.noop.periodicity)
        self.assertEqual(copied.extractor.noop.periodicity, "acf")
        inactive = inactive_sequence_parameters(original.extractor.get_params())
        self.assertIn("noop__min_samples", inactive)
        self.assertNotIn("noop__periodicity", inactive)
        self.assertNotIn("noop__band", inactive_sequence_parameters(copied.extractor.get_params()))
        self.assertNotIn("noop", sequence_feature_families(original.extractor.get_params()))
        self.assertIn("noop", sequence_feature_families(copied.extractor.get_params()))

    def test_grid_signature_ignores_disabled_numeric_values_without_rewriting(self):
        base = self.pipeline().get_params(deep=True)
        grid = {"estimator__extractor__noop__motion_stats": [None, "basic"],
                "estimator__extractor__noop__min_samples": [16, 24]}
        before = copy.deepcopy(grid)
        result = mb.deduplicate_param_grid(grid, base)
        self.assertEqual(len(result), 3)
        self.assertEqual(grid, before)
        self.assertTrue(all(set(row) == set(grid) for row in result))
        self.assertEqual(result[0]["estimator__extractor__noop__min_samples"], [16])
        a = {"estimator__extractor__noop__min_samples": 16}
        b = {"estimator__extractor__noop__min_samples": 24}
        self.assertEqual(mb.effective_candidate_signature(a, base), mb.effective_candidate_signature(b, base))
        a["estimator__extractor__noop__motion_stats"] = "basic"
        b["estimator__extractor__noop__motion_stats"] = "basic"
        self.assertNotEqual(mb.effective_candidate_signature(a, base), mb.effective_candidate_signature(b, base))

    def test_whole_extractor_objects_canonicalize_only_their_inactive_noop_values(self):
        a = {"extractor": extractor(options=NoopOptions(min_samples=16))}
        b = {"extractor": extractor(options=NoopOptions(min_samples=24))}
        self.assertEqual(mb.effective_candidate_signature(a), mb.effective_candidate_signature(b))
        b["extractor"].set_params(noop__motion_stats="basic")
        self.assertNotEqual(mb.effective_candidate_signature(a), mb.effective_candidate_signature(b))
        base = self.pipeline().get_params(deep=True)
        left = {"estimator__extractor": extractor(options=NoopOptions(motion_stats="basic", min_samples=16))}
        right = {"estimator__extractor": extractor(options=NoopOptions(motion_stats="basic", min_samples=24))}
        self.assertNotEqual(mb.effective_candidate_signature(left, base),
                            mb.effective_candidate_signature(right, base))
        right["estimator__extractor"].set_params(noop__min_samples=-1)
        self.assertIn("min_samples", mb.candidate_error(right, base))
        left = {"estimator__extractor__noop": NoopOptions(motion_stats="basic", min_samples=16)}
        right = {"estimator__extractor__noop": NoopOptions(motion_stats="basic", min_samples=24)}
        self.assertNotEqual(mb.effective_candidate_signature(left, base),
                            mb.effective_candidate_signature(right, base))

    def test_grid_keeps_huge_non_noop_ranges_unexpanded(self):
        base = self.pipeline().get_params(deep=True)
        grid = {f"estimator__unused_{index}": range(100) for index in range(12)}
        grid.update({"estimator__extractor__noop__motion_stats": [None, "basic"],
                     "estimator__extractor__noop__min_samples": [16, 24]})
        reduced = mb.deduplicate_param_grid(grid, base)
        self.assertEqual(len(reduced), 3)
        for subgrid in reduced:
            for index in range(12):
                self.assertIs(subgrid[f"estimator__unused_{index}"], grid[f"estimator__unused_{index}"])
        from sklearn.model_selection import ParameterGrid
        small = {"estimator__dense_units": [4, 8],
                 "estimator__extractor__noop__motion_stats": [None, "basic"],
                 "estimator__extractor__noop__min_samples": [16, 24]}
        self.assertEqual(len(ParameterGrid(mb.deduplicate_param_grid(small, base))), 6)

    def test_inactive_numeric_controls_do_not_reject_the_baseline(self):
        base = self.pipeline().get_params(deep=True)
        disabled = {"estimator__extractor__noop__min_samples": -1}
        self.assertIsNone(mb.candidate_error(disabled, base))
        enabled = dict(disabled, estimator__extractor__noop__motion_stats="basic")
        self.assertIn("min_samples", mb.candidate_error(enabled, base))

    def test_ga_and_bayes_fit_each_effective_noop_configuration_once(self):
        frame, labels = recordings()
        key = "estimator__extractor__noop__motion_stats"
        numeric = "estimator__extractor__noop__min_samples"
        searches = [mb.HierarchicalGASearchCV(
            self.pipeline(), param_grid={key: ECat([None, "basic"]), numeric: ECat([16, 24])},
            population_size=8, tournament_size=2, generations=0, cv=2,
            refit=False, verbose=0, random_state=3)]
        if mb.BayesSearchCV is not None:
            from skopt.space import Categorical
            searches.append(mb.HierarchicalBayesSearchCV(
                self.pipeline(), search_spaces={key: Categorical([None, "basic"]),
                                               numeric: Categorical([16, 24])},
                n_iter=8, cv=2, n_jobs=1, refit=False, verbose=0, random_state=3))
        for search in searches:
            with self.subTest(search=type(search).__name__):
                FIT_LOG.clear()
                search.fit(frame, labels)
                rows = search.cv_results_["params"]
                self.assertEqual(len(rows), 3)
                self.assertEqual(len(FIT_LOG), 6)
                self.assertTrue(all(set(row) == {key, numeric} for row in rows))

    def test_optional_matrix_routing_keeps_temporal_tensors_and_rejects_schema_drift(self):
        model = mb.MultiBranchSequenceClassifier()
        model.branch_groups_ = {"acc": [0]}
        model.noop_feature_names_ = ["noop_rms", "noop_rms_support"]
        values = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
        original = values.copy()
        chunks = {"X": values, "mask": np.ones((2, 3), dtype=np.float32),
                  "noop_features": np.array([[2., 1.], [3., 1.]], dtype=np.float32),
                  "noop_feature_names": model.noop_feature_names_}
        arrays = model._branches_from_chunks(chunks)
        np.testing.assert_array_equal(arrays["acc"], values[:, :, [0]])
        np.testing.assert_array_equal(arrays["noop"], chunks["noop_features"])
        np.testing.assert_array_equal(values, original)
        with self.assertRaisesRegex(InvalidExtractorParams, "schema"):
            model._branches_from_chunks(dict(chunks, noop_feature_names=["other", "other_support"]))
        with self.assertRaisesRegex(ValueError, "Unmapped feature"):
            mb.group_feature_names(["unregistered_feature"])
        model.noop_feature_names_ = []
        report_only = dict(chunks, noop_features=np.zeros((2, 0), dtype=np.float32),
                           noop_feature_names=[])
        self.assertEqual(set(model._branches_from_chunks(report_only)), {"acc"})


@unittest.skipUnless(mb.TENSORFLOW_AVAILABLE, "TensorFlow is required for actual model verification")
class NoopActualModelTests(unittest.TestCase):
    def tearDown(self):
        mb.tf.keras.backend.clear_session()

    def test_all_off_matches_legacy_training_architecture_weights_and_predictions(self):
        frame, labels = recordings()
        fitted = []
        for child in (False, True):
            mb.tf.keras.utils.set_random_seed(123)
            model = classifier(extractor(child=child)).fit(frame, labels)
            fitted.append((model, model.predict_proba(frame).to_numpy(),
                           [value.copy() for value in model.model_.get_weights()]))
        self.assertEqual(fitted[0][0].input_order_, fitted[1][0].input_order_)
        self.assertEqual([layer.name for layer in fitted[0][0].model_.layers],
                         [layer.name for layer in fitted[1][0].model_.layers])
        self.assertEqual(fitted[1][0].noop_feature_names_, [])
        for left, right in zip(fitted[0][2], fitted[1][2]):
            np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(fitted[0][1], fitted[1][1])

    def test_active_extractor_reaches_real_fit_validation_and_sequence_predictions(self):
        frame, labels = recordings()
        model = classifier(extractor(options=NoopOptions(motion_stats="basic"))).fit(frame, labels)
        self.assertIn("noop", model.input_order_)
        self.assertGreater(model.model_.get_layer("noop_input").output.shape[-1], 0)
        self.assertTrue(np.isfinite(model.history_["loss"]).all())
        self.assertTrue(np.isfinite(model.history_["val_loss"]).all())
        probabilities = model.predict_proba(frame)
        self.assertEqual(len(probabilities), frame.sequence_id.nunique())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1., atol=1e-6)
        self.assertIn("noop", set(model.branch_summary().branch))

    def test_static_branch_forward_and_training_for_every_temporal_backbone(self):
        for architecture in ("bigru_attention", "cnn_attention_pool", "phase_attention", "bert"):
            with self.subTest(architecture=architecture):
                mb.tf.keras.backend.clear_session()
                model = mb.build_multibranch_model(
                    {"acc": (16, 3)}, {"noop": 4}, 2, sequence_length=16,
                    temporal_architecture=architecture, use_bigru=architecture == "bigru_attention",
                    branch_cnn_filters=4, branch_cnn_depth=1, branch_dropout=0.,
                    gru_units=4, cnn_filters=4, cnn_depth=1, cnn_dropout=0.,
                    bert_d_model=8, bert_num_heads=2, bert_num_layers=1, bert_ff_dim=8,
                    bert_dropout=0., bert_attention_dropout=0., bert_pooling="avg",
                    static_branch_units=4, static_dropout=0., fusion_units=4,
                    dense_units=4, dropout=0., use_batchnorm=False)
                inputs = [np.ones((2, 16, 3), np.float32), np.ones((2, 4), np.float32),
                          np.ones((2, 16), np.float32)]
                loss = np.asarray(model.train_on_batch(inputs, np.array([0, 1])))
                self.assertTrue(np.isfinite(loss).all())
                output = model(inputs, training=False).numpy()
                self.assertEqual(output.shape, (2, 2))
                np.testing.assert_allclose(output.sum(axis=1), 1., atol=1e-6)

    def test_supervised_and_masked_pretraining_keep_noop_input_connected(self):
        frame, labels = recordings()
        for mode in ("supervised", "masked_ssl"):
            with self.subTest(mode=mode):
                ext = extractor(options=NoopOptions(motion_stats="basic"), pretraining_source="all")
                model = classifier(
                    ext, pretraining_mode=mode, pretraining_epochs=1, pretraining_batch_size=4,
                    pretraining_early_stopping_patience=0, pretraining_use_lr_scheduler=False,
                    temporal_architecture="bert", bert_d_model=8, bert_num_heads=2,
                    bert_num_layers=1, bert_ff_dim=8, bert_dropout=0.,
                    bert_attention_dropout=0., bert_pooling="avg", branch_cnn_filters=4,
                    branch_cnn_depth=1, branch_dropout=0., fusion_units=4,
                    ssl_decoder_units=4, ssl_span_length=3).fit(frame, labels)
                self.assertTrue(model.pretraining_completed_)
                self.assertIn("noop", model.input_order_)
                self.assertTrue(np.isfinite(model.pretraining_history_["loss"]).all())
                self.assertTrue(np.isfinite(model.predict_proba(frame)).all().all())


if __name__ == "__main__":
    unittest.main()
