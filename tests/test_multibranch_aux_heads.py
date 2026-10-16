"""Auxiliary-label alignment and unattended-search regression tests."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace, MethodType
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from test_multibranch_utils_params import mba, ECat
from multibranch_utils import candidate_score, refit_valid_candidate


class DummyCandidate(BaseEstimator):
    def __init__(self, bad=False, invalid_candidate_policy='penalize', invalid_candidate_score=-1.0,
                 refit_bad=False):
        self.bad = bad
        self.invalid_candidate_policy = invalid_candidate_policy
        self.invalid_candidate_score = invalid_candidate_score
        self.refit_bad = refit_bad

    def fit(self, X, y):
        if self.refit_bad and len(X) == 6:
            raise ValueError('full-data failure')
        self.fit_failed_ = self.bad
        return self

    def score(self, X, y):
        return -1.0 if self.fit_failed_ else (0.9 if self.refit_bad else 0.6)


class AuxiliaryTests(unittest.TestCase):
    def test_schema_clone_and_search_serialisation(self):
        heads = [{'name': 'short_head', 'target': 'action', 'window_size': 50}]
        estimator = mba.MultiBranchSequenceClassifier(custom_aux_heads=heads)
        self.assertEqual(clone(estimator).custom_aux_heads, heads)
        for value in (heads, json.dumps(heads)):
            self.assertEqual(mba._normalise_aux_heads(value)[0]['window_size'], 50)
        space = mba.prepare_multibranch_evolutionary_space({'custom_aux_heads': ECat([[], heads])})
        for choice in space['custom_aux_heads'].choices:
            mba._normalise_aux_heads(choice)
        self.assertNotIn('dropout', heads[0])  # defaults never mutate sklearn parameters

    def test_invalid_specs_are_rejected(self):
        for spec in [{'name': 'predictions', 'target': 'x'}, {'name': 'a', 'target': 'x', 'window_size': 0},
                     {'name': 'a', 'target': 'x', 'task': 'unknown'},
                     {'name': 'a', 'target': 'x', 'loss_weight': -1}]:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                mba._normalise_aux_heads([spec])

    def test_aux_targets_align_repeated_chunks_and_numeric_values(self):
        est = mba.MultiBranchSequenceClassifier(custom_aux_heads=[
            {'name': 'action_head', 'target': 'action'},
            {'name': 'numeric_head', 'target': 'numeric', 'task': 'regression'},
        ])
        est.extractor_ = SimpleNamespace(sequence_col='sequence_id')
        est.custom_aux_heads_ = mba._normalise_aux_heads(est.custom_aux_heads)
        est.custom_aux_encoders_ = {'action_head': mba.LabelEncoder().fit(['a', 'b'])}
        est.sample_weight_ = None
        y = pd.DataFrame({'sequence_id': [1, 2], 'action': ['a', 'b'], 'numeric': [0.5, 2.5]})
        targets, weights = est._prepare_training_targets(np.array([1, 0, 1]), [2, 1, 2], y)
        np.testing.assert_array_equal(targets['action_head'], [1, 0, 1])
        np.testing.assert_array_equal(targets['numeric_head'], [[2.5], [0.5], [2.5]])
        self.assertEqual(set(targets), set(weights))
        for value in weights.values():
            self.assertEqual(len(value), 3)
        with self.assertRaises(ValueError):
            est._aux_column_labels([3], y, 'action')

    def test_penalty_logs_failure_and_raise_policy_propagates(self):
        with tempfile.TemporaryDirectory() as tmp:
            est = mba.MultiBranchSequenceClassifier(invalid_candidate_policy='penalize',
                candidate_failure_log=str(Path(tmp) / 'failures.jsonl'))
            with patch.object(mba.MultiBranchSequenceClassifier, '_fit_impl', side_effect=ValueError('bad combo')):
                est.fit(None, None)
                self.assertEqual(est.score(None, None), -1.0)
                self.assertIn('bad combo', json.loads(Path(est.candidate_failure_log).read_text())['error'])
                est.set_params(invalid_candidate_policy='raise')
                with self.assertRaises(ValueError):
                    est.fit(None, None)

    def test_fit_routes_all_heads_and_phase_provider_per_partition(self):
        from base_utils import SequenceExtractor
        extractor = SequenceExtractor(output_format='chunks', chunk_window_size=2, chunk_stride=2)
        def fit(extractor_self, frame):
            extractor_self.feature_names_in_ = np.array(['acc_x'])
            return extractor_self
        def transform(extractor_self, frame):
            ids = np.repeat(frame['sequence_id'].drop_duplicates().to_numpy(), 2)
            return {'X': np.zeros((len(ids), 2, 1), dtype=np.float32),
                    'mask': np.ones((len(ids), 2), dtype=bool),
                    'sequence_ids': ids, 'feature_names': ['acc_x']}
        extractor.fit = MethodType(fit, extractor)
        extractor.transform_chunks = MethodType(transform, extractor)
        partitions = []
        def provider(frame, chunks):
            partitions.append(set(frame['sequence_id']))
            return np.zeros(chunks['mask'].shape, dtype=np.int32)
        model = SimpleNamespace()
        def record_fit(*args, **kwargs):
            model.fit_args, model.fit_kwargs = args, kwargs
            return SimpleNamespace(history={'loss': [1.0]})
        model.fit = record_fit
        est = mba.MultiBranchSequenceClassifier(extractor=extractor, validation_split=0.25,
            class_weight_mode='balanced', early_stopping_patience=0, use_orientation_aux_head=True,
            use_phase_aux_head=True, phase_label_provider=provider,
            custom_aux_heads=[{'name': 'window_head', 'target': 'action', 'window_size': 2}],
            random_state=13)
        frame = pd.DataFrame({'sequence_id': np.repeat(['a', 'b', 'c', 'd'], 4),
                              'sequence_counter': list(range(4))*4, 'acc_x': np.zeros(16)})
        y = pd.DataFrame({'sequence_id': ['a', 'b', 'c', 'd'], 'bfrb': ['target']*4,
                          'orientation': ['upright']*4, 'action': ['scratch']*4})
        with patch.object(mba, 'TENSORFLOW_AVAILABLE', True), \
             patch.object(mba, 'clone', side_effect=lambda value: value), \
             patch.object(mba, 'build_multibranch_model', return_value=model) as build:
            est.fit(frame, y)
        self.assertEqual(len(partitions), 2)
        self.assertFalse(partitions[0] & partitions[1])
        self.assertEqual(build.call_args.kwargs['orientation_num_classes'], 1)
        self.assertEqual(build.call_args.kwargs['custom_aux_heads'][0]['num_classes'], 1)
        targets = model.fit_args[1]
        self.assertEqual(set(targets), {'predictions', 'orientation_aux', 'phase_aux', 'window_head'})
        validation_targets, validation_weights = model.fit_kwargs['validation_data'][1:]
        self.assertEqual(len(validation_weights['predictions']), len(validation_targets['predictions']))
        self.assertNotEqual(len(model.fit_kwargs['sample_weight']['predictions']), len(validation_weights['predictions']))

    def test_grid_all_failed_is_saved_without_refit(self):
        pipe = Pipeline([('estimator', DummyCandidate())])
        X, y = np.zeros((6, 1)), np.zeros(6)
        search = GridSearchCV(pipe, {'estimator__bad': [True]}, cv=2,
                              scoring=candidate_score, refit=False).fit(X, y)
        self.assertFalse(refit_valid_candidate(search, X, y))
        self.assertIsNone(search.best_estimator_)

    def test_evolutionary_all_failed_does_not_refit(self):
        X, y = np.zeros((6, 1)), np.zeros(6)
        from base_utils import GASearchCV
        search = GASearchCV(Pipeline([('estimator', DummyCandidate())]),
                            param_grid={'estimator__bad': ECat([True])}, cv=2,
                            population_size=2, tournament_size=2, generations=1,
                            scoring=candidate_score, refit=False, verbose=0).fit(X, y)
        self.assertFalse(refit_valid_candidate(search, X, y))

    def test_one_failed_fold_disqualifies_positive_mean_candidate(self):
        search = SimpleNamespace(estimator=Pipeline([('estimator', DummyCandidate())]), cv_results_={
            'params': [{'estimator__bad': False}], 'mean_test_score': [0.1],
            'split0_test_score': [-1.0], 'split1_test_score': [0.9],
        })
        self.assertFalse(refit_valid_candidate(search, np.zeros((6, 1)), np.zeros(6)))

    def test_refit_tries_next_candidate_on_full_data_failure(self):
        X, y = np.zeros((6, 1)), np.zeros(6)
        search = GridSearchCV(Pipeline([('estimator', DummyCandidate())]),
                              {'estimator__refit_bad': [True, False]}, cv=2,
                              scoring=candidate_score, refit=False).fit(X, y)
        self.assertTrue(refit_valid_candidate(search, X, y))
        self.assertFalse(search.best_params_['estimator__refit_bad'])
        self.assertEqual(len(search.refit_errors_), 1)

    def test_bayesian_all_penalties_do_not_stop_optimizer(self):
        try:
            from skopt import BayesSearchCV
            from skopt.space import Categorical
        except ImportError:
            self.skipTest('scikit-optimize unavailable')
        X, y = np.zeros((6, 1)), np.zeros(6)
        search = BayesSearchCV(Pipeline([('estimator', DummyCandidate())]),
                              {'estimator__bad': Categorical([True])}, cv=2, n_iter=2,
                              scoring=candidate_score, refit=False).fit(X, y)
        self.assertEqual(len(search.cv_results_['params']), 2)
        self.assertFalse(refit_valid_candidate(search, X, y))


@unittest.skipUnless(mba.TENSORFLOW_AVAILABLE, 'TensorFlow unavailable')
class AuxiliaryModelTests(unittest.TestCase):
    def test_window_and_regression_heads_train_and_ignore_padding(self):
        tf = mba.tf
        heads = mba._normalise_aux_heads([
            {'name': 'short_head', 'target': 'action', 'num_classes': 2, 'window_size': 3},
            {'name': 'numeric_head', 'target': 'numeric', 'task': 'regression', 'window_size': 6},
        ])
        model = mba.build_multibranch_model(
            time_series_shapes={'acc': (8, 3)}, static_shapes={}, n_classes=2,
            custom_aux_heads=heads, use_batchnorm=False, branch_num_conv_layers=1,
            branch_filters={'acc': 4}, branch_kernel_sizes={'acc': 1}, dense_units=4,
        )
        X = np.ones((2, 8, 3), dtype=np.float32)
        mask = np.array([[1]*8, [1]*5+[0]*3], dtype=np.float32)
        targets = {'predictions': np.array([0, 1]), 'short_head': np.array([1, 0]),
                   'numeric_head': np.array([[0.5], [2.0]], dtype=np.float32)}
        weights = {k: np.ones(2, dtype=np.float32) for k in targets}
        result = model.train_on_batch([X, mask], targets, sample_weight=weights)
        self.assertTrue(np.isfinite(result).all())
        pool = tf.keras.Model(model.inputs, model.get_layer('short_head_window_pool').output)
        X_changed = X.copy(); X_changed[1, 5:] = 999
        np.testing.assert_allclose(pool([X, mask]).numpy(), pool([X_changed, mask]).numpy(), atol=1e-5)

    def test_invalid_window_is_rejected_before_training(self):
        with self.assertRaisesRegex(ValueError, 'exceeds chunk length'):
            mba.build_multibranch_model(time_series_shapes={'acc': (8, 3)}, static_shapes={}, n_classes=2,
                custom_aux_heads=[{'name': 'long_head', 'target': 'action', 'num_classes': 2, 'window_size': 9}])

    def test_orientation_phase_and_window_heads_train_with_bert_cls(self):
        model = mba.build_multibranch_model(
            time_series_shapes={'acc': (8, 3)}, static_shapes={}, n_classes=2,
            temporal_architecture='bert', bert_num_layers=1, bert_d_model=8,
            bert_num_heads=2, bert_ff_dim=16, bert_pooling='cls', bert_use_cls_token=True,
            use_orientation_aux_head=True, orientation_num_classes=5,
            use_phase_aux_head=True, n_phases=3, use_batchnorm=False,
            custom_aux_heads=[{'name': 'short_head', 'target': 'action', 'num_classes': 2, 'window_size': 3}],
        )
        X = np.ones((2, 8, 3), dtype=np.float32)
        mask = np.ones((2, 8), dtype=np.float32)
        targets = {'predictions': np.array([0, 1]), 'orientation_aux': np.array([3, 4]),
                   'phase_aux': np.zeros((2, 8), dtype=np.int32), 'short_head': np.array([1, 0])}
        weights = {k: np.ones(v.shape[:2] if k == 'phase_aux' else (2,), dtype=np.float32)
                   for k, v in targets.items()}
        self.assertTrue(np.isfinite(model.train_on_batch([X, mask], targets, sample_weight=weights)).all())
