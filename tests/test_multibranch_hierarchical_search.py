"""Architecture-aware search regressions without expensive model training."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from test_multibranch_utils_params import mba, ECat, EInt, make_pipeline
from multibranch_utils import candidate_error, canonical_candidate
from multibranch_utils import (
    HierarchicalGASearchCV,
    HierarchicalBayesSearchCV,
    candidate_score,
    refit_valid_candidate,
)


FIT_LOG = []


class DryRunMultibranch(mba.MultiBranchSequenceClassifier):
    """Use the real estimator parameter contract, but record rather than train."""

    def fit(self, X, y=None, **fit_params):
        reason = candidate_error({}, self.get_params(deep=True), y=y,
                                 phase_labels_available='phase_labels' in fit_params)
        self.fit_failed_ = reason is not None
        if not self.fit_failed_:
            FIT_LOG.append(self.get_params(deep=True))
        return self

    def score(self, X, y=None):
        return -1.0 if self.fit_failed_ else 0.75


def dry_pipeline():
    real = make_pipeline().named_steps['estimator']
    return Pipeline([('estimator', DryRunMultibranch(**real.get_params(deep=False)))])


def toy_data():
    return np.zeros((8, 1)), pd.DataFrame({
        'sequence_id': list(range(8)), 'bfrb': ['a', 'b'] * 4,
        'orientation': ['upright'] * 8, 'action': ['scratch'] * 8,
    })


class CandidateHierarchyTests(unittest.TestCase):
    def setUp(self):
        self.base = make_pipeline().get_params(deep=True)
        self.X, self.y = toy_data()

    def test_penalty_uses_the_direct_estimators_configured_value(self):
        search = HierarchicalGASearchCV(DryRunMultibranch(invalid_candidate_score=-5.0), param_grid={})
        search._initialise_rules(self.y, {})
        self.assertEqual(search._penalty(), -5.0)

    def test_bert_retains_branch_cnn_but_discards_other_backbone_knobs(self):
        params = {
            'estimator__temporal_architecture': 'bert',
            'estimator__branch_cnn_filters': 16,
            'estimator__branch_cnn_depth': 3,
            'estimator__cnn_depth': 99,
            'estimator__gru_units': 99,
            'estimator__phase_attention_units': 99,
            'estimator__branch_num_conv_layers': 99,
        }
        active = canonical_candidate(params, self.base)
        self.assertEqual(active['estimator__branch_cnn_filters'], 16)
        self.assertEqual(active['estimator__branch_cnn_depth'], 3)
        for name in ('cnn_depth', 'gru_units', 'phase_attention_units', 'branch_num_conv_layers'):
            self.assertNotIn('estimator__' + name, active)
        self.assertIsNone(candidate_error(params, self.base, y=self.y))

    def test_irrelevant_invalid_values_do_not_disqualify_bert(self):
        params = {'estimator__temporal_architecture': 'bert',
                  'estimator__cnn_depth': -10, 'estimator__n_phases': 0,
                  'estimator__phase_attention_temperature': 0.0}
        self.assertIsNone(candidate_error(params, self.base, y=self.y))

    def test_bert_shape_and_cls_requirements_are_checked(self):
        for extra in ({'bert_d_model': 30, 'bert_num_heads': 4},
                      {'bert_pooling': 'cls', 'bert_use_cls_token': False}):
            params = {'estimator__temporal_architecture': 'bert',
                      **{'estimator__' + key: value for key, value in extra.items()}}
            with self.subTest(extra=extra):
                self.assertTrue(candidate_error(params, self.base, y=self.y))

    def test_bert_attention_pooling_keeps_attention_options(self):
        params = {'estimator__temporal_architecture': 'bert',
                  'estimator__bert_pooling': 'attention',
                  'estimator__attention_type': 'additive', 'estimator__attention_units': 17}
        active = canonical_candidate(params, self.base)
        self.assertEqual(active['estimator__attention_units'], 17)
        self.assertEqual(active['estimator__attention_type'], 'additive')
        params['estimator__bert_pooling'] = 'avg'
        active = canonical_candidate(params, self.base)
        self.assertNotIn('estimator__attention_units', active)
        self.assertNotIn('estimator__attention_type', active)

    def test_aux_flags_activate_temporal_branch_encoder_on_legacy_backbone(self):
        params = {'estimator__temporal_architecture': 'bigru_attention',
                  'estimator__use_bigru': False, 'estimator__use_orientation_aux_head': True,
                  'estimator__branch_cnn_depth': 3, 'estimator__branch_num_conv_layers': 7}
        active = canonical_candidate(params, self.base)
        self.assertEqual(active['estimator__branch_cnn_depth'], 3)
        self.assertNotIn('estimator__branch_num_conv_layers', active)
        self.assertIsNone(candidate_error(params, self.base, y=self.y))

    def test_orientation_requires_labels_only_when_enabled(self):
        labels = self.y.drop(columns='orientation')
        self.assertIsNone(candidate_error({'estimator__use_orientation_aux_head': False}, self.base, y=labels))
        self.assertTrue(candidate_error({'estimator__use_orientation_aux_head': True}, self.base, y=labels))

    def test_phase_aux_requires_supervision_without_forbidding_bert(self):
        params = {'estimator__temporal_architecture': 'bert',
                  'estimator__use_phase_aux_head': True, 'estimator__n_phases': 3}
        self.assertTrue(candidate_error(params, self.base, y=self.y))
        self.assertIsNone(candidate_error(params, self.base, y=self.y, phase_labels_available=True))
        supervised = dict(params, estimator__phase_label_provider=lambda X, chunks: None)
        self.assertIsNone(candidate_error(supervised, self.base, y=self.y))

    def test_window_heads_validate_target_and_chunk_span(self):
        chunk_length = self.base['estimator__extractor__chunk_window_size']
        good = [{'name': 'short_head', 'target': 'action', 'window_size': 4}]
        missing = [{'name': 'short_head', 'target': 'missing', 'window_size': 4}]
        oversized = [{'name': 'long_head', 'target': 'action', 'window_size': chunk_length + 1}]
        self.assertIsNone(candidate_error({'estimator__custom_aux_heads': good}, self.base, y=self.y))
        for heads in (missing, oversized):
            with self.subTest(heads=heads):
                self.assertTrue(candidate_error({'estimator__custom_aux_heads': heads}, self.base, y=self.y))

    def test_disabled_aux_weights_do_not_create_distinct_candidates(self):
        params = {'estimator__use_orientation_aux_head': False,
                  'estimator__use_phase_aux_head': False,
                  'estimator__orientation_loss_weight': 0.01,
                  'estimator__phase_loss_weight': 0.99}
        active = canonical_candidate(params, self.base)
        self.assertNotIn('estimator__orientation_loss_weight', active)
        self.assertNotIn('estimator__phase_loss_weight', active)

    def test_cnn_branch_encoder_alias_has_one_canonical_representation(self):
        fixed = {'estimator__temporal_architecture': 'bert'}
        alias = canonical_candidate(dict(fixed, estimator__branch_encoder_type='cnn'), self.base)
        name = canonical_candidate(dict(fixed, estimator__branch_encoder_type='conv1d'), self.base)
        self.assertEqual(alias, name)


class HierarchicalGATests(unittest.TestCase):
    def setUp(self):
        FIT_LOG.clear()
        self.X, self.y = toy_data()

    def search(self, space, **kwargs):
        controls = dict(cv=2, population_size=4, tournament_size=2,
                        generations=1, scoring=candidate_score, refit=False,
                        verbose=0, random_state=23)
        controls.update(kwargs)
        return HierarchicalGASearchCV(dry_pipeline(), param_grid=space, **controls)

    def test_inactive_genes_share_cache_and_are_not_applied(self):
        search = self.search({
            'estimator__temporal_architecture': ECat(['bert']),
            'estimator__cnn_filters': ECat([17, 19, 23]),
            'estimator__orientation_loss_weight': ECat([0.1, 0.2]),
        }).fit(self.X, self.y)
        self.assertEqual(len(search.cv_results_['params']), 1)
        self.assertEqual(len(FIT_LOG), 2)
        self.assertGreater(search.reused_candidate_count_, 0)
        for effective in FIT_LOG:
            self.assertEqual(effective['cnn_filters'], mba.MultiBranchSequenceClassifier().cnn_filters)
            self.assertEqual(effective['orientation_loss_weight'], mba.MultiBranchSequenceClassifier().orientation_loss_weight)
        params = search.cv_results_['params'][0]
        self.assertNotIn('estimator__cnn_filters', params)

    def test_all_invalid_space_terminates_with_finite_penalty_without_fit(self):
        search = self.search({
            'estimator__temporal_architecture': ECat(['bert']),
            'estimator__bert_d_model': ECat([30]),
            'estimator__bert_num_heads': ECat([4]),
        }, generations=0, population_size=2).fit(self.X, self.y)
        self.assertEqual(FIT_LOG, [])
        self.assertTrue(np.isfinite(search.cv_results_['mean_test_score']).all())
        self.assertTrue((search.cv_results_['mean_test_score'] < 0).all())
        self.assertGreater(search.rejected_candidate_count_, 0)
        self.assertFalse(refit_valid_candidate(search, self.X, self.y))

    def test_invalid_initial_proposals_are_resampled(self):
        draws = iter([30, 30, 32, 32, 32, 32, 32, 32])
        class SometimesInvalid(EInt):
            def sample(self, rng=None):
                return next(draws, 32)
        search = self.search({'estimator__temporal_architecture': ECat(['bert']),
                              'estimator__bert_num_heads': ECat([4]),
                              'estimator__bert_d_model': SometimesInvalid(30, 32)}, generations=0)
        search.fit(self.X, self.y)
        self.assertGreater(search.rejected_candidate_count_, 0)
        self.assertTrue(all(params['bert_d_model'] == 32 for params in FIT_LOG))
        self.assertTrue((search.cv_results_['mean_test_score'] >= 0).all())

    def test_aux_off_on_choices_train_both_hierarchies_and_report_parameter_union(self):
        search = self.search({
            'estimator__temporal_architecture': ECat(['bigru_attention']),
            'estimator__use_bigru': ECat([False]),
            'estimator__use_orientation_aux_head': ECat([False, True]),
            'estimator__orientation_loss_weight': ECat([0.2]),
            'estimator__branch_num_conv_layers': ECat([2]),
            'estimator__branch_cnn_depth': ECat([3]),
        }, population_size=8, generations=0).fit(self.X, self.y)
        result = search.cv_results_
        self.assertEqual({params['estimator__use_orientation_aux_head'] for params in result['params']}, {False, True})
        self.assertIn('param_estimator__branch_num_conv_layers', result)
        self.assertIn('param_estimator__branch_cnn_depth', result)
        self.assertIn('param_estimator__orientation_loss_weight', result)
        for params in result['params']:
            if params['estimator__use_orientation_aux_head']:
                self.assertIn('estimator__branch_cnn_depth', params)
                self.assertNotIn('estimator__branch_num_conv_layers', params)
            else:
                self.assertIn('estimator__branch_num_conv_layers', params)
                self.assertNotIn('estimator__branch_cnn_depth', params)

    def test_validation_phase_labels_alone_do_not_satisfy_training_supervision(self):
        search = self.search({'estimator__use_phase_aux_head': ECat([True])})
        search._initialise_rules(self.y, {'estimator__validation_phase_labels': np.zeros((1, 2))})
        self.assertTrue(search._candidate_error({'estimator__use_phase_aux_head': True}))

    def test_classes_keep_cloneable_inherited_constructor(self):
        search = self.search({'estimator__use_orientation_aux_head': ECat([False, True])})
        self.assertEqual(clone(search).get_params(deep=False)['population_size'], 4)


class HierarchicalBayesTests(unittest.TestCase):
    def setUp(self):
        FIT_LOG.clear()
        self.X, self.y = toy_data()
        try:
            from skopt.space import Categorical
            self.Categorical = Categorical
        except ImportError:
            self.skipTest('scikit-optimize unavailable')

    def search(self, space, **kwargs):
        controls = dict(cv=2, n_iter=3, n_jobs=1, scoring=candidate_score,
                        refit=False, verbose=0, random_state=23)
        controls.update(kwargs)
        return HierarchicalBayesSearchCV(dry_pipeline(), search_spaces=space, **controls)

    def test_equivalent_valid_proposals_are_evaluated_once(self):
        search = self.search({
            'estimator__temporal_architecture': self.Categorical(['bert']),
            'estimator__cnn_filters': self.Categorical([17, 19, 23]),
        }).fit(self.X, self.y)
        self.assertEqual(len(FIT_LOG), 2)
        self.assertEqual(len(search.cv_results_['params']), 1)
        self.assertTrue(search.search_exhausted_)
        self.assertEqual(search.unique_candidate_count_, 1)
        self.assertNotIn('estimator__cnn_filters', search.conditional_search_spaces_[0])
        self.assertTrue(np.isfinite(search.optimizer_results_[0].func_vals).all())
        self.assertEqual(len(search.optimizer_results_[0].func_vals), 1)

    def test_all_invalid_proposals_finish_and_tell_finite_penalties(self):
        search = self.search({
            'estimator__temporal_architecture': self.Categorical(['bert']),
            'estimator__bert_d_model': self.Categorical([30]),
            'estimator__bert_num_heads': self.Categorical([4]),
        }, n_iter=1).fit(self.X, self.y)
        self.assertEqual(FIT_LOG, [])
        values = search.optimizer_results_[0].func_vals
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue((values > 0).all())
        self.assertGreater(search.rejected_candidate_count_, 0)
        self.assertLessEqual(len(values), 33)
        self.assertFalse(refit_valid_candidate(search, self.X, self.y))

    def test_invalid_proposal_is_told_penalty_then_resampled_before_fit(self):
        search = self.search({
            'estimator__temporal_architecture': self.Categorical(['bert']),
            'estimator__bert_d_model': self.Categorical([30, 32]),
            'estimator__bert_num_heads': self.Categorical([4]),
        }, n_iter=1)
        # Bayesian vector order is sorted parameter-name order.
        with patch('skopt.Optimizer.ask', return_value=[30, 4, 'bert']), \
                patch('skopt.space.Space.rvs', return_value=[[32, 4, 'bert']]):
            search.fit(self.X, self.y)
        self.assertEqual(len(FIT_LOG), 2)
        self.assertTrue(all(params['bert_d_model'] == 32 for params in FIT_LOG))
        values = search.optimizer_results_[0].func_vals
        self.assertEqual(len(values), 2)
        self.assertGreater(values[0], 0)
        self.assertLess(values[1], 0)

    def test_nonfinite_cv_score_is_told_as_finite_penalty(self):
        search = self.search({'estimator__temporal_architecture': self.Categorical(['bert'])}, n_iter=1)
        with patch.object(DryRunMultibranch, 'score', return_value=np.nan):
            search.fit(self.X, self.y)
        values = search.optimizer_results_[0].func_vals
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue((values > 0).all())

    def test_classes_keep_cloneable_inherited_constructor(self):
        search = self.search({'estimator__use_orientation_aux_head': self.Categorical([False, True])})
        self.assertEqual(clone(search).get_params(deep=False)['n_iter'], 3)


if __name__ == '__main__':
    unittest.main()
