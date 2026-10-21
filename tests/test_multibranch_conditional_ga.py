"""Evolutionary searches retain the parameters the caller configured."""
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from test_multibranch_utils_params import mba, ECat, EInt, make_pipeline
from base_utils import prepare_evolutionary_space
from multibranch_utils import HierarchicalGASearchCV, candidate_score


PROPOSAL_FIT_LOG = []


class RecordingMultibranch(mba.MultiBranchSequenceClassifier):
    """Record the real estimator's fitted parameter contract without using a GPU."""

    def fit(self, X, y=None, **fit_params):
        self.fit_failed_ = False
        PROPOSAL_FIT_LOG.append(self.get_params(deep=True))
        return self

    def score(self, X, y=None):
        return 0.75


def recording_pipeline():
    real = make_pipeline().named_steps['estimator']
    return Pipeline([('estimator', RecordingMultibranch(**real.get_params(deep=False)))])


def proposal_data():
    return np.zeros((8, 1)), pd.DataFrame({
        'sequence_id': list(range(8)), 'bfrb': ['a', 'b'] * 4,
        'orientation': ['upright'] * 8, 'action': ['scratch'] * 8,
    })


class CountingInt(EInt):
    def __init__(self, lower, upper):
        super().__init__(lower, upper)
        self.draws = 0

    def sample(self, rng=None):
        self.draws += 1
        return super().sample(rng)


class ConditionalGATests(unittest.TestCase):
    def setUp(self):
        PROPOSAL_FIT_LOG.clear()
        self.X, self.y = proposal_data()

    def search(self, space, **kwargs):
        options = dict(cv=2, population_size=4, generations=1,
                       tournament_size=2, scoring=candidate_score, refit=False,
                       verbose=0, random_state=23)
        options.update(kwargs)
        return HierarchicalGASearchCV(recording_pipeline(), param_grid=space, **options)

    def initialise(self, search):
        search._initialise_rules(self.y, {})
        return np.random.default_rng(23)

    def assert_candidates_were_fitted(self, search):
        """Every reported parameter reaches both CV estimators unchanged."""
        self.assertEqual(len(PROPOSAL_FIT_LOG), 2 * len(search.cv_results_['params']))
        for candidate in search.cv_results_['params']:
            matches = [fitted for fitted in PROPOSAL_FIT_LOG if all(
                fitted[key.removeprefix('estimator__')] == value
                for key, value in candidate.items()
            )]
            self.assertEqual(len(matches), 2, candidate)

    def test_configured_genes_are_drawn_and_reach_fitted_estimators(self):
        gru = CountingInt(1, 1000000)
        cnn = CountingInt(1, 1000000)
        attention = CountingInt(1, 1000000)
        layers = CountingInt(1, 2)
        search = self.search({
            'estimator__temporal_architecture': ECat(['bert']),
            'estimator__bert_pooling': ECat(['avg']),
            'estimator__bert_num_layers': layers,
            'estimator__gru_units': gru,
            'estimator__cnn_filters': cnn,
            'estimator__attention_units': attention,
        }).fit(self.X, self.y)
        for descriptor in (gru, cnn, attention, layers):
            self.assertGreater(descriptor.draws, 0)
        for candidate in search.cv_results_['params']:
            self.assertEqual(set(candidate), set(search.param_grid))
        self.assert_candidates_were_fitted(search)

    def test_architecture_mutation_keeps_and_samples_every_configured_gene(self):
        bert = CountingInt(2, 2)
        gru = CountingInt(16, 16)
        search = self.search({}, mutation_probability=1.0)
        rng = self.initialise(search)
        original = {'estimator__temporal_architecture': 'bert',
                    'estimator__bert_num_layers': 5,
                    'estimator__use_bigru': False}
        space = prepare_evolutionary_space({
            'estimator__temporal_architecture': ECat(['bigru_attention']),
            'estimator__use_bigru': ECat([True]),
            'estimator__bert_num_layers': bert,
            'estimator__gru_units': gru,
        })
        mutated = search._mutate(original, space, rng)
        self.assertEqual(mutated['estimator__bert_num_layers'], 2)
        self.assertEqual(mutated['estimator__gru_units'], 16)
        self.assertTrue(mutated['estimator__use_bigru'])
        self.assertEqual((bert.draws, gru.draws), (1, 1))
        space['estimator__temporal_architecture'] = ECat(['bert'])
        mutated = search._mutate(mutated, space, rng)
        self.assertEqual(mutated['estimator__bert_num_layers'], 2)
        self.assertEqual(mutated['estimator__gru_units'], 16)
        self.assertTrue(mutated['estimator__use_bigru'])
        self.assertEqual((bert.draws, gru.draws), (2, 2))

    def test_auxiliary_switch_does_not_remove_legacy_genes(self):
        branch = CountingInt(3, 3)
        legacy = CountingInt(2, 2)
        search = self.search({}, mutation_probability=1.0)
        rng = self.initialise(search)
        space = prepare_evolutionary_space({
            'estimator__temporal_architecture': ECat(['bigru_attention']),
            'estimator__use_bigru': ECat([False]),
            'estimator__use_orientation_aux_head': ECat([True]),
            'estimator__branch_cnn_depth': branch,
            'estimator__branch_num_conv_layers': legacy,
        })
        mutated = search._mutate({
            'estimator__temporal_architecture': 'bigru_attention',
            'estimator__use_bigru': False,
            'estimator__use_orientation_aux_head': False,
            'estimator__branch_num_conv_layers': 7,
        }, space, rng)
        self.assertEqual(mutated['estimator__branch_cnn_depth'], 3)
        self.assertEqual(mutated['estimator__branch_num_conv_layers'], 2)
        self.assertEqual((branch.draws, legacy.draws), (1, 1))

    def test_pooling_and_se_switches_do_not_suppress_declared_descriptors(self):
        attention = CountingInt(13, 13)
        post = CountingInt(17, 17)
        se = CountingInt(8, 8)
        search = self.search({})
        rng = self.initialise(search)
        space = prepare_evolutionary_space({
            'estimator__post_fusion_attention_units': post,
            'estimator__attention_units': attention,
            'estimator__se_ratio': se,
            'estimator__temporal_architecture': ECat(['cnn_attention_pool']),
            'estimator__post_fusion_pooling': ECat(['avg']),
            'estimator__use_se': ECat([False]),
        })
        proposal = search._sample_individual(space, rng)
        self.assertEqual(proposal['estimator__post_fusion_attention_units'], 17)
        self.assertEqual(proposal['estimator__attention_units'], 13)
        self.assertEqual(proposal['estimator__se_ratio'], 8)
        self.assertEqual((attention.draws, post.draws, se.draws), (1, 1, 1))

    def test_disabled_tof_keeps_configured_encoder_and_spatial_genes(self):
        spatial = CountingInt(4, 4)
        search = self.search({})
        rng = self.initialise(search)
        space = prepare_evolutionary_space({
            'estimator__tof_spatial_filters': spatial,
            'estimator__tof_encoder_type': ECat(['spatial_cnn']),
            'estimator__extractor__tof_modes': ECat(['']),
            'estimator__temporal_architecture': ECat(['bert']),
        })
        proposal = search._parameter_proposal(space, rng)
        self.assertEqual(proposal['estimator__tof_encoder_type'], 'spatial_cnn')
        self.assertEqual(proposal['estimator__tof_spatial_filters'], 4)
        self.assertEqual(spatial.draws, 1)

    def test_crossover_retains_the_union_of_sparse_parent_parameters(self):
        search = self.search({}, crossover_probability=1.0)
        rng = self.initialise(search)
        bert = {'estimator__temporal_architecture': 'bert',
                'estimator__bert_num_layers': 2}
        gru = {'estimator__temporal_architecture': 'bigru_attention',
               'estimator__use_bigru': True, 'estimator__gru_units': 16}
        for _ in range(20):
            for child in search._crossover(bert, gru, rng):
                self.assertEqual(set(child), set(bert) | set(gru))
                self.assertEqual(child['estimator__bert_num_layers'], 2)
                self.assertEqual(child['estimator__gru_units'], 16)
                self.assertTrue(child['estimator__use_bigru'])

    def test_static_branch_depth_choices_are_distinct_and_fitted_unchanged(self):
        depths = [{'acc': 4, 'stft': 1, 'cwt': 1},
                  {'acc': 4, 'stft': 4, 'cwt': 4}]
        search = self.search(prepare_evolutionary_space({
            'estimator__temporal_architecture': ['bigru_attention'],
            'estimator__use_bigru': [False],
            'estimator__branch_num_conv_layers': depths,
        })).fit(self.X, self.y)
        self.assertEqual(search.unique_candidate_count_, 2)
        fitted = {json.loads(row['branch_num_conv_layers'])['stft']
                  for row in PROPOSAL_FIT_LOG}
        self.assertEqual(fitted, {1, 4})
        self.assert_candidates_were_fitted(search)

    def test_spectral_fields_survive_sampling_cache_and_fit(self):
        configs = [[{'name': 'custom', 'window_seconds': 1.0,
                     'nperseg': count, 'acc_modes': 'raw', 'scaling': 'density'}]
                   for count in (8, 32)]
        cwt = [{'name': 'wave', 'widths': [1, 2], 'n_scales': 8,
                'min_scale': 1, 'acc_modes': 'jerk'}]
        search = self.search(prepare_evolutionary_space({
            'estimator__extractor__stft_configs': configs,
            'estimator__extractor__cwt_configs': [cwt],
        })).fit(self.X, self.y)
        self.assertEqual(search.unique_candidate_count_, 2)
        self.assertEqual({json.loads(row['extractor__stft_configs'])[0]['nperseg']
                          for row in PROPOSAL_FIT_LOG}, {8, 32})
        for row in PROPOSAL_FIT_LOG:
            self.assertEqual(json.loads(row['extractor__cwt_configs']), cwt)
            self.assertEqual(json.loads(row['extractor__stft_configs'])[0]['scaling'], 'density')
            self.assertEqual(json.loads(row['extractor__stft_configs'])[0]['acc_modes'], 'raw')
        self.assert_candidates_were_fitted(search)

    def test_explicit_unused_switch_values_remain_distinct_fitted_candidates(self):
        search = self.search(prepare_evolutionary_space({
            'estimator__temporal_architecture': ['bert'],
            'estimator__use_bigru': [False, True],
            'estimator__gru_units': [16],
        })).fit(self.X, self.y)
        self.assertEqual(search.unique_candidate_count_, 2)
        self.assertEqual({row['use_bigru'] for row in PROPOSAL_FIT_LOG}, {False, True})
        self.assert_candidates_were_fitted(search)

    def test_bounded_finite_fallback_finds_unseen_full_parameter_choices(self):
        search = self.search({}, mutation_probability=0.0)
        rng = self.initialise(search)
        space = prepare_evolutionary_space({
            'estimator__temporal_architecture': ['bert'],
            'estimator__bert_num_layers': [1, 2],
            'estimator__gru_units': [16],
        })
        repeated = {'estimator__temporal_architecture': 'bert',
                    'estimator__bert_num_layers': 1, 'estimator__gru_units': 16}
        with patch.object(search, '_parameter_proposal', return_value=repeated):
            first = search._sample_individual(space, rng)
            second = search._sample_individual(space, rng)
        self.assertEqual(first['estimator__bert_num_layers'], 1)
        self.assertEqual(second['estimator__bert_num_layers'], 2)
        self.assertEqual(second['estimator__gru_units'], 16)
        self.assertFalse(search.search_exhausted_)
        with patch.object(search, '_parameter_proposal', return_value=repeated):
            search._sample_individual(space, rng)
        self.assertTrue(search.search_exhausted_)

    def test_continuous_retry_limit_does_not_claim_finite_exhaustion(self):
        from base_utils import EFloat
        search = self.search({})
        rng = self.initialise(search)
        space = prepare_evolutionary_space({
            'estimator__temporal_architecture': ['bert'],
            'estimator__learning_rate': EFloat(0.001, 0.01),
        })
        repeated = {'estimator__temporal_architecture': 'bert',
                    'estimator__learning_rate': 0.005}
        with patch.object(search, '_parameter_proposal', return_value=repeated):
            search._sample_individual(space, rng)
            search._sample_individual(space, rng)
        self.assertFalse(getattr(search, 'search_exhausted_', False))
        self.assertTrue(search.proposal_limit_reached_)

    def test_search_counters_reset_between_fits(self):
        search = self.search({
            'estimator__temporal_architecture': ECat(['bert']),
            'estimator__bert_num_layers': ECat([1, 2]),
        })
        search.fit(self.X, self.y)
        first_counts = (search.unique_candidate_count_, search.duplicate_proposal_count_)
        search.fit(self.X, self.y)
        self.assertEqual((search.unique_candidate_count_, search.duplicate_proposal_count_), first_counts)
        self.assertEqual(search.unique_candidate_count_, len(search.cv_results_['params']))

    def test_fit_releases_label_validation_and_enumeration_state(self):
        search = self.search({
            'estimator__temporal_architecture': ECat(['bert']),
            'estimator__use_orientation_aux_head': ECat([True]),
            'estimator__bert_num_layers': ECat([1]),
        }).fit(self.X, self.y)
        self.assertIsNone(search._rules_y_)
        self.assertEqual(search._label_validation_cache_, {})
        self.assertIsNone(search._finite_candidates_)
        self.assertEqual(search.unique_candidate_count_, 1)

    def test_failed_fit_also_releases_validation_state(self):
        search = self.search({}, population_size=0)
        with self.assertRaises(ValueError):
            search.fit(self.X, self.y)
        self.assertIsNone(search._rules_y_)
        self.assertEqual(search._label_validation_cache_, {})
        self.assertIsNone(search._finite_candidates_)


if __name__ == '__main__':
    unittest.main()
