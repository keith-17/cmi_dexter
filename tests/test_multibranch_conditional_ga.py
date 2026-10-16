"""Exercise conditional evolutionary proposals without constructing Keras models."""
import unittest
from unittest.mock import patch

import numpy as np

from test_multibranch_hierarchical_search import FIT_LOG, dry_pipeline, toy_data
from test_multibranch_utils_params import ECat, EInt
from base_utils import prepare_evolutionary_space
from multibranch_utils import HierarchicalGASearchCV, candidate_score


class CountingInt(EInt):
    def __init__(self, lower, upper):
        super().__init__(lower, upper)
        self.draws = 0

    def sample(self, rng=None):
        self.draws += 1
        return super().sample(rng)


class ConditionalGATests(unittest.TestCase):
    def setUp(self):
        FIT_LOG.clear()
        self.X, self.y = toy_data()

    def search(self, space, **kwargs):
        options = dict(cv=2, population_size=4, generations=1,
                       tournament_size=2, scoring=candidate_score, refit=False,
                       verbose=0, random_state=23)
        options.update(kwargs)
        return HierarchicalGASearchCV(dry_pipeline(), param_grid=space, **options)

    def initialise(self, search):
        search._initialise_rules(self.y, {})
        return np.random.default_rng(23)

    def test_inactive_gru_and_pooling_genes_are_never_drawn(self):
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
        self.assertEqual((gru.draws, cnn.draws, attention.draws), (0, 0, 0))
        self.assertGreater(layers.draws, 0)
        self.assertEqual({p['estimator__bert_num_layers']
                          for p in search.cv_results_['params']}, {1, 2})
        self.assertEqual(search.unique_candidate_count_, 2)
        self.assertEqual(len(FIT_LOG), 4)
        self.assertTrue(search.search_exhausted_)

    def test_architecture_mutation_initialises_newly_active_genes(self):
        bert = CountingInt(2, 2)
        gru = CountingInt(16, 16)
        search = self.search({}, mutation_probability=1.0)
        rng = self.initialise(search)
        original = {'estimator__temporal_architecture': 'bert',
                    'estimator__bert_num_layers': 5,
                    'estimator__use_bigru': False}
        gru_space = prepare_evolutionary_space({
            'estimator__temporal_architecture': ECat(['bigru_attention']),
            'estimator__use_bigru': ECat([True]),
            'estimator__bert_num_layers': bert,
            'estimator__gru_units': gru,
        })
        mutated = search._mutate(original, gru_space, rng)
        self.assertEqual(mutated['estimator__gru_units'], 16)
        self.assertTrue(mutated['estimator__use_bigru'])
        self.assertNotIn('estimator__bert_num_layers', mutated)
        self.assertEqual((bert.draws, gru.draws), (0, 1))
        bert_space = dict(gru_space,
                          estimator__temporal_architecture=ECat(['bert']))
        mutated = search._mutate(mutated, bert_space, rng)
        self.assertEqual(mutated['estimator__bert_num_layers'], 2)
        self.assertNotIn('estimator__gru_units', mutated)
        self.assertNotIn('estimator__use_bigru', mutated)
        self.assertEqual((bert.draws, gru.draws), (1, 1))

    def test_aux_switch_initialises_temporal_branch_and_removes_legacy_genes(self):
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
        self.assertNotIn('estimator__branch_num_conv_layers', mutated)
        self.assertEqual((branch.draws, legacy.draws), (1, 0))

    def test_pooling_switches_are_resolved_before_dependent_genes(self):
        attention = CountingInt(13, 13)
        post = CountingInt(17, 17)
        se = CountingInt(8, 8)
        search = self.search({}, mutation_probability=1.0)
        rng = self.initialise(search)
        # Deliberately put dependent descriptors first in user input.
        space = prepare_evolutionary_space({
            'estimator__post_fusion_attention_units': post,
            'estimator__attention_units': attention,
            'estimator__se_ratio': se,
            'estimator__temporal_architecture': ECat(['cnn_attention_pool']),
            'estimator__post_fusion_pooling': ECat(['avg']),
            'estimator__use_se': ECat([False]),
        })
        proposal = search._sample_individual(space, rng)
        self.assertNotIn('estimator__post_fusion_attention_units', proposal)
        self.assertNotIn('estimator__attention_units', proposal)
        self.assertNotIn('estimator__se_ratio', proposal)
        self.assertEqual((attention.draws, post.draws, se.draws), (0, 0, 0))

    def test_disabled_tof_modes_skip_encoder_and_spatial_genes(self):
        spatial = CountingInt(4, 4)
        search = self.search({})
        rng = self.initialise(search)
        space = prepare_evolutionary_space({
            'estimator__tof_spatial_filters': spatial,
            'estimator__tof_encoder_type': ECat(['spatial_cnn']),
            'estimator__extractor__tof_modes': ECat(['']),
            'estimator__temporal_architecture': ECat(['bert']),
        })
        proposal = search._sample_individual(space, rng)
        self.assertNotIn('estimator__tof_encoder_type', proposal)
        self.assertNotIn('estimator__tof_spatial_filters', proposal)
        self.assertEqual(spatial.draws, 0)

    def test_crossover_accepts_sparse_parents_of_different_architectures(self):
        search = self.search({}, crossover_probability=1.0)
        rng = self.initialise(search)
        bert = {'estimator__temporal_architecture': 'bert',
                'estimator__bert_num_layers': 2}
        gru = {'estimator__temporal_architecture': 'bigru_attention',
               'estimator__use_bigru': True, 'estimator__gru_units': 16}
        for _ in range(20):
            for child in search._crossover(bert, gru, rng):
                if child['estimator__temporal_architecture'] == 'bert':
                    self.assertIn('estimator__bert_num_layers', child)
                    self.assertNotIn('estimator__gru_units', child)
                else:
                    self.assertIn('estimator__gru_units', child)
                    self.assertNotIn('estimator__bert_num_layers', child)

    def test_bounded_finite_fallback_finds_unseen_active_alternatives(self):
        search = self.search({}, mutation_probability=0.0)
        rng = self.initialise(search)
        space = prepare_evolutionary_space({
            'estimator__temporal_architecture': ECat(['bert']),
            'estimator__bert_num_layers': ECat([1, 2]),
            'estimator__gru_units': EInt(1, 1000000),
        })
        repeated = {'estimator__temporal_architecture': 'bert',
                    'estimator__bert_num_layers': 1}
        with patch.object(search, '_conditional_proposal', return_value=repeated):
            first = search._sample_individual(space, rng)
            second = search._sample_individual(space, rng)
        self.assertEqual(first['estimator__bert_num_layers'], 1)
        self.assertEqual(second['estimator__bert_num_layers'], 2)
        self.assertLess(search.duplicate_proposal_count_, 2 * search._proposal_attempts)
        self.assertFalse(search.search_exhausted_)
        with patch.object(search, '_conditional_proposal', return_value=repeated):
            search._sample_individual(space, rng)
        self.assertTrue(search.search_exhausted_)

    def test_continuous_retry_limit_does_not_claim_finite_exhaustion(self):
        from base_utils import EFloat
        search = self.search({})
        rng = self.initialise(search)
        space = prepare_evolutionary_space({
            'estimator__temporal_architecture': ECat(['bert']),
            'estimator__learning_rate': EFloat(0.001, 0.01),
        })
        repeated = {'estimator__temporal_architecture': 'bert',
                    'estimator__learning_rate': 0.005}
        with patch.object(search, '_conditional_proposal', return_value=repeated):
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
        self.assertLessEqual(search.unique_candidate_count_,
                             search.population_size * (search.generations + 1))

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
