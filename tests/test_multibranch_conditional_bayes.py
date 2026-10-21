"""Conditional Bayesian budgets count effective candidates, without Keras training."""
import unittest
from unittest.mock import patch

from test_multibranch_hierarchical_search import FIT_LOG, dry_pipeline, toy_data
from multibranch_utils import HierarchicalBayesSearchCV, candidate_score


class ConditionalBayesianTests(unittest.TestCase):
    def setUp(self):
        FIT_LOG.clear()
        self.X, self.y = toy_data()
        try:
            from skopt.space import Categorical
        except ImportError:
            self.skipTest('scikit-optimize unavailable')
        self.Categorical = Categorical

    def search(self, space, **controls):
        settings = dict(n_iter=3, cv=2, scoring=candidate_score, refit=False,
                        n_jobs=1, random_state=23)
        settings.update(controls)
        return HierarchicalBayesSearchCV(dry_pipeline(), search_spaces=space, **settings)

    def test_bert_preserves_requested_gru_domains_and_fit_values(self):
        cat = self.Categorical
        search = self.search({
            'estimator__temporal_architecture': cat(['bert']),
            'estimator__bert_d_model': cat([32, 64, 128]),
            'estimator__gru_units': cat([4, 8, 16]),
            'estimator__gru_layers': cat([1, 2, 3, 4]),
        }).fit(self.X, self.y)
        self.assertEqual(search.unique_candidate_count_, 3)
        self.assertEqual(len(FIT_LOG), 6)
        for space in search.conditional_search_spaces_:
            self.assertEqual(set(space['estimator__gru_units'].categories), {4, 8, 16})
            self.assertEqual(set(space['estimator__gru_layers'].categories), {1, 2, 3, 4})
        for params in FIT_LOG:
            self.assertIn(params['gru_units'], (4, 8, 16))
            self.assertIn(params['gru_layers'], (1, 2, 3, 4))

    def test_repeated_optimizer_points_do_not_consume_the_distinct_fit_budget(self):
        cat = self.Categorical
        search = self.search({
            'estimator__temporal_architecture': cat(['bert']),
            'estimator__bert_d_model': cat([32, 64, 128]),
            'estimator__bert_num_heads': cat([4]),
        })
        with patch('skopt.Optimizer.ask', return_value=[32, 4, 'bert']) as ask, \
                patch('skopt.space.Space.rvs', return_value=[[32, 4, 'bert']]):
            search.fit(self.X, self.y)
        self.assertEqual(search.unique_candidate_count_, 3)
        self.assertEqual(len(search.cv_results_['params']), 3)
        self.assertEqual(len(FIT_LOG), 6)
        self.assertGreater(search.duplicate_proposal_count_, 0)
        self.assertTrue(all(not call.args[1:] and not call.kwargs for call in ask.call_args_list))

    def test_architecture_subspaces_share_one_budget_and_keep_all_requested_values(self):
        cat = self.Categorical
        search = self.search({
            'estimator__temporal_architecture': cat(['bert', 'bigru_attention']),
            'estimator__use_bigru': cat([True]),
            'estimator__gru_units': cat([8, 16, 32]),
        }, n_iter=4).fit(self.X, self.y)
        self.assertEqual(search.requested_candidate_count_, 4)
        self.assertEqual(search.unique_candidate_count_, 4)
        self.assertEqual(len(FIT_LOG), 8)
        params = search.cv_results_['params']
        architectures = [p['estimator__temporal_architecture'] for p in params]
        self.assertEqual(architectures.count('bert'), 2)
        self.assertEqual(architectures.count('bigru_attention'), 2)
        self.assertTrue(all(p['estimator__use_bigru'] is True for p in params))
        self.assertTrue(all(p['estimator__gru_units'] in (8, 16, 32) for p in params))

    def test_callback_stops_the_entire_search_after_a_real_evaluation(self):
        cat = self.Categorical
        search = self.search({
            'estimator__temporal_architecture': cat(['bert', 'phase_attention']),
            'estimator__bert_d_model': cat([32, 64]),
            'estimator__cnn_filters': cat([16, 32]),
        }, n_iter=10)
        callbacks = []
        search.fit(self.X, self.y, callback=lambda result: callbacks.append(result) or True)
        self.assertEqual(len(callbacks), 1)
        self.assertEqual(search.unique_candidate_count_, 1)
        self.assertEqual(len(FIT_LOG), 2)

    def test_small_finite_space_stops_instead_of_replaying_the_budget(self):
        search = self.search({'estimator__temporal_architecture': self.Categorical(['bert'])}, n_iter=50)
        search.fit(self.X, self.y)
        self.assertTrue(search.search_exhausted_)
        self.assertEqual(len(FIT_LOG), 2)
        self.assertEqual(len(search.optimizer_results_[0].func_vals), 1)

    def test_surrogate_history_defaults_to_one_model_and_respects_explicit_override(self):
        space = {
            'estimator__temporal_architecture': self.Categorical(['bert']),
            'estimator__bert_d_model': self.Categorical([16, 24, 32, 40, 48, 56]),
        }
        for queue_size in (None, 3):
            with self.subTest(queue_size=queue_size):
                optimizer_kwargs = {'n_initial_points': 1, 'acq_optimizer': 'sampling',
                                    'acq_optimizer_kwargs': {'n_points': 32}}
                if queue_size is not None:
                    optimizer_kwargs['model_queue_size'] = queue_size
                original_kwargs = dict(optimizer_kwargs)
                search = self.search(space, n_iter=4, optimizer_kwargs=optimizer_kwargs)
                search.fit(self.X, self.y)
                optimizer = search.optimizers_[0]
                expected = 1 if queue_size is None else queue_size
                self.assertEqual(optimizer.max_model_queue_size, expected)
                self.assertEqual(len(optimizer.models), expected)
                self.assertEqual(optimizer_kwargs, original_kwargs)

    def test_exhausted_invalid_grid_tells_one_penalty_instead_of_fitting_a_large_surrogate(self):
        from skopt import Optimizer

        search = self.search({
            'estimator__temporal_architecture': self.Categorical(['bert']),
            'estimator__bert_d_model': self.Categorical([2 * index + 1 for index in range(80)]),
            'estimator__bert_num_heads': self.Categorical([4]),
        }, n_iter=1)
        tell_sizes = []
        original_tell = Optimizer.tell

        def record_tell(optimizer, points, scores, *args, **kwargs):
            tell_sizes.append(len(points))
            return original_tell(optimizer, points, scores, *args, **kwargs)

        with patch.object(Optimizer, 'tell', new=record_tell):
            search.fit(self.X, self.y)
        self.assertEqual(FIT_LOG, [])
        self.assertTrue(search.search_exhausted_)
        self.assertEqual(tell_sizes, [1])
        self.assertEqual(len(search.optimizer_results_[0].func_vals), 1)
        self.assertEqual(search.optimizers_[0].models, [])
        self.assertEqual(len(search.cv_results_['params']), 1)


if __name__ == '__main__':
    unittest.main()
