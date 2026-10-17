"""Real Bayesian scheduling and durable partial results, without Keras fitting."""
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from test_multibranch_hierarchical_search import (
    FIT_LOG, DryRunMultibranch, dry_pipeline, toy_data,
)
from multibranch_utils import (
    HierarchicalBayesSearchCV, candidate_score, conditional_bayesian_spaces,
)


SOURCE_KEY = 'estimator__extractor__pretraining_source'
SOURCES = ['non_target', 'all', 'target', 'baseline']


@unittest.skipIf(HierarchicalBayesSearchCV is None, 'scikit-optimize unavailable')
class LargePretrainingBayesTests(unittest.TestCase):
    def setUp(self):
        from skopt.space import Categorical
        self.Categorical = Categorical
        FIT_LOG.clear()
        self.X, self.y = toy_data()

    def space(self, include_sources=True):
        space = {
            'estimator__temporal_architecture': self.Categorical(['bert']),
            'estimator__bert_d_model': self.Categorical(list(range(32, 513, 32))),
            'estimator__bert_num_heads': self.Categorical([4]),
        }
        if include_sources:
            space[SOURCE_KEY] = self.Categorical(SOURCES)
        return space

    def search(self, iterations=37, space=None):
        return HierarchicalBayesSearchCV(
            dry_pipeline(), search_spaces=self.space() if space is None else space,
            n_iter=iterations, cv=[(list(range(4)), list(range(4, 8)))],
            n_jobs=1, random_state=42, scoring=candidate_score,
            refit=False, return_train_score=True, error_score='raise',
            optimizer_kwargs={'n_initial_points': 3, 'acq_optimizer': 'sampling',
                              'acq_optimizer_kwargs': {'n_points': 256}},
        )

    def test_37_candidates_cover_sources_evenly_and_update_each_surrogate(self):
        search = self.search()

        def score(estimator, X, y=None):
            # Give the actual skopt surrogate a learnable, nonconstant objective.
            source = estimator.extractor.pretraining_source
            return 0.5 + SOURCES.index(source) / 100 + estimator.bert_d_model / 10000

        with TemporaryDirectory() as directory:
            path = Path(directory) / 'partial_cv.csv'
            search.candidate_checkpoint_path = path
            with patch.object(DryRunMultibranch, 'score', new=score):
                search.fit(self.X, self.y)
            checkpoint = pd.read_csv(path)
            self.assertFalse(path.with_name(path.name + '.tmp').exists())
        params = search.cv_results_['params']
        sources = [candidate[SOURCE_KEY] for candidate in params]
        self.assertEqual(search.requested_candidate_count_, 37)
        self.assertEqual(search.unique_candidate_count_, 37)
        self.assertEqual(len(params), 37)
        self.assertEqual(len(FIT_LOG), 37)
        self.assertEqual(set(sources[:4]), set(SOURCES))
        self.assertEqual(sorted(Counter(sources).values()), [9, 9, 9, 10])
        self.assertEqual(len({(candidate[SOURCE_KEY], candidate['estimator__bert_d_model'])
                              for candidate in params}), 37)
        self.assertEqual(len(search.optimizers_), 4)
        for optimizer in search.optimizers_:
            self.assertGreater(len(optimizer.yi), 3)
            self.assertEqual(len(optimizer.models), 1)
        self.assertEqual(search.candidate_checkpoint_count_, 37)
        self.assertEqual(len(checkpoint), 37)
        self.assertEqual(checkpoint['param_' + SOURCE_KEY].tolist(), sources)
        self.assertIn('mean_train_score', checkpoint)
        self.assertIn('mean_test_score', checkpoint)
        self.assertTrue(checkpoint['mean_test_score'].notna().all())

    def test_callback_stop_preserves_every_completed_candidate(self):
        search = self.search()
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'partial_cv.csv'
            search.candidate_checkpoint_path = path
            search.fit(self.X, self.y,
                       callback=lambda result: search.unique_candidate_count_ >= 2)
            checkpoint = pd.read_csv(path)
        self.assertEqual(len(search.cv_results_['params']), 2)
        self.assertEqual(len(checkpoint), 2)
        self.assertEqual(search.candidate_checkpoint_count_, 2)

    def test_five_candidate_pilot_covers_all_four_sources(self):
        search = self.search(iterations=5).fit(self.X, self.y)
        sources = [params[SOURCE_KEY] for params in search.cv_results_['params']]
        self.assertEqual(set(sources[:4]), set(SOURCES))
        self.assertEqual(sorted(Counter(sources).values()), [1, 1, 1, 2])
        self.assertIsNone(search.candidate_checkpoint_path)
        self.assertFalse(hasattr(search, 'candidate_checkpoint_count_'))

    def test_ordinary_and_fixed_source_spaces_keep_one_optimizer(self):
        ordinary = self.space(include_sources=False)
        self.assertEqual(len(conditional_bayesian_spaces(ordinary)), 1)
        search = self.search(iterations=2, space=ordinary).fit(self.X, self.y)
        self.assertEqual(len(search.optimizers_), 1)
        self.assertEqual(search.unique_candidate_count_, 2)
        fixed = dict(ordinary, **{SOURCE_KEY: self.Categorical(['target'])})
        compiled = conditional_bayesian_spaces(fixed)
        self.assertEqual(len(compiled), 1)
        self.assertEqual(compiled[0][SOURCE_KEY].categories, ('target',))


if __name__ == '__main__':
    unittest.main()
