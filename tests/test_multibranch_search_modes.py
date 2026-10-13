"""Validate saved-parameter selection and GA controls without training models."""
import ast
import json
import unittest
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, ParameterGrid

from test_multibranch_attention_sol_params import NOTEBOOK, GASearchCV, make_pipeline
from multibranch_experiment_utils import candidate_score


class SearchModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        cls.sources = [''.join(c.get('source', [])) for c in notebook['cells'] if c['cell_type'] == 'code']
        namespace = {}
        exec(next(s for s in cls.sources if 'BEST_PARAMS = ' in s), namespace)
        cls.best_params = namespace['BEST_PARAMS']

    def test_best_params_are_accepted_by_current_pipeline(self):
        configured = clone(make_pipeline()).set_params(**self.best_params)
        self.assertEqual(configured.get_params()['estimator__activation'], self.best_params['estimator__activation'])
        self.assertFalse(any('best_params_source' in s or 'BEST_PARAM_PRESETS' in s for s in self.sources))

    def test_best_mode_selects_one_exact_candidate_and_keeps_cv_settings(self):
        source = next(s for s in self.sources if 'search = HierarchicalGASearchCV(' in s)
        tree = ast.parse(source)
        construction = ast.Module(body=[tree.body[0]], type_ignores=[])
        namespace = dict(search_mode='best params', BEST_PARAMS=self.best_params,
                         pipeline=make_pipeline(), GridSearchCV=GridSearchCV,
                         cv_object=3, n_jobs=1, verbose=0, error_score='raise', candidate_score=candidate_score)
        with redirect_stdout(StringIO()):
            exec(compile(construction, str(NOTEBOOK), 'exec'), namespace)
        search = namespace['search']
        self.assertEqual(list(ParameterGrid(search.param_grid)), [self.best_params])
        self.assertEqual(search.cv, 3)
        self.assertEqual(search.n_jobs, 1)
        self.assertFalse(search.refit)

    def test_exactly_four_search_modes_are_supported(self):
        source = next(s for s in self.sources if 'experiment_name =' in s)
        tree = ast.parse(source)
        normalization = next(node for node in tree.body if isinstance(node, ast.If)
                             and 'search_mode' in ast.unparse(node.test))
        for mode in ('best params', 'evo', 'bayes', 'grid'):
            namespace = {'search_mode': mode}
            exec(compile(ast.Module(body=[normalization], type_ignores=[]), str(NOTEBOOK), 'exec'), namespace)
            self.assertEqual(namespace['search_mode'], mode)
        with self.assertRaisesRegex(ValueError, 'search_mode must be'):
            exec(compile(ast.Module(body=[normalization], type_ignores=[]), str(NOTEBOOK), 'exec'), {'search_mode': 'invalid'})

    def test_ga_rejects_list_controls_before_training(self):
        for controls in ({'population_size': [5, 10, 15]}, {'generations': -1},
                         {'tournament_size': 11}, {'mutation_probability': 1.1}):
            with self.subTest(controls=controls), self.assertRaises(ValueError):
                GASearchCV(estimator=make_pipeline(), **controls).fit(np.zeros((4, 1)), np.zeros(4))

    def test_ga_exposes_only_implemented_controls(self):
        params = GASearchCV(estimator=make_pipeline()).get_params(deep=False)
        self.assertFalse({'algorithm', 'keep_top_k', 'n_jobs'} & set(params))


if __name__ == '__main__':
    unittest.main()
