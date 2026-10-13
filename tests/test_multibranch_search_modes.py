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


class SearchModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        cls.sources = [''.join(c.get('source', [])) for c in notebook['cells'] if c['cell_type'] == 'code']
        namespace = {}
        exec(next(s for s in cls.sources if 'BEST_PARAM_PRESETS = ' in s), namespace)
        cls.presets = namespace['BEST_PARAM_PRESETS']

    def test_saved_presets_are_accepted_by_current_pipeline(self):
        for preset in self.presets.values():
            with self.subTest(source=preset['source_file']):
                configured = clone(make_pipeline()).set_params(**preset['params'])
                self.assertEqual(configured.get_params()['estimator__activation'], preset['params']['estimator__activation'])
                self.assertTrue(np.isfinite(preset['cv_score']))

    def test_best_mode_selects_one_exact_candidate_and_keeps_cv_settings(self):
        source = next(s for s in self.sources if 'search = GASearchCV(' in s)
        tree = ast.parse(source)
        construction = ast.Module(body=[tree.body[0]], type_ignores=[])
        for requested, expected in [('auto', '81'), ('80', '80'), ('81', '81')]:
            namespace = dict(search_mode='best_params', best_params_source=requested,
                             BEST_PARAM_PRESETS=self.presets, pipeline=make_pipeline(),
                             GridSearchCV=GridSearchCV, cv_object=3, n_jobs=1,
                             verbose=0, error_score='raise')
            with redirect_stdout(StringIO()):
                exec(compile(construction, str(NOTEBOOK), 'exec'), namespace)
            self.assertEqual(namespace['selected_source'], expected)
            search = namespace['search']
            self.assertEqual(list(ParameterGrid(search.param_grid)), [self.presets[expected]['params']])
            self.assertEqual(search.cv, 3)
            self.assertEqual(search.n_jobs, 1)
            self.assertTrue(search.refit)

    def test_invalid_preset_source_has_clear_error(self):
        source = next(s for s in self.sources if 'search = GASearchCV(' in s)
        construction = ast.Module(body=[ast.parse(source).body[0]], type_ignores=[])
        with self.assertRaisesRegex(ValueError, 'best_params_source'):
            exec(compile(construction, str(NOTEBOOK), 'exec'), dict(
                search_mode='best_params', best_params_source='missing', BEST_PARAM_PRESETS=self.presets))

    def test_best_params_alias_is_normalized(self):
        source = next(s for s in self.sources if 'experiment_name =' in s)
        tree = ast.parse(source)
        normalization = next(node for node in tree.body if isinstance(node, ast.If)
                             and "'bayes'" in ast.unparse(node.test))
        for alias in ('best params', 'best_params', 'best'):
            namespace = {'search_mode': alias}
            exec(compile(ast.Module(body=[normalization], type_ignores=[]), str(NOTEBOOK), 'exec'), namespace)
            self.assertEqual(namespace['search_mode'], 'best_params')

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
