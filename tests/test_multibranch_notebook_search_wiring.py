"""Keep notebook search wiring and saved diagnostics connected to the engine."""
import ast
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
LEGACY = ('notebooks/multibranch_alpha.ipynb',
          'archive/notebooks/multibranch-gesture-action-reguarlise.ipynb')
MODERN = ('notebooks/multibranch_v1.ipynb',
          'archive/notebooks/multibranch_v1_bayes_gpu.ipynb',
          'archive/notebooks/multibranch_v1_evo_gpu.ipynb')


def code_cells(relative):
    notebook = json.loads((ROOT / relative).read_text(encoding='utf-8'))
    return [''.join(cell.get('source', [])) for cell in notebook['cells']
            if cell['cell_type'] == 'code']


class NotebookConditionalSearchWiringTests(unittest.TestCase):
    def test_legacy_bayes_alias_uses_reloaded_conditional_class_and_optional_guard(self):
        for relative in LEGACY:
            with self.subTest(notebook=relative):
                sources = code_cells(relative)
                source = next(cell for cell in sources
                              if 'BayesSearchCV = multibranch_keras.' in cell)
                tree = ast.parse(source)
                reload_node = next(node for node in tree.body if isinstance(node, ast.Expr)
                                   and isinstance(node.value, ast.Call)
                                   and ast.unparse(node.value.func) == 'importlib.reload')
                alias = next(node for node in tree.body if isinstance(node, ast.Assign)
                             and any(isinstance(target, ast.Name) and target.id == 'BayesSearchCV'
                                     for target in node.targets))
                self.assertGreater(alias.lineno, reload_node.lineno)
                self.assertFalse(any(isinstance(node, ast.ImportFrom) and node.module == 'skopt'
                                     and any(name.name == 'BayesSearchCV' for name in node.names)
                                     for node in ast.walk(tree)))
                sentinel = object()
                for available in (True, False):
                    namespace = dict(SKOPT_AVAILABLE=available,
                                     multibranch_keras=SimpleNamespace(HierarchicalBayesSearchCV=sentinel))
                    exec(compile(ast.Module(body=[alias], type_ignores=[]), relative, 'exec'), namespace)
                    self.assertIs(namespace['BayesSearchCV'], sentinel if available else None)

    def test_modern_notebooks_write_actual_conditional_search_counters(self):
        for relative in MODERN:
            with self.subTest(notebook=relative), TemporaryDirectory() as directory:
                source = next(cell for cell in code_cells(relative)
                              if 'candidate_rule_metrics = {' in cell)
                tree = ast.parse(source)
                metric_nodes = [node for node in tree.body
                                if (isinstance(node, ast.Assign)
                                    and any(isinstance(target, ast.Name) and target.id == 'candidate_rule_metrics'
                                            for target in node.targets))
                                or (isinstance(node, ast.With) and '_candidate_rules.json' in ast.unparse(node))]
                self.assertEqual(len(metric_nodes), 2)
                code = compile(ast.Module(body=metric_nodes, type_ignores=[]), relative, 'exec')
                namespace = dict(search=SimpleNamespace(
                    cv_results_={'params': [{}, {}]}, unique_candidate_count_=7,
                    duplicate_proposal_count_=13, reused_candidate_count_=3,
                    rejected_candidate_count_=2, rejected_candidate_reasons_={'invalid shape': 2},
                    search_exhausted_=True, proposal_limit_reached_=True),
                    results_dir=Path(directory), experiment_name='test', timestamp='run', json=json)
                with redirect_stdout(StringIO()):
                    exec(code, namespace)
                saved = json.loads((Path(directory) / 'test_run_candidate_rules.json').read_text())
                self.assertEqual(saved, {
                    'unique_candidate_count': 7, 'duplicate_proposal_count': 13,
                    'reused_candidate_count': 3, 'rejected_candidate_count': 2,
                    'rejected_candidate_reasons': {'invalid shape': 2},
                    'search_exhausted': True, 'proposal_limit_reached': True,
                })
                namespace['search'] = SimpleNamespace(cv_results_={'params': [{}, {}, {}]})
                exec(code, namespace)
                self.assertEqual(namespace['candidate_rule_metrics']['unique_candidate_count'], 3)
                self.assertFalse(namespace['candidate_rule_metrics']['search_exhausted'])


if __name__ == '__main__':
    unittest.main()
