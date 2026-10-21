"""Check active and generated notebooks delegate their two saved artifacts."""
import ast
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.create_large_pretraining_notebooks import build_notebook
from scripts.pretraining_search_configuration import configure_result_outputs


def source(notebook, index):
    return ''.join(notebook['cells'][index]['source'])


class PretrainingNotebookOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebooks = {
            name: json.loads((ROOT / 'notebooks' / name).read_text(encoding='utf-8'))
            for name in ('multibranch_pretraining.ipynb', 'multibranch_ssl.ipynb')
        }
        cls.notebooks.update({f'generated-model-{model}': build_notebook(model)
                              for model in (1, 2)})

    def test_notebooks_delegate_all_file_writing_to_search_runner(self):
        for name, notebook in self.notebooks.items():
            with self.subTest(notebook=name):
                trees = [ast.parse(''.join(cell['source'])) for cell in notebook['cells']
                         if cell['cell_type'] == 'code']
                calls = [node for tree in trees for node in ast.walk(tree)
                         if isinstance(node, ast.Call)]
                for call in calls:
                    if isinstance(call.func, ast.Attribute):
                        self.assertNotIn(call.func.attr, {
                            'dump', 'to_csv', 'save', 'savefig', 'mkdir',
                            'write_text', 'write_bytes',
                        })
                    elif isinstance(call.func, ast.Name):
                        self.assertNotEqual(call.func.id, 'open')
                runners = [call for call in calls if isinstance(call.func, ast.Name)
                           and call.func.id == 'run_validation_searches']
                self.assertEqual(len(runners), 1)
                self.assertNotIn('quick_save', {keyword.arg for keyword in runners[0].keywords})

    def test_final_cell_reports_only_the_two_saved_artifacts(self):
        for name, notebook in self.notebooks.items():
            with self.subTest(notebook=name):
                output = StringIO()
                paths = {'experiments_path': Path('run/experiment_experiments.csv'),
                         'summary_path': Path('run/experiment_summary.json')}
                with redirect_stdout(output):
                    exec(compile(source(notebook, 24), name, 'exec'),
                         {'validation_runs': paths})
                self.assertEqual(output.getvalue().splitlines(), [
                    f"Experiments CSV: {paths['experiments_path']}",
                    f"Summary JSON: {paths['summary_path']}",
                ])

    def test_orientation_filter_keeps_exclusions_in_memory_without_files(self):
        frame = pd.DataFrame({
            'sequence_id': ['missing', 'retained', 'retained'],
            'rot_w': [np.nan, np.nan, 1.0],
            'rot_x': [np.nan, np.nan, 0.0],
            'rot_y': [np.nan, np.nan, 0.0],
            'rot_z': [np.nan, np.nan, 0.0],
        })
        for name, notebook in self.notebooks.items():
            with self.subTest(notebook=name), TemporaryDirectory() as directory:
                statements = ast.parse(source(notebook, 7)).body
                start = next(index for index, node in enumerate(statements)
                             if isinstance(node, ast.Assign)
                             and any(isinstance(target, ast.Name) and target.id == 'quaternion_cols'
                                     for target in node.targets))
                end = next(index for index in range(start, len(statements))
                           if isinstance(statements[index], ast.Delete))
                code = compile(ast.Module(body=statements[start:end + 1], type_ignores=[]),
                               name, 'exec')
                namespace = {'train_df': frame.copy(), 'np': np, 'pd': pd,
                             'results_dir': Path(directory), 'timestamp': 'run'}
                with redirect_stdout(StringIO()):
                    exec(code, namespace)
                self.assertEqual(namespace['excluded_orientation_ids'], ['missing'])
                pd.testing.assert_frame_equal(namespace['train_df'], frame.iloc[1:])
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_focused_cleanup_preserves_editable_choices_logs_and_figures(self):
        notebook = deepcopy(self.notebooks['multibranch_pretraining.ipynb'])
        original_choices = deepcopy(notebook['cells'][14])
        logs = [{'output_type': 'stream', 'name': 'stdout', 'text': ['Epoch 1/10\n']}]
        figure = {'output_type': 'display_data', 'data': {'image/png': 'stored figure'},
                  'metadata': {}}
        notebook['cells'][17]['outputs'] = deepcopy(logs)
        notebook['cells'][20]['outputs'] = [deepcopy(figure), {
            'output_type': 'stream', 'name': 'stdout',
            'text': ['Saved curves to old/path.png\n', 'Epochs run: 10\n'],
        }]
        configure_result_outputs(notebook)
        self.assertEqual(notebook['cells'][14], original_choices)
        self.assertEqual(notebook['cells'][17]['outputs'], logs)
        self.assertEqual(notebook['cells'][20]['outputs'], [figure, {
            'output_type': 'stream', 'name': 'stdout', 'text': ['Epochs run: 10\n'],
        }])


if __name__ == '__main__':
    unittest.main()
