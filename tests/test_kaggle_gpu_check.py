"""Ensure the temporary GPU check has a coherent budget and preserves defaults."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import kaggle_gpu_check as gpu_check


class GPUCheckTests(unittest.TestCase):
    def test_validation_budget_is_valid_and_local_notebook_stays_unchanged(self):
        local = ROOT / 'notebooks/multibranch_v1.ipynb'
        original = local.read_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            build = Path(temporary)
            with patch.object(gpu_check.workflow, 'BUILD', build), patch.object(
                gpu_check.workflow, 'cli', return_value='Kernel version 5 successfully pushed.'
            ):
                gpu_check.main()
            metadata = json.loads((build / 'kernel-metadata.json').read_text())
            notebook = json.loads((build / metadata['code_file']).read_text())
            budget = next(''.join(c['source']) for c in notebook['cells'] if c.get('id') == 'gpu-check-budget')
            namespace = dict(tournament_size=3, param_space={}, ECat=lambda values: values,
                             pipeline=SimpleNamespace(set_params=lambda **params: None))
            exec(budget, namespace)
            self.assertEqual(namespace['population_size'], 2)
            self.assertEqual(namespace['generations'], 0)
            self.assertLessEqual(namespace['tournament_size'], namespace['population_size'])
            self.assertEqual(namespace['param_space']['estimator__epochs'], [1])
            subset = next(''.join(c['source']) for c in notebook['cells'] if c.get('id') == 'gpu-check-data')
            rows = [
                {'sequence_id': f'{label}_{index}', 'target': label}
                for label, count in [('large', 20), ('small', 5)]
                for index in range(count)
                for _ in range(2)
            ]
            subset_namespace = dict(train_df=pd.DataFrame(rows), TARGET_COL='target', random_state=42)
            exec(subset, subset_namespace)
            retained = subset_namespace['train_df']
            self.assertEqual(retained.groupby('target')['sequence_id'].nunique().to_dict(),
                             {'large': 16, 'small': 5})
            self.assertTrue(retained.groupby('sequence_id').size().eq(2).all())
            manifest = json.loads((build / 'submitted.json').read_text())
            self.assertEqual(manifest['run_stage'], 'gpu_validation')
            self.assertEqual(manifest['submitted_source_sha256'], gpu_check.workflow.source_hash(notebook))
            self.assertEqual(manifest['csv_names'], ['multibranch_v1_cv.csv'])
        self.assertEqual(local.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
