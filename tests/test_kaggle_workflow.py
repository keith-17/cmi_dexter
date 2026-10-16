"""Offline integration checks; no GPU or Kaggle credentials required."""
import importlib.util
import base64
import io
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('workflow', ROOT / 'scripts/kaggle_experiment.py')
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


class WorkflowTests(unittest.TestCase):
    def test_quick_save_collects_the_configured_single_cv_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / 'output'
            output.mkdir()
            (output / 'multibranch_v1_cv.csv').write_text('mean_test_score\n0.7\n')
            workflow.collect(output, root / 'results', expected_names={'multibranch_v1_cv.csv'})
            self.assertTrue((root / 'results' / 'multibranch_v1_cv.csv').exists())

    def test_attached_source_uses_dataset_without_injecting_local_modules(self):
        original = json.loads((ROOT / 'notebooks/multibranch_v1.ipynb').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(workflow, 'BUILD', Path(temporary)):
                metadata = workflow.prepare(notebook_name='multibranch_v1.ipynb', use_attached_source=True)
                packaged = json.loads((Path(temporary) / metadata['code_file']).read_text())
                manifest = json.loads((Path(temporary) / 'submission.json').read_text())
            self.assertEqual([c['source'] for c in packaged['cells']], [c['source'] for c in original['cells']])
            self.assertEqual(manifest['source_mode'], 'attached_dataset')
            self.assertIn('keithmarange/eliaslongagain', metadata['dataset_sources'])

    def test_v1_packages_exact_local_modules_without_editing_local_notebook(self):
        path = ROOT / 'notebooks/multibranch_v1.ipynb'
        original_bytes = path.read_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(workflow, 'BUILD', Path(temporary)):
                metadata = workflow.prepare(notebook_name='multibranch_v1.ipynb')
                packaged = json.loads((Path(temporary) / metadata['code_file']).read_text())
            original = json.loads(original_bytes)
            self.assertEqual(packaged['cells'][0]['id'], 'kaggle-source-bootstrap')
            self.assertEqual([c['source'] for c in packaged['cells'][1:]],
                             [c['source'] for c in original['cells']])
            bootstrap = ast.parse(''.join(packaged['cells'][0]['source']))
            payload = next(node.args[0].value for node in ast.walk(bootstrap)
                           if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                           and node.func.attr == 'b64decode')
            with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as archive:
                for module in (ROOT / 'src').glob('*.py'):
                    self.assertEqual(archive.read(module.name), module.read_bytes())
        self.assertEqual(path.read_bytes(), original_bytes)

    def test_two_iteration_test_changes_only_packaged_iteration_count(self):
        original_bytes = (ROOT / 'notebooks/multibranch_alpha.ipynb').read_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(workflow, 'BUILD', Path(temporary)):
                metadata = workflow.prepare(iterations=2)
                packaged = json.loads((Path(temporary) / metadata['code_file']).read_text())
                original = json.loads(original_bytes)
                changed = 0
                self.assertEqual(packaged['cells'][0]['id'], 'kaggle-source-bootstrap')
                for before, after in zip(original['cells'], packaged['cells'][1:]):
                    expected, actual = ''.join(before['source']), ''.join(after['source'])
                    if expected != actual:
                        changed += 1
                        self.assertEqual(actual, expected.replace('n_iter = 37', 'n_iter = 2'))
                self.assertEqual(changed, 1)
                manifest = json.loads((Path(temporary) / 'submission.json').read_text())
                self.assertEqual(manifest['search_iterations'], 2)
                self.assertEqual(manifest['iteration_override'], 2)
        self.assertEqual(original_bytes, (ROOT / 'notebooks/multibranch_alpha.ipynb').read_bytes())

    def test_acknowledged_version_survives_prepare(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(workflow, 'BUILD', Path(temporary)):
                workflow.prepare()
                workflow.record_submission('Kernel version 33 successfully pushed.')
                submitted = (Path(temporary) / 'submitted.json').read_bytes()
                workflow.prepare()
                self.assertEqual((Path(temporary) / 'submitted.json').read_bytes(), submitted)
                self.assertEqual(json.loads(submitted)['version'], 33)

    def test_download_verifies_completed_remote_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build = root / 'build'
            build.mkdir()
            notebook = {'cells': [{'source': ['original experiment\n']}]}
            metadata = {'id': 'keithmarange/multibranch-attention', 'code_file': 'experiment.ipynb'}
            manifest = {'kernel': metadata['id'], 'version': 33,
                        'submitted_source_sha256': workflow.source_hash(notebook)}
            (build / 'submitted.json').write_text(json.dumps(manifest))

            def fake_cli(*args):
                if args[1] == 'status':
                    return 'KernelWorkerStatus.COMPLETE'
                target = Path(args[args.index('-p') + 1])
                target.mkdir(parents=True, exist_ok=True)
                if args[1] == 'pull':
                    (target / 'kernel-metadata.json').write_text(json.dumps(metadata))
                    (target / metadata['code_file']).write_text(json.dumps(notebook))
                elif args[1] == 'output':
                    self.assertIn('--file-pattern', args)
                    for name in workflow.CSV_NAMES:
                        (target / name).write_text('score\n0.5\n')
                return ''

            with patch.object(workflow, 'ROOT', root), patch.object(workflow, 'BUILD', build), patch.object(workflow, 'cli', side_effect=fake_cli):
                destination = workflow.download(metadata)
                self.assertEqual({p.name for p in destination.glob('*.csv')}, workflow.CSV_NAMES)
                self.assertEqual(json.loads((destination / 'local_submission.json').read_text())['version'], 33)
                manifest['submitted_source_sha256'] = 'stale source'
                (build / 'submitted.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(RuntimeError, 'differs from the submitted source'):
                    workflow.download(metadata)

    def test_wait_downloads_only_after_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(workflow, 'BUILD', Path(temporary)), patch.object(workflow, 'cli', side_effect=['KernelWorkerStatus.RUNNING', 'KernelWorkerStatus.COMPLETE']), patch.object(workflow.time, 'sleep') as sleep, patch.object(workflow, 'download', return_value='results') as download:
                metadata = {'id': 'keithmarange/multibranch-attention'}
                self.assertEqual(workflow.wait_for_results(metadata, 60), 'results')
                sleep.assert_called_once_with(60)
                download.assert_called_once_with(metadata)

    def test_alpha_packages_a_local_snapshot_and_preserves_notebook_sources(self):
        original_bytes = (ROOT / 'notebooks/multibranch_alpha.ipynb').read_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(workflow, 'BUILD', Path(temporary)):
                metadata = workflow.prepare()
                notebook = json.loads((Path(temporary) / metadata['code_file']).read_text())
            original = json.loads(original_bytes)
            self.assertEqual(len(notebook['cells']), len(original['cells']) + 1)
            self.assertEqual(notebook['cells'][0]['id'], 'kaggle-source-bootstrap')
            for expected, actual in zip(original['cells'], notebook['cells'][1:]):
                self.assertEqual(expected['source'], actual['source'])
            full_text = '\n'.join(''.join(c.get('source', [])) for c in notebook['cells'])
            self.assertNotIn('MODULE_SNAPSHOT', full_text)
            self.assertTrue(metadata['enable_gpu'])
            self.assertIn('cmi-detect-behavior-with-sensor-data', metadata['competition_sources'])
        self.assertEqual(original_bytes, (ROOT / 'notebooks/multibranch_alpha.ipynb').read_bytes())

    def test_collect_nested_zipped_csv_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            downloaded = root / 'downloads'
            downloaded.mkdir()
            with zipfile.ZipFile(downloaded / 'output.zip', 'w') as archive:
                for name in workflow.CSV_NAMES:
                    archive.writestr('results_multibranch_attention_bfrb_bayesian/20261005/' + name, 'score\n0.5\n')
            workflow.collect(downloaded, root / 'results')
            self.assertEqual({p.name for p in (root / 'results').rglob('*.csv')}, workflow.CSV_NAMES)

    def test_incomplete_results_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, 'No complete'):
                workflow.collect(Path(temporary), Path(temporary) / 'results')

    def test_zip_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with zipfile.ZipFile(directory / 'output.zip', 'w') as archive:
                archive.writestr('../bad.csv', 'bad')
            with self.assertRaisesRegex(RuntimeError, 'Unsafe path'):
                workflow.collect(directory, directory / 'results')


if __name__ == '__main__':
    unittest.main()
