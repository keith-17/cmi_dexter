"""Offline packaging tests for the separate pretraining CPU workflow."""
import ast
import base64
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pretraining_workflow', ROOT / 'scripts/kaggle_pretraining_experiment.py')
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


def minimal_notebook():
    return {'cells': [{
        'cell_type': 'code', 'source': [
            "MODEL_VARIANT = 'model_1'\n",
            'FIRST_RUN = True\n',
            "search_mode = 'bayes'\n",
            'n_iter = 5\n',
            "experiment_name = 'multibranch_pretraining_v1'\n",
            'quick_save = True\n',
            'PRETRAINING_EPOCHS = 5\n',
            'FINETUNING_EPOCHS = 5\n',
            "PRETRAINING_SOURCE_CHOICES = ['non_target', 'all', 'target', 'baseline']\n",
            'BASE_LEARNING_RATE = 0.0001\n',
            "FIRST_RUN_FIXED_PARAMS = {\n",
            "    'estimator__augmentor__prob': 0.0,\n",
            "    'estimator__augmentor__per_aug_prob': 0.0,\n",
            "    'estimator__dropout': 0.0,\n",
            "    'estimator__weight_decay': 0.0,\n",
            "    'estimator__label_smoothing': 0.0,\n",
            "    'estimator__use_batchnorm': False,\n",
            "    'estimator__use_lr_scheduler': False,\n",
            "    'estimator__pretraining_use_lr_scheduler': False,\n",
            "    'estimator__pretraining_weight_decay': 0.0,\n",
            "    'estimator__pretraining_label_smoothing': 0.0,\n",
            "    'estimator__early_stopping_patience': 0,\n",
            "    'estimator__pretraining_early_stopping_patience': 0,\n",
            '}\n',
        ], 'outputs': [{'text': 'old output'}], 'execution_count': 1,
    }], 'metadata': {}, 'nbformat': 4, 'nbformat_minor': 5}


class PretrainingWorkflowTests(unittest.TestCase):
    def fixture(self, root, notebook=None):
        (root / 'notebooks').mkdir()
        (root / 'src').mkdir()
        (root / 'kaggle').mkdir()
        (root / 'notebooks' / workflow.NOTEBOOK_NAME).write_text(json.dumps(notebook or minimal_notebook()), encoding='utf-8')
        for module in sorted(workflow.REQUIRED_MODULES):
            (root / 'src' / module).write_bytes(('# ' + module + '\r\nVALUE = 1\r\n').encode())
        metadata = json.loads((ROOT / 'kaggle/pretraining-kernel-metadata.json').read_text(encoding='utf-8'))
        (root / 'kaggle/pretraining-kernel-metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
        return metadata

    def large_fixtures(self, root):
        pilot_metadata = self.fixture(root)
        profiles = {}
        for model in (1, 2):
            profile = f'model{model}-large'
            filename = 'multibranch_pretraining.ipynb' if model == 1 else 'multibranch_ssl.ipynb'
            notebook = minimal_notebook()
            notebook['cells'][0]['source'] = [
                f"MODEL_VARIANT = 'model_{model}'\n", 'FIRST_RUN = False\n',
                "search_mode = 'bayes'\n", 'n_iter = 37\n',
                f"experiment_name = 'multibranch_pretraining_model{model}_large'\n",
                'quick_save = True\n', 'LARGE_RUN = True\n',
                "PRETRAINING_SOURCE_CHOICES = ['non_target', 'all', 'target', 'baseline']\n",
                'COVER_PRETRAINING_SOURCES = True\n',
                "INITIAL_PRETRAINING_SOURCES = ['non_target', 'all', 'target', 'baseline']\n",
                "LARGE_FIXED_PARAMS = {'estimator__bert_d_model': 128}\n",
            ]
            metadata = {**pilot_metadata, 'id': f'owner/{profile}', 'code_file': filename}
            (root / 'notebooks' / filename).write_text(json.dumps(notebook), encoding='utf-8')
            (root / 'kaggle' / f'pretraining-model{model}-large-kernel-metadata.json').write_text(
                json.dumps(metadata), encoding='utf-8')
            profiles[profile] = metadata
        return profiles

    def test_profile_defaults_keep_all_three_experiments_isolated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profiles = self.large_fixtures(root)
            with patch.object(workflow, 'ROOT', root), patch.object(workflow, 'BUILD', root / 'build'), \
                    patch.object(workflow, 'METADATA', root / 'kaggle/pretraining-kernel-metadata.json'), \
                    patch.object(workflow.subprocess, 'run', return_value=SimpleNamespace(stdout='ssl/v0/develop\n')):
                workflow.prepare(first_run=True)
                pilot_before = (root / 'build/submission.json').read_bytes()
                for profile, metadata in profiles.items():
                    with self.subTest(profile=profile):
                        self.assertEqual(workflow.prepare(profile=profile), metadata)
                        build = root / 'build' / profile
                        manifest = json.loads((build / 'submission.json').read_text())
                        self.assertEqual(manifest['profile'], profile)
                        self.assertEqual(manifest['kernel'], metadata['id'])
                        self.assertEqual(manifest['notebook'], metadata['code_file'])
                        self.assertEqual(manifest['search_iterations'], 37)
                        self.assertTrue(manifest['configuration']['LARGE_RUN'])
                        self.assertTrue(manifest['configuration']['COVER_PRETRAINING_SOURCES'])
                        self.assertEqual(len(manifest['configuration']['INITIAL_PRETRAINING_SOURCES']), 4)
                        self.assertEqual(manifest['configuration']['LARGE_FIXED_PARAMS'],
                                         {'estimator__bert_d_model': 128})
                        workflow.record_submission('Kernel version 1 successfully pushed.', profile=profile)
                        submitted = json.loads((build / 'submitted.json').read_text())
                        self.assertEqual(submitted['kernel'], metadata['id'])
                self.assertEqual((root / 'build/submission.json').read_bytes(), pilot_before)
                self.assertFalse((root / 'build/submitted.json').exists())

    def test_large_profile_honours_explicit_notebook_and_packaged_iteration_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.large_fixtures(root)
            original = root / 'notebooks/multibranch_pretraining.ipynb'
            override = root / 'notebooks/custom-model1.ipynb'
            override.write_bytes(original.read_bytes())
            with patch.object(workflow, 'ROOT', root), patch.object(workflow, 'BUILD', root / 'build'), \
                    patch.object(workflow.subprocess, 'run', return_value=SimpleNamespace(stdout='ssl/v0/develop\n')):
                metadata = workflow.prepare(iterations=39, notebook_name=override.name, profile='model1-large')
            manifest = json.loads((root / 'build/model1-large/submission.json').read_text())
            self.assertEqual(manifest['notebook'], override.name)
            self.assertEqual(manifest['search_iterations'], 39)
            self.assertEqual(workflow.literal_configuration(json.loads(override.read_text()))['n_iter'], 37)
            packaged = json.loads((root / 'build/model1-large' / metadata['code_file']).read_text())
            self.assertEqual(workflow.literal_configuration(packaged)['n_iter'], 39)

    def test_large_profile_refuses_wrong_model_and_pilot_flag(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.large_fixtures(root)
            with patch.object(workflow, 'ROOT', root), patch.object(workflow, 'BUILD', root / 'build'), \
                    patch.object(workflow.subprocess, 'run', return_value=SimpleNamespace(stdout='ssl/v0/develop\n')):
                with self.assertRaisesRegex(RuntimeError, 'requires MODEL_VARIANT'):
                    workflow.prepare(notebook_name='multibranch_ssl.ipynb', profile='model1-large')
                with self.assertRaisesRegex(RuntimeError, 'reserved for'):
                    workflow.prepare(first_run=True, profile='model1-large')
                with self.assertRaisesRegex(RuntimeError, 'Unknown experiment profile'):
                    workflow.prepare(profile='unrecognised')

    def test_cli_routes_status_pull_and_default_notebook_to_selected_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profiles = self.large_fixtures(root)
            with patch.object(workflow, 'ROOT', root), patch.object(workflow, 'BUILD', root / 'build'):
                for profile, metadata in profiles.items():
                    for action in ('status', 'pull'):
                        with self.subTest(profile=profile, action=action), \
                                patch.object(workflow.sys, 'argv', ['workflow', action, '--profile', profile]), \
                                patch.object(workflow, 'cli') as cli:
                            workflow.main()
                            self.assertEqual(cli.call_args.args[:3], ('kernels', action, metadata['id']))
                            if action == 'pull':
                                self.assertEqual(cli.call_args.args[4], str(root / 'build' / profile / 'remote'))
                with patch.object(workflow.sys, 'argv', ['workflow', 'prepare', '--profile', 'model2-large']), \
                        patch.object(workflow, 'prepare') as prepare:
                    workflow.main()
                    prepare.assert_called_once_with(None, None, False, profile='model2-large')
                with patch.object(workflow.sys, 'argv', ['workflow', 'push', '--profile', 'model2-large']), \
                        patch.object(workflow, 'prepare', return_value=profiles['model2-large']), \
                        patch.object(workflow, 'cli', return_value='Kernel version 2 successfully pushed.') as cli, \
                        patch.object(workflow, 'record_submission') as record:
                    workflow.main()
                    cli.assert_called_once_with('kernels', 'push', '-p', str(root / 'build/model2-large'))
                    record.assert_called_once_with('Kernel version 2 successfully pushed.', profile='model2-large')

    def test_download_refuses_wrong_profile_even_when_kernel_matches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build = root / 'build/model1-large'
            build.mkdir(parents=True)
            metadata = {'id': 'owner/model1-large'}
            (build / 'submitted.json').write_text(json.dumps({
                'kernel': metadata['id'], 'profile': 'model2-large'}))
            with patch.object(workflow, 'ROOT', root), patch.object(workflow, 'BUILD', root / 'build'), \
                    patch.object(workflow, 'cli', return_value='KernelWorkerStatus.COMPLETE'):
                with self.assertRaisesRegex(RuntimeError, 'Submitted profile differs'):
                    workflow.download(metadata, profile='model1-large')

    def test_large_download_uses_its_own_submission_and_result_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            notebook = minimal_notebook()
            metadata = {'id': 'owner/model2-large', 'code_file': 'model2.ipynb'}
            build = root / 'build/model2-large'
            build.mkdir(parents=True)
            manifest = {'kernel': metadata['id'], 'profile': 'model2-large', 'version': 1,
                        'submitted_source_sha256': workflow.source_hash(notebook),
                        'csv_names': ['model2_cv.csv']}
            (build / 'submitted.json').write_text(json.dumps(manifest))
            (root / 'build/submitted.json').write_text('pilot submission must stay untouched')

            def fake_cli(*args):
                self.assertEqual(args[2], metadata['id'])
                if args[1] == 'status':
                    return 'KernelWorkerStatus.COMPLETE'
                target = Path(args[args.index('-p') + 1])
                self.assertTrue(target.is_relative_to(build))
                target.mkdir(parents=True, exist_ok=True)
                if args[1] == 'pull':
                    (target / 'kernel-metadata.json').write_text(json.dumps(metadata))
                    (target / metadata['code_file']).write_text(json.dumps(notebook))
                elif args[1] == 'output':
                    (target / 'model2_cv.csv').write_text('score\n0.6\n')
                return ''

            with patch.object(workflow, 'ROOT', root), patch.object(workflow, 'BUILD', root / 'build'), \
                    patch.object(workflow, 'cli', side_effect=fake_cli):
                destination = workflow.download(metadata, profile='model2-large')
                self.assertEqual(destination.parent, root / 'results/kaggle_pretraining/model2-large')
                self.assertTrue((destination / 'model2_cv.csv').exists())
                self.assertEqual(json.loads((destination / 'local_submission.json').read_text()), manifest)
                self.assertEqual((root / 'build/submitted.json').read_text(), 'pilot submission must stay untouched')

    def test_packages_exact_modules_and_preserves_local_notebook(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = self.fixture(root)
            local = root / 'notebooks' / workflow.NOTEBOOK_NAME
            before = local.read_bytes()
            with patch.object(workflow, 'ROOT', root), patch.object(workflow, 'BUILD', root / 'build'), \
                    patch.object(workflow, 'METADATA', root / 'kaggle/pretraining-kernel-metadata.json'), \
                    patch.object(workflow.subprocess, 'run', return_value=SimpleNamespace(stdout='ssl/v0/develop\n')):
                actual_metadata = workflow.prepare(first_run=True)
            self.assertEqual(metadata, actual_metadata)
            self.assertEqual(local.read_bytes(), before)
            packaged = json.loads((root / 'build' / metadata['code_file']).read_text(encoding='utf-8'))
            self.assertEqual(packaged['cells'][0]['id'], 'kaggle-pretraining-source-bootstrap')
            original = json.loads(before)
            self.assertEqual(packaged['cells'][1]['source'], original['cells'][0]['source'])
            self.assertEqual(packaged['cells'][1]['outputs'], [])
            self.assertIsNone(packaged['cells'][1]['execution_count'])
            tree = ast.parse(''.join(packaged['cells'][0]['source']))
            payload = next(node.args[0].value for node in ast.walk(tree)
                           if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                           and node.func.attr == 'b64decode')
            with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as archive:
                self.assertEqual(set(archive.namelist()), workflow.REQUIRED_MODULES)
                for module in (root / 'src').glob('*.py'):
                    self.assertEqual(archive.read(module.name), module.read_bytes())
            manifest = json.loads((root / 'build/submission.json').read_text())
            self.assertEqual(manifest['search_iterations'], 5)
            self.assertEqual(manifest['device'], 'cpu')
            self.assertTrue(manifest['first_run_validated'])
            self.assertEqual(manifest['csv_names'], ['multibranch_pretraining_v1_cv.csv'])
            self.assertEqual(manifest['submitted_source_sha256'], workflow.source_hash(packaged))

    def test_first_run_validation_refuses_wrong_model_or_budget(self):
        configuration = workflow.literal_configuration(minimal_notebook())
        workflow.validate_first_run(configuration, 5)
        for key, value in [('MODEL_VARIANT', 'model_2'), ('FIRST_RUN', False), ('search_mode', 'grid')]:
            with self.subTest(key=key):
                with self.assertRaises(RuntimeError):
                    workflow.validate_first_run({**configuration, key: value}, 5)
        with self.assertRaisesRegex(RuntimeError, 'exactly five'):
            workflow.validate_first_run(configuration, 6)

    def test_protocol_quick_run_embeds_helper_and_requires_complete_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            notebook = minimal_notebook()
            stage_choices = {
                'VALIDATION_PARAM_SPACE': {'validation_mode': ['sequence', 'subject'], 'group_column': ['subject']},
                'PRETRAINING_PARAM_SPACE': {'estimator__pretraining_epochs': [5, 30]},
                'FIRST_RUN_PARAM_SPACE': {'estimator__epochs': [5, 10]},
                'EVOLUTIONARY_STAGE_OVERRIDES': {'estimator__pretraining_learning_rate': [0.00001, 0.0001]},
            }
            notebook['cells'][0]['source'].extend(f'{name} = {value!r}\n' for name, value in stage_choices.items())
            metadata = self.fixture(root, notebook)
            helper = root / 'src/pretraining_search.py'
            helper.write_text('# protocol search helper snapshot\n', encoding='utf-8')
            with patch.object(workflow, 'ROOT', root), patch.object(workflow, 'BUILD', root / 'build'), \
                    patch.object(workflow, 'METADATA', root / 'kaggle/pretraining-kernel-metadata.json'), \
                    patch.object(workflow.subprocess, 'run', return_value=SimpleNamespace(stdout='ssl/v0/develop\n')):
                workflow.prepare()
            manifest = json.loads((root / 'build/submission.json').read_text())
            self.assertEqual(manifest['csv_names'], ['multibranch_pretraining_v1_experiments.csv'])
            self.assertEqual(manifest['result_names'], [
                'multibranch_pretraining_v1_experiments.csv',
                'multibranch_pretraining_v1_summary.json',
            ])
            for name, value in stage_choices.items():
                self.assertEqual(manifest['configuration'][name], value)
            self.assertIn(str(Path('src') / 'pretraining_search.py'), manifest['sha256'])
            packaged = json.loads((root / 'build' / metadata['code_file']).read_text())
            tree = ast.parse(''.join(packaged['cells'][0]['source']))
            payload = next(node.args[0].value for node in ast.walk(tree)
                           if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                           and node.func.attr == 'b64decode')
            with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as archive:
                self.assertEqual(archive.read('pretraining_search.py'), helper.read_bytes())
            self.assertEqual(manifest['submitted_source_sha256'], workflow.source_hash(packaged))

    def test_protocol_notebook_requires_packaged_search_helper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            notebook = minimal_notebook()
            notebook['cells'][0]['source'].append("VALIDATION_PARAM_SPACE = {'validation_mode': ['sequence']}\n")
            self.fixture(root, notebook)
            with patch.object(workflow, 'ROOT', root), patch.object(workflow, 'BUILD', root / 'build'), \
                    patch.object(workflow, 'METADATA', root / 'kaggle/pretraining-kernel-metadata.json'), \
                    patch.object(workflow.subprocess, 'run', return_value=SimpleNamespace(stdout='ssl/v0/develop\n')):
                with self.assertRaisesRegex(RuntimeError, 'pretraining_search.py'):
                    workflow.prepare()

    def test_legacy_protocol_result_collection_preserves_separate_complete_result_sets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            downloaded = root / 'downloaded'
            expected = {f'experiment_{suffix}.csv' for suffix in ('cv', 'best', 'holdout')}
            for protocol in ('sequence', 'subject__subject'):
                directory = downloaded / 'run/validation_protocols' / protocol
                directory.mkdir(parents=True)
                for name in expected:
                    (directory / name).write_text('score\n0.7\n')
            workflow.collect(downloaded, root / 'results', expected)
            for protocol in ('sequence', 'subject__subject'):
                directory = root / 'results/run/validation_protocols' / protocol
                self.assertEqual({path.name for path in directory.glob('*.csv')}, expected)

    def test_result_collection_requires_csv_and_json_in_the_same_run(self):
        expected = {'experiment_experiments.csv', 'experiment_summary.json'}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            downloaded = root / 'downloaded'
            first = downloaded / 'first-run'
            second = downloaded / 'second-run'
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            (first / 'experiment_experiments.csv').write_text('mean_test_score\n0.7\n')
            (second / 'experiment_summary.json').write_text('{"holdout_score": 0.6}')
            with self.assertRaisesRegex(RuntimeError, 'No complete result set'):
                workflow.collect(downloaded, root / 'results', expected)
            self.assertFalse((root / 'results').exists())

            summary = '{"holdout_score": 0.8, "best_params": {"epochs": 5}}'
            (first / 'experiment_summary.json').write_text(summary)
            (first / 'unwanted.csv').write_text('unused\n1\n')
            workflow.collect(downloaded, root / 'results', expected)
            target = root / 'results/first-run'
            self.assertEqual({path.name for path in target.iterdir()}, expected)
            self.assertEqual((target / 'experiment_summary.json').read_text(), summary)
            self.assertFalse((root / 'results/second-run').exists())

    def test_download_requests_and_collects_the_experiments_csv_and_summary_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build = root / 'build'
            build.mkdir()
            notebook = minimal_notebook()
            metadata = {'id': 'owner/new-kernel', 'code_file': workflow.NOTEBOOK_NAME}
            expected = ['experiment_experiments.csv', 'experiment_summary.json']
            manifest = {'kernel': metadata['id'], 'version': 1,
                        'submitted_source_sha256': workflow.source_hash(notebook),
                        'csv_names': [expected[0]], 'result_names': expected}
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
                    pattern = args[args.index('--file-pattern') + 1]
                    self.assertIn(r'experiment_experiments\.csv', pattern)
                    self.assertIn(r'experiment_summary\.json', pattern)
                    output = target / 'run'
                    output.mkdir()
                    (output / expected[0]).write_text('mean_test_score\n0.7\n')
                    (output / expected[1]).write_text('{"holdout_score": 0.8}')
                return ''

            with patch.object(workflow, 'ROOT', root), patch.object(workflow, 'BUILD', build), \
                    patch.object(workflow, 'cli', side_effect=fake_cli):
                destination = workflow.download(metadata)
            self.assertEqual({path.name for path in (destination / 'run').iterdir()}, set(expected))
            self.assertTrue((destination / 'local_submission.json').exists())

    def test_first_run_refuses_regularisation_augmentation_zero_lr_and_missing_sources(self):
        configuration = workflow.literal_configuration(minimal_notebook())
        for key, value in [('estimator__dropout', 0.1), ('estimator__augmentor__prob', 0.5),
                           ('estimator__pretraining_use_lr_scheduler', True)]:
            with self.subTest(key=key):
                invalid = {**configuration, 'FIRST_RUN_FIXED_PARAMS': {
                    **configuration['FIRST_RUN_FIXED_PARAMS'], key: value}}
                with self.assertRaisesRegex(RuntimeError, 'must be disabled'):
                    workflow.validate_first_run(invalid, 5)
        with self.assertRaisesRegex(RuntimeError, 'positive BASE_LEARNING_RATE'):
            workflow.validate_first_run({**configuration, 'BASE_LEARNING_RATE': 0.0}, 5)
        with self.assertRaisesRegex(RuntimeError, 'pretraining sources'):
            workflow.validate_first_run({**configuration, 'PRETRAINING_SOURCE_CHOICES': ['non_target']}, 5)

    def test_iteration_override_changes_packaged_configuration_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            local = root / 'notebooks' / workflow.NOTEBOOK_NAME
            before = local.read_bytes()
            with patch.object(workflow, 'ROOT', root), patch.object(workflow, 'BUILD', root / 'build'), \
                    patch.object(workflow, 'METADATA', root / 'kaggle/pretraining-kernel-metadata.json'), \
                    patch.object(workflow.subprocess, 'run', return_value=SimpleNamespace(stdout='ssl/v0/develop\n')):
                metadata = workflow.prepare(iterations=2)
            packaged = json.loads((root / 'build' / metadata['code_file']).read_text())
            self.assertEqual(workflow.literal_configuration(packaged)['n_iter'], 2)
            self.assertEqual(local.read_bytes(), before)

    def test_rejects_notebook_functions_and_wrong_branch(self):
        for invalid in ('def helper():\n    return 1\n', 'f = lambda x: x\n'):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                notebook = minimal_notebook()
                notebook['cells'][0]['source'].append(invalid)
                self.fixture(root, notebook)
                with patch.object(workflow, 'ROOT', root), patch.object(workflow, 'BUILD', root / 'build'), \
                        patch.object(workflow, 'METADATA', root / 'kaggle/pretraining-kernel-metadata.json'), \
                        patch.object(workflow.subprocess, 'run', return_value=SimpleNamespace(stdout='ssl/v0/develop\n')):
                    with self.assertRaisesRegex(RuntimeError, 'no function definitions'):
                        workflow.prepare()
        with patch.object(workflow.subprocess, 'run', return_value=SimpleNamespace(stdout='main\n')):
            with self.assertRaisesRegex(RuntimeError, 'ssl/\*'):
                workflow.prepare()

    def test_requires_missing_pretraining_module_and_cpu_private_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = self.fixture(root)
            (root / 'src/pretraining_utils.py').unlink()
            with patch.object(workflow, 'ROOT', root), patch.object(workflow, 'BUILD', root / 'build'), \
                    patch.object(workflow, 'METADATA', root / 'kaggle/pretraining-kernel-metadata.json'), \
                    patch.object(workflow.subprocess, 'run', return_value=SimpleNamespace(stdout='ssl/v0/develop\n')):
                with self.assertRaisesRegex(RuntimeError, 'Required Python utility'):
                    workflow.prepare()
                for key, value, message in [('enable_gpu', True, 'CPU'), ('is_private', False, 'private')]:
                    invalid = {**metadata, key: value}
                    (root / 'kaggle/pretraining-kernel-metadata.json').write_text(json.dumps(invalid))
                    with self.assertRaisesRegex(RuntimeError, message):
                        workflow.prepare()

    def test_record_submission_requires_acknowledged_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            build = Path(temporary)
            (build / 'submission.json').write_text(json.dumps({'kernel': 'owner/new-kernel'}))
            with patch.object(workflow, 'BUILD', build):
                workflow.record_submission('Kernel version 1 successfully pushed.')
                self.assertEqual(json.loads((build / 'submitted.json').read_text())['version'], 1)
                with self.assertRaisesRegex(RuntimeError, 'did not acknowledge'):
                    workflow.record_submission('Unknown push status')

    def test_record_submission_tracks_actual_kaggle_slug(self):
        with tempfile.TemporaryDirectory() as temporary:
            build = Path(temporary)
            (build / 'submission.json').write_text(json.dumps({'kernel': 'owner/requested-name'}))
            with patch.object(workflow, 'BUILD', build):
                workflow.record_submission('Kernel version 1 successfully pushed. '
                                           'Please check progress at https://www.kaggle.com/code/owner/title-derived-name')
            manifest = json.loads((build / 'submitted.json').read_text())
            self.assertEqual(manifest['requested_kernel'], 'owner/requested-name')
            self.assertEqual(manifest['kernel'], 'owner/title-derived-name')
            self.assertEqual(manifest['url'], 'https://www.kaggle.com/code/owner/title-derived-name')

    def test_download_verifies_source_before_collecting_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build = root / 'build'
            build.mkdir()
            notebook = minimal_notebook()
            metadata = {'id': 'owner/new-kernel', 'code_file': workflow.NOTEBOOK_NAME}
            manifest = {'kernel': metadata['id'], 'version': 1,
                        'submitted_source_sha256': workflow.source_hash(notebook),
                        'csv_names': ['experiment_cv.csv']}
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
                    (target / 'experiment_cv.csv').write_text('score\n0.5\n')
                return ''

            with patch.object(workflow, 'ROOT', root), patch.object(workflow, 'BUILD', build), \
                    patch.object(workflow, 'cli', side_effect=fake_cli):
                destination = workflow.download(metadata)
                self.assertTrue((destination / 'experiment_cv.csv').exists())
                self.assertTrue((destination / 'local_submission.json').exists())
                manifest['submitted_source_sha256'] = 'old source'
                (build / 'submitted.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(RuntimeError, 'differs from submitted source'):
                    workflow.download(metadata)

    def test_collect_rejects_zip_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with zipfile.ZipFile(root / 'outputs.zip', 'w') as archive:
                archive.writestr('../escape.csv', 'bad')
            with self.assertRaisesRegex(RuntimeError, 'Unsafe path'):
                workflow.collect(root, root / 'results', {'experiment_cv.csv'})

    def test_metadata_is_separate_cpu_kernel_with_existing_sources(self):
        metadata = json.loads((ROOT / 'kaggle/pretraining-kernel-metadata.json').read_text())
        existing = json.loads((ROOT / 'kaggle/kernel-metadata.json').read_text())
        self.assertNotEqual(metadata['id'], existing['id'])
        self.assertFalse(metadata['enable_gpu'])
        self.assertFalse(metadata['enable_tpu'])
        self.assertTrue(metadata['is_private'])
        self.assertEqual(metadata['dataset_sources'], existing['dataset_sources'])
        self.assertEqual(metadata['competition_sources'], existing['competition_sources'])

    def test_actual_notebook_preserves_v1_layout_and_first_cpu_configuration(self):
        notebook = json.loads((ROOT / 'archive/notebooks' / workflow.NOTEBOOK_NAME).read_text(encoding='utf-8'))
        original = json.loads((ROOT / 'notebooks/multibranch_v1.ipynb').read_text(encoding='utf-8'))
        self.assertEqual([cell['cell_type'] for cell in notebook['cells']],
                         [cell['cell_type'] for cell in original['cells']])
        configuration = workflow.literal_configuration(notebook)
        workflow.validate_first_run(configuration, configuration['n_iter'])
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                tree = ast.parse(''.join(cell['source']))
                self.assertFalse(any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
                                     for node in ast.walk(tree)))
        sources = '\n'.join(''.join(cell['source']) for cell in notebook['cells'])
        self.assertIn('GRID_PARAM_SPACE', sources)
        self.assertIn('BAYESIAN_PARAM_SPACE', sources)
        self.assertIn('EVOLUTIONARY_PARAM_SPACE', sources)


if __name__ == '__main__':
    unittest.main()
