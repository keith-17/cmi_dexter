"""Package, submit and inspect the isolated multibranch pretraining CPU run.

The local notebook is never rewritten. Every run embeds the exact local src/*.py
bytes rather than relying on an older utility snapshot in an attached dataset.
"""
from __future__ import annotations

import argparse
import ast
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import zipfile


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / 'kaggle_build' / 'multibranch_pretraining'
METADATA = ROOT / 'kaggle' / 'pretraining-kernel-metadata.json'
NOTEBOOK_NAME = 'multibranch_pretraining_v1.ipynb'
REQUIRED_MODULES = {'base_utils.py', 'multibranch_utils.py', 'pretraining_utils.py'}


def source_hash(notebook):
    sources = [''.join(cell['source']) for cell in notebook['cells']]
    return hashlib.sha256(json.dumps(sources, ensure_ascii=False).encode('utf-8')).hexdigest()


def cli(*args):
    result = subprocess.run([sys.executable, '-m', 'kaggle', *args], check=True,
                            capture_output=True, text=True)
    print(result.stdout, end='')
    return result.stdout


def literal_configuration(notebook):
    """Read literal configuration values without importing or running notebook code."""
    values = {}
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        tree = ast.parse(''.join(cell['source']))
        for node in tree.body:
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign):
                targets, value = [node.target], node.value
            else:
                continue
            try:
                literal = ast.literal_eval(value)
            except (ValueError, TypeError):
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    values[target.id] = literal
    return values


def validate_first_run(configuration, iterations):
    """Require the explicit five-candidate model-1 baseline selected in the notebook."""
    if configuration.get('MODEL_VARIANT') != 'model_1':
        raise RuntimeError('First run requires MODEL_VARIANT = "model_1".')
    if configuration.get('FIRST_RUN') is not True:
        raise RuntimeError('First run requires FIRST_RUN = True (fixed feature/regularisation settings).')
    if configuration.get('search_mode') not in {'bayes', 'bayesian'}:
        raise RuntimeError('First run requires Bayesian search.')
    if iterations != 5:
        raise RuntimeError('First run requires exactly five Bayesian candidates.')
    if set(configuration.get('PRETRAINING_SOURCE_CHOICES', [])) != {'non_target', 'all', 'target', 'baseline'}:
        raise RuntimeError('First run must expose non_target, all, target and baseline pretraining sources.')
    base_learning_rate = configuration.get('BASE_LEARNING_RATE')
    if not isinstance(base_learning_rate, (int, float)) or base_learning_rate <= 0:
        raise RuntimeError('First run requires the existing positive BASE_LEARNING_RATE.')
    fixed = configuration.get('FIRST_RUN_FIXED_PARAMS')
    if not isinstance(fixed, dict):
        raise RuntimeError('First run requires the explicit FIRST_RUN_FIXED_PARAMS dictionary.')
    required_disabled = {
        'estimator__augmentor__prob', 'estimator__augmentor__per_aug_prob',
        'estimator__dropout', 'estimator__weight_decay', 'estimator__label_smoothing',
        'estimator__use_batchnorm', 'estimator__use_lr_scheduler',
        'estimator__pretraining_use_lr_scheduler', 'estimator__pretraining_weight_decay',
        'estimator__pretraining_label_smoothing', 'estimator__early_stopping_patience',
        'estimator__pretraining_early_stopping_patience',
    }
    if required_disabled - set(fixed):
        raise RuntimeError(f'First-run fixed settings missing: {sorted(required_disabled - set(fixed))}.')
    disabled_fragments = ('dropout', 'weight_decay', 'label_smoothing', 'use_batchnorm',
                          'use_lr_scheduler', 'early_stopping_patience', 'warmup_epochs',
                          'gradient_clip')
    for name, value in fixed.items():
        augmentation_disabled = '__augmentor__' in name and not name.endswith('crop_frac_range')
        if augmentation_disabled or any(fragment in name for fragment in disabled_fragments):
            if value not in (None, False, 0, 0.0):
                raise RuntimeError(f'First-run augmentation/regularisation must be disabled: {name}={value!r}.')
    if fixed.get('estimator__augmentor__crop_frac_range', (1.0, 1.0)) != (1.0, 1.0):
        raise RuntimeError('First-run crop augmentation must preserve the complete sequence.')


def prepare(iterations=None, notebook_name=NOTEBOOK_NAME, first_run=False):
    if iterations is not None and iterations < 1:
        raise RuntimeError('Iteration override must be a positive integer.')
    branch = subprocess.run(['git', 'branch', '--show-current'], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    if not branch.startswith('ssl/'):
        raise RuntimeError('Switch to the ssl/* development branch before preparing this experiment.')
    metadata = json.loads(METADATA.read_text(encoding='utf-8'))
    if metadata.get('enable_gpu') is not False or metadata.get('enable_tpu', False):
        raise RuntimeError('This workflow requires CPU metadata (GPU and TPU disabled).')
    if metadata.get('is_private') is not True:
        raise RuntimeError('This experiment must use a private Kaggle kernel.')
    notebook_path = ROOT / 'notebooks' / notebook_name
    if notebook_path.resolve().parent != (ROOT / 'notebooks').resolve():
        raise RuntimeError('Notebook must be directly inside the notebooks directory.')
    notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
    iteration_cells = []
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        tree = ast.parse(source)
        if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
               for node in ast.walk(tree)):
            raise RuntimeError('Notebook must contain configuration and calls, with no function definitions.')
        compile(source, str(notebook_path), 'exec')
        cell['outputs'] = []
        cell['execution_count'] = None
        if re.search(r'^n_iter\s*=\s*\d+\b', source, re.MULTILINE):
            iteration_cells.append(cell)
    if len(iteration_cells) != 1:
        raise RuntimeError('Expected exactly one literal n_iter configuration assignment.')
    configuration_cell = iteration_cells[0]
    configuration_source = ''.join(configuration_cell['source'])
    original_iterations = int(re.search(r'^n_iter\s*=\s*(\d+)\b', configuration_source,
                                        re.MULTILINE).group(1))
    if iterations is not None:
        configuration_source, count = re.subn(r'^(n_iter\s*=\s*)\d+\b',
                                              lambda match: match.group(1) + str(iterations),
                                              configuration_source, flags=re.MULTILINE)
        if count != 1:
            raise RuntimeError('Iteration override must change exactly one assignment.')
        configuration_cell['source'] = configuration_source.splitlines(keepends=True)
    actual_iterations = iterations if iterations is not None else original_iterations
    configuration = literal_configuration(notebook)
    if first_run:
        validate_first_run(configuration, actual_iterations)
    modules = sorted((ROOT / 'src').glob('*.py'))
    if not REQUIRED_MODULES <= {path.name for path in modules}:
        raise RuntimeError(f'Required Python utility modules are missing: {sorted(REQUIRED_MODULES)}.')
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as zipped:
        for module in modules:
            zipped.writestr(module.name, module.read_bytes())
    payload = base64.b64encode(archive.getvalue()).decode('ascii')
    bootstrap = (
        '# Install the exact local source snapshot for this Kaggle run.\n'
        'import base64, io, sys, zipfile\n'
        'from pathlib import Path\n'
        "_source_dir = Path.cwd() / 'src'\n"
        '_source_dir.mkdir(exist_ok=True)\n'
        f'with zipfile.ZipFile(io.BytesIO(base64.b64decode({payload!r}))) as _archive:\n'
        '    _archive.extractall(_source_dir)\n'
        'sys.path.insert(0, str(_source_dir))\n'
        "print('Exact local pretraining source snapshot installed.')\n"
    )
    notebook['cells'].insert(0, {
        'cell_type': 'code', 'id': 'kaggle-pretraining-source-bootstrap', 'metadata': {},
        'execution_count': None, 'outputs': [], 'source': bootstrap.splitlines(keepends=True),
    })
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / metadata['code_file']).write_text(json.dumps(notebook, indent=1), encoding='utf-8')
    (BUILD / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    experiment_name = configuration.get('experiment_name', 'multibranch_pretraining_v1')
    suffixes = ('cv',) if configuration.get('quick_save', False) else ('cv', 'best', 'holdout')
    manifest = {
        'branch': branch, 'kernel': metadata['id'], 'source_mode': 'local_snapshot',
        'dataset_sources': metadata.get('dataset_sources', []),
        'competition_sources': metadata.get('competition_sources', []),
        'device': 'cpu', 'first_run_validated': first_run,
        'configuration': {key: configuration[key] for key in (
            'MODEL_VARIANT', 'FIRST_RUN', 'experiment_name', 'search_mode',
            'population_size', 'generations', 'quick_save', 'n_splits', 'train_size',
            'PRETRAINING_EPOCHS', 'FINETUNING_EPOCHS', 'FIRST_RUN_FIXED_PARAMS',
            'FIRST_RUN_PRETRAINING_EPOCHS', 'FIRST_RUN_FINETUNING_EPOCHS',
            'BASE_LEARNING_RATE', 'PRETRAINING_SOURCE_CHOICES',
        ) if key in configuration},
        'search_iterations': actual_iterations, 'iteration_override': iterations,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'submitted_source_sha256': source_hash(notebook),
        'sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in [notebook_path, *modules]},
        'csv_names': [f'{experiment_name}_{suffix}.csv' for suffix in suffixes],
    }
    (BUILD / 'submission.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Prepared {metadata["id"]} from branch {branch}: {BUILD}')
    return metadata


def record_submission(output):
    match = re.search(r'Kernel version (\d+) successfully pushed', output)
    if not match:
        raise RuntimeError('Kaggle did not acknowledge a submitted version. Check status before retrying.')
    manifest = json.loads((BUILD / 'submission.json').read_text(encoding='utf-8'))
    manifest.update(version=int(match.group(1)), acknowledged_utc=datetime.now(timezone.utc).isoformat())
    (BUILD / 'submitted.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def collect(download_dir, destination, expected_names):
    expected_names = set(expected_names)
    for archive in download_dir.rglob('*.zip'):
        with zipfile.ZipFile(archive) as zipped:
            for name in zipped.namelist():
                if not (archive.parent / name).resolve().is_relative_to(download_dir.resolve()):
                    raise RuntimeError('Unsafe path in downloaded output archive.')
            zipped.extractall(archive.parent)
    groups = {}
    for path in download_dir.rglob('*.csv'):
        if path.name in expected_names:
            groups.setdefault(path.parent, set()).add(path.name)
    complete = [parent for parent, names in groups.items() if names == expected_names]
    if not complete:
        raise RuntimeError(f'No complete result set found: {sorted(expected_names)}. Inspect downloaded logs.')
    for parent in complete:
        target = destination / parent.relative_to(download_dir)
        target.mkdir(parents=True, exist_ok=True)
        for name in expected_names:
            shutil.copy2(parent / name, target / name)
        print(f'Results downloaded to {target}')


def download(metadata):
    status = cli('kernels', 'status', metadata['id'])
    if 'KernelWorkerStatus.COMPLETE' not in status:
        raise RuntimeError('Kernel is not complete; result download postponed.')
    submitted = BUILD / 'submitted.json'
    if not submitted.exists():
        raise RuntimeError('No acknowledged local submission; cannot associate remote outputs with this run.')
    manifest = json.loads(submitted.read_text(encoding='utf-8'))
    if manifest['kernel'] != metadata['id']:
        raise RuntimeError('Submitted kernel differs from configured pretraining kernel.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    downloaded = BUILD / 'downloads' / stamp
    remote = downloaded / 'remote'
    remote.mkdir(parents=True)
    cli('kernels', 'pull', metadata['id'], '-p', str(remote), '-m')
    remote_metadata = json.loads((remote / 'kernel-metadata.json').read_text(encoding='utf-8'))
    notebook = json.loads((remote / remote_metadata['code_file']).read_text(encoding='utf-8'))
    if source_hash(notebook) != manifest['submitted_source_sha256']:
        raise RuntimeError('Latest remote source differs from submitted source; outputs not collected.')
    expected_names = set(manifest['csv_names'])
    pattern = '(?:' + '|'.join(re.escape(name) for name in sorted(expected_names)) + ')$'
    cli('kernels', 'output', metadata['id'], '-p', str(downloaded), '--file-pattern', pattern)
    destination = ROOT / 'results' / 'kaggle_pretraining' / stamp
    collect(downloaded, destination, expected_names)
    shutil.copy2(submitted, destination / 'local_submission.json')
    for log in downloaded.glob('*.log'):
        shutil.copy2(log, destination / log.name)
    return destination


def wait_for_results(metadata, interval=60):
    while True:
        status = cli('kernels', 'status', metadata['id'])
        (BUILD / 'last_status.txt').write_text(status, encoding='utf-8')
        if 'KernelWorkerStatus.COMPLETE' in status:
            return download(metadata)
        if any(f'KernelWorkerStatus.{state}' in status for state in ('ERROR', 'CANCEL', 'CANCELLED', 'CANCELED')):
            (BUILD / 'failure.log').write_text(cli('kernels', 'logs', metadata['id']), encoding='utf-8')
            raise RuntimeError(f'Kaggle execution failed. Inspect {BUILD / "failure.log"}.')
        time.sleep(min(60, max(15, interval)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'push', 'status', 'logs', 'wait', 'download', 'pull'])
    parser.add_argument('--iterations', type=int, help='Override packaged n_iter only; local notebook stays unchanged.')
    parser.add_argument('--notebook', default=NOTEBOOK_NAME, help='Notebook filename within notebooks/.')
    parser.add_argument('--first-run', action='store_true', help='Require the explicit model-1 five-candidate Bayesian baseline.')
    parser.add_argument('--interval', type=int, default=60, help='Status polling interval, clamped to 15–60 seconds.')
    parser.add_argument('--follow', action='store_true', help='Continuously follow logs instead of one snapshot.')
    args = parser.parse_args()
    try:
        metadata = json.loads(METADATA.read_text(encoding='utf-8'))
        if args.action in {'prepare', 'push'}:
            metadata = prepare(args.iterations, args.notebook, args.first_run)
            if args.action == 'push':
                record_submission(cli('kernels', 'push', '-p', str(BUILD)))
        elif args.action == 'status':
            cli('kernels', 'status', metadata['id'])
        elif args.action == 'logs':
            if args.follow:
                subprocess.run([sys.executable, '-u', '-m', 'kaggle', 'kernels', 'logs', metadata['id'], '--follow'], check=True)
            else:
                cli('kernels', 'logs', metadata['id'])
        elif args.action == 'pull':
            cli('kernels', 'pull', metadata['id'], '-p', str(BUILD / 'remote'), '-m')
        elif args.action == 'wait':
            wait_for_results(metadata, args.interval)
        else:
            download(metadata)
    except subprocess.CalledProcessError as error:
        print(error.stdout or '', end='', file=sys.stderr)
        print(error.stderr or '', end='', file=sys.stderr)
        parser.exit(1, 'Kaggle command failed. Check CLI installation, authentication and competition access.\n')
    except (RuntimeError, FileNotFoundError, ValueError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
