"""Submit the current local notebook and src snapshot to Kaggle."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / 'kaggle_build' / 'multibranch_attention'
METADATA = ROOT / 'kaggle' / 'kernel-metadata.json'
CSV_NAMES = {f'multibranch_attention_{name}.csv' for name in ('cv', 'best', 'holdout')}


def source_hash(notebook):
    sources = [''.join(cell['source']) for cell in notebook['cells']]
    return hashlib.sha256(json.dumps(sources, ensure_ascii=False).encode('utf-8')).hexdigest()


def cli(*args):
    command = [sys.executable, '-m', 'kaggle', *args]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    print(result.stdout, end='')
    return result.stdout


def prepare():
    branch = subprocess.run(['git', 'branch', '--show-current'], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    if branch != 'multibranch/v0/develop':
        raise RuntimeError('Switch to multibranch/v0/develop before preparing this experiment.')
    metadata = json.loads(METADATA.read_text(encoding='utf-8'))
    notebook_path = ROOT / 'notebooks' / 'multibranch_attention.ipynb'
    notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
    modules = sorted((ROOT / 'src').glob('*.py'))
    if not {'data_utils.py', 'base_utils_qwen.py', 'multibranch_attention.py'} <= {p.name for p in modules}:
        raise RuntimeError('Required Python utility modules are missing from src/.')
    encoded = {p.name: base64.b64encode(p.read_bytes()).decode('ascii') for p in modules}
    bootstrap = '''import base64, importlib.util, json, os, subprocess, sys
from pathlib import Path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
workspace_root = Path('/kaggle/working')
os.chdir(workspace_root)
src_path = workspace_root / 'src'
src_path.mkdir(exist_ok=True)
for name, contents in MODULE_SNAPSHOT.items():
    (src_path / name).write_bytes(base64.b64decode(contents))
sys.path.insert(0, str(src_path))
# Keep Kaggle's GPU TensorFlow stack; install only missing experiment dependencies.
missing = [package for module, package in [('skopt', 'scikit-optimize==0.10.2'), ('pywt', 'PyWavelets>=1.6,<2')]
           if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])
import tensorflow as tf
if not tf.config.list_physical_devices('GPU'):
    raise RuntimeError('No TensorFlow GPU detected. Check Kaggle accelerator and GPU quota.')
data_dir = Path('/kaggle/input/cmi-detect-behavior-with-sensor-data')
if not (data_dir / 'train.csv').exists():
    matches = list(Path('/kaggle/input').rglob('train.csv'))
    matches = [p for p in matches if (p.parent / 'train_demographics.csv').exists()]
    if len(matches) != 1:
        raise FileNotFoundError('Attach CMI competition data and accept its rules on Kaggle.')
    data_dir = matches[0].parent
# Existing data_utils.find_data_root() discovers this exact directory first.
(workspace_root / 'data').symlink_to(data_dir, target_is_directory=True) if not (workspace_root / 'data').exists() else None
print('GPU:', tf.config.list_physical_devices('GPU'))
print('Data:', data_dir)
'''
    source = 'MODULE_SNAPSHOT = ' + repr(encoded) + '\n' + bootstrap
    cell = {'cell_type': 'code', 'id': 'kaggle-local-bootstrap', 'execution_count': None, 'metadata': {}, 'outputs': [],
            'source': source.splitlines(keepends=True)}
    notebook['cells'].insert(0, cell)
    for original in notebook['cells'][1:]:
        if original['cell_type'] == 'code':
            original['outputs'] = []
            original['execution_count'] = None
            compile(''.join(original['source']), str(notebook_path), 'exec')
    compile(source, '<kaggle bootstrap>', 'exec')
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / metadata['code_file']).write_text(json.dumps(notebook, indent=1), encoding='utf-8')
    (BUILD / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    manifest = {'branch': branch, 'kernel': metadata['id'],
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'submitted_source_sha256': source_hash(notebook),
                'sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [notebook_path, *modules]}}
    (BUILD / 'submission.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Prepared {metadata["id"]} from branch {branch}: {BUILD}')
    return metadata


def record_submission(output):
    match = re.search(r'Kernel version (\d+) successfully pushed', output)
    if not match:
        raise RuntimeError('Kaggle did not confirm a submitted version. Check status before retrying.')
    manifest = json.loads((BUILD / 'submission.json').read_text(encoding='utf-8'))
    manifest['version'] = int(match.group(1))
    manifest['acknowledged_utc'] = datetime.now(timezone.utc).isoformat()
    (BUILD / 'submitted.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def download(metadata):
    status = cli('kernels', 'status', metadata['id'])
    if 'KernelWorkerStatus.COMPLETE' not in status:
        raise RuntimeError('Kernel is not complete; result download postponed.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    downloaded = BUILD / 'downloads' / stamp
    downloaded.mkdir(parents=True)
    # The output/status APIs use the latest run, even when a version suffix is supplied.
    # Verify the completed remote source against the acknowledged local submission.
    submitted = BUILD / 'submitted.json'
    if not submitted.exists():
        raise RuntimeError('No acknowledged local submission. Refusing to label old outputs as this experiment.')
    manifest = json.loads(submitted.read_text(encoding='utf-8'))
    if manifest['kernel'] != metadata['id']:
        raise RuntimeError('Submitted kernel differs from the configured kernel.')
    remote = downloaded / 'remote'
    cli('kernels', 'pull', metadata['id'], '-p', str(remote), '-m')
    remote_metadata = json.loads((remote / 'kernel-metadata.json').read_text(encoding='utf-8'))
    notebook = json.loads((remote / remote_metadata['code_file']).read_text(encoding='utf-8'))
    if source_hash(notebook) != manifest['submitted_source_sha256']:
        raise RuntimeError('Latest completed remote source differs from the submitted source; outputs not collected.')
    cli('kernels', 'output', metadata['id'], '-p', str(downloaded),
        '-f', r'multibranch_attention_(cv|best|holdout)\.csv$')
    destination = ROOT / 'results' / 'kaggle' / stamp
    collect(downloaded, destination)
    shutil.copy2(submitted, destination / 'local_submission.json')
    for log in downloaded.glob('*.log'):
        shutil.copy2(log, destination / log.name)
    return destination


def wait_for_results(metadata, interval):
    previous = None
    while True:
        status = cli('kernels', 'status', metadata['id'])
        if status != previous:
            (BUILD / 'last_status.txt').write_text(status, encoding='utf-8')
            previous = status
        if 'KernelWorkerStatus.COMPLETE' in status:
            return download(metadata)
        if any(f'KernelWorkerStatus.{state}' in status for state in ('ERROR', 'CANCEL', 'CANCELLED', 'CANCELED')):
            (BUILD / 'failure.log').write_text(cli('kernels', 'logs', metadata['id']), encoding='utf-8')
            raise RuntimeError('Kaggle execution failed. See kaggle_build/multibranch_attention/failure.log.')
        time.sleep(interval)


def collect(download_dir, destination):
    # Some API versions package output as a zip; unpack with traversal protection.
    for archive in download_dir.rglob('*.zip'):
        with zipfile.ZipFile(archive) as zipped:
            for name in zipped.namelist():
                if not (archive.parent / name).resolve().is_relative_to(download_dir.resolve()):
                    raise RuntimeError('Unsafe path in downloaded output archive')
            zipped.extractall(archive.parent)
    groups = {}
    for path in download_dir.rglob('*.csv'):
        if path.name in CSV_NAMES:
            groups.setdefault(path.parent, set()).add(path.name)
    complete = [parent for parent, names in groups.items() if names == CSV_NAMES]
    if not complete:
        raise RuntimeError('No complete CV/best/holdout CSV set found. Check status and downloaded logs.')
    for parent in complete:
        target = destination / parent.relative_to(download_dir)
        target.mkdir(parents=True, exist_ok=True)
        for name in CSV_NAMES:
            shutil.copy2(parent / name, target / name)
        print(f'Results downloaded to {target}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['auth', 'prepare', 'push', 'status', 'logs', 'wait', 'download', 'pull'])
    parser.add_argument('--interval', type=int, default=60, help='Polling interval in seconds for wait (minimum 15).')
    args = parser.parse_args()
    metadata = json.loads(METADATA.read_text(encoding='utf-8'))
    try:
        if args.action == 'auth':
            subprocess.run([sys.executable, '-m', 'kaggle', 'auth', 'login'], check=True)
        elif args.action in ('prepare', 'push'):
            prepare()
            if args.action == 'push':
                record_submission(cli('kernels', 'push', '-p', str(BUILD)))
        elif args.action == 'status':
            cli('kernels', 'status', metadata['id'])
        elif args.action == 'logs':
            subprocess.run([sys.executable, '-u', '-m', 'kaggle', 'kernels', 'logs', metadata['id'], '--follow'], check=True)
        elif args.action == 'wait':
            wait_for_results(metadata, max(15, args.interval))
        elif args.action == 'pull':
            # Preserve the notebook being edited; remote copies go only in the build folder.
            cli('kernels', 'pull', metadata['id'], '-p', str(BUILD / 'remote'), '-m')
        else:
            download(metadata)
    except subprocess.CalledProcessError as error:
        print(error.stdout or '', end='', file=sys.stderr)
        print(error.stderr or '', end='', file=sys.stderr)
        parser.exit(1, 'Kaggle command failed. Check CLI installation, authentication, competition rules and GPU quota.\n')
    except (RuntimeError, FileNotFoundError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
