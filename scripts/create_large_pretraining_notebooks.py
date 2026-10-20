"""Build model-1/model-2 pretraining notebooks without overriding their candidates.

Notebook cells contain configuration and orchestration; extraction, training and
balanced Bayesian search stay in src/.
"""
import argparse
import ast
import json
from pathlib import Path
import re

if __package__:
    from .pretraining_search_configuration import configure_search_parameters
else:
    from pretraining_search_configuration import configure_search_parameters


ROOT = Path(__file__).resolve().parents[1]


def replace_assignment(source, name, value):
    source, count = re.subn(r'^' + re.escape(name) + r'\s*=.*$',
                            name + ' = ' + repr(value), source, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f'Expected one assignment for {name}, found {count}.')
    return source


def build_notebook(model):
    """Assemble a notebook in memory, preserving its editable model spaces."""
    notebook = json.loads((ROOT / 'archive/notebooks/multibranch_pretraining_v1.ipynb').read_text(encoding='utf-8'))

    def source(index):
        return ''.join(notebook['cells'][index]['source'])

    def set_source(index, value):
        notebook['cells'][index]['source'] = value.splitlines(keepends=True)

    name = f'multibranch_pretraining_model{model}_large'
    objective = 'supervised gesture pretraining' if model == 1 else 'masked-signal self-supervision'
    set_source(0, f"""# Multibranch pretraining: model {model}

Model {model} uses {objective}, followed by target-gesture fine-tuning.
Edit GRID_PARAM_SPACE and EVOLUTIONARY_PARAM_SPACE to choose model candidates;
PRETRAINING_PARAM_SPACE supplies the stage settings.

Validation defaults to complete sequences. Add 'subject' to VALIDATION_PARAM_SPACE
for a separate search with wearers held apart. Each protocol saves its own results.
Completed Bayesian candidates are saved to partial_cv.csv before the next trial.
""")
    config = source(3)
    for key, value in {
        'experiment_name': name, 'n_iter': 37, 'train_size': 0.7,
        'MODEL_VARIANT': f'model_{model}',
    }.items():
        config = replace_assignment(config, key, value)
    config += '\nBAYES_INITIAL_POINTS_PER_SOURCE = 3\n'
    set_source(3, config)

    training = source(17).replace(
        '        search_spaces=param_space,\n',
        "        search_spaces=param_space,\n"
        "        optimizer_kwargs={'n_initial_points': BAYES_INITIAL_POINTS_PER_SOURCE,\n"
        "                          'model_queue_size': 1, 'acq_optimizer': 'sampling',\n"
        "                          'acq_optimizer_kwargs': {'n_points': 256}},\n")
    set_source(17, training)
    configure_search_parameters(notebook)
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            tree = ast.parse(''.join(cell['source']))
            if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) for node in ast.walk(tree)):
                raise ValueError('Generated notebooks must not define functions.')
            cell['outputs'] = []
            cell['execution_count'] = None
        cell['metadata'].pop('execution', None)
        cell['metadata'].pop('papermill', None)
    return notebook


def generate(model, reference=None):
    # The optional argument remains accepted for existing callers. Reference
    # results no longer change any configured search candidates.
    notebook = build_notebook(model)
    filename = 'multibranch_pretraining.ipynb' if model == 1 else 'multibranch_ssl.ipynb'
    path = ROOT / 'notebooks' / filename
    path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    metadata = json.loads((ROOT / 'kaggle/pretraining-kernel-metadata.json').read_text(encoding='utf-8'))
    metadata.update(id=f'keithmarange/multibranch-model-{model}-large-cpu-bayes-37',
                    title=f'Multibranch model {model} large CPU Bayes 37', code_file=path.name)
    (ROOT / f'kaggle/pretraining-model{model}-large-kernel-metadata.json').write_text(
        json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path,
                        help='Deprecated compatibility option; reference results no longer override candidates.')
    parser.parse_args()
    for model in (1, 2):
        generate(model)


if __name__ == '__main__':
    main()
