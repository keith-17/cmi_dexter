"""Apply the focused evolutionary pilot configuration without altering model code."""
import ast
import json
from pathlib import Path
import re

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / 'notebooks' / 'multibranch_v1.ipynb'

CONTROLS = '''# Evolutionary pilot: at most 24 CV candidates, plus one final refit.
# Probabilities are per pair (crossover) and per parameter (mutation).
population_size = 6
generations = 3
tournament_size = 2
elitism = True
crossover_probability = 0.8
mutation_probability = 0.05

'''


def process_cell(cell):
    if cell.get('cell_type') != 'code':
        return
    source = ''.join(cell.get('source', []))
    if 'population_size =' in source and 'experiment_name =' in source:
        source = re.sub(r'# (?:Separate evolutionary search parameters[^\n]*|Evolutionary pilot:[^\n]*)\n.*?(?=orientation_filter_list =)', CONTROLS, source, flags=re.S)
    if 'EVOLUTIONARY_PARAM_SPACE = {' in source:
        before, evolution = source.split('EVOLUTIONARY_PARAM_SPACE = {', 1)
        body, after = evolution.split('# Build and validate', 1)
        body = body.replace("'bigru_attention', 'phase_attention', 'cnn_attention_pool', 'bert'", "'bigru_attention'")
        body = body.replace("'estimator__use_bigru': ECat([False, True])", "'estimator__use_bigru': ECat([True])")
        tree = ast.parse('EVOLUTIONARY_PARAM_SPACE = {' + body)
        dictionary = tree.body[0].value
        lines = ('EVOLUTIONARY_PARAM_SPACE = {' + body).splitlines(keepends=True)
        prefixes = ('estimator__bert_', 'estimator__phase_', 'estimator__cnn_', 'estimator__tof_spatial_')
        remove = {'estimator__n_phases', 'estimator__tof_gru_units',
                  'estimator__extractor__ideal_skew_threshold',
                  'estimator__extractor__problematic_features_threshold',
                  'estimator__extractor__problematic_feature_cols'}
        for key, value in reversed(list(zip(dictionary.keys, dictionary.values))):
            if key.value in remove or key.value.startswith(prefixes):
                del lines[key.lineno - 1:value.end_lineno]
        source = before + ''.join(lines) + '# Build and validate' + after
    if 'search = GASearchCV(' in source:
        source = re.sub(r'^\s*scoring=None,.*\n', '', source, flags=re.M)
        source = re.sub(r'^\s*(?:keep_top_k|algorithm)=.*\n', '', source, flags=re.M)
        before, ga = source.split('search = GASearchCV(', 1)
        boundary = re.search(r'\n(?:elif|else)\b', ga)
        ga_block, after = ga[:boundary.start()], ga[boundary.start():]
        ga_block = re.sub(r'^\s*n_jobs=n_jobs,\n', '', ga_block, flags=re.M)
        source = before + 'search = GASearchCV(' + ga_block + after
    compile(source, '<notebook cell>', 'exec')
    if source != ''.join(cell.get('source', [])):
        cell['source'] = source.splitlines(keepends=True)
        cell['outputs'] = []
        cell['execution_count'] = None
        cell['metadata'].pop('execution', None)
        cell['metadata'].pop('papermill', None)


def main():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        process_cell(cell)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
