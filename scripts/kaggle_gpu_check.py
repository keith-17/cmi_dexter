"""Submit a temporary GPU validation build; never edit experiment defaults."""
import json
from pathlib import Path

import kaggle_experiment as workflow


def code_cell(cell_id, source):
    compile(source, cell_id, 'exec')
    return dict(cell_type='code', id=cell_id, metadata={}, outputs=[],
                execution_count=None, source=source.splitlines(keepends=True))


def main():
    metadata = workflow.prepare(notebook_name='multibranch_v1.ipynb', use_attached_source=True)
    path = workflow.BUILD / metadata['code_file']
    notebook = json.loads(path.read_text(encoding='utf-8'))
    # Sample only after validating the complete input and recording exclusions.
    data_index = next(i for i, cell in enumerate(notebook['cells'])
                      if 'raw_train_df' in ''.join(cell.get('source', [])))
    notebook['cells'].insert(data_index + 1, code_cell('gpu-check-data', '''
print('GPU VALIDATION ONLY: up to 16 sequences per target class, 2 candidates, 1 epoch.')
_sequence_labels = train_df[['sequence_id', TARGET_COL]].drop_duplicates('sequence_id')
_selected_ids = _sequence_labels.groupby(TARGET_COL, group_keys=False).apply(
    lambda group: group.sample(n=min(16, len(group)), random_state=random_state)
)['sequence_id']
train_df = train_df[train_df['sequence_id'].isin(_selected_ids)].copy()
print('Validation subset sequences:', train_df['sequence_id'].nunique())
'''))
    train_index = next(i for i, cell in enumerate(notebook['cells'])
                       if 'search.fit(X_train' in ''.join(cell.get('source', [])))
    notebook['cells'].insert(train_index, code_cell('gpu-check-budget', '''
population_size = 2
generations = 0
tournament_size = min(tournament_size, population_size)
param_space = dict(param_space)
param_space['estimator__epochs'] = ECat([1])
error_score = 'raise'
pipeline.set_params(estimator__invalid_candidate_policy='raise')
'''))
    path.write_text(json.dumps(notebook, indent=1), encoding='utf-8')
    manifest_path = workflow.BUILD / 'submission.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest.update(run_stage='gpu_validation', diagnostic_overrides={
        'population_size': 2, 'generations': 0, 'tournament_size': 2, 'epochs': 1,
        'max_sequences_per_class': 16,
    }, submitted_source_sha256=workflow.source_hash(notebook))
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    workflow.record_submission(workflow.cli('kernels', 'push', '-p', str(workflow.BUILD)))
    acknowledged = workflow.BUILD / 'submitted.json'
    (workflow.BUILD / 'gpu_validation_submitted.json').write_bytes(acknowledged.read_bytes())


if __name__ == '__main__':
    main()
