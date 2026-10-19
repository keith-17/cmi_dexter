"""Move stage settings into the same editable candidate lists as model settings.

The transformation is shared by the large-notebook generator and the active
notebooks. It changes configuration and orchestration, not model behaviour.
"""

from __future__ import annotations

import ast
import re


_STAGE_SPACE = '''# Stage settings use the same candidate lists for Bayesian and evolutionary search.
# One value fixes a parameter; add values to search it. Baseline skips pretraining.
PRETRAINING_PARAM_SPACE = {
    'estimator__epochs': [150],
    'estimator__learning_rate': [1e-4],
    'estimator__pretraining_mode': [PRETRAINING_MODE],
    'estimator__pretraining_epochs': [30],
    'estimator__pretraining_batch_size': [128],
    'estimator__pretraining_learning_rate': [1e-4],
    'estimator__pretraining_optimizer_name': ['adamw'],
    'estimator__pretraining_weight_decay': [0.0],
    'estimator__pretraining_momentum': [0.0],
    'estimator__pretraining_gradient_clip_norm': [None],
    'estimator__pretraining_gradient_clip_value': [None],
    'estimator__pretraining_loss_name': ['sparse_cce'],
    'estimator__pretraining_label_smoothing': [0.0],
    'estimator__pretraining_class_weight_mode': ['balanced'],
    'estimator__pretraining_validation_split': [0.15],
    # The validation protocol sets these two grouping fields together.
    'estimator__pretraining_validation_group_column': [None],
    'estimator__validation_group_column': [None],
    'estimator__pretraining_early_stopping_patience': [8],
    'estimator__pretraining_use_lr_scheduler': [True],
    'estimator__pretraining_lr_scheduler_type': ['cosine'],
    'estimator__pretraining_lr_factor': [0.5],
    'estimator__pretraining_lr_patience': [3],
    'estimator__pretraining_min_lr': [1e-6],
    'estimator__pretraining_cosine_t_max': [30],
    'estimator__pretraining_warmup_epochs': [0],
    'estimator__pretraining_model_params': [None],
    'estimator__pretraining_augmentor': [None],
    'estimator__pretraining_shuffle': [True],
    'estimator__pretraining_verbose': [None],
    'estimator__finetuning_head_epochs': [0],
    'estimator__finetuning_head_learning_rate': [None],
    # None follows estimator__learning_rate, so that searched value stays active.
    'estimator__finetuning_learning_rate': [None],
    'estimator__ssl_mask_ratio': [0.15],
    'estimator__ssl_span_length': [8],
    'estimator__ssl_mask_value': [0.0],
    'estimator__ssl_mask_columns': [None],
    'estimator__ssl_mask_modalities_together': [True],
    'estimator__ssl_reconstruction_loss': ['huber'],
    'estimator__ssl_huber_delta': [1.0],
    'estimator__ssl_decoder_units': [128],
    'estimator__ssl_decoder_dropout': [0.0],
    'estimator__ssl_loss_channel_weights': [None],
    'estimator__ssl_standardize_targets': [True],
    'estimator__ssl_invalid_values': [None],
    'estimator__ssl_validation_mask_seed': [1729],
    'estimator__ssl_resample_mask_each_epoch': [True],
    'estimator__ssl_cache_sequences': [16],
    'estimator__extractor__pretraining_source': ['non_target', 'all', 'target', 'baseline'],
    'estimator__extractor__pretraining_label_column': ['gesture'],
    'estimator__extractor__pretraining_include_classes': [None],
    'estimator__extractor__pretraining_exclude_classes': [None],
    'estimator__extractor__pretraining_filters': [None],
    'estimator__extractor__pretraining_exclude_filters': [None],
    'estimator__extractor__pretraining_min_length': [None],
    'estimator__extractor__pretraining_max_length': [None],
    'estimator__extractor__pretraining_sample_fraction': [1.0],
    'estimator__extractor__pretraining_max_sequences': [None],
    'estimator__extractor__pretraining_max_sequences_per_class': [None],
    'estimator__extractor__pretraining_balance_classes': [False],
    'estimator__extractor__pretraining_random_state': [random_state],
    'estimator__extractor__finetuning_source': ['target'],
    'estimator__extractor__finetuning_label_column': [TARGET_COL],
    'estimator__extractor__finetuning_include_classes': [None],
    'estimator__extractor__finetuning_exclude_classes': [None],
    'estimator__extractor__finetuning_filters': [None],
    'estimator__extractor__finetuning_exclude_filters': [None],
    'estimator__extractor__stage_target_column': ['is_target'],
    'estimator__extractor__stage_target_values': [(1,)],
}

# Optional evolutionary numeric ranges, for example an EInt or EFloat value.
# Shared stage candidates above are used unless a key is overridden here.
EVOLUTIONARY_STAGE_OVERRIDES = {}

FIRST_RUN_PARAM_SPACE = {
    'estimator__pretraining_epochs': [5],
    'estimator__epochs': [5],
}
ACTIVE_STAGE_PARAM_SPACE = dict(PRETRAINING_PARAM_SPACE)
if FIRST_RUN:
    ACTIVE_STAGE_PARAM_SPACE.update(FIRST_RUN_PARAM_SPACE)

# Sequence splitting is the default. Adding 'subject' runs a separate full search.
VALIDATION_PARAM_SPACE = {
    'validation_mode': ['sequence'],
    'group_column': ['subject'],
}
'''

_TRAINING_SUFFIX = '''validation_runs = run_validation_searches(
    search, experiment_df,
    validation_param_space=VALIDATION_PARAM_SPACE,
    sensor_columns=sensor_columns,
    train_size=train_size, n_splits=n_splits, random_state=random_state,
    target_col=TARGET_COL, results_dir=results_dir,
    experiment_name=experiment_name, timestamp=timestamp, quick_save=quick_save,
)
last_run = validation_runs['last_run']
protocol_results = validation_runs['protocol_results']
search = last_run.get('search')
X_train, X_test = last_run.get('X_train'), last_run.get('X_test')
y_train, y_test = last_run.get('y_train'), last_run.get('y_test')
groups, cv_object = last_run.get('groups'), last_run.get('cv_object')
experiment_succeeded = last_run.get('experiment_succeeded', False)
eval_results = last_run.get('eval_results', {})
'''

_MANIFEST = '''with open(results_dir / f'{experiment_name}_configuration.json', 'w', encoding='utf-8') as stream:
    json.dump({'model_variant': MODEL_VARIANT, 'pretraining_mode': PRETRAINING_MODE,
               'iterations': n_iter, 'train_size': train_size,
               'validation_param_space': VALIDATION_PARAM_SPACE,
               'reference_holdout_score': REFERENCE_HOLDOUT_SCORE,
               'fixed_params': LARGE_FIXED_PARAMS, 'varying_params': LARGE_PARAM_SPACE,
               'pretraining_param_space': PRETRAINING_PARAM_SPACE,
               'active_stage_param_space': ACTIVE_STAGE_PARAM_SPACE,
               'first_run': FIRST_RUN, 'first_run_param_space': FIRST_RUN_PARAM_SPACE,
               'evolutionary_stage_overrides': EVOLUTIONARY_STAGE_OVERRIDES},
              stream, indent=2, default=str)
print('Reference-only unsupported parameters:', LARGE_UNSUPPORTED_REFERENCE_PARAMS)
print('Pretraining sources:', ACTIVE_STAGE_PARAM_SPACE['estimator__extractor__pretraining_source'])
print('Validation choices:', VALIDATION_PARAM_SPACE['validation_mode'])
print('Bayesian candidates per validation choice:', n_iter)

'''


def _assignment(source, name):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node
    raise ValueError(f'Missing notebook assignment: {name}')


def _replace_assignment(source, name, replacement):
    node = _assignment(source, name)
    lines = source.splitlines(keepends=True)
    return ''.join(lines[:node.lineno - 1]) + replacement.rstrip() + '\n' + ''.join(lines[node.end_lineno:])


def configure_search_parameters(notebook):
    """Configure stage search lists and matching validation in a large notebook."""
    def source(index):
        return ''.join(notebook['cells'][index]['source'])

    def set_source(index, value):
        notebook['cells'][index]['source'] = value.splitlines(keepends=True)

    # Already converted notebooks retain their user-edited candidate lists.
    if 'ACTIVE_STAGE_PARAM_SPACE' in source(14):
        return notebook

    imports = source(2)
    if 'from pretraining_search import run_validation_searches' not in imports:
        imports = imports.replace('import pretraining_utils as pretraining_keras\n',
                                  'import pretraining_utils as pretraining_keras\n'
                                  'from pretraining_search import run_validation_searches\n')
    set_source(2, imports)

    config = source(3)
    start = config.index('# The first CPU run')
    stop = config.index('PROBLEMATIC_IDEAL_SKEW_THRESHOLD', start)
    config = config[:start] + config[stop:]
    config = config.replace('quick_save = True         # True -> save only the CV CSV with best_score & holdout_score appended; False -> full save',
                            'quick_save = True         # Always save CV/best/holdout CSVs; False also saves model weights.')
    set_source(3, config)

    set_source(9, '''# Keep complete sequences available; each selected validation protocol splits them
# before stage selection and extraction. Sequence splitting remains the default.
experiment_df = train_df
del train_df
print('Available sequences:', experiment_df['sequence_id'].nunique())
''')

    constructor = source(11)
    stage_names = {
        'pretraining_source', 'pretraining_label_column', 'pretraining_include_classes',
        'pretraining_exclude_classes', 'pretraining_filters', 'pretraining_exclude_filters',
        'pretraining_min_length', 'pretraining_max_length', 'pretraining_sample_fraction',
        'pretraining_max_sequences', 'pretraining_max_sequences_per_class',
        'pretraining_balance_classes', 'pretraining_random_state', 'finetuning_source',
        'finetuning_label_column', 'finetuning_include_classes', 'finetuning_exclude_classes',
        'finetuning_filters', 'finetuning_exclude_filters', 'stage_target_column',
        'stage_target_values', 'pretraining_mode', 'pretraining_epochs',
        'pretraining_learning_rate', 'pretraining_use_lr_scheduler',
        'pretraining_early_stopping_patience', 'learning_rate', 'epochs',
        'pretraining_validation_group_column', 'validation_group_column',
    }
    constructor = ''.join(line for line in constructor.splitlines(keepends=True)
                          if not re.match(r'\s*(' + '|'.join(stage_names) + r')\s*=', line))
    constructor = constructor.replace('    primary_target=TARGET_COL,\n',
                                      '    primary_target=TARGET_COL,\n'
                                      '    pretraining_validation_group_column=None,\n'
                                      '    validation_group_column=None,\n')
    set_source(11, constructor)

    params = source(14)
    params = _replace_assignment(params, 'PRETRAINING_PARAM_SPACE', _STAGE_SPACE)
    params = params.replace('GRID_PARAM_SPACE.update(PRETRAINING_PARAM_SPACE)\n'
                            'for _key, _choices in PRETRAINING_PARAM_SPACE.items():\n'
                            '    EVOLUTIONARY_PARAM_SPACE[_key] = ECat(_choices)\n', '')
    # The old pilot block was overwritten by the large block; retain only its
    # requested five-epoch limits, now declared alongside the stage lists.
    start = params.index('# Preserve the V1 full spaces separately')
    stop = params.index('# Large-run overrides', start)
    params = params[:start] + params[stop:]
    params = params.replace('LARGE_FIXED_PARAMS.update({key: values[0] for key, values in PRETRAINING_PARAM_SPACE.items()})\n', '')
    stage_keys = {ast.literal_eval(key) for key in _assignment(_STAGE_SPACE, 'PRETRAINING_PARAM_SPACE').value.keys}
    # Stage defaults now have one editable home. Keep all other existing large
    # architecture settings verbatim, including any user edits.
    lines = params.splitlines(keepends=True)
    inside_large = False
    kept = []
    for line in lines:
        if line.startswith('LARGE_FIXED_PARAMS.update({'):
            inside_large = True
        key = re.match(r"\s*['\"]([^'\"]+)['\"]\s*:", line)
        if not (inside_large and key and key.group(1) in stage_keys):
            kept.append(line)
        if inside_large and line.startswith('})'):
            inside_large = False
    params = ''.join(kept)
    params = params.replace('for _key, _value in LARGE_FIXED_PARAMS.items():\n',
                            'LARGE_FIXED_PARAMS = {key: value for key, value in LARGE_FIXED_PARAMS.items()\n'
                            '                      if key not in PRETRAINING_PARAM_SPACE}\n'
                            'for _key, _value in LARGE_FIXED_PARAMS.items():\n')
    params = params.replace("    'estimator__extractor__pretraining_source': PRETRAINING_SOURCE_CHOICES,\n", '')
    params = params.replace('GRID_PARAM_SPACE.update(LARGE_PARAM_SPACE)\n',
                            'GRID_PARAM_SPACE.update(LARGE_PARAM_SPACE)\n'
                            'GRID_PARAM_SPACE.update(ACTIVE_STAGE_PARAM_SPACE)\n')
    params = params.replace('EVOLUTIONARY_PARAM_SPACE = {key: ECat(values) for key, values in GRID_PARAM_SPACE.items()}\n',
                            'EVOLUTIONARY_PARAM_SPACE = {key: ECat(values) for key, values in GRID_PARAM_SPACE.items()}\n'
                            'EVOLUTIONARY_PARAM_SPACE.update(EVOLUTIONARY_STAGE_OVERRIDES)\n'
                            'if FIRST_RUN:\n'
                            '    EVOLUTIONARY_PARAM_SPACE.update({key: ECat(values) for key, values in FIRST_RUN_PARAM_SPACE.items()})\n')
    start = params.index("with open(results_dir / f'{experiment_name}_configuration.json'")
    stop = params.index('# ============================================================\n# Build and validate', start)
    params = params[:start] + _MANIFEST + params[stop:]
    params = params.replace("print('Epochs:', PRETRAINING_EPOCHS, 'pretraining +', FINETUNING_EPOCHS, 'fine-tuning; LR:', BASE_LEARNING_RATE)",
                            "print('Epoch candidates:', ACTIVE_STAGE_PARAM_SPACE['estimator__pretraining_epochs'], 'pretraining;',\n"
                            "      ACTIVE_STAGE_PARAM_SPACE['estimator__epochs'], 'fine-tuning; LR:',\n"
                            "      ACTIVE_STAGE_PARAM_SPACE['estimator__learning_rate'])")
    set_source(14, params)

    training = source(17)
    start = training.index('search.candidate_checkpoint_path')
    set_source(17, training[:start] + _TRAINING_SUFFIX)
    set_source(20, source(20).replace('save_dir = results_dir / timestamp', "save_dir = last_run['results_dir']"))
    set_source(22, '''# Each protocol's holdout was evaluated once after its own search/refit.
if experiment_succeeded:
    print('Last validation protocol:', last_run['validation_mode'])
    print('Holdout competition score:', eval_results['competition_score'])
pd.DataFrame(protocol_results)
''')
    set_source(24, '''# CV, best, holdout, histories and diagnostics are saved separately per protocol.
protocol_summary = pd.DataFrame(protocol_results)
print(protocol_summary.to_string(index=False))
print('Results:', validation_runs['results_dir'])
''')

    markdown = source(0)
    markdown = markdown.replace('The four pretraining sources share the same architecture ranges and receive\n9 or 10 evaluations each. Their first visits are interleaved. Each source uses\nthree initial proposals, followed by Bayesian adaptation.',
                                'The configured pretraining sources share the same architecture ranges.\nTheir first visits are interleaved, followed by Bayesian adaptation.')
    markdown = re.sub(r'Validation defaults to complete sequences,.*?old score remains a reference\.',
                      "Validation defaults to complete sequences. Add 'subject' to VALIDATION_PARAM_SPACE\n"
                      'only to run a separate search with wearers held apart. Each protocol saves its\n'
                      'own CV, best and holdout CSVs. The old score remains a reference.', markdown, flags=re.DOTALL)
    set_source(0, markdown)

    for index in (2, 3, 9, 11, 14, 17, 20, 22, 24):
        ast.parse(source(index))
        notebook['cells'][index]['outputs'] = []
        notebook['cells'][index]['execution_count'] = None
    return notebook
