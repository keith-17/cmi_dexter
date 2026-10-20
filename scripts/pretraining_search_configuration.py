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

_VALIDATION_SPACE = """# Sequence splitting is the default. Add 'subject' for a separate full search.
VALIDATION_PARAM_SPACE = {
    'validation_mode': ['sequence'],
    'group_column': ['subject'],
}
"""

_PARAMETER_SUFFIX = """# Add stage settings without replacing either editable search space.
for _key, _choices in PRETRAINING_PARAM_SPACE.items():
    GRID_PARAM_SPACE.setdefault(_key, _choices)
    EVOLUTIONARY_PARAM_SPACE.setdefault(_key, ECat(_choices))
BEST_PARAMS = {key: choices[0] for key, choices in GRID_PARAM_SPACE.items()}

# Build the search spaces from the configured candidates.
if SKOPT_AVAILABLE:
    BAYESIAN_PARAM_SPACE = multibranch_keras.prepare_multibranch_bayesian_space(GRID_PARAM_SPACE)
else:
    BAYESIAN_PARAM_SPACE = GRID_PARAM_SPACE
EVOLUTIONARY_SEARCH_SPACE = multibranch_keras.prepare_multibranch_evolutionary_space(EVOLUTIONARY_PARAM_SPACE)

_pipeline_params = pipeline.get_params(deep=True)
for space_name, candidate_space in (
    ('grid', GRID_PARAM_SPACE), ('bayesian', BAYESIAN_PARAM_SPACE),
    ('evolutionary', EVOLUTIONARY_SEARCH_SPACE),
):
    unknown_params = set(candidate_space) - set(_pipeline_params)
    if unknown_params:
        raise ValueError(f'{space_name}: unknown parameters={sorted(unknown_params)}')

if search_mode == 'bayes':
    param_space = BAYESIAN_PARAM_SPACE
elif search_mode == 'evo':
    param_space = EVOLUTIONARY_SEARCH_SPACE
else:
    param_space = GRID_PARAM_SPACE
"""


def _assignment(source, name):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node
    raise ValueError(f'Missing notebook assignment: {name}')


def configure_search_parameters(notebook):
    """Keep editable model/stage candidates and configure validation orchestration."""
    def source(index):
        return ''.join(notebook['cells'][index]['source'])

    def set_source(index, value):
        notebook['cells'][index]['source'] = value.splitlines(keepends=True)

    params = source(14)
    try:
        validation = ast.get_source_segment(params, _assignment(params, 'VALIDATION_PARAM_SPACE'))
    except ValueError:
        validation = _VALIDATION_SPACE

    imports = source(2)
    if 'from pretraining_search import run_validation_searches' not in imports:
        imports = imports.replace('import pretraining_utils as pretraining_keras\n',
                                  'import pretraining_utils as pretraining_keras\n'
                                  'from pretraining_search import run_validation_searches\n')
    set_source(2, imports)

    config = source(3)
    if '# The first CPU run' in config:
        start = config.index('# The first CPU run')
        stop = config.index('PROBLEMATIC_IDEAL_SKEW_THRESHOLD', start)
        config = config[:start] + config[stop:]
    config_lines = config.splitlines(keepends=True)
    for node in reversed(ast.parse(config).body):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in {
                'FIRST_RUN', 'LARGE_RUN', 'COVER_PRETRAINING_SOURCES',
                'REFERENCE_HOLDOUT_SCORE', 'REFERENCE_BEST_PARAMS',
                'PROBLEMATIC_IDEAL_SKEW_THRESHOLD', 'PROBLEMATIC_FEATURES_THRESHOLD',
                'PROBLEMATIC_FEATURE_COLUMNS',
            } for target in node.targets
        ):
            del config_lines[node.lineno - 1:node.end_lineno]
    config = ''.join(config_lines)
    config = config.replace('quick_save = True         # True -> save only the CV CSV with best_score & holdout_score appended; False -> full save',
                            'quick_save = True         # Always save CV/best/holdout CSVs; False also saves model weights.')
    if 'VALIDATION_PARAM_SPACE =' not in config:
        config += '\n' + validation.rstrip() + '\n'
    set_source(3, config)

    set_source(9, """# Each validation protocol splits complete sequences before stage selection.
experiment_df = train_df
del train_df
print('Available sequences:', experiment_df['sequence_id'].nunique())
""")

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
        'ideal_skew_threshold', 'problematic_features_threshold', 'problematic_feature_cols',
    }
    constructor = ''.join(line for line in constructor.splitlines(keepends=True)
                          if not re.match(r'\s*(' + '|'.join(stage_names) + r')\s*=', line))
    constructor = constructor.replace('    primary_target=TARGET_COL,\n',
                                      '    primary_target=TARGET_COL,\n'
                                      '    pretraining_validation_group_column=None,\n'
                                      '    validation_group_column=None,\n')
    set_source(11, constructor)

    # Preserve both user-editable model spaces verbatim. Discard generated overrides
    # after them, including pilot settings and automatic parameter expansion.
    evolutionary = _assignment(params, 'EVOLUTIONARY_PARAM_SPACE')
    prefix = ''.join(params.splitlines(keepends=True)[:evolutionary.end_lineno])
    stage = _assignment(params, 'PRETRAINING_PARAM_SPACE')
    stage_source = ast.get_source_segment(params, stage)
    if any(name.id in {'FIRST_RUN', 'PRETRAINING_EPOCHS', 'BASE_LEARNING_RATE'}
           for name in ast.walk(stage) if isinstance(name, ast.Name)):
        stage_source = ast.get_source_segment(
            _STAGE_SPACE, _assignment(_STAGE_SPACE, 'PRETRAINING_PARAM_SPACE'))
    else:
        # Older converted notebooks duplicated fine-tuning settings here.
        # They already belong to GRID_PARAM_SPACE / EVOLUTIONARY_PARAM_SPACE.
        lines = stage_source.splitlines(keepends=True)
        remove_lines = set()
        for key, value in zip(stage.value.keys, stage.value.values):
            if isinstance(key, ast.Constant) and key.value in {'estimator__epochs', 'estimator__learning_rate'}:
                remove_lines.update(range(key.lineno - stage.lineno, value.end_lineno - stage.lineno + 1))
        stage_source = ''.join(line for index, line in enumerate(lines) if index not in remove_lines)
    set_source(14, prefix.rstrip() + '\n\n' + stage_source.rstrip() + '\n\n' + _PARAMETER_SUFFIX)

    training = source(17)
    start = training.find('search.candidate_checkpoint_path')
    if start < 0:
        start = training.find('search.fit(X_train, y_train, groups=groups)')
    if start < 0:
        start = training.index('validation_runs = run_validation_searches(')
    training = training[:start].replace(
        '# Preserve saved values even when outside the current search ranges.',
        '# Use the first configured grid choices for this single-candidate run.')
    set_source(17, training + _TRAINING_SUFFIX)
    set_source(20, source(20).replace('save_dir = results_dir / timestamp', "save_dir = last_run['results_dir']"))
    set_source(22, """# Each protocol's holdout was evaluated once after its own search/refit.
if experiment_succeeded:
    print('Last validation protocol:', last_run['validation_mode'])
    print('Holdout competition score:', eval_results['competition_score'])
pd.DataFrame(protocol_results)
""")
    set_source(24, """# CV, best, holdout, histories and diagnostics are saved separately per protocol.
protocol_summary = pd.DataFrame(protocol_results)
print(protocol_summary.to_string(index=False))
print('Results:', validation_runs['results_dir'])
""")

    for index in (2, 3, 9, 11, 14, 17, 20, 22, 24):
        ast.parse(source(index))
        notebook['cells'][index]['outputs'] = []
        notebook['cells'][index]['execution_count'] = None
    return notebook
