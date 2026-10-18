"""Create the pretraining notebook from the existing V1 notebook layout.

The generated notebook deliberately contains configuration and orchestration only:
all extraction, training, masking, transfer and scoring live in src utilities.
"""

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    notebook = json.loads((ROOT / "notebooks/multibranch_v1.ipynb").read_text(encoding="utf-8"))

    def source(index):
        return "".join(notebook["cells"][index]["source"])

    def replace(index, text):
        notebook["cells"][index]["source"] = text.strip("\n").splitlines(keepends=True)

    replace(0, r'''
# Multi-branch pretraining V1 experiment

The same configuration → preparation → parameters → training → analysis/save layout as `multibranch_v1.ipynb`. Sequence extraction and stage selection come from `base_utils.py`; the advanced models are in `pretraining_utils.py` and reuse the existing multibranch architecture.

Choose `MODEL_VARIANT = 'model_1'` for supervised non-target pretraining or `'model_2'` for masked-signal self-supervised pretraining. Both finish with target-gesture fine-tuning. Configure class lists, metadata filters, source domains, sequence lengths and sampling inside the sequence extractor settings below. No functions are defined in this notebook.

### Search modes
Choose `search_mode = 'best params'`, `'evo'`, `'bayes'`, or `'grid'`. Every model, extractor and augmentation leaf parameter appears in all three search spaces; omitted candidate ranges remain explicit fixed defaults.

### First CPU run
`FIRST_RUN=True` selects model 1, five Bayesian candidates searching BERT width and pretraining source, the extractor's default feature spaces, and five epochs per stage. Source choices are non-target, all, target, and baseline (skip pretraining completely). This is a CPU pilot; set `FIRST_RUN=False` to use the full original search ranges and 150 fine-tuning epochs. Augmentations, regularisation, early stopping and scheduling are disabled for the pilot. The fixed learning rate remains the original saved configuration's `1e-4`.

The holdout and CV partitions separate wearers before either training stage. Non-target sequences remain in each training partition for pretraining; only the configured fine-tuning domain is scored.

### Kaggle workflow
```powershell
& .venv-kaggle\Scripts\python.exe scripts/kaggle_pretraining_experiment.py prepare --notebook multibranch_pretraining_v1.ipynb --iterations 5
& .venv-kaggle\Scripts\python.exe scripts/kaggle_pretraining_experiment.py push --notebook multibranch_pretraining_v1.ipynb --iterations 5
```
''')
    replace(1, '''# Multi-Branch BERT / Attention — Transfer and Self-Supervised Learning

Model 1 predicts individual pretraining gesture labels, transfers the shared branch/fusion/BERT weights, then fine-tunes the downstream classifier. Model 2 learns masked sensor reconstruction before the same fine-tuning stage. Switch between them in the configuration cell; architecture and preprocessing controls remain shared.

The notebook follows the existing V1 layout: configuration, data loading, split, estimator, grid/Bayesian/evolutionary parameters, holdout evaluation and saving.
''')
    replace(2, source(2).replace(
        "import multibranch_utils as multibranch_keras\nimportlib.reload(multibranch_keras)",
        "import multibranch_utils as multibranch_keras\nimport pretraining_utils as pretraining_keras\nimportlib.reload(multibranch_keras)\nimportlib.reload(pretraining_keras)"
    ))
    config = source(3).replace("experiment_name = 'multibranch_v1'", "experiment_name = 'multibranch_pretraining_v1'")
    config = config.replace("n_iter = 37", "n_iter = 5")
    config = config.replace("problematic_sequence_bool = False", "problematic_sequence_bool = True")
    config = config.replace("target_only_bool = True", "target_only_bool = False  # Keep both domains until the extractor selects each stage.")
    config += r'''

# Model 1 = supervised transfer; model 2 = masked-signal SSL then fine-tuning.
MODEL_VARIANT = 'model_1'  # 'model_1' or 'model_2'
FIRST_RUN = True
MODEL_MODES = {'model_1': 'supervised', 'model_2': 'masked_ssl'}
if MODEL_VARIANT not in MODEL_MODES:
    raise ValueError("MODEL_VARIANT must be 'model_1' or 'model_2'.")
PRETRAINING_MODE = MODEL_MODES[MODEL_VARIANT]

# The first CPU run uses 5 epochs per stage; full experiments retain V1's 150.
FIRST_RUN_PRETRAINING_EPOCHS = 5
FIRST_RUN_FINETUNING_EPOCHS = 5
PRETRAINING_EPOCHS = FIRST_RUN_PRETRAINING_EPOCHS if FIRST_RUN else 10
FINETUNING_EPOCHS = FIRST_RUN_FINETUNING_EPOCHS if FIRST_RUN else 150
BASE_LEARNING_RATE = 1e-4  # Original V1 saved learning rate, kept nonzero and fixed.
PRETRAINING_SOURCE = 'non_target' if MODEL_VARIANT == 'model_1' else 'all'
PRETRAINING_SOURCE_CHOICES = ['non_target', 'all', 'target', 'baseline']
# 'baseline' skips the pretraining stage entirely, for a matched supervised control.
PRETRAINING_LABEL_COLUMN = 'gesture'  # Preserve individual non-target gesture labels.
PRETRAINING_INCLUDE_CLASSES = None  # Example: ['Write name in air', 'Wave hello']
PRETRAINING_EXCLUDE_CLASSES = None
PRETRAINING_FILTERS = None  # Dict of metadata column -> allowed values.
PRETRAINING_EXCLUDE_FILTERS = None
PRETRAINING_MIN_LENGTH = None
PRETRAINING_MAX_LENGTH = None
PRETRAINING_SAMPLE_FRACTION = 1.0
PRETRAINING_MAX_SEQUENCES = None
PRETRAINING_MAX_SEQUENCES_PER_CLASS = None
PRETRAINING_BALANCE_CLASSES = False
FINETUNING_SOURCE = 'target'
FINETUNING_INCLUDE_CLASSES = None
FINETUNING_EXCLUDE_CLASSES = None
FINETUNING_FILTERS = None
FINETUNING_EXCLUDE_FILTERS = None
STAGE_TARGET_COLUMN = 'is_target'
STAGE_TARGET_VALUES = (1,)

# All records from a wearer stay in one holdout/CV partition, including pretraining.
GROUP_COLUMN = 'subject'
PROBLEMATIC_IDEAL_SKEW_THRESHOLD = 1.6
PROBLEMATIC_FEATURES_THRESHOLD = 7
PROBLEMATIC_FEATURE_COLUMNS = ['acc_x', 'acc_y', 'acc_z', 'rot_x', 'rot_y', 'rot_w', 'rot_z']
'''
    replace(3, config)
    prep = source(7).replace(
        "if target_only_bool:\n    keep_rows &= train_df['is_target'].eq(1)",
        "if target_only_bool:\n    raise ValueError('Keep target_only_bool=False; select target/non-target within the stage extractor.')"
    )
    replace(7, prep)
    replace(9, '''# Preserve the V1 data-source and row-aligned-label layout, with wearer grouping.
if GROUP_COLUMN not in train_df:
    raise ValueError(f'Missing group column {GROUP_COLUMN!r}; use a wearer/session identifier.')
sequence_df = train_df[['sequence_id', GROUP_COLUMN]].drop_duplicates('sequence_id').sort_values('sequence_id')
if sequence_df[GROUP_COLUMN].nunique() < 3:
    raise ValueError('Need at least three wearers/groups for independent holdout and CV training.')
splitter = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
train_idx, test_idx = next(splitter.split(sequence_df, groups=sequence_df[GROUP_COLUMN]))
train_ids = sequence_df.iloc[train_idx]['sequence_id']
test_ids = sequence_df.iloc[test_idx]['sequence_id']
train_sample_df = train_df[train_df['sequence_id'].isin(train_ids)].copy()
hold_out_df = train_df[train_df['sequence_id'].isin(test_ids)].copy()

X_train = train_sample_df
X_test = hold_out_df
# Keep both source domains in X/y; fitting selects pretraining and fine-tuning separately.
label_columns = [column for column in X_train.columns
                 if column not in sensor_columns and column not in {'row_id', 'sequence_counter'}]
y_train = X_train[label_columns].copy()
y_test = X_test[label_columns].copy()
groups = X_train[GROUP_COLUMN]

print('Train sequences:', X_train['sequence_id'].nunique())
print('Holdout sequences:', X_test['sequence_id'].nunique())
print('Train wearers:', X_train[GROUP_COLUMN].nunique(), 'Holdout wearers:', X_test[GROUP_COLUMN].nunique())
assert set(X_train[GROUP_COLUMN]).isdisjoint(set(X_test[GROUP_COLUMN]))
del train_df, train_sample_df, hold_out_df
''')
    replace(11, '''# Explicit [{}] is equivalent to the extractor's default STFT/CWT configuration.
# It also keeps those defaults serialisable in grid/Bayesian/evolutionary spaces.
extractor = SequenceExtractor(
    output_format='chunks',
    stft_configs=[{}],
    cwt_configs=[{}],
    filter_problematic_sequences=problematic_sequence_bool,
    ideal_skew_threshold=PROBLEMATIC_IDEAL_SKEW_THRESHOLD,
    problematic_features_threshold=PROBLEMATIC_FEATURES_THRESHOLD,
    problematic_feature_cols=PROBLEMATIC_FEATURE_COLUMNS,
    pretraining_source=PRETRAINING_SOURCE,
    pretraining_label_column=PRETRAINING_LABEL_COLUMN,
    pretraining_include_classes=PRETRAINING_INCLUDE_CLASSES,
    pretraining_exclude_classes=PRETRAINING_EXCLUDE_CLASSES,
    pretraining_filters=PRETRAINING_FILTERS,
    pretraining_exclude_filters=PRETRAINING_EXCLUDE_FILTERS,
    pretraining_min_length=PRETRAINING_MIN_LENGTH,
    pretraining_max_length=PRETRAINING_MAX_LENGTH,
    pretraining_sample_fraction=PRETRAINING_SAMPLE_FRACTION,
    pretraining_max_sequences=PRETRAINING_MAX_SEQUENCES,
    pretraining_max_sequences_per_class=PRETRAINING_MAX_SEQUENCES_PER_CLASS,
    pretraining_balance_classes=PRETRAINING_BALANCE_CLASSES,
    pretraining_random_state=random_state,
    finetuning_source=FINETUNING_SOURCE,
    finetuning_label_column=TARGET_COL,
    finetuning_include_classes=FINETUNING_INCLUDE_CLASSES,
    finetuning_exclude_classes=FINETUNING_EXCLUDE_CLASSES,
    finetuning_filters=FINETUNING_FILTERS,
    finetuning_exclude_filters=FINETUNING_EXCLUDE_FILTERS,
    stage_target_column=STAGE_TARGET_COLUMN,
    stage_target_values=STAGE_TARGET_VALUES,
)

clf = pretraining_keras.AdvancedMultiBranchSequenceClassifier(
    extractor=extractor,
    augmentor=SensorAugmentor(
        sequence_col='sequence_id', counter_col='sequence_counter',
        prob=0.0, per_aug_prob=0.0,
    ),
    primary_target=TARGET_COL,
    pretraining_mode=PRETRAINING_MODE,
    pretraining_epochs=PRETRAINING_EPOCHS,
    pretraining_learning_rate=BASE_LEARNING_RATE,
    pretraining_use_lr_scheduler=False,
    pretraining_early_stopping_patience=0 if FIRST_RUN else 3,
    pretraining_validation_group_column=GROUP_COLUMN,
    validation_group_column=GROUP_COLUMN,
    temporal_architecture='bert',
    use_bigru=False,
    learning_rate=BASE_LEARNING_RATE,
    epochs=FINETUNING_EPOCHS,
    use_orientation_aux_head=USE_ORIENTATION_AUX_HEAD,
    use_phase_aux_head=USE_PHASE_AUX_HEAD,
    custom_aux_heads=NO_CUSTOM_AUX_HEADS,
    orientation_loss_weight=ORIENTATION_LOSS_WEIGHT,
    phase_loss_weight=PHASE_LOSS_WEIGHT,
    phase_label_provider=PHASE_LABEL_PROVIDER,
    invalid_candidate_policy=INVALID_CANDIDATE_POLICY,
    candidate_failure_log=str(results_dir / f'{timestamp}_candidate_failures.jsonl'),
    verbose=epoch_verbosity,
    random_state=random_state,
)
pipeline = Pipeline([('estimator', clf)])
''')

    params = source(14)
    boundary = params.index("# ============================================================\n# Fixed data/runtime context")
    params = params[:boundary] + r'''
# ============================================================
# Pretraining / SSL and stage-selection parameters
# All fields below can be edited or expanded into additional candidates.
# Stage selection is performed afresh inside each training fold.
# ============================================================

PRETRAINING_PARAM_SPACE = {
    'estimator__pretraining_mode': [PRETRAINING_MODE],
    'estimator__pretraining_epochs': [PRETRAINING_EPOCHS],
    'estimator__pretraining_batch_size': [64],
    'estimator__pretraining_learning_rate': [BASE_LEARNING_RATE],
    'estimator__pretraining_optimizer_name': [None],
    'estimator__pretraining_weight_decay': [0.0],
    'estimator__pretraining_momentum': [0.0],
    'estimator__pretraining_gradient_clip_norm': [None],
    'estimator__pretraining_gradient_clip_value': [None],
    'estimator__pretraining_loss_name': ['sparse_cce'],
    'estimator__pretraining_label_smoothing': [0.0],
    'estimator__pretraining_class_weight_mode': [None],
    'estimator__pretraining_validation_split': [0.15],
    'estimator__pretraining_validation_group_column': [GROUP_COLUMN],
    'estimator__validation_group_column': [GROUP_COLUMN],
    'estimator__pretraining_early_stopping_patience': [0 if FIRST_RUN else 3],
    'estimator__pretraining_use_lr_scheduler': [False],
    'estimator__pretraining_lr_scheduler_type': ['plateau'],
    'estimator__pretraining_lr_factor': [0.5],
    'estimator__pretraining_lr_patience': [3],
    'estimator__pretraining_min_lr': [1e-6],
    'estimator__pretraining_cosine_t_max': [100],
    'estimator__pretraining_warmup_epochs': [0],
    'estimator__pretraining_model_params': [None],
    'estimator__pretraining_augmentor': [None],
    'estimator__pretraining_shuffle': [True],
    'estimator__pretraining_verbose': [None],
    'estimator__finetuning_head_epochs': [0],
    'estimator__finetuning_head_learning_rate': [None],
    'estimator__finetuning_learning_rate': [None],
    # These masks define the SSL prediction task, separately from augmentation.
    'estimator__ssl_mask_ratio': [0.15],
    'estimator__ssl_span_length': [8],
    'estimator__ssl_mask_value': [0.0],
    'estimator__ssl_mask_columns': [None],
    'estimator__ssl_mask_modalities_together': [True],
    'estimator__ssl_reconstruction_loss': ['huber'],
    'estimator__ssl_huber_delta': [1.0],
    'estimator__ssl_decoder_units': [64],
    'estimator__ssl_decoder_dropout': [0.0],
    'estimator__ssl_loss_channel_weights': [None],
    'estimator__ssl_standardize_targets': [True],
    'estimator__ssl_invalid_values': [None],
    'estimator__ssl_validation_mask_seed': [1729],
    'estimator__ssl_resample_mask_each_epoch': [True],
    'estimator__ssl_cache_sequences': [16],
    'estimator__extractor__pretraining_source': PRETRAINING_SOURCE_CHOICES,
    'estimator__extractor__pretraining_label_column': [PRETRAINING_LABEL_COLUMN],
    'estimator__extractor__pretraining_include_classes': [PRETRAINING_INCLUDE_CLASSES],
    'estimator__extractor__pretraining_exclude_classes': [PRETRAINING_EXCLUDE_CLASSES],
    'estimator__extractor__pretraining_filters': [PRETRAINING_FILTERS],
    'estimator__extractor__pretraining_exclude_filters': [PRETRAINING_EXCLUDE_FILTERS],
    'estimator__extractor__pretraining_min_length': [PRETRAINING_MIN_LENGTH],
    'estimator__extractor__pretraining_max_length': [PRETRAINING_MAX_LENGTH],
    'estimator__extractor__pretraining_sample_fraction': [PRETRAINING_SAMPLE_FRACTION],
    'estimator__extractor__pretraining_max_sequences': [PRETRAINING_MAX_SEQUENCES],
    'estimator__extractor__pretraining_max_sequences_per_class': [PRETRAINING_MAX_SEQUENCES_PER_CLASS],
    'estimator__extractor__pretraining_balance_classes': [PRETRAINING_BALANCE_CLASSES],
    'estimator__extractor__pretraining_random_state': [random_state],
    'estimator__extractor__finetuning_source': [FINETUNING_SOURCE],
    'estimator__extractor__finetuning_label_column': [TARGET_COL],
    'estimator__extractor__finetuning_include_classes': [FINETUNING_INCLUDE_CLASSES],
    'estimator__extractor__finetuning_exclude_classes': [FINETUNING_EXCLUDE_CLASSES],
    'estimator__extractor__finetuning_filters': [FINETUNING_FILTERS],
    'estimator__extractor__finetuning_exclude_filters': [FINETUNING_EXCLUDE_FILTERS],
    'estimator__extractor__stage_target_column': [STAGE_TARGET_COLUMN],
    'estimator__extractor__stage_target_values': [STAGE_TARGET_VALUES],
}
GRID_PARAM_SPACE.update(PRETRAINING_PARAM_SPACE)
for _key, _choices in PRETRAINING_PARAM_SPACE.items():
    EVOLUTIONARY_PARAM_SPACE[_key] = ECat(_choices)

# Explicit coverage: every individual estimator/extractor/augmentor parameter.
_pipeline_params = pipeline.get_params(deep=True)
_leaf_params = {
    key: value for key, value in _pipeline_params.items()
    if key.startswith('estimator__')
    and key not in {'estimator__extractor', 'estimator__augmentor'}
}
for _key, _value in _leaf_params.items():
    GRID_PARAM_SPACE.setdefault(_key, [_value])
    EVOLUTIONARY_PARAM_SPACE.setdefault(_key, ECat([_value]))

# Preserve the V1 full spaces separately so the first run cannot erase them.
FULL_GRID_PARAM_SPACE = dict(GRID_PARAM_SPACE)
FULL_EVOLUTIONARY_PARAM_SPACE = dict(EVOLUTIONARY_PARAM_SPACE)

# All regularisation and augmentation controls are explicit in the pilot.
FIRST_RUN_FIXED_PARAMS = {
    'estimator__temporal_architecture': 'bert',
    'estimator__use_bigru': False,
    'estimator__augmentor__prob': 0.0,
    'estimator__augmentor__per_aug_prob': 0.0,
    'estimator__augmentor__augmentations': None,
    'estimator__augmentor__sensor_drop_prob': 0.0,
    'estimator__augmentor__noise_std': 0.0,
    'estimator__augmentor__jitter_sigma': 0.0,
    'estimator__augmentor__scaling_sigma': 0.0,
    'estimator__augmentor__channel_drop_prob': 0.0,
    'estimator__augmentor__time_shift_frac': 0.0,
    'estimator__augmentor__temporal_num_masks': 0,
    'estimator__augmentor__temporal_mask_frac': 0.0,
    'estimator__augmentor__warp_sigma': 0.0,
    'estimator__augmentor__warp_num_knots': 0,
    'estimator__augmentor__crop_frac_range': (1.0, 1.0),
    'estimator__pretraining_augmentor': None,
    'estimator__pretraining_model_params': None,
    'estimator__dropout': 0.0,
    'estimator__conv_dropout': 0.0,
    'estimator__static_dropout': 0.0,
    'estimator__gru_dropout': 0.0,
    'estimator__gru_recurrent_dropout': 0.0,
    'estimator__cnn_dropout': 0.0,
    'estimator__branch_dropout': 0.0,
    'estimator__fusion_dropout': 0.0,
    'estimator__bert_dropout': 0.0,
    'estimator__bert_attention_dropout': 0.0,
    'estimator__ssl_decoder_dropout': 0.0,
    'estimator__weight_decay': 0.0,
    'estimator__pretraining_weight_decay': 0.0,
    'estimator__label_smoothing': 0.0,
    'estimator__pretraining_label_smoothing': 0.0,
    'estimator__use_batchnorm': False,
    'estimator__gradient_clip_norm': None,
    'estimator__gradient_clip_value': None,
    'estimator__pretraining_gradient_clip_norm': None,
    'estimator__pretraining_gradient_clip_value': None,
    'estimator__early_stopping_patience': 0,
    'estimator__pretraining_early_stopping_patience': 0,
    'estimator__use_lr_scheduler': False,
    'estimator__pretraining_use_lr_scheduler': False,
    'estimator__warmup_epochs': 0,
    'estimator__pretraining_warmup_epochs': 0,
    'estimator__use_orientation_aux_head': False,
    'estimator__use_phase_aux_head': False,
    'estimator__custom_aux_heads': [],
    'estimator__loss_name': 'sparse_cce',
    'estimator__pretraining_loss_name': 'sparse_cce',
    'estimator__bert_num_layers': 2,
    'estimator__bert_num_heads': 4,
    'estimator__bert_ff_dim': 128,
    'estimator__bert_pooling': 'attention',
    'estimator__bert_use_cls_token': False,
    'estimator__bert_positional_encoding': 'learned',
}
# The nonzero learning rate and stage lengths are deliberately fixed for the pilot.
FIRST_RUN_FIXED_PARAMS.update({
    'estimator__learning_rate': BASE_LEARNING_RATE,
    'estimator__pretraining_learning_rate': BASE_LEARNING_RATE,
    'estimator__finetuning_learning_rate': BASE_LEARNING_RATE,
    'estimator__finetuning_head_learning_rate': BASE_LEARNING_RATE,
    'estimator__epochs': FINETUNING_EPOCHS,
    'estimator__pretraining_epochs': PRETRAINING_EPOCHS,
    'estimator__pretraining_mode': PRETRAINING_MODE,
})
# The same defaults used by the instantiated extractor, including selection/filter controls.
FIRST_RUN_EXTRACTOR_PARAMS = {
    'estimator__extractor__' + key: value
    for key, value in extractor.get_params(deep=False).items()
}

if FIRST_RUN:
    # Freeze every other dimension at the configured estimator defaults.
    # This includes the extractor's default feature spaces, not BEST_PARAMS features.
    GRID_PARAM_SPACE = {key: [value] for key, value in _leaf_params.items()}
    for _key, _value in FIRST_RUN_EXTRACTOR_PARAMS.items():
        GRID_PARAM_SPACE[_key] = [_value]
    for _key, _value in FIRST_RUN_FIXED_PARAMS.items():
        GRID_PARAM_SPACE[_key] = [_value]
    # Five valid BERT widths and four source conditions, including no pretraining.
    # Bayesian n_iter=5 evaluates five proposals from these available combinations.
    GRID_PARAM_SPACE['estimator__bert_d_model'] = [32, 64, 96, 128, 160]
    GRID_PARAM_SPACE['estimator__extractor__pretraining_source'] = PRETRAINING_SOURCE_CHOICES
    EVOLUTIONARY_PARAM_SPACE = {key: ECat(choices) for key, choices in GRID_PARAM_SPACE.items()}
    BEST_PARAMS = {key: values[0] for key, values in GRID_PARAM_SPACE.items()}
else:
    # The supplied V1 best configuration can also be used with either stage objective.
    BEST_PARAMS.update({key: choices[0] for key, choices in PRETRAINING_PARAM_SPACE.items()})

# ============================================================
# Build and validate
# ============================================================
if SKOPT_AVAILABLE:
    BAYESIAN_PARAM_SPACE = multibranch_keras.prepare_multibranch_bayesian_space(GRID_PARAM_SPACE)
else:
    BAYESIAN_PARAM_SPACE = GRID_PARAM_SPACE
EVOLUTIONARY_SEARCH_SPACE = multibranch_keras.prepare_multibranch_evolutionary_space(EVOLUTIONARY_PARAM_SPACE)

for space_name, candidate_space in (
    ('grid', GRID_PARAM_SPACE), ('bayesian', BAYESIAN_PARAM_SPACE),
    ('evolutionary', EVOLUTIONARY_SEARCH_SPACE),
):
    unknown_params = set(candidate_space) - set(_pipeline_params)
    missing_params = set(_leaf_params) - set(candidate_space)
    if unknown_params or missing_params:
        raise ValueError(f'{space_name}: unknown={sorted(unknown_params)}, missing={sorted(missing_params)}')

if search_mode == 'bayes':
    param_space = BAYESIAN_PARAM_SPACE
elif search_mode == 'evo':
    param_space = EVOLUTIONARY_SEARCH_SPACE
else:
    param_space = GRID_PARAM_SPACE
print(f'Validated {len(GRID_PARAM_SPACE)} Grid/Bayesian and {len(EVOLUTIONARY_PARAM_SPACE)} Evolutionary parameters.')
print('Model:', MODEL_VARIANT, 'pretraining:', PRETRAINING_MODE, 'first CPU pilot:', FIRST_RUN)
print('Epochs:', PRETRAINING_EPOCHS, 'pretraining +', FINETUNING_EPOCHS, 'fine-tuning; LR:', BASE_LEARNING_RATE)
'''
    replace(14, params)
    # Pretraining belongs in the saved training record alongside fine-tuning.
    replace(20, source(20).replace(
        "history = best_estimator.get_history_dict()   # returns self.history_",
        "history = best_estimator.get_history_dict()   # fine-tuning history\n"
        "    pretraining_history = getattr(best_estimator, 'pretraining_history_', {})\n"
        "    with open(save_dir / 'pretraining_history.json', 'w') as stream:\n"
        "        json.dump(pretraining_history, stream, indent=2, default=str)\n"
        "    print('Pretraining history:', pretraining_history)"
    ))
    replace(22, '''if experiment_succeeded:
    best_model = search.best_estimator_
    best_estimator = best_model.named_steps['estimator']
    # The held-out pool contains both domains; evaluate only the configured final task.
    X_test_final = best_estimator.extractor_.select_finetuning_sequences(X_test, y_test)
    y_test_final = y_test.loc[X_test_final.index].copy()
    y_pred = best_model.predict(X_test_final)
    expected_holdout_ids = pd.Index(X_test_final['sequence_id'].unique())
    if y_pred.empty or not y_pred.index.is_unique or not y_pred.index.isin(expected_holdout_ids).all():
        raise ValueError('Holdout predictions must contain distinct IDs from the final-task holdout domain.')
    # Prediction extraction applies the same problematic-sequence eligibility filter.
    excluded_holdout_ids = expected_holdout_ids.difference(y_pred.index).tolist()
    y_test_final = y_test_final[y_test_final['sequence_id'].isin(y_pred.index)].copy()
    with open(results_dir / f'{timestamp}_holdout_exclusions.json', 'w') as stream:
        json.dump({'reason': 'excluded by fitted extractor during prediction',
                   'sequence_ids': excluded_holdout_ids,
                   'requested_sequences': int(len(expected_holdout_ids)),
                   'evaluated_sequences': int(len(y_pred)),
                   'evaluation_scope': 'configured fine-tuning domain accepted by fitted extractor'},
                  stream, indent=2)
    print('Holdout sequences excluded by extractor:', len(excluded_holdout_ids))
    eval_results = evaluate_holdout(y_test_final, y_pred, target_col=TARGET_COL, verbose=True)
    print('Holdout competition score / macro F1:', eval_results['competition_score'])
''')
    replace(24, source(24).replace(
        "'target': TARGET_COL,",
        "'target': TARGET_COL,\n        'model_variant': MODEL_VARIANT,\n        'pretraining_mode': PRETRAINING_MODE,"
    ))

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            text = "".join(cell["source"])
            tree = ast.parse(text)
            if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) for node in ast.walk(tree)):
                raise ValueError(f"Notebook cell {index} defines a function")
            cell["execution_count"] = None
            cell["outputs"] = []
        cell["metadata"].pop("execution", None)
        cell["metadata"].pop("papermill", None)
    destination = ROOT / "archive/notebooks/multibranch_pretraining_v1.ipynb"
    destination.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
