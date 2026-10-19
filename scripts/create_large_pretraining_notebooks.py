"""Build reproducible large model-1/model-2 notebooks from the V1 layout.

Reference results are parsed as JSON data. Notebook cells contain configuration
and calls only; extraction, training and balanced Bayesian search stay in src/.
"""
import argparse
import ast
import json
from pathlib import Path
from pprint import pformat
import re

if __package__:
    from .create_pretraining_notebook import configure_validation
    from .pretraining_search_configuration import configure_search_parameters
else:
    from create_pretraining_notebook import configure_validation
    from pretraining_search_configuration import configure_search_parameters


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'kaggle' / 'pretraining-large-reference.json'


def replace_assignment(source, name, value):
    source, count = re.subn(r'^' + re.escape(name) + r'\s*=.*$',
                            name + ' = ' + repr(value), source, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f'Expected one assignment for {name}, found {count}.')
    return source


def generate(model, reference):
    notebook = json.loads((ROOT / 'archive/notebooks/multibranch_pretraining_v1.ipynb').read_text(encoding='utf-8'))

    def source(index):
        return ''.join(notebook['cells'][index]['source'])

    def set_source(index, value):
        notebook['cells'][index]['source'] = value.splitlines(keepends=True)

    name = f'multibranch_pretraining_model{model}_large'
    set_source(0, f'''# Large multibranch pretraining: model {model}

37 Bayesian candidates on CPU, using the supplied 0.811823992 holdout result
as a feature/capacity reference. Model 1 uses supervised gesture pretraining;
model 2 uses masked-signal self-supervision. Both fine-tune target gestures.

The four pretraining sources share the same architecture ranges and receive
9 or 10 evaluations each. Their first visits are interleaved. Each source uses
three initial proposals, followed by Bayesian adaptation. Completed candidates
are saved to partial_cv.csv before the next trial.

The reference was the original branch-attention model. Here, BERT uses the
active branch_cnn_* controls: 128 filters, four layers and kernel seven for
every temporal branch. This approximates its large acc/rotation/ToF branches;
the thermal branch is larger and the old branch_dense_units does not apply.
STFT/CWT use static_branch_units=100. BERT widths 128/192/256, depths 2/3,
heads 4/8, feed-forward widths 256/512 and dense heads 128/256 are searched.

The reference sensor features and quality filter are restored. Fine-tuning is
capped at 150 epochs with patience 20; pretraining at 30 with patience 8.
AdamW starts at 1e-4, with the reference cosine schedule and batch normalization.
Augmentation, dropout, weight decay and label smoothing remain zero.

Outer holdout and CV split wearers 70/30, and internal validation also separates
wearers. The old holdout score is a reference, not a directly comparable result.
''')
    config = source(3)
    for key, value in {
        'experiment_name': name, 'n_iter': 37, 'train_size': 0.7,
        'MODEL_VARIANT': f'model_{model}', 'FIRST_RUN': False,
        'PRETRAINING_EPOCHS': 30, 'FINETUNING_EPOCHS': 150,
    }.items():
        config = replace_assignment(config, key, value)
    config += '\n\nLARGE_RUN = True\nCOVER_PRETRAINING_SOURCES = True\n'
    config += 'BAYES_INITIAL_POINTS_PER_SOURCE = 3\n'
    config += 'REFERENCE_HOLDOUT_SCORE = ' + repr(reference['holdout_score']) + '\n'
    config += 'REFERENCE_BEST_PARAMS = ' + pformat(reference['best_params'], sort_dicts=True) + '\n'
    set_source(3, config)
    params = source(14)
    marker = '# ============================================================\n# Build and validate'
    if params.count(marker) != 1:
        raise ValueError('Missing parameter-space build boundary.')
    large_block = '''# Large-run overrides use the supplied best result as data, not instructions.
# Retain every leaf parameter explicitly; vary only the dimensions listed below.
GRID_PARAM_SPACE = {key: [value] for key, value in _leaf_params.items()}
LARGE_FIXED_PARAMS = {key: value for key, value in REFERENCE_BEST_PARAMS.items()
                      if key in _pipeline_params}
LARGE_UNSUPPORTED_REFERENCE_PARAMS = sorted(set(REFERENCE_BEST_PARAMS) - set(_pipeline_params))
LARGE_FIXED_PARAMS.update({key: values[0] for key, values in PRETRAINING_PARAM_SPACE.items()})
LARGE_FIXED_PARAMS.update({
    'estimator__temporal_architecture': 'bert',
    'estimator__use_bigru': False,
    'estimator__branch_encoder_type': 'conv1d',
    'estimator__branch_cnn_filters': 128,
    'estimator__branch_cnn_depth': 4,
    'estimator__branch_cnn_kernel_size': 7,
    'estimator__static_branch_units': 100,
    'estimator__fusion_type': 'concat',
    'estimator__fusion_units': 256,
    'estimator__bert_pooling': 'attention',
    'estimator__bert_positional_encoding': 'learned',
    'estimator__bert_use_cls_token': False,
    'estimator__activation': 'relu',
    'estimator__bert_activation': 'gelu',
    'estimator__batch_size': 128,
    'estimator__epochs': FINETUNING_EPOCHS,
    'estimator__early_stopping_patience': 20,
    'estimator__optimizer_name': 'adamw',
    'estimator__learning_rate': BASE_LEARNING_RATE,
    'estimator__loss_name': 'focal',
    'estimator__class_weight_mode': 'balanced',
    'estimator__validation_split': 0.15,
    'estimator__validation_group_column': GROUP_COLUMN,
    'estimator__use_batchnorm': True,
    'estimator__use_lr_scheduler': True,
    'estimator__lr_scheduler_type': 'cosine',
    'estimator__cosine_t_max': 50,
    'estimator__min_lr': 1e-6,
    'estimator__pretraining_mode': PRETRAINING_MODE,
    'estimator__pretraining_epochs': PRETRAINING_EPOCHS,
    'estimator__pretraining_batch_size': 128,
    'estimator__pretraining_learning_rate': BASE_LEARNING_RATE,
    'estimator__pretraining_optimizer_name': 'adamw',
    'estimator__pretraining_class_weight_mode': 'balanced',
    'estimator__pretraining_loss_name': 'sparse_cce',
    'estimator__pretraining_validation_split': 0.15,
    'estimator__pretraining_validation_group_column': GROUP_COLUMN,
    'estimator__pretraining_early_stopping_patience': 8,
    'estimator__pretraining_use_lr_scheduler': True,
    'estimator__pretraining_lr_scheduler_type': 'cosine',
    'estimator__pretraining_cosine_t_max': 30,
    'estimator__pretraining_min_lr': 1e-6,
    'estimator__pretraining_augmentor': None,
    'estimator__pretraining_model_params': None,
    'estimator__finetuning_learning_rate': BASE_LEARNING_RATE,
    'estimator__finetuning_head_epochs': 0,
    'estimator__use_orientation_aux_head': False,
    'estimator__use_phase_aux_head': False,
    'estimator__custom_aux_heads': [],
    'estimator__augmentor__augmentations': None,
    'estimator__augmentor__prob': 0.0,
    'estimator__augmentor__per_aug_prob': 0.0,
    'estimator__ssl_decoder_units': 128,
    'estimator__ssl_cache_sequences': 16,
    'estimator__ssl_mask_ratio': 0.15,
    'estimator__ssl_span_length': 8,
})
# Preserve the requested zero regularisation settings across both stages.
for _key in _leaf_params:
    if ('dropout' in _key or 'weight_decay' in _key or 'label_smoothing' in _key):
        LARGE_FIXED_PARAMS[_key] = 0.0
    if 'gradient_clip' in _key:
        LARGE_FIXED_PARAMS[_key] = None
    if '__augmentor__' in _key and _key.endswith(('sigma', 'prob', 'noise_std',
                                                'time_shift_frac', 'temporal_mask_frac',
                                                'temporal_num_masks', 'warp_num_knots')):
        LARGE_FIXED_PARAMS[_key] = 0.0
for _key, _value in LARGE_FIXED_PARAMS.items():
    GRID_PARAM_SPACE[_key] = [_value]
LARGE_PARAM_SPACE = {
    'estimator__extractor__pretraining_source': PRETRAINING_SOURCE_CHOICES,
    'estimator__bert_d_model': [128, 192, 256],
    'estimator__bert_num_layers': [2, 3],
    'estimator__bert_num_heads': [4, 8],
    'estimator__bert_ff_dim': [256, 512],
    'estimator__dense_units': [128, 256],
}
GRID_PARAM_SPACE.update(LARGE_PARAM_SPACE)
EVOLUTIONARY_PARAM_SPACE = {key: ECat(values) for key, values in GRID_PARAM_SPACE.items()}
BEST_PARAMS = {key: values[0] for key, values in GRID_PARAM_SPACE.items()}
pipeline.set_params(**BEST_PARAMS)
with open(results_dir / f'{experiment_name}_configuration.json', 'w', encoding='utf-8') as stream:
    json.dump({'model_variant': MODEL_VARIANT, 'pretraining_mode': PRETRAINING_MODE,
               'iterations': n_iter, 'train_size': train_size, 'group_column': GROUP_COLUMN,
               'reference_holdout_score': REFERENCE_HOLDOUT_SCORE,
               'fixed_params': LARGE_FIXED_PARAMS, 'varying_params': LARGE_PARAM_SPACE},
              stream, indent=2, default=str)
print('Reference-only unsupported parameters:', LARGE_UNSUPPORTED_REFERENCE_PARAMS)
print('Balanced source budget: 9 or 10 candidates each, interleaved; total:', n_iter)

'''
    params = params.replace(marker, large_block + marker)
    set_source(14, params)
    training = source(17).replace(
        '        search_spaces=param_space,\n',
        "        search_spaces=param_space,\n"
        "        optimizer_kwargs={'n_initial_points': BAYES_INITIAL_POINTS_PER_SOURCE,\n"
        "                          'model_queue_size': 1, 'acq_optimizer': 'sampling',\n"
        "                          'acq_optimizer_kwargs': {'n_points': 256}},\n")
    training = training.replace(
        'search.fit(X_train, y_train, groups=groups)',
        "search.candidate_checkpoint_path = results_dir / 'partial_cv.csv'\n"
        'search.fit(X_train, y_train, groups=groups)')
    set_source(17, training)
    configure_validation(notebook)
    configure_search_parameters(notebook)
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            tree = ast.parse(''.join(cell['source']))
            if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) for node in ast.walk(tree)):
                raise ValueError('Generated notebooks must not define functions.')
            cell['outputs'] = []
            cell['execution_count'] = None
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
    parser.add_argument('--reference', type=Path, help='Import the supplied JSON result as immutable reference data.')
    args = parser.parse_args()
    if args.reference:
        reference = json.loads(args.reference.read_text(encoding='utf-8-sig'))
        if not isinstance(reference.get('best_params'), dict) or not isinstance(reference.get('holdout_score'), (int, float)):
            raise ValueError('Reference must contain best_params and holdout_score.')
        REFERENCE.write_text(json.dumps(reference, indent=2) + '\n', encoding='utf-8')
    else:
        reference = json.loads(REFERENCE.read_text(encoding='utf-8'))
    for model in (1, 2):
        generate(model, reference)


if __name__ == '__main__':
    main()
