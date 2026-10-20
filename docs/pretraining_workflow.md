# Multibranch pretraining experiments

The active notebooks are `notebooks/multibranch_pretraining.ipynb` (supervised
pretraining) and `notebooks/multibranch_ssl.ipynb` (masked SSL). Both follow the
multibranch v1 workflow. The original five-candidate pilot is retained at
`archive/notebooks/multibranch_pretraining_v1.ipynb` for reproducibility; the
legacy `pilot` workflow profile resolves that archived notebook automatically.
It calls the sequence extractor in `src/base_utils.py` and the staged models in
`src/pretraining_utils.py`. Set `MODEL_VARIANT` to `model_1` for supervised
gesture pretraining or `model_2` for masked-signal self-supervision. The branches
and fine-tuning model remain those of `src/multibranch_utils.py`.

Edit `GRID_PARAM_SPACE` for grid/Bayesian candidates and
`EVOLUTIONARY_PARAM_SPACE` for evolutionary candidates. These declared ranges
control the search directly. `PRETRAINING_PARAM_SPACE` adds shared stage settings
such as pretraining epochs, learning rate, source domains, labels, class filters,
sampling limits and target-domain definitions. Existing entries in either main
space take precedence. Fine-tuning epochs and learning rate live in those main
spaces; `finetuning_learning_rate=None` follows the selected `learning_rate`.

Both objectives retain the four source candidates: `non_target`, `all`, `target`
and `baseline`. The baseline skips pretraining and trains the downstream model
from scratch. Class lists and metadata filters apply inside each training fold.
For evolutionary integer or continuous ranges, put `EInt` or `EFloat` entries
directly in `EVOLUTIONARY_PARAM_SPACE`.

`VALIDATION_PARAM_SPACE` defaults to `validation_mode: ['sequence']` and
`group_column: ['subject']`. Sequence mode restores the original balanced outer
sequence split, uses complete-sequence CV groups, and reserves sequences for
internal validation. Subjects may appear on both sides. Subject separation runs
only if you explicitly add `'subject'` to the mode choices. The group-column
choice applies only to subject mode.

When both modes are listed, each gets a separate full search with matching outer
holdout, CV and internal validation. Each search receives the configured candidate
budget. Their CV, best and holdout CSVs and histories are saved in separate
`validation_protocols` folders; `validation_summary.csv` identifies each result.
The scores are not ranked together. Held-out sequences are excluded from
preprocessing and both training stages in either mode.

## First CPU run

The archived pilot's `FIRST_RUN = True` fixes default feature settings, disables
augmentations, regularisation and learning-rate scheduling, and keeps the base
learning rate positive. It uses five pretraining epochs and five fine-tuning
epochs per fit for the initial CPU search. Set `FIRST_RUN = False` to use the
full configurable search spaces and training budgets. The active notebooks
use their declared epoch candidates directly.

The first five Bayesian candidates search across the four source choices and
five BERT widths. Five sampled candidates are not an exhaustive comparison of
all source/width combinations.

Run these commands from the repository on `ssl/v0/develop`, using the existing
Kaggle environment:

```powershell
.venv-kaggle\Scripts\python.exe scripts/kaggle_pretraining_experiment.py prepare --first-run --iterations 5
.venv-kaggle\Scripts\python.exe scripts/kaggle_pretraining_experiment.py push --first-run --iterations 5
```

`prepare` writes a build only. `push` prepares and submits the private CPU
notebook `keithmarange/multibranch-pretraining-v1-cpu`. Its metadata is in
`kaggle/pretraining-kernel-metadata.json`, independent of the existing
multibranch notebooks. The submission attaches the same competition and source
dataset as the original workflow.

Every submission embeds the exact current `src/*.py` bytes in a bootstrap cell.
The attached dataset supplies the original data; stale utility copies in that
dataset cannot replace the embedded source. The local notebook is not rewritten.
Builds and source hashes are recorded in `kaggle_build/multibranch_pretraining/`.

## Monitor and collect

```powershell
.venv-kaggle\Scripts\python.exe scripts/kaggle_pretraining_experiment.py status
.venv-kaggle\Scripts\python.exe scripts/kaggle_pretraining_experiment.py logs
.venv-kaggle\Scripts\python.exe scripts/kaggle_pretraining_experiment.py logs --follow
.venv-kaggle\Scripts\python.exe scripts/kaggle_pretraining_experiment.py download
```

`download` requires a completed run and an acknowledged local submission. It
checks that the remote notebook source matches that submission before collecting
the configured result CSVs under `results/kaggle_pretraining/`. `pull` saves a
remote copy inside the build directory without replacing the local notebook.
`wait` polls status at intervals of 15–60 seconds and collects completed results.

To run another model or budget, change notebook configuration and omit
`--first-run`. `--iterations` changes only the packaged candidate count; it never
edits the local notebook.

## Large model searches

The large profiles have independent private CPU kernels, notebook files, build
directories and acknowledged submission manifests. Submitting model 2 cannot
replace the model-1 run or its local provenance.

```powershell
.venv-kaggle\Scripts\python.exe scripts/kaggle_pretraining_experiment.py push --profile model1-large --iterations 37
.venv-kaggle\Scripts\python.exe scripts/kaggle_pretraining_experiment.py push --profile model2-large --iterations 37
.venv-kaggle\Scripts\python.exe scripts/kaggle_pretraining_experiment.py status --profile model1-large
.venv-kaggle\Scripts\python.exe scripts/kaggle_pretraining_experiment.py download --profile model2-large
```

`scripts/create_large_pretraining_notebooks.py` recreates both notebooks from the
V1 pretraining layout. The active notebooks use their declared parameter ranges;
reference results and large-run presets do not overwrite those choices.
Regenerating notebooks replaces local notebook edits, so edit the active files
directly when configuring an experiment.

Bayesian compilation splits the four pretraining sources into fixed-source
subspaces under one shared budget and visits them in round-robin order. A
37-candidate run allocates 9/9/9/10 evaluations, with all four sources visited in
the first four valid candidates. Each optimizer uses three initial points, then
adapts to its observations. The search writes an atomic `partial_cv.csv` after
each completed evaluation; final CSV and holdout evaluation retain the V1 flow.
