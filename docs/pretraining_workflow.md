# Multibranch pretraining experiments

`notebooks/multibranch_pretraining_v1.ipynb` follows the multibranch v1 workflow.
It calls the sequence extractor in `src/base_utils.py` and the staged models in
`src/pretraining_utils.py`. Set `MODEL_VARIANT` to `model_1` for supervised
gesture pretraining or `model_2` for masked-signal self-supervision. The branches
and fine-tuning model remain those of `src/multibranch_utils.py`.

Pretraining sequence filters are part of the extractor configuration, including
target/non-target selection, gesture classes and the problematic-column filter.
Both objectives expose four source candidates: `non_target`, `all`, `target`
and `baseline`. The baseline skips pretraining and trains the same downstream
model from scratch. Class lists can further restrict the three pretraining
sources.
Pretraining data must stay inside the training wearers for each evaluation
split. Held-out wearer recordings are not unlabelled training data.

## First CPU run

The notebook's `FIRST_RUN = True` fixes default feature settings, disables
augmentations, regularisation and learning-rate scheduling, and keeps the base
learning rate positive. It uses five pretraining epochs and five fine-tuning
epochs per fit for the initial CPU search. Set `FIRST_RUN = False` to use the
full configurable search spaces and training budgets.

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
