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
V1 pretraining layout and `kaggle/pretraining-large-reference.json`. That reference
contains the supplied result (holdout 0.8118239920); its JSON values are treated
as configuration data. The original downloaded file is never edited.

Both profiles use the reference feature spaces, chunk length 200, 70/30 wearer
splits, 128-filter four-layer temporal CNN branches, static width 100, AdamW at
1e-4, cosine scheduling and batch normalization. Fine-tuning allows 150 epochs
with patience 20; pretraining allows 30 with patience 8. Augmentation, dropout,
weight decay and label smoothing are zero. BERT width, depth, heads, feed-forward
width and final dense width vary over explicitly listed large settings.

The reference predates the temporal BERT builder. Its per-branch CNN dictionaries
do not control BERT; the active `branch_cnn_*` parameters are set explicitly here.
Thermal processing is therefore larger than in the original, and its 225-unit
branch embeddings are not reproduced. The original score also used a different
partition, so it must not be treated as a directly comparable wearer holdout.

Bayesian compilation splits the four pretraining sources into fixed-source
subspaces under one shared budget and visits them in round-robin order. A
37-candidate run allocates 9/9/9/10 evaluations, with all four sources visited in
the first four valid candidates. Each optimizer uses three initial points, then
adapts to its observations. The search writes an atomic `partial_cv.csv` after
each completed evaluation; final CSV and holdout evaluation retain the V1 flow.
