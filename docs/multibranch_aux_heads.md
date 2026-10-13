# Optional auxiliary heads in multibranch_v1

Set `CUSTOM_AUX_HEADS` in the notebook's configuration cell. The built-in
orientation and phase switches already search `[False, True]` in
`GRID_PARAM_SPACE` (also used by Bayesian search) and `ECat([False, True])` in
`EVOLUTIONARY_PARAM_SPACE`. Each head can run independently or together.
`CUSTOM_AUX_HEAD_OPTIONS` searches every subset of the configured heads, so
short and long heads are independently optional: neither, short only, long only,
or both. N configured heads produce `2**N` choices; an empty list produces a
single disabled choice. To force a built-in head off
or on, replace its search choices with `[False]` or `[True]` (wrapped in `ECat`
for evolution). `USE_ORIENTATION_AUX_HEAD` and `USE_PHASE_AUX_HEAD` supply the
estimator's direct-fit defaults; the search choices override them.

```python
USE_ORIENTATION_AUX_HEAD = True
CUSTOM_AUX_HEADS = [
    {'name': 'short_window', 'target': TARGET_COL, 'window_size': 50,
     'hidden_units': [32], 'dropout': 0.1, 'loss_weight': 0.2},
    {'name': 'long_window', 'target': TARGET_COL, 'window_size': 150,
     'hidden_units': [64], 'loss_weight': 0.2},
    {'name': 'action_aux', 'target': 'gesture_action', 'loss_weight': 0.1},
]
```

Custom heads predict any available **sequence target column**: its value must be
present and constant within each sequence. Use `task='classification'` (default)
or `task='regression'` for a numeric target. Classification uses sparse categorical
cross entropy; regression uses MSE. Class counts are inferred using training
labels only. Names must be unique and cannot use the built-in output names.

Without `window_size`, a head reads the shared pooled representation. With a
window size, it pools the last N valid timesteps of the shared temporal
representation. Short and long heads have their own dense layers and contribute
their weighted losses to training. They share the branch encoders and temporal
backbone, so a GRU or transformer can carry information from outside the pooled
span. They do not create separate extractor branches or independently cropped
encoders. Sizes are extracted timesteps, must be positive, and cannot exceed the
current chunk length. Padding is excluded from pooling.

Orientation requires an `orientation` target column. Its class count is inferred
from training labels. `ORIENTATION_LOSS_WEIGHT` controls its training contribution.

Phase supervision needs real timestep labels. Set `PHASE_LABEL_PROVIDER` to a
callable `(partition_dataframe, extracted_chunks) -> labels`, where `labels` has
shape `(number_of_chunks, timesteps)` and integer IDs in `[0, n_phases)` at valid
timesteps. It runs separately for the training and internal validation partitions
of every CV fold, and again on refit. It must produce labels aligned with the
current extractor/chunk settings; these can vary across candidates. No phase
labels are guessed from orientation, gestures, or relative position. Without a
provider, a phase-head candidate is penalised. The estimator API also accepts
explicit `phase_labels` and `validation_phase_labels` for direct fits with fixed
extraction; the provider is the safer option for CV. A phase-attention backbone
can run without a supervised phase head.

With `INVALID_CANDIDATE_POLICY='penalize'`, failed fits or scoring receive `-1`
(valid F1 scores are nonnegative). Failure reasons and parameters are written to
`*_candidate_failures.jsonl`. Grid, Bayesian, and evolutionary searches continue,
including when every candidate fails. Only candidates with successful scores
in every fold are eligible for refit. If a full-data refit fails, the next eligible
candidate is tried and the reason is saved in `*_refit_failures.json`. If none
works, CV results are saved and plots/holdout/model export are skipped.

Use `INVALID_CANDIDATE_POLICY='raise'` to stop at a fit/scoring error while
debugging. Scores and holdout predictions always use the primary target; auxiliary
outputs influence training but do not replace the primary prediction.

Bayesian and evolutionary search use hierarchical candidate rules before
training. The selected temporal backbone controls which temporal settings are
active; branch CNN settings still apply with a BERT backbone. Auxiliary heads
are independent switches and can coexist with any supported backbone. Settings
for inactive backbones or disabled heads are reset to a canonical value so
irrelevant mutations do not produce duplicate experiments. Missing head labels,
invalid custom-head windows, BERT CLS mismatches, and spatial ToF without raw
pixels are rejected with a finite penalty before a model is built. Runtime
fit/scoring failures still use the existing logging and safe-refit path.

`*_candidate_rules.json` records the rule rejection count/reasons and the count
of cached candidate evaluations reused during search. The notebook retains
`flatten_1d` ToF by default; comments show how to enable spatial ToF choices and
their required raw input. Evolution searches all four temporal backbones. Its
current population 10 and generations 4 allow at most 50 population evaluations
before refit; cached repeats use no additional training.
