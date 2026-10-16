# Multibranch and random forest code review

The active notebooks now share two utility modules:

- `src/base_utils.py`: data discovery, balanced sequence splitting, signal cleaning,
  feature extraction, augmentation, random forest wrapper, scoring, and generic search utilities.
- `src/multibranch_utils.py`: neural architectures, multibranch estimator,
  candidate validation, hierarchical search, and safe refitting.

`sequence_explorer_utils.py` remains separate because it implements the interactive
explorer. The duplicate qwen/sol implementations and separate data/candidate/search
helper modules were removed. The sol implementations supplied the canonical behavior:
they contained every earlier API plus the numerical fixes and temporal/auxiliary heads.
Active notebooks, tests, scripts, and Kaggle source packaging use the new modules.
Archived notebooks under `legacy/` and `legacy multi modal/` were left as historical files.
Old serialized models that record the removed module paths need retraining or an
explicit migration before loading.

## Findings and implemented changes

| Cost or defect | Change |
| --- | --- |
| Notebooks retained raw, filtered, split, and copied input frames simultaneously. V1 also copied every sensor into `y`. | Load into one training frame, combine filters, use the split frames directly as `X`, retain only row-aligned metadata in `y`, and release the unsplit input. |
| CSV inference stored sensor readings as float64, including 320 ToF channels. | Read sensor columns directly as float32. Counters and metadata keep their existing dtypes; numerical signal calculations still explicitly use float64 where needed. |
| Gesture parsing repeatedly split identical strings at every timestep. | Parse unique gesture strings once and map the results to rows. |
| Balanced splitting grouped the full wide sensor table. | Group compact sequence/subject/gesture metadata, then select whole sequences once. |
| Read-only IMU, rotation, STFT and CWT transforms copied the full input, including unrelated sensors. | Read only the required columns and construct separate outputs. STFT/CWT summaries use positional broadcasting rather than repeated DataFrame column insertion. |
| Disabled motion filtering and augmentation still copied the full input. | Reuse the input when no work is enabled. Active transforms continue to protect caller data. Augmentation now selects sensor columns so numeric targets and metadata are preserved. |
| Signal cleaning ran Python interpolation callbacks even when no values were missing. | Skip imputation on complete data and use native grouped forward/backward fill. Backward fill remains inside each sequence. |
| Global context performed two groupings and two column insertions per feature. Fit materialized context merely to obtain column names. | Aggregate context once per sequence, broadcast one block, and derive its schema without allocating context values during fit. |
| Model fitting extracted the same training features once in fit and again in transform. | A transient `fit_transform` path extracts once. It stores no training-frame cache, and custom extractor overrides keep their existing behavior. |
| Chunking retained a list of every grouped DataFrame. | Iterate group positions and retain float32 windows, avoiding the collection of wide grouped frames. |
| Multibranch prediction retained the combined tensor after allocating modality arrays. Static pooling created another full-size multiplication temporary. | Release combined features before runtime prediction; mask the private static branch buffer in place, preserving float32 reduction results. Training partitions are released after phase providers finish reading them. |
| Label alignment/scoring copied all of `y`. | Select the required target columns before deduplicating or aligning. |
| Random forest parallelized both candidate evaluation and tree fitting. Queued fits could duplicate the raw dataset. | Default to one candidate at a time, parallelize the reduced tree-fitting workload, and set search `pre_dispatch='n_jobs'`. |
| RF target-only filtering used an integer Series as a DataFrame column selector. | Select rows using an explicit boolean target mask. |
| Kaggle's local-source mode did not consistently embed the actual source used locally. | Package the canonical modules for every local-snapshot run, including alpha and the GPU diagnostic. |

## Measurements

A deterministic synthetic input contains 48 sequences of 96 rows, with 320 ToF
channels, accelerometer, quaternion and temperature readings: 4,608 × 334 columns,
11.74 MiB resident input. The configuration enables raw ToF and global context,
disables STFT/CWT, and produces 192 × 48 × 1,956 float32 chunk features.

| Preprocessing | Previous sol path | Consolidated path |
| --- | ---: | ---: |
| Peak traced allocations, excluding resident input | 302.03 MiB | 193.41 MiB |
| Elapsed time with allocation tracing enabled | 37.22 s | 2.95 s |

That is approximately **36% lower peak traced allocation** in this case. Tensor,
mask, sequence ID and feature-name bytes have the same SHA-256 digest:
`6074d9106627cbc1dad1738fc63f6bc98ec66739c36d6471906c4446b3ea874f`.
This comparison uses identical float64 input for both versions; the separate
notebook float32 loading change is not included in the result. Traced allocations
are not process RSS or GPU memory, and these timings should not be extrapolated
to a full training run.

Run the current benchmark with:

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_preprocessing.py
```

The script also accepts `--source PATH --separate-fit-transform` to profile an
older standalone base utility file using its original fit-then-transform path.

## Validation and remaining limits

The regression suite covers numerical signal behavior, sequence-safe validation,
search/candidate contracts, auxiliary heads, data packaging, transient feature
reuse, input immutability and tensor release. Loading/splitting cells from all six
training notebooks were also executed against sample data: float32 sensor input,
disjoint sequence sets and row-aligned metadata labels were checked.

The local TensorFlow installation fails during import under Python 3.9.1
(`TypeError: unhashable type: 'list'` in `typing.py`), so Keras execution tests
are skipped. No full GPU search or complete-dataset retraining was performed.
Float32 CSV ingress rounds sensor values earlier than before; expect small
numerical or model-score changes from that deliberate memory tradeoff.

Two structural costs remain:

- STFT/CWT sequence summaries are still broadcast across source timesteps and
  chunks before being pooled back into static branch vectors. One million rows
  × 100 float64 summary features alone requires about 763 MiB. Eliminating this
  requires separate static and temporal extractor outputs.
- Raw ToF emits both imputed values and validity channels; global context then
  triples the feature count. Overlapping windows further increase tensor size.
  Existing feature choices, window caps, strides and experiment budgets remain
  configurable rather than being reduced during cleanup.

Some existing search choices are incompatible: for example, six attention heads
do not divide any current BERT width candidate, and `generic_aux_heads` references
the placeholder `your_numeric_column`. Candidate rules reject these before
training, but pruning such choices would reduce proposal overhead. The user's
configured search spaces were preserved.

The subsequent [conditional search review](deep_learning_efficiency_review.md)
documents the architecture-aware Bayesian spaces and evolutionary genes that
now prevent inactive controls from consuming useful search evaluations, together
with verified references to the supplied deep learning books.

RF and the older root gesture notebook retain an outer augmentation pipeline.
Its effective augmentation is disabled by the current settings. If enabled,
that pipeline also transforms evaluation inputs; move augmentation inside
training before using it for a fair augmented-model comparison.
