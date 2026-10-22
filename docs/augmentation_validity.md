# Augmentation validity and failure reporting

## Reproduction and historical limits

Before changing the implementation, the local functions were exercised with
finite acceleration `[0, 0, 9.81]` and quaternion `[1, 0, 0, 0]`.
The unchanged input passed `SignalCleaner(linear_acc_mode='baseline')`.
Forced rotation sensor dropout and channel dropout each created zero
quaternions on every sample. Temporal masking with fraction zero and one mask
still zeroed one sample. All three reproduced:

```
Acceleration and quaternion samples must be finite; quaternions nonzero
```

The actual classifier path reproduced the same failure through
`_augment_training_sequences` -> `_extract_training_chunks` ->
`SequenceExtractor.fit_transform` -> `SignalCleaner`, including the chained
`InvalidExtractorParams` and original `ValueError`. The first invalid samples
appeared in raw augmentation, before cleaning and before model construction.
Separately, zero-strength time shifting moved samples for fixed seed 0.

The supplied v8 results report 37 attempts, 19 valid and 18 failed, with all 18
failures occurring in enabled candidates with positive dropout probabilities.
That correlation does not identify which operation caused every historical
failure. No verified copy of the v8 executed dataset modules was found locally.
Cached embedded source archives identify an earlier version 2 and omit
`pretraining_search.py`; they cannot establish the identity of the v8 source.
The local reproduction establishes the current implementation's mechanisms.

## Geometry and augmentation policy

For the native `SequenceExtractor` in multibranch training, the order is:

1. Split by sequence (and the configured grouping), before augmentation.
2. Fit preprocessing on the existing training domain, with raw inputs intact.
3. Clean and interpolate, remove gravity, resample, calculate rotation-derived
   features and spectra, and assemble model features.
4. Apply the configured augmentation operations to training features, in their
   existing configured order, then create the existing chunks/model inputs.
5. Extract validation and holdout inputs without the temporary augmentation hook.

This includes the fine-tuning and separately configured source-training
augmentors. The hook is removed in `finally`, including when extraction fails.
Neither the reconstruction targets nor validation labels are augmented.
Masked SSL rejects enabled crop/resize and time-shift source augmentation that
would move input timesteps without moving reconstruction targets and masks.

Zero quaternion *features* now mean a masked model input. They never enter
gravity removal, quaternion normalization, rotation derivatives or interpolation.
Acceleration dropout also happens after gravity removal, avoiding artificial
derived acceleration from zeroing a raw acceleration sensor before subtracting
gravity. Noise, jitter, scaling, warping, cropping and shifting use the same
processed feature boundary. These operations are feature perturbations, not
physical changes of raw device orientation.

Standalone `SensorAugmentor.fit_transform` still accepts raw DataFrames (also
the fallback for custom extractors/augmentors). All raw `acc_*`, `rot_*`,
`rot6d_*` and `ang_vel_*` columns remain intact together. This conservative
fallback cannot drop raw IMU sensors. Eligible already processed `lin_acc_*`,
ToF and thermal columns can still change. Raw IMU augmentation requires the
native processed-feature integration, or an explicit call to
`transform_features` on already computed features. Numeric labels, sequence IDs,
counters, row order/index, shape and caller-owned input are preserved.

The cleaner's original quaternion validity guard is unchanged. Invalid original
orientations are not replaced with identity quaternions or removed.

Zero jitter/noise/scaling/warp strength, zero dropout probability, zero temporal
mask fraction/count, zero shift fraction and full-length crop `(1, 1)` are true
no-ops. Primitive no-ops do not consume RNG draws. Empty and single-sample
sequences are supported by shifting, cropping and warping.

## Optional diagnostics and scores

Enable `SensorAugmentor(diagnostics=True)` (or the existing nested sklearn
parameter `estimator__augmentor__diagnostics`) for per-operation checks.
Defaults retain at most five sequence reports and five representative locations
per validity category; configure `diagnostic_max_sequences` and
`diagnostic_max_locations` to change these bounds. Normal execution does not
scan validity after every augmentation operation.

Reports include the candidate identifier, partition, seed and RNG state,
sequence ID, row index/position and sample counter where available, selected
operations in execution order, operation parameters, the first offending
operation, pre-existing invalid-input status, non-finite counts and locations,
zero/near-zero raw quaternion norms, and the exception/traceback. Raw missing
values may be intentionally interpolated by the cleaner; input diagnostics
record them without replacing that policy. Near-zero nonzero quaternions are
reported without changing the original cleaner guard.

Candidate failure records distinguish augmentation, extraction, training and
scoring. Existing negative search penalties remain optimizer inputs, rather than
being presented as measured performance. The experiments CSV adds
`score_status`, `valid_score`, `failure_penalty` and `failed_fold_count`;
the summary separates valid and failed candidates and retains bounded worker
failure records with traceback/context. An individual failed CV fold makes the
candidate's `valid_score` absent even when its penalized mean is positive.

## Notebook reporting

The current runner returns top-level `experiments_path` and `summary_path`.
Both references in the current notebook already matched that real schema.
No alternative keys or `.get()` fallback are used. The added reporting test
calls the real runner on a preflight failure (no fitting), verifies that both
files exist, and executes the notebook's reporting cell with the returned dict.
The supplied historical `KeyError` cannot be reproduced against this local pair;
the unverified v8 runtime schema remains a limitation. Existing saved evaluation
results and notebook outputs are preserved.

## Verification

Small fixed-seed unit and extraction-only integration tests are in:

- `tests/test_sensor_augmentation.py`
- `tests/test_augmentation_integration.py`
- `tests/test_augmentation_failure_reporting.py`
- `tests/test_pretraining_reporting_schema.py`

These cover valid nontrivial and single-component quaternions, individual
operations and their stack, short sequences, no-ops, metadata/row alignment,
cloning/reproducibility, no input mutation, bounded optional diagnostics,
training/validation boundaries and actual cleaner/extractor integration without
neural-network training. Existing signal-processing, preprocessing, search and
notebook suites provide compatibility checks. No Kaggle training is launched.

Verification in the local `.venv` used `python -m unittest discover -s tests`
with the following filename patterns:

| Pattern | Result |
| --- | --- |
| `test_sensor_augmentation.py` | 21 passed |
| `test_augmentation*.py` | 21 passed (9 failure reporting + 12 integration) |
| `test_pretraining_reporting_schema.py` | 1 passed |
| `test_multibranch*.py` | 159 run, OK, 17 optional tests skipped |
| `test_signal_processing_math.py` | 16 passed |
| `test_base_utils_memory.py` | 9 passed |
| `test_search_dependency_audit.py` | 22 passed |
| `test_pretraining_validation_search.py` | 11 passed |
| `test_pretraining_notebook_outputs.py` | 4 passed |

The existing stage helper contract
`test_searched_epochs_and_separate_learning_rates_reach_training_calls` also
passed after preserving calls that do not initialize a source augmentor.
Python compilation of the four changed modules and `git diff --check` passed.

A broader `test_pretraining*.py` run exposed three pre-existing notebook
expectation failures (model 1 chunk lengths, source choices and learning rates).
AST comparison of `HEAD` and the working notebook confirmed all three choices
were already identical before these changes; they were deliberately preserved.
Three CPU training tests are blocked by the installed TensorFlow/Keras import:
`TypeError: unhashable type: 'list'` inside TensorFlow/Python 3.9 typing.
The unchanged import block from `HEAD` produces the same unavailable flag.
These tests did not start training. The relevant model-free checks above pass.

An independent existing extractor limitation remains: resampling with
`linear_acc_modes='raw'` can discard `lin_acc_*` columns before their feature
extractor consumes them. This also reproduces without augmentation. The
resampling regression here exercises baseline gravity removal and quaternion
SLERP without requesting that separate corrected-acceleration output mode.
