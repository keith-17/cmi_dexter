# Optional NOOP-inspired motion summaries

`notebooks/multibranch_pretraining.ipynb` is updated in place on the
`multibranch/v2/develop` branch; V2 names the branch, not a new notebook.
All existing non-NOOP Grid, Bayesian and Evolutionary entries retain their
original ranges. The new extractor produces chunk-local static features in a
separate `noop` model branch; it does not append constant copies to every
timestep or replace acceleration, rotation, ToF, temperature, STFT or CWT inputs.

## Configuration and exact off behavior

```python
from noop_sequence_extractor import NoopSequenceExtractor, NoopOptions

# Exact legacy/off behavior, including the original chunk dictionary.
baseline = NoopSequenceExtractor(noop=None, output_format="chunks")
also_off = NoopSequenceExtractor(noop=NoopOptions(), output_format="chunks")

# Enable only the requested methods. None, rather than a boolean, disables each family.
candidate = NoopSequenceExtractor(
    output_format="chunks",
    noop=NoopOptions(
        motion_stats="acc_ac_rms|jerk_rms|rotation_mean",
        periodicity="acf_harmonic_checked",
        spectral_stats="peak_fraction|band_fractions",
        quality="report",
        signal="raw",
        band=(0.5, 8.0),
        bands=((0.5, 2.0), (2.0, 4.0), (4.0, 8.0)),
        min_cycles=2.0,
        min_samples=16,
        max_gap_samples=0,
        detrend="constant",
    ),
)
```

The search prefix is `estimator__extractor__noop__`. The extractor exposes
`noop__` defaults through `get_params(deep=True)` even when `noop=None`;
`set_params(noop__motion_stats="axis_rms")` lazily creates a `NoopOptions`
object. Shallow parameters retain `noop=None` for exact off-state cloning.
Setting all four method fields to `None` disables all calculations. The notebook's
explicit `BEST_PARAMS` contains the 215 winning Candidate 21 values from
`experiments/noop_candidate21/reference_params.json`, plus canonical NOOP defaults
with all four methods `None`. This baseline uses 2,000 epochs, no early stopping,
and final epoch weights. Later search-space cells do not overwrite these values
with the first Grid choices. Conditional
candidate signatures ignore numeric controls that cannot affect the selected
methods, and the notebook's Grid path deduplicates those effective candidates.

| Option | Choices | Default |
|---|---|---|
| `motion_stats` | `None`; any pipe-delimited subset of `acc_ac_rms`, `axis_rms`, `jerk_rms`, `rotation_mean`, `rotation_rms`; aliases `basic`, `full` | `None` |
| `periodicity` | `None`, `acf`, `acf_harmonic_checked` | `None` |
| `spectral_stats` | `None`; any pipe-delimited subset of `peak_fraction`, `peak_neighbourhood_fraction`, `band_fractions`, `entropy`; alias `all` | `None` |
| `quality` | `None`, `report`, `features` | `None` |
| `signal` | `raw`, `linear`, `angular_speed`; selects periodicity, spectral and quality source | `raw` |
| `band`, `bands` | Frequency bounds in Hz; tuple/list or JSON-encoded equivalents | `(0.5, 8.0)`; `((0.5, 2.0), (2.0, 4.0), (4.0, 8.0))` |
| `min_cycles` | Positive cycles required at a candidate frequency | `2.0` |
| `min_samples` | Minimum usable sample count | `16` |
| `max_gap_samples` | Maximum bounded missing run eligible for interpolation; zero disables it | `0` |
| `acf_harmonic_tolerance` | Relative tolerance for magnitude double-frequency ambiguity | `0.1` |
| `spectral_neighbourhood_hz` | Half-width around a spectral peak | `0.5` |
| `detrend` | `constant`, `linear` | `constant` |

These initial frequency ranges are editable motion-analysis settings, not
validated BFRB diagnostic bands. They are not copied from NOOP's gait or heart
rate limits. Bounds are checked against the configured sampling rate's Nyquist
frequency. The saved sampling configuration is still an assumption about the
recordings unless independently verified from acquisition metadata.

## Features, existing CMI capabilities, and provenance

The reference is the inspected NOOP fork, `keith-17/cr7_kimmitch`, commit
`2a1de7609843b095505274ad6fc811757210be80`. The implementation adapts the
mathematical methods to CMI's signals, units, missing data and model chunks.

| Addition | NOOP source | CMI adaptation and limits |
|---|---|---|
| Acceleration AC RMS | `Packages/StrandAnalytics/Sources/StrandAnalytics/ImuFeatureExtractor.swift`, `extract` | Standard deviation of raw acceleration magnitude over observed chunk samples. This does **not** subtract a gravity vector. CMI already has raw and corrected magnitudes; the optional addition is the compact chunk summary. |
| Axis RMS and jerk RMS | Same `extract` function's energy and first-difference summaries | Axis RMS retains DC, including gravity. Jerk uses the vector difference divided by the actual configured interval, and includes only adjacent observed samples with consecutive counters. CMI already has axis jerk in `IMUExtractor.transform`; this is RMS aggregation, not a new sensor or derivative family. |
| Rotation mean/RMS | Same `extract` function's gyro-magnitude mean | CMI uses quaternion-derived body angular velocity in **rad/s**, with quaternion validity and adjacent-sample checks. It is not the reference's measured gyroscope in degrees/s. CMI's `RotationExtractor.transform` already supplies this angular velocity. |
| ACF strength/frequency and harmonic ambiguity | `ImuFeatureExtractor.extract`; `Packages/WhoopProtocol/Sources/WhoopProtocol/PpgHr.swift`, `detrend`, `acf`, `estimate`, `derivePpgHr` | Normalized autocorrelation on signed axes or angular-speed magnitude, limited to supported frequencies and a contiguous run. The harmonic-checked variant flags possible magnitude frequency doubling; it does not guess a corrected cadence. |
| Restricted-band concentration | `Packages/StrandAnalytics/Sources/StrandAnalytics/SleepStagerV2.swift`, `respRegularity` | A Hann-windowed, one-sided PSD supplies peak fraction, peak-neighbourhood fraction, configurable band fractions and normalized entropy. CMI already has STFT band fractions/entropy; the distinction is this explicit chunk-local, support-aware summary, computed before any sequence-wide summary is broadcast. |
| Quality/support | `PpgHr.derivePpgHr` consecutive-run segmentation and missing-estimate handling | Preserve observation provenance; do not compress gaps into an artificial clock. `report` emits per-chunk diagnostics without adding quality model columns. `features` also adds observed, imputed, longest-run and counter-gap fractions. Existing CMI validity and padding masks remain in place. |

An unsupported scalar is zero with a separate `_supported` column. Padding does
not count as observed data. An unavailable estimate is distinguishable from a
measured zero. Periodicity and spectral calculations use the longest valid
contiguous run inside each model chunk, after optional bounded interpolation.
Counter discontinuities split runs; rows on either side are not concatenated.

The enabled extractor returns `noop_features` with shape `[chunks, features]`
and `noop_feature_names` in a fixed order. `noop_quality`, when requested,
contains aligned diagnostics. The estimator saves `noop_feature_names_` and
uses the same branch assembly for supervised fitting, pretraining, masked SSL,
validation and prediction. Feature extraction and augmentation order must keep
the summaries consistent with the actual model inputs.

The model's sampling interval, outer chunk duration and spectral duration are
different controls. A 250-sample chunk at 100 Hz is 2.5 seconds. An ACF frequency
must fit at least `min_cycles` into its usable contiguous run; for a full
2.5-second run and two required cycles this excludes frequencies below 0.8 Hz.
Padding and denser FFT samples do not create additional resolving power.

## Permutation importance in the trained model

The notebook's final section uses the **saved internal validation IDs** of its
fitted `AdvancedMultiBranchSequenceClassifier` and the corresponding rows from
`X_train`/`y_train`. It then applies the fitted extractor's fine-tuning-domain
selection: reserved raw validation IDs may also contain non-target recordings
whose labels are outside the trained final-task classes. It does not use the final holdout, train a random-forest
proxy, inspect attention weights, or refit the neural network.

```python
from multibranch_utils import compute_permutation_importance

importance = compute_permutation_importance(
    fitted_pipeline, X_internal, y_internal,
    repeats=3,
    random_state=42,
    include_noop_individual=True,
    output_dir="results/internal_importance",
    prefix="matched_noop_internal",
)
```

The helper extracts model inputs **once**. It computes baseline probabilities
through the actual fitted network, averages window probabilities per sequence
using the classifier's aggregation rule, and calculates macro-F1 over the
fitted classifier's fixed class set. Each permutation exchanges complete
sequence blocks, preserving chunk order, timesteps and masks. Donors must have
the same number of chunks and exact valid-timestep layout; singleton layouts
remain unchanged. The report states the fraction of sequences that could
actually move, and raises rather than reporting fabricated importance when no
compatible pair exists.

The default named groups are:

- `motion_coherent`: available acceleration, derived IMU, rotation, STFT, CWT
  and NOOP summaries move together from the same donor sequence.
- `tof`, `thermo`, `noop`: each available branch moves separately.
- Individual NOOP columns, including their support indicators, are also
  evaluated one column at a time. These marginal permutations can break
  correlations, so compare them with the coherent group result.

The score is `baseline_macro_f1 - permuted_macro_f1`. A larger positive drop
indicates stronger dependence of this trained model on that input. Negative
values are retained. Redundant/correlated inputs can conceal individual
importance; a low score does not prove a sensor contains no useful information.
This is model dependence, not causal attribution. Internal validation may
already have guided checkpoint/parameter selection, so these results remain
exploratory and do not replace a fresh final evaluation.

When an output directory is supplied, the helper saves:

- `<prefix>.csv`: means, repeat standard deviations, per-repeat scores and
  permutation coverage.
- `<prefix>.json`: the same scores plus scope, sequence IDs, baseline and donor
  policy.
- `<prefix>.png`: a plot explicitly labelled as internal-validation sequence
  macro-F1 drop in the trained multibranch network.

The returned DataFrame's `attrs` contains scope, baseline, coverage and saved
paths. Original input arrays and fitted weights are never mutated; an exception
during prediction leaves those inputs intact.

## Search configuration

Choose `search_mode = 'grid'`, `'bayes'`, or `'evo'` to use the corresponding
editable search space. The Grid, Bayesian and Evolutionary declarations remain
independent. `search_mode = 'best params'` evaluates the saved `BEST_PARAMS`.

For a baseline/NOOP comparison, keep split manifests, seeds, preprocessing,
augmentation, architecture, optimizer, epochs and checkpoint policy identical.
Use the exact-off extractor for baseline, then change only NOOP method choices.
Log the effective parameters and feature names, including support columns.
Do not tune features against the final holdout. The explicit baseline
`BEST_PARAMS` now carries the discussed 2,000-epoch winning configuration;
non-NOOP search ranges remain independently editable and preserved.

The focused tests check optional notebook wiring and preservation of editable search declarations,
whole-sequence donors, padding preservation, internal-partition guards,
single-extraction behavior, per-column perturbation, input restoration on
failure, zero importance for an input-independent predictor, and CSV/JSON/plot
output. A fixture predictor tests the permutation machinery without requiring
TensorFlow; actual neural training must be verified separately where the
TensorFlow runtime is available.
