# Search dependency audit beyond the backbone example

The audit followed constructor parameters into the actual model, callbacks,
augmentor and native sequence feature pipeline. A parameter is removed only when
its selected consumer is disabled, replaced, or absent. Equivalent configurations
share a cache key in both Bayesian and evolutionary search. Controls are resolved
before dependent genes, and the Bayesian compiler considers switch alternatives
before removing dimensions.

## Training and model dependencies

| Condition | Settings that do not change the effective model/training |
| --- | --- |
| Optimizer is not SGD | `momentum`. This implementation does not forward it to RMSprop either. |
| Optimizer is not AdamW | `weight_decay`. The other optimizer constructors do not receive it. |
| Loss is focal | `label_smoothing`; the implemented sparse focal loss does not read it. |
| LR scheduler disabled | `lr_scheduler_type`, `lr_factor`, `lr_patience`, `min_lr`, `cosine_t_max`, `warmup_epochs`. |
| Cosine scheduler selected | Plateau-only `lr_factor` and `lr_patience`. |
| Plateau scheduler selected | Cosine-only `cosine_t_max` and `warmup_epochs`. With `validation_split=0`, plateau creates no callback, so its factor, patience and minimum rate are also unused. |
| Native augmentation disabled by either probability being zero | Amplitudes, masks, crop, dropout, warp and RNG settings cannot affect input values. Equivalent disabled probabilities are canonicalized; an explicit empty augmentation list is preserved so defaults cannot accidentally re-enable transforms. |
| Only selected augmentations enabled | Fields belonging to unselected transforms: e.g. `noise_std` with jitter-only augmentation, or `warp_num_knots` without magnitude warping. |
| No STFT/CWT feature family can be produced | `static_branch_units` and `static_dropout`. |
| Only a spatial ToF temporal branch remains | Generic branch Conv1D encoder controls and SE controls. Shared phase-attention CNN residual/BN settings remain active when that shared CNN exists. A generic accelerometer/rotation/temperature/flattened-ToF branch reactivates its controls. |
| Legacy branch mapping contains disabled sensor entries | Unused entries in `branch_filters`, `branch_kernel_sizes`, `branch_pool_sizes` and per-branch convolution depths. Canonical mappings explicitly retain the builder's actual fallback for enabled branches. |
| Legacy custom filter list exceeds its branch depth | Unused list tail. Scalar widths and repeated terminal widths also normalize to one equivalent list without expanding deep architectures in memory. |
| Non-attention pooling, linear attention, SE off, auxiliary heads off | Attention widths/types, `se_ratio`, disabled auxiliary loss weights and class-width settings are conditional too; these earlier guards remain in place. |

## Native feature extraction dependencies

| Condition | Unused settings |
| --- | --- |
| Problematic-sequence filtering off | `ideal_skew_threshold`, `problematic_features_threshold`, `problematic_feature_cols`. |
| Kalman filtering off | `kalman_process_noise`, `kalman_measurement_noise`. |
| Dead-reckoning columns have no downstream consumer | `use_dead_reckoning`, `dead_reckoning_detrend`. Built-in IMU selectors use original axes and spectral defaults explicitly exclude these derived columns. The pipeline now skips calculating them in this case. |
| A spectral config explicitly selects a dead-reckoning column | Those settings remain active and the columns are still computed. Standalone `MotionFilter` and custom extractor subclasses retain their behavior. |
| Resampling off | All four target sampling rates plus ToF/temperature native rates. IMU and rotation native rates remain conservative because physical derivatives or spectra can consume them. |
| ToF extraction off | Sensor cap; fill policy too when resampling cannot consume it. |
| Rotation extraction off | `fix_quaternion_sign`. |
| Linear acceleration correction off | `acceleration_units`, `linear_acc_frame`, `gravity_magnitude`, `use_highpass_fallback`. |
| Smoothed acceleration not selected | `smooth_alpha`. |
| EWM smoothing selected, or smoothing absent | Rolling `window_size`, unless the high-pass fallback may still need it. |
| Chunk output | `frame_stats`. |
| Frame output | Window size, stride/ratio, padding, crop/maxlen, window caps/policies and `add_global_context`. Frame aggregation uses entire sequences and base feature statistics. Unused context values are no longer built during frame transform. |
| Ratio stride selected | Absolute `chunk_stride`. Otherwise `chunk_stride_ratio` is unused. |
| No window cap | `window_cap_policy`. |
| Cropping disabled / unlimited | `maxlen` under `sequence_crop_mode='none'`; crop policy under `maxlen=None`. Joint crop controls are still drawn before final canonicalization so search can enable cropping. |
| STFT/CWT config list disabled | All corresponding global spectral defaults. |
| Every spectral config supplies a field | The matching global default is unused. If even one stack entry needs the default, it remains active. |
| STFT uses `window_seconds` | `nperseg` is overridden, both in global defaults and within config dictionaries. |
| CWT config supplies explicit `widths` | Generated-grid `n_scales` and `min_scale` are unused. Maximum scale remains because the implementation still validates it. |
| Spectral config supplies `acc_modes` | Both spectral implementations store this argument but select inputs via `signal_columns`; changing `acc_modes` does not select derivatives. It is removed from native config cache keys. |

Recognized exact aliases also share candidates: `cnn`/`conv1d`,
`extended_kalman`/`kalman`, STFT `density`/`psd` and
`spectrum`/`power_spectrum`, and case aliases accepted by optimizer/loss builders.

## Saved notebook changes

- V1's saved Bayesian dictionary now compiles from **141 to 85 potentially active
  dimensions**. The earlier backbone-focused pass produced 110. Controllers
  that can enable consumers are retained.
- RF's grid space falls from **59 to 41 entries** and its Bayesian space from
  **60 to 42**. Nine ignored frame/chunk controls and nine augmentation controls
  are removed from each. The RF pipeline fixes `per_aug_prob=0.0`, so varying its
  augmentation probabilities/amplitudes could never augment a training sample.
- These changes remove unused choices; active model/feature choices are retained.

RF also has a coupled tree-size constraint. The installed sklearn implementation
in `sklearn/tree/_classes.py` sets
`min_samples_split = max(min_samples_split, 2 * min_samples_leaf)`.
With the notebook's leaf choices `[10, 30, 50]` and split choices `[10, 30, 50]`,
the nine raw pairs describe **five effective pairs**. All three split values
are equivalent at leaf size 30 and at leaf size 50. RF Bayesian search now uses
conditional leaf spaces with explicit budgets whose sum equals the original
`n_iter`. Fractional counts, weighted priors and tiny budgets remain unchanged
instead of making assumptions about fold size or excluding leaf choices.
An actual fitted-forest regression checks identical tree features, thresholds,
values and probabilities for equivalent integer pairs.

## Cases deliberately not collapsed without data or training evidence

Some apparently equivalent settings depend on the input sequences or the actual
training trajectory. These remain distinct:

- Window caps that no sequence reaches, padding values for batches that never
  pad, or crop lengths larger than every input sequence.
- Different ratios that happen to round to the same integer stride for a
  particular selected window length.
- Early-stopping patience values that happen to produce the same stopping epoch.
- Batch sizes larger than a particular training set, or balanced class weights
  on an exactly balanced set.
- Spectral or modality settings whose requested input columns happen to be
  missing in a particular dataset.

Custom extractor/augmentor subclasses are not assumed to follow native parameter
dependencies. Conservative Bayesian fallback may keep extra proposal dimensions
when controller products are large; canonical caching still prevents repeating
an already evaluated effective candidate.

## Verification

`tests/test_search_dependency_audit.py` checks the guards and their active
counterexamples, Bayesian switch preservation, controller mutation, disabled
descriptor draw counts, native output equivalence for unconsumed dead reckoning,
explicit spectral consumers, frame context avoidance and the RF search dictionaries.
The existing numerical, memory, search and notebook wiring suites remain in use.
