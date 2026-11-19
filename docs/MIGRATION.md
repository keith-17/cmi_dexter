# Migration Guide (`MIGRATION.md`) — v3 to v4

`cmi_dexter` v4 consolidates the entire Honeycomb foundation into **`src/base_utils.py`** and renames **`src/utils_v3.py` → `src/utils.py`** (where all multi-branch PyTorch extractors, augmentors, classifiers, and search classes inherit cleanly from `base_utils` parent classes). Existing constructor defaults, `NoopOptions`, `noop_*` column names, and `base_utils.SequenceExtractor` defaults remain backward-compatible.

> **Module Rename (`utils_v3` → `utils`)**: Replace `import utils_v3` / `from utils_v3 import ...` with `import utils` / `from utils import ...`. All shared Honeycomb parent classes, domain cells, OSS backends, physics, descriptors, and `FEATURE_REGISTRY` live in `base_utils` and are re-exported by `utils`.

---

## 1. Key Behavioral Fixes (Automatic Improvements)

| Area | v3 Behavior | v4 Behavior | Action Needed |
|---|---|---|---|
| **Fake `tsfresh` `approximate_entropy`** | `_tsfresh_features` computed a fabricated `log1p(mean_abs_change/...)` proxy instead of calling `tsfresh`. | Deleted. `static_backend="tsfresh"` invokes genuine batched `tsfresh.extract_features()`. | None; values for `tsfresh__*approximate_entropy*` now match real `tsfresh`. |
| **Hand-rolled `tsfel` 16-feature bypass** | `_tsfel_features` bypassed `tsfel` for default features and lacked `fractal` domain. | Deleted. `static_backend="tsfel"` invokes genuine `tsfel.time_series_features_extractor()` across all 4 domains (`statistical`, `temporal`, `spectral`, `fractal`). | None; optionally explore `tsfel_domains="all"` or `tsfel_features="fractal_complexity"`. |
| **`'all'` mode alias** | `_pipe_modes` rejected `'all'` on `acc_modes`, `rotation_modes`, `thm_modes`, `tof_modes`, `static_channels`, `tsfel_domains`, `power_branches`. | Supported everywhere via `features.physics.parse_pipe_modes(..., support_all_alias=True)`. | None. |
| **Partial-NaN quaternion gravity removal** | `q_obs.all()` discarded quaternion gravity removal for the entire sequence if a single row had NaN quaternion. | Per-row `remove_gravity_quaternion()` applies exact `scipy.spatial.transform.Rotation` gravity removal on every valid row and falls back only on invalid rows. | None; optionally set `linear_acc_frame="world"`. |
| **`base_utils.py` STFT / CWT rates** | Hardcoded `sampling_rate=20` even on 100 Hz IMU columns (5× frequency shift). | `column_sampling_rates` map & `sampling_rate="auto"` (from `dt`) wired automatically by `SequenceExtractor`. | None; pass explicit `column_sampling_rates` if calling `STFTExtractor`/`CWTExtractor` standalone. |
| **`base_utils.py` raw gravity integration** | Integrated raw `acc_z ≈ 9.81 m/s²` directly via `cumsum` (quadratic drift). | Gated by `allow_raw_gravity_integration=False` (raises `InvalidExtractorParams` unless `integration_highpass_hz > 0` or linear acceleration is used). | Use `linear_acc_modes="velocity|displacement"` or set `integration_highpass_hz=0.3`. |
| **`base_utils.py` `MotionFilter`** | `extended_kalman` was a 1-D scalar filter misnamed as EKF. | Canonical name `kalman_1d` added; `kalman` kept as alias; `extended_kalman` emits `DeprecationWarning`. | Migrate `'extended_kalman'` -> `'kalman_1d'`. |
| **`base_utils.py` `window_size` collision** | `SignalCleaner` and `IMUExtractor` shared one ambiguous `window_size`. | Split into explicit `highpass_window_size` (`SignalCleaner`) and `smoothing_window_size` (`IMUExtractor`) while retaining `window_size` fallback. | Prefer explicit `highpass_window_size` / `smoothing_window_size`. |
| **L2-consistent `acc_gyro_coupling`** | Mixed `mean(||omega||)` with `std(||a||)`. | Uses consistent L2-RMS ratio `RMS(||omega||) / (RMS(||a|| - mean) + 1e-4)`. | None. |
| **`ProblematicSequenceFilter`** | Present in `src/base_utils.py`. | **100% untouched** (never edited, deleted, or enabled). | None. |

---

## 2. Signal Descriptor Renaming (Legacy vs Canonical Scientific Names)

Set `descriptor_naming="canonical"` on `MultiModalSequenceExtractor` to emit canonical scientific column names; keep default `descriptor_naming="legacy"` for exact v3 `noop_*` compatibility. Both names are recorded in `extractor.feature_manifest_`.

| Legacy Alias (`descriptor_naming="legacy"`) | Canonical Scientific Name (`descriptor_naming="canonical"`) | Module |
|---|---|---|
| `noop_acc_ac_rms` | `dyn_acc_magnitude_ac_rms` | `signal_descriptors.dynamic_kinematics` |
| `noop_acc_{x,y,z}_rms` | `dyn_acc_{x,y,z}_rms` | `signal_descriptors.dynamic_kinematics` |
| `noop_jerk_rms` | `dyn_acc_jerk_rms` | `signal_descriptors.dynamic_kinematics` |
| `noop_rotation_mean_rad_s` | `dyn_angular_speed_mean_rad_s` | `signal_descriptors.dynamic_kinematics` |
| `noop_rotation_rms_rad_s` | `dyn_angular_speed_rms_rad_s` | `signal_descriptors.dynamic_kinematics` |
| `noop_gyro_energy_dps` | `dyn_angular_speed_mean_deg_s` | `signal_descriptors.dynamic_kinematics` |
| `noop_posture_var` | `dyn_posture_covariance_trace` | `signal_descriptors.dynamic_kinematics` |
| `noop_motion_intensity_mean` | `dyn_vector_delta_mean` | `signal_descriptors.dynamic_kinematics` |
| `noop_motion_variance` | `dyn_vector_delta_variance` | `signal_descriptors.dynamic_kinematics` |
| `noop_motion_cv` | `dyn_vector_delta_cv` | `signal_descriptors.dynamic_kinematics` |
| `noop_jerk_floor` | `dyn_quiescent_jerk_floor` | `signal_descriptors.dynamic_kinematics` |
| `noop_move_fraction` / `noop_still_fraction` | `dyn_jerk_active_fraction` / `dyn_jerk_still_fraction` | `signal_descriptors.dynamic_kinematics` |
| `noop_jerk_gate_ratio` | `dyn_jerk_peak_to_floor_ratio` | `signal_descriptors.dynamic_kinematics` |
| `noop_acc_gyro_corr` / `noop_gyro_to_acc_ratio` | `dyn_acc_gyro_envelope_corr` / `dyn_gyro_to_acc_l2_rms_ratio` | `signal_descriptors.dynamic_kinematics` |
| `noop_dyn_amp` *(new)* | `dyn_local_baseline_amplitude` | `signal_descriptors.dynamic_kinematics` |
| `noop_dog_variability` | `walch_dog_variability` | `signal_descriptors.walch_dog` |
| `noop_baevsky_amo_pct` / `mxdmn` / `si` | `baevsky_mode_amplitude_pct` / `baevsky_robust_range_mxdmn` / `baevsky_stress_index` | `signal_descriptors.baevsky_stress` |
| `noop_acf_frequency_hz` / `strength` / `magnitude_harmonic` | `acf_dominant_frequency_hz` / `acf_dominant_strength` / `acf_magnitude_double_harmonic_flag` | `signal_descriptors.autocorrelation_periodicity` |
| `noop_cadence_hz` / `cadence_sublag_hz` / `cadence_strength` / `acf_sublag_concavity` | `acf_cadence_fundamental_hz` / `acf_parabolic_sublag_fundamental_hz` / `acf_cadence_strength` / `acf_parabolic_peak_concavity` | `signal_descriptors.autocorrelation_periodicity` |
| `noop_spectral_peak_fraction` / `neighbourhood` / `entropy_normalized` / `peakedness` | `psd_hann_peak_power_fraction` / `psd_hann_peak_neighbourhood_fraction` / `psd_hann_shannon_entropy_norm` / `psd_hann_herfindahl_peakedness` | `signal_descriptors.hann_spectral` |
| `noop_observed_fraction` / `imputed` / `longest_contiguous` / `counter_gap` | `qual_observed_sample_fraction` / `qual_imputed_sample_fraction` / `qual_longest_contiguous_run_fraction` / `qual_counter_gap_fraction` | `signal_descriptors.signal_quality` |
| `noop_{thm,tof}_{drift_slope,dev_rms,motion_cv}` | `branch_{thm,tof}_{linear_drift_slope_per_s,baseline_deviation_rms,step_delta_cv}` | `signal_descriptors.branch_drift` |

---

## 3. New Optional v4 Extractor & Augmentor Parameters (All Default Off / Backward-Compatible)

- **Modality & Preprocessing**: `modalities="all"`, `canonicalize_handedness=False`, `handedness_col="handedness"`, `imu_feature_preset=None`, `linear_acc_frame="body"`, `lag_diff_scales=None`, `filter_mode="none"`, `tof_spike_threshold=None`, `tof_invalid_sentinels=(-1.0,)`, `baseline_mode="none"`, `min_observed_fraction=0.0`, `scale_observed_only=False`, `descriptor_naming="legacy"`.
- **Extended Branch Modes**: `rotation_modes` adds `rot6d`, `angular_distance`, `angular_accel`, `angular_jerk`; `tof_modes` adds `block_2x2`, `block_4x4`.
- **OSS Backends**: `static_backend` adds `catch22`, `tsflex`, `seglearn`, `sktime` alongside real `tsfel` and `tsfresh`.
- **Pluggable Domain Families**: `orientation_features`, `vibration_features`, `tof_spatial_features`, `tof_temporal_features`, `thermal_features`, `cross_modal_features`, `events_complexity_features`, `window_seconds_list`, `sequence_aggregation_features`, `extra_feature_families`.
- **Augmentor**: `TSAugmentor(..., per_modality_augmentation=False)`.
