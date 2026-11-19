# Changelog

All notable changes to `cmi_dexter` are documented in this file.

## [4.1.0] — 2026-11-19 (PyTorch Pretraining/SSL, Notebook Consolidation & 2–5 Model Ensemble)

### Added
- **PyTorch Pretraining & Masked-Signal SSL (`src/pretraining_ssl_utils.py`)**:
  - `PretrainingAndSSLMixin`, `MaskedSSLReconstructionHead`, `transfer_torch_backbone_weights`, `corrupt_raw_dataframe_spans`, `select_pretraining_sequences`, and `PretrainedMultiBranchTSAIClassifier`.
  - Integrated into `MultiBranchTSAIClassifier` in `src/utils.py` with all pretraining/SSL knobs defaulting strictly to `0` / `0.0` / `'none'` / `False` for zero-overhead default execution.
- **Model Bundle Persistence & `MultiBranchEnsemble` (`Ensemble`) in `src/utils.py`**:
  - `MultiBranchTSAIClassifier.save_bundle` / `load_bundle` persist `weights.pt`, `config.json`, `oof_proba.csv`, `holdout_proba.csv`, and `manifest.json` cleanly without pickled torch modules.
  - `MultiBranchEnsemble(BaseSequenceClassifier)` supports **2 to 5+ models** (either loading saved bundles via `use_saved_bundles=True` or end-to-end GroupKFold OOF training via `use_saved_bundles=False`) across `'hierarchical'` (Stage-1 BFRB gate + IMU-specialist boost + Stage-2 gesture routing), `'soft_vote'`, and `'stacking'` modes.
- **Consolidated Notebooks & Test Suite (`tests/test_pretraining_ssl_and_ensemble.py`)**:
  - Renamed `notebooks/multi-branch_version_3.ipynb` → `notebooks/multibranch.ipynb` with pretraining/SSL search leaves set to `0` and bundle saving enabled.
  - Added `notebooks/multibranch_ensemble.ipynb` in the exact style of `multibranch.ipynb`.

### Removed
- Removed deprecated legacy notebooks (`notebooks/multibranch_v1.ipynb`, `notebooks/multibranch_pretraining.ipynb`, `notebooks/multibranch_ssl.ipynb`) while keeping `notebooks/multibranch_alpha.ipynb`.
- Removed dead legacy Keras helper modules (`src/pretraining_utils.py`, `src/pretraining_search.py`, `src/noop_sequence_extractor.py`) and obsolete tests tied strictly to them.

## [4.0.0] — 2026-11-19 (`ALGO_VERSION = "4.0.0"`, `FEATURE_VERSION = "4.0.0"`)

### Added
- **Consolidated 2-File Honeycomb Parent–Child Architecture (`src/base_utils.py` + `src/utils.py`)**:
  - Renamed `src/utils_v3.py` → `src/utils.py` and consolidated all shared physics, Honeycomb preprocessing/feature cells, OSS backends, signal descriptors, augmentation adapter, and search drivers cleanly into `src/base_utils.py`.
  - Established full parent–child inheritance across `base_utils.py` → `utils.py`:
    - `HoneycombBase` → `BaseSequenceExtractor` → `SequenceExtractor` (`base_utils.py`) & `MultiModalSequenceExtractor` (`utils.py`)
    - `HoneycombBase` → `BaseAugmentor` → `SensorAugmentor` (`base_utils.py`) & `TSAugmentor` (`utils.py`)
    - `HoneycombBase` → `WindowFeatureCell` → 11 concrete Honeycomb cells (`OrientationPostureCell`, `VibrationMultiaxialCell`, `ToFSpatialCell`, `ToFTemporalCell`, `ThermalDynamicsCell`, `CrossModalCell`, `EventsComplexityCell`, `MultiScaleCell`, `SequenceAggregationCell`, `SignalDescriptorCell`, `ExampleMeanCell`)
    - `FeatureBackend` → 6 concrete OSS library backends (`TSFELBackend`, `TSFreshBackend`, `Catch22Backend`, `TSFlexBackend`, `SeglearnBackend`, `SktimeBackend`)
    - `BaseSequenceClassifier` → `RandomForestSequenceClassifier` (`base_utils.py`) & `MultiBranchTSAIClassifier` (`utils.py`)
    - `BaseSearchCV` → `GASearchCV` (`base_utils.py`) → `MultiBranchGASearchCV` (`utils.py`) & `HierarchicalBayesSearchCV` (`utils.py`)
    - `InvalidParams` → `InvalidExtractorParams`
- **Shared Physics Kernel & Provenance (`PhysicsKernel`, `FEATURE_REGISTRY`, `FeatureManifest` in `src/base_utils.py`)**:
  - Single canonical source of truth for constants, sampling-rate helpers, polyphase resampling, SciPy `Rotation` quaternion/rot6d/Euler/angular-velocity math, per-row gravity removal (`body` and `world` frames), configurable ToF sentinel cleaning (`DEFAULT_TOF_INVALID_SENTINELS = (-1.0,)`), periodic-Hann PSD, and STFT edge-coverage compensation.
  - Pluggable `FEATURE_REGISTRY` (`FeatureFamilySpec`, `FeatureBackend`, `register_extension`) and JSON-serializable `FeatureManifest` emitted on every `fit` / `featurize`.
- **Scientific Signal Descriptors (`SignalDescriptorOptions` / `NoopOptions`, `SignalDescriptorCell` in `src/base_utils.py`)**:
  - Clean-room calibration-invariant `dyn_amp` (`dyn_local_baseline_amplitude`), L2-consistent `acc_gyro_coupling`, Walch DoG, Baevsky Stress Index, ACF periodicity, Hann PSD, signal quality, branch drift, and selectable `descriptor_naming="legacy" | "canonical"`.
- **Six Genuine Open-Source Static Feature Backends (`FeatureBackend` children in `src/base_utils.py`)**:
  - Real `tsfel` (all 4 domains including `fractal`), real batched `tsfresh`, `pycatch22` (`catch24=True`), `tsflex` (plus MIT `mad-lab-fau/sleep_analysis` multi-scale actigraphy bank fixing upstream D6 off-by-one), `seglearn` (`FeatureRep`), and `sktime` (`SummaryTransformer`, `Catch22`, `MiniRocket`).
- **Honeycomb Preprocessing & Domain Cells (`src/base_utils.py` + `src/utils.py`)**:
  - `HandednessCanonicalizer` (exact left-to-right mirror reflection, SO(3) quaternion conjugation, THM/ToF 3↔5 swap, 8×8 horizontal grid mirror, optional 180° Z mount hook).
  - `DenoisingFilterCell` (zero-phase Butterworth low/high/bandpass, median, SavGol, VisuShrink wavelet denoising, `reject_tof_spikes`).
  - All 7 domain cells (`OrientationPostureCell`, `VibrationMultiaxialCell`, `ToFSpatialCell`, `ToFTemporalCell` + `block_2x2`/`block_4x4`, `ThermalDynamicsCell`, `CrossModalCell`, `EventsComplexityCell`), `MultiScaleCell`, and leakage-free `SequenceAggregationCell`.
  - First-class `modalities="all" | "imu" | "imu|thm" | "imu|tof"` regime, `imu_feature_preset="public_core" | "public_core_jerk"`, `lag_diff_scales`, `baseline_mode`, `scale_observed_only`.
- **Swappable Augmentation Adapter (`TsaugAdapter` in `src/base_utils.py`)**:
  - Isolated archived `tsaug` behind `TsaugAdapter` with NumPy/SciPy fallback and `per_modality_augmentation=True` (uniform THM/ToF noise + automatic quaternion unit-norm preservation).
- **Comprehensive Tests & Living Docs**: 6 new test suites + renamed `tests/test_utils.py` (`test_physics_unification`, `test_signal_descriptors`, `test_oss_parity`, `test_no_reinvented_wheel`, `test_registry`, `test_domain_features`, `test_utils`), `requirements-features.txt`, `docs/{DECISIONS,REFERENCES,ARCHITECTURE,MIGRATION,CALCULATION_CATALOG}.md`, `SUMMARY.md`, and `README.md`.

### Fixed
- Deleted fabricated `_tsfresh_features` `approximate_entropy` proxy and hand-rolled `_tsfel_features` 16-feature bypass.
- Fixed `'all'` mode alias support across all extractor and descriptor mode parameters (`acc_modes`, `rotation_modes`, `thm_modes`, `tof_modes`, `static_channels`, `tsfel_domains`, `power_branches`).
- Fixed `InvalidParams` class-identity mismatch across `importlib.reload(utils)` by defining `InvalidParams` once in `src/base_utils.py`.
- Fixed all-or-nothing quaternion gravity removal (`q_obs.all()` -> per-row `remove_gravity_quaternion`).
- Fixed `base_utils.py` `STFTExtractor` / `CWTExtractor` hardcoded 20 Hz sampling-rate bug via `column_sampling_rates` and `sampling_rate="auto"`; added `sqrt(coverage)` amplitude compensation and raw-gravity integration guard (`allow_raw_gravity_integration`, `integration_highpass_hz`); split `highpass_window_size` / `smoothing_window_size`; added `kalman_1d` and deprecated misnamed `extended_kalman`.
- Updated `notebooks/multi-branch_version_3.ipynb` to import `base_utils` / `utils`, with curated multi-option v4 search spaces, BFRB spectral bands, `simulate_hidden_test = False` (opt-in), and `run_permutation_importance = True`.

### Preserved Unchanged
- `ProblematicSequenceFilter(HoneycombBase)` in `src/base_utils.py` was **left 100% untouched**.
