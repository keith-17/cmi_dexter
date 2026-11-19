# Architecture & Engineering Decision Log (`DECISIONS.md`)

All significant design decisions made during the v4 Honeycomb refactor of `cmi_dexter`, together with concrete empirical/technical rationale and trade-offs.

---

## D1. Consolidated 2-File Honeycomb Architecture (`src/base_utils.py` → `src/utils.py`)
- **Decision**: Consolidate all shared physics (`PhysicsKernel`), Honeycomb preprocessing and window feature cells (`HoneycombBase`, `WindowFeatureCell`), six genuine open-source backends (`FeatureBackend`), modular signal descriptors (`SignalDescriptorOptions` / `NoopOptions`, `SignalDescriptorCell`), augmentation adapter (`TsaugAdapter`, `BaseAugmentor`), and abstract parents (`BaseSequenceExtractor`, `BaseSequenceClassifier`, `BaseSearchCV`) directly into [src/base_utils.py](file:///c:/Users/maran/OneDrive/Documents/Git%20Profile/cmi_dexter/src/base_utils.py). Rename `src/utils_v3.py` → [src/utils.py](file:///c:/Users/maran/OneDrive/Documents/Git%20Profile/cmi_dexter/src/utils.py) as the child PyTorch multi-branch & unified search module inheriting from `base_utils` parent classes everywhere possible.
- **Rationale**: Eliminates 36 scattered small files and duplicated helpers (`find_data_root`, `ECat`/`EInt`/`EFloat`, search/extractor/augmentor/classifier boilerplate), simplifies Kaggle packaging and notebook imports to two primary modules (`base_utils` and `utils`), keeps `InvalidParams` class identity stable across `importlib.reload(utils)`, and enforces true Honeycomb OOP composition (`HoneycombBase` → `WindowFeatureCell` / `BaseSequenceExtractor` / `BaseAugmentor`).
- **Trade-off**: `src/base_utils.py` is larger as a single file, organized into 10 clearly delimited architectural sections with strict parent–child class boundaries.

## D2. `ProblematicSequenceFilter` Left 100% Untouched
- **Decision**: Keep `ProblematicSequenceFilter(HoneycombBase)` in `src/base_utils.py` byte-for-byte identical to `HEAD`—never edited, replaced, deleted, or silently enabled.
- **Rationale**: Explicit project invariant to preserve exact backward compatibility for existing exploratory pipelines and prevent unrequested data filtering. Verified by `test_problematic_sequence_filter_remains_untouched` and git diff against `HEAD`.

## D3. Library-First Static Feature Backends (Deleting Hand-Rolled Proxies)
- **Decision**: Delete the hand-rolled `_tsfresh_features` bypass (including its fabricated `log1p(mean_abs_change/...)` proxy disguised as `approximate_entropy`) and the hand-rolled `_tsfel_features` 16-feature bypass. Route `static_backend` dynamically through `FEATURE_REGISTRY.get_backend()` supporting six real `FeatureBackend` children in `src/base_utils.py`: `TSFELBackend`, `TSFreshBackend`, `Catch22Backend`, `TSFlexBackend`, `SeglearnBackend`, and `SktimeBackend`.
- **Rationale**: Ensures every named library feature computes the authentic published algorithm (`tsfresh.extract_features`, `tsfel.time_series_features_extractor` including the `fractal` domain, `pycatch22.catch22_all(catch24=True)`, `tsflex.features.FeatureCollection`, `seglearn.transform.FeatureRep`, and `sktime` `SummaryTransformer` / `Catch22` / `MiniRocket`).
- **Trade-off**: Calling real `tsfel`/`tsfresh` is slightly slower per window than NumPy proxies, so fast C-backed `pycatch22` (`catch22`) and lightweight `seglearn` / `none` are exposed alongside `tsfel` in search spaces.

## D4. Scientific Naming with Full Backward-Compatible `noop_*` Aliases
- **Decision**: Structure physiological/kinematic descriptors cleanly under `SignalDescriptorOptions` (`NoopOptions` alias) and `SignalDescriptorCell(WindowFeatureCell)` in `src/base_utils.py`, with canonical scientific names (`descriptor_naming="canonical"`) and 1-to-1 legacy `noop_*` aliases (`descriptor_naming="legacy"`, default).
- **Rationale**: Clear scientific/algorithmic names make feature manifests and ablation reports self-documenting, while preserving `NoopOptions` and `noop_*` column names by default guarantees zero breaking changes for existing pipelines and tests.

## D5. Clean-Room & License Compliance Policy for Reference Repositories
- **Decision**: Audit every external reference repository license before writing code:
  - `mad-lab-fau/sleep_analysis` (**MIT License**): Multi-scale rolling actigraphy feature bank adapted using `tsflex.features.FeatureCollection` with attribution in `TSFlexBackend` (`src/base_utils.py`), fixing upstream D6 off-by-one (`np.arange(1, 20)` omitting scale 20 -> `range(1, window_size + 1)` with `include_max_window=True`).
  - `OpenStrap/analytics` & `OpenStrap/edge` (**AGPL-3.0**): Zero code copied; clean-room mathematical concepts only (e.g., calibration-invariant local-baseline `dyn_amp` from Straczkiewicz et al., 2023).
  - `Joffcom/my-whoop`, `patrickloeber/whoop-analyzer`, `A1fred00-datascience/WHOOP` (**No LICENSE file**): Zero code copied; domain workflow ideas only (and explicitly rejected `A1fred00-datascience/WHOOP`'s pre-split scaling pattern to preserve strict fold-level leakage hygiene).

## D6. Isolating Archived `tsaug` Behind `TsaugAdapter` (`src/base_utils.py`)
- **Decision**: Route all `tsaug` imports through a single `TsaugAdapter` class in `src/base_utils.py` (with zero `tsaug` imports anywhere in `src/utils.py` or other files), with a deterministic NumPy/SciPy fallback and optional `per_modality_augmentation=True` (Gaussian noise + unit-norm quaternion renormalization for IMU; uniform noise without multiplicative gain distortion for calibrated °C/mm thermopile/ToF).
- **Rationale**: Upstream `arundo/tsaug` is archived/unmaintained; isolating it behind one adapter class ensures future replacement requires touching only `TsaugAdapter`.

## D7. Configurable ToF Invalid Sentinels (Verified Against Real CMI Data)
- **Decision**: Keep `DEFAULT_TOF_INVALID_SENTINELS = (-1.0,)` while exposing `tof_invalid_sentinels` as a configurable parameter (default `(-1.0,)`).
- **Rationale**: Empirical audit of real `data/train.csv` and `data/sample.csv` confirmed ToF readings lie strictly in `[-1.0, 249.0]`, where `-1.0` represents no return, `0.0` is a valid near-contact reading, and `255` / `4000` do not appear.

## D8. Pluggable `WindowFeatureCell`, `FEATURE_REGISTRY` & `FeatureManifest` Provenance
- **Decision**: All 7 domain feature cells (`OrientationPostureCell`, `VibrationMultiaxialCell`, `ToFSpatialCell`, `ToFTemporalCell`, `ThermalDynamicsCell`, `CrossModalCell`, `EventsComplexityCell`) and any future add-ons inherit from `WindowFeatureCell(HoneycombBase)` and register onto `FEATURE_REGISTRY` in `src/base_utils.py`. Both `SequenceExtractor` (`base_utils.py`) and `MultiModalSequenceExtractor` (`utils.py`) inherit manifest and registry dispatch from `BaseSequenceExtractor(HoneycombBase)`.
- **Rationale**: Satisfies strict Open-Closed extensibility—adding a new sensor or feature family requires only subclassing `WindowFeatureCell` (like `ExampleMeanCell`) and calling `.register(FEATURE_REGISTRY)`.

## D9. First-Class IMU-Only Regime (`modalities="imu"`) & Opt-In Hidden-Test Simulation
- **Decision**: Add `modalities="all" | "imu" | "imu|thm" | "imu|tof"` to `MultiModalSequenceExtractor`, keep `simulate_hidden_test = False` (opt-in) and enable `run_permutation_importance = True` in `notebooks/multi-branch_version_3.ipynb`.
- **Rationale**: Supports clean IMU-only specialist modeling (emitting zero `thm`/`tof` branch or static columns) while keeping default notebook validation fast on normal holdout splits.

## D10. Mixed-Rate Time-Aligned Cross-Modal Synchronization
- **Decision**: In `CrossModalCell` (`src/base_utils.py`), project paired signals with differing native rates (e.g., 100 Hz IMU vs 20 Hz THM/ToF) onto a shared uniform time grid in physical seconds (`min(fs_a, fs_b)`) before computing lagged correlations (in seconds: `-0.2, -0.1, 0.0, +0.1, +0.2 s`), spectral coherence (`scipy.signal.coherence`), and conditional contact statistics.
- **Rationale**: Sample-index pairing across 100 Hz and 20 Hz branches is physically meaningless; physical-second alignment ensures lags, coherence bands, and pre/post-contact thermal slopes have exact physical meaning regardless of target resampling rates.
