# CMI Dexter `v4.0.0` Refactor & Empirical Evaluation Summary

**Branch:** `multibranch/v4/develop`  
**Versions:** `ALGO_VERSION = "4.0.0"`, `FEATURE_VERSION = "4.0.0"`  
**Test Suite Verification:** **174 / 174 unit tests passing** across 15 suites (`0` failures, `0` errors) in `ds_kaggle`  
**Notebook Hidden-Test Simulation:** Disabled (`simulate_hidden_test = False` in `notebooks/multi-branch_version_3.ipynb`)  
**Preservation Guarantee:** `ProblematicSequenceFilter` in `src/base_utils.py` was left **100% untouched** (verified byte-identical to `HEAD`, 1,850 bytes).

---

## 1. Empirical Before/After Results & Per-Module Verdicts

### Protocol (Zero Leakage, Normal Holdout Only)
- **Fast sample benchmark (`data/sample.csv`):** All **`n = 37` sequences across `31` subjects** (`9` BFRB target classes + `non_bfrb`), joined with `data/train_demographics.csv` (`handedness`).
- **Balanced larger real-data check (`data/train.csv` subset):** **`n = 120` sequences across `61` distinct subjects** (5 balanced classes × 24 sequences each: `Eyelash - pull hair`, `Forehead - pull hairline`, `Neck - pinch skin`, `Neck - scratch`, `non_bfrb`), joined with `data/train_demographics.csv`.
- **CV Split:** 3-fold subject-grouped `GroupKFold(n_splits=3)` on `subject` (no subject ever appears in both train and validation). All scalers, feature selectors (`static_top_k`), and classifiers are fit strictly inside each fold's training split.
- **Metric:** Exact competition metric `competition_score = (binary_f1 + gesture_f1) / 2.0` on normal holdout (no hidden-test simulation).

### 1A. End-to-End Neural Comparison (`MultiBranchTSAIClassifier`, Conv1d, 3-fold subject `GroupKFold`)

| Dataset (`n` / subjects) | Configuration | `comp` (`Δ`) | `binary_f1` | `gesture_f1` | Time (3 folds) | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `data/sample.csv` (`n=37` / `31` subj, 5 epochs) | `neural_00_baseline_v3` (Before) | `0.4039` | `0.7869` | `0.0208` | 15.9s | Baseline |
| `data/sample.csv` (`n=37` / `31` subj, 5 epochs) | `neural_01_v4_bundle_topk32` (v4 + `static_top_k=32` + per-modality aug) | **`0.4174`** (`+0.0135`) | **`0.7931`** | **`0.0417`** | 32.0s | **`worked`** |
| `data/train.csv` subset (`n=120` / `61` subj, 8 epochs) | `neural_00_baseline_v3` (Before) | `0.7068` | `0.9149` | `0.4987` | 11.4s | Baseline |
| `data/train.csv` subset (`n=120` / `61` subj, 8 epochs) | `neural_01_v4_bundle_topk64` (v4 + `static_top_k=64` + per-modality aug) | **`0.7414`** (`+0.0346`) | **`0.9534`** | **`0.5294`** | 52.6s | **`worked`** |

---

### 1B. Per-Module Tabular Ablations (`ExtraTreesClassifier`, 3-fold subject `GroupKFold`)

Baselines (`00_baseline_v3_default`, 3 static cols):
- `data/sample.csv` (`n=37` / `31` subj): **`comp = 0.2897`** (`bin = 0.5294`, `gest = 0.0500`), `4.9s`
- `data/train.csv` subset (`n=120` / `61` subj): **`comp = 0.7378`** (`bin = 0.9167`, `gest = 0.5589`), `5.0s`

| Module / Config | Static Cols | `sample.csv` (`n=37`) `comp` (`Δ`) | `sample.csv` Time | `train.csv` (`n=120`) `comp` (`Δ`) | `train.csv` `bin` / `gest` | Verdict & Notes |
| --- | ---: | ---: | ---: | ---: | :---: | --- |
| `09_dom_public_core_rot6d` (`public_core_jerk` + `rot6d` + `angular_distance` + `lag_diff=(1,2,5)`) | 3 | **`0.3792`** (`+0.0895`) | 2.8s | **`0.7709`** (`+0.0331`) | `0.9436` / `0.5981` | **`worked`** — top consistent fast win across both datasets with 0 static bloat. |
| `10_dom_orientation` (`orientation_posture`) | 23 | **`0.3698`** (`+0.0801`) | 3.1s | **`0.7542`** (`+0.0164`) | `0.9474` / `0.5609` | **`worked`** — consistent lift on both datasets in ~3s (`sample.csv`). |
| `06_oss_seglearn` (`FeatureRep`) | 39 | **`0.3583`** (`+0.0686`) | 7.7s | **`0.7597`** (`+0.0219`) | `0.9424` / `0.5771` | **`worked`** — fast, compact, consistent lift on both datasets. |
| `08_dom_handedness` (X-mirror + SO(3) conj + THM/ToF 3↔5 + 8×8 flip) | 3 | **`0.3370`** (`+0.0473`) | 3.5s | **`0.7657`** (`+0.0279`) | `0.9381` / `0.5932` | **`worked`** — pure physics win on both datasets, 0 added columns. |
| `02_oss_tsfel_real` (4-domain real TSFEL) | 67 | **`0.3005`** (`+0.0108`) | 19.7s | **`0.7825`** (`+0.0447`) | `0.9436` / `0.6215` | **`worked`** — #2 on `train.csv` (`n=120`) and positive on `sample.csv`. |
| `05_oss_tsflex` (incl. MIT `sleep_analysis` actigraphy bank, D6 fixed) | 99 | **`0.2955`** (`+0.0058`) | 56.4s | **`0.7833`** (`+0.0455`) | `0.9479` / `0.6187` | **`worked`** — #1 on `train.csv` (`n=120`) and positive on `sample.csv`; slowest backend. |
| `04_oss_catch22` (`pycatch22`, `catch24=True`) | 103 | `0.2563` (`-0.0334`) | 5.4s | **`0.7646`** (`+0.0268`) | `0.9326` / `0.5965` | **`worked` (`n=120`)** / **`inconclusive` (`n=37`)** — fastest strong OSS backend at `n=120` (`5.8s`); doubled `gest` on `sample.csv` (`0.0500→0.1125`) but dipped on `bin`. |
| `03_oss_tsfresh_minimal` (real batched `tsfresh`) | 67 | `0.2778` (`-0.0119`) | 12.6s | **`0.7483`** (`+0.0105`) | `0.9474` / `0.5491` | **`worked` (`n=120`)** / **`inconclusive` (`n=37`)** — replaces fabricated v3 proxy. |
| `07_oss_sktime` (`SummaryTransformer`) | 39 | **`0.3648`** (`+0.0751`) | 21.6s | `0.7398` (`+0.0020`) | `0.9534` / `0.5262` | **`worked` (`n=37`)** / **`inconclusive` (`n=120`)** — strong `binary_f1` on both. |
| `13_dom_thermal` (`thermal_dynamics`) | 34 | **`0.3413`** (`+0.0516`) | 3.5s | `0.7397` (`+0.0019`) | `0.9130` / `0.5664` | **`worked` (`n=37`)** / **`inconclusive` (`n=120`)** — fast (`3.5s`), mild positive lift. |
| `16_dom_seq_agg` (`sequence_aggregation_features`) | 338 | **`0.4095`** (`+0.1198`) | 9.5s | `0.7155` (`-0.0223`) | `0.9375` / `0.4935` | **`inconclusive` (wide: 338 cols)** — #1 on `sample.csv`, diluted unpruned trees on `train.csv`; pair with `static_top_k`. |
| `12_dom_tof_spatial_temporal` (`tof_spatial` + `tof_temporal`) | 81 | **`0.3595`** (`+0.0698`) | 7.4s | `0.7210` (`-0.0168`) | `0.9167` / `0.5253` | **`inconclusive` (wide: 81 cols)** — strong on `sample.csv` (`gest=0.1307`), mild dip unpruned on `train.csv`; pair with `static_top_k`. |
| `11_dom_vibration` (`vibration_multiaxial`) | 90 | `0.2960` (`+0.0063`) | 5.4s | `0.7216` (`-0.0162`) | `0.9529` / `0.4902` | **`inconclusive`** — lifts `binary_f1` (`0.9529`), 90 unpruned cols dilute gesture splits; prune with `static_top_k`. |
| `18_v4_curated_bundle` (combined domain + `catch22` + descriptors) | 476 | `0.2326` (`-0.0571`) | 26.4s | **`0.7667`** (`+0.0289`) | `0.9534` / `0.5800` | **`worked` (`n=120` & pruned neural)** / **`didnt` (unpruned tabular on `n=37`)** — 476 unpruned cols overfit `25` train seqs/fold on `sample.csv`, but win with `static_top_k` in neural (`+0.0135` on `n=37`, `+0.0346` on `n=120`). |
| `01_signal_descriptors_full` (`SignalDescriptorOptions` full) | 119 | `0.2366` (`-0.0531`) | 7.3s | `0.7366` (`-0.0012`) | `0.9175` / `0.5556` | **`didnt` (unpruned standalone)** — 119 unpruned cols dilute small folds; use compact subsets or `static_top_k`. |
| `15_dom_events_complexity` (peaks, DWT, PSD, Hjorth, DFA/AMI) | 42 | `0.2437` (`-0.0460`) | 5.2s | `0.7351` (`-0.0027`) | `0.9319` / `0.5383` | **`didnt` / `inconclusive`** — negative on `sample.csv`, neutral on `train.csv`. |
| `14_dom_cross_modal` (`cross_modal`) | 49 | `0.2508` (`-0.0389`) | 8.6s | `0.6810` (`-0.0568`) | `0.9189` / `0.4431` | **`didnt`** — consistent drop on both datasets on short `0.8s` windows (`8–40` samples); only test with longer windows (`>= 1.5s`) + `static_top_k`. |
| `17_modalities_imu_only` (`modalities="imu"`) | 183 | `0.2631` (`-0.0266`) | 6.8s | `0.6474` (`-0.0904`) | `0.9312` / `0.3636` | **`didnt` (as full-sensor replacement)** — expected since THM/ToF are excluded; intended strictly as an optional IMU-only specialist. |

> **Synthetic CMI smoke check (`n = 48` seqs, `12` subjects, 3-fold GroupKFold):** All evaluated configs (`00`, `01`, `04`, `18`) hit `comp = 1.0000` (`bin = 1.0000`, `gest = 1.0000`) — pipeline sanity verified (`inconclusive` for ranking due to ceiling saturation).

---

## 2. Consolidated 2-File Honeycomb Architecture (`src/base_utils.py` + `src/utils.py`)

1. **Consolidated Honeycomb Parent Layer (`src/base_utils.py`):**
   - Folded all shared physics (`PhysicsKernel`), registry (`FEATURE_REGISTRY`, `FeatureManifest`), preprocessing cells (`HandednessCanonicalizer`, `DenoisingFilterCell`), signal descriptors (`SignalDescriptorOptions` / `NoopOptions`, `SignalDescriptorCell`), 11 `WindowFeatureCell(HoneycombBase)` subclasses (`OrientationPostureCell`, `VibrationMultiaxialCell`, `ToFSpatialCell`, `ToFTemporalCell`, `ThermalDynamicsCell`, `CrossModalCell`, `EventsComplexityCell`, `MultiScaleCell`, `SequenceAggregationCell`, `SignalDescriptorCell`, `ExampleMeanCell`), 6 genuine `FeatureBackend` children (`TSFELBackend`, `TSFreshBackend`, `Catch22Backend`, `TSFlexBackend`, `SeglearnBackend`, `SktimeBackend`), `TsaugAdapter`, and shared search descriptors (`ECat`, `EInt`, `EFloat`) directly into `src/base_utils.py` — removing the 36 scattered v4 sub-files.
   - Introduced clean abstract parent classes: `BaseSequenceExtractor(HoneycombBase)` → `SequenceExtractor`, `BaseAugmentor(HoneycombBase)` → `SensorAugmentor`, `BaseSequenceClassifier` → `RandomForestSequenceClassifier`, `BaseSearchCV` → `GASearchCV`, and `InvalidParams` → `InvalidExtractorParams`.
   - Hardened `SequenceExtractor` STFT/CWT column rates & `sqrt(coverage)` amplitude compensation, gravity-gated integration, split highpass/smoothing windows, `kalman_1d` alias, SciPy `Rotation`, `extra_feature_families` hook, and `feature_manifest_` — while keeping **`ProblematicSequenceFilter(HoneycombBase)` 100% untouched**.
2. **Child Multi-Branch PyTorch & Unified Search Layer (`src/utils.py`, renamed from `src/utils_v3.py`):**
   - Deleted `src/utils_v3.py` and refactored `src/utils.py` to inherit cleanly from `base_utils` parents everywhere possible:
     - `MultiModalSequenceExtractor(BaseSequenceExtractor)` (full `HoneycombBase` transformer with `.transform()` + `.transform_chunks()`, dynamic `FEATURE_REGISTRY` dispatch, and `feature_manifest_`)
     - `TSAugmentor(BaseAugmentor)` (per-modality tensor augmentation via `TsaugAdapter` with quaternion unit-norm protection)
     - `MultiBranchTSAIClassifier(BaseSequenceClassifier)`
     - `MultiBranchGASearchCV(BaseGASearchCV)` (aliased as `GASearchCV` / `HierarchicalGASearchCV`) and `HierarchicalBayesSearchCV(BaseSearchCV)`
   - Deleted v3's fabricated `_tsfresh_features` (`approximate_entropy` proxy), hand-rolled `_tsfel_features` bypass, inline duplicated descriptor logic, and duplicated `find_data_root` / search helpers.
3. **Notebook, Packaging, Docs & Tests:**
   - `notebooks/multi-branch_version_3.ipynb`: Updated imports to `base_utils` / `utils`, expanded Grid / Bayesian / Evolutionary search spaces to cover all v4 leaf parameters (zero `.update()`, zero `FunctionDef`, empty cell outputs), BFRB spectral bands, `simulate_hidden_test = False`, and `run_permutation_importance = True`.
   - `scripts/kaggle_experiment.py`: Packs clean `src/*.py` snapshot.
   - Renamed `tests/test_utils_v3.py` → `tests/test_utils.py` and updated all 6 new test suites (`test_physics_unification`, `test_signal_descriptors`, `test_oss_parity`, `test_no_reinvented_wheel`, `test_registry`, `test_domain_features`) including explicit Honeycomb parent–child `issubclass(...)` assertions.
