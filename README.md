# `cmi_dexter` — Multimodal Wearable Sensor Pipeline for CMI BFRB Detection (v4)

Modular, library-first multimodal time-series feature extraction, augmentation, and deep/tabular modeling pipeline for the **CMI Detect Behavior with Sensor Data** competition (100 Hz wrist IMU + 20 Hz 5-channel Thermopile + 20 Hz 5×8×8 Time-of-Flight proximity grids).

> **Research Disclaimer**: This repository is a research and competition machine-learning codebase for wearable movement pattern recognition. None of the signal descriptors, stress/variability indices, or classifiers are medical devices or validated for clinical diagnosis or treatment.

---

## Key Capabilities (v4.0.0 — Consolidated Honeycomb 2-File Architecture)

- **Clean 2-File Parent–Child Honeycomb Layout ([src/base_utils.py](file:///c:/Users/maran/OneDrive/Documents/Git%20Profile/cmi_dexter/src/base_utils.py) + [src/utils.py](file:///c:/Users/maran/OneDrive/Documents/Git%20Profile/cmi_dexter/src/utils.py))**: All shared physics (`PhysicsKernel`), Honeycomb preprocessing & window feature cells (`HoneycombBase`, `WindowFeatureCell`), six real OSS feature backends (`FeatureBackend`), modular signal descriptors (`SignalDescriptorOptions` / `NoopOptions`, `SignalDescriptorCell`), augmentation adapter (`TsaugAdapter`, `BaseAugmentor`), and abstract parents (`BaseSequenceExtractor`, `BaseSequenceClassifier`, `BaseSearchCV`) live in `src/base_utils.py`. Multi-branch PyTorch & unified search children (`MultiModalSequenceExtractor`, `TSAugmentor`, `MultiBranchTSAIClassifier`, `MultiBranchGASearchCV`, `HierarchicalBayesSearchCV`) inherit directly from `base_utils` in `src/utils.py` (renamed from `src/utils_v3.py`).
- **Unified Shared Physics (`PhysicsKernel` in `src/base_utils.py`)**: Per-modality sample-rate handling (`100 Hz` IMU, `20 Hz` THM/ToF), anti-aliased polyphase resampling, exact `scipy.spatial.transform.Rotation` math (`rot6d`, Euler, body angular velocity, per-row `body`/`world` gravity removal), configurable ToF invalid sentinels (`(-1.0,)` default), periodic-Hann PSD, and STFT edge-coverage compensation serving both `SequenceExtractor` and `MultiModalSequenceExtractor`.
- **Six Genuine Open-Source Static Backends (`FeatureBackend` children in `src/base_utils.py`)**: `TSFELBackend` (4 domains incl. `fractal`), `TSFreshBackend` (batched), `Catch22Backend` (`pycatch22`, `catch24=True`), `TSFlexBackend` (incl. MIT `mad-lab-fau/sleep_analysis` multi-scale actigraphy bank with D6 fix), `SeglearnBackend` (`FeatureRep`), and `SktimeBackend` (`SummaryTransformer`, `Catch22`, `MiniRocket`).
- **Pluggable Honeycomb Domain Cells (`WindowFeatureCell` + `FEATURE_REGISTRY` in `src/base_utils.py`)**: `HandednessCanonicalizer`, `DenoisingFilterCell`, `OrientationPostureCell`, `VibrationMultiaxialCell`, `ToFSpatialCell`, `ToFTemporalCell`, `ThermalDynamicsCell`, `CrossModalCell`, `EventsComplexityCell`, `MultiScaleCell`, `SequenceAggregationCell`, `SignalDescriptorCell`, and `ExampleMeanCell` — all tracked via JSON-serializable `feature_manifest_`.
- **Per-Modality Augmentation & Optional IMU-Only Mode**: Native `modalities="imu"` specialist mode (zero THM/ToF columns) and per-modality augmentation with quaternion unit-norm protection via `TsaugAdapter` (`src/base_utils.py`) and `TSAugmentor` (`src/utils.py`).
- **PyTorch Pretraining, Masked-Signal SSL & Ensembling ([src/pretraining_ssl_utils.py](file:///c:/Users/maran/OneDrive/Documents/Git%20Profile/cmi_dexter/src/pretraining_ssl_utils.py) + `MultiBranchEnsemble`)**:
  - `notebooks/multibranch.ipynb` (consolidated from `multi-branch_version_3.ipynb`) supports supervised gesture pretraining, masked-signal SSL reconstruction, and head warmup with all pretraining/SSL params defaulting to `0`, and saves model ensemble bundles (`weights.pt`, `config.json`, `oof_proba.csv`, `holdout_proba.csv`, `manifest.json`).
  - `notebooks/multibranch_ensemble.ipynb` scales cleanly across **2 to 5 models** using `MultiBranchEnsemble` (`Ensemble`) — supporting both loading saved bundles and end-to-end multi-model training across `'hierarchical'` (Stage-1 binary `bfrb` gate + modality-aware routing), `'soft_vote'`, and `'stacking'` modes.
  - `notebooks/multibranch_alpha.ipynb` stays intact on `src/base_utils.py` + `src/multibranch_utils.py`.
- **Preserved Legacy Filter**: `ProblematicSequenceFilter(HoneycombBase)` in [src/base_utils.py](file:///c:/Users/maran/OneDrive/Documents/Git%20Profile/cmi_dexter/src/base_utils.py) is preserved 100% untouched.

---

## Installation & Pinned Environment

Pinned versions verified on Python 3.10 (`ds_kaggle` Conda environment) are tracked in [requirements-features.txt](file:///c:/Users/maran/OneDrive/Documents/Git%20Profile/cmi_dexter/requirements-features.txt):

```bash
pip install -r requirements-features.txt
```

> **Note on `tsaug`**: Upstream [`arundo/tsaug`](https://github.com/arundo/tsaug) is archived/unmaintained. All `tsaug` imports are isolated inside `TsaugAdapter` in [src/base_utils.py](file:///c:/Users/maran/OneDrive/Documents/Git%20Profile/cmi_dexter/src/base_utils.py), which automatically falls back to NumPy/SciPy equivalents if `tsaug` is unavailable and allows swapping augmentation backends in one class.

---

## Quickstart Configuration Example

```python
import sys
sys.path.insert(0, "src")

from utils import MultiModalSequenceExtractor, NoopOptions, make_synthetic_cmi_df

df = make_synthetic_cmi_df(n_subjects=4, sequences_per_subject=4, n_timesteps=100)

ext = MultiModalSequenceExtractor(
    window_seconds=1.0,
    canonicalize_handedness=True,
    imu_feature_preset="public_core_jerk",
    rotation_modes="quaternion|angular_velocity|rot6d|angular_distance",
    tof_modes="raw|pooled_stats|sensor_stats|block_2x2",
    static_backend="catch22",                # 'none' | 'tsfel' | 'tsfresh' | 'catch22' | 'tsflex' | 'seglearn' | 'sktime'
    noop=NoopOptions(motion_stats="kimmitch", periodicity="acf_kimmitch", spectral_stats="all"),
    descriptor_naming="canonical",           # 'legacy' (noop_*) or 'canonical' (scientific names)
    orientation_features=True,
    vibration_features=True,
    tof_spatial_features=True,
    tof_temporal_features=True,
    thermal_features=True,
    cross_modal_features=True,
    events_complexity_features=True,
)
chunks = ext.fit(df).transform_chunks(df)
print("Static features:", chunks["static"].shape)
print("Manifest summary:", ext.feature_manifest_.to_dict()["families"])
```

### IMU-Only Hidden-Test Specialist Mode
```python
ext_imu = MultiModalSequenceExtractor(
    window_seconds=1.0,
    modalities="imu",                        # omits THM/ToF branches and static columns completely
    imu_feature_preset="public_core_jerk",
    static_backend="catch22",
    orientation_features=True,
    vibration_features=True,
)
```

---

## Running the Test Suite

```powershell
$env:PATH = "C:\Users\maran\anaconda3\envs\ds_kaggle\Library\bin;" + $env:PATH
$env:PYTHONPATH = "src;tests"
C:\Users\maran\anaconda3\envs\ds_kaggle\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

---

## Documentation Map

- [SUMMARY.md](file:///c:/Users/maran/OneDrive/Documents/Git%20Profile/cmi_dexter/SUMMARY.md) — Executive summary, empirical GroupKFold before/after results, and per-module verdicts
- [docs/ARCHITECTURE.md](file:///c:/Users/maran/OneDrive/Documents/Git%20Profile/cmi_dexter/docs/ARCHITECTURE.md) — System Mermaid diagram, package layout, and 5-step plugin guide
- [docs/CALCULATION_CATALOG.md](file:///c:/Users/maran/OneDrive/Documents/Git%20Profile/cmi_dexter/docs/CALCULATION_CATALOG.md) — Full mathematical formula, library, units, edge-case, and test catalog
- [docs/DECISIONS.md](file:///c:/Users/maran/OneDrive/Documents/Git%20Profile/cmi_dexter/docs/DECISIONS.md) — Engineering decision log (D1–D10)
- [docs/MIGRATION.md](file:///c:/Users/maran/OneDrive/Documents/Git%20Profile/cmi_dexter/docs/MIGRATION.md) — v3 → v4 migration guide and canonical-to-legacy alias table
- [docs/REFERENCES.md](file:///c:/Users/maran/OneDrive/Documents/Git%20Profile/cmi_dexter/docs/REFERENCES.md) — Pinned libraries, verified URLs, repository license audit, and papers
- [CHANGELOG.md](file:///c:/Users/maran/OneDrive/Documents/Git%20Profile/cmi_dexter/CHANGELOG.md) — Release history (`4.0.0`)
- [KAGGLE_WORKFLOW.md](file:///c:/Users/maran/OneDrive/Documents/Git%20Profile/cmi_dexter/KAGGLE_WORKFLOW.md) — Kaggle dataset snapshot & notebook execution workflow (`scripts/kaggle_experiment.py` packs nested `src/**/*.py`)
