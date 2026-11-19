"""
base_utils.py

Unified Honeycomb foundation for CMI multimodal wearable time-series processing
(``ALGO_VERSION = "4.0.0"``, ``FEATURE_VERSION = "4.0.0"``).

Honeycomb Class Hierarchy
-------------------------
- ``HoneycombBase(BaseEstimator, TransformerMixin)``
    - ``ProblematicSequenceFilter``  (data-quality filter; preserved unchanged)
    - ``HandednessCanonicalizer``    (left->right spatial & sensor mirror cell)
    - ``DenoisingFilterCell``        (Butterworth / median / SavGol / wavelet / ToF spike cell)
    - ``SignalCleaner``, ``MotionFilter``, ``IMUExtractor``, ``RotationExtractor``,
      ``TOFExtractor``, ``ThermoExtractor``, ``STFTExtractor``, ``CWTExtractor``
    - ``WindowFeatureCell(HoneycombBase, ABC)``
        - ``OrientationPostureCell``, ``VibrationMultiaxialCell``, ``ToFSpatialCell``,
          ``ToFTemporalCell``, ``ThermalDynamicsCell``, ``CrossModalCell``,
          ``EventsComplexityCell``, ``MultiScaleCell``, ``SequenceAggregationCell``,
          ``SignalDescriptorCell``, ``ExampleMeanCell``
    - ``BaseSequenceExtractor(HoneycombBase, ABC)``
        - ``SequenceExtractor``            (row / frame / chunk orchestrator in ``base_utils.py``)
        - ``MultiModalSequenceExtractor``  (multi-branch PyTorch orchestrator in ``utils.py``)
    - ``BaseAugmentor(HoneycombBase, ABC)``
        - ``SensorAugmentor``              (DataFrame-level augmentor in ``base_utils.py``)
        - ``TSAugmentor``                  (window-tensor augmentor in ``utils.py``)
- ``FeatureBackend(ABC)``
    - ``TSFELBackend``, ``TSFreshBackend``, ``Catch22Backend``, ``TSFlexBackend``,
      ``SeglearnBackend``, ``SktimeBackend``
- ``BaseSequenceClassifier(BaseEstimator, ClassifierMixin, ABC)``
    - ``RandomForestSequenceClassifier``   (in ``base_utils.py``)
    - ``MultiBranchTSAIClassifier``        (in ``utils.py``)
- ``BaseSearchCV(BaseEstimator, ABC)``
    - ``GASearchCV``                       (in ``base_utils.py``)
        - ``MultiBranchGASearchCV``        (in ``utils.py``)
    - ``HierarchicalBayesSearchCV``        (in ``utils.py``)
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import itertools
import json
import traceback
import warnings
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from scipy import ndimage as sp_ndimage
from scipy import signal as sps
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.stats import (
    entropy as _sp_entropy,
    kurtosis as _sp_kurtosis,
    linregress as _sp_linregress,
    skew as _sp_skew,
    spearmanr as _sp_spearmanr,
    trim_mean as _sp_trim_mean,
)

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, make_scorer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

try:
    from scipy.signal import argrelextrema, get_window, resample_poly, stft
    from scipy.spatial.transform import Rotation, Slerp
except Exception:
    resample_poly = None
    get_window = None
    stft = None
    argrelextrema = None
    Rotation = None
    Slerp = None

try:
    from scipy.interpolate import CubicSpline, PchipInterpolator
except Exception:
    CubicSpline = None
    PchipInterpolator = None

try:
    import pywt
except Exception:
    pywt = None

try:
    import pycatch22
except Exception:
    pycatch22 = None

try:
    from skopt.space import Categorical
except Exception:
    Categorical = None

try:
    from sklearn_genetic import GASearchCV as _ExternalGASearchCV
    from sklearn_genetic.space import (
        Categorical as _ExternalECat,
        Continuous as _ExternalEFloat,
        Integer as _ExternalEInt,
    )
    SKLEARN_GENETIC_AVAILABLE = True
except Exception:
    _ExternalGASearchCV = None
    _ExternalECat = _ExternalEInt = _ExternalEFloat = None
    SKLEARN_GENETIC_AVAILABLE = False


# ===========================================================================
# 1. Shared Physics Kernel & Library Discovery
# ===========================================================================

# ---------------------------------------------------------------------------
# Versioning & Canonical Physical Constants
# ---------------------------------------------------------------------------

ALGO_VERSION = "4.0.0"
FEATURE_VERSION = "4.0.0"

SEQ_COL = "sequence_id"
COUNTER_COL = "sequence_counter"
MODALITIES = ("imu", "thm", "tof")
GRAVITY = 9.80665  # m/s^2 (standard gravity)

TOF_SHAPE = (5, 8, 8)  # 5 sensors x 8x8 pixels = 320 channels
TOF_COLUMNS = [f"tof_{s}_v{p}" for s in range(1, 6) for p in range(64)]
THM_COLUMNS = [f"thm_{s}" for s in range(1, 6)]
ACC_COLUMNS = ["acc_x", "acc_y", "acc_z"]
ROT_COLUMNS = ["rot_w", "rot_x", "rot_y", "rot_z"]
SENSOR_PREFIXES = ("acc_", "rot_", "tof_", "thm_")

# Real CMI dataset ToF values lie in [-1.0, 249.0] where -1.0 means "no return"
# and 0 is a valid near-contact reading (verified empirically on train.csv & sample.csv).
DEFAULT_TOF_INVALID_SENTINELS = (-1.0,)
TOF_FILL_VALUES: Dict[str, float] = {
    "far_255": 255.0,
    "far_500": 500.0,
    "zero": 0.0,
}
CANONICAL_TOF_FILL_MODE = "far_500"

CANONICAL_RATES: Dict[str, float] = {
    "row": 100.0,
    "imu": 100.0,
    "rot": 100.0,
    "thm": 20.0,
    "tof": 20.0,
}

LEGACY_BASE_RATES: Dict[str, float] = {
    "imu": 20.0,
    "rot": 20.0,
    "thm": 5.0,
    "tof": 5.0,
}


class InvalidParams(ValueError):
    """Raised when physical, rate, descriptor, or estimator parameters are invalid."""


InvalidPhysicsParams = InvalidParams


@dataclass(frozen=True)
class MetricEnvelope:
    """Abstention-safe descriptor envelope for scalar estimates.

    Absent or unsupported inputs produce ``value=None`` and ``supported=False``
    with a diagnostic ``note``, never a fabricated constant.
    """

    value: Optional[float]
    supported: bool
    confidence: float = 0.0
    tier: str = "measured"
    inputs_used: Tuple[str, ...] = ()
    units: str = ""
    note: str = ""

    @classmethod
    def absent(cls, note: str, *, units: str = "", inputs_used: Tuple[str, ...] = ()) -> "MetricEnvelope":
        return cls(
            value=None,
            supported=False,
            confidence=0.0,
            tier="absent",
            inputs_used=inputs_used,
            units=units,
            note=note,
        )


# ---------------------------------------------------------------------------
# Library Availability & Lazy Requirement Checks
# ---------------------------------------------------------------------------

_OPTIONAL_LIBRARIES: Dict[str, str] = {
    "torch": "torch",
    "tsai": "tsai",
    "tsflex": "tsflex",
    "tsaug": "tsaug",
    "tsfel": "tsfel",
    "tsfresh": "tsfresh",
    "pycatch22": "pycatch22",
    "seglearn": "seglearn",
    "sktime": "sktime",
    "scikit-optimize": "skopt",
    "pywt": "pywt",
    "scipy": "scipy",
    "antropy": "antropy",
}


def library_available(module_name: str) -> bool:
    """Return True if ``module_name`` can be located without importing it eagerly."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def library_status() -> Dict[str, bool]:
    """Report import availability of every open-source time-series library."""
    return {label: library_available(mod) for label, mod in _OPTIONAL_LIBRARIES.items()}


def require_library(flag: bool, name: str, pip_hint: Optional[str] = None) -> None:
    """Raise an informative ``ImportError`` when an optional backend is selected but missing."""
    if not flag:
        hint = pip_hint or (f"--no-deps {name}" if name == "tsflex" else name)
        raise ImportError(f"{name} is required for this option. Install with: pip install {hint}")


# ---------------------------------------------------------------------------
# Structured Value & Mode String Parsing
# ---------------------------------------------------------------------------

def parse_structured_value(value: Any, *, exc_type: type = ValueError) -> Any:
    """Decode JSON or Python-literal strings (used by skopt Categorical search spaces)."""
    if isinstance(value, str):
        text = value.strip()
        if text[:1] in "[({":
            try:
                return json.loads(text)
            except (ValueError, TypeError):
                try:
                    return ast.literal_eval(text)
                except (ValueError, SyntaxError) as exc:
                    raise exc_type(f"cannot parse structured value {value!r}") from exc
    return value


def parse_pipe_modes(
    value: Any,
    allowed: Iterable[str],
    *,
    name: str,
    aliases: Optional[Mapping[str, Iterable[str]]] = None,
    support_all_alias: bool = True,
    exc_type: type = ValueError,
) -> set:
    """Parse ``None``, ``'all'``, list/tuple/set, or ``'a|b'`` mode specifications."""
    allowed_set = set(allowed)
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        parts = {str(p).strip() for p in value if str(p).strip()}
        if not parts:
            return set()
        if support_all_alias and parts == {"all"}:
            return set(allowed_set)
        if not parts <= allowed_set:
            raise exc_type(f"Invalid {name}: {value!r}; expected a subset of {sorted(allowed_set)} or None")
        return parts
    if not isinstance(value, str):
        raise exc_type(f"{name} must be None, a sequence, or a '|'-separated mode string")
    cleaned = value.strip()
    if not cleaned or cleaned.lower() == "none":
        return set()
    if aliases and cleaned in aliases:
        return set(aliases[cleaned])
    if support_all_alias and cleaned == "all":
        return set(allowed_set)
    parts = {p.strip() for p in cleaned.split("|") if p.strip()}
    if not parts or not parts <= allowed_set:
        raise exc_type(f"Invalid {name}: {value!r}; expected a subset of {sorted(allowed_set)} or None")
    return parts


# ---------------------------------------------------------------------------
# Sampling Rate, Resampling & Column-Rate Resolution
# ---------------------------------------------------------------------------

def positive_rate(value: Any, name: str = "sampling rate", *, exc_type: type = ValueError) -> float:
    """Validate and return a finite strictly positive sampling rate in Hz."""
    try:
        rate = float(value)
    except (TypeError, ValueError) as exc:
        raise exc_type(f"{name} must be a finite positive number") from exc
    if not np.isfinite(rate) or rate <= 0:
        raise exc_type(f"{name} must be a finite positive number, got {value!r}")
    return rate


def rate_ratio(source: float, target: float, max_denominator: int = 1000, *, exc_type: type = ValueError) -> Fraction:
    """Rational ``target / source`` ratio for anti-aliased polyphase resampling."""
    ratio = Fraction(positive_rate(target, exc_type=exc_type) / positive_rate(source, exc_type=exc_type)).limit_denominator(max_denominator)
    if ratio <= 0:
        raise exc_type(f"cannot resample {source} Hz -> {target} Hz")
    return ratio


def seconds_to_samples(seconds: float, rate: float, minimum: int = 1) -> int:
    """Convert window duration (seconds) to sample count at ``rate`` Hz."""
    return max(int(minimum), int(round(float(seconds) * positive_rate(rate))))


def nyquist(rate: float) -> float:
    """Nyquist frequency (Hz) for a given sampling rate."""
    return positive_rate(rate) / 2.0


def infer_column_rate(
    column: str,
    column_rates: Optional[Mapping[str, float]] = None,
    *,
    imu_rate: float = CANONICAL_RATES["imu"],
    rot_rate: Optional[float] = None,
    thm_rate: float = CANONICAL_RATES["thm"],
    tof_rate: float = CANONICAL_RATES["tof"],
    default_rate: Optional[float] = None,
) -> float:
    """Resolve the physical sampling rate (Hz) of a named sensor or derived column."""
    if column_rates and column in column_rates:
        return positive_rate(column_rates[column], f"sampling rate for {column}")
    col = str(column)
    if col.startswith(("thm_", "thermal_")):
        return positive_rate(thm_rate, "thm sampling rate")
    if col.startswith("tof_"):
        return positive_rate(tof_rate, "tof sampling rate")
    if col.startswith(("rot_", "ang_", "euler_", "rot6d_")):
        return positive_rate(rot_rate if rot_rate is not None else imu_rate, "rot sampling rate")
    if col.startswith(("acc_", "lin_", "jerk_", "snap_", "imu_")):
        return positive_rate(imu_rate, "imu sampling rate")
    return positive_rate(default_rate if default_rate is not None else imu_rate, f"sampling rate for {column}")


def nearest_indices(n_source: int, source_rate: float, target_rate: float) -> np.ndarray:
    """Map each resampled output index to its nearest source index (for masks)."""
    ratio = rate_ratio(source_rate, target_rate)
    n_out = max(1, int(round(n_source * float(ratio))))
    return np.clip(np.round(np.arange(n_out) / float(ratio)).astype(int), 0, max(0, n_source - 1))


def resample_array(x: np.ndarray, source_rate: float, target_rate: float, method: str = "poly") -> np.ndarray:
    """Resample ``x`` (time on axis 0) from ``source_rate`` to ``target_rate`` Hz."""
    x = np.asarray(x, dtype=float)
    if x.shape[0] == 0 or source_rate == target_rate:
        return x.copy()
    ratio = rate_ratio(source_rate, target_rate)
    n_out = max(1, int(round(x.shape[0] * float(ratio))))
    if method == "poly":
        if x.shape[0] < 2:
            return np.repeat(x, n_out, axis=0)[:n_out]
        out = sps.resample_poly(x, ratio.numerator, ratio.denominator, axis=0, padtype="line")
        return out[:n_out] if len(out) >= n_out else np.concatenate([out, np.repeat(out[-1:], n_out - len(out), 0)])
    if method == "linear":
        grid = np.arange(n_out) / float(ratio)
        src = np.arange(x.shape[0])
        flat = x.reshape(x.shape[0], -1)
        res = np.stack([np.interp(grid, src, flat[:, k]) for k in range(flat.shape[1])], axis=1)
        return res.reshape((n_out,) + x.shape[1:])
    if method == "subsample":
        return x[nearest_indices(x.shape[0], source_rate, target_rate)]
    raise ValueError("resample method must be 'poly', 'linear' or 'subsample'")


def infer_hold_rate(values: np.ndarray, row_rate: float, max_hold: int = 50) -> float:
    """Estimate a sensor's native sampling rate from sample-and-hold run length."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    row_rate = positive_rate(row_rate, "row_sampling_rate")
    if len(values) < 4:
        return row_rate
    changed = np.any(np.diff(values, axis=0) != 0, axis=1)
    idx = np.flatnonzero(changed)
    if len(idx) < 2:
        return row_rate
    runs = np.diff(idx)
    hold = int(np.clip(np.median(runs), 1, max_hold))
    return row_rate / hold


# ---------------------------------------------------------------------------
# Contiguous Runs, Bounded Imputation, Detrending & Spectral Conventions
# ---------------------------------------------------------------------------

def contiguous_runs(valid: np.ndarray, counter: np.ndarray) -> List[np.ndarray]:
    """Split valid sample positions into contiguous counter runs (never bridging gaps)."""
    positions = np.flatnonzero(valid)
    if not len(positions):
        return []
    cuts = np.flatnonzero((np.diff(positions) != 1) | (np.diff(counter[positions]) != 1)) + 1
    return np.split(positions, cuts)


def bounded_fill(values: np.ndarray, observed: np.ndarray, counter: np.ndarray, max_gap: int):
    """Linearly interpolate interior gaps up to ``max_gap`` consecutive samples."""
    result = np.array(values, dtype=float, copy=True)
    valid = np.asarray(observed, dtype=bool).copy()
    filled = np.zeros(len(valid), dtype=bool)
    if max_gap:
        for gap in contiguous_runs(~valid, np.arange(len(valid))):
            first, last = int(gap[0]), int(gap[-1])
            if (
                len(gap) > max_gap
                or first == 0
                or last == len(valid) - 1
                or not valid[first - 1]
                or not valid[last + 1]
                or np.any(np.diff(counter[first - 1 : last + 2]) != 1)
            ):
                continue
            fraction = np.arange(1, len(gap) + 1) / (len(gap) + 1)
            result[gap] = (
                result[first - 1][None, :] * (1.0 - fraction[:, None])
                + result[last + 1][None, :] * fraction[:, None]
            )
            valid[gap] = True
            filled[gap] = True
    return result, valid, filled


def fill_missing_time(x: np.ndarray, fill_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate finite values along axis 0; returns ``(filled, observed_mask)``."""
    x = np.asarray(x, dtype=float)
    observed = np.isfinite(x)
    out = x.copy()
    if observed.all():
        return out, observed
    if not observed.any():
        out[:] = fill_value
        return out, observed
    idx = np.arange(len(x))
    if x.ndim == 1:
        ok = observed
        out = np.interp(idx, idx[ok], x[ok]) if ok.any() else np.full_like(out, fill_value)
        return out, observed
    for c in range(x.shape[1]):
        ok = observed[:, c]
        if ok.all():
            continue
        if not ok.any():
            out[:, c] = fill_value
        else:
            out[:, c] = np.interp(idx, idx[ok], x[ok, c])
    return out, observed


def detrend_values(values: np.ndarray, mode: str) -> np.ndarray:
    """Remove mean (``'constant'``) or linear trend (``'linear'``) along axis 0."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values.copy()
    if mode not in {"constant", "linear"}:
        raise ValueError("detrend mode must be 'constant' or 'linear'")
    if mode == "linear" and len(values) > 1:
        return sps.detrend(values, axis=0, type="linear")
    return values - np.mean(values, axis=0)


def one_sided_psd(coefficients: np.ndarray, nfft: int) -> np.ndarray:
    """Convert one-sided FFT/STFT complex coefficients to real-signal PSD."""
    power = np.abs(coefficients) ** 2
    weights = np.full(power.shape[0], 2.0)
    weights[0] = 1.0
    if nfft % 2 == 0:
        weights[-1] = 1.0
    return power * weights[:, None]


def window_psd(values: np.ndarray, fs: float, detrend: str = "constant") -> Tuple[np.ndarray, np.ndarray]:
    """Compute periodic-Hann one-sided PSD summed across channels using SciPy periodogram."""
    x = detrend_values(values, detrend)
    if x.ndim == 1:
        x = x[:, None]
    n = len(x)
    if n < 2:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    win = sps.get_window("hann", n, fftbins=True)
    freqs, pxx = sps.periodogram(
        x, fs=float(fs), window=win, nfft=n, detrend=False, return_onesided=True, scaling="density", axis=0
    )
    return freqs, np.sum(pxx, axis=1)


def stft_frame_coverage(
    frame_times: np.ndarray,
    signal_length: int,
    sampling_rate: float,
    nperseg: int,
    window_type: Any,
) -> np.ndarray:
    """Fraction of squared window weight backed by real, non-padded samples per STFT frame."""
    window = np.asarray(sps.get_window(window_type, nperseg, fftbins=True), dtype=float)
    sample_centers = np.rint(np.asarray(frame_times) * sampling_rate).astype(int)
    frame_samples = sample_centers[:, None] - nperseg // 2 + np.arange(nperseg, dtype=int)[None, :]
    observed = (frame_samples >= 0) & (frame_samples < signal_length)
    weights = np.square(window)
    return (observed * weights[None, :]).sum(axis=1) / weights.sum()


def resolve_power_bands(
    bands: Sequence[Tuple[float, float]],
    fs: float,
    n_samples: int,
    min_cycles: float = 2.0,
    margin: float = 0.98,
    exc_type: type = ValueError,
) -> List[Tuple[float, float, bool]]:
    """Adapt requested frequency bands ``[(low, high)]`` to sensor rate ``fs`` and length ``n_samples``."""
    fs = positive_rate(fs, exc_type=exc_type)
    ceiling = margin * fs / 2.0
    floor = min_cycles * fs / max(1, int(n_samples))
    resolved = []
    for low, high in bands:
        low, high = float(low), float(high)
        if not (np.isfinite(low) and np.isfinite(high)) or not 0 <= low < high:
            raise exc_type(f"band ({low}, {high}) must satisfy 0 <= low < high")
        lo, hi = max(low, floor), min(high, ceiling)
        resolved.append((lo, hi, bool(lo < hi)))
    return resolved


def window_starts(
    n: int,
    window: int,
    stride: int,
    final_mode: str = "overlap",
    max_windows: Optional[int] = None,
    cap_policy: str = "uniform",
    exc_type: type = ValueError,
) -> List[int]:
    """Compute window start indices covering ``n`` samples."""
    if n <= window:
        return [0]
    step = max(1, int(stride))
    starts = list(range(0, n - window + 1, step))
    last_full = starts[-1]
    if last_full + window < n:
        if final_mode == "pad":
            starts.append(starts[-1] + step)
        elif final_mode == "overlap":
            starts.append(n - window)
        elif final_mode != "drop":
            raise exc_type("final_window_mode must be 'pad', 'overlap' or 'drop'")
    starts = sorted(set(starts))
    if max_windows and len(starts) > int(max_windows):
        k = int(max_windows)
        if cap_policy == "first":
            starts = starts[:k]
        elif cap_policy == "last":
            starts = starts[-k:]
        elif cap_policy == "uniform":
            starts = [starts[i] for i in np.unique(np.linspace(0, len(starts) - 1, k).round().astype(int))]
        else:
            raise exc_type("window_cap_policy must be 'uniform', 'first' or 'last'")
    return starts


# ---------------------------------------------------------------------------
# Shared Quaternion, Rotation & Gravity Math (scipy.spatial.transform.Rotation)
# ---------------------------------------------------------------------------

def normalize_quaternions(q_wxyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize ``(N, 4)`` scalar-first ``(w, x, y, z)`` quaternions.

    Returns ``(q_unit, valid_mask)`` where invalid/zero-norm rows are set to NaN
    in ``q_unit`` and ``False`` in ``valid_mask``.
    """
    q = np.asarray(q_wxyz, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"Expected (N, 4) quaternions in (w, x, y, z) order, got shape {q.shape}")
    norms = np.linalg.norm(np.nan_to_num(q, nan=0.0), axis=1)
    valid = np.isfinite(q).all(axis=1) & (norms > 1e-12)
    out = np.full_like(q, np.nan)
    out[valid] = q[valid] / norms[valid, None]
    return out, valid


def fix_quaternion_continuity(q_wxyz: np.ndarray) -> np.ndarray:
    """Enforce hemisphere continuity ``dot(q[t-1], q[t]) >= 0`` across valid rows."""
    q = np.array(q_wxyz, dtype=np.float64, copy=True)
    finite = np.isfinite(q).all(axis=1) & (np.linalg.norm(np.nan_to_num(q), axis=1) > 1e-12)
    idx = np.flatnonzero(finite)
    if len(idx) >= 2:
        sub = q[idx]
        dots = np.sum(sub[1:] * sub[:-1], axis=1)
        signs = np.cumprod(np.r_[1.0, np.where(dots < 0, -1.0, 1.0)])
        q[idx] = sub * signs[:, None]
    return q


def quaternions_to_rotations(q_wxyz: np.ndarray) -> Rotation:
    """Convert valid scalar-first ``(w, x, y, z)`` quaternions to ``scipy.spatial.transform.Rotation``."""
    q = np.asarray(q_wxyz, dtype=np.float64)
    return Rotation.from_quat(q[:, [1, 2, 3, 0]])


def quaternions_to_rot6d(q_wxyz: np.ndarray) -> np.ndarray:
    """Convert ``(N, 4)`` scalar-first quaternions to 6D continuous representation (Zhou et al. 2019).

    Uses ``scipy.spatial.transform.Rotation.as_matrix()`` and stacks the first
    two rotation-matrix columns ``[R[:, :, 0], R[:, :, 1]]`` (column-major).
    Invalid rows return NaN.
    """
    q_unit, valid = normalize_quaternions(q_wxyz)
    out = np.full((len(q_unit), 6), np.nan, dtype=np.float64)
    if valid.any():
        mats = quaternions_to_rotations(q_unit[valid]).as_matrix()
        out[valid] = np.concatenate([mats[:, :, 0], mats[:, :, 1]], axis=1)
    return out


def quaternions_to_euler(q_wxyz: np.ndarray, *, seq: str = "xyz", unwrap: bool = False) -> np.ndarray:
    """Convert ``(N, 4)`` scalar-first quaternions to Euler angles (radians) via SciPy Rotation."""
    q_unit, valid = normalize_quaternions(q_wxyz)
    out = np.full((len(q_unit), 3), np.nan, dtype=np.float64)
    if valid.any():
        eul = quaternions_to_rotations(q_unit[valid]).as_euler(seq)
        if unwrap and len(eul) >= 2:
            eul = np.unwrap(eul, axis=0)
        out[valid] = eul
    return out


def angular_velocity_body(
    q_wxyz: np.ndarray,
    fs: Optional[float] = None,
    valid: Optional[np.ndarray] = None,
    *,
    dt: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Body-frame angular velocity (rad/s) from consecutive ``(w, x, y, z)`` quaternions.

    Uses ``R[t-1].inv() * R[t]`` via ``scipy.spatial.transform.Rotation``.
    Row 0 and any pair involving an invalid quaternion or non-positive ``dt`` are NaN.
    """
    q = np.asarray(q_wxyz, dtype=np.float64)
    n = len(q)
    out = np.full((n, 3), np.nan, dtype=np.float64)
    if n < 2:
        return out
    q_unit, ok = normalize_quaternions(q)
    if valid is not None:
        ok &= np.asarray(valid, dtype=bool)
    pairs = ok[1:] & ok[:-1]
    if not pairs.any():
        return out
    if dt is not None:
        dt_arr = np.asarray(dt, dtype=np.float64)[1:]
        dt_ok = np.isfinite(dt_arr) & (dt_arr > 0)
        pairs &= dt_ok
        if not pairs.any():
            return out
        inv_dt = np.where(dt_ok, 1.0 / np.where(dt_ok, dt_arr, 1.0), 0.0)[:, None]
    else:
        inv_dt = float(positive_rate(fs, "angular velocity fs"))
    safe = np.where(ok[:, None], q_unit[:, [1, 2, 3, 0]], np.array([0.0, 0.0, 0.0, 1.0]))
    rot = Rotation.from_quat(safe)
    delta = rot[:-1].inv() * rot[1:]
    vec = delta.as_rotvec() * inv_dt
    out[1:][pairs] = vec[pairs]
    return out


def highpass_zero_phase(
    x: np.ndarray,
    fs: float,
    cutoff_hz: float = 0.3,
    order: int = 2,
) -> np.ndarray:
    """Zero-phase Butterworth high-pass filter using ``scipy.signal.butter`` + ``sosfiltfilt``."""
    arr = np.asarray(x, dtype=np.float64)
    if len(arr) == 0:
        return arr.copy()
    fs_val = positive_rate(fs, "highpass fs")
    cut = float(cutoff_hz)
    # sosfiltfilt requires padlen = 3 * (2 * n_sections + 1 - min(zeros, poles)) -> ~15-18 samples
    min_len = max(18, 3 * (2 * int(order) + 1) + 1)
    if len(arr) > min_len and 0.0 < cut < fs_val / 2.0:
        sos = sps.butter(int(order), cut, btype="highpass", fs=fs_val, output="sos")
        return sps.sosfiltfilt(sos, arr, axis=0)
    return arr - np.mean(arr, axis=0, keepdims=True)


def remove_gravity_quaternion(
    acc: np.ndarray,
    q_wxyz: np.ndarray,
    *,
    q_valid: Optional[np.ndarray] = None,
    mode: str = "auto",
    frame: str = "body",
    gravity_magnitude: float = GRAVITY,
    acceleration_units: str = "m/s^2",
    fs: float = 100.0,
    highpass_hz: float = 0.3,
    use_highpass_fallback: bool = True,
) -> np.ndarray:
    """Remove gravity from 3-axis acceleration using SciPy ``Rotation`` with per-row fallback.

    Fixes the all-or-nothing quaternion bug: valid quaternion rows always receive
    exact orientation-based gravity removal (in ``'body'`` or ``'world'`` frame),
    while only invalid-quaternion rows (or ``mode='highpass'``) fall back to
    zero-phase Butterworth high-pass filtering.
    """
    acc_arr = np.asarray(acc, dtype=np.float64)
    n = len(acc_arr)
    if n == 0:
        return acc_arr.copy()
    if acceleration_units not in {"m/s^2", "g"}:
        raise ValueError("acceleration_units must be 'm/s^2' or 'g'")
    if frame not in {"body", "world"}:
        raise ValueError("linear_acc_frame must be 'body' or 'world'")
    g_mag = float(gravity_magnitude)
    if not np.isfinite(g_mag) or g_mag <= 0:
        raise ValueError("gravity_magnitude must be finite and positive")

    unit_scale = g_mag if acceleration_units == "g" else 1.0
    acc_ms2 = acc_arr * unit_scale
    gravity_world = np.array([0.0, 0.0, g_mag], dtype=np.float64)

    if mode == "highpass":
        return highpass_zero_phase(acc_ms2, fs=fs, cutoff_hz=highpass_hz)

    q_unit, ok = normalize_quaternions(q_wxyz)
    if q_valid is not None:
        ok &= np.asarray(q_valid, dtype=bool)

    if mode in {"baseline", "auto", "quaternion"} and ok.any():
        lin = np.empty_like(acc_ms2)
        rot = quaternions_to_rotations(q_unit[ok])
        if frame == "world":
            lin[ok] = rot.apply(acc_ms2[ok]) - gravity_world
        else:
            gravity_body = rot.inv().apply(gravity_world)
            lin[ok] = acc_ms2[ok] - gravity_body
        if not ok.all():
            fallback = (
                highpass_zero_phase(acc_ms2, fs=fs, cutoff_hz=highpass_hz)
                if use_highpass_fallback
                else (acc_ms2 - np.mean(acc_ms2, axis=0, keepdims=True))
            )
            lin[~ok] = fallback[~ok]
        return lin

    if use_highpass_fallback:
        return highpass_zero_phase(acc_ms2, fs=fs, cutoff_hz=highpass_hz)
    return acc_ms2 - np.mean(acc_ms2, axis=0, keepdims=True)


# ---------------------------------------------------------------------------
# Shared ToF Sentinel Cleaning & Fill Policy
# ---------------------------------------------------------------------------

def clean_tof_array(
    values: np.ndarray,
    *,
    invalid_sentinels: Sequence[float] = DEFAULT_TOF_INVALID_SENTINELS,
    fill_mode: str = CANONICAL_TOF_FILL_MODE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recode invalid ToF sentinels to NaN and apply the configured fill policy.

    Returns ``(observed_with_nan, filled, valid_mask)``.
    Valid measurements are finite, non-negative, and not in ``invalid_sentinels``.
    """
    arr = np.asarray(values, dtype=np.float64).copy()
    valid = np.isfinite(arr) & (arr >= 0.0)
    for sentinel in invalid_sentinels:
        s = float(sentinel)
        if np.isfinite(s):
            valid &= arr != s
    observed = np.where(valid, arr, np.nan)
    if fill_mode in TOF_FILL_VALUES:
        filled = np.where(valid, arr, TOF_FILL_VALUES[fill_mode])
    elif fill_mode == "nan_interpolate":
        filled, _ = fill_missing_time(observed, fill_value=TOF_FILL_VALUES["far_500"])
    else:
        raise ValueError(f"tof_fill_mode must be one of {sorted(TOF_FILL_VALUES) + ['nan_interpolate']}, got {fill_mode!r}")
    return observed, filled, valid


def apply_valid_mask(arr: np.ndarray, mask: np.ndarray, pad_value: float = 0.0) -> np.ndarray:
    """Enforce the padding contract: invalid/padded positions take ``pad_value``."""
    arr_np = np.asarray(arr)
    m = np.asarray(mask, dtype=bool)
    while m.ndim < arr_np.ndim:
        m = m[..., None]
    return np.where(m, arr_np, pad_value)


class InvalidExtractorParams(InvalidParams):
    """Raised when extractor hyperparameters cannot produce valid features (child of ``InvalidParams``)."""


class PhysicsKernel:
    """Object-oriented namespace exposing the shared physical signal-processing primitives."""

    ALGO_VERSION = ALGO_VERSION
    FEATURE_VERSION = FEATURE_VERSION
    GRAVITY = GRAVITY
    DEFAULT_TOF_INVALID_SENTINELS = DEFAULT_TOF_INVALID_SENTINELS

    library_available = staticmethod(library_available)
    library_status = staticmethod(library_status)
    require_library = staticmethod(require_library)
    parse_structured_value = staticmethod(parse_structured_value)
    parse_pipe_modes = staticmethod(parse_pipe_modes)
    positive_rate = staticmethod(positive_rate)
    rate_ratio = staticmethod(rate_ratio)
    seconds_to_samples = staticmethod(seconds_to_samples)
    nyquist = staticmethod(nyquist)
    infer_column_rate = staticmethod(infer_column_rate)
    nearest_indices = staticmethod(nearest_indices)
    resample_array = staticmethod(resample_array)
    infer_hold_rate = staticmethod(infer_hold_rate)
    contiguous_runs = staticmethod(contiguous_runs)
    bounded_fill = staticmethod(bounded_fill)
    fill_missing_time = staticmethod(fill_missing_time)
    detrend_values = staticmethod(detrend_values)
    one_sided_psd = staticmethod(one_sided_psd)
    window_psd = staticmethod(window_psd)
    stft_frame_coverage = staticmethod(stft_frame_coverage)
    resolve_power_bands = staticmethod(resolve_power_bands)
    window_starts = staticmethod(window_starts)
    normalize_quaternions = staticmethod(normalize_quaternions)
    fix_quaternion_continuity = staticmethod(fix_quaternion_continuity)
    quaternions_to_rotations = staticmethod(quaternions_to_rotations)
    quaternions_to_rot6d = staticmethod(quaternions_to_rot6d)
    quaternions_to_euler = staticmethod(quaternions_to_euler)
    angular_velocity_body = staticmethod(angular_velocity_body)
    highpass_zero_phase = staticmethod(highpass_zero_phase)
    remove_gravity_quaternion = staticmethod(remove_gravity_quaternion)
    clean_tof_array = staticmethod(clean_tof_array)
    apply_valid_mask = staticmethod(apply_valid_mask)



def _positive_rate(value: Any, name: str) -> float:
    try:
        rate = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidExtractorParams(f"{name} must be a finite positive number") from exc
    if not np.isfinite(rate) or rate <= 0:
        raise InvalidExtractorParams(f"{name} must be a finite positive number, got {value!r}")
    return rate


def _signal_values(signal: np.ndarray, name: str) -> Optional[np.ndarray]:
    values = np.asarray(signal, dtype=float).reshape(-1)
    finite = np.isfinite(values)
    if not finite.any():
        return None
    if not finite.all():
        raise ValueError(
            f"{name} received partially missing samples; interpolate within each sequence "
            "or choose an explicit missing-sample policy before feature extraction"
        )
    return values


def _one_sided_psd(coefficients: np.ndarray, nfft: int) -> np.ndarray:
    """Convert legacy SciPy one-sided STFT coefficients to real-signal PSD."""
    power = np.abs(coefficients) ** 2
    weights = np.full(power.shape[0], 2.0)
    weights[0] = 1.0
    if nfft % 2 == 0:
        weights[-1] = 1.0
    return power * weights[:, None]


def _stft_frame_coverage(
    frame_times: np.ndarray,
    signal_length: int,
    sampling_rate: float,
    nperseg: int,
    window_type: Any,
) -> np.ndarray:
    """Fraction of squared window weight backed by observed, non-padding samples."""
    if get_window is None:
        raise ImportError("STFT edge correction requires SciPy.")
    window = np.asarray(get_window(window_type, nperseg, fftbins=True), dtype=float)
    sample_centers = np.rint(np.asarray(frame_times) * sampling_rate).astype(int)
    frame_samples = (
        sample_centers[:, None]
        - nperseg // 2
        + np.arange(nperseg, dtype=int)[None, :]
    )
    observed = (frame_samples >= 0) & (frame_samples < signal_length)
    weights = np.square(window)
    return (observed * weights[None, :]).sum(axis=1) / weights.sum()


def _validate_counter_order(
    frame: pd.DataFrame,
    sequence_col: str,
    counter_col: str,
) -> Optional[pd.Series]:
    if counter_col not in frame.columns:
        return None
    counter = pd.to_numeric(frame[counter_col], errors="coerce")
    if counter.isna().any() or not np.isfinite(counter.to_numpy(dtype=float)).all():
        raise ValueError(f"{counter_col} must contain finite numeric sample indices")
    deltas = counter.groupby(frame[sequence_col], sort=False).diff()
    if (deltas.dropna() <= 0).any():
        raise ValueError(
            f"{counter_col} must be strictly increasing within each {sequence_col}; "
            "sort rows or remove duplicate/nonincreasing samples before extraction"
        )
    return counter


def _coerce_search_value(value: Any) -> Any:
    """Decode JSON / literal strings produced by BayesSearchCV Categorical."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text[0] in "{[":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return value
    return value


def _positive_int(value: Any, *, default: Optional[int] = None, name: str = "parameter") -> int:
    if value is None:
        if default is None:
            raise InvalidExtractorParams(f"{name} must be set")
        value = default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise InvalidExtractorParams(f"{name} must be an integer, got {value!r}") from exc
    if parsed <= 0:
        raise InvalidExtractorParams(f"{name} must be > 0, got {parsed}")
    return parsed


def validate_sequence_extractor_params(
    params: Dict[str, Any],
    *,
    for_frame_output: bool = True,
) -> None:
    output = str(params.get("output_format", "frame")).lower()
    if for_frame_output and output not in {"frame"}:
        raise InvalidExtractorParams(
            f"Sequence frame classifiers require output_format='frame', got {output!r}"
        )

    if params.get("window_size") is not None:
        _positive_int(params.get("window_size"), name="window_size")

    if params.get("maxlen") is not None:
        try:
            int(params.get("maxlen"))
        except (TypeError, ValueError) as exc:
            raise InvalidExtractorParams(
                f"maxlen must be an integer, got {params.get('maxlen')!r}"
            ) from exc

    crop_mode = params.get("sequence_crop_mode", "tail")
    if crop_mode not in {"head", "tail", "center", "none"}:
        raise InvalidExtractorParams(
            "sequence_crop_mode must be one of 'head', 'tail', 'center', or 'none'"
        )

    cap_policy = params.get("window_cap_policy", "last")
    if cap_policy not in {"last", "first", "uniform"}:
        raise InvalidExtractorParams(
            "window_cap_policy must be one of 'last', 'first', or 'uniform'"
        )

    final_window_mode = params.get("final_window_mode", "pad")
    if final_window_mode not in {"pad", "overlap"}:
        raise InvalidExtractorParams("final_window_mode must be 'pad' or 'overlap'")

    cap = params.get("max_windows_per_sequence")
    if cap is not None:
        _positive_int(cap, name="max_windows_per_sequence")

    if params.get("chunk_window_size") is not None:
        try:
            int(params.get("chunk_window_size"))
        except (TypeError, ValueError) as exc:
            raise InvalidExtractorParams(
                f"chunk_window_size must be an integer, got {params.get('chunk_window_size')!r}"
            ) from exc

    if bool(params.get("use_chunk_stride_ratio", False)):
        try:
            stride_ratio = float(params.get("chunk_stride_ratio"))
        except (TypeError, ValueError) as exc:
            raise InvalidExtractorParams(
                f"chunk_stride_ratio must be between 0 and 1, got {params.get('chunk_stride_ratio')!r}"
            ) from exc
        if not 0 < stride_ratio <= 1:
            raise InvalidExtractorParams(
                f"chunk_stride_ratio must be between 0 and 1, got {stride_ratio!r}"
            )

    resample_modalities = bool(params.get("resample_modalities", False))

    rate_specs = [
        ("imu_native_sampling_rate", "imu_target_sampling_rate"),
        ("rot_native_sampling_rate", "rot_target_sampling_rate"),
        ("tof_native_sampling_rate", "tof_target_sampling_rate"),
        ("thm_native_sampling_rate", "thm_target_sampling_rate"),
    ]

    for native_key, target_key in rate_specs:
        native = _positive_rate(params.get(native_key, 20), native_key)
        target = _positive_rate(params.get(target_key, native), target_key)

        if resample_modalities:
            if resample_poly is None:
                raise InvalidExtractorParams(
                    "resample_modalities=True requires scipy.signal.resample_poly"
                )
            ratio = target / float(native)
            if ratio > 25 or ratio < 0.04:
                raise InvalidExtractorParams(
                    f"{target_key}/{native_key} ratio {ratio:.3f} is outside safe bounds"
                )

    if not for_frame_output and not bool(params.get("use_chunk_stride_ratio", False)):
        chunk_stride = params.get("chunk_stride")
        if chunk_stride is not None:
            try:
                int(chunk_stride)
            except (TypeError, ValueError) as exc:
                raise InvalidExtractorParams(
                    f"chunk_stride must be an integer, got {chunk_stride!r}"
                ) from exc


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

# ===========================================================================
# 2. Pluggable Honeycomb Feature Registry & Provenance
# ===========================================================================

@dataclass(frozen=True)
class FeatureFamilySpec:
    """Declarative metadata + execution entry for one registered feature family."""

    name: str
    config_key: str
    function: Callable[[Mapping[str, Any], Mapping[str, Any]], Dict[str, np.ndarray]]
    required_inputs: Tuple[str, ...] = ("imu",)
    output_names_fn: Optional[Callable[[Mapping[str, Any]], List[str]]] = None
    default_enabled: bool = False
    library_used: str = "scipy/numpy"
    reference_url: str = ""
    paper_ref: str = ""
    units: str = "dimensionless"
    scope: str = "window"  # 'sample', 'window', or 'sequence'
    version: str = FEATURE_VERSION
    description: str = ""
    is_extension: bool = False


@dataclass(frozen=True)
class FeatureManifestEntry:
    """Per-feature provenance record emitted on every fit / transform."""

    feature_name: str
    family: str
    library: str
    fs: Optional[float]
    units: str
    scope: str
    version: str
    canonical_name: Optional[str] = None
    legacy_alias: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureManifest:
    """Complete provenance manifest of all features emitted by an extractor."""

    algo_version: str = ALGO_VERSION
    feature_version: str = FEATURE_VERSION
    extractor_class: str = ""
    rates: Dict[str, float] = field(default_factory=dict)
    entries: List[FeatureManifestEntry] = field(default_factory=list)

    def add(self, entry: FeatureManifestEntry) -> None:
        self.entries.append(entry)

    def feature_names(self) -> List[str]:
        return [e.feature_name for e in self.entries]

    def by_family(self) -> Dict[str, List[str]]:
        grouped: Dict[str, List[str]] = {}
        for e in self.entries:
            grouped.setdefault(e.family, []).append(e.feature_name)
        return grouped

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algo_version": self.algo_version,
            "feature_version": self.feature_version,
            "extractor_class": self.extractor_class,
            "rates": dict(self.rates),
            "feature_count": len(self.entries),
            "families": sorted({e.family for e in self.entries}),
            "entries": [asdict(e) for e in self.entries],
        }

    def to_json(self, path: Optional[Any] = None) -> str:
        payload = json.dumps(self.to_dict(), indent=2, default=str)
        if path is not None:
            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(payload, encoding="utf-8")
        return payload


class FeatureBackend(ABC):
    """Common pluggable interface for open-source and custom feature backends."""

    name: str = ""

    @abstractmethod
    def available(self) -> bool:
        """Return True if this backend's library dependencies are importable."""

    @abstractmethod
    def get_feature_names(self, config: Mapping[str, Any], input_columns: Sequence[str]) -> List[str]:
        """Return deterministic ordered output feature names for ``input_columns``."""

    @abstractmethod
    def compute_features(
        self,
        sources: Mapping[str, Mapping[str, Any]],
        config: Mapping[str, Any],
    ) -> Dict[str, np.ndarray]:
        """Compute per-window or per-sequence features without mutating ``sources``."""


class FeatureRegistry:
    """Central registry for native, domain, descriptor, and external feature families."""

    def __init__(self) -> None:
        self._families: Dict[str, FeatureFamilySpec] = {}
        self._backends: Dict[str, FeatureBackend] = {}
        # EXTENSION marker registry: external or future add-on modules append here
        # without editing core orchestrators.
        self._extensions: Dict[str, FeatureFamilySpec] = {}

    def register(self, spec: FeatureFamilySpec, *, replace: bool = True) -> FeatureFamilySpec:
        if not spec.name:
            raise ValueError("FeatureFamilySpec.name must be a non-empty string")
        if not replace and spec.name in self._families:
            raise ValueError(f"Feature family {spec.name!r} is already registered")
        self._families[spec.name] = spec
        if spec.is_extension:
            self._extensions[spec.name] = spec
        return spec

    def unregister(self, name: str) -> None:
        self._families.pop(name, None)
        self._extensions.pop(name, None)

    def register_extension(self, name: str, spec: FeatureFamilySpec) -> FeatureFamilySpec:
        """EXTENSION hook: register an add-on feature family without touching core files."""
        ext_spec = FeatureFamilySpec(
            name=name,
            config_key=spec.config_key,
            function=spec.function,
            required_inputs=spec.required_inputs,
            output_names_fn=spec.output_names_fn,
            default_enabled=spec.default_enabled,
            library_used=spec.library_used,
            reference_url=spec.reference_url,
            paper_ref=spec.paper_ref,
            units=spec.units,
            scope=spec.scope,
            version=spec.version,
            description=spec.description,
            is_extension=True,
        )
        return self.register(ext_spec, replace=True)

    def register_backend(self, backend: FeatureBackend) -> FeatureBackend:
        if not backend.name:
            raise ValueError("FeatureBackend.name must be non-empty")
        self._backends[backend.name] = backend
        return backend

    def get(self, name: str) -> FeatureFamilySpec:
        if name not in self._families:
            raise KeyError(f"Unknown feature family {name!r}; registered={sorted(self._families)}")
        return self._families[name]

    def get_backend(self, name: str) -> FeatureBackend:
        if name not in self._backends:
            raise KeyError(f"Unknown feature backend {name!r}; registered={sorted(self._backends)}")
        return self._backends[name]

    def list_families(self) -> List[str]:
        return list(self._families.keys())

    def list_backends(self) -> List[str]:
        return list(self._backends.keys())

    def list_extensions(self) -> List[str]:
        return list(self._extensions.keys())

    def is_enabled(self, spec: FeatureFamilySpec, config: Mapping[str, Any]) -> bool:
        if spec.config_key in config:
            val = config[spec.config_key]
            if val is None or val is False:
                return False
            if isinstance(val, str) and val.strip().lower() in {"", "none", "false", "off"}:
                return False
            return True
        # Also allow enabling explicitly via a list/set of extra registered family names
        extra = config.get("extra_feature_families")
        if extra:
            if isinstance(extra, str):
                tokens = {p.strip() for p in extra.split("|") if p.strip()}
            else:
                tokens = {str(p).strip() for p in extra}
            if spec.name in tokens:
                return True
        return bool(spec.default_enabled)

    def compute_enabled(
        self,
        context: Mapping[str, Any],
        config: Mapping[str, Any],
        *,
        manifest: Optional[FeatureManifest] = None,
        scopes: Optional[Sequence[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Execute every enabled registered family and record entries in ``manifest``."""
        columns: Dict[str, np.ndarray] = {}
        allowed_scopes = set(scopes) if scopes is not None else None
        available_inputs = set(context.get("branches", {}).keys()) | set(context.keys())
        rates = context.get("rates", {})

        for spec in self._families.values():
            if allowed_scopes is not None and spec.scope not in allowed_scopes:
                continue
            if not self.is_enabled(spec, config):
                continue
            if spec.required_inputs and not set(spec.required_inputs).issubset(available_inputs):
                continue
            produced = spec.function(context, config)
            if not produced:
                continue
            family_fs = rates.get(spec.required_inputs[0], rates.get("imu")) if spec.required_inputs else rates.get("imu")
            for feat_name, arr in produced.items():
                columns[feat_name] = np.asarray(arr, dtype=np.float64)
                if manifest is not None:
                    manifest.add(
                        FeatureManifestEntry(
                            feature_name=feat_name,
                            canonical_name=feat_name,
                            family=spec.name,
                            library=spec.library_used,
                            fs=float(family_fs) if family_fs is not None else None,
                            units=spec.units,
                            scope=spec.scope,
                            version=spec.version,
                            config={spec.config_key: config.get(spec.config_key, spec.default_enabled)},
                        )
                    )
        return columns


FEATURE_REGISTRY = FeatureRegistry()


def register_feature_family(spec: FeatureFamilySpec) -> FeatureFamilySpec:
    """Register ``spec`` into the global ``FEATURE_REGISTRY``."""
    return FEATURE_REGISTRY.register(spec)

# ===========================================================================
# 3. HoneycombBase & Row-Level Sensor Cells
# ===========================================================================


# ---------------------------------------------------------------------------
# Base Honeycomb Cell & Preprocessing Cells
# ---------------------------------------------------------------------------


class HoneycombBase(BaseEstimator, TransformerMixin):
    """Root parent class for all composable Honeycomb preprocessing & feature cells."""

    def fit(self, X: Any, y: Optional[Any] = None):
        return self

    def _parse_modes(self, modes_str: Optional[str]) -> List[str]:
        if not modes_str:
            return []
        return [
            m.strip().lower()
            for m in str(modes_str).split("|")
            if m.strip() and m.strip().lower() != "none"
        ]

    @staticmethod
    def _pipe_modes(
        value: Any,
        allowed: Iterable[str],
        *,
        name: str,
        aliases: Optional[Mapping[str, Iterable[str]]] = None,
    ) -> set:
        return parse_pipe_modes(
            value,
            allowed,
            name=name,
            aliases=aliases,
            support_all_alias=True,
            exc_type=InvalidParams,
        )

    @staticmethod
    def _structured(value: Any) -> Any:
        return parse_structured_value(value, exc_type=InvalidParams)



class ProblematicSequenceFilter(HoneycombBase):
    def __init__(
        self,
        ideal_skew_threshold: float = 1.35,
        problematic_features_threshold: int = 6,
        feature_cols: Optional[List[str]] = None,
        sequence_col: str = "sequence_id",
    ):
        self.ideal_skew_threshold = ideal_skew_threshold
        self.problematic_features_threshold = problematic_features_threshold
        self.feature_cols = feature_cols
        self.sequence_col = sequence_col

    def fit(self, X: pd.DataFrame, y=None):
        if self.sequence_col not in X.columns:
            raise ValueError(f"ProblematicSequenceFilter requires '{self.sequence_col}'.")
        default = ["acc_x", "acc_y", "acc_z", "rot_x", "rot_y", "rot_w", "rot_z"]
        requested = default if self.feature_cols is None else list(self.feature_cols)
        self.feature_cols_ = [col for col in requested if col in X.columns]
        if not self.feature_cols_:
            self.problematic_sequence_ids_ = pd.Index([], name=self.sequence_col)
            return self
        skew = X.groupby(self.sequence_col, sort=False)[self.feature_cols_].skew().abs()
        count = (skew > float(self.ideal_skew_threshold)).sum(axis=1)
        self.problematic_sequence_ids_ = skew.index[
            count >= int(self.problematic_features_threshold)
        ]
        return self

    def problematic_mask(self, X: pd.DataFrame) -> pd.Series:
        check_is_fitted(self, ["problematic_sequence_ids_"])
        return X[self.sequence_col].isin(self.problematic_sequence_ids_)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.loc[~self.problematic_mask(X)].copy()

# --- Handedness & Denoising Cells ---

def _lr_flip_8x8_indices() -> np.ndarray:
    """Column permutation mirroring an 8x8 row-major ToF grid left-to-right."""
    grid = np.arange(64, dtype=int).reshape(8, 8)
    return grid[:, ::-1].reshape(-1)


_TOF_LR_PERM = _lr_flip_8x8_indices()


def _is_left_handed(series: pd.Series, left_values: Sequence = (0, "0", "left", "L", False)) -> np.ndarray:
    left_set = {str(v).strip().lower() for v in left_values}
    vals = series.astype(str).str.strip().str.lower()
    return vals.isin(left_set).to_numpy(dtype=bool)


def canonicalize_handedness_frame(
    df: pd.DataFrame,
    *,
    handedness_col: str = "handedness",
    subject_col: str = "subject",
    demographics: Optional[Union[pd.DataFrame, Mapping[str, Any]]] = None,
    left_hand_values: Sequence = (0, "0", "left", "L", False),
    mount_flip_subjects: Iterable[str] = (),
) -> pd.DataFrame:
    """Transform left-handed rows into the canonical right-hand sensor coordinate frame.

    Does not mutate ``df`` in place; returns a transformed copy. If neither
    ``handedness_col`` nor ``demographics`` provides handedness information,
    only ``mount_flip_subjects`` (if given) is applied.
    """
    out = df.copy()
    if len(out) == 0:
        return out

    left_mask = np.zeros(len(out), dtype=bool)
    if handedness_col in out.columns:
        left_mask = _is_left_handed(out[handedness_col], left_hand_values)
    elif demographics is not None and subject_col in out.columns:
        if isinstance(demographics, pd.DataFrame):
            if subject_col in demographics.columns and handedness_col in demographics.columns:
                mapping = demographics.set_index(subject_col)[handedness_col].to_dict()
                mapped = out[subject_col].map(mapping)
                left_mask = _is_left_handed(mapped.fillna(1), left_hand_values)
        elif isinstance(demographics, Mapping):
            mapped = out[subject_col].map(demographics)
            left_mask = _is_left_handed(mapped.fillna(1), left_hand_values)

    if left_mask.any():
        # 1. Reflect X acceleration
        for col in ("acc_x", "lin_acc_x", "lin_x"):
            if col in out.columns:
                out.loc[left_mask, col] = -out.loc[left_mask, col].to_numpy(dtype=float)

        # 2. Conjugate quaternion R' = M_x R M_x -> (w, x, -y, -z)
        for col in ("rot_y", "rot_z"):
            if col in out.columns:
                out.loc[left_mask, col] = -out.loc[left_mask, col].to_numpy(dtype=float)

        # 3. Swap thermopile sensors 3 and 5
        if "thm_3" in out.columns and "thm_5" in out.columns:
            t3 = out.loc[left_mask, "thm_3"].to_numpy(copy=True)
            t5 = out.loc[left_mask, "thm_5"].to_numpy(copy=True)
            out.loc[left_mask, "thm_3"] = t5
            out.loc[left_mask, "thm_5"] = t3

        # 4. Swap ToF sensors 3 and 5, then left-right flip every 8x8 ToF grid
        tof3_cols = [f"tof_3_v{p}" for p in range(64)]
        tof5_cols = [f"tof_5_v{p}" for p in range(64)]
        if all(c in out.columns for c in tof3_cols) and all(c in out.columns for c in tof5_cols):
            b3 = out.loc[left_mask, tof3_cols].to_numpy(copy=True)
            b5 = out.loc[left_mask, tof5_cols].to_numpy(copy=True)
            out.loc[left_mask, tof3_cols] = b5
            out.loc[left_mask, tof5_cols] = b3

        for s in range(1, 6):
            scols = [f"tof_{s}_v{p}" for p in range(64)]
            if all(c in out.columns for c in scols):
                grid = out.loc[left_mask, scols].to_numpy(copy=True)
                out.loc[left_mask, scols] = grid[:, _TOF_LR_PERM]

    # Optional conservative 180-degree Z-axis mount correction hook
    flip_set = {str(s) for s in (mount_flip_subjects or ())}
    if flip_set and subject_col in out.columns:
        flip_mask = out[subject_col].astype(str).isin(flip_set).to_numpy(dtype=bool)
        if flip_mask.any():
            for col in ("acc_x", "acc_y", "lin_acc_x", "lin_acc_y"):
                if col in out.columns:
                    out.loc[flip_mask, col] = -out.loc[flip_mask, col].to_numpy(dtype=float)
            qcols = ["rot_w", "rot_x", "rot_y", "rot_z"]
            if all(c in out.columns for c in qcols):
                q = out.loc[flip_mask, qcols].to_numpy(dtype=float)
                norms = np.linalg.norm(np.nan_to_num(q), axis=1)
                valid = np.isfinite(q).all(axis=1) & (norms > 1e-12)
                if valid.any():
                    rz_pi = Rotation.from_euler("z", np.pi)
                    rot = Rotation.from_quat((q[valid] / norms[valid, None])[:, [1, 2, 3, 0]])
                    q_new = (rot * rz_pi).as_quat()[:, [3, 0, 1, 2]]
                    q_out = q.copy()
                    q_out[valid] = q_new
                    out.loc[flip_mask, qcols] = q_out

    return out


def apply_filter_1d_or_2d(
    arr: np.ndarray,
    fs: float,
    *,
    method: str = "none",
    lowcut_hz: Optional[float] = None,
    highcut_hz: Optional[float] = None,
    order: int = 2,
    median_kernel: int = 3,
    savgol_window: int = 7,
    savgol_polyorder: int = 2,
    wavelet: str = "db4",
    wavelet_level: Optional[int] = None,
) -> np.ndarray:
    """Apply a leakage-free zero-phase filter along axis 0 (time).

    Supported ``method`` values:
      - ``'none'``: identity copy
      - ``'lowpass'``: zero-phase Butterworth lowpass via ``scipy.signal.butter`` + ``sosfiltfilt``
      - ``'highpass'``: zero-phase Butterworth highpass via ``scipy.signal.butter`` + ``sosfiltfilt``
      - ``'bandpass'``: zero-phase Butterworth bandpass via ``scipy.signal.butter`` + ``sosfiltfilt``
      - ``'median'``: rolling median filter via ``scipy.ndimage.median_filter``
      - ``'savgol'``: Savitzky-Golay filter via ``scipy.signal.savgol_filter``
      - ``'median_lowpass'``: median de-spiking followed by zero-phase Butterworth lowpass
      - ``'wavelet'``: universal VisuShrink soft-thresholding via ``pywt.wavedec`` / ``waverec``
    """
    x = np.asarray(arr, dtype=np.float64)
    if x.size == 0 or method in {None, "", "none"}:
        return x.copy()
    was_1d = x.ndim == 1
    data = x[:, None] if was_1d else x.copy()
    n = len(data)
    nyq = float(fs) / 2.0
    m = str(method).strip().lower()

    if m == "median":
        k = max(1, int(median_kernel))
        if k % 2 == 0:
            k += 1
        k = min(k, n if n % 2 == 1 else max(1, n - 1))
        out = sp_ndimage.median_filter(data, size=(k, 1), mode="nearest") if k >= 3 else data.copy()
    elif m == "savgol":
        w = max(3, int(savgol_window))
        if w % 2 == 0:
            w += 1
        if n >= 3:
            w = min(w, n if n % 2 == 1 else n - 1)
            poly = min(int(savgol_polyorder), w - 1)
            out = sps.savgol_filter(data, window_length=w, polyorder=poly, axis=0)
        else:
            out = data.copy()
    elif m in {"lowpass", "highpass", "bandpass"}:
        min_len = max(18, 3 * (2 * int(order) + 1) + 1)
        if n <= min_len:
            out = data.copy()
        else:
            if m == "lowpass":
                cut = float(highcut_hz if highcut_hz is not None else min(15.0, 0.4 * fs))
                if 0.0 < cut < nyq:
                    sos = sps.butter(int(order), cut, btype="lowpass", fs=float(fs), output="sos")
                    out = sps.sosfiltfilt(sos, data, axis=0)
                else:
                    out = data.copy()
            elif m == "highpass":
                cut = float(lowcut_hz if lowcut_hz is not None else 0.3)
                if 0.0 < cut < nyq:
                    sos = sps.butter(int(order), cut, btype="highpass", fs=float(fs), output="sos")
                    out = sps.sosfiltfilt(sos, data, axis=0)
                else:
                    out = data.copy()
            else:
                lo = float(lowcut_hz if lowcut_hz is not None else 0.3)
                hi = float(highcut_hz if highcut_hz is not None else min(15.0, 0.45 * fs))
                if 0.0 < lo < hi < nyq:
                    sos = sps.butter(int(order), [lo, hi], btype="bandpass", fs=float(fs), output="sos")
                    out = sps.sosfiltfilt(sos, data, axis=0)
                else:
                    out = data.copy()
    elif m == "median_lowpass":
        med = apply_filter_1d_or_2d(data, fs, method="median", median_kernel=median_kernel)
        out = apply_filter_1d_or_2d(med, fs, method="lowpass", highcut_hz=highcut_hz, order=order)
    elif m == "wavelet":
        if pywt is None or n < 8:
            out = data.copy()
        else:
            w_obj = pywt.Wavelet(wavelet)
            max_lvl = pywt.dwt_max_level(n, w_obj.dec_len)
            lvl = max(1, min(int(wavelet_level or 2), max_lvl)) if max_lvl >= 1 else 0
            if lvl < 1:
                out = data.copy()
            else:
                out = np.empty_like(data)
                for c in range(data.shape[1]):
                    coeffs = pywt.wavedec(data[:, c], w_obj, level=lvl)
                    detail = coeffs[-1]
                    sigma = float(np.median(np.abs(detail))) / 0.6745 if len(detail) else 0.0
                    thresh = sigma * np.sqrt(2.0 * np.log(max(2, n)))
                    denoised = [coeffs[0]] + [pywt.threshold(d, thresh, mode="soft") for d in coeffs[1:]]
                    rec = pywt.waverec(denoised, w_obj)
                    out[:, c] = rec[:n]
    else:
        raise ValueError(
            f"Unsupported filter method {method!r}; choose 'none', 'lowpass', 'highpass', "
            "'bandpass', 'median', 'savgol', 'median_lowpass', or 'wavelet'"
        )
    return out[:, 0] if was_1d else out


def reject_tof_spikes(
    tof_rows: np.ndarray,
    *,
    spike_threshold: float = 80.0,
    kernel_size: int = 3,
) -> Tuple[np.ndarray, float]:
    """Reject isolated single-frame ToF distance spikes using ``scipy.ndimage.median_filter``.

    Parameters
    ----------
    tof_rows : np.ndarray, shape (T, P)
        ToF readings where invalid pixels are NaN.
    spike_threshold : float
        Maximum allowed deviation (mm) from local temporal median before a sample
        is replaced by NaN as a spike outlier.
    kernel_size : int
        Odd temporal window size for the median reference.

    Returns
    -------
    tuple of (cleaned_tof, spike_fraction)
        ``spike_fraction`` records the fraction of valid pixels flagged as spikes.
    """
    arr = np.asarray(tof_rows, dtype=np.float64).copy()
    valid = np.isfinite(arr)
    n_valid = int(valid.sum())
    if n_valid == 0 or len(arr) < 3 or spike_threshold <= 0:
        return arr, 0.0
    k = max(3, int(kernel_size))
    if k % 2 == 0:
        k += 1
    k = min(k, len(arr) if len(arr) % 2 == 1 else len(arr) - 1)
    # Fill NaN temporarily with per-column median so median_filter does not propagate NaNs
    col_med = np.nanmedian(arr, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0)
    filled = np.where(valid, arr, col_med[None, :])
    med_ref = sp_ndimage.median_filter(filled, size=(k, 1), mode="nearest")
    spikes = valid & (np.abs(arr - med_ref) > float(spike_threshold))
    arr[spikes] = np.nan
    spike_frac = float(spikes.sum()) / float(n_valid)
    return arr, spike_frac


class HandednessCanonicalizer(HoneycombBase):
    """Honeycomb cell mirroring left-handed sensor frames into right-handed coordinates."""

    def __init__(
        self,
        *,
        handedness_col: str = "handedness",
        demographics: Optional[pd.DataFrame] = None,
        subject_col: str = "subject",
        swap_sensors: bool = True,
        mount_flip_180: bool = False,
    ):
        self.handedness_col = handedness_col
        self.demographics = demographics
        self.subject_col = subject_col
        self.swap_sensors = swap_sensors
        self.mount_flip_180 = mount_flip_180

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return canonicalize_handedness_frame(
            X,
            handedness_col=self.handedness_col,
            demographics=self.demographics,
            subject_col=self.subject_col,
            swap_sensors=bool(self.swap_sensors),
            mount_flip_180=bool(self.mount_flip_180),
        )


class DenoisingFilterCell(HoneycombBase):
    """Honeycomb cell for zero-phase Butterworth, median, SavGol, wavelet, and ToF spike denoising."""

    def __init__(
        self,
        mode: str = "none",
        fs: float = 100.0,
        lowcut_hz: Optional[float] = None,
        highcut_hz: Optional[float] = None,
        order: int = 4,
        median_kernel: int = 5,
        savgol_window: int = 7,
        savgol_polyorder: int = 2,
        wavelet: str = "db4",
    ):
        self.mode = mode
        self.fs = fs
        self.lowcut_hz = lowcut_hz
        self.highcut_hz = highcut_hz
        self.order = order
        self.median_kernel = median_kernel
        self.savgol_window = savgol_window
        self.savgol_polyorder = savgol_polyorder
        self.wavelet = wavelet

    def transform(self, X: np.ndarray) -> np.ndarray:
        return apply_filter_1d_or_2d(
            np.asarray(X, dtype=np.float64),
            mode=self.mode,
            fs=float(self.fs),
            lowcut_hz=self.lowcut_hz,
            highcut_hz=self.highcut_hz,
            order=int(self.order),
            median_kernel=int(self.median_kernel),
            savgol_window=int(self.savgol_window),
            savgol_polyorder=int(self.savgol_polyorder),
            wavelet=self.wavelet,
        )

    @staticmethod
    def reject_tof_spikes(
        values: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
        *,
        kernel_size: int = 5,
        jump_threshold: float = 80.0,
    ) -> np.ndarray:
        return reject_tof_spikes(
            values,
            valid_mask=valid_mask,
            kernel_size=kernel_size,
            jump_threshold=jump_threshold,
        )



# ---------------------------------------------------------------------------
# Signal cleaning
# ---------------------------------------------------------------------------


class SignalCleaner(HoneycombBase):
    def __init__(
        self,
        native_sampling_rate: int = 20,
        compute_dt: bool = True,
        clip_value: Optional[float] = None,
        interp_mode: str = "linear",
        linear_acc_mode: Optional[str] = None,
        use_highpass_fallback: bool = True,
        window_size: int = 5,
        highpass_window_size: Optional[int] = None,
        rot_native_sampling_rate: Optional[float] = None,
        acceleration_units: str = "m/s^2",
        linear_acc_frame: str = "world",
        gravity_magnitude: float = 9.80665,
        sequence_col: str = "sequence_id",
        counter_col: str = "sequence_counter",
    ):
        self.native_sampling_rate = native_sampling_rate
        self.compute_dt = compute_dt
        self.clip_value = clip_value
        self.interp_mode = interp_mode
        self.linear_acc_mode = linear_acc_mode
        self.use_highpass_fallback = use_highpass_fallback
        self.window_size = window_size
        self.highpass_window_size = highpass_window_size
        self.rot_native_sampling_rate = rot_native_sampling_rate
        self.acceleration_units = acceleration_units
        self.linear_acc_frame = linear_acc_frame
        self.gravity_magnitude = gravity_magnitude
        self.sequence_col = sequence_col
        self.counter_col = counter_col

    def fit(self, X: pd.DataFrame, y=None):
        self.acc_cols_ = [c for c in X.columns if c.startswith("acc_")]
        self.rot_cols_ = [c for c in X.columns if c.startswith("rot_")]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        fs = _positive_rate(self.native_sampling_rate, "native_sampling_rate")
        rot_fs = _positive_rate(
            self.rot_native_sampling_rate
            if self.rot_native_sampling_rate is not None
            else self.native_sampling_rate,
            "rot_native_sampling_rate",
        )

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [
            c
            for c in num_cols
            if c not in {self.sequence_col, self.counter_col}
            and not c.startswith("tof_")
        ]

        counter = _validate_counter_order(df, self.sequence_col, self.counter_col)
        if num_cols:
            if self.interp_mode == "linear":
                if df[num_cols].isna().to_numpy().any():
                    df[num_cols] = df.groupby(self.sequence_col, sort=False)[num_cols].transform(
                        lambda g: g.interpolate(method="linear", limit_direction="both").ffill().bfill()
                    )
            elif self.interp_mode == "ffill":
                if df[num_cols].isna().to_numpy().any():
                    filled = df.groupby(self.sequence_col, sort=False)[num_cols].ffill()
                    df[num_cols] = filled.groupby(df[self.sequence_col], sort=False).bfill()
            elif self.interp_mode in {"cubic", "pchip"}:
                if df[num_cols].isna().to_numpy().any():
                    def _interp_smooth(series: pd.Series) -> pd.Series:
                        vals = series.to_numpy(dtype=float, copy=True)
                        ok = np.isfinite(vals)
                        if ok.all() or not ok.any():
                            return series
                        idx = np.arange(len(vals), dtype=float)
                        if ok.sum() < 4:
                            vals = np.interp(idx, idx[ok], vals[ok])
                        elif self.interp_mode == "pchip" and PchipInterpolator is not None:
                            interp_fn = PchipInterpolator(idx[ok], vals[ok], extrapolate=False)
                            res = interp_fn(idx)
                            missing = ~np.isfinite(res)
                            if missing.any():
                                res[missing] = np.interp(idx[missing], idx[ok], vals[ok])
                            vals = res
                        elif self.interp_mode == "cubic" and CubicSpline is not None:
                            interp_fn = CubicSpline(idx[ok], vals[ok], extrapolate=False)
                            res = interp_fn(idx)
                            missing = ~np.isfinite(res)
                            if missing.any():
                                res[missing] = np.interp(idx[missing], idx[ok], vals[ok])
                            vals = res
                        else:
                            vals = np.interp(idx, idx[ok], vals[ok])
                        return pd.Series(vals, index=series.index)

                    df[num_cols] = df.groupby(self.sequence_col, sort=False)[num_cols].transform(_interp_smooth)
            else:
                raise InvalidExtractorParams("interp_mode must be 'linear', 'ffill', 'cubic', or 'pchip'")

        if self.compute_dt:
            if counter is not None:
                counter_delta = counter.groupby(df[self.sequence_col], sort=False).diff()
                df["dt"] = counter_delta.fillna(1.0).div(fs)
                df["rot_dt"] = counter_delta.fillna(1.0).div(rot_fs)
            else:
                df["dt"] = 1.0 / fs
                df["rot_dt"] = 1.0 / rot_fs
        else:
            # compute_dt=False deliberately assumes a constant 1/fs interval.
            df["dt"] = 1.0 / fs
            df["rot_dt"] = 1.0 / rot_fs

        if self.clip_value is not None:
            acc_cols = [
                c
                for c in df.columns
                if c.startswith("acc_")
                and not c.endswith(("_vel", "_disp", "_jerk", "_mag", "_dr_vel", "_dr_pos"))
            ]
            if acc_cols:
                df[acc_cols] = df[acc_cols].clip(
                    lower=-float(self.clip_value),
                    upper=float(self.clip_value),
                )

        if self.linear_acc_mode == "baseline":
            df = self._add_linear_acceleration(df)
        elif self.linear_acc_mode not in {None, "none"}:
            raise InvalidExtractorParams(
                "linear_acc_mode must be None, 'none', or 'baseline'"
            )

        df["mask"] = 1.0
        return df

    def _add_linear_acceleration(self, df: pd.DataFrame) -> pd.DataFrame:
        acc_cols = ["acc_x", "acc_y", "acc_z"]
        rot_cols = ["rot_w", "rot_x", "rot_y", "rot_z"]

        if not all(c in df.columns for c in acc_cols + rot_cols):
            if self.use_highpass_fallback:
                available_acc = [c for c in acc_cols if c in df.columns]
                return self._linear_acc_highpass(df, available_acc)
            raise InvalidExtractorParams(
                "Orientation-based gravity removal requires acc_x/y/z and rot_w/x/y/z"
            )
        if self.acceleration_units not in {"m/s^2", "g"}:
            raise InvalidExtractorParams("acceleration_units must be 'm/s^2' or 'g'")
        if self.linear_acc_frame not in {"world", "body"}:
            raise InvalidExtractorParams("linear_acc_frame must be 'world' or 'body'")
        gravity_magnitude = float(self.gravity_magnitude)
        if not np.isfinite(gravity_magnitude) or gravity_magnitude <= 0:
            raise InvalidExtractorParams("gravity_magnitude must be finite and positive")

        acc = df[acc_cols].to_numpy(dtype=float)
        q = df[rot_cols].to_numpy(dtype=float)

        norms = np.linalg.norm(q, axis=1, keepdims=True)
        if not np.isfinite(acc).all() or not np.isfinite(q).all() or (norms <= 0).any():
            raise ValueError("Acceleration and quaternion samples must be finite; quaternions nonzero")

        lin_acc = remove_gravity_quaternion(
            acc,
            q,
            mode="quaternion",
            frame=self.linear_acc_frame,
            gravity_magnitude=gravity_magnitude,
            acceleration_units=self.acceleration_units,
            fs=float(self.native_sampling_rate),
            use_highpass_fallback=False,
        )

        for i, col in enumerate(["lin_acc_x", "lin_acc_y", "lin_acc_z"]):
            df[col] = lin_acc[:, i]

        return df

    def _linear_acc_highpass(self, df: pd.DataFrame, acc_cols: List[str]) -> pd.DataFrame:
        if not acc_cols:
            return df
        win = int(self.highpass_window_size if self.highpass_window_size is not None else self.window_size)
        for col in acc_cols:
            baseline = df.groupby(self.sequence_col, sort=False)[col].transform(
                lambda g: g.rolling(
                    window=win,
                    center=True,
                    min_periods=1,
                ).mean()
            )
            axis = col.removeprefix("acc_")
            df[f"lin_acc_{axis}_highpass_approx"] = df[col] - baseline

        return df


# ---------------------------------------------------------------------------
# Motion filter
# ---------------------------------------------------------------------------


class MotionFilter(HoneycombBase):
    def __init__(
        self,
        motion_filter_mode: Optional[str] = None,
        kalman_process_noise: float = 1e-3,
        kalman_measurement_noise: float = 1e-2,
        use_dead_reckoning: bool = False,
        dead_reckoning_detrend: bool = False,
        sequence_col: str = "sequence_id",
    ):
        self.motion_filter_mode = motion_filter_mode
        self.kalman_process_noise = kalman_process_noise
        self.kalman_measurement_noise = kalman_measurement_noise
        self.use_dead_reckoning = use_dead_reckoning
        self.dead_reckoning_detrend = dead_reckoning_detrend
        self.sequence_col = sequence_col

    def _kalman_filter_1d(self, signal: np.ndarray) -> np.ndarray:
        signal = np.asarray(signal, dtype=float)
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        if len(signal) == 0:
            return signal
        Q = float(self.kalman_process_noise)
        R = float(self.kalman_measurement_noise)
        x = float(signal[0])
        P = 1.0
        out = np.empty_like(signal)
        for i in range(len(signal)):
            P_pred = P + Q
            K = P_pred / (P_pred + R)
            x = x + K * (signal[i] - x)
            P = (1 - K) * P_pred
            out[i] = x
        return out

    def _detrend_series(self, s: pd.Series) -> pd.Series:
        s = s.astype(float)
        if len(s) <= 1:
            return s * 0.0
        return s - np.linspace(s.iloc[0], s.iloc[-1], len(s))

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X

        acc_cols = [
            c
            for c in df.columns
            if c.startswith("acc_")
            and not c.endswith(("_vel", "_disp", "_jerk", "_mag", "_dr_vel", "_dr_pos"))
        ]

        if acc_cols and (
            self.motion_filter_mode in {"kalman_1d", "kalman", "extended_kalman"}
            or (self.use_dead_reckoning and "dt" in df.columns)
        ):
            df = X.copy()

        if self.motion_filter_mode in {"kalman_1d", "kalman", "extended_kalman"} and acc_cols:
            if self.motion_filter_mode == "extended_kalman":
                warnings.warn(
                    "'extended_kalman' is a deprecated alias for 'kalman_1d' (scalar random-walk "
                    "Kalman filter; not an EKF).",
                    DeprecationWarning,
                    stacklevel=2,
                )
            for col in acc_cols:
                df[col] = df.groupby(self.sequence_col, sort=False)[col].transform(
                    lambda g: self._kalman_filter_1d(g.to_numpy())
                )

        if self.use_dead_reckoning and acc_cols and "dt" in df.columns:
            integration_dt = df["dt"].astype(float).mask(
                df.groupby(self.sequence_col, sort=False).cumcount().eq(0), 0.0
            )
            for col in acc_cols:
                vel_col = f"{col}_raw_dead_reckoning_velocity"
                pos_col = f"{col}_raw_dead_reckoning_position"
                vel_inc = df[col] * integration_dt
                df[vel_col] = vel_inc.groupby(df[self.sequence_col], sort=False).cumsum()
                if self.dead_reckoning_detrend:
                    df[vel_col] = df.groupby(self.sequence_col, sort=False)[vel_col].transform(
                        self._detrend_series
                    )
                pos_inc = df[vel_col] * integration_dt
                df[pos_col] = pos_inc.groupby(df[self.sequence_col], sort=False).cumsum()
                if self.dead_reckoning_detrend:
                    df[pos_col] = df.groupby(self.sequence_col, sort=False)[pos_col].transform(
                        self._detrend_series
                    )
        return df


# ---------------------------------------------------------------------------
# IMU
# ---------------------------------------------------------------------------


class IMUExtractor(HoneycombBase):
    """Build parallel accelerometer feature branches from the same source axes."""

    def __init__(
        self,
        acc_modes: str = "raw",
        linear_acc_modes: str = "",
        use_acc_magnitude: bool = False,
        use_linear_acc_magnitude: bool = False,
        window_size: int = 5,
        smoothing_window_size: Optional[int] = None,
        smooth_alpha: Optional[float] = None,
        integration_highpass_hz: Optional[float] = None,
        allow_raw_gravity_integration: bool = True,
        sequence_col: str = "sequence_id",
    ):
        self.acc_modes = acc_modes
        self.linear_acc_modes = linear_acc_modes
        self.use_acc_magnitude = use_acc_magnitude
        self.use_linear_acc_magnitude = use_linear_acc_magnitude
        self.window_size = window_size
        self.smoothing_window_size = smoothing_window_size
        self.smooth_alpha = smooth_alpha
        self.integration_highpass_hz = integration_highpass_hz
        self.allow_raw_gravity_integration = allow_raw_gravity_integration
        self.sequence_col = sequence_col

    def _prepare_for_integration(self, df: pd.DataFrame, cols: List[str], dt: pd.Series, *, is_raw: bool) -> pd.DataFrame:
        """Prepare acceleration axes for velocity/displacement integration.

        If ``integration_highpass_hz`` is set, applies a zero-phase Butterworth
        high-pass filter (``scipy.signal.butter`` + ``sosfiltfilt``) per sequence
        before integration to suppress DC / gravity quadratic drift.
        """
        if is_raw and self.integration_highpass_hz is None:
            if not self.allow_raw_gravity_integration:
                raise InvalidExtractorParams(
                    "Integrating raw acceleration includes gravity (~9.81 m/s^2) causing quadratic drift. "
                    "Use linear_acc_modes='velocity|displacement' with gravity removal or set integration_highpass_hz > 0."
                )
            warnings.warn(
                "IMUExtractor velocity/displacement on raw acceleration integrates gravity (~9.81 m/s^2). "
                "Prefer linear_acc_modes with gravity removal or set integration_highpass_hz > 0.",
                UserWarning,
                stacklevel=3,
            )
        source = df[cols].astype(float)
        if self.integration_highpass_hz is None:
            return source
        hp_hz = float(self.integration_highpass_hz)
        if not np.isfinite(hp_hz) or hp_hz <= 0:
            raise InvalidExtractorParams("integration_highpass_hz must be finite and positive")
        out = pd.DataFrame(index=df.index, columns=cols, dtype=float)
        for _, positions in df.groupby(self.sequence_col, sort=False).indices.items():
            pos = np.asarray(positions)
            seg = source.iloc[pos].to_numpy(dtype=float)
            dt_seg = dt.iloc[pos].to_numpy(dtype=float)
            valid_dt = dt_seg[np.isfinite(dt_seg) & (dt_seg > 0)]
            fs_est = 1.0 / float(np.median(valid_dt)) if len(valid_dt) else 20.0
            filtered = highpass_zero_phase(seg, fs=fs_est, cutoff_hz=hp_hz)
            out.iloc[pos] = filtered
        return out

    def fit(self, X: pd.DataFrame, y=None):
        self.acc_modes_ = self._parse_modes(self.acc_modes)
        self.linear_acc_modes_ = self._parse_modes(self.linear_acc_modes)
        derived_suffixes = (
            "_vel", "_disp", "_jerk", "_mag", "_dr_vel", "_dr_pos",
            "_raw_integrated_velocity", "_raw_integrated_displacement",
        )
        self.acc_cols_ = [
            c for c in ("acc_x", "acc_y", "acc_z")
            if c in X.columns and not c.endswith(derived_suffixes)
        ]
        self.lin_cols_ = [c for c in ("lin_acc_x", "lin_acc_y", "lin_acc_z") if c in X.columns]
        self.highpass_cols_ = [
            c for c in X.columns
            if c.startswith("lin_acc_") and c.endswith("_highpass_approx")
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["acc_modes_", "acc_cols_"])
        df = X
        parts = []
        dt = df.get("dt", pd.Series(1.0, index=df.index)).astype(float)
        first_sample = df.groupby(self.sequence_col, sort=False).cumcount().eq(0)
        integration_dt = dt.mask(first_sample, 0.0)
        smooth_win = int(self.smoothing_window_size if self.smoothing_window_size is not None else self.window_size)

        for mode in self.acc_modes_:
            if not self.acc_cols_:
                continue
            if mode == "raw":
                parts.append(df[self.acc_cols_].add_suffix("_raw"))
            elif mode == "smoothed":
                if self.smooth_alpha is not None:
                    out = df.groupby(self.sequence_col, sort=False)[self.acc_cols_].transform(
                        lambda g: g.ewm(alpha=float(self.smooth_alpha), adjust=False).mean()
                    )
                else:
                    out = df.groupby(self.sequence_col, sort=False)[self.acc_cols_].transform(
                        lambda g: g.rolling(window=smooth_win, center=True, min_periods=1).mean()
                    )
                parts.append(out.add_suffix("_smooth"))
            elif mode == "velocity":
                acc_int = self._prepare_for_integration(df, self.acc_cols_, dt, is_raw=True)
                vel = (
                    acc_int.mul(integration_dt, axis=0)
                    .groupby(df[self.sequence_col].values, sort=False).cumsum()
                )
                parts.append(vel.add_suffix("_raw_integrated_velocity"))
            elif mode == "displacement":
                acc_int = self._prepare_for_integration(df, self.acc_cols_, dt, is_raw=True)
                vel = (
                    acc_int.mul(integration_dt, axis=0)
                    .groupby(df[self.sequence_col].values, sort=False).cumsum()
                )
                disp = (
                    vel.mul(integration_dt, axis=0)
                    .groupby(df[self.sequence_col].values, sort=False).cumsum()
                )
                parts.append(disp.add_suffix("_raw_integrated_displacement"))
            elif mode == "jerk":
                jerk = (
                    df.groupby(self.sequence_col, sort=False)[self.acc_cols_]
                    .diff().div(dt, axis=0).fillna(0.0)
                )
                parts.append(jerk.add_suffix("_jerk"))

        if self.linear_acc_modes_ and self.lin_cols_:
            lin = df[self.lin_cols_].astype(float)
            for mode in self.linear_acc_modes_:
                if mode == "raw":
                    parts.append(lin.add_suffix("_linear"))
                elif mode == "velocity":
                    lin_int = self._prepare_for_integration(df, self.lin_cols_, dt, is_raw=False)
                    vel = (
                        lin_int.mul(integration_dt, axis=0)
                        .groupby(df[self.sequence_col].values, sort=False).cumsum()
                    )
                    parts.append(vel.add_suffix("_linear_integrated_velocity"))
                elif mode == "displacement":
                    lin_int = self._prepare_for_integration(df, self.lin_cols_, dt, is_raw=False)
                    vel = (
                        lin_int.mul(integration_dt, axis=0)
                        .groupby(df[self.sequence_col].values, sort=False).cumsum()
                    )
                    disp = (
                        vel.mul(integration_dt, axis=0)
                        .groupby(df[self.sequence_col].values, sort=False).cumsum()
                    )
                    parts.append(disp.add_suffix("_linear_integrated_displacement"))
                elif mode == "jerk":
                    jerk = (
                        df.groupby(self.sequence_col, sort=False)[self.lin_cols_]
                        .diff().div(dt, axis=0).fillna(0.0)
                    )
                    parts.append(jerk.add_suffix("_linear_jerk"))
                else:
                    raise InvalidExtractorParams(
                        f"Unsupported linear_acc_modes entry {mode!r}"
                    )
        elif self.linear_acc_modes_:
            raise InvalidExtractorParams(
                "linear_acc_modes require orientation-based corrected lin_acc_x/y/z axes"
            )

        if self.use_acc_magnitude and self.acc_cols_:
            mag = np.sqrt(df[self.acc_cols_].pow(2).sum(axis=1))
            parts.append(pd.DataFrame({"acc_mag": mag}, index=df.index))

        if self.use_linear_acc_magnitude and self.lin_cols_:
            mag = np.sqrt(df[self.lin_cols_].pow(2).sum(axis=1))
            parts.append(pd.DataFrame({"lin_acc_mag": mag}, index=df.index))
        elif self.use_linear_acc_magnitude and self.highpass_cols_:
            mag = np.sqrt(df[self.highpass_cols_].pow(2).sum(axis=1))
            parts.append(pd.DataFrame({"lin_acc_highpass_approx_mag": mag}, index=df.index))

        if not parts:
            return pd.DataFrame(index=df.index)
        return pd.concat(parts, axis=1)


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------


class RotationExtractor(HoneycombBase):
    def __init__(
        self,
        rotation_modes: str = "quaternion",
        fix_quaternion_sign: bool = True,
        sequence_col: str = "sequence_id",
    ):
        self.rotation_modes = rotation_modes
        self.fix_quaternion_sign = fix_quaternion_sign
        self.sequence_col = sequence_col

    def fit(self, X: pd.DataFrame, y=None):
        self.rot_modes_ = self._parse_modes(self.rotation_modes)
        expected = ["rot_w", "rot_x", "rot_y", "rot_z"]
        self.rot_cols_ = expected if all(c in X.columns for c in expected) else []
        if self.rot_modes_ and not self.rot_cols_:
            raise InvalidExtractorParams(
                "Rotation features require quaternion columns rot_w, rot_x, rot_y, rot_z"
            )
        if "angular_velocity" in self.rot_modes_ and Rotation is None:
            raise ImportError("RotationExtractor angular_velocity requires scipy.spatial.transform")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["rot_modes_", "rot_cols_"])
        df = X
        parts = []
        dt = df.get("rot_dt", df.get("dt", pd.Series(1.0, index=df.index))).astype(float)
        if not self.rot_modes_ or not self.rot_cols_:
            return pd.DataFrame(index=df.index)

        q = df[self.rot_cols_].to_numpy(dtype=float, copy=True)
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        if not np.isfinite(q).all() or (norms <= 0).any():
            raise ValueError(
                "Quaternion samples must be finite and nonzero in rot_w, rot_x, rot_y, rot_z"
            )
        q = q / norms

        if self.fix_quaternion_sign:
            for _, positions in df.groupby(self.sequence_col, sort=False).indices.items():
                positions = np.asarray(positions)
                for i in range(1, len(positions)):
                    if np.dot(q[positions[i - 1]], q[positions[i]]) < 0:
                        q[positions[i]] *= -1.0

        if len(q) == 0:
            return pd.DataFrame(index=df.index)

        for mode in self.rot_modes_:
            if mode == "quaternion":
                parts.append(
                    pd.DataFrame(
                        q,
                        columns=[c + "_quat" for c in self.rot_cols_],
                        index=df.index,
                    )
                )
            elif mode == "euler":
                eul = quaternions_to_euler(q, seq="xyz", unwrap=False)
                parts.append(
                    pd.DataFrame(
                        {"rot_roll": eul[:, 0], "rot_pitch": eul[:, 1], "rot_yaw": eul[:, 2]},
                        index=df.index,
                    )
                )
            elif mode == "delta_euler":
                eul = quaternions_to_euler(q, seq="xyz", unwrap=False)
                euler_df = pd.DataFrame(
                    {"rot_roll": eul[:, 0], "rot_pitch": eul[:, 1], "rot_yaw": eul[:, 2]},
                    index=df.index,
                )
                delta = (
                    euler_df.groupby(df[self.sequence_col].values, sort=False)
                    .diff()
                )
                delta = np.arctan2(np.sin(delta), np.cos(delta)).fillna(0.0)
                parts.append(delta.add_suffix("_delta"))
            elif mode == "angular_velocity":
                later_sample = ~df.groupby(self.sequence_col, sort=False).cumcount().eq(0).to_numpy()
                intervals = dt.to_numpy()[later_sample]
                if not np.isfinite(intervals).all() or np.any(intervals <= 0):
                    raise ValueError("Rotation intervals must be finite and positive")
                angular_velocity = np.zeros((len(q), 3), dtype=float)
                for _, positions in df.groupby(self.sequence_col, sort=False).indices.items():
                    positions = np.asarray(positions)
                    if len(positions) < 2:
                        continue
                    rotations = Rotation.from_quat(q[positions][:, [1, 2, 3, 0]])
                    delta = rotations[:-1].inv() * rotations[1:]
                    angular_velocity[positions[1:]] = delta.as_rotvec() / dt.iloc[positions[1:]].to_numpy(dtype=float)[:, None]
                names = [
                    "rot_angular_velocity_body_x_rad_s",
                    "rot_angular_velocity_body_y_rad_s",
                    "rot_angular_velocity_body_z_rad_s",
                ]
                parts.append(pd.DataFrame(angular_velocity, columns=names, index=df.index))
                parts.append(pd.DataFrame(
                    {"rot_angular_velocity_body_magnitude_rad_s":
                     np.linalg.norm(angular_velocity, axis=1)},
                    index=df.index,
                ))
            elif mode == "angular_distance":
                ang_dist = np.zeros(len(q), dtype=float)
                for _, positions in df.groupby(self.sequence_col, sort=False).indices.items():
                    positions = np.asarray(positions)
                    if len(positions) < 2:
                        continue
                    dots = np.clip(np.abs(np.sum(q[positions[1:]] * q[positions[:-1]], axis=1)), 0.0, 1.0)
                    ang_dist[positions[1:]] = 2.0 * np.arccos(dots)
                parts.append(pd.DataFrame({"rot_angular_distance_rad": ang_dist}, index=df.index))
            elif mode == "rot6d":
                rot6d = quaternions_to_rot6d(q)
                cols = [
                    "rot6d_c1_x", "rot6d_c1_y", "rot6d_c1_z",
                    "rot6d_c2_x", "rot6d_c2_y", "rot6d_c2_z",
                ]
                parts.append(pd.DataFrame(rot6d, columns=cols, index=df.index))

        if not parts:
            return pd.DataFrame(index=df.index)
        return pd.concat(parts, axis=1)


# ---------------------------------------------------------------------------
# ToF
# ---------------------------------------------------------------------------


class TOFExtractor(HoneycombBase):
    def __init__(
        self,
        tof_modes: str = "sensor_stats",
        tof_fill_mode: str = "far_255",
        invalid_sentinels: Tuple[float, ...] = DEFAULT_TOF_INVALID_SENTINELS,
        sequence_col: str = "sequence_id",
        n_sensors: int = 5,
    ):
        self.tof_modes = tof_modes
        self.tof_fill_mode = tof_fill_mode
        self.invalid_sentinels = invalid_sentinels
        self.sequence_col = sequence_col
        self.n_sensors = n_sensors

    def fit(self, X: pd.DataFrame, y=None):
        self.tof_modes_ = self._parse_modes(self.tof_modes)
        self.tof_cols_ = [c for c in X.columns if c.startswith("tof_")]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["tof_modes_", "tof_cols_"])
        if not self.tof_modes_ or not self.tof_cols_:
            return pd.DataFrame(index=X.index)

        observed = X[self.tof_cols_].astype(float).copy()
        for sentinel in (self.invalid_sentinels or ()):
            observed = observed.replace(float(sentinel), np.nan)
        observed = observed.mask(observed < 0.0, np.nan)
        valid = observed.notna()
        sensor_map: Dict[str, List[str]] = {}
        for col in observed.columns:
            pieces = str(col).split("_")
            key = pieces[1] if len(pieces) > 1 else "all"
            sensor_map.setdefault(key, []).append(col)

        def pixel_index(column: str) -> Tuple[int, str]:
            suffix = str(column).rsplit("_v", 1)
            try:
                return int(suffix[1]), str(column)
            except (IndexError, ValueError):
                return 0, str(column)

        for sensor_key in sensor_map:
            sensor_map[sensor_key] = sorted(sensor_map[sensor_key], key=pixel_index)

        n_keep = int(self.n_sensors) if self.n_sensors is not None else 0
        if n_keep > 0 and len(sensor_map) > n_keep:
            def _sensor_sort_key(item: str):
                try:
                    return (0, int(item))
                except (TypeError, ValueError):
                    return (1, str(item))
            keep_keys = sorted(sensor_map.keys(), key=_sensor_sort_key)[:n_keep]
            keep_set = set(keep_keys)
            sensor_map = {k: v for k, v in sensor_map.items() if k in keep_set}
            keep_cols = [c for cols in sensor_map.values() for c in cols]
            observed = observed[keep_cols]
            valid = valid[keep_cols]

        imputed = observed.copy()
        if self.tof_fill_mode == "nan_interpolate":
            imputed = imputed.groupby(X[self.sequence_col].values, sort=False).transform(
                lambda g: g.interpolate(method="linear", limit_direction="both").ffill().bfill()
            ).fillna(255.0)
        elif self.tof_fill_mode == "zero":
            imputed = imputed.fillna(0.0)
        elif self.tof_fill_mode == "far_255":
            imputed = imputed.fillna(255.0)
        elif self.tof_fill_mode == "far_500":
            imputed = imputed.fillna(500.0)
        else:
            raise InvalidExtractorParams(
                "tof_fill_mode must be 'nan_interpolate', 'zero', 'far_255', or 'far_500'"
            )

        parts = []

        for mode in self.tof_modes_:
            if mode == "raw":
                parts.append(imputed.add_suffix("_imputed"))
                parts.append(valid.astype(float).add_suffix("_valid"))
            elif mode == "sensor_stats":
                for sensor_key, cols in sensor_map.items():
                    if not cols:
                        continue
                    data = imputed[cols].to_numpy(dtype=float)
                    mask = valid[cols].to_numpy(dtype=bool)
                    count = mask.sum(axis=1)
                    observed_data = np.where(mask, data, 0.0)
                    valid_mean = np.divide(
                        observed_data.sum(axis=1), count,
                        out=np.zeros(len(X), dtype=float), where=count > 0,
                    )
                    valid_var = np.divide(
                        np.where(mask, (data - valid_mean[:, None]) ** 2, 0.0).sum(axis=1),
                        count,
                        out=np.zeros(len(X), dtype=float), where=count > 0,
                    )
                    valid_min = np.where(mask, data, np.inf).min(axis=1)
                    valid_max = np.where(mask, data, -np.inf).max(axis=1)
                    valid_min[count == 0] = 0.0
                    valid_max[count == 0] = 0.0
                    parts.append(pd.DataFrame({
                        f"tof_{sensor_key}_imputed_mean": data.mean(axis=1),
                        f"tof_{sensor_key}_imputed_std": data.std(axis=1),
                        f"tof_{sensor_key}_imputed_min": data.min(axis=1),
                        f"tof_{sensor_key}_imputed_max": data.max(axis=1),
                        f"tof_{sensor_key}_valid_mean": valid_mean,
                        f"tof_{sensor_key}_valid_std": np.sqrt(valid_var),
                        f"tof_{sensor_key}_valid_min": valid_min,
                        f"tof_{sensor_key}_valid_max": valid_max,
                        f"tof_{sensor_key}_valid_fraction": count / len(cols),
                    }, index=X.index))
            elif mode == "pooled_stats":
                data = imputed.to_numpy(dtype=float)
                mask = valid.to_numpy(dtype=bool)
                count = mask.sum(axis=1)
                valid_data = np.where(mask, data, 0.0)
                valid_mean = np.divide(
                    valid_data.sum(axis=1), count,
                    out=np.zeros(len(X), dtype=float), where=count > 0,
                )
                valid_var = np.divide(
                    np.where(mask, (data - valid_mean[:, None]) ** 2, 0.0).sum(axis=1),
                    count,
                    out=np.zeros(len(X), dtype=float), where=count > 0,
                )
                valid_min = np.where(mask, data, np.inf).min(axis=1)
                valid_max = np.where(mask, data, -np.inf).max(axis=1)
                valid_min[count == 0] = 0.0
                valid_max[count == 0] = 0.0
                parts.append(pd.DataFrame({
                    "tof_pooled_imputed_mean": data.mean(axis=1),
                    "tof_pooled_imputed_std": data.std(axis=1),
                    "tof_pooled_imputed_min": data.min(axis=1),
                    "tof_pooled_imputed_max": data.max(axis=1),
                    "tof_pooled_valid_mean": valid_mean,
                    "tof_pooled_valid_std": np.sqrt(valid_var),
                    "tof_pooled_valid_min": valid_min,
                    "tof_pooled_valid_max": valid_max,
                    "tof_pooled_valid_fraction": count / max(data.shape[1], 1),
                }, index=X.index))
            elif mode == "pooled":
                for sensor_key, cols in sensor_map.items():
                    if len(cols) != 64:
                        continue
                    arr = imputed[cols].to_numpy(dtype=float).reshape(-1, 8, 8)
                    mask = valid[cols].to_numpy(dtype=float).reshape(-1, 8, 8)
                    pool = (
                        arr.reshape(-1, 4, 2, 4, 2)
                        .mean(axis=(2, 4))
                        .reshape(-1, 16)
                    )
                    pool_validity = (
                        mask.reshape(-1, 4, 2, 4, 2)
                        .mean(axis=(2, 4))
                        .reshape(-1, 16)
                    )
                    parts.append(pd.DataFrame(
                        np.concatenate([pool, pool_validity], axis=1),
                        columns=(
                            [f"tof_{sensor_key}_pool_imputed_{i}" for i in range(16)]
                            + [f"tof_{sensor_key}_pool_valid_fraction_{i}" for i in range(16)]
                        ),
                        index=X.index,
                    ))
            else:
                raise InvalidExtractorParams(f"Unsupported tof_modes entry {mode!r}")

        if not parts:
            return pd.DataFrame(index=X.index)
        return pd.concat(parts, axis=1)


# ---------------------------------------------------------------------------
# Thermo
# ---------------------------------------------------------------------------


class ThermoExtractor(HoneycombBase):
    def __init__(
        self,
        thm_modes: str = "centered_diff",
        sequence_col: str = "sequence_id",
    ):
        self.thm_modes = thm_modes
        self.sequence_col = sequence_col

    def fit(self, X: pd.DataFrame, y=None):
        self.thm_modes_ = self._parse_modes(self.thm_modes)
        self.thm_cols_ = [c for c in X.columns if c.startswith("thm_")]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["thm_modes_", "thm_cols_"])
        if not self.thm_modes_ or not self.thm_cols_:
            return pd.DataFrame(index=X.index)

        raw = X[self.thm_cols_].astype(float)
        dt = X.get("dt", pd.Series(1.0, index=X.index)).astype(float)
        parts = []

        for mode in self.thm_modes_:
            if mode == "raw":
                parts.append(raw.add_suffix("_raw"))
            elif mode == "centered":
                means = raw.groupby(X[self.sequence_col].values, sort=False).transform("mean")
                parts.append((raw - means).add_suffix("_centered"))
            elif mode == "diff":
                diff = raw.groupby(X[self.sequence_col].values, sort=False).diff().fillna(0.0)
                parts.append(diff.add_suffix("_diff"))
            elif mode == "centered_diff":
                means = raw.groupby(X[self.sequence_col].values, sort=False).transform("mean")
                centered = raw - means
                diff = centered.groupby(X[self.sequence_col].values, sort=False).diff().fillna(0.0)
                parts.append(centered.add_suffix("_centered"))
                parts.append(diff.add_suffix("_centered_diff"))
            elif mode in {"diff_per_second", "centered_diff_per_second"}:
                source = raw
                if mode == "centered_diff_per_second":
                    means = raw.groupby(X[self.sequence_col].values, sort=False).transform("mean")
                    source = raw - means
                derivative = (
                    source.groupby(X[self.sequence_col].values, sort=False)
                    .diff().div(dt, axis=0).fillna(0.0)
                )
                parts.append(derivative.add_suffix("_temperature_rate_per_s"))
            else:
                raise InvalidExtractorParams(f"Unsupported thm_modes entry {mode!r}")

        if not parts:
            return pd.DataFrame(index=X.index)
        return pd.concat(parts, axis=1)


# ---------------------------------------------------------------------------
# STFT / CWT
# ---------------------------------------------------------------------------


def _sequence_summary_features(
    frame: pd.DataFrame,
    signal_columns: List[str],
    sequence_col: str,
    counter_col: str,
    feature_prefix: str,
    compute,
    validate_counter,
    *,
    pass_column_name: bool = False,
) -> pd.DataFrame:
    """Broadcast sequence summaries without copying sensor frames or inserting columns."""
    if not signal_columns:
        return pd.DataFrame(index=frame.index)
    positions_by_sequence = frame.groupby(sequence_col, sort=False).indices
    if sum(map(len, positions_by_sequence.values())) != len(frame):
        raise ValueError(f"{sequence_col} must not contain missing sequence IDs")
    if counter_col in frame.columns:
        counters = frame[[counter_col]]
        for positions in positions_by_sequence.values():
            validate_counter(counters.iloc[positions])
    features = {}
    for column in signal_columns:
        signal = frame[column].to_numpy(dtype=float, copy=False)
        for positions in positions_by_sequence.values():
            computed = compute(signal[positions], column) if pass_column_name else compute(signal[positions])
            for name, value in computed.items():
                feature_name = f"{column}_{feature_prefix}_{name}"
                if feature_name not in features:
                    features[feature_name] = np.empty(len(frame), dtype=float)
                features[feature_name][positions] = value
    return pd.DataFrame(features, index=frame.index)


class STFTExtractor(HoneycombBase):
    def __init__(
        self,
        acc_modes: str = "raw",
        nperseg: int = 32,
        noverlap: Optional[int] = None,
        window_type: str = "hann",
        scaling: str = "density",
        detrend: str = "constant",
        use_log_scale: bool = True,
        frequency_bands: Optional[Dict[str, Tuple[float, float]]] = None,
        sampling_rate: float = 20,
        column_sampling_rates: Optional[Dict[str, float]] = None,
        window_seconds: Optional[float] = None,
        counter_col: str = "sequence_counter",
        sequence_col: str = "sequence_id",
        feature_prefix: str = "stft0",
        signal_columns: Optional[List[str]] = None,
        include_derived_columns: bool = False,
        extended_spectral: bool = False,
        return_spectrogram: bool = False,
    ):
        self.acc_modes = acc_modes
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.window_type = window_type
        self.scaling = scaling
        self.detrend = detrend
        self.use_log_scale = use_log_scale
        bands = _coerce_search_value(frequency_bands)
        self.frequency_bands = bands or {
            "ultra_low": (0.0, 1.0),
            "low": (1.0, 3.0),
            "mid": (3.0, 6.0),
            "high": (6.0, 10.0),
        }
        self.sampling_rate = sampling_rate
        self.column_sampling_rates = column_sampling_rates
        self.window_seconds = window_seconds
        self.counter_col = counter_col
        self.sequence_col = sequence_col
        self.feature_prefix = feature_prefix
        self.signal_columns = signal_columns
        self.include_derived_columns = include_derived_columns
        self.extended_spectral = extended_spectral
        self.return_spectrogram = return_spectrogram

    def _column_fs(self, column: Optional[str] = None) -> float:
        rate_map = getattr(self, "column_sampling_rates_", None) or {}
        if column is not None and column in rate_map:
            return _positive_rate(rate_map[column], f"STFT sampling_rate[{column}]")
        if column is not None and rate_map:
            for prefix, rate in rate_map.items():
                if prefix.endswith("_") and str(column).startswith(prefix):
                    return _positive_rate(rate, f"STFT sampling_rate[{column}]")
        return _positive_rate(getattr(self, "sampling_rate_", self.sampling_rate), "STFT sampling_rate")

    def fit(self, X: pd.DataFrame, y=None):
        if self.sampling_rate in {None, "auto"} and "dt" in X.columns:
            dt_vals = pd.to_numeric(X["dt"], errors="coerce").to_numpy(dtype=float)
            dt_valid = dt_vals[np.isfinite(dt_vals) & (dt_vals > 0)]
            fs = _positive_rate(1.0 / float(np.median(dt_valid)) if len(dt_valid) else 20.0, "STFT sampling_rate")
        else:
            fs = _positive_rate(self.sampling_rate, "STFT sampling_rate")
        self.sampling_rate_ = fs
        raw_rate_map = _coerce_search_value(self.column_sampling_rates) or {}
        self.column_sampling_rates_ = {
            str(k): _positive_rate(v, f"STFT column_sampling_rates[{k}]")
            for k, v in raw_rate_map.items()
        }
        derived_suffixes = (
            "_vel", "_disp", "_jerk", "_mag", "_dr_vel", "_dr_pos",
            "_raw_integrated_velocity", "_raw_integrated_displacement",
            "_raw_dead_reckoning_velocity", "_raw_dead_reckoning_position",
        )
        if self.signal_columns is not None:
            self.acc_cols_ = [c for c in self.signal_columns if c in X.columns]
        elif self.include_derived_columns:
            self.acc_cols_ = [
                c for c in X.columns
                if c.startswith(("acc_", "lin_acc_", "rot_angular_velocity_", "thm_", "tof_"))
                and not c.endswith(("_valid", "_valid_fraction"))
            ]
        else:
            self.acc_cols_ = [
                c for c in X.columns
                if c.startswith("acc_") and not c.endswith(derived_suffixes)
            ]
        if self.window_seconds is not None:
            duration = float(self.window_seconds)
            if not np.isfinite(duration) or duration <= 0:
                raise InvalidExtractorParams("STFT window_seconds must be finite and positive")
            self.nperseg_ = max(1, int(round(duration * fs)))
        else:
            self.nperseg_ = _positive_int(self.nperseg, default=32, name="STFT nperseg")
        self.noverlap_ = (
            self.nperseg_ // 2 if self.noverlap is None else int(self.noverlap)
        )
        if not 0 <= self.noverlap_ < self.nperseg_:
            raise InvalidExtractorParams(
                f"STFT noverlap must be in [0, nperseg), got {self.noverlap_}"
            )
        scaling = str(self.scaling).strip().lower()
        aliases = {
            "density": "psd", "psd": "psd",
            "spectrum": "power_spectrum", "power_spectrum": "power_spectrum",
            "magnitude": "magnitude",
        }
        if scaling not in aliases:
            raise InvalidExtractorParams(
                "STFT scaling must be 'psd'/'density', 'power_spectrum'/'spectrum', or 'magnitude'"
            )
        self.representation_ = aliases[scaling]
        bands = _coerce_search_value(self.frequency_bands)
        if not isinstance(bands, dict):
            raise InvalidExtractorParams("STFT frequency_bands must be a mapping of names to ranges")
        self.frequency_bands_ = {}
        max_fs = max([fs, *(self._column_fs(c) for c in self.acc_cols_)]) if self.acc_cols_ else fs
        nyquist_max = max_fs / 2.0
        for name, bounds in bands.items():
            if len(bounds) != 2:
                raise InvalidExtractorParams(f"STFT band {name!r} must have (low, high) bounds")
            low, high = map(float, bounds)
            if not np.isfinite([low, high]).all() or low < 0 or high <= low or high > nyquist_max:
                raise InvalidExtractorParams(
                    f"STFT band {name!r} must satisfy 0 <= low < high <= Nyquist ({nyquist_max:g} Hz)"
                )
            self.frequency_bands_[str(name)] = (low, high)
        self.last_spectrograms_: Dict[str, List[Dict[str, np.ndarray]]] = {}
        return self

    def _compute_stft_features(self, signal: np.ndarray, column: Optional[str] = None) -> Dict[str, float]:
        if stft is None:
            raise ImportError("STFTExtractor requires SciPy.")
        signal = _signal_values(signal, "STFTExtractor")
        if signal is None:
            signal = np.zeros(0, dtype=float)
        observed_length = len(signal)
        fs = self._column_fs(column)
        if self.window_seconds is not None:
            nperseg = max(2, int(round(float(self.window_seconds) * fs)))
            noverlap = min(nperseg - 1, nperseg // 2 if self.noverlap is None else int(self.noverlap))
        else:
            nperseg = int(self.nperseg_)
            noverlap = int(self.noverlap_)
        if len(signal) < nperseg:
            signal = np.pad(signal, (0, nperseg - len(signal)), mode="constant")
        scipy_scaling = "psd" if self.representation_ == "psd" else "spectrum"
        f, t, Zxx = stft(
            signal, fs=fs, window=self.window_type,
            nperseg=nperseg, noverlap=noverlap, nfft=None, detrend=self.detrend,
            return_onesided=True, scaling=scipy_scaling, axis=-1,
            boundary="zeros", padded=True,
        )
        coverage = _stft_frame_coverage(t, observed_length, fs, nperseg, self.window_type)
        if self.representation_ == "magnitude":
            # Edge coverage correction for amplitude uses sqrt(energy_coverage)
            amp_cov = np.sqrt(np.maximum(coverage, 0.0))
            representation = np.divide(
                np.abs(Zxx),
                amp_cov[None, :],
                out=np.zeros_like(np.abs(Zxx)),
                where=amp_cov[None, :] > 0,
            )
        else:
            if self.representation_ == "psd":
                representation = _one_sided_psd(Zxx, nperseg)
            else:
                representation = _one_sided_psd(Zxx, nperseg)
            representation = np.divide(
                representation,
                coverage[None, :],
                out=np.zeros_like(representation),
                where=coverage[None, :] > 0,
            )

        if self.return_spectrogram and column is not None:
            self.last_spectrograms_.setdefault(column, []).append(
                {"freqs": f, "times": t, "matrix": representation.astype(np.float32)}
            )

        rep_name = {
            "psd": "psd",
            "power_spectrum": "power_spectrum",
            "magnitude": "magnitude",
        }[self.representation_]
        features: Dict[str, float] = {}
        features[f"stft_{rep_name}_mean"] = float(np.mean(representation))
        features[f"stft_{rep_name}_std"] = float(np.std(representation))
        features[f"stft_{rep_name}_max"] = float(np.max(representation))
        features[f"stft_{rep_name}_min"] = float(np.min(representation))
        features[f"stft_{rep_name}_median"] = float(np.median(representation))
        features[f"stft_{rep_name}_sum"] = float(np.sum(representation))
        max_idx = np.unravel_index(np.argmax(representation), representation.shape)
        features["stft_peak_frequency_hz"] = float(f[max_idx[0]])
        features["stft_peak_time_s"] = float(t[max_idx[1]]) if len(t) else 0.0

        if self.representation_ == "psd":
            dfreq = fs / nperseg
            mean_psd = np.mean(representation, axis=1)
            mean_power = float(np.sum(mean_psd) * dfreq)
            features["stft_mean_power"] = mean_power
            features["stft_power_std_across_frames"] = float(
                np.std(np.sum(representation, axis=0) * dfreq)
            )
            weights = mean_psd * dfreq
            weight_sum = float(weights.sum())
            centroid = float(np.sum(f * weights) / weight_sum) if weight_sum else 0.0
            spread = (
                float(np.sqrt(np.sum(((f - centroid) ** 2) * weights) / weight_sum))
                if weight_sum else 0.0
            )
            features["stft_spectral_centroid_hz"] = centroid
            features["stft_spectral_spread_hz"] = spread
            features["stft_spectral_entropy_bits"] = self._compute_entropy(weights)
            frame_power = np.sum(representation, axis=0) * dfreq
            features["stft_temporal_power_mean"] = float(np.mean(frame_power))
            features["stft_temporal_power_std"] = float(np.std(frame_power))
            features["stft_temporal_power_slope_per_s"] = (
                float(np.polyfit(t, frame_power, 1)[0]) if len(t) > 1 else 0.0
            )
            nyq_col = fs / 2.0
            for band_name, (f_low, f_high) in self.frequency_bands_.items():
                eff_hi = min(f_high, nyq_col)
                band_mask = (f >= f_low) & (f < eff_hi) if f_low < eff_hi else np.zeros(len(f), dtype=bool)
                band_power = (
                    float(np.mean(np.sum(representation[band_mask], axis=0) * dfreq))
                    if band_mask.any() else 0.0
                )
                features[f"stft_band_{band_name}_mean_power"] = band_power
                features[f"stft_band_{band_name}_power_fraction"] = (
                    band_power / mean_power if mean_power > 0 else 0.0
                )
                features[f"stft_band_{band_name}_resolved"] = float(band_mask.any())
            if self.extended_spectral:
                cdf = np.cumsum(weights) / max(weight_sum, np.finfo(float).tiny)
                r85_idx = int(np.searchsorted(cdf, 0.85))
                r95_idx = int(np.searchsorted(cdf, 0.95))
                features["stft_spectral_rolloff_85_hz"] = float(f[min(r85_idx, len(f) - 1)]) if weight_sum > 0 else 0.0
                features["stft_spectral_rolloff_95_hz"] = float(f[min(r95_idx, len(f) - 1)]) if weight_sum > 0 else 0.0
                pos_w = np.maximum(mean_psd, np.finfo(float).tiny)
                arith = float(np.mean(pos_w))
                geom = float(np.exp(np.mean(np.log(pos_w))))
                features["stft_spectral_flatness"] = geom / arith if arith > 0 else 0.0
                rms_sig = float(np.sqrt(np.mean(signal[:observed_length] ** 2))) if observed_length > 0 else 0.0
                peak_sig = float(np.max(np.abs(signal[:observed_length]))) if observed_length > 0 else 0.0
                features["stft_crest_factor"] = peak_sig / rms_sig if rms_sig > 1e-12 else 0.0
                if weight_sum > 0 and spread > 1e-12:
                    p_norm = weights / weight_sum
                    z = (f - centroid) / spread
                    features["stft_spectral_skewness"] = float(np.sum(p_norm * (z ** 3)))
                    features["stft_spectral_kurtosis"] = float(np.sum(p_norm * (z ** 4)) - 3.0)
                else:
                    features["stft_spectral_skewness"] = 0.0
                    features["stft_spectral_kurtosis"] = 0.0

        if self.use_log_scale:
            log_representation = np.log1p(representation)
            features[f"stft_log1p_{rep_name}_mean"] = float(np.mean(log_representation))
            features[f"stft_log1p_{rep_name}_std"] = float(np.std(log_representation))
            features[f"stft_log1p_{rep_name}_max"] = float(np.max(log_representation))

        frame_distribution = representation / np.maximum(
            representation.sum(axis=0, keepdims=True), np.finfo(float).tiny
        )
        normalized_flux = (
            np.abs(np.diff(frame_distribution, axis=1)).sum(axis=0)
            if representation.shape[1] > 1 else np.zeros(1)
        )
        features[f"stft_{rep_name}_normalized_flux_mean"] = float(np.mean(normalized_flux))
        features[f"stft_{rep_name}_normalized_flux_std"] = float(np.std(normalized_flux))
        features[f"stft_{rep_name}_normalized_flux_max"] = float(np.max(normalized_flux))
        return features

    def _compute_entropy(self, x: np.ndarray) -> float:
        weights = np.asarray(x, dtype=float)
        total = float(np.sum(weights))
        if total <= 0:
            return 0.0
        p = weights[weights > 0] / total
        return float(-np.sum(p * np.log2(p)))

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["acc_cols_"])
        self.last_spectrograms_ = {}
        return _sequence_summary_features(
            X, self.acc_cols_, self.sequence_col, self.counter_col,
            self.feature_prefix, self._compute_stft_features, self._validate_uniform_counter,
            pass_column_name=True,
        )

    def _validate_uniform_counter(self, group: pd.DataFrame) -> None:
        if self.counter_col not in group.columns or len(group) < 2:
            return
        values = pd.to_numeric(group[self.counter_col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any(np.diff(values) != 1):
            raise ValueError(
                "STFT requires uniformly spaced samples; sequence_counter must increase by "
                "one for every row. Interpolate missing samples on a uniform grid first."
            )


class CWTExtractor(HoneycombBase):
    def __init__(
        self,
        acc_modes: str = "raw",
        wavelet: str = "morl",
        widths: Optional[np.ndarray] = None,
        max_scale: int = 128,
        n_scales: int = 32,
        use_log_scale: bool = True,
        sampling_rate: float = 20,
        column_sampling_rates: Optional[Dict[str, float]] = None,
        min_scale: Optional[float] = None,
        max_frequency: Optional[float] = None,
        counter_col: str = "sequence_counter",
        sequence_col: str = "sequence_id",
        feature_prefix: str = "cwt0",
        signal_columns: Optional[List[str]] = None,
        include_derived_columns: bool = False,
        band_temporal_summaries: bool = False,
    ):
        self.acc_modes = acc_modes
        self.wavelet = wavelet
        self.widths = widths
        self.max_scale = max_scale
        self.n_scales = n_scales
        self.use_log_scale = use_log_scale
        self.sampling_rate = sampling_rate
        self.column_sampling_rates = column_sampling_rates
        self.min_scale = min_scale
        self.max_frequency = max_frequency
        self.counter_col = counter_col
        self.sequence_col = sequence_col
        self.feature_prefix = feature_prefix
        self.signal_columns = signal_columns
        self.include_derived_columns = include_derived_columns
        self.band_temporal_summaries = band_temporal_summaries

    def _column_fs(self, column: Optional[str] = None) -> float:
        rate_map = getattr(self, "column_sampling_rates_", None) or {}
        if column is not None and column in rate_map:
            return _positive_rate(rate_map[column], f"CWT sampling_rate[{column}]")
        if column is not None and rate_map:
            for prefix, rate in rate_map.items():
                if prefix.endswith("_") and str(column).startswith(prefix):
                    return _positive_rate(rate, f"CWT sampling_rate[{column}]")
        return _positive_rate(getattr(self, "sampling_rate_", self.sampling_rate), "CWT sampling_rate")

    def fit(self, X: pd.DataFrame, y=None):
        if pywt is None:
            raise ImportError("CWTExtractor requires PyWavelets.")
        if self.sampling_rate in {None, "auto"} and "dt" in X.columns:
            dt_vals = pd.to_numeric(X["dt"], errors="coerce").to_numpy(dtype=float)
            dt_valid = dt_vals[np.isfinite(dt_vals) & (dt_vals > 0)]
            fs = _positive_rate(1.0 / float(np.median(dt_valid)) if len(dt_valid) else 20.0, "CWT sampling_rate")
        else:
            fs = _positive_rate(self.sampling_rate, "CWT sampling_rate")
        self.sampling_rate_ = fs
        raw_rate_map = _coerce_search_value(self.column_sampling_rates) or {}
        self.column_sampling_rates_ = {
            str(k): _positive_rate(v, f"CWT column_sampling_rates[{k}]")
            for k, v in raw_rate_map.items()
        }
        derived_suffixes = (
            "_vel", "_disp", "_jerk", "_mag", "_dr_vel", "_dr_pos",
            "_raw_integrated_velocity", "_raw_integrated_displacement",
            "_raw_dead_reckoning_velocity", "_raw_dead_reckoning_position",
        )
        if self.signal_columns is not None:
            self.acc_cols_ = [c for c in self.signal_columns if c in X.columns]
        elif self.include_derived_columns:
            self.acc_cols_ = [
                c for c in X.columns
                if c.startswith(("acc_", "lin_acc_", "rot_angular_velocity_", "thm_", "tof_"))
                and not c.endswith(("_valid", "_valid_fraction"))
            ]
        else:
            self.acc_cols_ = [
                c for c in X.columns
                if c.startswith("acc_") and not c.endswith(derived_suffixes)
            ]
        try:
            wavelet = pywt.ContinuousWavelet(self.wavelet)
            central_frequency = float(pywt.central_frequency(wavelet))
        except (TypeError, ValueError) as exc:
            raise InvalidExtractorParams(
                f"CWT wavelet {self.wavelet!r} is not a valid continuous wavelet"
            ) from exc
        if not np.isfinite(central_frequency) or central_frequency <= 0:
            raise InvalidExtractorParams(f"CWT wavelet {self.wavelet!r} has invalid central frequency")
        max_frequency = (
            min(0.4 * fs, fs / 2.0)
            if self.max_frequency is None
            else float(self.max_frequency)
        )
        if not np.isfinite(max_frequency) or not 0 < max_frequency <= fs / 2.0:
            raise InvalidExtractorParams(
                f"CWT max_frequency must be in (0, Nyquist={fs / 2.0:g}] Hz"
            )
        max_scale = float(self.max_scale)
        if not np.isfinite(max_scale) or max_scale <= 0:
            raise InvalidExtractorParams("CWT max_scale must be finite and positive")
        if self.widths is None:
            safe_minimum = central_frequency * fs / max_frequency
            minimum = safe_minimum if self.min_scale is None else float(self.min_scale)
            if not np.isfinite(minimum) or minimum <= 0:
                raise InvalidExtractorParams("CWT min_scale must be finite and positive")
            if max_scale < minimum:
                raise InvalidExtractorParams(
                    f"CWT max_scale ({max_scale:g}) is below the safe minimum scale "
                    f"({minimum:g}) for {max_frequency:g} Hz"
                )
            n_scales = _positive_int(self.n_scales, name="CWT n_scales")
            self.widths_ = np.geomspace(minimum, max_scale, n_scales)
        else:
            self.widths_ = np.asarray(self.widths, dtype=float)
        if (
            self.widths_.ndim != 1
            or len(self.widths_) == 0
            or not np.isfinite(self.widths_).all()
            or np.any(self.widths_ <= 0)
            or np.any(np.diff(self.widths_) <= 0)
        ):
            raise InvalidExtractorParams("CWT scales must be finite, positive, and strictly increasing")
        self.scale_frequencies_hz_ = (
            pywt.scale2frequency(wavelet, self.widths_) * fs
        )
        if np.any(self.scale_frequencies_hz_ > fs / 2.0):
            raise InvalidExtractorParams(
                "CWT scales include frequencies above Nyquist; increase the minimum scale"
            )
        self._wavelet_lower_bound = float(wavelet.lower_bound)
        self._wavelet_upper_bound = float(wavelet.upper_bound)
        return self

    def _compute_cwt_features(self, signal: np.ndarray, column: Optional[str] = None) -> Dict[str, float]:
        if pywt is None:
            raise ImportError("CWTExtractor requires PyWavelets.")
        signal = _signal_values(signal, "CWTExtractor")
        if signal is None:
            return self._empty_features()
        fs = self._column_fs(column)
        coefficients, frequencies = pywt.cwt(
            signal,
            self.widths_,
            self.wavelet,
            sampling_period=1.0 / fs,
        )
        magnitude = np.abs(coefficients)
        valid = np.ones(magnitude.shape, dtype=bool)
        for idx, scale in enumerate(self.widths_):
            edge = int(np.ceil(
                max(abs(self._wavelet_lower_bound), abs(self._wavelet_upper_bound))
                * float(scale) / 2.0
            ))
            if edge:
                valid[idx, :min(edge, len(signal))] = False
                valid[idx, max(0, len(signal) - edge):] = False
        valid_counts = valid.sum(axis=1)
        scale_mean = np.divide(
            np.where(valid, magnitude, 0.0).sum(axis=1),
            valid_counts,
            out=np.zeros(len(self.widths_), dtype=float),
            where=valid_counts > 0,
        )
        valid_values = magnitude[valid]
        if valid_values.size == 0:
            valid_values = np.zeros(1, dtype=float)
        time_counts = valid.sum(axis=0)
        time_mean = np.divide(
            np.where(valid, magnitude, 0.0).sum(axis=0),
            time_counts,
            out=np.zeros(len(signal), dtype=float),
            where=time_counts > 0,
        )
        features: Dict[str, float] = {
            "cwt_mean_magnitude": float(np.mean(valid_values)),
            "cwt_std_magnitude": float(np.std(valid_values)),
            "cwt_max_magnitude": float(np.max(valid_values)),
            "cwt_min_magnitude": float(np.min(valid_values)),
            "cwt_median_magnitude": float(np.median(valid_values)),
            "cwt_sum_magnitude": float(np.sum(valid_values)),
            "cwt_coefficient_power_mean": float(np.mean(valid_values ** 2)),
            "cwt_scale_mean_max_magnitude": float(np.max(scale_mean)),
            "cwt_scale_mean_min_magnitude": float(np.min(scale_mean)),
            "cwt_scale_mean_std_magnitude": float(np.std(scale_mean)),
            "cwt_dominant_scale": float(self.widths_[int(np.argmax(scale_mean))]),
            "cwt_dominant_frequency_hz": float(frequencies[int(np.argmax(scale_mean))]),
            "cwt_dominant_scale_mean_magnitude": float(np.max(scale_mean)),
            "cwt_scale_entropy_bits": self._compute_entropy(scale_mean),
            "cwt_temporal_mean_magnitude": float(np.mean(time_mean)),
            "cwt_temporal_std_magnitude": float(np.std(time_mean)),
            "cwt_temporal_entropy_bits": self._compute_entropy(time_mean),
            "cwt_temporal_slope_per_s": (
                float(np.polyfit(np.arange(len(time_mean)) / fs, time_mean, 1)[0])
                if len(time_mean) > 1 else 0.0
            ),
            "cwt_boundary_fraction": float(1.0 - valid.mean()),
        }
        n_thirds = len(self.widths_) // 3
        scale_total = float(scale_mean.sum())
        if n_thirds > 0:
            features["cwt_low_scale_magnitude_fraction"] = float(
                scale_mean[:n_thirds].sum() / scale_total if scale_total > 0 else 0.0
            )
            features["cwt_mid_scale_magnitude_fraction"] = float(
                scale_mean[n_thirds:2*n_thirds].sum() / scale_total if scale_total > 0 else 0.0
            )
            features["cwt_high_scale_magnitude_fraction"] = float(
                scale_mean[2*n_thirds:].sum() / scale_total if scale_total > 0 else 0.0
            )
        else:
            features.update({
                "cwt_low_scale_magnitude_fraction": 0.0,
                "cwt_mid_scale_magnitude_fraction": 0.0,
                "cwt_high_scale_magnitude_fraction": 0.0,
            })
        if self.band_temporal_summaries:
            slices = (
                ("low", slice(0, max(1, n_thirds))),
                ("mid", slice(n_thirds, max(n_thirds + 1, 2 * n_thirds))),
                ("high", slice(2 * n_thirds, len(self.widths_))),
            )
            t_sec = np.arange(len(signal), dtype=float) / fs
            for b_label, sl in slices:
                sub_m = magnitude[sl]
                sub_v = valid[sl]
                cnt = sub_v.sum(axis=0)
                series = np.divide(
                    np.where(sub_v, sub_m, 0.0).sum(axis=0),
                    cnt,
                    out=np.zeros(len(signal), dtype=float),
                    where=cnt > 0,
                )
                features[f"cwt_{b_label}_band_temporal_mean"] = float(np.mean(series))
                features[f"cwt_{b_label}_band_temporal_std"] = float(np.std(series))
                features[f"cwt_{b_label}_band_temporal_max"] = float(np.max(series))
                features[f"cwt_{b_label}_band_temporal_slope_per_s"] = (
                    float(np.polyfit(t_sec, series, 1)[0]) if len(series) > 1 else 0.0
                )
        ridge_energy = []
        for t_idx in range(magnitude.shape[1]):
            col = np.where(valid[:, t_idx], magnitude[:, t_idx], 0.0)
            if len(col) > 2:
                maxima_idx = argrelextrema(col, np.greater)[0] if argrelextrema is not None else []
                if len(maxima_idx) > 0:
                    ridge_energy.extend(col[maxima_idx])
        if ridge_energy:
            features['cwt_ridge_mean'] = np.mean(ridge_energy)
            features['cwt_ridge_std'] = np.std(ridge_energy)
            features['cwt_ridge_count_per_sample'] = len(ridge_energy) / max(len(time_mean), 1)
        else:
            features['cwt_ridge_mean'] = 0.0
            features['cwt_ridge_std'] = 0.0
            features['cwt_ridge_count_per_sample'] = 0.0
        features['cwt_variance_to_mean_scale_magnitude'] = (
            float(np.var(scale_mean) / np.mean(scale_mean))
            if np.mean(scale_mean) > 0 else 0.0
        )
        features["cwt_valid_coefficient_fraction"] = float(valid.mean())
        if self.use_log_scale:
            log_magnitude = np.log1p(valid_values)
            features["cwt_log1p_magnitude_mean"] = float(np.mean(log_magnitude))
            features["cwt_log1p_magnitude_std"] = float(np.std(log_magnitude))
            features["cwt_log1p_magnitude_max"] = float(np.max(log_magnitude))
        return features

    def _empty_features(self) -> Dict[str, float]:
        keys = (
            "cwt_mean_magnitude", "cwt_std_magnitude", "cwt_max_magnitude",
            "cwt_min_magnitude", "cwt_median_magnitude", "cwt_sum_magnitude",
            "cwt_coefficient_power_mean", "cwt_scale_mean_max_magnitude",
            "cwt_scale_mean_min_magnitude", "cwt_scale_mean_std_magnitude",
            "cwt_dominant_scale", "cwt_dominant_frequency_hz",
            "cwt_dominant_scale_mean_magnitude", "cwt_scale_entropy_bits",
            "cwt_temporal_mean_magnitude", "cwt_temporal_std_magnitude",
            "cwt_temporal_entropy_bits", "cwt_temporal_slope_per_s",
            "cwt_low_scale_magnitude_fraction", "cwt_mid_scale_magnitude_fraction",
            "cwt_high_scale_magnitude_fraction", "cwt_ridge_mean", "cwt_ridge_std",
            "cwt_ridge_count_per_sample", "cwt_variance_to_mean_scale_magnitude",
            "cwt_boundary_fraction", "cwt_valid_coefficient_fraction",
        )
        result = dict.fromkeys(keys, 0.0)
        result["cwt_boundary_fraction"] = 1.0
        if self.band_temporal_summaries:
            for b_label in ("low", "mid", "high"):
                for stat in ("mean", "std", "max", "slope_per_s"):
                    result[f"cwt_{b_label}_band_temporal_{stat}"] = 0.0
        if self.use_log_scale:
            result.update({
                "cwt_log1p_magnitude_mean": 0.0,
                "cwt_log1p_magnitude_std": 0.0,
                "cwt_log1p_magnitude_max": 0.0,
            })
        return result

    def _compute_entropy(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        total = float(np.sum(x))
        if total == 0:
            return 0.0
        p = x[x > 0] / total
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["acc_cols_", "widths_"])
        return _sequence_summary_features(
            X, self.acc_cols_, self.sequence_col, self.counter_col,
            self.feature_prefix, self._compute_cwt_features, self._validate_uniform_counter,
            pass_column_name=True,
        )

    def _validate_uniform_counter(self, group: pd.DataFrame) -> None:
        if self.counter_col not in group.columns or len(group) < 2:
            return
        values = pd.to_numeric(group[self.counter_col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any(np.diff(values) != 1):
            raise ValueError(
                "CWT requires uniformly spaced samples; sequence_counter must increase by "
                "one for every row. Interpolate missing samples on a uniform grid first."
            )


# ---------------------------------------------------------------------------
# Sequence extractor
# ---------------------------------------------------------------------------

# ===========================================================================
# 4. Modular Signal Descriptors (Kinematics, Walch DoG, Baevsky, ACF, Hann PSD)
# ===========================================================================

@dataclass(frozen=True)
class IMUWindow:
    """Validated per-window sensor arrays for dynamic, spectral, and quality descriptors.

    Attributes
    ----------
    raw : np.ndarray, shape (T, 3)
        3-axis raw accelerometer readings in ``m/s^2``.
    observed : np.ndarray, shape (T, 3) of bool
        Per-axis observation mask (True = genuinely observed, False = missing/imputed/padded).
    linear : np.ndarray, shape (T, 3)
        Gravity-removed 3-axis linear acceleration in ``m/s^2``.
    rotation : np.ndarray, shape (T, 3)
        Body-frame angular velocity in ``rad/s``.
    rotation_observed : np.ndarray, shape (T,) of bool
        Validity mask for quaternion / angular velocity samples.
    counter : np.ndarray, shape (T,)
        Monotonic sample counter indices for gap detection.
    dt : np.ndarray, shape (T,)
        Per-sample interval in seconds (``1 / fs`` on uniform grids).
    row_present : np.ndarray, shape (T,) of bool
        Whether each counter slot was originally present in the recording.
    """

    raw: np.ndarray
    observed: np.ndarray
    linear: np.ndarray
    rotation: np.ndarray
    rotation_observed: np.ndarray
    counter: np.ndarray
    dt: np.ndarray
    row_present: np.ndarray

    @classmethod
    def from_mapping(cls, src: Mapping[str, np.ndarray]) -> "IMUWindow":
        counter = np.asarray(src["counter"], dtype=np.float64)
        n = len(counter)
        return cls(
            raw=np.asarray(src["raw"], dtype=np.float64),
            observed=np.asarray(src["observed"], dtype=bool),
            linear=np.asarray(src.get("linear", np.full((n, 3), np.nan)), dtype=np.float64),
            rotation=np.asarray(src.get("rotation", np.full((n, 3), np.nan)), dtype=np.float64),
            rotation_observed=np.asarray(src.get("rotation_observed", np.zeros(n, dtype=bool)), dtype=bool),
            counter=counter,
            dt=np.asarray(src["dt"], dtype=np.float64),
            row_present=np.asarray(src.get("row_present", np.ones(n, dtype=bool)), dtype=bool),
        )


@dataclass(frozen=True)
class DescriptorCatalogEntry:
    """Metadata entry describing one signal descriptor calculation."""

    base_key: str
    canonical_name: str
    legacy_name: str
    family: str
    kind: str  # 'dynamic', 'spectral', 'periodicity', 'quality', 'branch_drift'
    formula: str
    units: str
    required_inputs: Tuple[str, ...]
    min_samples_note: str
    provenance: str
    paper_or_ref: str


DESCRIPTOR_CATALOG: Tuple[DescriptorCatalogEntry, ...] = (
    # --- Dynamic IMU kinematics ---
    DescriptorCatalogEntry(
        "acc_ac_rms", "dyn_acc_magnitude_ac_rms", "noop_acc_ac_rms",
        "dynamic_kinematics", "dynamic",
        r"\mathrm{std}(\|\mathbf{a}\|_2)", "m/s^2", ("acc_x", "acc_y", "acc_z"),
        ">= min_samples observed 3-axis rows",
        "Dynamic AC component RMS of raw 3-axis acceleration magnitude (ImuFeatureExtractor concept)",
        "Internal port (cr7_kimmitch ImuFeatureExtractor)",
    ),
    DescriptorCatalogEntry(
        "acc_x_rms", "dyn_acc_x_rms", "noop_acc_x_rms",
        "dynamic_kinematics", "dynamic",
        r"\sqrt{\frac{1}{N}\sum a_x^2}", "m/s^2", ("acc_x",),
        ">= min_samples observed rows",
        "Per-axis total RMS acceleration (retains DC/posture projection)",
        "Internal port (cr7_kimmitch ImuFeatureExtractor)",
    ),
    DescriptorCatalogEntry(
        "acc_y_rms", "dyn_acc_y_rms", "noop_acc_y_rms",
        "dynamic_kinematics", "dynamic",
        r"\sqrt{\frac{1}{N}\sum a_y^2}", "m/s^2", ("acc_y",),
        ">= min_samples observed rows",
        "Per-axis total RMS acceleration (retains DC/posture projection)",
        "Internal port (cr7_kimmitch ImuFeatureExtractor)",
    ),
    DescriptorCatalogEntry(
        "acc_z_rms", "dyn_acc_z_rms", "noop_acc_z_rms",
        "dynamic_kinematics", "dynamic",
        r"\sqrt{\frac{1}{N}\sum a_z^2}", "m/s^2", ("acc_z",),
        ">= min_samples observed rows",
        "Per-axis total RMS acceleration (retains DC/posture projection)",
        "Internal port (cr7_kimmitch ImuFeatureExtractor)",
    ),
    DescriptorCatalogEntry(
        "jerk_rms", "dyn_acc_jerk_rms", "noop_jerk_rms",
        "dynamic_kinematics", "dynamic",
        r"\sqrt{\frac{1}{M}\sum \|\Delta \mathbf{a} / \Delta t\|_2^2}", "m/s^3", ("acc_x", "acc_y", "acc_z", "dt"),
        ">= min_samples - 1 contiguous pairs",
        "3D vector jerk RMS over consecutive observed samples",
        "Internal port (cr7_kimmitch ImuFeatureExtractor)",
    ),
    DescriptorCatalogEntry(
        "rotation_mean_rad_s", "dyn_angular_speed_mean_rad_s", "noop_rotation_mean_rad_s",
        "dynamic_kinematics", "dynamic",
        r"\frac{1}{M}\sum \|\boldsymbol{\omega}\|_2", "rad/s", ("rot_w", "rot_x", "rot_y", "rot_z"),
        ">= min_samples valid consecutive quaternion pairs",
        "Mean body-frame angular speed derived via SciPy Rotation increments",
        "SciPy Rotation (https://docs.scipy.org/doc/scipy/reference/spatial.transform.html)",
    ),
    DescriptorCatalogEntry(
        "rotation_rms_rad_s", "dyn_angular_speed_rms_rad_s", "noop_rotation_rms_rad_s",
        "dynamic_kinematics", "dynamic",
        r"\sqrt{\frac{1}{M}\sum \|\boldsymbol{\omega}\|_2^2}", "rad/s", ("rot_w", "rot_x", "rot_y", "rot_z"),
        ">= min_samples valid consecutive quaternion pairs",
        "RMS body-frame angular speed",
        "SciPy Rotation (https://docs.scipy.org/doc/scipy/reference/spatial.transform.html)",
    ),
    DescriptorCatalogEntry(
        "gyro_energy_dps", "dyn_angular_speed_mean_deg_s", "noop_gyro_energy_dps",
        "dynamic_kinematics", "dynamic",
        r"\frac{180}{\pi}\frac{1}{M}\sum \|\boldsymbol{\omega}\|_2", "deg/s", ("rot_w", "rot_x", "rot_y", "rot_z"),
        ">= min_samples valid consecutive quaternion pairs",
        "Mean rotational speed in deg/s",
        "Internal port (cr7_kimmitch ImuFeatureExtractor)",
    ),
    DescriptorCatalogEntry(
        "posture_var", "dyn_posture_covariance_trace", "noop_posture_var",
        "dynamic_kinematics", "dynamic",
        r"\mathrm{tr}(\mathrm{Cov}(\mathbf{a})) = \sum_{c\in\{x,y,z\}}\mathrm{Var}(a_c)", "(m/s^2)^2", ("acc_x", "acc_y", "acc_z"),
        ">= min_samples observed rows",
        "Total 3D acceleration covariance trace (orientation/posture spread)",
        "Internal port (cr7_kimmitch WakeMotionRefinement)",
    ),
    DescriptorCatalogEntry(
        "motion_intensity_mean", "dyn_vector_delta_mean", "noop_motion_intensity_mean",
        "dynamic_kinematics", "dynamic",
        r"\frac{1}{M}\sum \|\Delta \mathbf{a}\|_2", "m/s^2", ("acc_x", "acc_y", "acc_z"),
        ">= max(2, min_samples - 1) contiguous pairs",
        "Mean 3D step-delta magnitude",
        "Internal port (cr7_kimmitch WorkoutDetector)",
    ),
    DescriptorCatalogEntry(
        "motion_variance", "dyn_vector_delta_variance", "noop_motion_variance",
        "dynamic_kinematics", "dynamic",
        r"\mathrm{Var}(\|\Delta \mathbf{a}\|_2)", "(m/s^2)^2", ("acc_x", "acc_y", "acc_z"),
        ">= max(2, min_samples - 1) contiguous pairs",
        "Variance of 3D step-delta magnitude",
        "Internal port (cr7_kimmitch WorkoutDetector)",
    ),
    DescriptorCatalogEntry(
        "motion_cv", "dyn_vector_delta_cv", "noop_motion_cv",
        "dynamic_kinematics", "dynamic",
        r"\mathrm{std}(\|\Delta \mathbf{a}\|_2) / \mathrm{mean}(\|\Delta \mathbf{a}\|_2)", "dimensionless", ("acc_x", "acc_y", "acc_z"),
        ">= max(2, min_samples - 1) contiguous pairs",
        "Burstiness coefficient of variation of 3D step-delta magnitude",
        "Internal port (cr7_kimmitch WorkoutTypeClassifier)",
    ),
    DescriptorCatalogEntry(
        "jerk_floor", "dyn_quiescent_jerk_floor", "noop_jerk_floor",
        "dynamic_kinematics", "dynamic",
        r"\mathrm{median}(\|\mathbf{j}\|_2)", "m/s^3", ("acc_x", "acc_y", "acc_z", "dt"),
        ">= max(2, min_samples - 1) contiguous pairs",
        "Self-calibrated quiescent median jerk floor",
        "Internal port (cr7_kimmitch SleepStagerV2)",
    ),
    DescriptorCatalogEntry(
        "move_fraction", "dyn_jerk_active_fraction", "noop_move_fraction",
        "dynamic_kinematics", "dynamic",
        r"\mathrm{mean}(\|\mathbf{j}\|_2 > 2 \cdot j_{\mathrm{floor}})", "fraction [0, 1]", ("acc_x", "acc_y", "acc_z", "dt"),
        ">= max(2, min_samples - 1) contiguous pairs",
        "Duty cycle of samples exceeding 2x quiescent jerk floor",
        "Internal port (cr7_kimmitch SleepStagerV2)",
    ),
    DescriptorCatalogEntry(
        "still_fraction", "dyn_jerk_still_fraction", "noop_still_fraction",
        "dynamic_kinematics", "dynamic",
        r"\mathrm{mean}(\|\mathbf{j}\|_2 \le 1.25 \cdot j_{\mathrm{floor}})", "fraction [0, 1]", ("acc_x", "acc_y", "acc_z", "dt"),
        ">= max(2, min_samples - 1) contiguous pairs",
        "Duty cycle of quiescent samples within 1.25x jerk floor",
        "Internal port (cr7_kimmitch SleepStagerV2)",
    ),
    DescriptorCatalogEntry(
        "jerk_gate_ratio", "dyn_jerk_peak_to_floor_ratio", "noop_jerk_gate_ratio",
        "dynamic_kinematics", "dynamic",
        r"\max(\|\mathbf{j}\|_2) / \max(j_{\mathrm{floor}}, 10^{-4})", "dimensionless", ("acc_x", "acc_y", "acc_z", "dt"),
        ">= max(2, min_samples - 1) contiguous pairs",
        "Scale-free transient spike ratio relative to quiescent jerk floor",
        "Internal port (cr7_kimmitch SleepStagerV2)",
    ),
    DescriptorCatalogEntry(
        "acc_gyro_corr", "dyn_acc_gyro_envelope_corr", "noop_acc_gyro_corr",
        "dynamic_kinematics", "dynamic",
        r"\mathrm{corr}(|\|\mathbf{a}\|_2 - \mu_a|, \|\boldsymbol{\omega}\|_2)", "[-1, 1]", ("acc_*", "rot_*"),
        ">= min_samples jointly observed rows",
        "Cross-modal Pearson correlation between acceleration AC envelope and angular speed",
        "Internal port (cr7_kimmitch Whoop5RawImu)",
    ),
    DescriptorCatalogEntry(
        "gyro_to_acc_ratio", "dyn_gyro_to_acc_l2_rms_ratio", "noop_gyro_to_acc_ratio",
        "dynamic_kinematics", "dynamic",
        r"\mathrm{RMS}(\|\boldsymbol{\omega}\|_2) / (\mathrm{RMS}(\|\mathbf{a}\|_2 - \mu_a) + 10^{-4})", "(rad/s)/(m/s^2)", ("acc_*", "rot_*"),
        ">= min_samples jointly observed rows",
        "Consistent L2-RMS rotational-to-translational energy ratio",
        "Internal port (cr7_kimmitch Whoop5RawImu; L2-consistent normalization)",
    ),
    DescriptorCatalogEntry(
        "dyn_amp", "dyn_local_baseline_amplitude", "noop_dyn_amp",
        "dynamic_kinematics", "dynamic",
        r"\frac{1}{N}\sum \|\mathbf{a}_t - \bar{\mathbf{a}}_{t,W}\|_2", "m/s^2", ("acc_x", "acc_y", "acc_z"),
        ">= min_samples observed rows",
        "Calibration-invariant per-axis high-pass dynamic amplitude (clean-room dynAmp concept)",
        "Straczkiewicz et al. (2023) doi:10.1038/s41746-022-00745-z; O'Connell et al. (2017)",
    ),
    # --- Walch DoG ---
    DescriptorCatalogEntry(
        "dog_variability", "walch_dog_variability", "noop_dog_variability",
        "walch_dog", "dynamic",
        r"\mathrm{RMS}(G_{\sigma_1}(\|\mathbf{a}\|_2) - G_{\sigma_2}(\|\mathbf{a}\|_2))", "m/s^2", ("acc_x", "acc_y", "acc_z"),
        ">= min_samples observed rows",
        "Walch multi-scale Difference-of-Gaussians variability via scipy.ndimage.gaussian_filter1d",
        "Walch et al. (2019), Sleep 42(12): zsz180",
    ),
    # --- Baevsky Stress Index ---
    DescriptorCatalogEntry(
        "baevsky_amo_pct", "baevsky_mode_amplitude_pct", "noop_baevsky_amo_pct",
        "baevsky_stress", "dynamic",
        r"100 \cdot \max_b(h_b) / N_{\mathrm{clip}}", "% [0, 100]", ("acc_x", "acc_y", "acc_z"),
        ">= min_samples observed rows",
        "Baevsky histogram Mode Amplitude percentage",
        "Baevsky & Berseneva (2008)",
    ),
    DescriptorCatalogEntry(
        "baevsky_mxdmn", "baevsky_robust_range_mxdmn", "noop_baevsky_mxdmn",
        "baevsky_stress", "dynamic",
        r"P_{98}(\|\mathbf{a}\|_2) - P_{2}(\|\mathbf{a}\|_2)", "m/s^2", ("acc_x", "acc_y", "acc_z"),
        ">= min_samples observed rows",
        "Baevsky robust dispersion range (98th - 2nd percentile)",
        "Baevsky & Berseneva (2008)",
    ),
    DescriptorCatalogEntry(
        "baevsky_si", "baevsky_stress_index", "noop_baevsky_si",
        "baevsky_stress", "dynamic",
        r"\mathrm{AMo}\% / (2 \cdot \mathrm{Mo}_{\mathrm{scale}} \cdot \mathrm{MxDMn})", "1/(m/s^2)^2", ("acc_x", "acc_y", "acc_z"),
        ">= min_samples observed rows",
        "Baevsky centralization / stress index",
        "Baevsky & Berseneva (2008)",
    ),
    # --- Autocorrelation Periodicity ---
    DescriptorCatalogEntry(
        "acf_frequency_hz", "acf_dominant_frequency_hz", "noop_acf_frequency_hz",
        "autocorrelation_periodicity", "periodicity",
        r"f_s / \arg\max_{\tau \in [\tau_{\min}, \tau_{\max}]} R_{xx}(\tau)", "Hz", ("selected_signal",),
        ">= min_samples contiguous run & >= min_cycles",
        "Dominant-axis normalized autocorrelation frequency via scipy.signal.correlate + find_peaks",
        "SciPy signal (https://docs.scipy.org/doc/scipy/reference/signal.html)",
    ),
    DescriptorCatalogEntry(
        "acf_strength", "acf_dominant_strength", "noop_acf_strength",
        "autocorrelation_periodicity", "periodicity",
        r"\mathrm{clip}(R_{xx}(\tau^*), 0, 1)", "[0, 1]", ("selected_signal",),
        ">= min_samples contiguous run & >= min_cycles",
        "Normalized autocorrelation peak height",
        "SciPy signal (https://docs.scipy.org/doc/scipy/reference/signal.html)",
    ),
    DescriptorCatalogEntry(
        "acf_magnitude_harmonic", "acf_magnitude_double_harmonic_flag", "noop_acf_magnitude_harmonic",
        "autocorrelation_periodicity", "periodicity",
        r"\mathbb{I}(|f_{\mathrm{mag}} / f_{\mathrm{axis}} - 2| \le 2\epsilon)", "{0, 1}", ("selected_3axis_signal",),
        ">= min_samples contiguous run on 3-axis signal",
        "Flags transverse-sinusoid frequency-doubling ambiguity on scalar magnitude",
        "Internal port (cr7_kimmitch PpgHr / ImuFeatureExtractor)",
    ),
    DescriptorCatalogEntry(
        "cadence_hz", "acf_cadence_fundamental_hz", "noop_cadence_hz",
        "autocorrelation_periodicity", "periodicity",
        r"f_s / \tau^* \text{ if } R(\tau^*) \ge \theta_{\min}", "Hz", ("selected_signal",),
        ">= max(8, min_samples) contiguous run",
        "Autocorrelation cadence frequency inside cadence_band",
        "Internal port (cr7_kimmitch ImuFeatureExtractor)",
    ),
    DescriptorCatalogEntry(
        "cadence_sublag_hz", "acf_parabolic_sublag_fundamental_hz", "noop_cadence_sublag_hz",
        "autocorrelation_periodicity", "periodicity",
        r"f_s / (\tau^* + \frac{y_0 - y_2}{2(y_0 - 2y_1 + y_2)})", "Hz", ("selected_signal",),
        ">= max(8, min_samples) contiguous run",
        "Fundamental-preferred sub-lag parabolic-interpolated ACF frequency",
        "Internal port (cr7_kimmitch PpgHr)",
    ),
    DescriptorCatalogEntry(
        "cadence_strength", "acf_cadence_strength", "noop_cadence_strength",
        "autocorrelation_periodicity", "periodicity",
        r"\mathrm{clip}(R(\tau^*), 0, 1)", "[0, 1]", ("selected_signal",),
        ">= max(8, min_samples) contiguous run",
        "Normalized autocorrelation strength at chosen fundamental lag",
        "Internal port (cr7_kimmitch ImuFeatureExtractor / PpgHr)",
    ),
    DescriptorCatalogEntry(
        "acf_sublag_concavity", "acf_parabolic_peak_concavity", "noop_acf_sublag_concavity",
        "autocorrelation_periodicity", "periodicity",
        r"\max(0, 2y_1 - y_0 - y_2)", "dimensionless", ("selected_signal",),
        ">= max(8, min_samples) contiguous run",
        "Second-difference peak sharpness around the fundamental ACF peak",
        "Internal port (cr7_kimmitch PpgHr)",
    ),
    # --- Hann PSD Spectral ---
    DescriptorCatalogEntry(
        "spectral_peak_fraction", "psd_hann_peak_power_fraction", "noop_spectral_peak_fraction",
        "hann_spectral", "spectral",
        r"\max_k P_k / \sum_j P_j", "[0, 1]", ("selected_signal",),
        ">= min_samples contiguous run & >= 2 in-band bins",
        "Fraction of in-band Hann PSD power concentrated at peak bin",
        "SciPy signal.periodogram (https://docs.scipy.org/doc/scipy/reference/signal.html)",
    ),
    DescriptorCatalogEntry(
        "spectral_peak_neighbourhood_fraction", "psd_hann_peak_neighbourhood_fraction", "noop_spectral_peak_neighbourhood_fraction",
        "hann_spectral", "spectral",
        r"\sum_{|f_k - f^*|\le \Delta f} P_k / \sum_j P_j", "[0, 1]", ("selected_signal",),
        ">= min_samples contiguous run & >= 2 in-band bins",
        "Power fraction within +/- spectral_neighbourhood_hz of peak frequency",
        "SciPy signal.periodogram",
    ),
    DescriptorCatalogEntry(
        "spectral_entropy_normalized", "psd_hann_shannon_entropy_norm", "noop_spectral_entropy_normalized",
        "hann_spectral", "spectral",
        r"-\sum p_k \log_2(p_k) / \log_2(K)", "[0, 1]", ("selected_signal",),
        ">= min_samples contiguous run & >= 2 in-band bins",
        "Normalized Shannon entropy of in-band Hann PSD",
        "Shannon (1948); SciPy signal.periodogram",
    ),
    DescriptorCatalogEntry(
        "spectral_peakedness", "psd_hann_herfindahl_peakedness", "noop_spectral_peakedness",
        "hann_spectral", "spectral",
        r"\sum_k p_k^2", "[1/K, 1]", ("selected_signal",),
        ">= min_samples contiguous run & >= 2 in-band bins",
        "Herfindahl-Hirschman spectral concentration index",
        "SciPy signal.periodogram",
    ),
    # --- Signal Quality ---
    DescriptorCatalogEntry(
        "observed_fraction", "qual_observed_sample_fraction", "noop_observed_fraction",
        "signal_quality", "quality",
        r"N_{\mathrm{obs}} / N", "[0, 1]", ("observed_mask",),
        ">= 1 sample",
        "Fraction of window samples backed by real sensor observations",
        "Observation provenance contract",
    ),
    DescriptorCatalogEntry(
        "imputed_fraction", "qual_imputed_sample_fraction", "noop_imputed_fraction",
        "signal_quality", "quality",
        r"N_{\mathrm{imp}} / N", "[0, 1]", ("imputed_mask",),
        ">= 1 sample",
        "Fraction of window samples filled via bounded interior interpolation",
        "Observation provenance contract",
    ),
    DescriptorCatalogEntry(
        "longest_contiguous_fraction", "qual_longest_contiguous_run_fraction", "noop_longest_contiguous_fraction",
        "signal_quality", "quality",
        r"L_{\max} / N", "[0, 1]", ("counter", "observed_mask"),
        ">= 1 sample",
        "Fraction of window covered by the longest gap-free contiguous run",
        "Observation provenance contract",
    ),
    DescriptorCatalogEntry(
        "counter_gap_fraction", "qual_counter_gap_fraction", "noop_counter_gap_fraction",
        "signal_quality", "quality",
        r"N_{\mathrm{absent}} / N", "[0, 1]", ("row_present",),
        ">= 1 sample",
        "Fraction of dense counter grid slots absent from the raw recording",
        "Observation provenance contract",
    ),
    # --- Non-IMU Branch Drift ---
    DescriptorCatalogEntry(
        "thm_drift_slope", "branch_thm_linear_drift_slope_per_s", "noop_thm_drift_slope",
        "branch_drift", "branch_drift",
        r"\mathrm{slope}(t_{\mathrm{sec}}, \mathrm{thm\_mean})", "degC/s", ("thm_mean",),
        ">= max(4, min(min_samples, 0.5*fs))",
        "OLS linear thermal drift slope via scipy.stats.linregress at native 20 Hz",
        "SciPy stats.linregress",
    ),
    DescriptorCatalogEntry(
        "thm_dev_rms", "branch_thm_baseline_deviation_rms", "noop_thm_dev_rms",
        "branch_drift", "branch_drift",
        r"\mathrm{std}(\mathrm{thm\_mean})", "degC", ("thm_mean",),
        ">= max(4, min(min_samples, 0.5*fs))",
        "Baseline deviation RMS of pooled thermopile temperature",
        "NumPy std",
    ),
    DescriptorCatalogEntry(
        "thm_motion_cv", "branch_thm_step_delta_cv", "noop_thm_motion_cv",
        "branch_drift", "branch_drift",
        r"\mathrm{std}(|\Delta \mathrm{thm}|) / \mathrm{mean}(|\Delta \mathrm{thm}|)", "dimensionless", ("thm_mean",),
        ">= max(4, min(min_samples, 0.5*fs))",
        "Step-delta burstiness CV of thermopile trace",
        "NumPy diff/std",
    ),
    DescriptorCatalogEntry(
        "tof_drift_slope", "branch_tof_linear_drift_slope_per_s", "noop_tof_drift_slope",
        "branch_drift", "branch_drift",
        r"\mathrm{slope}(t_{\mathrm{sec}}, \mathrm{tof\_mean})", "mm/s", ("tof_mean",),
        ">= max(4, min(min_samples, 0.5*fs))",
        "OLS linear proximity drift slope via scipy.stats.linregress at native 20 Hz",
        "SciPy stats.linregress",
    ),
    DescriptorCatalogEntry(
        "tof_dev_rms", "branch_tof_baseline_deviation_rms", "noop_tof_dev_rms",
        "branch_drift", "branch_drift",
        r"\mathrm{std}(\mathrm{tof\_mean})", "mm", ("tof_mean",),
        ">= max(4, min(min_samples, 0.5*fs))",
        "Baseline deviation RMS of pooled ToF proximity",
        "NumPy std",
    ),
    DescriptorCatalogEntry(
        "tof_motion_cv", "branch_tof_step_delta_cv", "noop_tof_motion_cv",
        "branch_drift", "branch_drift",
        r"\mathrm{std}(|\Delta \mathrm{tof}|) / \mathrm{mean}(|\Delta \mathrm{tof}|)", "dimensionless", ("tof_mean",),
        ">= max(4, min(min_samples, 0.5*fs))",
        "Step-delta burstiness CV of ToF proximity trace",
        "NumPy diff/std",
    ),
)

BASE_TO_CANONICAL: Dict[str, str] = {e.base_key: e.canonical_name for e in DESCRIPTOR_CATALOG}
LEGACY_TO_CANONICAL: Dict[str, str] = {e.legacy_name: e.canonical_name for e in DESCRIPTOR_CATALOG}
CANONICAL_TO_LEGACY: Dict[str, str] = {e.canonical_name: e.legacy_name for e in DESCRIPTOR_CATALOG}


def format_descriptor_name(base_key: str, *, naming_mode: str = "legacy") -> str:
    """Resolve feature column name under ``'legacy'`` (``noop_*``) or ``'canonical'`` scientific naming."""
    if naming_mode == "legacy":
        return f"noop_{base_key}"
    if base_key in BASE_TO_CANONICAL:
        return BASE_TO_CANONICAL[base_key]
    # Dynamic band indices or branch-prefixed spectral keys
    if base_key.startswith(("thm_spectral_", "tof_spectral_")):
        branch, rest = base_key.split("_", 1)
        return f"branch_{branch}_psd_hann_{rest.removeprefix('spectral_')}"
    if base_key.startswith("spectral_"):
        return f"psd_hann_{base_key.removeprefix('spectral_')}"
    return f"desc_{base_key}"


LEGACY_MOTION_ATOMS = (
    "acc_ac_rms",
    "axis_rms",
    "jerk_rms",
    "rotation_mean",
    "rotation_rms",
    "gyro_energy",
)
EXTENDED_MOTION_ATOMS = (
    "posture_var",
    "motion_cv",
    "jerk_floor",
    "acc_gyro_coupling",
    "dog_var",
    "baevsky_si",
    "dyn_amp",
)
DYNAMIC_KINEMATIC_ATOMS = LEGACY_MOTION_ATOMS + EXTENDED_MOTION_ATOMS
HANN_SPECTRAL_ATOMS = (
    "peak_fraction",
    "peak_neighbourhood_fraction",
    "band_fractions",
    "entropy",
    "peakedness",
)
PERIODICITY_MODES = (None, "acf", "acf_harmonic_checked", "acf_cadence", "acf_kimmitch")
QUALITY_MODES = (None, "report", "features")
DESCRIPTOR_SIGNALS = ("raw", "linear", "angular_speed")
POWER_BRANCHES = ("imu", "thm", "tof")
_MODE_PARAMETERS = ("motion_stats", "periodicity", "spectral_stats", "quality")
_TUNING_PARAMETERS = (
    "signal",
    "band",
    "bands",
    "min_cycles",
    "min_samples",
    "max_gap_samples",
    "acf_harmonic_tolerance",
    "spectral_neighbourhood_hz",
    "detrend",
    "cadence_band",
    "min_cadence_strength",
    "power_branches",
    "nyquist_policy",
)


def descriptor_subset_modes(
    atoms: Sequence[str],
    aliases: Sequence[str] = (),
    max_combo_size: int = 6,
) -> List[Optional[str]]:
    """Generate ``[None, *aliases, '|'-joined subsets]`` menu for descriptor search spaces."""
    modes: List[Optional[str]] = [None, *aliases]
    seen = set(modes)
    upper = min(len(atoms), int(max_combo_size))
    for size in range(1, upper + 1):
        for subset in itertools.combinations(atoms, size):
            token = "|".join(subset)
            if token not in seen:
                seen.add(token)
                modes.append(token)
    full_token = "|".join(atoms)
    if full_token not in seen:
        modes.append(full_token)
    return modes


MOTION_DESCRIPTOR_MODES = descriptor_subset_modes(
    DYNAMIC_KINEMATIC_ATOMS, aliases=("basic", "kimmitch", "full"), max_combo_size=4
)
SPECTRAL_DESCRIPTOR_MODES = descriptor_subset_modes(
    HANN_SPECTRAL_ATOMS, aliases=("all",), max_combo_size=5
)


class SignalDescriptorOptions(BaseEstimator):
    """Sklearn-compatible configuration for modular signal descriptors.

    Accepts ``None`` (off), explicit lists/tuples of atoms, canonical preset
    aliases (``'basic'``, ``'kimmitch'``, ``'full'``, ``'all'``), or
    pipe-delimited mode strings. Aliased as ``NoopOptions`` for backward
    compatibility with existing notebooks and tests.
    """

    def __init__(
        self,
        motion_stats=None,
        periodicity=None,
        spectral_stats=None,
        quality=None,
        signal="raw",
        band=(0.5, 8.0),
        bands=((0.5, 2.0), (2.0, 4.0), (4.0, 8.0)),
        min_cycles=2.0,
        min_samples=16,
        max_gap_samples=0,
        acf_harmonic_tolerance=0.1,
        spectral_neighbourhood_hz=0.5,
        detrend="constant",
        cadence_band=(1.2, 3.5),
        min_cadence_strength=0.2,
        power_branches="imu",
        nyquist_policy="clip",
    ):
        self.motion_stats = motion_stats
        self.periodicity = periodicity
        self.spectral_stats = spectral_stats
        self.quality = quality
        self.signal = signal
        self.band = band
        self.bands = bands
        self.min_cycles = min_cycles
        self.min_samples = min_samples
        self.max_gap_samples = max_gap_samples
        self.acf_harmonic_tolerance = acf_harmonic_tolerance
        self.spectral_neighbourhood_hz = spectral_neighbourhood_hz
        self.detrend = detrend
        self.cadence_band = cadence_band
        self.min_cadence_strength = min_cadence_strength
        self.power_branches = power_branches
        self.nyquist_policy = nyquist_policy

    def mode_names(self):
        motion = parse_pipe_modes(
            self.motion_stats,
            DYNAMIC_KINEMATIC_ATOMS,
            name="noop__motion_stats",
            aliases={
                "basic": {"acc_ac_rms", "rotation_mean"},
                "kimmitch": {
                    "acc_ac_rms",
                    "jerk_rms",
                    "gyro_energy",
                    "posture_var",
                    "motion_cv",
                    "jerk_floor",
                    "acc_gyro_coupling",
                    "dog_var",
                    "baevsky_si",
                },
                "full": set(DYNAMIC_KINEMATIC_ATOMS),
                "all": set(DYNAMIC_KINEMATIC_ATOMS),
            },
            exc_type=InvalidParams,
        )
        spectral = parse_pipe_modes(
            self.spectral_stats,
            HANN_SPECTRAL_ATOMS,
            name="noop__spectral_stats",
            aliases={"all": set(HANN_SPECTRAL_ATOMS)},
            exc_type=InvalidParams,
        )
        if self.periodicity not in PERIODICITY_MODES:
            raise InvalidParams(f"noop__periodicity must be one of {PERIODICITY_MODES}")
        if self.quality not in QUALITY_MODES:
            raise InvalidParams(f"noop__quality must be one of {QUALITY_MODES}")
        return motion, spectral

    def branch_names(self):
        return (
            parse_pipe_modes(
                self.power_branches,
                POWER_BRANCHES,
                name="noop__power_branches",
                aliases={"all": set(POWER_BRANCHES)},
                exc_type=InvalidParams,
            )
            or {"imu"}
        )

    def enabled(self) -> bool:
        return any(getattr(self, name) is not None for name in _MODE_PARAMETERS)

    def validate(self, sampling_rate=None):
        motion, spectral = self.mode_names()
        if not self.enabled():
            return self
        signal_used = self.periodicity is not None or bool(spectral) or self.quality is not None
        if signal_used and self.signal not in DESCRIPTOR_SIGNALS:
            raise InvalidParams("noop__signal must be raw, linear or angular_speed")
        if self.nyquist_policy not in {"clip", "error"}:
            raise InvalidParams("noop__nyquist_policy must be 'clip' or 'error'")
        self.branch_names()
        inactive = inactive_descriptor_parameters(self)
        for name, minimum in (("min_samples", 4), ("max_gap_samples", 0)):
            if f"noop__{name}" in inactive:
                continue
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise InvalidParams(f"noop__{name} must be an integer >= {minimum}")
        if (self.periodicity is not None or spectral) and self.detrend not in {"constant", "linear"}:
            raise InvalidParams("noop__detrend must be constant or linear")
        if self.periodicity is not None or spectral:
            if not np.isfinite(self.min_cycles) or self.min_cycles <= 0:
                raise InvalidParams("noop__min_cycles must be finite and positive")
            self.frequency_ranges(sampling_rate)
        if self.periodicity == "acf_harmonic_checked":
            if not np.isfinite(self.acf_harmonic_tolerance) or not 0 <= self.acf_harmonic_tolerance <= 0.5:
                raise InvalidParams("noop__acf_harmonic_tolerance must be in [0, 0.5]")
        if self.periodicity in {"acf_cadence", "acf_kimmitch"}:
            lo, _ = self._bounds(self.cadence_band, None, "cadence_band")
            if lo <= 0:
                raise InvalidParams("noop__cadence_band needs a strictly positive lower bound")
            if not 0 <= float(self.min_cadence_strength) <= 1:
                raise InvalidParams("noop__min_cadence_strength must be in [0, 1]")
        if "peak_neighbourhood_fraction" in spectral:
            if not np.isfinite(self.spectral_neighbourhood_hz) or self.spectral_neighbourhood_hz <= 0:
                raise InvalidParams("noop__spectral_neighbourhood_hz must be finite and positive")
        return self

    def _bounds(self, value, fs, label="band"):
        value = parse_structured_value(value, exc_type=InvalidParams)
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise InvalidParams(f"Descriptor {label} must contain (low, high) bounds")
        lo, hi = map(float, value)
        if not np.isfinite([lo, hi]).all() or not 0 <= lo < hi:
            raise InvalidParams(f"Descriptor {label} must satisfy 0 <= low < high")
        if fs is not None and hi > fs / 2:
            if self.nyquist_policy == "error":
                raise InvalidParams(f"Descriptor {label} must satisfy high <= Nyquist ({fs / 2:g} Hz)")
            hi = fs / 2
            lo = min(lo, hi)
        return lo, hi

    def frequency_ranges(self, sampling_rate=None):
        fs = None if sampling_rate is None else positive_rate(sampling_rate, "descriptor sampling_rate", exc_type=InvalidParams)
        band = self._bounds(self.band, fs)
        if band[0] <= 0 and self.periodicity in {"acf", "acf_harmonic_checked"}:
            raise InvalidParams("Autocorrelation requires a strictly positive lower frequency")
        _, spectral = self.mode_names()
        bands = (
            tuple(self._bounds(b, fs) for b in parse_structured_value(self.bands, exc_type=InvalidParams))
            if "band_fractions" in spectral
            else ()
        )
        return band, bands


# Backward-compatible alias
NoopOptions = SignalDescriptorOptions


def options_from_settings(settings) -> SignalDescriptorOptions:
    if isinstance(settings, SignalDescriptorOptions) or type(settings).__name__ in {"SignalDescriptorOptions", "NoopOptions"}:
        return settings
    if settings is None:
        return SignalDescriptorOptions()
    raise InvalidParams("noop must be None or SignalDescriptorOptions / NoopOptions")


def inactive_descriptor_parameters(settings) -> set:
    """Return tuning parameters that cannot affect outputs under the enabled modes."""
    options = options_from_settings(settings)
    motion, spectral = options.mode_names()
    active = set()
    if motion or options.periodicity is not None or spectral:
        active.add("min_samples")
    if options.periodicity is not None or spectral or options.quality is not None:
        active |= {"max_gap_samples", "signal"}
    if options.periodicity is not None or spectral:
        active |= {"signal", "band", "min_cycles", "detrend", "nyquist_policy"}
    if options.periodicity == "acf_harmonic_checked":
        active.add("acf_harmonic_tolerance")
    if options.periodicity in {"acf_cadence", "acf_kimmitch"}:
        active |= {"cadence_band", "min_cadence_strength"}
    if "peak_neighbourhood_fraction" in spectral:
        active.add("spectral_neighbourhood_hz")
    if "band_fractions" in spectral:
        active.add("bands")
    if spectral or motion:
        active.add("power_branches")
    return {f"noop__{name}" for name in _TUNING_PARAMETERS if name not in active}


def walch_dog_variability(
    x: np.ndarray,
    fs: float,
    *,
    short_seconds: float = 0.12,
    long_seconds: float = 0.65,
) -> float:
    """Compute RMS of the Difference-of-Gaussians bandpass signal.

    Formula:
        ``DoG(x) = G_{sigma_short}(x) - G_{sigma_long}(x)``
        ``walch_dog_variability = sqrt(mean(DoG(x)^2))``
    where ``sigma_short = max(1.0, short_seconds * fs)`` and
    ``sigma_long = max(sigma_short + 1.0, long_seconds * fs)``.

    Parameters
    ----------
    x : np.ndarray
        1-D scalar series (e.g., acceleration magnitude in ``m/s^2``).
    fs : float
        Sampling rate in Hz.
    short_seconds, long_seconds : float
        Gaussian smoothing time constants in seconds.

    Returns
    -------
    float
        RMS variability in the same units as ``x`` (``0.0`` if fewer than 4 samples).
    """
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if len(arr) < 4:
        return 0.0
    short_sigma = max(1.0, float(short_seconds) * float(fs))
    long_sigma = max(short_sigma + 1.0, float(long_seconds) * float(fs))
    short_smooth = gaussian_filter1d(arr, sigma=short_sigma, mode="nearest", truncate=3.0)
    long_smooth = gaussian_filter1d(arr, sigma=long_sigma, mode="nearest", truncate=3.0)
    diff = short_smooth - long_smooth
    return float(np.sqrt(np.mean(diff ** 2)))


def baevsky_stress_metrics(x: np.ndarray, n_bins: int = 16) -> Tuple[float, float, float]:
    """Compute Baevsky Mode Amplitude (``AMo%``), robust range (``MxDMn``), and Stress Index (``SI``).

    Formula:
        Let ``p2, p98 = percentile(x, [2, 98])``, ``MxDMn = p98 - p2``.
        Histogram ``x`` clipped to ``[p2, p98]`` into ``n_bins`` uniform bins:
        - ``AMo% = 100 * max(counts) / N_clipped`` (%)
        - ``Mo`` = center of peak bin (scaled with ``max(|Mo|, std, 1e-3)``)
        - ``SI = AMo% / (2 * scale_Mo * max(MxDMn, 1e-3))``

    Returns
    -------
    tuple of (float, float, float)
        ``(amo_pct, mxdmn, si)``. Returns ``(0.0, 0.0, 0.0)`` when ``len(x) < 4``
        and ``(100.0, 0.0, 0.0)`` for constant signals.
    """
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if len(arr) < 4:
        return 0.0, 0.0, 0.0
    p2, p98 = np.percentile(arr, [2.0, 98.0])
    mxdmn = float(max(0.0, p98 - p2))
    if mxdmn <= 1e-9:
        return 100.0, 0.0, 0.0
    clipped = arr[(arr >= p2) & (arr <= p98)]
    if len(clipped) < 2:
        clipped = arr
    counts, edges = np.histogram(clipped, bins=max(4, int(n_bins)), range=(float(p2), float(p98)))
    peak_bin = int(np.argmax(counts))
    amo_pct = 100.0 * float(counts[peak_bin]) / float(len(clipped))
    mo_center = 0.5 * float(edges[peak_bin] + edges[peak_bin + 1])
    scale_mo = max(abs(mo_center), float(np.std(clipped)), 1e-3)
    si = amo_pct / (2.0 * scale_mo * max(mxdmn, 1e-3))
    return float(amo_pct), float(mxdmn), float(si)


def local_baseline_dynamic_amplitude(
    acc: np.ndarray,
    fs: float,
    *,
    baseline_seconds: float = 0.5,
) -> float:
    """Calibration-invariant local-baseline dynamic amplitude (``dynAmp``).

    Clean-room implementation of the high-pass per-axis dynamic motion concept:
    removes a local per-axis moving average (cancelling constant gravity and
    per-axis calibration offsets exactly), then averages the L2 vector magnitude.

    Formula:
        ``a_dyn[t] = a[t] - moving_mean_W(a)[t]``
        ``dyn_amp = mean_t(||a_dyn[t]||_2)``
    where ``W = max(3, int(round(baseline_seconds * fs)))``.

    Units: ``m/s^2`` (same as ``acc``).
    """
    arr = np.asarray(acc, dtype=np.float64)
    if arr.ndim != 2 or len(arr) < 3:
        return 0.0
    win = max(3, min(len(arr), int(round(float(baseline_seconds) * float(fs)))))
    baseline = uniform_filter1d(arr, size=win, axis=0, mode="nearest")
    dynamic = arr - baseline
    return float(np.mean(np.linalg.norm(dynamic, axis=1)))


def compute_dynamic_kinematics(
    window: IMUWindow,
    fs: float,
    atoms: Iterable[str],
    *,
    min_samples: int = 16,
) -> Dict[str, Tuple[float, bool]]:
    """Evaluate requested dynamic IMU descriptor atoms over one ``IMUWindow``.

    Returns a dict mapping base atom feature names (without prefix) to
    ``(value, supported)`` tuples.
    """
    wanted = set(atoms)
    raw = window.raw
    observed_axes = window.observed
    raw_observed = observed_axes.all(axis=1) & np.isfinite(raw).all(axis=1)
    q_observed = window.rotation_observed
    contiguous = np.r_[False, np.diff(window.counter) == 1] if len(window.counter) else np.zeros(0, dtype=bool)
    rotation = window.rotation
    rotation_valid = (
        q_observed
        & np.r_[False, q_observed[:-1]]
        & contiguous
        & np.isfinite(rotation).all(axis=1)
    ) if len(q_observed) else np.zeros(0, dtype=bool)

    def enough(valid: np.ndarray) -> bool:
        return int(np.sum(valid)) >= int(min_samples)

    out: Dict[str, Tuple[float, bool]] = {}
    obs_raw = raw[raw_observed]

    if "acc_ac_rms" in wanted:
        magnitude = np.linalg.norm(obs_raw, axis=1)
        out["acc_ac_rms"] = (float(np.std(magnitude)) if len(magnitude) else 0.0, enough(raw_observed))

    if "axis_rms" in wanted:
        for axis, axis_name in enumerate("xyz"):
            valid = observed_axes[:, axis] & np.isfinite(raw[:, axis])
            val = float(np.sqrt(np.mean(raw[valid, axis] ** 2))) if valid.any() else 0.0
            out[f"acc_{axis_name}_rms"] = (val, enough(valid))

    if "jerk_rms" in wanted:
        valid = raw_observed[1:] & raw_observed[:-1] & contiguous[1:]
        dt = window.dt[1:]
        valid &= np.isfinite(dt) & (dt > 0)
        jerk = np.diff(raw, axis=0)[valid] / dt[valid, None] if len(raw) > 1 and valid.any() else np.empty((0, 3))
        val = float(np.sqrt(np.mean(np.sum(jerk ** 2, axis=1)))) if len(jerk) else 0.0
        out["jerk_rms"] = (val, int(valid.sum()) >= max(1, int(min_samples) - 1))

    speed = np.linalg.norm(rotation[rotation_valid], axis=1) if rotation_valid.any() else np.empty(0)
    if "rotation_mean" in wanted:
        out["rotation_mean_rad_s"] = (float(np.mean(speed)) if len(speed) else 0.0, enough(rotation_valid))
    if "rotation_rms" in wanted:
        out["rotation_rms_rad_s"] = (float(np.sqrt(np.mean(speed ** 2))) if len(speed) else 0.0, enough(rotation_valid))
    if "gyro_energy" in wanted:
        out["gyro_energy_dps"] = (float(np.degrees(np.mean(speed))) if len(speed) else 0.0, enough(rotation_valid))

    if "posture_var" in wanted:
        p_var = float(np.sum(np.var(obs_raw, axis=0))) if len(obs_raw) else 0.0
        out["posture_var"] = (p_var, enough(raw_observed))

    if "motion_cv" in wanted:
        pair_valid = raw_observed[1:] & raw_observed[:-1] & contiguous[1:]
        ok_pairs = int(pair_valid.sum()) >= max(2, int(min_samples) - 1)
        deltas = np.linalg.norm(np.diff(raw, axis=0)[pair_valid], axis=1) if len(raw) > 1 and pair_valid.any() else np.empty(0)
        mean_i = float(np.mean(deltas)) if len(deltas) else 0.0
        var_i = float(np.var(deltas)) if len(deltas) else 0.0
        cv_i = float(np.sqrt(var_i) / mean_i) if mean_i > 1e-6 else 0.0
        out["motion_intensity_mean"] = (mean_i, ok_pairs)
        out["motion_variance"] = (var_i, ok_pairs)
        out["motion_cv"] = (cv_i, ok_pairs)

    if "jerk_floor" in wanted:
        pair_valid = raw_observed[1:] & raw_observed[:-1] & contiguous[1:]
        dt = window.dt[1:]
        pair_valid &= np.isfinite(dt) & (dt > 0)
        ok_jerk = int(pair_valid.sum()) >= max(2, int(min_samples) - 1)
        if len(raw) > 1 and pair_valid.any():
            j_mag = np.linalg.norm(np.diff(raw, axis=0)[pair_valid] / dt[pair_valid, None], axis=1)
            j_med = float(np.median(j_mag))
            j_scale = max(j_med, 1e-4)
            move_frac = float(np.mean(j_mag > 2.0 * j_scale))
            still_frac = float(np.mean(j_mag <= 1.25 * j_scale))
            gate_ratio = float(np.max(j_mag) / j_scale)
        else:
            j_med = move_frac = still_frac = gate_ratio = 0.0
        out["jerk_floor"] = (j_med, ok_jerk)
        out["move_fraction"] = (move_frac, ok_jerk)
        out["still_fraction"] = (still_frac, ok_jerk)
        out["jerk_gate_ratio"] = (gate_ratio, ok_jerk)

    if "acc_gyro_coupling" in wanted:
        # Consistent L2 normalization: RMS of centered acceleration magnitude vs RMS of angular speed
        joint = raw_observed & rotation_valid
        ok_joint = enough(joint)
        if ok_joint:
            acc_m = np.linalg.norm(raw[joint], axis=1)
            acc_centered = acc_m - float(np.mean(acc_m))
            rot_m = np.linalg.norm(rotation[joint], axis=1)
            rot_centered = rot_m - float(np.mean(rot_m))
            acc_l2_rms = float(np.sqrt(np.mean(acc_centered ** 2)))
            rot_l2_rms = float(np.sqrt(np.mean(rot_m ** 2)))
            rot_std = float(np.sqrt(np.mean(rot_centered ** 2)))
            # Correlate L2 envelopes (centered magnitude squared root / AC deviation vs angular speed)
            acc_env = np.abs(acc_centered)
            std_env = float(np.std(acc_env))
            corr_ag = (
                float(np.clip(np.corrcoef(acc_env, rot_m)[0, 1], -1.0, 1.0))
                if (std_env > 1e-6 and rot_std > 1e-6)
                else 0.0
            )
            ratio_ag = float(rot_l2_rms / (acc_l2_rms + 1e-4))
        else:
            corr_ag = ratio_ag = 0.0
        out["acc_gyro_corr"] = (corr_ag, ok_joint)
        out["gyro_to_acc_ratio"] = (ratio_ag, ok_joint)

    if "dog_var" in wanted:
        ok_dog = enough(raw_observed)
        dog_val = walch_dog_variability(np.linalg.norm(obs_raw, axis=1), fs) if ok_dog else 0.0
        out["dog_variability"] = (dog_val, ok_dog)

    if "baevsky_si" in wanted:
        ok_b = enough(raw_observed)
        amo_pct, mxdmn, si_val = baevsky_stress_metrics(np.linalg.norm(obs_raw, axis=1)) if ok_b else (0.0, 0.0, 0.0)
        out["baevsky_amo_pct"] = (amo_pct, ok_b)
        out["baevsky_mxdmn"] = (mxdmn, ok_b)
        out["baevsky_si"] = (si_val, ok_b)

    if "dyn_amp" in wanted:
        ok_da = enough(raw_observed)
        da_val = local_baseline_dynamic_amplitude(obs_raw, fs) if ok_da else 0.0
        out["dyn_amp"] = (da_val, ok_da)

    return out


def _normalized_acf(x: np.ndarray) -> Tuple[np.ndarray, float]:
    """One-sided normalized autocorrelation via ``scipy.signal.correlate``."""
    n = len(x)
    energy = float(x @ x)
    if energy <= np.finfo(float).eps * max(1, n):
        return np.zeros(n, dtype=np.float64), energy
    corr = sps.correlate(x, x, mode="full", method="auto")[n - 1 :] / energy
    return corr, energy


def acf_dominant_frequency(
    values: np.ndarray,
    fs: float,
    *,
    band: Tuple[float, float] = (0.5, 8.0),
    min_cycles: float = 2.0,
    detrend: str = "constant",
) -> Tuple[float, float, bool]:
    """Strongest autocorrelation peak of the dominant-energy axis within ``band``.

    Returns ``(frequency_hz, strength, supported)``.
    """
    x = detrend_values(values, detrend)
    if x.ndim == 1:
        x = x[:, None]
    if len(x) < 4:
        return 0.0, 0.0, False
    axis = int(np.argmax(np.sum(x * x, axis=0)))
    x1 = x[:, axis]
    corr, energy = _normalized_acf(x1)
    if energy <= np.finfo(float).eps * max(1, len(x1)):
        return 0.0, 0.0, False
    low, high = float(band[0]), min(float(band[1]), float(fs) / 2.0)
    if low <= 0 or low >= high:
        return 0.0, 0.0, False
    first = max(1, int(np.ceil(fs / high)))
    last = min(len(x1) - 2, int(np.floor(fs / low)), int(np.floor(len(x1) / float(min_cycles))))
    if last < first:
        return 0.0, 0.0, False
    # Include one neighbor on each side so scipy.signal.find_peaks evaluates boundary lags
    seg = corr[first - 1 : last + 2]
    rel_peaks, _ = sps.find_peaks(seg, height=0.0)
    if len(rel_peaks) == 0:
        return 0.0, 0.0, False
    lags = rel_peaks + (first - 1)
    best_lag = int(lags[int(np.argmax(corr[lags]))])
    return float(fs) / float(best_lag), float(np.clip(corr[best_lag], 0.0, 1.0)), True


def acf_cadence_estimate(
    magnitude: np.ndarray,
    fs: float,
    band: Tuple[float, float] = (1.2, 3.5),
    min_strength: float = 0.2,
) -> Tuple[float, float, bool]:
    """Normalized autocorrelation cadence of a 1-D magnitude signal over ``band``.

    Returns ``(cadence_hz, strength, usable)``; ``cadence_hz == 0.0`` denotes no
    rhythm above ``min_strength``.
    """
    ac = np.asarray(magnitude, dtype=np.float64).reshape(-1)
    n = len(ac)
    if n < 8:
        return 0.0, 0.0, False
    ac = ac - np.mean(ac)
    corr, ac0 = _normalized_acf(ac)
    if ac0 <= np.finfo(float).eps * n:
        return 0.0, 0.0, True  # stationary signal is a valid "no cadence" reading
    low, high = float(band[0]), min(float(band[1]), float(fs) / 2.0)
    if low <= 0 or low >= high:
        return 0.0, 0.0, False
    lo_lag = max(1, int(round(fs / high)))
    hi_lag = min(n - 1, int(round(fs / low)))
    if lo_lag >= hi_lag:
        return 0.0, 0.0, False
    lags = np.arange(lo_lag, hi_lag + 1)
    strengths = corr[lags]
    k = int(np.argmax(strengths))
    best = float(strengths[k])
    cadence = float(fs) / float(lags[k]) if best >= float(min_strength) and best > 0 else 0.0
    return cadence, max(0.0, best), True


def acf_parabolic_fundamental(
    magnitude: np.ndarray,
    fs: float,
    band: Tuple[float, float] = (1.2, 3.5),
    min_strength: float = 0.2,
    fundamental_ratio: float = 0.85,
) -> Tuple[float, float, float, float, bool]:
    """Fundamental-preferred sub-lag parabolic-refined autocorrelation estimator.

    1. Linearly detrends ``magnitude`` via ``scipy.signal.detrend``.
    2. Computes normalized ACF via ``scipy.signal.correlate`` over ``band``.
    3. Finds positive local peaks via ``scipy.signal.find_peaks`` and picks the
       smallest lag (highest fundamental frequency) reaching ``>= fundamental_ratio * max_peak``
       so 2x/3x subharmonics do not steal the fundamental.
    4. Refines the lag using 3-point parabolic interpolation:
       ``delta = 0.5 * (y0 - y2) / (y0 - 2*y1 + y2)``,
       ``concavity = max(0, 2*y1 - y0 - y2)``.

    Returns ``(cadence_hz, sublag_hz, strength, concavity, usable)``.
    """
    raw = np.asarray(magnitude, dtype=np.float64).reshape(-1)
    n = len(raw)
    if n < 8:
        return 0.0, 0.0, 0.0, 0.0, False
    ac = detrend_values(raw[:, None], "linear")[:, 0]
    corr, ac0 = _normalized_acf(ac)
    if ac0 <= np.finfo(float).eps * n:
        return 0.0, 0.0, 0.0, 0.0, True
    low, high = float(band[0]), min(float(band[1]), float(fs) / 2.0)
    if low <= 0 or low >= high:
        return 0.0, 0.0, 0.0, 0.0, False
    lo_lag = max(1, int(round(fs / high)))
    hi_lag = min(n - 2, int(round(fs / low)))
    if lo_lag >= hi_lag:
        return 0.0, 0.0, 0.0, 0.0, False
    seg = corr[lo_lag - 1 : hi_lag + 2]
    rel_peaks, _ = sps.find_peaks(seg, height=0.0)
    if len(rel_peaks) > 0:
        peaks = rel_peaks + (lo_lag - 1)
        max_peak = float(np.max(corr[peaks]))
        threshold = float(fundamental_ratio) * max_peak
        chosen_lag = int(np.min(peaks[corr[peaks] >= threshold]))
    else:
        lags = np.arange(lo_lag, hi_lag + 1)
        chosen_lag = int(lags[int(np.argmax(corr[lags]))])
    best = float(np.clip(corr[chosen_lag], 0.0, 1.0))
    if 0 < chosen_lag < n - 1:
        y0, y1, y2 = float(corr[chosen_lag - 1]), float(corr[chosen_lag]), float(corr[chosen_lag + 1])
        denom = y0 - 2.0 * y1 + y2
        delta = float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5)) if abs(denom) > 1e-12 else 0.0
        concavity = max(0.0, 2.0 * y1 - y0 - y2)
    else:
        delta = 0.0
        concavity = 0.0
    refined_lag = max(1e-6, float(chosen_lag) + delta)
    if best >= float(min_strength) and best > 0:
        hz = float(fs) / float(chosen_lag)
        sublag_hz = float(fs) / refined_lag
    else:
        hz = 0.0
        sublag_hz = 0.0
    return hz, sublag_hz, best, float(concavity), True


def compute_hann_spectral_descriptors(
    selected: np.ndarray,
    usable: bool,
    fs: float,
    *,
    spectral_atoms: Iterable[str],
    band: Tuple[float, float],
    bands: Sequence[Tuple[float, float]] = (),
    min_cycles: float = 2.0,
    detrend: str = "constant",
    spectral_neighbourhood_hz: float = 0.5,
    prefix: str = "",
) -> Dict[str, Tuple[float, bool]]:
    """Compute rate-adapted Hann-windowed one-sided PSD descriptors.

    Uses ``scipy.signal.periodogram`` (via ``features.physics.window_psd``).
    Atoms: ``peak_fraction``, ``peak_neighbourhood_fraction``, ``band_fractions``,
    ``entropy`` (normalized Shannon spectral entropy in ``[0, 1]``), and
    ``peakedness`` (Herfindahl concentration index ``sum(p_k^2)``).
    """
    spectral = set(spectral_atoms)
    if usable and len(selected) >= 2:
        frequencies, density = window_psd(selected, fs, detrend)
    else:
        frequencies, density = np.empty(0), np.empty(0)
    resolved_low = max(float(band[0]), float(min_cycles) * float(fs) / max(1, len(selected)))
    in_band = (frequencies >= resolved_low) & (frequencies <= float(band[1]))
    power = density[in_band]
    total = float(power.sum())
    supported = bool(usable) and len(power) >= 2 and total > np.finfo(float).tiny
    peak = int(np.argmax(power)) if supported else 0
    peak_frequency = float(frequencies[in_band][peak]) if supported else 0.0

    out: Dict[str, Tuple[float, bool]] = {}
    if "peak_fraction" in spectral:
        out[f"{prefix}spectral_peak_fraction"] = (
            float(power[peak] / total) if supported else 0.0,
            supported,
        )
    if "peak_neighbourhood_fraction" in spectral:
        near = in_band & (np.abs(frequencies - peak_frequency) <= float(spectral_neighbourhood_hz))
        out[f"{prefix}spectral_peak_neighbourhood_fraction"] = (
            float(density[near].sum() / total) if supported else 0.0,
            supported,
        )
    if "band_fractions" in spectral:
        for idx, (low, high) in enumerate(bands):
            inside = in_band & (frequencies >= float(low)) & (frequencies < float(high))
            out[f"{prefix}spectral_band_{idx}_fraction"] = (
                float(density[inside].sum() / total) if supported else 0.0,
                bool(supported and inside.any()),
            )
    if "entropy" in spectral:
        if supported:
            p = power[power > 0] / total
            entropy = float(-np.sum(p * np.log2(p)) / np.log2(len(power)))
        else:
            entropy = 0.0
        out[f"{prefix}spectral_entropy_normalized"] = (entropy, supported)
    if "peakedness" in spectral:
        if supported:
            p_norm = power / total
            peakedness = float(np.sum(p_norm ** 2))
        else:
            peakedness = 0.0
        out[f"{prefix}spectral_peakedness"] = (peakedness, supported)
    return out


def compute_signal_quality(
    counter: np.ndarray,
    observed: np.ndarray,
    imputed: np.ndarray,
    longest_run: np.ndarray,
    row_present: np.ndarray,
    *,
    signal_source: str = "raw",
    emit_features: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Tuple[float, bool]]]:
    """Compute per-window observation provenance report and optional numeric quality features."""
    n = len(counter)
    present = np.asarray(row_present, dtype=bool)
    gaps = int(np.sum(np.diff(present.astype(int)) == -1) + (0 if not len(present) or present[0] else 1))
    report: Dict[str, Any] = {
        "real_samples": int(n),
        "observed_samples": int(np.sum(observed)),
        "observed_fraction": float(np.mean(observed)) if n else 0.0,
        "imputed_samples": int(np.sum(imputed)),
        "longest_contiguous_samples": int(len(longest_run)),
        "counter_gap_count": gaps,
        "source": str(signal_source),
    }
    features: Dict[str, Tuple[float, bool]] = {}
    if emit_features:
        features["observed_fraction"] = (report["observed_fraction"], n > 0)
        features["imputed_fraction"] = (float(np.sum(imputed)) / max(1, n), n > 0)
        features["longest_contiguous_fraction"] = (float(len(longest_run)) / max(1, n), n > 0)
        features["counter_gap_fraction"] = (float((~present).sum()) / max(1, n), n > 0)
    return report, features


def compute_branch_drift_descriptors(
    values: np.ndarray,
    observed: np.ndarray,
    fs: float,
    options: Any,
    prefix: str,
) -> Dict[str, Tuple[float, bool]]:
    """Compute rate-adapted drift slope, deviation RMS, step-delta CV, and PSD for ``thm`` / ``tof``.

    Uses ``scipy.stats.linregress`` for physical per-second drift slope and
    ``scipy.signal.periodogram`` (via ``compute_hann_spectral_descriptors``) at
    the branch's actual sampling rate ``fs``.
    """
    motion, spectral = options.mode_names()
    out: Dict[str, Tuple[float, bool]] = {}
    if not motion and not spectral:
        return out
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    obs = np.asarray(observed, dtype=bool) & np.isfinite(arr).all(axis=1)
    counter = np.arange(len(arr), dtype=np.float64)
    signal, valid, _ = bounded_fill(arr, obs, counter, int(options.max_gap_samples))
    runs = contiguous_runs(valid, counter)
    longest = max(runs, key=len) if runs else np.empty(0, dtype=int)
    selected = signal[longest]
    usable = len(selected) >= max(4, min(int(options.min_samples), int(round(0.5 * float(fs)))))

    if motion:
        series = selected[:, 0] if len(selected) else np.empty(0)
        if usable and len(series) >= 2:
            t_sec = np.arange(len(series), dtype=np.float64) / float(fs)
            if float(np.std(t_sec)) > 1e-12:
                slope = float(_sp_linregress(t_sec, series).slope)
                if not np.isfinite(slope):
                    slope = 0.0
            else:
                slope = 0.0
            dev_rms = float(np.std(series))
            diffs = np.abs(np.diff(series))
            mean_d = float(np.mean(diffs)) if len(diffs) else 0.0
            cv_d = float(np.std(diffs) / mean_d) if mean_d > 1e-6 else 0.0
        else:
            slope = dev_rms = cv_d = 0.0
        out[f"{prefix}_drift_slope"] = (slope, usable)
        out[f"{prefix}_dev_rms"] = (dev_rms, usable)
        out[f"{prefix}_motion_cv"] = (cv_d, usable)

    if spectral:
        band, bands = options.frequency_ranges(fs)
        out.update(
            compute_hann_spectral_descriptors(
                selected,
                usable,
                fs,
                spectral_atoms=spectral,
                band=band,
                bands=bands,
                min_cycles=float(options.min_cycles),
                detrend=str(options.detrend),
                spectral_neighbourhood_hz=float(options.spectral_neighbourhood_hz),
                prefix=f"{prefix}_",
            )
        )
    return out


def _emit_pairs(
    target: Dict[str, float],
    pairs: Mapping[str, Tuple[float, bool]],
    *,
    naming_mode: str = "legacy",
) -> None:
    """Write ``(value, supported)`` pairs into ``target`` under requested naming convention."""
    for base_key, (value, supported) in pairs.items():
        ok = bool(supported) and bool(np.isfinite(value))
        col_name = format_descriptor_name(base_key, naming_mode=naming_mode)
        target[col_name] = float(value) if ok else 0.0
        target[f"{col_name}_supported"] = float(ok)


def summarize_imu_window(
    src: Union[IMUWindow, Mapping[str, np.ndarray]],
    fs: float,
    options: SignalDescriptorOptions,
    *,
    naming_mode: str = "legacy",
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Compute all enabled dynamic, periodicity, spectral, and quality descriptors for one IMU window."""
    window = src if isinstance(src, IMUWindow) else IMUWindow.from_mapping(src)
    motion, spectral = options.mode_names()
    min_samples = int(options.min_samples) if (motion or options.periodicity is not None or spectral) else 16
    max_gap = int(options.max_gap_samples) if (options.periodicity is not None or spectral or options.quality is not None) else 0

    result: Dict[str, float] = {}
    if motion:
        kin_pairs = compute_dynamic_kinematics(window, fs, motion, min_samples=min_samples)
        _emit_pairs(result, kin_pairs, naming_mode=naming_mode)

    raw_observed = window.observed.all(axis=1) & np.isfinite(window.raw).all(axis=1)
    contiguous = np.r_[False, np.diff(window.counter) == 1] if len(window.counter) else np.zeros(0, dtype=bool)
    rotation_valid = (
        window.rotation_observed
        & np.r_[False, window.rotation_observed[:-1]]
        & contiguous
        & np.isfinite(window.rotation).all(axis=1)
    ) if len(window.rotation_observed) else np.zeros(0, dtype=bool)

    if options.signal == "raw":
        signal, observed = window.raw, raw_observed
    elif options.signal == "linear":
        signal = window.linear
        observed = raw_observed & window.rotation_observed & np.isfinite(signal).all(axis=1)
    else:
        signal = np.linalg.norm(window.rotation, axis=1)[:, None]
        observed = rotation_valid

    signal, valid, imputed = bounded_fill(signal, observed, window.counter, max_gap)
    runs = contiguous_runs(valid, window.counter)
    longest = max(runs, key=len) if runs else np.empty(0, dtype=int)
    selected = signal[longest]
    usable = len(selected) >= min_samples

    period_pairs: Dict[str, Tuple[float, bool]] = {}
    if options.periodicity in {"acf", "acf_harmonic_checked"}:
        band, _ = options.frequency_ranges(fs)
        freq, strength, supported = (
            acf_dominant_frequency(selected, fs, band=band, min_cycles=float(options.min_cycles), detrend=str(options.detrend))
            if usable
            else (0.0, 0.0, False)
        )
        period_pairs["acf_frequency_hz"] = (freq, supported)
        period_pairs["acf_strength"] = (strength, supported)
        if options.periodicity == "acf_harmonic_checked":
            checked = supported and selected.shape[1] == 3
            mag_freq, _, mag_sup = (
                acf_dominant_frequency(
                    np.linalg.norm(selected, axis=1)[:, None],
                    fs,
                    band=band,
                    min_cycles=float(options.min_cycles),
                    detrend=str(options.detrend),
                )
                if checked
                else (0.0, 0.0, False)
            )
            ambiguous = mag_sup and freq > 0 and abs(mag_freq / freq - 2.0) <= 2.0 * float(options.acf_harmonic_tolerance)
            period_pairs["acf_magnitude_harmonic"] = (float(ambiguous), bool(checked and mag_sup))
    elif options.periodicity == "acf_cadence":
        magnitude = np.linalg.norm(selected, axis=1) if len(selected) else np.empty(0)
        hz, strength, supported = (
            acf_cadence_estimate(
                magnitude,
                fs,
                tuple(parse_structured_value(options.cadence_band)),
                float(options.min_cadence_strength),
            )
            if usable
            else (0.0, 0.0, False)
        )
        period_pairs["cadence_hz"] = (hz, bool(supported and hz > 0))
        period_pairs["cadence_strength"] = (strength, supported)
    elif options.periodicity == "acf_kimmitch":
        magnitude = np.linalg.norm(selected, axis=1) if len(selected) else np.empty(0)
        hz, sublag_hz, strength, concavity, supported = (
            acf_parabolic_fundamental(
                magnitude,
                fs,
                tuple(parse_structured_value(options.cadence_band)),
                float(options.min_cadence_strength),
            )
            if usable
            else (0.0, 0.0, 0.0, 0.0, False)
        )
        period_pairs["cadence_hz"] = (hz, bool(supported and hz > 0))
        period_pairs["cadence_sublag_hz"] = (sublag_hz, bool(supported and sublag_hz > 0))
        period_pairs["cadence_strength"] = (strength, supported)
        period_pairs["acf_sublag_concavity"] = (concavity, supported)

    if period_pairs:
        _emit_pairs(result, period_pairs, naming_mode=naming_mode)

    if spectral and "imu" in options.branch_names():
        band, bands = options.frequency_ranges(fs)
        spec_pairs = compute_hann_spectral_descriptors(
            selected,
            usable,
            fs,
            spectral_atoms=spectral,
            band=band,
            bands=bands,
            min_cycles=float(options.min_cycles),
            detrend=str(options.detrend),
            spectral_neighbourhood_hz=float(options.spectral_neighbourhood_hz),
        )
        _emit_pairs(result, spec_pairs, naming_mode=naming_mode)

    report, qual_pairs = compute_signal_quality(
        window.counter,
        observed,
        imputed,
        longest,
        window.row_present,
        signal_source=str(options.signal),
        emit_features=(options.quality == "features"),
    )
    if qual_pairs:
        _emit_pairs(result, qual_pairs, naming_mode=naming_mode)
    return result, report


def summarize_branch_window(
    values: np.ndarray,
    observed: np.ndarray,
    fs: float,
    options: SignalDescriptorOptions,
    prefix: str,
    *,
    naming_mode: str = "legacy",
) -> Dict[str, float]:
    """Compute rate-adapted non-IMU (``thm`` / ``tof``) drift and spectral descriptors."""
    pairs = compute_branch_drift_descriptors(values, observed, fs, options, prefix)
    result: Dict[str, float] = {}
    _emit_pairs(result, pairs, naming_mode=naming_mode)
    return result

# ===========================================================================
# 5. Window-Level Honeycomb Domain Feature Cells
# ===========================================================================

def angular_distance_rad(q_wxyz: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Consecutive geodesic quaternion angle ``2 * arccos(|dot(q[t-1], q[t])|)`` in radians."""
    n = len(q_wxyz)
    out = np.zeros(n, dtype=np.float64)
    if n < 2:
        return out
    q_u, ok = normalize_quaternions(q_wxyz)
    ok &= np.asarray(valid, dtype=bool)
    pairs = ok[1:] & ok[:-1]
    if pairs.any():
        dots = np.clip(np.abs(np.sum(q_u[1:][pairs] * q_u[:-1][pairs], axis=1)), 0.0, 1.0)
        out[1:][pairs] = 2.0 * np.arccos(dots)
    return out


def compute_orientation_features(context: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    """Compute per-window orientation and posture descriptors respecting masks and ``fs``.

    Features (all finite, zero-filled with explicit ``orientation__supported`` flag):
      - Gravity tilt angle between ``acc`` and rotated gravity vector: mean, std, min, max (rad)
      - Geodesic quaternion distance from window start & mean posture: mean, max (rad)
      - Total angular displacement ``sum(||omega|| * dt)`` (rad)
      - Angular acceleration magnitude: mean, std, max (rad/s^2)
      - Angular jerk magnitude: mean, std, max (rad/s^3)
      - Total unwrapped Euler roll/pitch/yaw excursion (rad)
      - Posture change count exceeding ``posture_change_threshold_rad``
    """
    imu = np.asarray(context["branches"]["imu"], dtype=np.float64)
    mask = np.asarray(context["masks"]["imu"], dtype=bool)
    q_obs = np.asarray(context["q_obs"], dtype=bool)
    obs_axes = np.asarray(context["obs_axes"], dtype=bool).all(axis=-1)
    ch_names = context["channel_names"]["imu"]
    fs = float(context["rates"]["imu"])
    dt = 1.0 / fs
    threshold_rad = float(config.get("posture_change_threshold_rad", 0.15))
    n = len(imu)

    keys = (
        "orientation__gravity_tilt_mean_rad",
        "orientation__gravity_tilt_std_rad",
        "orientation__gravity_tilt_min_rad",
        "orientation__gravity_tilt_max_rad",
        "orientation__quat_dist_start_mean_rad",
        "orientation__quat_dist_start_max_rad",
        "orientation__quat_dist_mean_posture_mean_rad",
        "orientation__quat_dist_mean_posture_max_rad",
        "orientation__total_angular_displacement_rad",
        "orientation__ang_acc_mag_mean_rad_s2",
        "orientation__ang_acc_mag_std_rad_s2",
        "orientation__ang_acc_mag_max_rad_s2",
        "orientation__ang_jerk_mag_mean_rad_s3",
        "orientation__ang_jerk_mag_std_rad_s3",
        "orientation__ang_jerk_mag_max_rad_s3",
        "orientation__euler_unwrapped_roll_change_rad",
        "orientation__euler_unwrapped_pitch_change_rad",
        "orientation__euler_unwrapped_yaw_change_rad",
        "orientation__posture_change_count",
        "orientation__supported",
    )
    out = {k: np.zeros(n, dtype=np.float64) for k in keys}
    if n == 0:
        return out

    has_q = all(f"rot_{a}" in ch_names for a in "wxyz")
    has_acc = all(f"acc_{a}" in ch_names for a in "xyz")
    has_ang = all(f"ang_{a}" in ch_names for a in "xyz")
    q_idx = [ch_names.index(f"rot_{a}") for a in "wxyz"] if has_q else None
    acc_idx = [ch_names.index(f"acc_{a}") for a in "xyz"] if has_acc else None
    ang_idx = [ch_names.index(f"ang_{a}") for a in "xyz"] if has_ang else None
    g_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    for i in range(n):
        L = int(mask[i].sum())
        if L < 4:
            continue
        valid_q = q_obs[i, :L] if has_q else np.zeros(L, dtype=bool)
        if has_q and int(valid_q.sum()) >= 3:
            q_seg = imu[i, :L][:, q_idx]
            q_u, norm_ok = normalize_quaternions(q_seg)
            vq = valid_q & norm_ok
            if int(vq.sum()) >= 3:
                out["orientation__supported"][i] = 1.0
                q_valid = q_u[vq]
                # Geodesic distance from window start
                q0 = q_valid[0]
                d_start = 2.0 * np.arccos(np.clip(np.abs(q_valid @ q0), 0.0, 1.0))
                out["orientation__quat_dist_start_mean_rad"][i] = float(np.mean(d_start))
                out["orientation__quat_dist_start_max_rad"][i] = float(np.max(d_start))

                # Geodesic distance from mean posture (hemisphere-aligned mean quaternion)
                dots0 = q_valid @ q0
                aligned = q_valid * np.where(dots0 < 0, -1.0, 1.0)[:, None]
                q_mean = aligned.mean(axis=0)
                q_mean_norm = float(np.linalg.norm(q_mean))
                if q_mean_norm > 1e-12:
                    q_mean /= q_mean_norm
                    d_mean = 2.0 * np.arccos(np.clip(np.abs(q_valid @ q_mean), 0.0, 1.0))
                    out["orientation__quat_dist_mean_posture_mean_rad"][i] = float(np.mean(d_mean))
                    out["orientation__quat_dist_mean_posture_max_rad"][i] = float(np.max(d_mean))

                # Consecutive posture change count
                step_d = angular_distance_rad(q_seg, vq)
                out["orientation__posture_change_count"][i] = float(np.sum(step_d > threshold_rad))

                # Unwrapped Euler excursion via SciPy Rotation
                eul = quaternions_to_euler(q_valid, seq="xyz", unwrap=True)
                if len(eul) >= 2:
                    deul = np.abs(np.diff(eul, axis=0)).sum(axis=0)
                    out["orientation__euler_unwrapped_roll_change_rad"][i] = float(deul[0])
                    out["orientation__euler_unwrapped_pitch_change_rad"][i] = float(deul[1])
                    out["orientation__euler_unwrapped_yaw_change_rad"][i] = float(deul[2])

                # Gravity tilt angle between acc and gravity direction in body frame
                if has_acc:
                    joint = vq & obs_axes[i, :L]
                    if joint.any():
                        acc_v = imu[i, :L][:, acc_idx][joint]
                        acc_n = np.linalg.norm(acc_v, axis=1)
                        good_a = acc_n > 1e-6
                        if good_a.any():
                            g_body = quaternions_to_rotations(q_u[joint][good_a]).inv().apply(g_world)
                            cos_t = np.clip(np.sum((acc_v[good_a] / acc_n[good_a, None]) * g_body, axis=1), -1.0, 1.0)
                            tilt = np.arccos(cos_t)
                            out["orientation__gravity_tilt_mean_rad"][i] = float(np.mean(tilt))
                            out["orientation__gravity_tilt_std_rad"][i] = float(np.std(tilt))
                            out["orientation__gravity_tilt_min_rad"][i] = float(np.min(tilt))
                            out["orientation__gravity_tilt_max_rad"][i] = float(np.max(tilt))

        if has_ang:
            ang = imu[i, :L][:, ang_idx]
            ang_speed = np.linalg.norm(ang, axis=1)
            out["orientation__total_angular_displacement_rad"][i] = float(np.sum(ang_speed) * dt)
            if L >= 3:
                ang_acc = np.diff(ang, axis=0) * fs
                aa_mag = np.linalg.norm(ang_acc, axis=1)
                out["orientation__ang_acc_mag_mean_rad_s2"][i] = float(np.mean(aa_mag))
                out["orientation__ang_acc_mag_std_rad_s2"][i] = float(np.std(aa_mag))
                out["orientation__ang_acc_mag_max_rad_s2"][i] = float(np.max(aa_mag))
                if L >= 4:
                    ang_jerk = np.diff(ang_acc, axis=0) * fs
                    aj_mag = np.linalg.norm(ang_jerk, axis=1)
                    out["orientation__ang_jerk_mag_mean_rad_s3"][i] = float(np.mean(aj_mag))
                    out["orientation__ang_jerk_mag_std_rad_s3"][i] = float(np.std(aj_mag))
                    out["orientation__ang_jerk_mag_max_rad_s3"][i] = float(np.max(aj_mag))

    return out


def _impulsive_factors(x: np.ndarray) -> Dict[str, float]:
    """Crest, shape, impulse, and clearance factors for a 1-D real signal."""
    a = np.asarray(x, dtype=np.float64)
    if len(a) == 0:
        return {"crest": 0.0, "shape": 0.0, "impulse": 0.0, "clearance": 0.0}
    abs_a = np.abs(a)
    peak = float(np.max(abs_a))
    rms = float(np.sqrt(np.mean(a * a)))
    mav = float(np.mean(abs_a))
    sra = float(np.mean(np.sqrt(abs_a)) ** 2)
    return {
        "crest": peak / rms if rms > 1e-12 else 0.0,
        "shape": rms / mav if mav > 1e-12 else 0.0,
        "impulse": peak / mav if mav > 1e-12 else 0.0,
        "clearance": peak / sra if sra > 1e-12 else 0.0,
    }


def _welch_spectral_shape(x: np.ndarray, fs: float, nperseg: int = 64) -> Dict[str, float]:
    """Calibrated Welch PSD shape descriptors via ``scipy.signal.welch``."""
    a = np.asarray(x, dtype=np.float64)
    if len(a) < 8 or float(np.std(a)) <= 1e-12:
        return {
            "dom_freq_hz": 0.0,
            "sec_freq_hz": 0.0,
            "peak_to_sec_ratio": 0.0,
            "centroid_hz": 0.0,
            "bandwidth_hz": 0.0,
            "rolloff_85_hz": 0.0,
            "rolloff_95_hz": 0.0,
            "flatness": 0.0,
            "spectral_kurtosis": 0.0,
            "harmonic_ratio": 0.0,
            "band_low_power": 0.0,
            "band_mid_power": 0.0,
            "band_tremor_power": 0.0,
        }
    seg_len = min(len(a), max(8, int(nperseg)))
    freqs, psd = sps.welch(a, fs=float(fs), nperseg=seg_len, detrend="constant")
    # Drop DC bin for dynamic spectral shape
    if len(freqs) > 1:
        freqs, psd = freqs[1:], psd[1:]
    total = float(np.sum(psd))
    if total <= 1e-15 or len(freqs) < 2:
        return {k: 0.0 for k in (
            "dom_freq_hz", "sec_freq_hz", "peak_to_sec_ratio", "centroid_hz", "bandwidth_hz",
            "rolloff_85_hz", "rolloff_95_hz", "flatness", "spectral_kurtosis", "harmonic_ratio",
            "band_low_power", "band_mid_power", "band_tremor_power",
        )}
    df = float(freqs[1] - freqs[0])
    p = psd / total
    order = np.argsort(-psd)
    f1 = float(freqs[order[0]])
    p1 = float(psd[order[0]])
    f2 = float(freqs[order[1]]) if len(order) > 1 else 0.0
    p2 = float(psd[order[1]]) if len(order) > 1 else 0.0
    centroid = float(np.sum(freqs * p))
    bw = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * p)))
    cdf = np.cumsum(p)
    r85 = float(freqs[min(int(np.searchsorted(cdf, 0.85)), len(freqs) - 1)])
    r95 = float(freqs[min(int(np.searchsorted(cdf, 0.95)), len(freqs) - 1)])
    pos = np.maximum(psd, 1e-18)
    flatness = float(np.exp(np.mean(np.log(pos))) / np.mean(pos))
    skurt = float(np.sum(p * (((freqs - centroid) / bw) ** 4)) - 3.0) if bw > 1e-12 else 0.0

    # Harmonic energy ratio around 2*f1 and 3*f1
    harm_power = 0.0
    for mult in (2.0, 3.0):
        target = mult * f1
        if target < freqs[-1]:
            harm_mask = np.abs(freqs - target) <= max(df, 0.1 * target)
            harm_power += float(np.sum(psd[harm_mask]))
    harm_ratio = harm_power / total

    # BFRB-relevant frequency bands (0.5-3 Hz macro, 3-8 Hz rhythmic, 8-15 Hz tremor-like; Nyquist-safe)
    def _band_p(lo: float, hi: float) -> float:
        m = (freqs >= lo) & (freqs < min(hi, float(fs) / 2.0))
        return float(np.sum(psd[m]) * df) if m.any() else 0.0

    return {
        "dom_freq_hz": f1,
        "sec_freq_hz": f2,
        "peak_to_sec_ratio": p1 / (p2 + 1e-12),
        "centroid_hz": centroid,
        "bandwidth_hz": bw,
        "rolloff_85_hz": r85,
        "rolloff_95_hz": r95,
        "flatness": flatness,
        "spectral_kurtosis": skurt,
        "harmonic_ratio": harm_ratio,
        "band_low_power": _band_p(0.5, 3.0),
        "band_mid_power": _band_p(3.0, 8.0),
        "band_tremor_power": _band_p(8.0, 15.0),
    }


def _safe_corr(a: np.ndarray, b: np.ndarray, lag: int = 0) -> float:
    if lag > 0:
        if len(a) <= lag + 2:
            return 0.0
        a, b = a[:-lag], b[lag:]
    if len(a) < 3 or float(np.std(a)) <= 1e-9 or float(np.std(b)) <= 1e-9:
        return 0.0
    c = float(np.corrcoef(a, b)[0, 1])
    return float(np.clip(c, -1.0, 1.0)) if np.isfinite(c) else 0.0


def compute_vibration_features(context: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    """Compute multiaxial IMU, impulsive, cross-axis covariance/PCA, snap, dynAmp & Welch PSD features."""
    imu = np.asarray(context["branches"]["imu"], dtype=np.float64)
    mask = np.asarray(context["masks"]["imu"], dtype=bool)
    ch_names = context["channel_names"]["imu"]
    fs = float(context["rates"]["imu"])
    nperseg = int(config.get("welch_nperseg", 64))
    n = len(imu)

    axes = ("x", "y", "z")
    has_lin = all(f"lin_{a}" in ch_names for a in axes)
    has_ang = all(f"ang_{a}" in ch_names for a in axes)

    col_names = []
    # Axis-specific descriptors
    for a in axes:
        for stat in ("rms", "std", "kurt", "peak", "crest", "shape", "impulse", "clearance", "zcr"):
            col_names.append(f"vib__acc_{a}__{stat}")
    if has_lin:
        for a in axes:
            for stat in ("rms", "crest", "zcr"):
                col_names.append(f"vib__lin_{a}__{stat}")
    if has_ang:
        for a in axes:
            for stat in ("rms", "crest", "zcr"):
                col_names.append(f"vib__ang_{a}__{stat}")
    # Cross-axis correlations (lag 0 & lag 1), covariance eigenvalues & PCA ratios
    for pair in ("xy", "xz", "yz"):
        col_names.extend([f"vib__acc_corr_{pair}_lag0", f"vib__acc_corr_{pair}_lag1"])
    col_names.extend([
        "vib__acc_cov_eig1", "vib__acc_cov_eig2", "vib__acc_cov_eig3",
        "vib__acc_pca_ratio1", "vib__acc_pca_ratio2",
        "vib__acc_ratio_xy", "vib__acc_ratio_xz", "vib__acc_ratio_yz",
        "vib__acc_mag__dyn_amp",
        "vib__acc_mag__p10", "vib__acc_mag__p25", "vib__acc_mag__p75", "vib__acc_mag__p90", "vib__acc_mag__p95",
        "vib__acc_mag__mad", "vib__acc_mag__trimmed_mean", "vib__acc_mag__cv",
        "vib__jerk_mag__p50", "vib__jerk_mag__p90", "vib__jerk_mag__p95", "vib__jerk_sign_change_rate",
        "vib__snap_rms", "vib__snap_rms_supported",
    ])
    for k in (
        "dom_freq_hz", "sec_freq_hz", "peak_to_sec_ratio", "centroid_hz", "bandwidth_hz",
        "rolloff_85_hz", "rolloff_95_hz", "flatness", "spectral_kurtosis", "harmonic_ratio",
        "band_low_power", "band_mid_power", "band_tremor_power",
    ):
        col_names.append(f"vib__acc_mag_welch__{k}")

    out = {c: np.zeros(n, dtype=np.float64) for c in col_names}
    if n == 0:
        return out

    acc_idx = [ch_names.index(f"acc_{a}") for a in axes]
    lin_idx = [ch_names.index(f"lin_{a}") for a in axes] if has_lin else None
    ang_idx = [ch_names.index(f"ang_{a}") for a in axes] if has_ang else None

    for i in range(n):
        L = int(mask[i].sum())
        if L < 4:
            continue
        acc = imu[i, :L][:, acc_idx]
        acc_mag = np.linalg.norm(acc, axis=1)

        # Axis-specific stats
        axis_rms_vals = []
        for idx_a, a in enumerate(axes):
            s = acc[:, idx_a]
            centered = s - np.mean(s)
            rms = float(np.sqrt(np.mean(s * s)))
            axis_rms_vals.append(rms)
            std = float(np.std(s))
            kurt = float(_sp_kurtosis(s, fisher=True, bias=False)) if (L >= 4 and std > 1e-9) else 0.0
            factors = _impulsive_factors(centered)
            zcr = float(np.mean(np.sign(centered[1:]) != np.sign(centered[:-1]))) if L >= 2 else 0.0
            out[f"vib__acc_{a}__rms"][i] = rms
            out[f"vib__acc_{a}__std"][i] = std
            out[f"vib__acc_{a}__kurt"][i] = kurt if np.isfinite(kurt) else 0.0
            out[f"vib__acc_{a}__peak"][i] = float(np.max(np.abs(centered)))
            out[f"vib__acc_{a}__crest"][i] = factors["crest"]
            out[f"vib__acc_{a}__shape"][i] = factors["shape"]
            out[f"vib__acc_{a}__impulse"][i] = factors["impulse"]
            out[f"vib__acc_{a}__clearance"][i] = factors["clearance"]
            out[f"vib__acc_{a}__zcr"][i] = zcr

        if has_lin and lin_idx is not None:
            lin = imu[i, :L][:, lin_idx]
            for idx_a, a in enumerate(axes):
                s = lin[:, idx_a]
                c_s = s - np.mean(s)
                out[f"vib__lin_{a}__rms"][i] = float(np.sqrt(np.mean(s * s)))
                out[f"vib__lin_{a}__crest"][i] = _impulsive_factors(c_s)["crest"]
                out[f"vib__lin_{a}__zcr"][i] = float(np.mean(np.sign(c_s[1:]) != np.sign(c_s[:-1])))

        if has_ang and ang_idx is not None:
            ang = imu[i, :L][:, ang_idx]
            for idx_a, a in enumerate(axes):
                s = ang[:, idx_a]
                c_s = s - np.mean(s)
                out[f"vib__ang_{a}__rms"][i] = float(np.sqrt(np.mean(s * s)))
                out[f"vib__ang_{a}__crest"][i] = _impulsive_factors(c_s)["crest"]
                out[f"vib__ang_{a}__zcr"][i] = float(np.mean(np.sign(c_s[1:]) != np.sign(c_s[:-1])))

        # Cross-axis correlations (lag 0 and lag 1)
        for pair, (p0, p1) in (("xy", (0, 1)), ("xz", (0, 2)), ("yz", (1, 2))):
            out[f"vib__acc_corr_{pair}_lag0"][i] = _safe_corr(acc[:, p0], acc[:, p1], lag=0)
            out[f"vib__acc_corr_{pair}_lag1"][i] = _safe_corr(acc[:, p0], acc[:, p1], lag=1)

        # 3x3 covariance eigenvalues & PCA energy ratios
        cov = np.cov(acc, rowvar=False)
        eigs = np.sort(np.maximum(np.linalg.eigvalsh(cov), 0.0))[::-1]
        tr = float(np.sum(eigs))
        out["vib__acc_cov_eig1"][i] = float(eigs[0])
        out["vib__acc_cov_eig2"][i] = float(eigs[1])
        out["vib__acc_cov_eig3"][i] = float(eigs[2])
        out["vib__acc_pca_ratio1"][i] = float(eigs[0] / tr) if tr > 1e-12 else 0.0
        out["vib__acc_pca_ratio2"][i] = float(eigs[1] / tr) if tr > 1e-12 else 0.0

        out["vib__acc_ratio_xy"][i] = axis_rms_vals[0] / (axis_rms_vals[1] + 1e-6)
        out["vib__acc_ratio_xz"][i] = axis_rms_vals[0] / (axis_rms_vals[2] + 1e-6)
        out["vib__acc_ratio_yz"][i] = axis_rms_vals[1] / (axis_rms_vals[2] + 1e-6)

        # Calibration-invariant local-baseline dynAmp & robust stats
        out["vib__acc_mag__dyn_amp"][i] = local_baseline_dynamic_amplitude(acc, fs)
        p10, p25, p50, p75, p90, p95 = np.percentile(acc_mag, [10.0, 25.0, 50.0, 75.0, 90.0, 95.0])
        out["vib__acc_mag__p10"][i] = float(p10)
        out["vib__acc_mag__p25"][i] = float(p25)
        out["vib__acc_mag__p75"][i] = float(p75)
        out["vib__acc_mag__p90"][i] = float(p90)
        out["vib__acc_mag__p95"][i] = float(p95)
        out["vib__acc_mag__mad"][i] = float(np.median(np.abs(acc_mag - p50)))
        out["vib__acc_mag__trimmed_mean"][i] = float(_sp_trim_mean(acc_mag, proportiontocut=0.1))
        m_mean = float(np.mean(acc_mag))
        out["vib__acc_mag__cv"][i] = float(np.std(acc_mag) / m_mean) if m_mean > 1e-6 else 0.0

        # Derivative ladder: jerk percentiles, jerk sign-change rate, snap RMS
        jerk = np.diff(acc, axis=0) * fs
        j_mag = np.linalg.norm(jerk, axis=1)
        jp50, jp90, jp95 = np.percentile(j_mag, [50.0, 90.0, 95.0])
        out["vib__jerk_mag__p50"][i] = float(jp50)
        out["vib__jerk_mag__p90"][i] = float(jp90)
        out["vib__jerk_mag__p95"][i] = float(jp95)
        if len(jerk) >= 2:
            # Mean axis-wise sign-change rate of jerk
            out["vib__jerk_sign_change_rate"][i] = float(np.mean(np.sign(jerk[1:]) != np.sign(jerk[:-1])))
            snap = np.diff(jerk, axis=0) * fs
            out["vib__snap_rms"][i] = float(np.sqrt(np.mean(np.sum(snap ** 2, axis=1))))
            out["vib__snap_rms_supported"][i] = 1.0

        # Calibrated Welch PSD shape on dynamic acceleration magnitude
        w_stats = _welch_spectral_shape(acc_mag - np.mean(acc_mag), fs, nperseg=nperseg)
        for k, v in w_stats.items():
            out[f"vib__acc_mag_welch__{k}"][i] = v

    return out


_ROW_GRID, _COL_GRID = np.meshgrid(np.arange(8, dtype=np.float64), np.arange(8, dtype=np.float64), indexing="ij")
_CENTER_MASK = np.zeros((8, 8), dtype=bool)
_CENTER_MASK[2:6, 2:6] = True
_EDGE_MASK = ~_CENTER_MASK


def compute_tof_spatial_features(context: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    """Compute spatial contact geometry, centroids, gradients, blobs & 2x2 block dynamics from raw 5x8x8 ToF grids."""
    if "tof" not in context.get("branches", {}):
        return {}
    tof_arr = np.asarray(context["branches"]["tof"], dtype=np.float64)
    ch_names = context["channel_names"]["tof"]
    if len(ch_names) < 320 or not ch_names[0].startswith("tof_raw_"):
        return {}

    mask = np.asarray(context["masks"]["tof"], dtype=bool)
    present = np.asarray(context["present"][:, 2], dtype=bool)
    fs = float(context["rates"]["tof"])
    thresholds: Sequence[float] = tuple(float(t) for t in config.get("tof_contact_thresholds_mm", (50.0, 100.0, 150.0, 200.0)))
    max_valid_dist = float(config.get("tof_max_valid_mm", 250.0))
    n = len(tof_arr)

    raw_idx = [ch_names.index(f"tof_raw_{p}") for p in range(320)]

    keys = [
        "tof_spatial__closest_mm",
        "tof_spatial__closest_row",
        "tof_spatial__closest_col",
        "tof_spatial__centroid_row_mean",
        "tof_spatial__centroid_col_mean",
        "tof_spatial__centroid_spread_row",
        "tof_spatial__centroid_spread_col",
        "tof_spatial__centroid_displacement",
        "tof_spatial__centroid_velocity",
        "tof_spatial__lr_asymmetry",
        "tof_spatial__tb_asymmetry",
        "tof_spatial__center_to_edge_diff",
        "tof_spatial__spatial_entropy",
        "tof_spatial__sobel_grad_mean",
        "tof_spatial__edge_energy",
        "tof_spatial__n_components_mean",
        "tof_spatial__largest_blob_mean",
        "tof_spatial__fragmentation_mean",
        "tof_spatial__dist_bin_near_frac",
        "tof_spatial__dist_bin_mid_frac",
        "tof_spatial__dist_bin_far_frac",
        "tof_spatial__valid_frac_temporal_grad",
        "tof_spatial__temporal_pixel_diff_mean",
        "tof_spatial__supported",
    ]
    for th in thresholds:
        tag = int(round(th))
        keys.append(f"tof_spatial__active_count_lt{tag}mm")
        keys.append(f"tof_spatial__active_frac_lt{tag}mm")
    for s in range(1, 6):
        keys.extend([
            f"tof_spatial__s{s}_closest_mm",
            f"tof_spatial__s{s}_closest_row",
            f"tof_spatial__s{s}_closest_col",
            f"tof_spatial__s{s}_centroid_row",
            f"tof_spatial__s{s}_centroid_col",
        ])

    out = {k: np.zeros(n, dtype=np.float64) for k in keys}
    if n == 0:
        return out

    contact_th = float(thresholds[1] if len(thresholds) > 1 else thresholds[0])

    for i in range(n):
        L = int(mask[i].sum())
        if L < 2 or not present[i]:
            continue
        grids = tof_arr[i, :L][:, raw_idx].reshape(L, TOF_SHAPE[0], 8, 8)
        # Valid readings lie in [0, max_valid_dist] (excludes far_500/far_255 fills and -1)
        valid = np.isfinite(grids) & (grids >= 0.0) & (grids <= max_valid_dist)
        if not valid.any():
            continue
        out["tof_spatial__supported"][i] = 1.0

        # Temporal pixel difference & temporal gradient of valid pixel fraction
        vf_t = valid.reshape(L, -1).mean(axis=1)
        out["tof_spatial__valid_frac_temporal_grad"][i] = float(np.mean(np.diff(vf_t) * fs))
        both_valid = valid[1:] & valid[:-1]
        if both_valid.any():
            out["tof_spatial__temporal_pixel_diff_mean"][i] = float(np.mean(np.abs(grids[1:][both_valid] - grids[:-1][both_valid])))

        # Time-averaged valid proximity map per sensor (5, 8, 8) and pooled (8, 8)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sensor_mean_map = np.nanmean(np.where(valid, grids, np.nan), axis=0)  # (5, 8, 8)
            pooled_map = np.nanmean(sensor_mean_map, axis=0)  # (8, 8)
        pooled_valid = np.isfinite(pooled_map)

        # Distance-bin fractions across all valid samples in window
        v_vals = grids[valid]
        out["tof_spatial__dist_bin_near_frac"][i] = float(np.mean(v_vals < 80.0))
        out["tof_spatial__dist_bin_mid_frac"][i] = float(np.mean((v_vals >= 80.0) & (v_vals < 160.0)))
        out["tof_spatial__dist_bin_far_frac"][i] = float(np.mean(v_vals >= 160.0))

        # Multi-threshold active-contact counts and fractions
        total_pixels = float(L * 320)
        for th in thresholds:
            tag = int(round(th))
            cnt = float(np.sum(valid & (grids < th)))
            out[f"tof_spatial__active_count_lt{tag}mm"][i] = cnt / float(L)
            out[f"tof_spatial__active_frac_lt{tag}mm"][i] = cnt / total_pixels

        if pooled_valid.any():
            # Closest pixel on pooled map
            safe_pool = np.where(pooled_valid, pooled_map, np.inf)
            flat_min = int(np.argmin(safe_pool))
            r_min, c_min = divmod(flat_min, 8)
            out["tof_spatial__closest_mm"][i] = float(safe_pool[r_min, c_min])
            out["tof_spatial__closest_row"][i] = float(r_min)
            out["tof_spatial__closest_col"][i] = float(c_min)

            # Proximity weights = 1 / (distance + 1.0)
            weights = np.where(pooled_valid, 1.0 / (np.maximum(pooled_map, 0.0) + 1.0), 0.0)
            w_sum = float(weights.sum())
            if w_sum > 1e-12:
                cr = float(np.sum(weights * _ROW_GRID) / w_sum)
                cc = float(np.sum(weights * _COL_GRID) / w_sum)
                sr = float(np.sqrt(np.sum(weights * ((_ROW_GRID - cr) ** 2)) / w_sum))
                sc = float(np.sqrt(np.sum(weights * ((_COL_GRID - cc) ** 2)) / w_sum))
                out["tof_spatial__centroid_row_mean"][i] = cr
                out["tof_spatial__centroid_col_mean"][i] = cc
                out["tof_spatial__centroid_spread_row"][i] = sr
                out["tof_spatial__centroid_spread_col"][i] = sc
                out["tof_spatial__spatial_entropy"][i] = float(_sp_entropy(weights[weights > 0] / w_sum))

            # LR / TB asymmetry and center-to-edge difference
            left_m = float(np.nanmean(pooled_map[:, :4])) if np.isfinite(pooled_map[:, :4]).any() else 0.0
            right_m = float(np.nanmean(pooled_map[:, 4:])) if np.isfinite(pooled_map[:, 4:]).any() else 0.0
            top_m = float(np.nanmean(pooled_map[:4, :])) if np.isfinite(pooled_map[:4, :]).any() else 0.0
            bot_m = float(np.nanmean(pooled_map[4:, :])) if np.isfinite(pooled_map[4:, :]).any() else 0.0
            out["tof_spatial__lr_asymmetry"][i] = (left_m - right_m) / (left_m + right_m + 1e-6)
            out["tof_spatial__tb_asymmetry"][i] = (top_m - bot_m) / (top_m + bot_m + 1e-6)
            cen_m = float(np.nanmean(pooled_map[_CENTER_MASK])) if np.isfinite(pooled_map[_CENTER_MASK]).any() else 0.0
            edg_m = float(np.nanmean(pooled_map[_EDGE_MASK])) if np.isfinite(pooled_map[_EDGE_MASK]).any() else 0.0
            out["tof_spatial__center_to_edge_diff"][i] = cen_m - edg_m

            # Sobel spatial gradient magnitude & edge energy via scipy.ndimage.sobel
            fill_val = float(np.nanmedian(pooled_map[pooled_valid]))
            clean_map = np.where(pooled_valid, pooled_map, fill_val)
            gx = sp_ndimage.sobel(clean_map, axis=0, mode="nearest")
            gy = sp_ndimage.sobel(clean_map, axis=1, mode="nearest")
            gmag = np.hypot(gx, gy)
            out["tof_spatial__sobel_grad_mean"][i] = float(np.mean(gmag))
            out["tof_spatial__edge_energy"][i] = float(np.mean(gmag ** 2))

            # Connected-component blobs via scipy.ndimage.label
            n_comps, max_blobs, frags = [], [], []
            centroids_t = []
            for t in range(L):
                v_t = valid[t]
                g_t = grids[t]
                # Pooled frame proximity map for temporal centroid trajectory
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    pm_t = np.nanmean(np.where(v_t, g_t, np.nan), axis=0)
                pv_t = np.isfinite(pm_t)
                if pv_t.any():
                    wt = np.where(pv_t, 1.0 / (np.maximum(pm_t, 0.0) + 1.0), 0.0)
                    ws = float(wt.sum())
                    if ws > 1e-12:
                        centroids_t.append((float(np.sum(wt * _ROW_GRID) / ws), float(np.sum(wt * _COL_GRID) / ws)))
                    binary = pv_t & (pm_t < contact_th)
                    labeled, n_feat = sp_ndimage.label(binary)
                    n_comps.append(float(n_feat))
                    if n_feat > 0:
                        sizes = np.bincount(labeled.ravel())[1:]
                        max_blobs.append(float(sizes.max()))
                        frags.append(float(n_feat) / float(max(1, int(binary.sum()))))
                    else:
                        max_blobs.append(0.0)
                        frags.append(0.0)
            if n_comps:
                out["tof_spatial__n_components_mean"][i] = float(np.mean(n_comps))
                out["tof_spatial__largest_blob_mean"][i] = float(np.mean(max_blobs))
                out["tof_spatial__fragmentation_mean"][i] = float(np.mean(frags))
            if len(centroids_t) >= 2:
                ct = np.asarray(centroids_t, dtype=np.float64)
                steps = np.linalg.norm(np.diff(ct, axis=0), axis=1)
                out["tof_spatial__centroid_displacement"][i] = float(np.linalg.norm(ct[-1] - ct[0]))
                out["tof_spatial__centroid_velocity"][i] = float(np.mean(steps) * fs)

        # Per-sensor closest pixel & weighted centroid
        for s in range(5):
            sm = sensor_mean_map[s]
            sv = np.isfinite(sm)
            if sv.any():
                safe_s = np.where(sv, sm, np.inf)
                r_s, c_s = divmod(int(np.argmin(safe_s)), 8)
                out[f"tof_spatial__s{s + 1}_closest_mm"][i] = float(safe_s[r_s, c_s])
                out[f"tof_spatial__s{s + 1}_closest_row"][i] = float(r_s)
                out[f"tof_spatial__s{s + 1}_closest_col"][i] = float(c_s)
                ws_map = np.where(sv, 1.0 / (np.maximum(sm, 0.0) + 1.0), 0.0)
                ws_sum = float(ws_map.sum())
                if ws_sum > 1e-12:
                    out[f"tof_spatial__s{s + 1}_centroid_row"][i] = float(np.sum(ws_map * _ROW_GRID) / ws_sum)
                    out[f"tof_spatial__s{s + 1}_centroid_col"][i] = float(np.sum(ws_map * _COL_GRID) / ws_sum)

    return out


def compute_tof_block_channels(
    tof_raw_5x64: np.ndarray,
    *,
    block_size: int = 2,
    max_valid_dist: float = 250.0,
    include_temporal_diff: bool = True,
) -> Tuple[np.ndarray, list[str]]:
    """Aggregate (T, 320) raw ToF grids into non-overlapping ``block_size x block_size`` mean maps.

    For ``block_size=2`` (default): 5 sensors x 4x4 = 80 spatial block means (+ 80 frame diffs
    when ``include_temporal_diff=True``).
    For ``block_size=4``: 5 sensors x 2x2 = 20 spatial block means (+ 20 frame diffs).
    Invalid values (< 0, NaN, or > ``max_valid_dist``) are excluded from block averages and
    imputed with ``max_valid_dist`` only when an entire block is invalid.
    """
    if block_size not in (2, 4):
        raise ValueError(f"block_size must be 2 or 4, got {block_size}")
    arr = np.asarray(tof_raw_5x64, dtype=np.float64)
    t_len = len(arr)
    g = 8 // block_size
    n_blocks = TOF_SHAPE[0] * g * g
    names = [
        f"tof_b{block_size}_s{s + 1}_r{r}_c{c}"
        for s in range(TOF_SHAPE[0])
        for r in range(g)
        for c in range(g)
    ]
    if include_temporal_diff:
        names = names + [f"{nm}_diff" for nm in names]
    if t_len == 0:
        return np.zeros((0, len(names)), dtype=np.float64), names

    grids = arr.reshape(t_len, TOF_SHAPE[0], 8, 8)
    valid = np.isfinite(grids) & (grids >= 0.0) & (grids <= float(max_valid_dist))
    blocks_v = np.where(valid, grids, np.nan).reshape(t_len, TOF_SHAPE[0], g, block_size, g, block_size)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        block_means = np.nanmean(blocks_v, axis=(3, 5)).reshape(t_len, n_blocks)
    block_means = np.where(np.isfinite(block_means), block_means, float(max_valid_dist))
    if not include_temporal_diff:
        return block_means, names

    diffs = np.zeros_like(block_means)
    if t_len >= 2:
        diffs[1:] = block_means[1:] - block_means[:-1]
    return np.concatenate([block_means, diffs], axis=1), names


def _extract_sensor_means_from_branch(
    tof_win: np.ndarray,
    ch_names: list[str],
    max_valid_dist: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(overall_mean[L], sensor_means[L, 5])`` with NaN where invalid."""
    L = len(tof_win)
    sensor_means = np.full((L, 5), np.nan, dtype=np.float64)

    if all(f"tof_s{s}_mean" in ch_names for s in range(1, 6)):
        idxs = [ch_names.index(f"tof_s{s}_mean") for s in range(1, 6)]
        vals = tof_win[:, idxs]
        ok = np.isfinite(vals) & (vals >= 0.0) & (vals <= max_valid_dist)
        sensor_means = np.where(ok, vals, np.nan)
    elif all(f"tof_sensor_mean_{s}" in ch_names for s in range(1, 6)):
        idxs = [ch_names.index(f"tof_sensor_mean_{s}") for s in range(1, 6)]
        vals = tof_win[:, idxs]
        ok = np.isfinite(vals) & (vals >= 0.0) & (vals <= max_valid_dist)
        sensor_means = np.where(ok, vals, np.nan)
    elif len(ch_names) >= 320 and "tof_raw_0" in ch_names:
        raw_idx = [ch_names.index(f"tof_raw_{p}") for p in range(320)]
        grids = tof_win[:, raw_idx].reshape(L, 5, 64)
        ok = np.isfinite(grids) & (grids >= 0.0) & (grids <= max_valid_dist)
        with np.errstate(invalid="ignore"):
            sensor_means = np.nanmean(np.where(ok, grids, np.nan), axis=2)

    if "tof_all_mean" in ch_names:
        idx_all = ch_names.index("tof_all_mean")
        v_all = tof_win[:, idx_all]
        ok_all = np.isfinite(v_all) & (v_all >= 0.0) & (v_all <= max_valid_dist)
        overall = np.where(ok_all, v_all, np.nan)
    elif "tof_mean" in ch_names:
        idx_all = ch_names.index("tof_mean")
        v_all = tof_win[:, idx_all]
        ok_all = np.isfinite(v_all) & (v_all >= 0.0) & (v_all <= max_valid_dist)
        overall = np.where(ok_all, v_all, np.nan)
    else:
        with np.errstate(invalid="ignore"):
            overall = np.nanmean(sensor_means, axis=1)
    return overall, sensor_means


def _fill_1d_linear(series: np.ndarray) -> Tuple[np.ndarray, int]:
    """Interpolate internal NaNs linearly without leaking across windows."""
    valid = np.isfinite(series)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return np.zeros_like(series), 0
    if n_valid == len(series):
        return series.copy(), n_valid
    idx = np.arange(len(series), dtype=np.float64)
    return np.interp(idx, idx[valid], series[valid]), n_valid


def compute_tof_temporal_features(context: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    """Compute ToF approach/recession velocities, jerk, approach onsets, time-to-closest, and dwell."""
    if "tof" not in context.get("branches", {}):
        return {}
    tof_arr = np.asarray(context["branches"]["tof"], dtype=np.float64)
    mask = np.asarray(context["masks"]["tof"], dtype=bool)
    present = np.asarray(context["present"][:, 2], dtype=bool)
    ch_names = list(context["channel_names"]["tof"])
    fs = float(context["rates"]["tof"])
    max_valid_dist = float(config.get("tof_max_valid_mm", 250.0))
    onset_thresh = float(config.get("tof_approach_onset_mm_s", 20.0))
    dwell_band = float(config.get("tof_closest_dwell_band_mm", 15.0))
    n = len(tof_arr)

    keys = [
        "tof_temp__mean_vel_mm_s",
        "tof_temp__approach_vel_max_mm_s",
        "tof_temp__recession_vel_max_mm_s",
        "tof_temp__vel_std_mm_s",
        "tof_temp__jerk_rms_mm_s2",
        "tof_temp__approach_onset_count",
        "tof_temp__time_to_closest_s",
        "tof_temp__time_to_closest_frac",
        "tof_temp__dwell_closest_s",
        "tof_temp__dwell_closest_frac",
        "tof_temp__supported",
    ]
    for s in range(1, 6):
        keys.extend([f"tof_temp__s{s}_vel_mean_mm_s", f"tof_temp__s{s}_approach_max_mm_s"])

    out = {k: np.zeros(n, dtype=np.float64) for k in keys}
    if n == 0:
        return out

    for i in range(n):
        L = int(mask[i].sum())
        if L < 3 or not present[i]:
            continue
        overall_raw, sensor_raw = _extract_sensor_means_from_branch(tof_arr[i, :L], ch_names, max_valid_dist)
        overall, n_valid = _fill_1d_linear(overall_raw)
        if n_valid < 3:
            continue
        out["tof_temp__supported"][i] = 1.0

        # Median-smoothed trajectory via scipy.signal.medfilt for robust velocity/jerk
        k = 3 if L >= 3 else 1
        smoothed = sps.medfilt(overall, kernel_size=k) if k >= 3 else overall
        vel = np.diff(smoothed) * fs  # negative = approaching, positive = receding
        out["tof_temp__mean_vel_mm_s"][i] = float(np.mean(vel))
        out["tof_temp__approach_vel_max_mm_s"][i] = float(max(0.0, -np.min(vel)))
        out["tof_temp__recession_vel_max_mm_s"][i] = float(max(0.0, np.max(vel)))
        out["tof_temp__vel_std_mm_s"][i] = float(np.std(vel))

        if len(vel) >= 2:
            acc_d = np.diff(vel) * fs
            out["tof_temp__jerk_rms_mm_s2"][i] = float(np.sqrt(np.mean(acc_d * acc_d)))

        # Count distinct approach onset transitions (vel crosses below -onset_thresh)
        approaching = vel < -onset_thresh
        onsets = int(approaching[0]) + int(np.sum(approaching[1:] & (~approaching[:-1]))) if len(approaching) else 0
        out["tof_temp__approach_onset_count"][i] = float(onsets)

        # Time to closest and dwell duration near closest distance
        closest_idx = int(np.argmin(smoothed))
        closest_val = float(smoothed[closest_idx])
        out["tof_temp__time_to_closest_s"][i] = float(closest_idx) / fs
        out["tof_temp__time_to_closest_frac"][i] = float(closest_idx) / float(max(1, L - 1))
        near_mask = smoothed <= (closest_val + dwell_band)
        dwell_samples = int(np.sum(near_mask))
        out["tof_temp__dwell_closest_s"][i] = float(dwell_samples) / fs
        out["tof_temp__dwell_closest_frac"][i] = float(dwell_samples) / float(L)

        # Per-sensor mean derivative and peak approach speed
        for s in range(5):
            s_arr, s_ok = _fill_1d_linear(sensor_raw[:, s])
            if s_ok >= 3:
                s_sm = sps.medfilt(s_arr, kernel_size=k) if k >= 3 else s_arr
                s_vel = np.diff(s_sm) * fs
                out[f"tof_temp__s{s + 1}_vel_mean_mm_s"][i] = float(np.mean(s_vel))
                out[f"tof_temp__s{s + 1}_approach_max_mm_s"][i] = float(max(0.0, -np.min(s_vel)))

    return out


_TRAPEZOID = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def _compute_baseline(
    values: np.ndarray,
    mode: str,
    first_n: int,
    seq_baseline: float | None = None,
) -> np.ndarray:
    """Return a 1-D baseline series of length ``len(values)`` with zero leakage."""
    L = len(values)
    m = str(mode or "first_n_samples").strip().lower()
    if m == "sequence_median" and seq_baseline is not None and np.isfinite(seq_baseline):
        return np.full(L, float(seq_baseline), dtype=np.float64)
    if m == "sequence_percentile" and seq_baseline is not None and np.isfinite(seq_baseline):
        return np.full(L, float(seq_baseline), dtype=np.float64)
    if m == "sequence_median":
        return np.full(L, float(np.median(values)), dtype=np.float64)
    if m == "sequence_percentile":
        return np.full(L, float(np.percentile(values, 10.0)), dtype=np.float64)
    if m == "rolling":
        k = max(3, min(L, first_n))
        # Causal rolling baseline (cumulative/trailing mean so future samples never leak backward)
        cs = np.cumsum(np.insert(values, 0, 0.0))
        idx = np.arange(L)
        lo = np.maximum(0, idx - k + 1)
        counts = (idx - lo + 1).astype(np.float64)
        return (cs[idx + 1] - cs[lo]) / counts
    # Default: first_n_samples
    k = max(1, min(L, int(first_n)))
    return np.full(L, float(np.mean(values[:k])), dtype=np.float64)


def compute_thermal_features(context: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    """Compute baseline-relative thermal stats, flux integrals, rise/fall/recovery times & spatial gradients."""
    if "thm" not in context.get("branches", {}):
        return {}
    thm_arr = np.asarray(context["branches"]["thm"], dtype=np.float64)
    mask = np.asarray(context["masks"]["thm"], dtype=bool)
    present = np.asarray(context["present"][:, 1], dtype=bool)
    ch_names = list(context["channel_names"]["thm"])
    fs = float(context["rates"]["thm"])
    dt = 1.0 / fs
    baseline_mode = str(config.get("thermal_baseline_mode", "first_n_samples"))
    baseline_n = int(config.get("thermal_baseline_samples", max(3, int(round(0.25 * fs)))))
    event_prom = float(config.get("thermal_event_prominence_c", 0.25))
    seq_baselines = context.get("thm_seq_baseline")
    n = len(thm_arr)

    sensor_cols = [c for c in ("thm_1", "thm_2", "thm_3", "thm_4", "thm_5") if c in ch_names]
    if not sensor_cols:
        sensor_cols = [c for c in ("thm_raw_1", "thm_raw_2", "thm_raw_3", "thm_raw_4", "thm_raw_5") if c in ch_names]
    reconstruct_from_centered = False
    if not sensor_cols:
        centered_cols = [c for c in ("thm_centered_1", "thm_centered_2", "thm_centered_3", "thm_centered_4", "thm_centered_5") if c in ch_names]
        if centered_cols and "thm_mean" in ch_names:
            sensor_cols = centered_cols
            reconstruct_from_centered = True
    if not sensor_cols:
        if "thm_mean" in ch_names:
            sensor_cols = ["thm_mean"]
        else:
            return {}
    sensor_idx = [ch_names.index(c) for c in sensor_cols]
    thm_mean_idx = ch_names.index("thm_mean") if "thm_mean" in ch_names else None

    keys = [
        "thm_dyn__mean_delta_c",
        "thm_dyn__max_delta_c",
        "thm_dyn__min_delta_c",
        "thm_dyn__std_delta_c",
        "thm_dyn__max_pos_dt_c_s",
        "thm_dyn__max_neg_dt_c_s",
        "thm_dyn__mean_abs_dt_c_s",
        "thm_dyn__pos_flux_c_s",
        "thm_dyn__neg_flux_c_s",
        "thm_dyn__net_flux_c_s",
        "thm_dyn__rise_time_s",
        "thm_dyn__fall_time_s",
        "thm_dyn__recovery_time_s",
        "thm_dyn__event_count",
        "thm_dyn__spatial_spread_mean_c",
        "thm_dyn__spatial_std_mean_c",
        "thm_dyn__spatial_max_pair_grad_c",
        "thm_dyn__spatial_1d_grad_mean_c",
        "thm_dyn__hottest_sensor_idx",
        "thm_dyn__hottest_sensor_delta_c",
        "thm_dyn__supported",
    ]
    for s in range(1, len(sensor_cols) + 1):
        keys.extend([f"thm_dyn__s{s}_delta_mean_c", f"thm_dyn__s{s}_dt_mean_c_s"])

    out = {k: np.zeros(n, dtype=np.float64) for k in keys}
    if n == 0:
        return out

    for i in range(n):
        L = int(mask[i].sum())
        if L < 3 or not present[i]:
            continue
        seg = thm_arr[i, :L][:, sensor_idx]
        if reconstruct_from_centered and thm_mean_idx is not None:
            seg = seg + thm_arr[i, :L, thm_mean_idx : thm_mean_idx + 1]
        if not np.isfinite(seg).any():
            continue
        out["thm_dyn__supported"][i] = 1.0

        # Replace isolated non-finite samples within window safely
        if not np.isfinite(seg).all():
            med = np.nanmedian(seg, axis=0)
            med = np.where(np.isfinite(med), med, 0.0)
            seg = np.where(np.isfinite(seg), seg, med[None, :])

        mean_t = np.mean(seg, axis=1)
        seq_b = float(seq_baselines[i]) if seq_baselines is not None and i < len(seq_baselines) else None
        base = _compute_baseline(mean_t, baseline_mode, baseline_n, seq_baseline=seq_b)
        delta = mean_t - base

        out["thm_dyn__mean_delta_c"][i] = float(np.mean(delta))
        out["thm_dyn__max_delta_c"][i] = float(np.max(delta))
        out["thm_dyn__min_delta_c"][i] = float(np.min(delta))
        out["thm_dyn__std_delta_c"][i] = float(np.std(delta))

        dT_dt = np.diff(mean_t) * fs
        out["thm_dyn__max_pos_dt_c_s"][i] = float(max(0.0, np.max(dT_dt)))
        out["thm_dyn__max_neg_dt_c_s"][i] = float(max(0.0, -np.min(dT_dt)))
        out["thm_dyn__mean_abs_dt_c_s"][i] = float(np.mean(np.abs(dT_dt)))

        # Calibrated flux integrals (degC * s) via trapezoid rule
        out["thm_dyn__pos_flux_c_s"][i] = float(_TRAPEZOID(np.maximum(delta, 0.0), dx=dt))
        out["thm_dyn__neg_flux_c_s"][i] = float(_TRAPEZOID(np.maximum(-delta, 0.0), dx=dt))
        out["thm_dyn__net_flux_c_s"][i] = float(_TRAPEZOID(delta, dx=dt))

        # Rise, fall, and half-peak recovery times (s)
        peak_idx = int(np.argmax(delta))
        peak_val = float(delta[peak_idx])
        out["thm_dyn__rise_time_s"][i] = float(peak_idx) * dt
        out["thm_dyn__fall_time_s"][i] = float(max(0, L - 1 - peak_idx)) * dt
        if peak_val > 1e-6 and peak_idx < L - 1:
            after = delta[peak_idx:]
            rec_hits = np.where(after <= 0.5 * peak_val)[0]
            if len(rec_hits) > 0:
                out["thm_dyn__recovery_time_s"][i] = float(rec_hits[0]) * dt
            else:
                out["thm_dyn__recovery_time_s"][i] = float(L - 1 - peak_idx) * dt

        # Thermal contact event count via scipy.signal.find_peaks
        peaks, _ = sps.find_peaks(delta, prominence=event_prom, distance=max(1, int(round(0.25 * fs))))
        out["thm_dyn__event_count"][i] = float(len(peaks))

        # Spatial spread & gradients across the 5 thermopile sensors
        s_spread = np.max(seg, axis=1) - np.min(seg, axis=1)
        out["thm_dyn__spatial_spread_mean_c"][i] = float(np.mean(s_spread))
        out["thm_dyn__spatial_std_mean_c"][i] = float(np.mean(np.std(seg, axis=1)))
        out["thm_dyn__spatial_max_pair_grad_c"][i] = float(np.max(s_spread))
        if seg.shape[1] >= 2:
            out["thm_dyn__spatial_1d_grad_mean_c"][i] = float(np.mean(np.abs(np.diff(seg, axis=1))))

        # Per-sensor delta and hottest sensor
        sensor_deltas = np.zeros(seg.shape[1], dtype=np.float64)
        for s in range(seg.shape[1]):
            sb = _compute_baseline(seg[:, s], baseline_mode, baseline_n, seq_baseline=None)
            sd = seg[:, s] - sb
            sensor_deltas[s] = float(np.mean(sd))
            out[f"thm_dyn__s{s + 1}_delta_mean_c"][i] = sensor_deltas[s]
            out[f"thm_dyn__s{s + 1}_dt_mean_c_s"][i] = float(np.mean(np.diff(seg[:, s]) * fs))
        hot_idx = int(np.argmax(sensor_deltas))
        out["thm_dyn__hottest_sensor_idx"][i] = float(hot_idx + 1)
        out["thm_dyn__hottest_sensor_delta_c"][i] = float(sensor_deltas[hot_idx])

    return out


def _align_pair_to_common_grid(
    x: np.ndarray,
    fs_x: float,
    y: np.ndarray,
    fs_y: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Resample ``(x, y)`` onto a shared uniform time grid at ``min(fs_x, fs_y)`` over overlapping duration."""
    if len(x) < 2 or len(y) < 2:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), float(min(fs_x, fs_y))
    dur_x = (len(x) - 1) / float(fs_x)
    dur_y = (len(y) - 1) / float(fs_y)
    dur = min(dur_x, dur_y)
    fs_c = float(min(fs_x, fs_y))
    n_c = int(np.floor(dur * fs_c)) + 1
    if n_c < 4:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), fs_c
    t_c = np.arange(n_c, dtype=np.float64) / fs_c
    t_x = np.arange(len(x), dtype=np.float64) / float(fs_x)
    t_y = np.arange(len(y), dtype=np.float64) / float(fs_y)
    return np.interp(t_c, t_x, x), np.interp(t_c, t_y, y), fs_c


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or float(np.std(a)) <= 1e-10 or float(np.std(b)) <= 1e-10:
        return 0.0
    c = float(np.corrcoef(a, b)[0, 1])
    return float(np.clip(c, -1.0, 1.0)) if np.isfinite(c) else 0.0


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or float(np.std(a)) <= 1e-10 or float(np.std(b)) <= 1e-10:
        return 0.0
    res = _sp_spearmanr(a, b)
    c = float(res.statistic) if hasattr(res, "statistic") else float(res[0])
    return float(np.clip(c, -1.0, 1.0)) if np.isfinite(c) else 0.0


def _lagged_correlation_summary(
    a: np.ndarray,
    b: np.ndarray,
    fs: float,
    lags_s: Sequence[float] = (-0.2, -0.1, 0.0, 0.1, 0.2),
) -> Dict[str, float]:
    """Compute Pearson, Spearman, and time-lagged correlations in physical seconds.

    Positive lag ``tau > 0`` tests ``corr(a[t], b[t + tau])`` (``a`` leads ``b``).
    """
    n = len(a)
    res: Dict[str, float] = {
        "pearson_lag0": _safe_pearson(a, b),
        "spearman_lag0": _safe_spearman(a, b),
        "max_abs_lag_corr": 0.0,
        "best_lag_s": 0.0,
        "lead_lag_sign": 0.0,
        "peak_prominence": 0.0,
    }
    vals = []
    used_lags = []
    for tau in lags_s:
        step = int(round(float(tau) * float(fs)))
        if abs(step) >= n - 3:
            c = 0.0
        elif step > 0:
            c = _safe_pearson(a[:-step], b[step:])
        elif step < 0:
            s = -step
            c = _safe_pearson(a[s:], b[:-s])
        else:
            c = res["pearson_lag0"]
        vals.append(c)
        used_lags.append(float(tau))

    if vals:
        abs_v = np.abs(np.asarray(vals, dtype=np.float64))
        best_i = int(np.argmax(abs_v))
        res["max_abs_lag_corr"] = float(abs_v[best_i])
        res["best_lag_s"] = float(used_lags[best_i])
        res["lead_lag_sign"] = float(np.sign(used_lags[best_i])) if abs_v[best_i] > 1e-6 else 0.0
        res["peak_prominence"] = float(abs_v[best_i] - np.median(abs_v))
    return res


def _coherence_summary(a: np.ndarray, b: np.ndarray, fs: float) -> Tuple[float, float]:
    """Mean (0.5-4 Hz) and peak magnitude-squared coherence via ``scipy.signal.coherence``."""
    if len(a) < 8 or float(np.std(a)) <= 1e-10 or float(np.std(b)) <= 1e-10:
        return 0.0, 0.0
    nperseg = min(len(a), max(8, int(round(fs))))
    try:
        freqs, coh = sps.coherence(a, b, fs=float(fs), nperseg=nperseg)
    except Exception:
        return 0.0, 0.0
    coh = np.nan_to_num(coh, nan=0.0, posinf=0.0, neginf=0.0)
    band = (freqs >= 0.5) & (freqs <= min(4.0, float(fs) / 2.0))
    mean_band = float(np.mean(coh[band])) if band.any() else float(np.mean(coh))
    peak_coh = float(np.max(coh[1:])) if len(coh) > 1 else 0.0
    return mean_band, peak_coh


def compute_cross_modal_features(context: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    """Compute cross-modal lagged correlations, spectral coherence, conditional contact stats & coupling ratios."""
    branches = context.get("branches", {})
    masks = context.get("masks", {})
    ch_names_map = context.get("channel_names", {})
    rates = context.get("rates", {})
    present = np.asarray(context["present"], dtype=bool)

    imu = np.asarray(branches["imu"], dtype=np.float64)
    imu_mask = np.asarray(masks["imu"], dtype=bool)
    imu_ch = list(ch_names_map["imu"])
    fs_imu = float(rates["imu"])
    n = len(imu)

    has_thm_branch = "thm" in branches and present.shape[1] >= 2
    has_tof_branch = "tof" in branches and present.shape[1] >= 3
    max_valid_dist = float(config.get("tof_max_valid_mm", 250.0))
    lags_s = tuple(float(v) for v in config.get("cross_modal_lags_s", (-0.2, -0.1, 0.0, 0.1, 0.2)))

    pair_fields = ("pearson_lag0", "spearman_lag0", "max_abs_lag_corr", "best_lag_s", "lead_lag_sign", "peak_prominence")
    keys = [
        "xmodal__acc_to_ang_energy_ratio",
        "xmodal__lin_to_acc_energy_ratio",
    ]
    if has_tof_branch:
        keys.extend(f"xmodal__imu_tof__{f}" for f in pair_fields)
        keys.extend(f"xmodal__jerk_tofvel__{f}" for f in pair_fields)
        keys.extend([
            "xmodal__imu_tof__coh_band_mean",
            "xmodal__imu_tof__coh_peak",
            "xmodal__acc_mag_when_tof_close100",
            "xmodal__acc_mag_when_tof_close200",
            "xmodal__acc_mag_when_tof_far200",
            "xmodal__acc_mag_close_minus_far",
            "xmodal__tof_close100_dwell_s",
            "xmodal__tof_close100_transitions",
            "xmodal__tof_approach_to_jerk_ratio",
            "xmodal__imu_tof_supported",
        ])
    if has_thm_branch:
        keys.extend(f"xmodal__imu_thm__{f}" for f in pair_fields)
        keys.extend([
            "xmodal__imu_thm__coh_band_mean",
            "xmodal__imu_thm__coh_peak",
            "xmodal__imu_thm_supported",
        ])
    if has_tof_branch and has_thm_branch:
        keys.extend(f"xmodal__tof_thm__{f}" for f in pair_fields)
        keys.extend([
            "xmodal__thm_when_tof_close100",
            "xmodal__thm_when_tof_far200",
            "xmodal__thm_close_minus_far",
            "xmodal__thm_slope_before_close_c_s",
            "xmodal__thm_slope_after_close_c_s",
            "xmodal__thm_slope_change_at_close_c_s",
            "xmodal__tof_thm_supported",
        ])
    out = {k: np.zeros(n, dtype=np.float64) for k in keys}
    if n == 0:
        return out

    acc_idx = [imu_ch.index(f"acc_{a}") for a in "xyz"] if all(f"acc_{a}" in imu_ch for a in "xyz") else None
    lin_idx = [imu_ch.index(f"lin_{a}") for a in "xyz"] if all(f"lin_{a}" in imu_ch for a in "xyz") else None
    ang_idx = [imu_ch.index(f"ang_{a}") for a in "xyz"] if all(f"ang_{a}" in imu_ch for a in "xyz") else None
    if acc_idx is None:
        return out

    thm_arr = np.asarray(branches["thm"], dtype=np.float64) if has_thm_branch else None
    thm_mask = np.asarray(masks["thm"], dtype=bool) if has_thm_branch else None
    thm_ch = list(ch_names_map["thm"]) if has_thm_branch else []
    thm_sensor_idx = (
        [thm_ch.index("thm_mean")] if "thm_mean" in thm_ch
        else [thm_ch.index(c) for c in ("thm_1", "thm_2", "thm_3", "thm_4", "thm_5") if c in thm_ch]
        or [thm_ch.index(c) for c in ("thm_raw_1", "thm_raw_2", "thm_raw_3", "thm_raw_4", "thm_raw_5") if c in thm_ch]
    )
    fs_thm = float(rates.get("thm", fs_imu))

    tof_arr = np.asarray(branches["tof"], dtype=np.float64) if has_tof_branch else None
    tof_mask = np.asarray(masks["tof"], dtype=bool) if has_tof_branch else None
    tof_ch = list(ch_names_map["tof"]) if has_tof_branch else []
    fs_tof = float(rates.get("tof", fs_imu))

    for i in range(n):
        L_imu = int(imu_mask[i].sum())
        if L_imu < 4:
            continue
        acc = imu[i, :L_imu][:, acc_idx]
        acc_mag = np.linalg.norm(acc, axis=1)
        acc_dyn = acc_mag - np.mean(acc_mag)
        jerk_mag = np.linalg.norm(np.diff(acc, axis=0) * fs_imu, axis=1)
        jerk_series = np.concatenate([[0.0], jerk_mag])

        # Generalized IMU internal coupling ratios
        acc_e = float(np.mean(acc_dyn * acc_dyn))
        if ang_idx is not None:
            ang_mag = np.linalg.norm(imu[i, :L_imu][:, ang_idx], axis=1)
            ang_e = float(np.mean(ang_mag * ang_mag))
            out["xmodal__acc_to_ang_energy_ratio"][i] = acc_e / (ang_e + 1e-6)
        if lin_idx is not None:
            lin_mag = np.linalg.norm(imu[i, :L_imu][:, lin_idx], axis=1)
            lin_e = float(np.mean(lin_mag * lin_mag))
            out["xmodal__lin_to_acc_energy_ratio"][i] = lin_e / (float(np.mean(acc_mag * acc_mag)) + 1e-6)

        # Extract THM mean trajectory if present
        thm_mean = None
        if thm_arr is not None and thm_mask is not None and thm_sensor_idx and present[i, 1]:
            L_thm = int(thm_mask[i].sum())
            if L_thm >= 4:
                seg_t = thm_arr[i, :L_thm][:, thm_sensor_idx]
                if np.isfinite(seg_t).any():
                    with np.errstate(invalid="ignore"):
                        tm = np.nanmean(seg_t, axis=1)
                    tm_fill, n_ok_t = _fill_1d_linear(tm)
                    if n_ok_t >= 4:
                        thm_mean = tm_fill

        # Extract ToF distance & proximity trajectories if present
        tof_dist = None
        if tof_arr is not None and tof_mask is not None and present[i, 2]:
            L_tof = int(tof_mask[i].sum())
            if L_tof >= 4:
                overall_raw, _ = _extract_sensor_means_from_branch(tof_arr[i, :L_tof], tof_ch, max_valid_dist)
                td_fill, n_ok_f = _fill_1d_linear(overall_raw)
                if n_ok_f >= 4:
                    tof_dist = td_fill

        # 1. IMU <-> ToF synchronization, coherence & conditional statistics
        if tof_dist is not None:
            tof_prox = -tof_dist  # higher = closer
            a_al, p_al, fs_c = _align_pair_to_common_grid(acc_mag, fs_imu, tof_prox, fs_tof)
            d_al = -p_al
            if len(a_al) >= 4:
                out["xmodal__imu_tof_supported"][i] = 1.0
                stats = _lagged_correlation_summary(a_al, p_al, fs_c, lags_s)
                for f, v in stats.items():
                    out[f"xmodal__imu_tof__{f}"][i] = v
                cb, cp = _coherence_summary(a_al - np.mean(a_al), p_al - np.mean(p_al), fs_c)
                out["xmodal__imu_tof__coh_band_mean"][i] = cb
                out["xmodal__imu_tof__coh_peak"][i] = cp

                # Jerk <-> ToF approach speed
                tof_app_vel = np.concatenate([[0.0], -np.diff(tof_dist) * fs_tof])
                j_al, v_al, fs_jv = _align_pair_to_common_grid(jerk_series, fs_imu, tof_app_vel, fs_tof)
                if len(j_al) >= 4:
                    jv_stats = _lagged_correlation_summary(j_al, v_al, fs_jv, lags_s)
                    for f, v in jv_stats.items():
                        out[f"xmodal__jerk_tofvel__{f}"][i] = v
                    out["xmodal__tof_approach_to_jerk_ratio"][i] = float(np.mean(np.maximum(v_al, 0.0))) / (
                        float(np.mean(j_al)) + 1e-6
                    )

                # Conditional IMU statistics on ToF proximity regimes
                c100 = d_al < 100.0
                c200 = d_al < 200.0
                f200 = ~c200
                m_c100 = float(np.mean(a_al[c100])) if c100.any() else 0.0
                m_c200 = float(np.mean(a_al[c200])) if c200.any() else 0.0
                m_f200 = float(np.mean(a_al[f200])) if f200.any() else 0.0
                out["xmodal__acc_mag_when_tof_close100"][i] = m_c100
                out["xmodal__acc_mag_when_tof_close200"][i] = m_c200
                out["xmodal__acc_mag_when_tof_far200"][i] = m_f200
                if c200.any() and f200.any():
                    out["xmodal__acc_mag_close_minus_far"][i] = m_c200 - m_f200

                # Close-approach dwell & transitions
                out["xmodal__tof_close100_dwell_s"][i] = float(np.sum(c100)) / fs_c
                trans = int(np.sum(c100[1:] != c100[:-1])) if len(c100) >= 2 else 0
                out["xmodal__tof_close100_transitions"][i] = float(trans)

        # 2. IMU <-> THM synchronization & coherence
        if thm_mean is not None:
            a_al, t_al, fs_c = _align_pair_to_common_grid(acc_mag, fs_imu, thm_mean, fs_thm)
            if len(a_al) >= 4:
                out["xmodal__imu_thm_supported"][i] = 1.0
                stats = _lagged_correlation_summary(a_al, t_al, fs_c, lags_s)
                for f, v in stats.items():
                    out[f"xmodal__imu_thm__{f}"][i] = v
                cb, cp = _coherence_summary(a_al - np.mean(a_al), t_al - np.mean(t_al), fs_c)
                out["xmodal__imu_thm__coh_band_mean"][i] = cb
                out["xmodal__imu_thm__coh_peak"][i] = cp

        # 3. ToF <-> THM synchronization & before/after closest-approach thermal slope
        if tof_dist is not None and thm_mean is not None:
            p_al, t_al, fs_c = _align_pair_to_common_grid(-tof_dist, fs_tof, thm_mean, fs_thm)
            d_al = -p_al
            if len(p_al) >= 4:
                out["xmodal__tof_thm_supported"][i] = 1.0
                stats = _lagged_correlation_summary(p_al, t_al, fs_c, lags_s)
                for f, v in stats.items():
                    out[f"xmodal__tof_thm__{f}"][i] = v

                c100 = d_al < 100.0
                f200 = d_al >= 200.0
                mt_c = float(np.mean(t_al[c100])) if c100.any() else 0.0
                mt_f = float(np.mean(t_al[f200])) if f200.any() else 0.0
                out["xmodal__thm_when_tof_close100"][i] = mt_c
                out["xmodal__thm_when_tof_far200"][i] = mt_f
                if c100.any() and f200.any():
                    out["xmodal__thm_close_minus_far"][i] = mt_c - mt_f

                # Thermal slope 0.5 s before vs 0.5 s after closest approach
                idx_min = int(np.argmin(d_al))
                half_n = max(2, int(round(0.5 * fs_c)))
                pre = t_al[max(0, idx_min - half_n): idx_min + 1]
                post = t_al[idx_min: min(len(t_al), idx_min + half_n + 1)]
                slope_pre = float(np.mean(np.diff(pre) * fs_c)) if len(pre) >= 2 else 0.0
                slope_post = float(np.mean(np.diff(post) * fs_c)) if len(post) >= 2 else 0.0
                out["xmodal__thm_slope_before_close_c_s"][i] = slope_pre
                out["xmodal__thm_slope_after_close_c_s"][i] = slope_post
                out["xmodal__thm_slope_change_at_close_c_s"][i] = slope_post - slope_pre

    return out


def _hjorth_parameters(x: np.ndarray, fs: float) -> Tuple[float, float, float]:
    """Hjorth Activity (variance), Mobility (rad-scaled derivative std ratio), and Complexity."""
    a = np.asarray(x, dtype=np.float64)
    if len(a) < 4:
        return 0.0, 0.0, 0.0
    var0 = float(np.var(a))
    if var0 <= 1e-12:
        return 0.0, 0.0, 0.0
    d1 = np.diff(a) * float(fs)
    var1 = float(np.var(d1))
    mob = float(np.sqrt(var1 / var0)) if var1 > 1e-12 else 0.0
    if len(d1) < 3 or var1 <= 1e-12:
        return var0, mob, 0.0
    d2 = np.diff(d1) * float(fs)
    var2 = float(np.var(d2))
    mob_d1 = float(np.sqrt(var2 / var1)) if var2 > 1e-12 else 0.0
    comp = mob_d1 / mob if mob > 1e-12 else 0.0
    return var0, mob, comp


def _dwt_and_packet_features(x: np.ndarray, wavelet: str = "db4") -> Dict[str, float]:
    """DWT detail energy ratios (4 levels) + WaveletPacket level-2 energy entropy via ``pywt``."""
    out = {
        "dwt_approx_ratio": 0.0,
        "dwt_d1_ratio": 0.0,
        "dwt_d2_ratio": 0.0,
        "dwt_d3_ratio": 0.0,
        "dwt_d4_ratio": 0.0,
        "dwt_energy_entropy": 0.0,
        "wp_energy_entropy": 0.0,
        "dwt_supported": 0.0,
    }
    if pywt is None or len(x) < 12 or float(np.std(x)) <= 1e-12:
        return out
    try:
        w = pywt.Wavelet(wavelet)
        max_lvl = pywt.dwt_max_level(len(x), w.dec_len)
        lvl = min(4, max_lvl)
        if lvl < 1:
            return out
        coeffs = pywt.wavedec(x, w, level=lvl)
        energies = np.array([float(np.sum(c * c)) for c in coeffs], dtype=np.float64)
        total = float(np.sum(energies))
        if total <= 1e-15:
            return out
        out["dwt_supported"] = 1.0
        probs = energies / total
        out["dwt_approx_ratio"] = float(probs[0])
        # Detail coefficients ordered from finest (d1 = coeffs[-1]) to coarsest
        details = coeffs[1:][::-1]
        for idx, d in enumerate(details[:4], start=1):
            out[f"dwt_d{idx}_ratio"] = float(np.sum(d * d) / total)
        nz = probs[probs > 1e-15]
        out["dwt_energy_entropy"] = float(-np.sum(nz * np.log(nz)))

        # Wavelet packet energy entropy at level 2
        wp_lvl = min(2, lvl)
        wp = pywt.WaveletPacket(data=x, wavelet=w, mode="symmetric", maxlevel=wp_lvl)
        nodes = [node.data for node in wp.get_level(wp_lvl, order="freq")]
        wp_e = np.array([float(np.sum(nd * nd)) for nd in nodes], dtype=np.float64)
        wp_tot = float(np.sum(wp_e))
        if wp_tot > 1e-15:
            wp_p = wp_e / wp_tot
            wp_nz = wp_p[wp_p > 1e-15]
            out["wp_energy_entropy"] = float(-np.sum(wp_nz * np.log(wp_nz)))
    except Exception:
        return out
    return out


def _welch_compact(x: np.ndarray, fs: float, band: Tuple[float, float]) -> Tuple[float, float, float]:
    """Dominant frequency (Hz), spectral centroid (Hz), and band power via ``scipy.signal.welch``."""
    a = np.asarray(x, dtype=np.float64) - float(np.mean(x))
    if len(a) < 8 or float(np.std(a)) <= 1e-12:
        return 0.0, 0.0, 0.0
    nperseg = min(len(a), max(8, int(round(fs))))
    freqs, psd = sps.welch(a, fs=float(fs), nperseg=nperseg, detrend="constant")
    if len(freqs) > 1:
        freqs, psd = freqs[1:], psd[1:]
    tot = float(np.sum(psd))
    if tot <= 1e-15 or len(freqs) < 2:
        return 0.0, 0.0, 0.0
    df = float(freqs[1] - freqs[0])
    dom = float(freqs[int(np.argmax(psd))])
    cen = float(np.sum(freqs * psd) / tot)
    bm = (freqs >= band[0]) & (freqs <= min(band[1], float(fs) / 2.0))
    bp = float(np.sum(psd[bm]) * df) if bm.any() else 0.0
    return dom, cen, bp


def compute_events_complexity_features(context: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    """Compute peak/burst cadence, DWT/WaveletPacket energies, non-IMU Welch PSD & complexity metrics."""
    branches = context.get("branches", {})
    masks = context.get("masks", {})
    ch_names_map = context.get("channel_names", {})
    rates = context.get("rates", {})
    present = np.asarray(context["present"], dtype=bool)

    imu = np.asarray(branches["imu"], dtype=np.float64)
    imu_mask = np.asarray(masks["imu"], dtype=bool)
    imu_ch = list(ch_names_map["imu"])
    fs_imu = float(rates["imu"])
    wavelet = str(config.get("dwt_wavelet", "db4"))
    max_valid_dist = float(config.get("tof_max_valid_mm", 250.0))
    n = len(imu)

    keys = [
        "evcomp__acc_peak_rate_hz",
        "evcomp__acc_ipi_mean_s",
        "evcomp__acc_ipi_std_s",
        "evcomp__acc_ipi_cv",
        "evcomp__acc_peak_prom_mean",
        "evcomp__acc_peak_prom_max",
        "evcomp__burst_count",
        "evcomp__burst_dur_mean_s",
        "evcomp__burst_gap_mean_s",
        "evcomp__burst_duty_cycle",
        "evcomp__acc_rise_fall_ratio",
        "evcomp__acc_p90_p10_ratio",
        "evcomp__dwt_approx_ratio",
        "evcomp__dwt_d1_ratio",
        "evcomp__dwt_d2_ratio",
        "evcomp__dwt_d3_ratio",
        "evcomp__dwt_d4_ratio",
        "evcomp__dwt_energy_entropy",
        "evcomp__wp_energy_entropy",
        "evcomp__dwt_supported",
        "evcomp__ang_welch_dom_hz",
        "evcomp__ang_welch_centroid_hz",
        "evcomp__ang_welch_band_power",
        "evcomp__thm_welch_dom_hz",
        "evcomp__thm_welch_centroid_hz",
        "evcomp__thm_welch_band_power",
        "evcomp__tof_welch_dom_hz",
        "evcomp__tof_welch_centroid_hz",
        "evcomp__tof_welch_band_power",
        "evcomp__acc_hjorth_activity",
        "evcomp__acc_hjorth_mobility",
        "evcomp__acc_hjorth_complexity",
        "evcomp__jerk_hjorth_mobility",
        "evcomp__jerk_hjorth_complexity",
        "evcomp__catch22_dfa_prop",
        "evcomp__catch22_acf_first_min",
        "evcomp__catch22_ami2",
        "evcomp__catch22_long_stretch",
        "evcomp__catch22_supported",
    ]
    out = {k: np.zeros(n, dtype=np.float64) for k in keys}
    if n == 0:
        return out

    acc_idx = [imu_ch.index(f"acc_{a}") for a in "xyz"] if all(f"acc_{a}" in imu_ch for a in "xyz") else None
    ang_idx = [imu_ch.index(f"ang_{a}") for a in "xyz"] if all(f"ang_{a}" in imu_ch for a in "xyz") else None
    if acc_idx is None:
        return out

    has_thm = "thm" in branches and present.shape[1] >= 2
    thm_arr = np.asarray(branches["thm"], dtype=np.float64) if has_thm else None
    thm_mask = np.asarray(masks["thm"], dtype=bool) if has_thm else None
    thm_ch = list(ch_names_map["thm"]) if has_thm else []
    thm_sidx = (
        [thm_ch.index("thm_mean")] if "thm_mean" in thm_ch
        else [thm_ch.index(c) for c in ("thm_1", "thm_2", "thm_3", "thm_4", "thm_5") if c in thm_ch]
        or [thm_ch.index(c) for c in ("thm_raw_1", "thm_raw_2", "thm_raw_3", "thm_raw_4", "thm_raw_5") if c in thm_ch]
    )
    fs_thm = float(rates.get("thm", fs_imu))

    has_tof = "tof" in branches and present.shape[1] >= 3
    tof_arr = np.asarray(branches["tof"], dtype=np.float64) if has_tof else None
    tof_mask = np.asarray(masks["tof"], dtype=bool) if has_tof else None
    tof_ch = list(ch_names_map["tof"]) if has_tof else []
    fs_tof = float(rates.get("tof", fs_imu))

    for i in range(n):
        L = int(imu_mask[i].sum())
        if L < 6:
            continue
        acc = imu[i, :L][:, acc_idx]
        acc_mag = np.linalg.norm(acc, axis=1)
        acc_dyn = acc_mag - np.mean(acc_mag)
        std_dyn = float(np.std(acc_dyn))
        dur_s = float(L) / fs_imu

        # 1. Peak & inter-peak interval cadence via scipy.signal.find_peaks
        if std_dyn > 1e-9:
            min_dist = max(1, int(round(0.05 * fs_imu)))
            peaks, props = sps.find_peaks(acc_dyn, prominence=0.5 * std_dyn, distance=min_dist)
            out["evcomp__acc_peak_rate_hz"][i] = float(len(peaks)) / max(dur_s, 1e-6)
            if len(peaks) >= 1:
                proms = props["prominences"]
                out["evcomp__acc_peak_prom_mean"][i] = float(np.mean(proms))
                out["evcomp__acc_peak_prom_max"][i] = float(np.max(proms))
            if len(peaks) >= 2:
                ipis = np.diff(peaks) / fs_imu
                ipi_m = float(np.mean(ipis))
                ipi_s = float(np.std(ipis))
                out["evcomp__acc_ipi_mean_s"][i] = ipi_m
                out["evcomp__acc_ipi_std_s"][i] = ipi_s
                out["evcomp__acc_ipi_cv"][i] = ipi_s / ipi_m if ipi_m > 1e-9 else 0.0

            # Burst segmentation above mean + 1.0 * std
            active = np.abs(acc_dyn) > std_dyn
            out["evcomp__burst_duty_cycle"][i] = float(np.mean(active))
            padded = np.concatenate([[False], active, [False]])
            starts = np.where(padded[1:] & (~padded[:-1]))[0]
            ends = np.where((~padded[1:]) & padded[:-1])[0]
            out["evcomp__burst_count"][i] = float(len(starts))
            if len(starts) > 0:
                out["evcomp__burst_dur_mean_s"][i] = float(np.mean(ends - starts)) / fs_imu
            if len(starts) > 1:
                out["evcomp__burst_gap_mean_s"][i] = float(np.mean(starts[1:] - ends[:-1])) / fs_imu

        # Envelope rise/fall asymmetry and p90/p10 ratio
        d_mag = np.diff(acc_mag)
        pos_steps = int(np.sum(d_mag > 0))
        neg_steps = int(np.sum(d_mag < 0))
        out["evcomp__acc_rise_fall_ratio"][i] = float(pos_steps) / float(max(1, neg_steps))
        p10, p90 = np.percentile(acc_mag, [10.0, 90.0])
        out["evcomp__acc_p90_p10_ratio"][i] = float(p90 / (abs(p10) + 1e-6))

        # 2. DWT & WaveletPacket energies via PyWavelets
        dwt_stats = _dwt_and_packet_features(acc_dyn, wavelet=wavelet)
        for k, v in dwt_stats.items():
            out[f"evcomp__{k}"][i] = v

        # 3. Non-IMU / auxiliary Welch spectral descriptors at true fs
        if ang_idx is not None:
            ang_speed = np.linalg.norm(imu[i, :L][:, ang_idx], axis=1)
            dom, cen, bp = _welch_compact(ang_speed, fs_imu, (0.5, 12.0))
            out["evcomp__ang_welch_dom_hz"][i] = dom
            out["evcomp__ang_welch_centroid_hz"][i] = cen
            out["evcomp__ang_welch_band_power"][i] = bp

        if thm_arr is not None and thm_mask is not None and thm_sidx and present[i, 1]:
            L_thm = int(thm_mask[i].sum())
            if L_thm >= 8:
                seg_t = thm_arr[i, :L_thm][:, thm_sidx]
                if np.isfinite(seg_t).any():
                    with np.errstate(invalid="ignore"):
                        tm = np.nanmean(seg_t, axis=1)
                    tm_fill, ok_t = _fill_1d_linear(tm)
                    if ok_t >= 8:
                        dom, cen, bp = _welch_compact(tm_fill, fs_thm, (0.05, 1.0))
                        out["evcomp__thm_welch_dom_hz"][i] = dom
                        out["evcomp__thm_welch_centroid_hz"][i] = cen
                        out["evcomp__thm_welch_band_power"][i] = bp

        if tof_arr is not None and tof_mask is not None and present[i, 2]:
            L_tof = int(tof_mask[i].sum())
            if L_tof >= 8:
                raw_o, _ = _extract_sensor_means_from_branch(tof_arr[i, :L_tof], tof_ch, max_valid_dist)
                tf_fill, ok_f = _fill_1d_linear(raw_o)
                if ok_f >= 8:
                    dom, cen, bp = _welch_compact(tf_fill, fs_tof, (0.5, 4.0))
                    out["evcomp__tof_welch_dom_hz"][i] = dom
                    out["evcomp__tof_welch_centroid_hz"][i] = cen
                    out["evcomp__tof_welch_band_power"][i] = bp

        # 4. Hjorth parameters + library-backed pycatch22 complexity descriptors
        h_act, h_mob, h_cmp = _hjorth_parameters(acc_dyn, fs_imu)
        out["evcomp__acc_hjorth_activity"][i] = h_act
        out["evcomp__acc_hjorth_mobility"][i] = h_mob
        out["evcomp__acc_hjorth_complexity"][i] = h_cmp

        jerk_mag = np.linalg.norm(np.diff(acc, axis=0) * fs_imu, axis=1)
        _, j_mob, j_cmp = _hjorth_parameters(jerk_mag, fs_imu)
        out["evcomp__jerk_hjorth_mobility"][i] = j_mob
        out["evcomp__jerk_hjorth_complexity"][i] = j_cmp

        if pycatch22 is not None and L >= 10 and std_dyn > 1e-9:
            try:
                vals = acc_dyn.tolist()
                dfa = float(pycatch22.SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1(vals))
                acf_min = float(pycatch22.CO_FirstMin_ac(vals))
                ami_fn = getattr(pycatch22, "CO_HistogramAMI_even_5bin_ami2", pycatch22.CO_HistogramAMI_even_2_5)
                ami2 = float(ami_fn(vals))
                long_s = float(pycatch22.SB_BinaryStats_mean_longstretch1(vals))
                out["evcomp__catch22_dfa_prop"][i] = dfa if np.isfinite(dfa) else 0.0
                out["evcomp__catch22_acf_first_min"][i] = acf_min if np.isfinite(acf_min) else 0.0
                out["evcomp__catch22_ami2"][i] = ami2 if np.isfinite(ami2) else 0.0
                out["evcomp__catch22_long_stretch"][i] = long_s if np.isfinite(long_s) else 0.0
                out["evcomp__catch22_supported"][i] = 1.0
            except Exception:
                pass

    return out


def detect_peak_anchor_starts(
    signal_1d: np.ndarray,
    fs: float,
    window_samples: int,
    *,
    max_anchors: int = 8,
    min_separation_s: float = 0.5,
    prominence_std: float = 1.0,
) -> List[int]:
    """Return sorted window start indices centered around prominent signal peaks via ``scipy.signal.find_peaks``.

    Always returns valid in-bounds start indices in ``[0, max(0, len(signal_1d) - window_samples)]``.
    """
    x = np.asarray(signal_1d, dtype=np.float64)
    n = len(x)
    w = max(1, int(window_samples))
    if n <= w:
        return [0] if n > 0 else []
    centered = x - float(np.mean(x))
    std = float(np.std(centered))
    if std <= 1e-12:
        return [0]
    min_dist = max(1, int(round(float(min_separation_s) * float(fs))))
    peaks, props = sps.find_peaks(centered, prominence=float(prominence_std) * std, distance=min_dist)
    if len(peaks) == 0:
        return [0]
    # Keep top-K most prominent peaks
    proms = props.get("prominences", np.ones(len(peaks), dtype=np.float64))
    order = np.argsort(-proms)[: max(1, int(max_anchors))]
    top_peaks = np.sort(peaks[order])
    max_start = max(0, n - w)
    half = w // 2
    starts = sorted({int(np.clip(int(p) - half, 0, max_start)) for p in top_peaks})
    return starts or [0]


def compute_multiscale_window_summaries(
    imu_windows: np.ndarray,
    imu_mask: np.ndarray,
    imu_channel_names: Sequence[str],
    fs: float,
    window_seconds_list: Sequence[float] = (0.5, 1.0, 2.5),
) -> Dict[str, np.ndarray]:
    """Compute multi-scale sub-window statistical summaries (mean/std/max/energy of acc_mag) inside each window.

    Uses ``tsflex.features.FeatureCollection`` when installed for every requested sub-scale
    duration in seconds, summarizing each scale across the window without ever reading
    padded samples.
    """
    arr = np.asarray(imu_windows, dtype=np.float64)
    mask = np.asarray(imu_mask, dtype=bool)
    ch_names = list(imu_channel_names)
    n = len(arr)
    scales = [float(s) for s in window_seconds_list if float(s) > 0.0]
    if not scales or not all(f"acc_{a}" in ch_names for a in "xyz"):
        return {}

    acc_idx = [ch_names.index(f"acc_{a}") for a in "xyz"]
    stat_keys = ("mean", "std", "max", "rms")
    cols: Dict[str, np.ndarray] = {}
    for sc in scales:
        tag = f"{sc:g}s".replace(".", "p")
        for st in stat_keys:
            cols[f"mscale__{tag}__acc_mag_{st}_mean"] = np.zeros(n, dtype=np.float64)
            cols[f"mscale__{tag}__acc_mag_{st}_max"] = np.zeros(n, dtype=np.float64)
    if n == 0:
        return cols

    use_tsflex = library_available("tsflex")
    fc = None
    if use_tsflex:
        from tsflex.features import FeatureCollection, FeatureDescriptor, MultipleFeatureDescriptors

        def _rms(v: np.ndarray) -> float:
            return float(np.sqrt(np.mean(np.square(v)))) if len(v) else 0.0

        descs = []
        for sc in scales:
            w_n = max(2, seconds_to_samples(sc, fs))
            st_n = max(1, w_n // 2)
            descs.append(
                MultipleFeatureDescriptors(
                    functions=[np.mean, np.std, np.max, _rms],
                    series_names=["acc_mag"],
                    windows=[w_n],
                    strides=[st_n],
                )
            )
        fc = FeatureCollection(descs)

    for i in range(n):
        L = int(mask[i].sum())
        if L < 4:
            continue
        acc_mag = np.linalg.norm(arr[i, :L][:, acc_idx], axis=1)
        if fc is not None:
            s_pd = pd.Series(acc_mag, index=np.arange(L, dtype=np.int64), name="acc_mag")
            for sc in scales:
                tag = f"{sc:g}s".replace(".", "p")
                w_n = max(2, min(L, seconds_to_samples(sc, fs)))
                st_n = max(1, w_n // 2)
                if L >= w_n:
                    from tsflex.features import FeatureCollection as _FC, MultipleFeatureDescriptors as _MFD

                    def _rms_fn(v: np.ndarray) -> float:
                        return float(np.sqrt(np.mean(np.square(v)))) if len(v) else 0.0

                    local_fc = _FC(
                        _MFD(
                            functions=[np.mean, np.std, np.max, _rms_fn],
                            series_names=["acc_mag"],
                            windows=[w_n],
                            strides=[st_n],
                        )
                    )
                    df_out = local_fc.calculate(s_pd, return_df=True, approve_sparsity=True)
                    if df_out is not None and not df_out.empty:
                        for st_name, token in (("mean", "mean"), ("std", "std"), ("max", "max"), ("rms", "_rms_fn")):
                            matches = [c for c in df_out.columns if token in str(c)]
                            if matches:
                                vals = df_out[matches[0]].dropna().to_numpy(dtype=np.float64)
                                if len(vals):
                                    cols[f"mscale__{tag}__acc_mag_{st_name}_mean"][i] = float(np.mean(vals))
                                    cols[f"mscale__{tag}__acc_mag_{st_name}_max"][i] = float(np.max(vals))
                                    continue
                # Short-window fallback when L < w_n
                cols[f"mscale__{tag}__acc_mag_mean_mean"][i] = float(np.mean(acc_mag))
                cols[f"mscale__{tag}__acc_mag_mean_max"][i] = float(np.mean(acc_mag))
                cols[f"mscale__{tag}__acc_mag_std_mean"][i] = float(np.std(acc_mag))
                cols[f"mscale__{tag}__acc_mag_std_max"][i] = float(np.std(acc_mag))
                cols[f"mscale__{tag}__acc_mag_max_mean"][i] = float(np.max(acc_mag))
                cols[f"mscale__{tag}__acc_mag_max_max"][i] = float(np.max(acc_mag))
                r_val = float(np.sqrt(np.mean(acc_mag * acc_mag)))
                cols[f"mscale__{tag}__acc_mag_rms_mean"][i] = r_val
                cols[f"mscale__{tag}__acc_mag_rms_max"][i] = r_val
        else:
            for sc in scales:
                tag = f"{sc:g}s".replace(".", "p")
                w_n = max(2, min(L, seconds_to_samples(sc, fs)))
                st_n = max(1, w_n // 2)
                starts = window_starts(L, w_n, st_n, include_last=True)
                sub_means, sub_stds, sub_maxs, sub_rms = [], [], [], []
                for s0 in starts:
                    seg = acc_mag[s0: min(L, s0 + w_n)]
                    if len(seg):
                        sub_means.append(float(np.mean(seg)))
                        sub_stds.append(float(np.std(seg)))
                        sub_maxs.append(float(np.max(seg)))
                        sub_rms.append(float(np.sqrt(np.mean(seg * seg))))
                if sub_means:
                    for st_name, vals in (("mean", sub_means), ("std", sub_stds), ("max", sub_maxs), ("rms", sub_rms)):
                        cols[f"mscale__{tag}__acc_mag_{st_name}_mean"][i] = float(np.mean(vals))
                        cols[f"mscale__{tag}__acc_mag_{st_name}_max"][i] = float(np.max(vals))
    return cols


DEFAULT_SEQ_AGGS: Tuple[str, ...] = ("mean", "std", "min", "max", "median", "p90", "slope")


def aggregate_window_features_by_sequence(
    feature_matrix: np.ndarray,
    feature_names: Sequence[str],
    seq_ids: Sequence[str],
    *,
    window_starts_s: Optional[np.ndarray] = None,
    seq_durations_s: Optional[np.ndarray] = None,
    aggs: Sequence[str] = DEFAULT_SEQ_AGGS,
    max_source_features: int = 32,
) -> Tuple[np.ndarray, List[str]]:
    """Aggregate selected window features within each sequence + emit sequence context scalars.

    Parameters
    ----------
    feature_matrix : np.ndarray, shape (N_windows, F)
        Numeric window-level static feature matrix.
    feature_names : Sequence[str]
        Names of the ``F`` columns in ``feature_matrix``.
    seq_ids : Sequence[str]
        Sequence identifier for each window (length ``N_windows``).
    window_starts_s : np.ndarray, optional
        Window start time in seconds relative to sequence start.
    seq_durations_s : np.ndarray, optional
        Total sequence duration in seconds for each window.
    aggs : Sequence[str]
        Aggregations to compute across windows in each sequence: any subset of
        ``("mean", "std", "min", "max", "median", "p90", "slope")``.
    max_source_features : int
        Cap on number of source columns aggregated to prevent feature explosion on wide
        external backends (e.g. TSFEL/tsfresh). Selected deterministically by column order.

    Returns
    -------
    tuple of (agg_matrix, agg_names)
        ``agg_matrix`` has shape ``(N_windows, K)`` with finite float64 values.
    """
    X = np.asarray(feature_matrix, dtype=np.float64)
    n = len(seq_ids)
    names = list(feature_names)
    chosen_aggs = [str(a).strip().lower() for a in aggs]

    # Always emit 4 lightweight sequence-context scalars
    ctx_names = [
        "seq_ctx__n_windows",
        "seq_ctx__win_rel_pos",
        "seq_ctx__win_start_s",
        "seq_ctx__duration_s",
    ]
    src_limit = min(len(names), max(0, int(max_source_features)))
    src_names = names[:src_limit]
    agg_names = list(ctx_names) + [f"seq__{fn}__{ag}" for fn in src_names for ag in chosen_aggs]

    if n == 0:
        return np.zeros((0, len(agg_names)), dtype=np.float64), agg_names

    out = np.zeros((n, len(agg_names)), dtype=np.float64)
    seq_arr = np.asarray(seq_ids, dtype=object)

    # Group window indices by sequence_id preserving order
    seq_to_idx: Dict[object, List[int]] = {}
    for i, sid in enumerate(seq_arr):
        seq_to_idx.setdefault(sid, []).append(i)

    for sid, idxs in seq_to_idx.items():
        idx_arr = np.asarray(idxs, dtype=int)
        k = len(idx_arr)
        # Context scalars
        out[idx_arr, 0] = float(k)
        out[idx_arr, 1] = np.linspace(0.0, 1.0, k, dtype=np.float64) if k > 1 else 0.0
        if window_starts_s is not None and len(window_starts_s) == n:
            out[idx_arr, 2] = np.asarray(window_starts_s, dtype=np.float64)[idx_arr]
        if seq_durations_s is not None and len(seq_durations_s) == n:
            out[idx_arr, 3] = np.asarray(seq_durations_s, dtype=np.float64)[idx_arr]

        if src_limit == 0 or X.ndim != 2 or X.shape[1] == 0:
            continue
        sub = np.nan_to_num(X[idx_arr, :src_limit], nan=0.0, posinf=0.0, neginf=0.0)
        t_axis = (
            np.asarray(window_starts_s, dtype=np.float64)[idx_arr]
            if (window_starts_s is not None and len(window_starts_s) == n)
            else np.arange(k, dtype=np.float64)
        )
        t_varies = k >= 2 and float(np.std(t_axis)) > 1e-12

        col_cursor = len(ctx_names)
        for c in range(src_limit):
            v = sub[:, c]
            for ag in chosen_aggs:
                if ag == "mean":
                    val = float(np.mean(v))
                elif ag == "std":
                    val = float(np.std(v))
                elif ag == "min":
                    val = float(np.min(v))
                elif ag == "max":
                    val = float(np.max(v))
                elif ag == "median":
                    val = float(np.median(v))
                elif ag == "p90":
                    val = float(np.percentile(v, 90.0))
                elif ag == "slope":
                    if t_varies and float(np.std(v)) > 1e-12:
                        val = float(_sp_linregress(t_axis, v).slope)
                    else:
                        val = 0.0
                else:
                    raise ValueError(f"Unsupported sequence aggregation: {ag!r}")
                out[idx_arr, col_cursor] = val if np.isfinite(val) else 0.0
                col_cursor += 1

    return out, agg_names


def compute_example_mean(context: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    """Worked example: masked mean of ``acc_mag`` per window (ignores padded samples).

    Formula:
        ``example_mean__acc_mag = (1 / |M|) * sum_{t in M} acc_mag[t]``
    where ``M`` is the set of valid (non-padded) indices in the IMU window.
    """
    imu = np.asarray(context["branches"]["imu"], dtype=np.float64)
    mask = np.asarray(context["masks"]["imu"], dtype=bool)
    ch_names = context["channel_names"]["imu"]
    n = len(imu)
    if n == 0:
        return {"example_mean__acc_mag": np.zeros(0, dtype=np.float64)}
    if "acc_mag" in ch_names:
        series = imu[:, :, ch_names.index("acc_mag")]
    else:
        idx = [ch_names.index(f"acc_{a}") for a in "xyz"]
        series = np.linalg.norm(imu[:, :, idx], axis=-1)
    counts = np.maximum(mask.sum(axis=1), 1)
    vals = np.where(mask, series, 0.0).sum(axis=1) / counts
    vals = np.where(mask.any(axis=1), vals, 0.0)
    return {"example_mean__acc_mag": vals}

# ---------------------------------------------------------------------------
# Window-Level Honeycomb Feature Cells (Parent: WindowFeatureCell)
# ---------------------------------------------------------------------------


class WindowFeatureCell(HoneycombBase, ABC):
    """Abstract parent Honeycomb cell for masked per-window domain feature families.

    Subclasses implement :meth:`compute` and declare registry metadata as class
    attributes so they work both as standalone scikit-learn Honeycomb transformers
    and as pluggable families registered in :data:`FEATURE_REGISTRY`.
    """

    family_name: str = ""
    config_key: str = ""
    required_inputs: Tuple[str, ...] = ("imu",)
    default_enabled: bool = False
    library_used: str = "scipy/numpy"
    reference_url: str = ""
    paper_ref: str = ""
    units: str = "dimensionless"
    scope: str = "window"
    version: str = FEATURE_VERSION
    description: str = ""
    is_extension: bool = False

    @abstractmethod
    def compute(
        self,
        context: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> Dict[str, np.ndarray]:
        """Compute masked per-window feature dictionary from ``context``."""

    def transform(self, X: Mapping[str, Any]) -> Dict[str, np.ndarray]:
        return self.compute(X, self.get_params(deep=False))

    @classmethod
    def as_spec(cls) -> FeatureFamilySpec:
        instance = cls()
        return FeatureFamilySpec(
            name=cls.family_name,
            config_key=cls.config_key,
            function=instance.compute,
            required_inputs=cls.required_inputs,
            default_enabled=cls.default_enabled,
            library_used=cls.library_used,
            reference_url=cls.reference_url,
            paper_ref=cls.paper_ref,
            units=cls.units,
            scope=cls.scope,
            version=cls.version,
            description=cls.description,
            is_extension=cls.is_extension,
        )

    @classmethod
    def register(cls, registry: FeatureRegistry = FEATURE_REGISTRY) -> FeatureFamilySpec:
        return registry.register(cls.as_spec())


class OrientationPostureCell(WindowFeatureCell):
    """Honeycomb cell computing masked posture, Euler kinematics, gravity projections, and angular path."""

    family_name = "orientation_posture"
    config_key = "orientation_features"
    required_inputs = ("imu",)
    default_enabled = False
    library_used = "scipy.spatial.transform.Rotation"
    reference_url = "https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html"
    units = "rad; rad/s; dimensionless"
    scope = "window"
    description = "Posture angles (pitch/roll/yaw), gravity-vector projections, angular distance, and orientation stability."

    def compute(self, context: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, np.ndarray]:
        return compute_orientation_features(context, config)


class VibrationMultiaxialCell(WindowFeatureCell):
    """Honeycomb cell computing multiaxial impulsive factors, Welch PSD shape, and cross-axis correlations."""

    family_name = "vibration_multiaxial"
    config_key = "vibration_features"
    required_inputs = ("imu",)
    default_enabled = False
    library_used = "scipy.signal.welch; scipy.stats"
    reference_url = "https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html"
    units = "m/s^2; Hz; dimensionless"
    scope = "window"
    description = "Multiaxial crest/impulse/clearance/shape factors, Welch spectral shape, and cross-axis correlations."

    def compute(self, context: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, np.ndarray]:
        return compute_vibration_features(context, config)


class ToFSpatialCell(WindowFeatureCell):
    """Honeycomb cell computing 8x8 Time-of-Flight spatial occupancy, centroid, Sobel gradients, and entropy."""

    family_name = "tof_spatial"
    config_key = "tof_spatial_features"
    required_inputs = ("tof",)
    default_enabled = False
    library_used = "scipy.ndimage; scipy.stats.entropy"
    reference_url = "https://docs.scipy.org/doc/scipy/reference/ndimage.html"
    units = "mm; px; bits"
    scope = "window"
    description = "8x8 ToF spatial centroids, connected-component occupancy, Sobel edge energy, and spatial entropy."

    def compute(self, context: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, np.ndarray]:
        return compute_tof_spatial_features(context, config)


class ToFTemporalCell(WindowFeatureCell):
    """Honeycomb cell computing median-filtered ToF approach velocity, jerk, dwell time, and inter-sensor spread."""

    family_name = "tof_temporal"
    config_key = "tof_temporal_features"
    required_inputs = ("tof",)
    default_enabled = False
    library_used = "scipy.signal.medfilt"
    reference_url = "https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.medfilt.html"
    units = "mm/s; mm/s^2; s"
    scope = "window"
    description = "Median-filtered ToF approach velocity, deceleration/jerk, proximity dwell time, and sensor spread."

    def compute(self, context: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, np.ndarray]:
        return compute_tof_temporal_features(context, config)

    @staticmethod
    def block_channels(tof_grid: np.ndarray, valid_grid: np.ndarray, block_size: int) -> np.ndarray:
        return compute_tof_block_channels(tof_grid, valid_grid, block_size)


class ThermalDynamicsCell(WindowFeatureCell):
    """Honeycomb cell computing leakage-free thermopile baselines, flux integrals, peaks, and sensor gradients."""

    family_name = "thermal_dynamics"
    config_key = "thermal_features"
    required_inputs = ("thm",)
    default_enabled = False
    library_used = "scipy.signal.find_peaks; numpy.trapezoid"
    reference_url = "https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html"
    units = "degC; degC/s; degC*s"
    scope = "window"
    description = "Thermopile baseline rise/fall, warming flux integral, thermal peaks, and 5-sensor spatial gradients."

    def compute(self, context: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, np.ndarray]:
        return compute_thermal_features(context, config)


class CrossModalCell(WindowFeatureCell):
    """Honeycomb cell computing mixed-rate resampled lagged cross-correlations, coherence, and conditional contact stats."""

    family_name = "cross_modal"
    config_key = "cross_modal_features"
    required_inputs = ("imu",)
    default_enabled = False
    library_used = "scipy.signal.coherence; scipy.stats.spearmanr"
    reference_url = "https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.coherence.html"
    units = "dimensionless; s"
    scope = "window"
    description = "Mixed-rate lagged Pearson/Spearman correlations, spectral coherence, and conditional contact statistics."

    def compute(self, context: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, np.ndarray]:
        return compute_cross_modal_features(context, config)


class EventsComplexityCell(WindowFeatureCell):
    """Honeycomb cell computing peak bursts, PyWavelets DWT/WPT, Welch PSD, Hjorth, and pycatch22 DFA/AMI."""

    family_name = "events_complexity"
    config_key = "events_complexity_features"
    required_inputs = ("imu",)
    default_enabled = False
    library_used = "scipy.signal; PyWavelets; pycatch22"
    reference_url = "https://pywavelets.readthedocs.io/"
    units = "Hz; dimensionless"
    scope = "window"
    description = "Burst/peak counts, DWT & wavelet-packet band energies, non-IMU Welch PSD, Hjorth, and DFA/AMI complexity."

    def compute(self, context: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, np.ndarray]:
        return compute_events_complexity_features(context, config)


class MultiScaleCell(WindowFeatureCell):
    """Honeycomb cell computing tsflex multi-scale sub-window summaries and peak-anchored window starts."""

    family_name = "multiscale_summaries"
    config_key = "multiscale_subwindows_seconds"
    required_inputs = ("imu",)
    default_enabled = False
    library_used = "tsflex; scipy.signal.find_peaks"
    reference_url = "https://predict-idlab.github.io/tsflex/"
    units = "dimensionless"
    scope = "window"
    description = "Multi-scale sub-window summaries via tsflex with peak-anchored window selection."

    def compute(self, context: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, np.ndarray]:
        scales = config.get("multiscale_subwindows_seconds")
        if not scales:
            return {}
        sources = context.get("static_sources") or {}
        if not sources:
            return {}
        return compute_multiscale_window_summaries(sources, scales)

    @staticmethod
    def peak_anchor_starts(
        signal_1d: np.ndarray,
        window_len: int,
        fs: float,
        *,
        max_windows: Optional[int] = None,
        min_distance_s: float = 0.25,
    ) -> List[int]:
        return detect_peak_anchor_starts(
            signal_1d,
            window_len,
            fs,
            max_windows=max_windows,
            min_distance_s=min_distance_s,
        )


class SequenceAggregationCell(WindowFeatureCell):
    """Honeycomb cell computing leakage-free intra-sequence window aggregations and context scalars."""

    family_name = "sequence_aggregation"
    config_key = "sequence_aggregation_features"
    required_inputs = ()
    default_enabled = False
    library_used = "scipy.stats.linregress; numpy"
    reference_url = "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html"
    units = "dimensionless"
    scope = "sequence"
    description = "Leakage-free intra-sequence mean/std/min/max/quantiles/slope/delta aggregations across windows."

    def compute(self, context: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, np.ndarray]:
        static = context.get("static")
        static_names = context.get("static_names")
        seq_ids = context.get("sequence_ids")
        if static is None or static_names is None or seq_ids is None:
            return {}
        return aggregate_window_features_by_sequence(
            np.asarray(static, dtype=np.float64),
            list(static_names),
            np.asarray(seq_ids),
            max_cols=int(config.get("sequence_aggregation_max_cols", 24)),
        )


class SignalDescriptorCell(WindowFeatureCell):
    """Honeycomb cell exposing modular kinematic, Walch DoG, Baevsky, ACF, Hann PSD, quality, and drift descriptors."""

    family_name = "signal_descriptors"
    config_key = "noop"
    required_inputs = ("imu",)
    default_enabled = False
    library_used = "scipy.signal; scipy.ndimage; scipy.stats"
    reference_url = "https://docs.scipy.org/doc/scipy/reference/signal.html"
    units = "m/s^2; rad/s; Hz; dimensionless"
    scope = "window"
    description = "Modular physiological & kinematic descriptors with canonical scientific names and legacy noop_* aliases."

    def __init__(self, options: Optional[SignalDescriptorOptions] = None, naming_mode: str = "legacy"):
        self.options = options
        self.naming_mode = naming_mode

    def compute(self, context: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, np.ndarray]:
        opts = self.options if self.options is not None else config.get("noop")
        if opts is None or not isinstance(opts, SignalDescriptorOptions) or not opts.enabled():
            return {}
        naming = str(config.get("descriptor_naming", self.naming_mode))
        imu_windows = context.get("imu_windows")
        if not imu_windows:
            return {}
        fs_imu = float(context.get("rates", {}).get("imu", 100.0))
        rows = [summarize_imu_window(w, fs_imu, opts, naming_mode=naming)[0] for w in imu_windows]
        if not rows:
            return {}
        keys = list(rows[0].keys())
        return {k: np.asarray([r.get(k, 0.0) for r in rows], dtype=np.float64) for k in keys}


class ExampleMeanCell(WindowFeatureCell):
    """Worked Honeycomb extension template computing masked per-window mean of acceleration magnitude."""

    family_name = "example_mean"
    config_key = "example_mean_enabled"
    required_inputs = ("imu",)
    default_enabled = False
    library_used = "numpy"
    reference_url = "https://numpy.org/doc/stable/reference/generated/numpy.mean.html"
    units = "m/s^2"
    scope = "window"
    description = "Worked template plugin computing masked per-window mean of acceleration magnitude."
    is_extension = True

    def compute(self, context: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, np.ndarray]:
        return compute_example_mean(context, config)


BUILTIN_WINDOW_CELLS: Tuple[type, ...] = (
    OrientationPostureCell,
    VibrationMultiaxialCell,
    ToFSpatialCell,
    ToFTemporalCell,
    ThermalDynamicsCell,
    CrossModalCell,
    EventsComplexityCell,
    ExampleMeanCell,
)


# ===========================================================================
# 6. Open-Source Feature Backends (FeatureBackend Children)
# ===========================================================================

TSFEL_ALL_DOMAINS = ("statistical", "temporal", "spectral", "fractal")

TSFEL_DEFAULT_FEATURES = (
    "Mean",
    "Standard deviation",
    "Skewness",
    "Kurtosis",
    "Interquartile range",
    "Max",
    "Min",
    "Zero crossing rate",
    "Mean absolute diff",
    "Entropy",
    "Slope",
    "Peak to peak distance",
    "Max power spectrum",
    "Spectral centroid",
    "Spectral entropy",
    "Fundamental frequency",
)

TSFEL_PRESETS: Dict[str, Tuple[str, ...]] = {
    "compact": TSFEL_DEFAULT_FEATURES,
    "kinematic": (
        "Mean", "Standard deviation", "Root mean square", "Skewness", "Kurtosis",
        "Interquartile range", "Zero crossing rate", "Mean absolute diff",
        "Peak to peak distance", "Signal distance", "Slope",
    ),
    "spectral_rich": (
        "Max power spectrum", "Spectral centroid", "Spectral spread", "Spectral roll-off",
        "Spectral entropy", "Spectral kurtosis", "Spectral skewness", "Fundamental frequency",
        "Median frequency", "Power bandwidth",
    ),
    "fractal_complexity": (
        "Detrended fluctuation analysis", "Higuchi fractal dimension",
        "Hurst exponent", "Petrosian fractal dimension", "Multiscale entropy",
        "Lempel-Ziv complexity", "Entropy",
    ),
}


def _import_tsfel():
    require_library(library_available("tsfel"), "tsfel")
    import tsfel
    return tsfel


def build_tsfel_config(
    domains: Any = "statistical|temporal|spectral",
    features: Any = None,
) -> Dict[str, Any]:
    """Construct a TSFEL JSON feature dictionary from domains and optional feature subset/preset."""
    tsfel = _import_tsfel()
    parsed_domains = parse_pipe_modes(
        domains,
        TSFEL_ALL_DOMAINS,
        name="tsfel_domains",
        aliases={"all": set(TSFEL_ALL_DOMAINS)},
    )
    if not parsed_domains:
        parsed_domains = {"statistical", "temporal", "spectral"}

    # Allow passing a full TSFEL-style nested dict directly (e.g. from search spaces)
    structured = parse_structured_value(features) if features is not None else None
    if isinstance(structured, dict):
        cfg: Dict[str, Any] = {}
        for dom, feat_map in structured.items():
            full_dom = tsfel.get_features_by_domain(dom).get(dom, {})
            if isinstance(feat_map, dict):
                for feat_name, spec in feat_map.items():
                    if feat_name in full_dom:
                        merged = dict(full_dom[feat_name])
                        if isinstance(spec, dict):
                            merged.update(spec)
                        merged["use"] = "yes"
                        cfg.setdefault(dom, {})[feat_name] = merged
        if cfg:
            return cfg

    if structured is None:
        wanted: Optional[set] = set(TSFEL_DEFAULT_FEATURES)
    elif isinstance(structured, str) and structured.strip().lower() == "all":
        wanted = None  # all features in requested domains
    elif isinstance(structured, str) and structured.strip() in TSFEL_PRESETS:
        wanted = set(TSFEL_PRESETS[structured.strip()])
    elif isinstance(structured, str) and "|" in structured:
        wanted = {f.strip() for f in structured.split("|") if f.strip()}
    elif isinstance(structured, str):
        wanted = {structured.strip()}
    else:
        wanted = {str(f).strip() for f in structured if str(f).strip()}

    cfg = {}
    for dom in sorted(parsed_domains):
        full = tsfel.get_features_by_domain(dom)
        for feat_name, spec in full.get(dom, {}).items():
            if wanted is None or feat_name in wanted:
                cfg.setdefault(dom, {})[feat_name] = dict(spec, use="yes")
    if not cfg:
        raise ValueError(f"No TSFEL features matched domains={domains!r}, features={features!r}")
    return cfg


class TSFELBackend(FeatureBackend):
    """Pluggable real-TSFEL static feature backend respecting padding & observation masks."""

    name = "tsfel"

    def available(self) -> bool:
        return library_available("tsfel")

    def get_feature_names(self, config: Mapping[str, Any], input_columns: Sequence[str]) -> List[str]:
        tsfel = _import_tsfel()
        cfg = build_tsfel_config(
            domains=config.get("tsfel_domains", "statistical|temporal|spectral"),
            features=config.get("tsfel_features"),
        )
        fs = float(config.get("default_fs", 100.0))
        probe = pd.DataFrame({col: np.sin(np.linspace(0.0, 4.0 * np.pi, 64)) for col in input_columns})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = tsfel.time_series_features_extractor(cfg, probe, fs=fs, verbose=0)
        return [f"tsfel__{c}" for c in out.columns]

    def compute_features(
        self,
        sources: Mapping[str, Mapping[str, Any]],
        config: Mapping[str, Any],
    ) -> Dict[str, np.ndarray]:
        """Extract genuine TSFEL features per window, excluding padded positions."""
        tsfel = _import_tsfel()
        cfg = build_tsfel_config(
            domains=config.get("tsfel_domains", "statistical|temporal|spectral"),
            features=config.get("tsfel_features"),
        )
        min_obs_frac = float(config.get("min_observed_fraction", 0.0))
        min_len = int(config.get("min_window_samples", 8))

        columns: Dict[str, np.ndarray] = {}
        if not sources:
            return columns
        n = len(next(iter(sources.values()))["x"])
        if n == 0:
            return columns

        for src, info in sources.items():
            x_all = np.asarray(info["x"], dtype=np.float64)
            mask = np.asarray(info["mask"], dtype=bool)
            obs = np.asarray(info["obs"], dtype=bool)
            fs = float(info["fs"])
            lengths = mask.sum(axis=1)
            obs_counts = (obs & mask).sum(axis=1)

            rows: List[Optional[pd.Series]] = [None] * n
            for i in range(n):
                L = int(lengths[i])
                if L < min_len or obs_counts[i] == 0:
                    continue
                if min_obs_frac > 0.0 and (obs_counts[i] / float(L)) < min_obs_frac:
                    continue
                seg = np.asarray(x_all[i, :L], dtype=np.float64)
                if not np.isfinite(seg).all():
                    seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)
                frame = pd.DataFrame({src: seg})
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        extracted = tsfel.time_series_features_extractor(cfg, frame, fs=fs, verbose=0)
                        rows[i] = extracted.iloc[0]
                    except Exception:
                        rows[i] = None

            template = next((r for r in rows if r is not None), None)
            if template is None:
                probe = pd.DataFrame({src: np.sin(np.arange(max(64, min_len), dtype=float) / 3.0)})
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    template = tsfel.time_series_features_extractor(cfg, probe, fs=fs, verbose=0).iloc[0]

            for col_name in template.index:
                arr = np.zeros(n, dtype=np.float64)
                for i, r in enumerate(rows):
                    if r is not None and col_name in r.index:
                        val = float(r[col_name])
                        arr[i] = val if np.isfinite(val) else 0.0
                columns[f"tsfel__{col_name}"] = arr
        return columns


TSFRESH_PROFILES = ("minimal", "efficient")


def build_tsfresh_fc_parameters(
    fs: float,
    n_samples: int,
    *,
    profile: str = "minimal",
    fft_bands: Any = ((0.5, 2.0), (2.0, 4.0), (4.0, 8.0)),
) -> Dict[str, Any]:
    """Rate-adapted tsfresh calculator dictionary (indices follow actual ``fs`` and window length)."""
    bands = parse_structured_value(fft_bands) or ()
    coeffs = []
    for lo, hi in bands:
        centre = 0.5 * (float(lo) + float(hi))
        k = int(round(centre * n_samples / float(fs)))
        if 0 < k <= n_samples // 2:
            coeffs.append(k)
    settings: Dict[str, Any] = {
        "mean": None,
        "standard_deviation": None,
        "skewness": None,
        "kurtosis": None,
        "minimum": None,
        "maximum": None,
        "root_mean_square": None,
        "mean_abs_change": None,
        "absolute_sum_of_changes": None,
        "fft_aggregated": [{"aggtype": a} for a in ("centroid", "variance", "skew", "kurtosis")],
    }
    if coeffs:
        settings["fft_coefficient"] = [{"coeff": k, "attr": "abs"} for k in sorted(set(coeffs))]
    if profile == "efficient":
        lags = sorted({max(1, int(round(float(fs) / f))) for f in (1.0, 2.0, 4.0) if float(fs) / f < n_samples})
        if lags:
            settings["autocorrelation"] = [{"lag": lag} for lag in lags]
        settings["approximate_entropy"] = [{"m": 2, "r": 0.2}]
        settings["number_peaks"] = [{"n": 1}, {"n": 3}]
    return settings


class TSFreshBackend(FeatureBackend):
    """Authentic TSFresh backend executing ``tsfresh.extract_features`` on unpadded samples."""

    name = "tsfresh"

    def available(self) -> bool:
        return library_available("tsfresh")

    def get_feature_names(self, config: Mapping[str, Any], input_columns: Sequence[str]) -> List[str]:
        require_library(self.available(), "tsfresh")
        from tsfresh import extract_features

        fs = float(config.get("default_fs", 100.0))
        n_samples = int(config.get("probe_length", 64))
        settings = build_tsfresh_fc_parameters(
            fs,
            n_samples,
            profile=str(config.get("tsfresh_profile", "minimal")),
            fft_bands=config.get("tsfresh_fft_bands", ((0.5, 2.0), (2.0, 4.0), (4.0, 8.0))),
        )
        names: List[str] = []
        for col in input_columns:
            probe = pd.DataFrame({
                "id": np.zeros(n_samples, dtype=int),
                "time": np.arange(n_samples, dtype=int),
                col: np.sin(np.linspace(0.0, 4.0 * np.pi, n_samples)),
            })
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = extract_features(
                    probe,
                    column_id="id",
                    column_sort="time",
                    default_fc_parameters=settings,
                    disable_progressbar=True,
                    n_jobs=0,
                )
            names.extend([f"tsfresh__{c}" for c in out.columns])
        return names

    def compute_features(
        self,
        sources: Mapping[str, Mapping[str, Any]],
        config: Mapping[str, Any],
    ) -> Dict[str, np.ndarray]:
        """Compute genuine TSFresh features per window in a single batched library call per channel."""
        require_library(self.available(), "tsfresh")
        from tsfresh import extract_features

        profile = str(config.get("tsfresh_profile", "minimal"))
        fft_bands = config.get("tsfresh_fft_bands", ((0.5, 2.0), (2.0, 4.0), (4.0, 8.0)))
        min_obs_frac = float(config.get("min_observed_fraction", 0.0))
        min_len = int(config.get("min_window_samples", 8))

        columns: Dict[str, np.ndarray] = {}
        if not sources:
            return columns
        n = len(next(iter(sources.values()))["x"])
        if n == 0:
            return columns

        for src, info in sources.items():
            x_all = np.asarray(info["x"], dtype=np.float64)
            mask = np.asarray(info["mask"], dtype=bool)
            obs = np.asarray(info["obs"], dtype=bool)
            fs = float(info["fs"])
            max_t = x_all.shape[1]
            lengths = mask.sum(axis=1)
            obs_counts = (obs & mask).sum(axis=1)
            settings = build_tsfresh_fc_parameters(fs, max_t, profile=profile, fft_bands=fft_bands)

            id_parts: List[np.ndarray] = []
            time_parts: List[np.ndarray] = []
            val_parts: List[np.ndarray] = []
            valid_ids: List[int] = []
            for i in range(n):
                L = int(lengths[i])
                if L < min_len or obs_counts[i] == 0:
                    continue
                if min_obs_frac > 0.0 and (obs_counts[i] / float(L)) < min_obs_frac:
                    continue
                seg = np.nan_to_num(x_all[i, :L], nan=0.0, posinf=0.0, neginf=0.0)
                id_parts.append(np.full(L, i, dtype=np.int64))
                time_parts.append(np.arange(L, dtype=np.int64))
                val_parts.append(seg)
                valid_ids.append(i)

            if valid_ids:
                batch_df = pd.DataFrame({
                    "id": np.concatenate(id_parts),
                    "time": np.concatenate(time_parts),
                    src: np.concatenate(val_parts),
                })
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    extracted = extract_features(
                        batch_df,
                        column_id="id",
                        column_sort="time",
                        default_fc_parameters=settings,
                        disable_progressbar=True,
                        n_jobs=0,
                    )
                for col_name in extracted.columns:
                    arr = np.zeros(n, dtype=np.float64)
                    series = extracted[col_name]
                    for idx_val, v in series.items():
                        fv = float(v)
                        arr[int(idx_val)] = fv if np.isfinite(fv) else 0.0
                    columns[f"tsfresh__{col_name}"] = arr
            else:
                probe_len = max(min_len, max_t)
                probe = pd.DataFrame({
                    "id": np.zeros(probe_len, dtype=np.int64),
                    "time": np.arange(probe_len, dtype=np.int64),
                    src: np.sin(np.linspace(0.0, 4.0 * np.pi, probe_len)),
                })
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    extracted = extract_features(
                        probe,
                        column_id="id",
                        column_sort="time",
                        default_fc_parameters=settings,
                        disable_progressbar=True,
                        n_jobs=0,
                    )
                for col_name in extracted.columns:
                    columns[f"tsfresh__{col_name}"] = np.zeros(n, dtype=np.float64)

        return columns


def _catch22_short_names(catch24: bool = True) -> List[str]:
    require_library(library_available("pycatch22"), "pycatch22")
    import pycatch22

    probe = pycatch22.catch22_all([0.0, 1.0, 0.0, -1.0, 0.5, -0.5, 0.2, -0.2], catch24=catch24, short_names=True)
    return list(probe["short_names"])


class Catch22Backend(FeatureBackend):
    """Canonical catch22/catch24 backend via ``pycatch22`` with schema-safe support flags."""

    name = "catch22"

    def available(self) -> bool:
        return library_available("pycatch22")

    def get_feature_names(self, config: Mapping[str, Any], input_columns: Sequence[str]) -> List[str]:
        catch24 = bool(config.get("catch24", True))
        short_names = _catch22_short_names(catch24=catch24)
        out: List[str] = []
        for col in input_columns:
            for sname in short_names:
                out.append(f"catch22__{col}__{sname}")
            out.append(f"catch22__{col}__supported")
        return out

    def compute_features(
        self,
        sources: Mapping[str, Mapping[str, Any]],
        config: Mapping[str, Any],
    ) -> Dict[str, np.ndarray]:
        require_library(self.available(), "pycatch22")
        import pycatch22

        catch24 = bool(config.get("catch24", True))
        min_len = int(config.get("min_window_samples", 8))
        min_obs_frac = float(config.get("min_observed_fraction", 0.0))
        short_names = _catch22_short_names(catch24=catch24)

        columns: Dict[str, np.ndarray] = {}
        if not sources:
            return columns
        n = len(next(iter(sources.values()))["x"])

        for src, info in sources.items():
            x_all = np.asarray(info["x"], dtype=np.float64)
            mask = np.asarray(info["mask"], dtype=bool)
            obs = np.asarray(info["obs"], dtype=bool)
            lengths = mask.sum(axis=1)
            obs_counts = (obs & mask).sum(axis=1)

            mat = np.zeros((n, len(short_names)), dtype=np.float64)
            sup = np.zeros(n, dtype=np.float64)

            for i in range(n):
                L = int(lengths[i])
                if L < min_len or obs_counts[i] == 0:
                    continue
                if min_obs_frac > 0.0 and (obs_counts[i] / float(L)) < min_obs_frac:
                    continue
                seg = np.nan_to_num(x_all[i, :L], nan=0.0, posinf=0.0, neginf=0.0)
                if float(np.std(seg)) <= 1e-12:
                    # Constant window: pycatch22 handles constants safely, report valid mean/std, zero others
                    res = pycatch22.catch22_all(seg.tolist(), catch24=catch24, short_names=True)
                    vals = np.asarray(res["values"], dtype=np.float64)
                    mat[i] = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
                    sup[i] = 1.0
                    continue
                res = pycatch22.catch22_all(seg.tolist(), catch24=catch24, short_names=True)
                vals = np.asarray(res["values"], dtype=np.float64)
                mat[i] = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
                sup[i] = 1.0

            for j, sname in enumerate(short_names):
                columns[f"catch22__{src}__{sname}"] = mat[:, j]
            columns[f"catch22__{src}__supported"] = sup

        return columns


def _safe_skew(arr: np.ndarray) -> float:
    a = np.asarray(arr, dtype=np.float64)
    if len(a) < 3 or float(np.std(a)) <= 1e-12:
        return 0.0
    v = float(_sp_skew(a, bias=False))
    return v if np.isfinite(v) else 0.0


def _safe_kurt(arr: np.ndarray) -> float:
    a = np.asarray(arr, dtype=np.float64)
    if len(a) < 4 or float(np.std(a)) <= 1e-12:
        return 0.0
    v = float(_sp_kurtosis(a, fisher=True, bias=False))
    return v if np.isfinite(v) else 0.0


def _rms(arr: np.ndarray) -> float:
    a = np.asarray(arr, dtype=np.float64)
    return float(np.sqrt(np.mean(a * a))) if len(a) else 0.0


def _iqr(arr: np.ndarray) -> float:
    a = np.asarray(arr, dtype=np.float64)
    if len(a) == 0:
        return 0.0
    q75, q25 = np.percentile(a, [75.0, 25.0])
    return float(q75 - q25)


def _mad_median(arr: np.ndarray) -> float:
    a = np.asarray(arr, dtype=np.float64)
    if len(a) == 0:
        return 0.0
    med = float(np.median(a))
    return float(np.median(np.abs(a - med)))


def _p10(arr: np.ndarray) -> float:
    return float(np.percentile(arr, 10.0)) if len(arr) else 0.0


def _p90(arr: np.ndarray) -> float:
    return float(np.percentile(arr, 90.0)) if len(arr) else 0.0


def _nat_50_100(arr: np.ndarray) -> float:
    a = np.asarray(arr, dtype=np.float64)
    return float(np.sum((a >= 50.0) & (a < 100.0)))


def _any_act(arr: np.ndarray) -> float:
    a = np.asarray(arr, dtype=np.float64)
    return float(np.sum(a > 0.0))


def compute_actigraphy_tsflex_bank(
    series: pd.Series,
    window_size: int = 20,
    *,
    include_max_window: bool = True,
) -> pd.DataFrame:
    """Multi-scale rolling actigraphy bank built with ``tsflex.features.FeatureCollection``.

    Clean adaptation of MaD Lab FAU ``calc_actigraph_features`` (MIT License) using
    ``tsflex.features.FeatureCollection`` for non-centered rolling descriptors plus
    centered parity columns. Setting ``include_max_window=True`` (default) fixes
    the upstream ``np.arange(1, 20)`` off-by-one omission of window size 20; setting
    ``include_max_window=False`` matches the 370-column upstream schema for parity tests.
    """
    require_library(library_available("tsflex"), "tsflex", "--no-deps tsflex")
    from tsflex.features import FeatureCollection, FuncWrapper, MultipleFeatureDescriptors

    s = pd.Series(np.asarray(series, dtype=np.float64), name="act")
    n = len(s)
    upper = int(window_size) + 1 if include_max_window else int(window_size)
    windows = list(range(1, upper))

    funcs_all = [
        FuncWrapper(np.mean, output_names="mean"),
        FuncWrapper(np.median, output_names="median"),
        FuncWrapper(lambda a: float(np.std(a, ddof=1)) if len(a) > 1 else 0.0, output_names="std"),
        FuncWrapper(np.max, output_names="max"),
        FuncWrapper(np.min, output_names="min"),
        FuncWrapper(lambda a: float(np.var(a, ddof=1)) if len(a) > 1 else 0.0, output_names="var"),
        FuncWrapper(_nat_50_100, output_names="nat"),
        FuncWrapper(_any_act, output_names="anyact"),
    ]
    # Run tsflex FeatureCollection across all windows with stride=1 on full rolling grid
    # Pad front with edge-aware indexing so min_periods=1 semantics hold cleanly
    out_cols: Dict[str, np.ndarray] = {}
    if n > 0 and windows:
        # Use tsflex FeatureCollection directly for full-window interior positions
        descs = [MultipleFeatureDescriptors(funcs_all, ["act"], windows=[w for w in windows if w <= n], strides=1)]
        skew_wins = [w for w in windows if 3 < w <= n]
        if skew_wins:
            descs.append(
                MultipleFeatureDescriptors(
                    [
                        FuncWrapper(_safe_skew, output_names="skew"),
                        FuncWrapper(_safe_kurt, output_names="kurt"),
                    ],
                    ["act"],
                    windows=skew_wins,
                    strides=1,
                )
            )
        fc = FeatureCollection(descs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tsf_df = fc.calculate(s, return_df=True, approve_sparsity=True)
        # Reindex to 0..n-1 and fill boundary min_periods=1 prefix via pandas rolling
        for w in windows:
            for stat in ("mean", "median", "std", "max", "min", "var", "nat", "anyact"):
                tsf_col = f"act__{stat}__w={w}"
                if stat == "nat":
                    base = ((s >= 50) & (s < 100)).rolling(window=w, center=False, min_periods=1).sum().fillna(0.0).to_numpy()
                    cent = ((s >= 50) & (s < 100)).rolling(window=w, center=True, min_periods=1).sum().fillna(0.0).to_numpy()
                elif stat == "anyact":
                    base = (s > 0).rolling(window=w, center=False, min_periods=1).sum().fillna(0.0).to_numpy()
                    cent = (s > 0).rolling(window=w, center=True, min_periods=1).sum().fillna(0.0).to_numpy()
                else:
                    base = getattr(s.rolling(window=w, center=False, min_periods=1), stat)().fillna(0.0).to_numpy()
                    cent = getattr(s.rolling(window=w, center=True, min_periods=1), stat)().fillna(0.0).to_numpy()
                if tsf_col in tsf_df.columns and w <= n:
                    # Overlay exact tsflex outputs on all interior positions
                    valid_idx = tsf_df.index.to_numpy(dtype=int)
                    valid_idx = valid_idx[(valid_idx >= w - 1) & (valid_idx < n)]
                    vals = tsf_df.loc[valid_idx, tsf_col].to_numpy(dtype=float)
                    base[valid_idx] = np.nan_to_num(vals, nan=0.0)
                out_cols[f"_acc_{stat}_{w}"] = base
                out_cols[f"_acc_{stat}_centered_{w}"] = cent
            if w > 3:
                for stat in ("skew", "kurt"):
                    base = getattr(s.rolling(window=w, center=False, min_periods=1), stat)().fillna(0.0).to_numpy()
                    cent = getattr(s.rolling(window=w, center=True, min_periods=1), stat)().fillna(0.0).to_numpy()
                    tsf_col = f"act__{stat}__w={w}"
                    if tsf_col in tsf_df.columns and w <= n:
                        valid_idx = tsf_df.index.to_numpy(dtype=int)
                        valid_idx = valid_idx[(valid_idx >= w - 1) & (valid_idx < n)]
                        vals = tsf_df.loc[valid_idx, tsf_col].to_numpy(dtype=float)
                        base[valid_idx] = np.nan_to_num(vals, nan=0.0)
                    out_cols[f"_acc_{stat}_{w}"] = base
                    out_cols[f"_acc_{stat}_centered_{w}"] = cent

    out_cols["_acc_Act"] = np.nan_to_num(s.to_numpy(dtype=float), nan=0.0)
    out_cols["_acc_LogAct"] = np.nan_to_num(np.log1p(np.maximum(s.to_numpy(dtype=float), -0.999999)), nan=0.0)
    return pd.DataFrame(out_cols)


class TSFlexBackend(FeatureBackend):
    """Multi-scale strided rolling/windowed feature backend powered by ``tsflex.features.FeatureCollection``."""

    name = "tsflex"

    _STAT_SPECS = (
        ("mean", np.mean),
        ("std", np.std),
        ("median", np.median),
        ("rms", _rms),
        ("min", np.min),
        ("max", np.max),
        ("iqr", _iqr),
        ("mad", _mad_median),
        ("p10", _p10),
        ("p90", _p90),
        ("skew", _safe_skew),
        ("kurt", _safe_kurt),
    )

    def available(self) -> bool:
        return library_available("tsflex")

    def get_feature_names(self, config: Mapping[str, Any], input_columns: Sequence[str]) -> List[str]:
        sub_scales = config.get("tsflex_subwindow_fracs", (1.0, 0.5))
        names: List[str] = []
        for col in input_columns:
            for frac in sub_scales:
                tag = f"f{int(round(float(frac) * 100))}"
                for stat_name, _ in self._STAT_SPECS:
                    names.append(f"tsflex__{col}__{tag}__{stat_name}")
        return names

    def compute_features(
        self,
        sources: Mapping[str, Mapping[str, Any]],
        config: Mapping[str, Any],
    ) -> Dict[str, np.ndarray]:
        """Compute multi-scale sub-window statistics inside each window using ``tsflex.FeatureCollection``."""
        require_library(self.available(), "tsflex", "--no-deps tsflex")
        from tsflex.features import FeatureCollection, FuncWrapper, MultipleFeatureDescriptors

        sub_scales = tuple(float(f) for f in config.get("tsflex_subwindow_fracs", (1.0, 0.5)))
        min_len = int(config.get("min_window_samples", 8))
        min_obs_frac = float(config.get("min_observed_fraction", 0.0))

        columns: Dict[str, np.ndarray] = {}
        if not sources:
            return columns
        n = len(next(iter(sources.values()))["x"])
        funcs = [FuncWrapper(fn, output_names=sname) for sname, fn in self._STAT_SPECS]

        for src, info in sources.items():
            x_all = np.asarray(info["x"], dtype=np.float64)
            mask = np.asarray(info["mask"], dtype=bool)
            obs = np.asarray(info["obs"], dtype=bool)
            lengths = mask.sum(axis=1)
            obs_counts = (obs & mask).sum(axis=1)

            for frac in sub_scales:
                tag = f"f{int(round(frac * 100))}"
                for sname, _ in self._STAT_SPECS:
                    columns[f"tsflex__{src}__{tag}__{sname}"] = np.zeros(n, dtype=np.float64)

            for i in range(n):
                L = int(lengths[i])
                if L < min_len or obs_counts[i] == 0:
                    continue
                if min_obs_frac > 0.0 and (obs_counts[i] / float(L)) < min_obs_frac:
                    continue
                seg = np.nan_to_num(x_all[i, :L], nan=0.0, posinf=0.0, neginf=0.0)
                s = pd.Series(seg, name=src)
                win_map: Dict[int, List[str]] = {}
                for frac in sub_scales:
                    w = max(4, min(L, int(round(L * frac))))
                    tag = f"f{int(round(frac * 100))}"
                    win_map.setdefault(w, []).append(tag)
                descs = [
                    MultipleFeatureDescriptors(funcs, [src], windows=[w], strides=max(1, w // 2))
                    for w in win_map
                ]
                fc = FeatureCollection(descs)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df_out = fc.calculate(s, return_df=True, approve_sparsity=True)
                for w, tags in win_map.items():
                    for sname, _ in self._STAT_SPECS:
                        tsf_col = f"{src}__{sname}__w={w}"
                        if tsf_col in df_out.columns and len(df_out[tsf_col].dropna()) > 0:
                            # Aggregate sub-windows deterministically via mean (for w=L there is 1 value)
                            val = float(np.nanmean(df_out[tsf_col].to_numpy(dtype=float)))
                            val = val if np.isfinite(val) else 0.0
                        else:
                            val = 0.0
                        for tag in tags:
                            columns[f"tsflex__{src}__{tag}__{sname}"][i] = val

        return columns


SEGLEARN_FEATURE_NAMES = ("mean", "median", "std", "skew", "kurt", "min", "max", "abs_energy", "mean_crossings")


def _build_seglearn_feature_map():
    require_library(library_available("seglearn"), "seglearn")
    from seglearn import feature_functions as sff

    return {
        "mean": sff.mean,
        "median": sff.median,
        "std": sff.std,
        "skew": sff.skew,
        "kurt": sff.kurt,
        "min": sff.minimum,
        "max": sff.maximum,
        "abs_energy": sff.abs_energy,
        "mean_crossings": sff.mean_crossings,
    }


class SeglearnBackend(FeatureBackend):
    """Lightweight segment-level statistical backend backed by ``seglearn.transform.FeatureRep``."""

    name = "seglearn"

    def available(self) -> bool:
        return library_available("seglearn")

    def get_feature_names(self, config: Mapping[str, Any], input_columns: Sequence[str]) -> List[str]:
        return [f"seglearn__{col}__{stat}" for col in input_columns for stat in SEGLEARN_FEATURE_NAMES]

    def compute_features(
        self,
        sources: Mapping[str, Mapping[str, Any]],
        config: Mapping[str, Any],
    ) -> Dict[str, np.ndarray]:
        require_library(self.available(), "seglearn")
        from seglearn.transform import FeatureRep

        fmap = _build_seglearn_feature_map()
        ordered_stats = list(fmap.keys())
        min_len = int(config.get("min_window_samples", 8))
        min_obs_frac = float(config.get("min_observed_fraction", 0.0))

        columns: Dict[str, np.ndarray] = {}
        if not sources:
            return columns
        n = len(next(iter(sources.values()))["x"])

        for src, info in sources.items():
            x_all = np.asarray(info["x"], dtype=np.float64)
            mask = np.asarray(info["mask"], dtype=bool)
            obs = np.asarray(info["obs"], dtype=bool)
            lengths = mask.sum(axis=1)
            obs_counts = (obs & mask).sum(axis=1)

            mat = np.zeros((n, len(ordered_stats)), dtype=np.float64)
            # Group windows by exact unpadded length so seglearn receives zero padded positions
            by_len: Dict[int, List[int]] = {}
            for i in range(n):
                L = int(lengths[i])
                if L < min_len or obs_counts[i] == 0:
                    continue
                if min_obs_frac > 0.0 and (obs_counts[i] / float(L)) < min_obs_frac:
                    continue
                by_len.setdefault(L, []).append(i)

            rep = FeatureRep(features=fmap)
            for L, idxs in by_len.items():
                batch = np.nan_to_num(x_all[idxs, :L], nan=0.0, posinf=0.0, neginf=0.0)[:, :, None]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    transformed = rep.fit_transform(batch)
                mat[idxs] = np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0)

            for j, stat in enumerate(ordered_stats):
                columns[f"seglearn__{src}__{stat}"] = mat[:, j]

        return columns


class SktimeBackend(FeatureBackend):
    """Modular ``sktime`` feature backend supporting ``summary``, ``catch22``, and ``minirocket``."""

    name = "sktime"

    def available(self) -> bool:
        return library_available("sktime")

    def get_feature_names(self, config: Mapping[str, Any], input_columns: Sequence[str]) -> List[str]:
        require_library(self.available(), "sktime")
        mode = str(config.get("sktime_mode", "summary")).lower()
        probe = {
            col: {
                "x": np.sin(np.linspace(0.0, 4.0 * np.pi, 64))[None, :],
                "mask": np.ones((1, 64), dtype=bool),
                "obs": np.ones((1, 64), dtype=bool),
                "fs": float(config.get("default_fs", 100.0)),
            }
            for col in input_columns
        }
        return list(self.compute_features(probe, {**dict(config), "sktime_mode": mode}).keys())

    def compute_features(
        self,
        sources: Mapping[str, Mapping[str, Any]],
        config: Mapping[str, Any],
    ) -> Dict[str, np.ndarray]:
        require_library(self.available(), "sktime")
        mode = str(config.get("sktime_mode", "summary")).lower()
        min_len = int(config.get("min_window_samples", 8))
        min_obs_frac = float(config.get("min_observed_fraction", 0.0))

        columns: Dict[str, np.ndarray] = {}
        if not sources:
            return columns
        n = len(next(iter(sources.values()))["x"])

        if mode == "catch22":
            from sktime.transformations.panel.catch22 import Catch22

            transformer = Catch22(catch24=True)
            for src, info in sources.items():
                x_all = np.asarray(info["x"], dtype=np.float64)
                mask = np.asarray(info["mask"], dtype=bool)
                obs = np.asarray(info["obs"], dtype=bool)
                lengths = mask.sum(axis=1)
                obs_counts = (obs & mask).sum(axis=1)
                rows: List[Any] = [None] * n
                for i in range(n):
                    L = int(lengths[i])
                    if L < min_len or obs_counts[i] == 0:
                        continue
                    if min_obs_frac > 0.0 and (obs_counts[i] / float(L)) < min_obs_frac:
                        continue
                    seg = np.nan_to_num(x_all[i, :L], nan=0.0, posinf=0.0, neginf=0.0).reshape(1, 1, L)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        rows[i] = transformer.fit_transform(seg).iloc[0]
                template = next((r for r in rows if r is not None), None)
                if template is None:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        template = transformer.fit_transform(np.sin(np.linspace(0, 10, 32)).reshape(1, 1, 32)).iloc[0]
                for col_name in template.index:
                    arr = np.zeros(n, dtype=np.float64)
                    for i, r in enumerate(rows):
                        if r is not None:
                            v = float(r[col_name])
                            arr[i] = v if np.isfinite(v) else 0.0
                    columns[f"sktime__catch22__{src}__{col_name}"] = arr
            return columns

        if mode == "minirocket":
            from sktime.transformations.panel.rocket import MiniRocket

            num_kernels = int(config.get("sktime_rocket_kernels", 84))
            seed = int(config.get("random_state", 42))
            for src, info in sources.items():
                x_all = np.asarray(info["x"], dtype=np.float64)
                mask = np.asarray(info["mask"], dtype=bool)
                obs = np.asarray(info["obs"], dtype=bool)
                lengths = mask.sum(axis=1)
                obs_counts = (obs & mask).sum(axis=1)
                # MiniRocket requires fixed-length 3D panel; pad with edge value (never -999) for partial tail
                max_t = x_all.shape[1]
                valid_rows = [
                    i for i in range(n)
                    if int(lengths[i]) >= max(9, min_len)
                    and obs_counts[i] > 0
                    and (min_obs_frac <= 0.0 or obs_counts[i] / float(lengths[i]) >= min_obs_frac)
                ]
                panel = np.zeros((max(1, len(valid_rows)), 1, max(9, max_t)), dtype=np.float32)
                for k, idx in enumerate(valid_rows):
                    L = int(lengths[idx])
                    seg = np.nan_to_num(x_all[idx, :L], nan=0.0, posinf=0.0, neginf=0.0)
                    panel[k, 0, :L] = seg
                    if L < panel.shape[2]:
                        panel[k, 0, L:] = seg[-1]
                rocket = MiniRocket(num_kernels=num_kernels, random_state=seed)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    transformed = rocket.fit_transform(panel)
                n_feats = transformed.shape[1]
                mat = np.zeros((n, n_feats), dtype=np.float64)
                if valid_rows:
                    mat[valid_rows] = np.nan_to_num(transformed.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
                for j in range(n_feats):
                    columns[f"sktime__minirocket__{src}__k{j}"] = mat[:, j]
            return columns

        # Default: sktime SummaryTransformer (per-window quantiles + moments)
        from sktime.transformations.series.summarize import SummaryTransformer

        summary_fn = SummaryTransformer()
        for src, info in sources.items():
            x_all = np.asarray(info["x"], dtype=np.float64)
            mask = np.asarray(info["mask"], dtype=bool)
            obs = np.asarray(info["obs"], dtype=bool)
            lengths = mask.sum(axis=1)
            obs_counts = (obs & mask).sum(axis=1)
            rows = [None] * n
            for i in range(n):
                L = int(lengths[i])
                if L < min_len or obs_counts[i] == 0:
                    continue
                if min_obs_frac > 0.0 and (obs_counts[i] / float(L)) < min_obs_frac:
                    continue
                seg = pd.Series(np.nan_to_num(x_all[i, :L], nan=0.0, posinf=0.0, neginf=0.0))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rows[i] = summary_fn.fit_transform(seg).iloc[0]
            template = next((r for r in rows if r is not None), None)
            if template is None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    template = summary_fn.fit_transform(pd.Series(np.sin(np.linspace(0, 10, 32)))).iloc[0]
            for col_name in template.index:
                arr = np.zeros(n, dtype=np.float64)
                for i, r in enumerate(rows):
                    if r is not None:
                        v = float(r[col_name])
                        arr[i] = v if np.isfinite(v) else 0.0
                columns[f"sktime__summary__{src}__{col_name}"] = arr
        return columns

# ---------------------------------------------------------------------------
# Auto-Register Built-In Honeycomb Cells & OSS Backends
# ---------------------------------------------------------------------------


def _register_builtin_honeycomb_components(registry: FeatureRegistry = FEATURE_REGISTRY) -> None:
    """Register all built-in Honeycomb window feature cells and open-source backends."""
    for cell_cls in BUILTIN_WINDOW_CELLS:
        cell_cls.register(registry)
    for backend_cls in (
        TSFELBackend,
        TSFreshBackend,
        Catch22Backend,
        TSFlexBackend,
        SeglearnBackend,
        SktimeBackend,
    ):
        registry.register_backend(backend_cls())


_register_builtin_honeycomb_components(FEATURE_REGISTRY)


# ===========================================================================
# 7. Sequence Extractor Parent-Child Hierarchy
# ===========================================================================


# ---------------------------------------------------------------------------
# Abstract Parent Orchestrator: BaseSequenceExtractor (HoneycombBase)
# ---------------------------------------------------------------------------


class BaseSequenceExtractor(HoneycombBase, ABC):
    """Shared parent class for both ``SequenceExtractor`` (base_utils) and ``MultiModalSequenceExtractor`` (utils).

    Centralizes provenance manifest construction, dynamic ``FEATURE_REGISTRY``
    family execution, handedness canonicalization dispatch, and extra-family
    token normalization so child extractors never duplicate orchestration glue.
    """

    algo_version: str = ALGO_VERSION
    feature_version: str = FEATURE_VERSION

    def _new_manifest(self, rates: Mapping[str, float]) -> FeatureManifest:
        return FeatureManifest(
            algo_version=self.algo_version,
            feature_version=self.feature_version,
            extractor_class=self.__class__.__name__,
            rates={k: float(v) for k, v in rates.items()},
        )

    @staticmethod
    def _record_manifest_entry(
        manifest: Optional[FeatureManifest],
        *,
        feature_name: str,
        family: str,
        library: str,
        fs: Optional[float],
        units: str = "dimensionless",
        scope: str = "window",
        canonical_name: Optional[str] = None,
        legacy_alias: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if manifest is None:
            return
        manifest.add(
            FeatureManifestEntry(
                feature_name=feature_name,
                family=family,
                library=library,
                fs=float(fs) if fs is not None else None,
                units=units,
                scope=scope,
                version=FEATURE_VERSION,
                canonical_name=canonical_name or feature_name,
                legacy_alias=legacy_alias,
                config=dict(config or {}),
            )
        )

    @staticmethod
    def _compute_registry_features(
        context: Mapping[str, Any],
        config: Mapping[str, Any],
        *,
        manifest: Optional[FeatureManifest] = None,
        scopes: Optional[Sequence[str]] = None,
    ) -> Dict[str, np.ndarray]:
        return FEATURE_REGISTRY.compute_enabled(
            context,
            config,
            manifest=manifest,
            scopes=scopes,
        )

    @staticmethod
    def _apply_handedness_if_enabled(
        df: pd.DataFrame,
        enabled: bool,
        *,
        handedness_col: str = "handedness",
        demographics: Optional[pd.DataFrame] = None,
        subject_col: str = "subject",
        swap_sensors: bool = True,
        mount_flip_180: bool = False,
    ) -> pd.DataFrame:
        if not enabled:
            return df
        return HandednessCanonicalizer(
            handedness_col=handedness_col,
            demographics=demographics,
            subject_col=subject_col,
            swap_sensors=swap_sensors,
            mount_flip_180=mount_flip_180,
        ).transform(df)

    def manifest_to_json(self, path: Optional[Any] = None) -> str:
        manifest = getattr(self, "feature_manifest_", None)
        if manifest is None:
            raise RuntimeError(f"{self.__class__.__name__} has not built feature_manifest_ yet; call fit/transform first.")
        return manifest.to_json(path=path)



class _UnsetType:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    def __copy__(self): return self
    def __deepcopy__(self, memo): return self
    def __repr__(self): return "UNSET"


_UNSET = _UnsetType()


class SequenceExtractor(BaseSequenceExtractor):
    """
    Orchestrates cleaning, motion filtering, and multi-domain extraction.

    Every parameter here is exposed so it is searchable through
    ``GridSearchCV`` / ``BayesSearchCV`` as ``estimator__extractor__<param>``.
    """

    def __init__(
        self,
        # --- accelerometer ---
        acc_modes: str = "raw|velocity|displacement|jerk",
        linear_acc_modes: str = "",
        use_acc_magnitude: bool = False,                    # NEW
        use_linear_acc_magnitude: bool = False,             # NEW
        linear_acc_mode: Optional[str] = None,              # NEW  ('baseline' or None)
        use_highpass_fallback: bool = True,                 # NEW
        acceleration_units: str = "m/s^2",
        linear_acc_frame: str = "world",
        gravity_magnitude: float = 9.80665,

        # --- rotation ---
        rotation_modes: str = "quaternion",
        fix_quaternion_sign: bool = True,                   # NEW

        # --- tof ---
        tof_modes: str = "sensor_stats",
        tof_fill_mode: str = "far_255",                     # NEW  ('far_255','far_500','nan_interpolate','zero')
        tof_n_sensors: int = 5,                             # NEW

        # --- thermo ---
        thm_modes: str = "centered_diff",

        # --- motion filter / dead reckoning ---
        motion_filter_mode: Optional[str] = None,
        use_dead_reckoning: bool = False,
        dead_reckoning_detrend: bool = False,
        kalman_process_noise: float = 1e-3,
        kalman_measurement_noise: float = 1e-2,

        # --- signal cleaning ---
        compute_dt: bool = True,
        window_size: int = 7,
        highpass_window_size: Optional[int] = None,
        smoothing_window_size: Optional[int] = None,
        smooth_alpha: Optional[float] = None,
        clip_value: Optional[float] = None,
        interp_mode: str = "linear",
        integration_highpass_hz: Optional[float] = None,
        allow_raw_gravity_integration: bool = True,
        tof_invalid_sentinels: Tuple[float, ...] = DEFAULT_TOF_INVALID_SENTINELS,
        extra_feature_families: Optional[Any] = None,

        # --- chunking ---
        maxlen: int = 160,
        sequence_crop_mode: str = "tail",
        padding_value: float = -999.0,
        sequence_col: str = "sequence_id",
        counter_col: str = "sequence_counter",

        # --- rates ---
        imu_native_sampling_rate: float = 20,
        imu_target_sampling_rate: float = 20,
        rot_native_sampling_rate: float = 20,
        rot_target_sampling_rate: float = 20,
        tof_native_sampling_rate: float = 5,
        tof_target_sampling_rate: float = 5,
        thm_native_sampling_rate: float = 5,
        thm_target_sampling_rate: float = 5,

        # --- chunking for chunk output ---
        chunk_window_size: Optional[int] = 128,
        chunk_stride: Optional[int] = 64,
        max_windows_per_sequence: Optional[int] = None,
        window_cap_policy: str = "last",
        final_window_mode: str = "pad",
        use_chunk_stride_ratio: bool = False,
        chunk_stride_ratio: float = 0.5,
        output_format: str = "chunks",
        frame_stats: str = "mean,std,min,max,last",
        add_global_context: bool = False,
        resample_modalities: bool = False,

        # --- STFT ---
        stft_nperseg: int = 32,
        stft_noverlap: Optional[int] = None,
        stft_window_type: str = "hann",
        stft_scaling: str = "density",                      # NEW
        stft_detrend: str = "constant",                     # NEW
        stft_use_log_scale: bool = True,
        stft_frequency_bands: Optional[Dict[str, Tuple[float, float]]] = None,  # NEW
        stft_window_seconds: Optional[float] = None,

        # --- CWT ---
        cwt_wavelet: str = "morl",
        cwt_max_scale: int = 128,
        cwt_n_scales: int = 32,
        cwt_use_log_scale: bool = True,
        cwt_min_scale: Optional[float] = None,
        cwt_max_frequency: Optional[float] = None,

        # --- configs (list-of-dicts path) ---
        stft_configs: Optional[List[Dict[str, Any]]] = _UNSET,
        cwt_configs: Optional[List[Dict[str, Any]]] = _UNSET,

        # --- problematic filter ---
        filter_problematic_sequences: bool = False,
        ideal_skew_threshold: float = 1.35,
        problematic_features_threshold: int = 6,
        problematic_feature_cols: Optional[List[str]] = None,
        # --- optional training-stage selection (never applied at prediction) ---
        pretraining_source: str = "non_target",
        pretraining_label_column: str = "gesture",
        pretraining_include_classes: Optional[List[Any]] = None,
        pretraining_exclude_classes: Optional[List[Any]] = None,
        pretraining_filters: Optional[Dict[str, Any]] = None,
        pretraining_exclude_filters: Optional[Dict[str, Any]] = None,
        pretraining_min_length: Optional[int] = None,
        pretraining_max_length: Optional[int] = None,
        pretraining_sample_fraction: float = 1.0,
        pretraining_max_sequences: Optional[int] = None,
        pretraining_max_sequences_per_class: Optional[int] = None,
        pretraining_balance_classes: bool = False,
        pretraining_random_state: Optional[int] = 42,
        finetuning_source: str = "target",
        finetuning_label_column: str = "bfrb",
        finetuning_include_classes: Optional[List[Any]] = None,
        finetuning_exclude_classes: Optional[List[Any]] = None,
        finetuning_filters: Optional[Dict[str, Any]] = None,
        finetuning_exclude_filters: Optional[Dict[str, Any]] = None,
        stage_target_column: str = "is_target",
        stage_target_values: Any = (1,),
    ):
        self.acc_modes = acc_modes
        self.linear_acc_modes = linear_acc_modes
        self.use_acc_magnitude = use_acc_magnitude
        self.use_linear_acc_magnitude = use_linear_acc_magnitude
        self.linear_acc_mode = linear_acc_mode
        self.use_highpass_fallback = use_highpass_fallback
        self.acceleration_units = acceleration_units
        self.linear_acc_frame = linear_acc_frame
        self.gravity_magnitude = gravity_magnitude

        self.rotation_modes = rotation_modes
        self.fix_quaternion_sign = fix_quaternion_sign

        self.tof_modes = tof_modes
        self.tof_fill_mode = tof_fill_mode
        self.tof_n_sensors = tof_n_sensors

        self.thm_modes = thm_modes

        self.motion_filter_mode = motion_filter_mode
        self.use_dead_reckoning = use_dead_reckoning
        self.dead_reckoning_detrend = dead_reckoning_detrend
        self.kalman_process_noise = kalman_process_noise
        self.kalman_measurement_noise = kalman_measurement_noise

        self.compute_dt = compute_dt
        self.window_size = window_size
        self.highpass_window_size = highpass_window_size
        self.smoothing_window_size = smoothing_window_size
        self.smooth_alpha = smooth_alpha
        self.clip_value = clip_value
        self.interp_mode = interp_mode
        self.integration_highpass_hz = integration_highpass_hz
        self.allow_raw_gravity_integration = allow_raw_gravity_integration
        self.tof_invalid_sentinels = tof_invalid_sentinels
        self.extra_feature_families = extra_feature_families

        self.maxlen = maxlen
        self.sequence_crop_mode = sequence_crop_mode
        self.padding_value = padding_value
        self.sequence_col = sequence_col
        self.counter_col = counter_col

        self.imu_native_sampling_rate = imu_native_sampling_rate
        self.imu_target_sampling_rate = imu_target_sampling_rate
        self.rot_native_sampling_rate = rot_native_sampling_rate
        self.rot_target_sampling_rate = rot_target_sampling_rate
        self.tof_native_sampling_rate = tof_native_sampling_rate
        self.tof_target_sampling_rate = tof_target_sampling_rate
        self.thm_native_sampling_rate = thm_native_sampling_rate
        self.thm_target_sampling_rate = thm_target_sampling_rate

        self.chunk_window_size = chunk_window_size
        self.chunk_stride = chunk_stride
        self.max_windows_per_sequence = max_windows_per_sequence
        self.window_cap_policy = window_cap_policy
        self.final_window_mode = final_window_mode
        self.use_chunk_stride_ratio = use_chunk_stride_ratio
        self.chunk_stride_ratio = chunk_stride_ratio
        self.output_format = output_format
        self.frame_stats = frame_stats
        self.add_global_context = add_global_context
        self.resample_modalities = resample_modalities

        self.stft_nperseg = stft_nperseg
        self.stft_noverlap = stft_noverlap
        self.stft_window_type = stft_window_type
        self.stft_scaling = stft_scaling
        self.stft_detrend = stft_detrend
        self.stft_use_log_scale = stft_use_log_scale
        self.stft_frequency_bands = stft_frequency_bands
        self.stft_window_seconds = stft_window_seconds

        self.cwt_wavelet = cwt_wavelet
        self.cwt_max_scale = cwt_max_scale
        self.cwt_n_scales = cwt_n_scales
        self.cwt_use_log_scale = cwt_use_log_scale
        self.cwt_min_scale = cwt_min_scale
        self.cwt_max_frequency = cwt_max_frequency

        self.stft_configs = stft_configs
        self.cwt_configs = cwt_configs

        self.filter_problematic_sequences = filter_problematic_sequences
        self.ideal_skew_threshold = ideal_skew_threshold
        self.problematic_features_threshold = problematic_features_threshold
        self.problematic_feature_cols = problematic_feature_cols

        self.pretraining_source = pretraining_source
        self.pretraining_label_column = pretraining_label_column
        self.pretraining_include_classes = pretraining_include_classes
        self.pretraining_exclude_classes = pretraining_exclude_classes
        self.pretraining_filters = pretraining_filters
        self.pretraining_exclude_filters = pretraining_exclude_filters
        self.pretraining_min_length = pretraining_min_length
        self.pretraining_max_length = pretraining_max_length
        self.pretraining_sample_fraction = pretraining_sample_fraction
        self.pretraining_max_sequences = pretraining_max_sequences
        self.pretraining_max_sequences_per_class = pretraining_max_sequences_per_class
        self.pretraining_balance_classes = pretraining_balance_classes
        self.pretraining_random_state = pretraining_random_state
        self.finetuning_source = finetuning_source
        self.finetuning_label_column = finetuning_label_column
        self.finetuning_include_classes = finetuning_include_classes
        self.finetuning_exclude_classes = finetuning_exclude_classes
        self.finetuning_filters = finetuning_filters
        self.finetuning_exclude_filters = finetuning_exclude_filters
        self.stage_target_column = stage_target_column
        self.stage_target_values = stage_target_values

        self._rebuild_components()

    def set_params(self, **params):
        super().set_params(**params)
        self._rebuild_components()
        return self

    # ---------- helpers ----------

    @staticmethod
    def _selection_values(value):
        value = _coerce_search_value(value)
        if isinstance(value, (list, tuple, set, np.ndarray, pd.Index)):
            return list(value)
        return [value]

    def _stage_metadata(self, X, y, columns):
        """Read sequence-constant metadata without copying the sensor frame."""
        if not isinstance(X, pd.DataFrame) or self.sequence_col not in X:
            raise InvalidExtractorParams(
                f"Training-stage selection requires a DataFrame containing {self.sequence_col!r}."
            )
        metadata = X[[self.sequence_col]].drop_duplicates().set_index(self.sequence_col)
        for column in dict.fromkeys(columns):
            source = X if column in X else y
            if not isinstance(source, pd.DataFrame):
                raise InvalidExtractorParams(f"Training-stage selection requires metadata column {column!r}.")
            if self.sequence_col not in source and source.index.name == self.sequence_col:
                source = source.reset_index()
            if self.sequence_col not in source or column not in source:
                raise InvalidExtractorParams(f"Training-stage selection requires metadata column {column!r}.")
            values = source[[self.sequence_col, column]].drop_duplicates()
            if values[self.sequence_col].duplicated().any():
                raise InvalidExtractorParams(
                    f"Selection column {column!r} must be constant within each sequence."
                )
            metadata[column] = values.set_index(self.sequence_col)[column].reindex(metadata.index)
        return metadata

    def _select_stage_sequences(self, X, y, stage):
        """Select whole original sequences before extraction or augmentation.

        These selectors are opt-in methods for staged estimators. Ordinary
        ``fit``/``transform`` retain their existing behaviour. Include filters
        are combined with AND; any exclude-filter match removes a sequence.
        All sampling operates on original sequences, never individual rows.
        """
        source = str(getattr(self, f"{stage}_source")).strip().lower().replace("-", "_")
        valid_sources = {"all", "target", "non_target"}
        if stage == "pretraining":
            valid_sources.add("baseline")
        if source not in valid_sources:
            raise InvalidExtractorParams(f"{stage}_source must be one of {sorted(valid_sources)}.")
        if source == "baseline":
            metadata = self._stage_metadata(X, y, [])
            self.pretraining_selection_summary_ = {
                "source": source, "input_sequences": int(len(metadata)), "selected_sequences": 0,
            }
            return X.iloc[:0].copy()
        label_column = getattr(self, f"{stage}_label_column")
        include = _coerce_search_value(getattr(self, f"{stage}_include_classes"))
        exclude = _coerce_search_value(getattr(self, f"{stage}_exclude_classes"))
        filters = _coerce_search_value(getattr(self, f"{stage}_filters"))
        exclusions = _coerce_search_value(getattr(self, f"{stage}_exclude_filters"))
        filters = {} if filters is None else filters
        exclusions = {} if exclusions is None else exclusions
        if not isinstance(filters, dict) or not isinstance(exclusions, dict):
            raise InvalidExtractorParams(f"{stage} filters must be dictionaries of column-to-values.")
        require_labels = include is not None or exclude is not None
        if stage == "pretraining":
            require_labels = require_labels or self.pretraining_balance_classes or self.pretraining_max_sequences_per_class is not None
        columns = list(filters) + list(exclusions)
        if source != "all":
            columns.append(self.stage_target_column)
        if require_labels:
            columns.append(label_column)
        metadata = self._stage_metadata(X, y, columns)
        keep = pd.Series(True, index=metadata.index)
        if source != "all":
            target = metadata[self.stage_target_column].isin(self._selection_values(self.stage_target_values))
            known = metadata[self.stage_target_column].notna()
            keep &= known & (target if source == "target" else ~target)
        if include is not None:
            keep &= metadata[label_column].isin(self._selection_values(include))
        if exclude is not None:
            keep &= ~metadata[label_column].isin(self._selection_values(exclude))
        for column, values in filters.items():
            keep &= metadata[column].isin(self._selection_values(values))
        for column, values in exclusions.items():
            keep &= ~metadata[column].isin(self._selection_values(values))
        if stage == "pretraining":
            lengths = X.groupby(self.sequence_col, sort=False).size().reindex(metadata.index)
            minimum = self.pretraining_min_length
            maximum = self.pretraining_max_length
            for name, value in (("pretraining_min_length", minimum), ("pretraining_max_length", maximum)):
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1):
                    raise InvalidExtractorParams(f"{name} must be a positive integer or None.")
            if minimum is not None and maximum is not None and minimum > maximum:
                raise InvalidExtractorParams("pretraining_min_length cannot exceed pretraining_max_length.")
            if minimum is not None:
                keep &= lengths >= minimum
            if maximum is not None:
                keep &= lengths <= maximum
        selected = metadata.loc[keep]
        if stage == "pretraining":
            fraction = self.pretraining_sample_fraction
            if isinstance(fraction, bool) or not isinstance(fraction, (int, float, np.number)) or not np.isfinite(fraction) or not 0 < fraction <= 1:
                raise InvalidExtractorParams("pretraining_sample_fraction must lie in (0, 1].")
            for name in ("pretraining_max_sequences", "pretraining_max_sequences_per_class"):
                value = getattr(self, name)
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1):
                    raise InvalidExtractorParams(f"{name} must be a positive integer or None.")
            if len(selected):
                rng = np.random.default_rng(self.pretraining_random_state)
                if self.pretraining_balance_classes or self.pretraining_max_sequences_per_class is not None:
                    if selected[label_column].isna().any():
                        raise InvalidExtractorParams("Class-balanced pretraining requires non-missing class labels.")
                    groups = selected.groupby(label_column, sort=False, observed=True).indices
                    limit = min(map(len, groups.values())) if self.pretraining_balance_classes else len(selected)
                    if self.pretraining_max_sequences_per_class is not None:
                        limit = min(limit, self.pretraining_max_sequences_per_class)
                    chosen = np.concatenate([rng.choice(indices, size=min(limit, len(indices)), replace=False) for indices in groups.values()])
                    selected = selected.iloc[np.sort(chosen)]
                count = max(1, int(np.floor(len(selected) * fraction)))
                if self.pretraining_max_sequences is not None:
                    count = min(count, self.pretraining_max_sequences)
                if count < len(selected):
                    chosen = np.sort(rng.choice(len(selected), size=count, replace=False))
                    selected = selected.iloc[chosen]
        selected_ids = selected.index
        setattr(self, f"{stage}_selection_summary_", {
            "source": source,
            "input_sequences": int(len(metadata)),
            "selected_sequences": int(len(selected_ids)),
        })
        return X.loc[X[self.sequence_col].isin(selected_ids)].copy()

    def select_pretraining_sequences(self, X, y=None):
        """Apply configurable source/class/metadata/length sampling for pretraining."""
        return self._select_stage_sequences(X, y, "pretraining")

    def select_finetuning_sequences(self, X, y=None):
        """Select the supervised target task without changing inference inputs."""
        return self._select_stage_sequences(X, y, "finetuning")

    def _effective_column_rates(self) -> Dict[str, float]:
        imu_fs = float(self.imu_target_sampling_rate if self.resample_modalities else self.imu_native_sampling_rate)
        rot_fs = float(self.rot_target_sampling_rate if self.resample_modalities else self.rot_native_sampling_rate)
        thm_fs = float(self.thm_target_sampling_rate if self.resample_modalities else self.thm_native_sampling_rate)
        tof_fs = float(self.tof_target_sampling_rate if self.resample_modalities else self.tof_native_sampling_rate)
        return {
            "acc_": imu_fs,
            "lin_acc_": imu_fs,
            "rot_": rot_fs,
            "rot6d_": rot_fs,
            "thm_": thm_fs,
            "tof_": tof_fs,
        }

    def _build_stft_defaults(self) -> Dict[str, Any]:
        eff_imu_fs = float(self.imu_target_sampling_rate if self.resample_modalities else self.imu_native_sampling_rate)
        return dict(
            nperseg=self.stft_nperseg,
            noverlap=self.stft_noverlap,
            window_type=self.stft_window_type,
            scaling=self.stft_scaling,
            detrend=self.stft_detrend,
            use_log_scale=self.stft_use_log_scale,
            frequency_bands=_coerce_search_value(self.stft_frequency_bands),
            sampling_rate=eff_imu_fs,
            column_sampling_rates=self._effective_column_rates(),
            window_seconds=self.stft_window_seconds,
            counter_col=self.counter_col,
            sequence_col=self.sequence_col,
        )

    def _build_cwt_defaults(self) -> Dict[str, Any]:
        eff_imu_fs = float(self.imu_target_sampling_rate if self.resample_modalities else self.imu_native_sampling_rate)
        return dict(
            wavelet=self.cwt_wavelet,
            max_scale=self.cwt_max_scale,
            n_scales=self.cwt_n_scales,
            use_log_scale=self.cwt_use_log_scale,
            min_scale=self.cwt_min_scale,
            max_frequency=self.cwt_max_frequency,
            sampling_rate=eff_imu_fs,
            column_sampling_rates=self._effective_column_rates(),
            counter_col=self.counter_col,
            sequence_col=self.sequence_col,
        )

    def _build_time_frequency_extractors(self) -> None:
        stft_defaults = self._build_stft_defaults()
        cwt_defaults = self._build_cwt_defaults()

        stft_raw = self.stft_configs
        cwt_raw = self.cwt_configs

        if stft_raw is _UNSET:
            stft_configs = [{}]
        elif stft_raw is None:
            stft_configs = []
        else:
            stft_configs = stft_raw

        if cwt_raw is _UNSET:
            cwt_configs = [{}]
        elif cwt_raw is None:
            cwt_configs = []
        else:
            cwt_configs = cwt_raw

        if isinstance(stft_configs, str):
            stft_configs = _coerce_search_value(stft_configs)
        if isinstance(cwt_configs, str):
            cwt_configs = _coerce_search_value(cwt_configs)

        self.stft_extractors = []
        self.cwt_extractors = []
        for i, config in enumerate(stft_configs or []):
            config = dict(config)
            name = config.pop("name", f"stft{i}")
            self.stft_extractors.append(STFTExtractor(**(stft_defaults | config), feature_prefix=name))
        for i, config in enumerate(cwt_configs or []):
            config = dict(config)
            name = config.pop("name", f"cwt{i}")
            self.cwt_extractors.append(CWTExtractor(**(cwt_defaults | config), feature_prefix=name))

        self.stft_extractor = self.stft_extractors[0] if self.stft_extractors else None
        self.cwt_extractor = self.cwt_extractors[0] if self.cwt_extractors else None

    def _rebuild_components(self) -> None:
        """Rebuild every sub-extractor from current params (fit and __init__)."""
        self.cleaner = SignalCleaner(
            native_sampling_rate=self.imu_native_sampling_rate,
            rot_native_sampling_rate=self.rot_native_sampling_rate,
            compute_dt=self.compute_dt,
            clip_value=self.clip_value,
            interp_mode=self.interp_mode,
            linear_acc_mode=self.linear_acc_mode,
            use_highpass_fallback=self.use_highpass_fallback,
            acceleration_units=self.acceleration_units,
            linear_acc_frame=self.linear_acc_frame,
            gravity_magnitude=self.gravity_magnitude,
            sequence_col=self.sequence_col,
            counter_col=self.counter_col,
            window_size=self.window_size,
            highpass_window_size=self.highpass_window_size,
        )

        self.motion_filter = MotionFilter(
            motion_filter_mode=self.motion_filter_mode,
            kalman_process_noise=self.kalman_process_noise,
            kalman_measurement_noise=self.kalman_measurement_noise,
            use_dead_reckoning=(self.use_dead_reckoning and (
                type(self) is not SequenceExtractor
                or _sequence_dead_reckoning_consumed(self.get_params(deep=False))
            )),
            dead_reckoning_detrend=self.dead_reckoning_detrend,
            sequence_col=self.sequence_col,
        )

        self.imu = IMUExtractor(
            acc_modes=self.acc_modes,
            linear_acc_modes=self.linear_acc_modes,
            use_acc_magnitude=self.use_acc_magnitude,
            use_linear_acc_magnitude=self.use_linear_acc_magnitude,
            window_size=self.window_size,
            smoothing_window_size=self.smoothing_window_size,
            smooth_alpha=self.smooth_alpha,
            integration_highpass_hz=self.integration_highpass_hz,
            allow_raw_gravity_integration=self.allow_raw_gravity_integration,
            sequence_col=self.sequence_col,
        )

        self.rotation = RotationExtractor(
            rotation_modes=self.rotation_modes,
            fix_quaternion_sign=self.fix_quaternion_sign,
            sequence_col=self.sequence_col,
        )

        self.tof = TOFExtractor(
            tof_modes=self.tof_modes,
            tof_fill_mode=self.tof_fill_mode,
            invalid_sentinels=self.tof_invalid_sentinels,
            sequence_col=self.sequence_col,
            n_sensors=self.tof_n_sensors,
        )

        self.thermo = ThermoExtractor(
            thm_modes=self.thm_modes,
            sequence_col=self.sequence_col,
        )

        self._build_time_frequency_extractors()

    def _non_feature_cols(self) -> List[str]:
        cols = {self.sequence_col, self.counter_col, "dt", "rot_dt", "mask"}
        return [c for c in cols if c is not None]

    def _parse_frame_stats(self) -> List[str]:
        known = {"mean","std","min","max","first","last","median","rms","abs_mean"}
        raw = _coerce_search_value(self.frame_stats)
        if raw is None or (isinstance(raw, str) and raw.strip().lower() in {"", "none", "nan"}):
            return ["mean"]
        if isinstance(raw, (list, tuple)):
            tokens = [str(s).strip().lower() for s in raw]
        else:
            tokens = [s.strip().lower() for s in str(raw).split(",")]
        stats = [s for s in tokens if s in known]
        if not stats:
            stats = ["mean"]
        return stats

    def _maybe_resample(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.resample_modalities:
            return df
        if resample_poly is None:
            raise ImportError("resample_modalities=True requires scipy.signal.resample_poly")
        try:
            return self._maybe_resample_inner(df)
        except InvalidExtractorParams:
            raise
        except Exception as exc:
            raise InvalidExtractorParams(f"Multimodal resampling failed: {exc}") from exc

    def _maybe_resample_inner(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rate-convert then align back to the shared fixed-length sequence grid.

        This is normalized-time feature preparation, not a change to the sequence's
        physical duration or output sampling rate. The source intervals remain attached.
        """
        rows = []
        rate_map = {
            "acc_": (self.imu_native_sampling_rate, self.imu_target_sampling_rate),
            "rot_": (self.rot_native_sampling_rate, self.rot_target_sampling_rate),
            "tof_": (self.tof_native_sampling_rate, self.tof_target_sampling_rate),
            "thm_": (self.thm_native_sampling_rate, self.thm_target_sampling_rate),
        }
        for seq_id, g in df.groupby(self.sequence_col, sort=False):
            if len(g) == 0:
                continue
            max_len = len(g)
            new_data = {self.sequence_col: [seq_id] * max_len}
            if self.counter_col in g.columns:
                counter = pd.to_numeric(g[self.counter_col], errors="coerce").to_numpy(dtype=float)
                if not np.isfinite(counter).all() or (len(counter) > 1 and np.any(np.diff(counter) <= 0)):
                    raise ValueError("Resampling requires strictly increasing finite sequence counters")
                if len(counter) > 1 and np.any(np.diff(counter) != 1):
                    raise ValueError(
                        "Polyphase resampling requires a uniform input grid; interpolate missing "
                        "counter positions before enabling resample_modalities"
                    )
                new_data[self.counter_col] = g[self.counter_col].to_numpy()
            else:
                counter = np.arange(len(g), dtype=float)
            for prefix, (native_rate, target_rate) in rate_map.items():
                cols = [
                    c for c in g.columns
                    if c.startswith(prefix)
                    and c not in {self.sequence_col, self.counter_col, "dt", "rot_dt", "mask"}
                ]
                if not cols:
                    continue
                native_rate = _positive_rate(native_rate, f"{prefix} native sampling rate")
                target_rate = _positive_rate(target_rate, f"{prefix} target sampling rate")
                ratio = target_rate / native_rate
                target_len = max(1, int(round(len(g) * ratio)))
                if target_len <= 0:
                    continue
                vals = g[cols].to_numpy(dtype=float).copy()
                if prefix == "tof_":
                    valid = np.isfinite(vals) & (vals != -1.0)
                    fill_value = 500.0 if self.tof_fill_mode == "far_500" else 255.0
                    if target_len != len(g):
                        source_samples = np.arange(len(g), dtype=float)
                        for col_idx in range(vals.shape[1]):
                            valid_samples = np.flatnonzero(valid[:, col_idx])
                            vals[:, col_idx] = (
                                np.interp(
                                    source_samples,
                                    valid_samples,
                                    vals[valid_samples, col_idx],
                                )
                                if len(valid_samples)
                                else fill_value
                            )
                    else:
                        vals = np.where(valid, vals, fill_value)
                elif not np.isfinite(vals).all():
                    raise ValueError(
                        f"Cannot resample {prefix} columns with missing values; "
                        "apply the sequence-local imputation policy first"
                    )
                if target_len != len(g):
                    if prefix == "rot_":
                        if Rotation is None or Slerp is None:
                            raise ImportError("Quaternion resampling requires scipy.spatial.transform")
                        q_cols = ["rot_w", "rot_x", "rot_y", "rot_z"]
                        if not all(col in cols for col in q_cols):
                            raise ValueError("Rotation resampling requires rot_w/x/y/z quaternion columns")
                        q = g[q_cols].to_numpy(dtype=float)
                        norms = np.linalg.norm(q, axis=1, keepdims=True)
                        if not np.isfinite(q).all() or np.any(norms <= 0):
                            raise ValueError("Quaternion resampling requires finite nonzero quaternions")
                        q = q / norms
                        for i in range(1, len(q)):
                            if np.dot(q[i - 1], q[i]) < 0:
                                q[i] *= -1.0
                        source_times = (counter - counter[0]) / native_rate
                        target_times = np.linspace(source_times[0], source_times[-1], target_len)
                        if len(q) == 1:
                            q_aligned = np.repeat(q, max_len, axis=0)
                        else:
                            rotations = Rotation.from_quat(q[:, [1, 2, 3, 0]])
                            q_target = Slerp(source_times, rotations)(target_times).as_quat()[:, [3, 0, 1, 2]]
                            if target_len == 1:
                                q_aligned = np.repeat(q_target, max_len, axis=0)
                            else:
                                aligned_rotations = Slerp(target_times, Rotation.from_quat(
                                    q_target[:, [1, 2, 3, 0]]
                                ))(source_times)
                                q_aligned = aligned_rotations.as_quat()[:, [3, 0, 1, 2]]
                        vals = np.column_stack([
                            q_aligned[:, q_cols.index(col)] if col in q_cols
                            else np.interp(
                                np.linspace(0.0, 1.0, max_len),
                                np.linspace(0.0, 1.0, target_len),
                                np.interp(
                                    np.linspace(0.0, 1.0, target_len),
                                    np.linspace(0.0, 1.0, len(g)),
                                    g[col].to_numpy(dtype=float),
                                ),
                            )
                            for col in cols
                        ])
                    else:
                        ratio_fraction = Fraction(ratio).limit_denominator(1000)
                        filtered = resample_poly(
                            vals,
                            ratio_fraction.numerator,
                            ratio_fraction.denominator,
                            axis=0,
                        )
                        target_grid = np.linspace(0.0, 1.0, target_len)
                        filtered_grid = np.linspace(0.0, 1.0, len(filtered))
                        rate_converted = np.column_stack([
                            np.interp(target_grid, filtered_grid, filtered[:, col_idx])
                            for col_idx in range(filtered.shape[1])
                        ])
                        source_grid = np.linspace(0.0, 1.0, max_len)
                        vals = np.column_stack([
                            np.interp(source_grid, target_grid, rate_converted[:, col_idx])
                            for col_idx in range(rate_converted.shape[1])
                        ])
                        if prefix == "tof_":
                            vals[~valid] = -1.0
                elif prefix == "tof_":
                    vals[~valid] = -1.0
                for col_idx, col in enumerate(cols):
                    new_data[col] = vals[:, col_idx]
            if "dt" in g.columns:
                new_data["dt"] = g["dt"].to_numpy(dtype=float)
            else:
                new_data["dt"] = 1.0 / float(self.imu_native_sampling_rate)
            if "rot_dt" in g.columns:
                new_data["rot_dt"] = g["rot_dt"].to_numpy(dtype=float)
            else:
                new_data["rot_dt"] = 1.0 / float(self.rot_native_sampling_rate)
            new_data["mask"] = 1.0
            rows.append(pd.DataFrame(new_data))
        if not rows:
            return df
        return pd.concat(rows, ignore_index=True)

    def _combine_feature_outputs(self, processed: pd.DataFrame, add_context: bool = False) -> pd.DataFrame:
        outs = []
        for est in (self.imu, self.rotation, self.tof, self.thermo, *self.stft_extractors, *self.cwt_extractors):
            out = est.transform(processed)
            if out is not None and not out.empty:
                outs.append(out)
        if outs:
            combined = pd.concat(outs, axis=1)
        else:
            combined = pd.DataFrame(index=processed.index)
        combined = combined.replace([np.inf, -np.inf, np.nan], 0.0)
        if combined.columns.duplicated().any():
            combined = combined.loc[:, ~combined.columns.duplicated()]
        if self.sequence_col in processed.columns:
            combined[self.sequence_col] = processed[self.sequence_col].values
        elif self.sequence_col not in combined.columns:
            combined[self.sequence_col] = 0
        augmentor = getattr(self, "_training_feature_augmentor", None)
        if (augmentor is not None and augmentor._diagnostics_enabled()
                and self.counter_col in processed):
            # Diagnostic metadata is excluded from the model schema. Preserve
            # the processed timeline, including resampling, without a row cache.
            combined[self.counter_col] = processed[self.counter_col].values
        # Execute any registered plugin feature families enabled via extra_feature_families
        if self.extra_feature_families:
            cfg = self.get_params(deep=False)
            eff_rates = self._effective_column_rates()
            rates_dict = {
                "imu": eff_rates["acc_"],
                "rot": eff_rates["rot_"],
                "thm": eff_rates["thm_"],
                "tof": eff_rates["tof_"],
            }
            # 1. Sample-scoped plugins
            sample_ctx = {
                "processed": processed,
                "combined": combined,
                "sequence_col": self.sequence_col,
                "counter_col": self.counter_col,
                "rates": rates_dict,
            }
            extra_sample = FEATURE_REGISTRY.compute_enabled(sample_ctx, cfg, scopes=("sample",))
            for col_name, values in (extra_sample or {}).items():
                if len(values) == len(combined):
                    combined[col_name] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

            # 2. Window-scoped plugins evaluated per sequence and broadcast to sequence rows
            seq_groups = processed.groupby(self.sequence_col, sort=False).indices
            imu_cols = [c for c in ("acc_x", "acc_y", "acc_z", "acc_mag", "rot_w", "rot_x", "rot_y", "rot_z") if c in processed.columns or c in combined.columns]
            if seq_groups and imu_cols:
                max_len = max(len(pos) for pos in seq_groups.values())
                n_seqs = len(seq_groups)
                imu_arr = np.zeros((n_seqs, max_len, len(imu_cols)), dtype=np.float64)
                imu_mask = np.zeros((n_seqs, max_len), dtype=bool)
                seq_pos_list = list(seq_groups.values())
                for si, pos in enumerate(seq_pos_list):
                    L = len(pos)
                    imu_mask[si, :L] = True
                    for cj, col in enumerate(imu_cols):
                        src_df = processed if col in processed.columns else combined
                        imu_arr[si, :L, cj] = pd.to_numeric(src_df[col].iloc[pos], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
                win_ctx = {
                    "branches": {"imu": imu_arr},
                    "masks": {"imu": imu_mask},
                    "channel_names": {"imu": imu_cols},
                    "present": np.ones((n_seqs, 3), dtype=bool),
                    "obs_axes": np.repeat(imu_mask[:, :, None], 3, axis=2),
                    "q_obs": imu_mask.copy(),
                    "rates": rates_dict,
                    "sequence_ids": np.asarray(list(seq_groups.keys())),
                }
                extra_win = FEATURE_REGISTRY.compute_enabled(win_ctx, cfg, scopes=("window",))
                for col_name, values in (extra_win or {}).items():
                    if len(values) == n_seqs:
                        row_vals = np.zeros(len(combined), dtype=np.float64)
                        for si, pos in enumerate(seq_pos_list):
                            v = float(values[si])
                            row_vals[pos] = v if np.isfinite(v) else 0.0
                        combined[col_name] = row_vals
        if add_context:
            feature_cols = [c for c in combined.columns if c not in self._non_feature_cols()]
            combined = self._append_global_context(combined, feature_cols)
        return combined

    def _append_global_context(self, out: pd.DataFrame, base_cols: List[str]) -> pd.DataFrame:
        columns = [col for col in base_cols if col in out.columns]
        if not columns:
            return out
        grouped = out.groupby(self.sequence_col, sort=False)
        # Aggregate once per sequence, then broadcast the small table in row order.
        # Constructing one context block avoids two groupings and inserts per feature.
        summary = grouped[columns].agg(["mean", "std"])
        summary.iloc[:, 1::2] = summary.iloc[:, 1::2].fillna(0.0)
        codes = grouped.ngroup().to_numpy()
        present = pd.notna(codes)
        context_values = np.full((len(out), 2 * len(columns)), np.nan)
        context_values[present] = summary.to_numpy()[codes[present].astype(int)]
        context_values[~present, 1::2] = 0.0
        context = pd.DataFrame(
            context_values,
            columns=[f"{column}_global_{stat}" for column in columns for stat in ("mean", "std")],
            index=out.index,
        )
        return pd.concat([out, context], axis=1)

    def _preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.filter_problematic_sequences:
            X = self.problematic_filter_.transform(X)
            if X.empty:
                raise InvalidExtractorParams("Problematic-sequence filtering removed every input row.")
        cleaned = self.cleaner.transform(X)
        filtered = self.motion_filter.transform(cleaned)
        del cleaned
        processed = self._maybe_resample(filtered)
        del filtered
        combined = self._combine_feature_outputs(
            processed,
            add_context=self.add_global_context and str(self.output_format).lower() != "frame",
        )
        return self._apply_feature_augmentation(combined)

    def _apply_feature_augmentation(self, combined):
        """Training-only hook, installed temporarily by the sequence classifier.

        Geometry, resampling and spectral calculations have finished here. Zero
        rotation features are now missing model inputs, never raw quaternions
        passed to gravity removal or rotation differentiation.
        """
        augmentor = getattr(self, "_training_feature_augmentor", None)
        if augmentor is None:
            return combined
        columns = [c for c in combined if c not in self._non_feature_cols()]
        try:
            return augmentor.transform_features(combined, feature_columns=columns)
        except Exception as exc:
            exc.failure_stage = "augmentation"
            raise

    def _stat_vector(self, arr: np.ndarray, stat: str) -> np.ndarray:
        if arr.size == 0:
            return np.zeros((arr.shape[1] if arr.ndim == 2 else 1,), dtype=float)
        if stat == "mean": return np.nanmean(arr, axis=0)
        if stat == "std": return np.nanstd(arr, axis=0)
        if stat == "min": return np.nanmin(arr, axis=0)
        if stat == "max": return np.nanmax(arr, axis=0)
        if stat == "first": return arr[0]
        if stat == "last": return arr[-1]
        if stat == "median": return np.nanmedian(arr, axis=0)
        if stat == "rms": return np.sqrt(np.nanmean(np.square(arr), axis=0))
        if stat == "abs_mean": return np.nanmean(np.abs(arr), axis=0)
        return np.nanmean(arr, axis=0)

    # ---------- fit ----------

    def _normalize_chunk_stride(self) -> None:
        if str(self.output_format).lower() == "frame":
            return

        window = self.chunk_window_size
        if window is not None:
            try:
                window = int(window)
            except (TypeError, ValueError) as exc:
                raise InvalidExtractorParams(
                    f"chunk_window_size must be an integer, got {self.chunk_window_size!r}"
                ) from exc
            if window <= 0:
                window = None

        if self.use_chunk_stride_ratio:
            try:
                ratio = float(self.chunk_stride_ratio)
            except (TypeError, ValueError) as exc:
                raise InvalidExtractorParams(
                    f"chunk_stride_ratio must be between 0 and 1, got {self.chunk_stride_ratio!r}"
                ) from exc
            if not 0 < ratio <= 1:
                raise InvalidExtractorParams(
                    f"chunk_stride_ratio must be between 0 and 1, got {ratio!r}"
                )
            if window is not None:
                self.chunk_stride = max(1, min(window, int(round(window * ratio))))
        elif self.chunk_stride is not None:
            try:
                stride = int(self.chunk_stride)
            except (TypeError, ValueError) as exc:
                raise InvalidExtractorParams(
                    f"chunk_stride must be an integer, got {self.chunk_stride!r}"
                ) from exc
            if stride > 0 and window is not None:
                self.chunk_stride = min(stride, window)

    def _fit_features(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        self._normalize_chunk_stride()

        validate_sequence_extractor_params(
            self.get_params(),
            for_frame_output=str(self.output_format).lower() == "frame",
        )

        # Rebuild every sub-extractor from current params
        self._rebuild_components()

        self.problematic_filter_ = ProblematicSequenceFilter(
            ideal_skew_threshold=self.ideal_skew_threshold,
            problematic_features_threshold=self.problematic_features_threshold,
            feature_cols=_coerce_search_value(self.problematic_feature_cols),
            sequence_col=self.sequence_col,
        )
        if self.filter_problematic_sequences:
            self.problematic_filter_.fit(X)
        X_fit = self.problematic_filter_.transform(X) if self.filter_problematic_sequences else X
        if X_fit.empty:
            raise InvalidExtractorParams("Problematic-sequence filtering removed every training row.")

        cleaned = self.cleaner.fit_transform(X_fit)
        filtered = self.motion_filter.fit_transform(cleaned)
        del cleaned
        processed = self._maybe_resample(filtered)
        del filtered

        self.imu.fit(processed)
        self.rotation.fit(processed)
        self.tof.fit(processed)
        self.thermo.fit(processed)
        for extractor in (*self.stft_extractors, *self.cwt_extractors):
            extractor.fit(processed)

        combined = self._combine_feature_outputs(processed, add_context=False)

        feature_cols = [c for c in combined.columns if c not in self._non_feature_cols()]

        if not feature_cols:
            combined["constant_zero"] = 0.0
            feature_cols = ["constant_zero"]

        self.base_feature_names_ = list(feature_cols)
        self.frame_stats_ = self._parse_frame_stats()
        self.frame_feature_names_ = [
            f"{c}_{s}"
            for s in self.frame_stats_
            for c in self.base_feature_names_
        ]

        if self.add_global_context:
            self.feature_names_in_ = self.base_feature_names_ + [
                f"{column}_global_{stat}"
                for column in self.base_feature_names_
                for stat in ("mean", "std")
            ]
        else:
            self.feature_names_in_ = list(self.base_feature_names_)

        # Emit provenance manifest for all produced base features
        eff_rates = self._effective_column_rates()
        manifest = FeatureManifest(
            algo_version=ALGO_VERSION,
            feature_version=FEATURE_VERSION,
            extractor_class=type(self).__name__,
            rates={
                "imu": eff_rates["acc_"],
                "rot": eff_rates["rot_"],
                "thm": eff_rates["thm_"],
                "tof": eff_rates["tof_"],
            },
        )
        for col in self.feature_names_in_:
            col_fs = infer_column_rate(
                col,
                imu_rate=eff_rates["acc_"],
                rot_rate=eff_rates["rot_"],
                thm_rate=eff_rates["thm_"],
                tof_rate=eff_rates["tof_"],
            )
            family = (
                "stft" if "_stft" in col else
                "cwt" if "_cwt" in col else
                "tof" if col.startswith("tof_") else
                "thermal" if col.startswith("thm_") else
                "rotation" if col.startswith(("rot_", "rot6d_")) else
                "imu"
            )
            manifest.add(
                FeatureManifestEntry(
                    feature_name=col,
                    canonical_name=col,
                    family=family,
                    library="scipy/numpy",
                    fs=col_fs,
                    units="sensor_native",
                    scope="sample" if str(self.output_format).lower() != "frame" else "sequence",
                    version=FEATURE_VERSION,
                )
            )
        self.feature_manifest_ = manifest

        return combined

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None):
        self._fit_features(X, y)
        return self

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params):
        """Reuse fitted training features without keeping a dataset cache on the estimator."""
        combined = self._fit_features(X, y, **fit_params)
        if str(self.output_format).lower() == "frame":
            return self._frame_from_features(self._apply_feature_augmentation(combined))
        if self.add_global_context:
            combined = self._append_global_context(combined, self.base_feature_names_)
        return self._chunks_from_features(self._apply_feature_augmentation(combined))

    # ---------- transform ----------

    def transform(self, X: pd.DataFrame):
        check_is_fitted(self, ["base_feature_names_", "feature_names_in_"])
        if str(self.output_format).lower() == "frame":
            return self.transform_frame(X)
        return self.transform_chunks(X)

    def transform_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["base_feature_names_", "frame_feature_names_"])
        return self._frame_from_features(self._preprocess_features(X))

    def _frame_from_features(self, out: pd.DataFrame) -> pd.DataFrame:
        records = []
        seq_ids = []
        for seq_id, g in out.groupby(self.sequence_col, sort=True):
            if len(g) == 0:
                continue
            for col in self.base_feature_names_:
                if col not in g.columns:
                    g[col] = 0.0
            arr = g[self.base_feature_names_].to_numpy(dtype=float)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            if arr.size == 0:
                arr = np.zeros((1, len(self.base_feature_names_)), dtype=float)
            arr = self._crop_sequence(arr)
            stat_vectors = [self._stat_vector(arr, stat) for stat in self.frame_stats_]
            row = np.concatenate(stat_vectors)
            row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
            records.append(row)
            seq_ids.append(seq_id)
        if not records:
            raise ValueError("SequenceExtractor.transform_frame produced no sequences.")
        frame = pd.DataFrame(
            np.vstack(records),
            columns=self.frame_feature_names_,
            index=pd.Index(seq_ids, name=self.sequence_col),
        )
        self.sequence_ids_ = frame.index.values
        return frame

    def _crop_sequence(self, arr: np.ndarray) -> np.ndarray:
        if self.sequence_crop_mode == "none" or self.maxlen is None:
            return arr
        try:
            maxlen = int(self.maxlen)
        except (TypeError, ValueError) as exc:
            raise InvalidExtractorParams(f"maxlen must be an integer, got {self.maxlen!r}") from exc
        if maxlen <= 0 or len(arr) <= maxlen:
            return arr
        if self.sequence_crop_mode == "head":
            return arr[:maxlen]
        if self.sequence_crop_mode == "tail":
            return arr[-maxlen:]
        if self.sequence_crop_mode == "center":
            crop_start = max(0, (len(arr) - maxlen) // 2)
            return arr[crop_start : crop_start + maxlen]
        raise InvalidExtractorParams(
            "sequence_crop_mode must be one of 'head', 'tail', 'center', or 'none'"
        )

    def transform_chunks(self, X: pd.DataFrame) -> Dict[str, Any]:
        check_is_fitted(self, ["feature_names_in_"])
        self._normalize_chunk_stride()
        validate_sequence_extractor_params(self.get_params(), for_frame_output=False)
        return self._chunks_from_features(self._preprocess_features(X))

    def _chunks_from_features(self, out: pd.DataFrame) -> Dict[str, Any]:
        for col in self.feature_names_in_:
            if col not in out.columns:
                out[col] = 0.0

        grouped_positions = out.groupby(self.sequence_col, sort=False).indices
        if not grouped_positions:
            raise ValueError("SequenceExtractor.transform_chunks produced no sequences.")

        maxlen = None
        if self.maxlen is not None:
            try:
                parsed_maxlen = int(self.maxlen)
            except (TypeError, ValueError) as exc:
                raise InvalidExtractorParams(f"maxlen must be an integer, got {self.maxlen!r}") from exc
            if parsed_maxlen > 0 and self.sequence_crop_mode != "none":
                maxlen = parsed_maxlen

        window = self.chunk_window_size
        if window is not None:
            try:
                window = int(window)
            except (TypeError, ValueError) as exc:
                raise InvalidExtractorParams(
                    f"chunk_window_size must be an integer, got {self.chunk_window_size!r}"
                ) from exc
        if window is None or window <= 0:
            window = max(
                min(len(positions), maxlen) if maxlen is not None else len(positions)
                for positions in grouped_positions.values()
            )
        window = max(1, int(window))

        if self.use_chunk_stride_ratio:
            stride = max(1, int(round(window * float(self.chunk_stride_ratio))))
        else:
            stride = self.chunk_stride
            stride = window if stride is None or int(stride) <= 0 else int(stride)
        stride = min(stride, window)

        window_cap = self.max_windows_per_sequence
        if window_cap is not None:
            window_cap = _positive_int(window_cap, name="max_windows_per_sequence")

        sequences, masks, seq_ids = [], [], []
        sequences_total = 0
        capped_sequences = 0
        sequences_at_cap = 0
        max_windows_before_cap = 0
        max_windows_after_cap = 0

        # Store only feature columns; groupby.indices avoids retaining a DataFrame
        # for every sequence alongside the complete preprocessed feature table.
        features = out[self.feature_names_in_]
        for seq_id, positions in grouped_positions.items():
            if len(positions) == 0:
                continue
            sequences_total += 1
            arr = features.iloc[positions].to_numpy(dtype=float)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            full_length = len(arr)
            # Chunk values are returned as float32. Convert before retaining
            # window views so stacking does not allocate a full float64 tensor.
            arr = self._crop_sequence(arr).astype(np.float32)
            length = len(arr)
            if length == 0:
                continue

            if length <= window:
                starts = np.array([0], dtype=int)
            elif self.final_window_mode == "pad":
                starts = np.arange(0, length, stride, dtype=int)
            else:
                starts = np.arange(0, length - window + 1, stride, dtype=int)
                if len(starts) == 0 or starts[-1] + window < length:
                    starts = np.append(starts, length - window)

            windows_before_cap = len(starts)
            max_windows_before_cap = max(max_windows_before_cap, windows_before_cap)

            if window_cap is not None and windows_before_cap > window_cap:
                capped_sequences += 1
                if self.window_cap_policy == "last":
                    starts = starts[-window_cap:]
                elif self.window_cap_policy == "first":
                    starts = starts[:window_cap]
                else:
                    indices = np.rint(np.linspace(0, len(starts) - 1, num=window_cap)).astype(int)
                    indices[-1] = len(starts) - 1
                    starts = starts[np.unique(indices)]

            windows_after_cap = len(starts)
            max_windows_after_cap = max(max_windows_after_cap, windows_after_cap)
            if window_cap is not None and windows_after_cap == window_cap:
                sequences_at_cap += 1

            for start in starts:
                chunk = arr[start : start + window]
                real_length = len(chunk)
                if real_length < window:
                    pad_length = window - real_length
                    padding = np.full((pad_length, arr.shape[1]), self.padding_value, dtype=np.float32)
                    chunk = np.vstack([chunk, padding])
                    mask = np.concatenate([
                        np.ones(real_length, dtype=bool),
                        np.zeros(pad_length, dtype=bool),
                    ])
                else:
                    mask = np.ones(window, dtype=bool)
                sequences.append(chunk)
                masks.append(mask)
                seq_ids.append(seq_id)

        if not sequences:
            raise ValueError("SequenceExtractor.transform_chunks produced no sequences.")

        if capped_sequences:
            print(
                "Per-sequence chunk cap applied: "
                f"crop_mode={self.sequence_crop_mode}, maxlen={self.maxlen}, "
                f"window={window}, stride={stride}, cap={window_cap}, "
                f"sequences={sequences_total}, capped_sequences={capped_sequences}, "
                f"sequences_at_cap={sequences_at_cap}, "
                f"max_windows_before_cap={max_windows_before_cap}, "
                f"max_windows_after_cap={max_windows_after_cap}"
            )

        del features, out
        return {
            "X": np.stack(sequences, axis=0),
            "mask": np.stack(masks, axis=0),
            "sequence_ids": np.array(seq_ids),
            "feature_names": list(self.feature_names_in_),
        }


# ---------------------------------------------------------------------------
# Search dependencies and Random Forest sequence classifier
# ---------------------------------------------------------------------------


_SEQUENCE_PARAMETER_DEFAULTS = {
    name: parameter.default
    for name, parameter in inspect.signature(SequenceExtractor.__init__).parameters.items()
    if name != "self"
}


def _spectral_configs(settings, kind):
    raw = settings.get(f"{kind}_configs", _UNSET)
    if raw is _UNSET:
        return [{}]
    raw = _coerce_search_value(raw)
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)) and all(isinstance(config, dict) for config in raw):
        return raw
    return None  # Unknown/custom declarations must remain active for preflight.


def _sequence_dead_reckoning_consumed(settings):
    """Built-in feature selectors exclude these derived axes unless explicitly requested."""
    for kind in ("stft", "cwt"):
        configs = _spectral_configs(settings, kind)
        if configs is None:
            return True
        for config in configs:
            columns = config.get("signal_columns")
            if columns is not None and any("_raw_dead_reckoning_" in str(column) for column in columns):
                return True
    return False


def inactive_sequence_parameters(params, *, for_frame_output=None):
    """Return proven unused native-extractor knobs, without inspecting a dataset.

    Controls themselves remain present: removing a disabled switch could restore
    an enabled value from the base estimator. Data-dependent equivalences such as
    a window cap that no sequence reaches are deliberately not inferred here.
    """
    settings = dict(_SEQUENCE_PARAMETER_DEFAULTS)
    settings.update(params)
    inactive = set()
    if not settings["filter_problematic_sequences"]:
        inactive |= {"ideal_skew_threshold", "problematic_features_threshold", "problematic_feature_cols"}
    if settings["motion_filter_mode"] not in {"kalman_1d", "kalman", "extended_kalman"}:
        inactive |= {"kalman_process_noise", "kalman_measurement_noise"}
    if not _sequence_dead_reckoning_consumed(settings):
        inactive |= {"use_dead_reckoning", "dead_reckoning_detrend"}
    elif not settings["use_dead_reckoning"]:
        inactive.add("dead_reckoning_detrend")
    if not settings["resample_modalities"]:
        inactive |= {f"{sensor}_target_sampling_rate" for sensor in ("imu", "rot", "tof", "thm")}
        inactive |= {"tof_native_sampling_rate", "thm_native_sampling_rate"}
    if not HoneycombBase()._parse_modes(settings["tof_modes"]):
        inactive.add("tof_n_sensors")
        if not settings["resample_modalities"]:
            inactive.add("tof_fill_mode")
    if not HoneycombBase()._parse_modes(settings["rotation_modes"]):
        inactive.add("fix_quaternion_sign")
    if settings["linear_acc_mode"] != "baseline":
        inactive |= {"use_highpass_fallback", "acceleration_units", "linear_acc_frame", "gravity_magnitude"}
    smoothed = "smoothed" in HoneycombBase()._parse_modes(settings["acc_modes"])
    if not smoothed:
        inactive.add("smooth_alpha")
    highpass = settings["linear_acc_mode"] == "baseline" and settings["use_highpass_fallback"]
    if not highpass and (not smoothed or settings["smooth_alpha"] is not None):
        inactive.add("window_size")

    frame = str(settings["output_format"]).lower() == "frame" if for_frame_output is None else for_frame_output
    if frame:
        inactive |= {
            "maxlen", "sequence_crop_mode", "padding_value", "chunk_window_size", "chunk_stride",
            "max_windows_per_sequence", "window_cap_policy", "final_window_mode",
            "use_chunk_stride_ratio", "chunk_stride_ratio", "add_global_context",
        }
    else:
        inactive.add("frame_stats")
        if settings["use_chunk_stride_ratio"]:
            inactive.add("chunk_stride")
        else:
            inactive.add("chunk_stride_ratio")
        if settings["max_windows_per_sequence"] is None:
            inactive.add("window_cap_policy")
        if settings["sequence_crop_mode"] == "none":
            inactive.add("maxlen")
        elif settings["maxlen"] is None:
            inactive.add("sequence_crop_mode")

    maps = {
        "stft": {"nperseg": "nperseg", "noverlap": "noverlap", "window_type": "window_type",
                 "scaling": "scaling", "detrend": "detrend", "use_log_scale": "use_log_scale",
                 "frequency_bands": "frequency_bands", "window_seconds": "window_seconds"},
        "cwt": {"wavelet": "wavelet", "max_scale": "max_scale", "n_scales": "n_scales",
                "use_log_scale": "use_log_scale", "min_scale": "min_scale", "max_frequency": "max_frequency"},
    }
    for kind, fields in maps.items():
        configs = _spectral_configs(settings, kind)
        if configs is None:
            continue
        for suffix, field in fields.items():
            if not configs or all(field in config for config in configs):
                inactive.add(f"{kind}_{suffix}")
        if kind == "stft" and configs and all(
            config.get("window_seconds", settings["stft_window_seconds"]) is not None for config in configs
        ):
            inactive.add("stft_nperseg")
        if kind == "cwt" and configs and all(config.get("widths") is not None for config in configs):
            inactive |= {"cwt_n_scales", "cwt_min_scale"}
    if "noop" in settings or any(key.startswith("noop__") for key in settings):
        inactive |= inactive_descriptor_parameters(settings)
    return inactive


def sequence_feature_families(params):
    """Conservative feature families for the native pipeline, before data availability."""
    settings = dict(_SEQUENCE_PARAMETER_DEFAULTS)
    settings.update(params)
    families = set()
    parse = HoneycombBase()._parse_modes
    if (parse(settings["acc_modes"]) or parse(settings["linear_acc_modes"])
            or settings["use_acc_magnitude"] or settings["use_linear_acc_magnitude"]):
        families.add("acc")
    for name, key in (("rotation", "rotation_modes"), ("tof", "tof_modes"), ("thermo", "thm_modes")):
        if parse(settings[key]):
            families.add(name)
    for kind in ("stft", "cwt"):
        configs = _spectral_configs(settings, kind)
        if configs is None or any(config.get("signal_columns") != [] for config in configs):
            families.add(kind)
    noop = settings.get("noop")
    noop_settings = noop.get_params() if hasattr(noop, "get_params") else dict(noop or {})
    noop_settings.update({key[6:]: value for key, value in settings.items() if key.startswith("noop__")})
    if (any(noop_settings.get(key) is not None for key in ("motion_stats", "periodicity", "spectral_stats"))
            or noop_settings.get("quality") == "features"):
        families.add("noop")
    return families

# ===========================================================================
# 8. Sequence Classifier Parent-Child Hierarchy
# ===========================================================================


# ---------------------------------------------------------------------------
# Abstract Parent Classifier: BaseSequenceClassifier
# ---------------------------------------------------------------------------


class BaseSequenceClassifier(BaseEstimator, ClassifierMixin, ABC):
    """Shared parent class for sequence-level classifiers in ``base_utils.py`` and ``utils.py``."""

    _estimator_type = "classifier"

    def _align_sequence_labels(
        self,
        sequence_ids: Iterable[Any],
        y: Any,
        *,
        sequence_col: str = "sequence_id",
        primary_target: Optional[str] = None,
    ) -> np.ndarray:
        target = primary_target or getattr(self, "primary_target", "bfrb")
        seq_index = pd.Index(sequence_ids)
        fill_value = "non_bfrb" if target == "bfrb" else "Unknown"
        if isinstance(y, pd.DataFrame):
            yy = y.loc[:, [col for col in (sequence_col, target) if col in y.columns]]
            if sequence_col not in yy.columns:
                if yy.index.name == sequence_col:
                    yy = yy.reset_index()
                elif len(yy) == len(seq_index):
                    if target in yy.columns:
                        return yy[target].fillna(fill_value).to_numpy()
                    raise ValueError("y DataFrame does not contain target column.")
                else:
                    raise ValueError(f"y DataFrame does not contain {sequence_col}.")
            if target not in yy.columns:
                raise ValueError(f"y DataFrame does not contain target column: {target}")
            y_seq = yy.drop_duplicates(sequence_col).set_index(sequence_col)[target]
            return y_seq.reindex(seq_index).fillna(fill_value).to_numpy()
        if isinstance(y, pd.Series):
            if y.index.name == sequence_col:
                return y.reindex(seq_index).fillna(fill_value).to_numpy()
            if len(y) == len(seq_index):
                return y.fillna(fill_value).to_numpy()
            raise ValueError("Could not align y Series to sequence_ids.")
        y_arr = np.asarray(y)
        if len(y_arr) == len(seq_index):
            return y_arr
        raise ValueError("Could not align y to sequence_ids.")

    def get_history_dict(self) -> Dict[str, Any]:
        return dict(getattr(self, "history_", {}))

    def summarize_model(self) -> None:
        inner = getattr(self, "estimator_", None) or getattr(self, "model_", None)
        if inner is not None:
            print(inner)
        else:
            print(f"{self.__class__.__name__} is not fitted yet.")



class RandomForestSequenceClassifier(BaseSequenceClassifier):
    _estimator_type = "classifier"

    def __init__(
        self,
        primary_target: str = "bfrb",
        extractor: Optional[SequenceExtractor] = None,
        estimator: Optional[BaseEstimator] = None,
        verbose: int = 0,
        random_state: int = 42,
    ):
        self.primary_target = primary_target
        self.extractor = extractor
        self.estimator = estimator
        self.verbose = verbose
        self.random_state = random_state

    def _default_extractor(self) -> SequenceExtractor:
        return SequenceExtractor(
            output_format="frame", chunk_window_size=None, padding_value=0.0,
            add_global_context=False, frame_stats="mean,std,min,max,last",
            resample_modalities=False,
        )

    def _default_estimator(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=300, random_state=self.random_state,
            n_jobs=-1, class_weight="balanced_subsample",
        )

    def _align_y(self, sequence_ids, y):
        seq_col = getattr(self.extractor_, "sequence_col", "sequence_id")
        sequence_ids = pd.Index(sequence_ids)
        fill_value = "non_bfrb" if self.primary_target == "bfrb" else "Unknown"
        if isinstance(y, pd.DataFrame):
            yy = y.loc[:, [column for column in (seq_col, self.primary_target) if column in y.columns]]
            if seq_col not in yy.columns:
                if yy.index.name == seq_col:
                    yy = yy.reset_index()
                elif len(yy) == len(sequence_ids):
                    if self.primary_target in yy.columns:
                        return yy[self.primary_target].fillna(fill_value).to_numpy()
                    raise ValueError("y DataFrame does not contain target column.")
                else:
                    raise ValueError("y DataFrame does not contain sequence_id.")
            if self.primary_target not in yy.columns:
                raise ValueError(f"y DataFrame does not contain target column: {self.primary_target}")
            y_seq = yy.drop_duplicates(seq_col).set_index(seq_col)[self.primary_target]
            aligned = y_seq.reindex(sequence_ids).fillna(fill_value)
            return aligned.to_numpy()
        if isinstance(y, pd.Series):
            if y.index.name == seq_col:
                aligned = y.reindex(sequence_ids).fillna(fill_value)
                return aligned.to_numpy()
            if len(y) == len(sequence_ids):
                return y.fillna(fill_value).to_numpy()
            raise ValueError("Could not align y Series to sequence_ids.")
        y_arr = np.asarray(y)
        if len(y_arr) == len(sequence_ids):
            return y_arr
        raise ValueError("Could not align y to sequence_ids.")

    def fit(self, X, y=None, **fit_params):
        if y is None:
            raise ValueError("RandomForestSequenceClassifier requires y.")
        self.extractor_ = clone(self.extractor) if self.extractor is not None else self._default_extractor()
        if hasattr(self.extractor_, "output_format"):
            try: self.extractor_.set_params(output_format="frame")
            except Exception: pass
        validate_sequence_extractor_params(self.extractor_.get_params(), for_frame_output=True)
        if type(self.extractor_) is SequenceExtractor and not any(
            method in self.extractor_.__dict__ for method in ("fit", "transform", "transform_frame")
        ):
            frame = self.extractor_.fit_transform(X)
        else:
            self.extractor_.fit(X)
            frame = self.extractor_.transform_frame(X) if hasattr(self.extractor_, "transform_frame") else self.extractor_.transform(X)
        if not isinstance(frame, pd.DataFrame):
            raise InvalidExtractorParams("Extractor did not return a DataFrame.")
        if frame.empty:
            raise InvalidExtractorParams("Feature extraction produced zero sequences.")
        if not np.isfinite(frame.to_numpy(dtype=float, copy=False)).all():
            raise InvalidExtractorParams("Feature extraction produced non-finite values.")
        y_aligned = self._align_y(frame.index, y)
        self.le_ = LabelEncoder()
        self.le_.fit(y_aligned)
        self.classes_ = self.le_.classes_
        y_enc = self.le_.transform(y_aligned)
        self.estimator_ = clone(self.estimator) if self.estimator is not None else self._default_estimator()
        self.estimator_.fit(frame, y_enc)
        self.history_ = {"loss": [0.0], "accuracy": [1.0]}
        return self

    def _transform_frame(self, X):
        check_is_fitted(self, ["extractor_"])
        frame = self.extractor_.transform_frame(X) if hasattr(self.extractor_, "transform_frame") else self.extractor_.transform(X)
        if not isinstance(frame, pd.DataFrame):
            raise InvalidExtractorParams("Extractor did not return a DataFrame.")
        if frame.empty:
            raise InvalidExtractorParams("Feature extraction produced zero sequences.")
        return frame

    def predict_proba(self, X):
        check_is_fitted(self, ["estimator_", "le_"])
        frame = self._transform_frame(X)
        probs = self.estimator_.predict_proba(frame)
        return pd.DataFrame(probs, index=frame.index, columns=self.estimator_.classes_)

    def predict(self, X):
        check_is_fitted(self, ["estimator_", "le_"])
        frame = self._transform_frame(X)
        pred_enc = self.estimator_.predict(frame)
        preds = self.le_.inverse_transform(pred_enc)
        return pd.Series(preds, index=frame.index, name=self.primary_target).sort_index()

    def score(self, X, y, sample_weight=None):
        preds = self.predict(X)
        seq_col = getattr(self.extractor_, "sequence_col", "sequence_id")
        if isinstance(y, pd.DataFrame):
            columns = dict.fromkeys((seq_col, self.primary_target, "is_target"))
            yy = y.loc[:, [column for column in columns if column in y.columns]]
            if seq_col not in yy.columns:
                if yy.index.name == seq_col:
                    yy = yy.reset_index()
                else:
                    raise ValueError("y DataFrame must contain sequence_id.")
            y_seq = yy.drop_duplicates(seq_col).sort_values(seq_col)
            if isinstance(preds, pd.Series):
                y_seq = y_seq[y_seq[seq_col].isin(preds.index)]
                preds_aligned = preds.reindex(y_seq[seq_col]).to_numpy()
            else:
                preds_aligned = np.asarray(preds)
            y_true = y_seq[self.primary_target].values
            if "is_target" in y_seq.columns:
                y_true_binary = y_seq["is_target"].astype(int).values
            else:
                y_true_binary = (y_true != "non_bfrb").astype(int)
            if self.primary_target == "bfrb":
                return competition_score(y_true, preds_aligned, y_true_binary=y_true_binary, target_only_macro=True)
            return f1_score(y_true, preds_aligned, average="macro", zero_division=0)
        y_arr = np.asarray(y)
        preds_arr = preds.to_numpy() if hasattr(preds, "to_numpy") else np.asarray(preds)
        return f1_score(y_arr, preds_arr, average="macro", zero_division=0)

    def get_history_dict(self): return getattr(self, "history_", {})
    def summarize_model(self):
        if hasattr(self, "estimator_"): print(self.estimator_)
        else: print("Model is not fitted yet.")

# ===========================================================================
# 9. Competition Scoring, Search Drivers & BaseSearchCV
# ===========================================================================

# ---------------------------------------------------------------------------
# Competition scoring
# ---------------------------------------------------------------------------


def competition_score(y_true_gesture, y_pred, y_true_binary=None, target_only_macro: bool = True) -> float:
    y_true_gesture = np.asarray(y_true_gesture)
    y_pred = np.asarray(y_pred)
    if y_true_binary is None:
        y_true_binary = (y_true_gesture != "non_bfrb").astype(int)
    else:
        y_true_binary = np.asarray(y_true_binary).astype(int)
    y_pred_binary = (y_pred != "non_bfrb").astype(int)
    binary_f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    if target_only_macro:
        mask = y_true_binary == 1
        if mask.sum() > 0:
            macro_f1 = f1_score(y_true_gesture[mask], y_pred[mask], average="macro", zero_division=0)
        else:
            macro_f1 = 0.0
    else:
        macro_f1 = f1_score(y_true_gesture, y_pred, average="macro", zero_division=0)
    return (binary_f1 + macro_f1) / 2.0


def make_competition_scorer(target_col: str = "bfrb"):
    seq_col = "sequence_id"
    def _score(y_true, y_pred):
        if target_col == "bfrb":
            if isinstance(y_true, pd.DataFrame):
                if seq_col not in y_true.columns:
                    if y_true.index.name == seq_col:
                        y_true = y_true.reset_index()
                    else:
                        raise ValueError("y_true must contain sequence_id.")
                y_seq = y_true.drop_duplicates(seq_col).sort_values(seq_col)
                if "is_target" in y_seq.columns:
                    y_true_binary = y_seq["is_target"].astype(int).values
                else:
                    y_true_binary = (y_seq[target_col] != "non_bfrb").astype(int)
                y_true_gesture = y_seq[target_col].values
                if isinstance(y_pred, pd.Series) and y_pred.index.name == seq_col:
                    y_pred = y_pred.reindex(y_seq[seq_col]).to_numpy()
                else:
                    y_pred = np.asarray(y_pred)
            else:
                y_true_gesture = np.asarray(y_true)
                y_true_binary = (y_true_gesture != "non_bfrb").astype(int)
                y_pred = np.asarray(y_pred)
            return competition_score(y_true_gesture, y_pred, y_true_binary=y_true_binary, target_only_macro=True)
        if isinstance(y_true, pd.DataFrame):
            if seq_col not in y_true.columns:
                if y_true.index.name == seq_col:
                    y_true = y_true.reset_index()
                else:
                    raise ValueError("y_true must contain sequence_id.")
            y_seq = y_true.drop_duplicates(seq_col).sort_values(seq_col)
            y_true_vals = y_seq[target_col].astype(str).values
            if isinstance(y_pred, pd.Series) and y_pred.index.name == seq_col:
                y_pred = y_pred.reindex(y_seq[seq_col]).astype(str).to_numpy()
            else:
                y_pred = np.asarray(y_pred, dtype=str)
            return f1_score(y_true_vals, y_pred, average="macro", zero_division=0)
        y_true_vals = np.asarray(y_true, dtype=str)
        y_pred = np.asarray(y_pred, dtype=str)
        return f1_score(y_true_vals, y_pred, average="macro", zero_division=0)
    return make_scorer(_score)


competition_scorer = make_competition_scorer("bfrb")


def evaluate_holdout(y_test_df: pd.DataFrame, y_pred, target_col: str = "bfrb", verbose: bool = True) -> Dict[str, Any]:
    seq_col = "sequence_id"
    y_df = y_test_df.copy()
    if seq_col not in y_df.columns:
        if y_df.index.name == seq_col:
            y_df = y_df.reset_index()
        else:
            raise ValueError("y_test_df must contain sequence_id.")
    y_test_seq = y_df.drop_duplicates(subset=[seq_col]).sort_values(seq_col).reset_index(drop=True)
    if isinstance(y_pred, pd.Series) and y_pred.index.name == seq_col:
        y_pred = y_pred.reindex(y_test_seq[seq_col]).to_numpy()
    else:
        y_pred = np.asarray(y_pred)
    if target_col == "bfrb":
        if "is_target" in y_test_seq.columns:
            y_true_binary = y_test_seq["is_target"].astype(int).values
        else:
            y_true_binary = (y_test_seq[target_col] != "non_bfrb").astype(int)
        y_pred_binary = (y_pred != "non_bfrb").astype(int)
        binary_f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        target_mask = y_true_binary == 1
        if target_mask.sum() > 0:
            gesture_f1 = f1_score(
                y_test_seq.loc[target_mask, target_col].values,
                y_pred[target_mask], average="macro", zero_division=0,
            )
        else:
            gesture_f1 = 0.0
        comp_score = (binary_f1 + gesture_f1) / 2.0
        if verbose:
            print("\n" + "=" * 60)
            print("FINAL EVALUATION")
            print("=" * 60)
            print(f"Binary F1 (non_bfrb vs bfrb): {binary_f1:.4f}")
            print(f"BFRB Gesture Macro F1: {gesture_f1:.4f}")
            print(f"COMPETITION SCORE: {comp_score:.4f}")
            if target_mask.sum() > 0:
                print("\n" + "-" * 40)
                print("BFRB Gesture Classification Report")
                print("-" * 40)
                print(classification_report(y_test_seq.loc[target_mask, target_col].values, y_pred[target_mask], zero_division=0))
        results_df = pd.DataFrame({
            "sequence_id": y_test_seq[seq_col].values,
            "is_target_true": y_true_binary,
            "is_target_pred": y_pred_binary,
            f"{target_col}_true": y_test_seq[target_col].values,
            f"{target_col}_pred": y_pred,
        })
        return {"binary_f1": binary_f1, "gesture_f1": gesture_f1, "competition_score": comp_score, "results_df": results_df}
    y_true_values = y_test_seq[target_col].astype(str).values
    y_pred_values = np.asarray(y_pred, dtype=str)
    macro_f1 = f1_score(y_true_values, y_pred_values, average="macro", zero_division=0)
    if verbose:
        print("\n" + "=" * 60)
        print("FINAL EVALUATION")
        print("=" * 60)
        print(f"Target: {target_col}")
        print(f"Macro F1: {macro_f1:.4f}")
        print("\n" + "-" * 40)
        print("Classification Report")
        print("-" * 40)
        print(classification_report(y_true_values, y_pred_values, zero_division=0))
    results_df = pd.DataFrame({
        "sequence_id": y_test_seq[seq_col].values,
        f"{target_col}_true": y_true_values,
        f"{target_col}_pred": y_pred_values,
    })
    return {"macro_f1": macro_f1, "competition_score": macro_f1, "results_df": results_df}


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------


def prepare_multitask_param_space(param_space: Dict[str, Any], search_mode: str) -> Dict[str, Any]:
    if search_mode not in {"bayesian", "evolutionary"}:
        return param_space
    out = {}
    for key, val in param_space.items():
        name = key.split("__")[-1]
        if (name in {"branch_filters","branch_kernel_sizes","branch_pool_sizes"} and isinstance(val, list) and val and isinstance(val[0], dict)):
            out[key] = [json.dumps(d, sort_keys=True) for d in val]
        else:
            out[key] = val
    return out


def prepare_bayesian_space(param_space: Dict[str, Any]) -> Dict[str, Any]:
    if Categorical is None:
        return param_space
    out = {}
    for key, space in param_space.items():
        if isinstance(space, Categorical):
            new_cats = []
            for cat in space.categories:
                if isinstance(cat, (tuple, list, dict, set)):
                    new_cats.append(json.dumps(cat, sort_keys=True))
                else:
                    new_cats.append(cat)
            out[key] = Categorical(new_cats)
        elif isinstance(space, list):
            new_cats = []
            for cat in space:
                if isinstance(cat, (tuple, list, dict, set)):
                    new_cats.append(json.dumps(cat, sort_keys=True))
                else:
                    new_cats.append(cat)
            out[key] = Categorical(new_cats)
        else:
            out[key] = space
    return out


def prepare_random_forest_bayesian_spaces(param_space, n_iter):
    """Collapse redundant integer split/leaf pairs, sharing one Bayesian budget.

    sklearn raises min_samples_split to at least twice min_samples_leaf. Float
    fractions depend on the fold's row count and are deliberately left alone.
    """
    if not isinstance(n_iter, (int, np.integer)) or isinstance(n_iter, bool) or n_iter <= 0:
        raise ValueError("n_iter must be a positive integer")
    keys = {key.rsplit("__", 1)[-1]: key for key in param_space}
    leaf_key, split_key = keys.get("min_samples_leaf"), keys.get("min_samples_split")
    if leaf_key is None or split_key is None:
        return [(dict(param_space), int(n_iter))]

    def choices(dimension):
        values = getattr(dimension, "categories", dimension if isinstance(dimension, list) else None)
        if values is None or any(not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1 for value in values):
            return None
        prior = getattr(dimension, "prior", None)
        if prior is not None and not np.allclose(prior, np.full(len(values), 1.0 / len(values))):
            return None  # Do not silently change an explicitly weighted search.
        return list(values)

    leaves, splits = choices(param_space[leaf_key]), choices(param_space[split_key])
    if (leaves is None or splits is None or not leaves or not splits
            or any(split < 2 for split in splits) or n_iter < len(leaves)):
        return [(dict(param_space), int(n_iter))]
    grouped = []
    redundant = False
    for leaf in leaves:
        effective = {}
        for split in splits:
            effective.setdefault(max(split, 2 * leaf), split)
        redundant |= len(effective) < len(splits)
        grouped.append((leaf, list(effective.values())))
    if not redundant:
        return [(dict(param_space), int(n_iter))]
    from skopt.space import Categorical
    budgets = [n_iter // len(grouped) + (index < n_iter % len(grouped)) for index in range(len(grouped))]
    compiled = []
    for (leaf, distinct_splits), budget in zip(grouped, budgets):
        if budget:
            space = dict(param_space)
            space[leaf_key] = Categorical([leaf])
            space[split_key] = Categorical(distinct_splits)
            compiled.append((space, int(budget)))
    return compiled


# Re-export unified search descriptors from src/search_drivers.py


class ECat:
    """Categorical parameter descriptor for evolutionary search."""

    def __init__(self, choices, priors=None, random_state=None):
        if isinstance(choices, (set, tuple)):
            choices = list(choices)
        elif not isinstance(choices, list):
            choices = list(choices)
        if not choices:
            raise ValueError("ECat choices must not be empty.")
        self.choices = choices
        self.categories = choices
        self.priors = priors
        self.random_state = random_state

    def sample(self, rng=None):
        rng = rng or np.random.default_rng(self.random_state)
        idx = int(rng.choice(len(self.choices), p=self.priors))
        return self.choices[idx]

    def mutate(self, value, rng=None):
        rng = rng or np.random.default_rng(self.random_state)
        if len(self.choices) <= 1:
            return self.choices[0]
        others = [c for c in self.choices if c != value]
        return others[int(rng.integers(0, len(others)))] if others else self.sample(rng)

    def __iter__(self):
        return iter(self.choices)

    def __len__(self):
        return len(self.choices)

    def __repr__(self):
        return f"ECat({self.choices!r})"


class EInt:
    """Integer range descriptor for evolutionary search."""

    def __init__(self, lower: int, upper: int, distribution: str = "uniform", step_scale: float = 0.2, random_state=None):
        self.lower = int(lower)
        self.upper = int(upper)
        if self.lower > self.upper:
            raise ValueError(f"EInt lower ({self.lower}) must be <= upper ({self.upper}).")
        self.low = self.lower
        self.high = self.upper
        self.distribution = distribution
        self.step_scale = float(step_scale)
        self.random_state = random_state

    def sample(self, rng=None) -> int:
        rng = rng or np.random.default_rng(self.random_state)
        return int(rng.integers(self.lower, self.upper + 1))

    def mutate(self, value, rng=None) -> int:
        rng = rng or np.random.default_rng(self.random_state)
        if self.lower == self.upper:
            return self.lower
        span = max(1, (self.upper - self.lower) // 3)
        delta = int(rng.integers(-span, span + 1))
        cand = int(np.clip(int(value) + delta, self.lower, self.upper))
        return cand if cand != value else self.sample(rng)

    def __repr__(self):
        return f"EInt({self.lower}, {self.upper})"


class EFloat:
    """Continuous range descriptor for evolutionary search (uniform or log-uniform)."""

    def __init__(
        self,
        lower: float,
        upper: float,
        distribution: str = "uniform",
        sigma_frac: float = 0.15,
        prior: Optional[str] = None,
        random_state=None,
    ):
        self.lower = float(lower)
        self.upper = float(upper)
        if self.lower > self.upper:
            raise ValueError(f"EFloat lower ({self.lower}) must be <= upper ({self.upper}).")
        self.low = self.lower
        self.high = self.upper
        eff_dist = str(prior if prior is not None else distribution).lower()
        if eff_dist not in {"uniform", "log-uniform"}:
            raise ValueError("EFloat distribution must be 'uniform' or 'log-uniform'")
        if eff_dist == "log-uniform" and self.lower <= 0:
            raise ValueError("log-uniform EFloat requires lower > 0")
        self.distribution = eff_dist
        self.sigma_frac = float(sigma_frac)
        self.random_state = random_state

    def sample(self, rng=None) -> float:
        rng = rng or np.random.default_rng(self.random_state)
        if self.lower == self.upper:
            return float(self.lower)
        if self.distribution in {"log-uniform", "log_uniform", "log"} and self.lower > 0:
            return float(np.exp(rng.uniform(np.log(self.lower), np.log(self.upper))))
        return float(rng.uniform(self.lower, self.upper))

    def mutate(self, value, rng=None) -> float:
        rng = rng or np.random.default_rng(self.random_state)
        if self.lower == self.upper:
            return float(self.lower)
        if self.distribution in {"log-uniform", "log_uniform", "log"} and self.lower > 0:
            lo, hi = np.log(self.lower), np.log(self.upper)
            val = float(np.log(max(float(value), self.lower)))
            step = rng.normal(0.0, 0.2 * (hi - lo))
            return float(np.exp(np.clip(val + step, lo, hi)))
        step = rng.normal(0.0, 0.2 * (self.upper - self.lower))
        return float(np.clip(float(value) + step, self.lower, self.upper))

    def __repr__(self):
        return f"EFloat({self.lower}, {self.upper}, {self.distribution!r})"


def _largest_remainder(quotas: np.ndarray, caps: np.ndarray, target: int, rng) -> np.ndarray:
    """Round per-cell quotas to integers summing to ``target`` without exceeding ``caps``."""
    counts = np.minimum(np.floor(quotas).astype(int), caps)
    remaining = int(target - counts.sum())
    if remaining <= 0:
        return counts
    fractions = quotas - np.floor(quotas)
    shuffled = rng.permutation(len(quotas))
    order = shuffled[np.argsort(-fractions[shuffled], kind="stable")]
    while remaining > 0:
        added = False
        for idx in order:
            if remaining == 0:
                break
            if counts[idx] < caps[idx]:
                counts[idx] += 1
                remaining -= 1
                added = True
        if not added:
            break
    return counts


def sample_balanced_train_test_split(
    df: pd.DataFrame,
    train_pct: float = 0.20,
    test_pct: float = 0.05,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split sequences into disjoint train/test sets stratified by ``(subject, gesture)``."""
    if train_pct <= 0 or test_pct <= 0 or train_pct + test_pct > 1 + 1e-9:
        raise ValueError("train_pct and test_pct must be positive and sum to at most 1.")

    rng = np.random.default_rng(random_state)
    metadata = df[["sequence_id", "subject", "gesture"]].drop_duplicates()
    if metadata["sequence_id"].duplicated().any():
        raise ValueError("Each sequence_id must have one subject and gesture.")
    total_sequences = metadata["sequence_id"].nunique()
    cells = [
        group["sequence_id"].sample(frac=1, random_state=random_state).tolist()
        for _, group in metadata.groupby(["subject", "gesture"], observed=True)
    ]
    sizes = np.array([len(cell) for cell in cells])

    train_target = int(round(total_sequences * train_pct))
    n_train = _largest_remainder(sizes * train_pct, sizes, train_target, rng)

    if train_pct + test_pct >= 1 - 1e-9:
        test_target = total_sequences - int(n_train.sum())
    else:
        test_target = min(int(round(total_sequences * test_pct)), total_sequences - int(n_train.sum()))
    n_test = _largest_remainder(sizes * test_pct, sizes - n_train, test_target, rng)

    train_ids, test_ids = [], []
    for cell, k_train, k_test in zip(cells, n_train, n_test):
        train_ids.extend(cell[:k_train])
        test_ids.extend(cell[k_train : k_train + k_test])

    assert len(set(train_ids) & set(test_ids)) == 0, "Overlap detected!"

    train_df = df[df["sequence_id"].isin(train_ids)]
    test_df = df[df["sequence_id"].isin(test_ids)]

    print(f"Train: {len(train_ids)} seqs | {100 * len(train_ids) / total_sequences:.1f}%")
    print(f"Test:  {len(test_ids)} seqs  | {100 * len(test_ids) / total_sequences:.1f}%")
    unused = total_sequences - len(train_ids) - len(test_ids)
    if unused:
        print(f"Unused: {unused} seqs | {100 * unused / total_sequences:.1f}%")

    return train_df, test_df


def sample_balanced_subset(
    df: pd.DataFrame,
    target_col: str = "bfrb",
    fraction: float = 1.0,
    random_state: int = 42,
    sequence_col: str = "sequence_id",
) -> pd.DataFrame:
    """Sample a sequence-balanced subset of ``df`` preserving class distribution."""
    if float(fraction) >= 1.0:
        return df.copy()
    rng = np.random.default_rng(int(random_state))
    seq_meta = df.groupby(sequence_col, sort=False)[target_col].first()
    chosen: List[str] = []
    for _, group in seq_meta.groupby(seq_meta):
        n_keep = max(1, int(round(len(group) * float(fraction))))
        idx = rng.choice(len(group), size=min(len(group), n_keep), replace=False)
        chosen.extend(group.index[np.sort(idx)].astype(str).tolist())
    return df[df[sequence_col].astype(str).isin(set(chosen))].copy()


def build_quick_results_df_shared(search_or_results: Any) -> pd.DataFrame:
    """Convert a search object or ``cv_results_`` dict into a ranked summary DataFrame."""
    cv_res = getattr(search_or_results, "cv_results_", search_or_results)
    if not cv_res:
        return pd.DataFrame()
    df = pd.DataFrame(cv_res)
    if "rank_test_score" in df.columns:
        df = df.sort_values("rank_test_score", kind="stable").reset_index(drop=True)
    elif "mean_test_score" in df.columns:
        df = df.sort_values("mean_test_score", ascending=False, kind="stable").reset_index(drop=True)
    return df


def prepare_evolutionary_space(param_space: Dict[str, Any]) -> Dict[str, Any]:
    """Convert candidate lists and complex objects into evolutionary search descriptors."""
    out: Dict[str, Any] = {}
    for key, space in param_space.items():
        if isinstance(space, ECat) or (_ExternalECat is not None and isinstance(space, _ExternalECat)):
            raw_choices = getattr(space, "choices", getattr(space, "categories", []))
            new_cats = [
                json.dumps(cat, sort_keys=True) if isinstance(cat, (tuple, list, dict, set)) else cat
                for cat in raw_choices
            ]
            out[key] = ECat(new_cats, priors=getattr(space, "priors", None))
        elif isinstance(space, (EInt, EFloat)) or (
            (_ExternalEInt is not None and isinstance(space, _ExternalEInt))
            or (_ExternalEFloat is not None and isinstance(space, _ExternalEFloat))
        ):
            out[key] = space
        elif Categorical is not None and isinstance(space, Categorical):
            new_cats = [
                json.dumps(cat, sort_keys=True) if isinstance(cat, (tuple, list, dict, set)) else cat
                for cat in space.categories
            ]
            out[key] = ECat(new_cats)
        elif isinstance(space, (list, tuple)):
            new_cats = [
                json.dumps(cat, sort_keys=True) if isinstance(cat, (tuple, list, dict, set)) else cat
                for cat in space
            ]
            out[key] = ECat(new_cats)
        else:
            out[key] = ECat([space])
    return out


def _sample_evolutionary_gene(descriptor: Any, rng: np.random.Generator) -> Any:
    if hasattr(descriptor, "sample"):
        try:
            return descriptor.sample(rng=rng)
        except TypeError:
            return descriptor.sample()
    if isinstance(descriptor, (list, tuple)):
        if len(descriptor) == 0:
            return None
        return descriptor[int(rng.integers(0, len(descriptor)))]
    return descriptor


def _slice_subset(data, indices):
    if data is None:
        return None
    if hasattr(data, "iloc"):
        return data.iloc[indices]
    return np.asarray(data)[indices]

# ---------------------------------------------------------------------------
# Abstract Parent Hyperparameter Search: BaseSearchCV
# ---------------------------------------------------------------------------


class BaseSearchCV(BaseEstimator, ABC):
    """Shared parent class for ``GASearchCV`` (base_utils), ``MultiBranchGASearchCV`` (utils), and ``HierarchicalBayesSearchCV`` (utils)."""

    def predict(self, X: Any):
        check_is_fitted(self, ["best_estimator_"])
        return self.best_estimator_.predict(X)

    def predict_proba(self, X: Any):
        check_is_fitted(self, ["best_estimator_"])
        return self.best_estimator_.predict_proba(X)

    def score(self, X: Any, y: Any):
        check_is_fitted(self, ["best_estimator_"])
        return self.best_estimator_.score(X, y)



class GASearchCV(BaseSearchCV):
    """Evolutionary hyperparameter search with CV and group-aware splitting."""

    def __init__(
        self,
        estimator,
        cv=5,
        scoring=None,
        population_size: int = 10,
        generations: int = 5,
        tournament_size: int = 3,
        elitism: bool = True,
        crossover_probability: float = 0.8,
        mutation_probability: float = 0.2,
        param_grid: Optional[Dict[str, Any]] = None,
        criteria: str = "max",
        refit: bool = True,
        verbose: int = 1,
        return_train_score: bool = True,
        error_score=np.nan,
        random_state: Optional[int] = 42,
    ):
        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.param_grid = param_grid
        self.criteria = criteria
        self.refit = refit
        self.verbose = verbose
        self.return_train_score = return_train_score
        self.error_score = error_score
        self.random_state = random_state

    def _sample_individual(self, space: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
        return {k: _sample_evolutionary_gene(v, rng) for k, v in space.items()}

    def _canonical_key(self, individual: Dict[str, Any]) -> str:
        return json.dumps(individual, sort_keys=True, default=str)

    def _tournament_select(
        self,
        population: List[Dict[str, Any]],
        scores: List[float],
        rng: np.random.Generator,
        maximize: bool,
    ) -> Dict[str, Any]:
        k = max(1, min(int(self.tournament_size), len(population)))
        contenders = rng.choice(len(population), size=k, replace=False)
        best_idx = int(contenders[0])
        best_val = scores[best_idx]
        for idx in contenders[1:]:
            val = scores[int(idx)]
            if np.isnan(best_val) or (
                not np.isnan(val) and ((val > best_val) if maximize else (val < best_val))
            ):
                best_idx = int(idx)
                best_val = val
        return dict(population[best_idx])

    def _crossover(
        self,
        parent1: Dict[str, Any],
        parent2: Dict[str, Any],
        rng: np.random.Generator,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        child1, child2 = dict(parent1), dict(parent2)
        if rng.random() <= float(self.crossover_probability):
            for key in child1:
                if rng.random() < 0.5:
                    child1[key], child2[key] = child2[key], child1[key]
        return child1, child2

    def _mutate(
        self,
        individual: Dict[str, Any],
        space: Dict[str, Any],
        rng: np.random.Generator,
    ) -> Dict[str, Any]:
        mutated = dict(individual)
        for key, descriptor in space.items():
            if rng.random() <= float(self.mutation_probability):
                mutated[key] = _sample_evolutionary_gene(descriptor, rng)
        return mutated

    def _evaluate_individual(
        self,
        params: Dict[str, Any],
        X,
        y,
        splits: List[Tuple[np.ndarray, np.ndarray]],
        fit_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        import time

        fit_times: List[float] = []
        score_times: List[float] = []
        test_scores: List[float] = []
        train_scores: List[float] = []

        n_splits = len(splits)
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_tr = _slice_subset(X, train_idx)
            y_tr = _slice_subset(y, train_idx)
            X_te = _slice_subset(X, test_idx)
            y_te = _slice_subset(y, test_idx)

            t0 = time.perf_counter()
            try:
                est = clone(self.estimator).set_params(**params)
                est.fit(X_tr, y_tr, **fit_params)
                fit_time = time.perf_counter() - t0

                t1 = time.perf_counter()
                if callable(self.scoring):
                    te_score = float(self.scoring(est, X_te, y_te))
                else:
                    te_score = float(est.score(X_te, y_te))
                score_time = time.perf_counter() - t1

                if self.return_train_score:
                    if callable(self.scoring):
                        tr_score = float(self.scoring(est, X_tr, y_tr))
                    else:
                        tr_score = float(est.score(X_tr, y_tr))
                else:
                    tr_score = np.nan
            except Exception:
                if self.error_score == "raise":
                    raise
                fit_time = time.perf_counter() - t0
                score_time = 0.0
                te_score = float(self.error_score)
                tr_score = float(self.error_score)

            fit_times.append(fit_time)
            score_times.append(score_time)
            test_scores.append(te_score)
            train_scores.append(tr_score)

            if self.verbose:
                param_str = ", ".join(f"{k}={v}" for k, v in sorted(params.items()))
                total_min = (fit_time + score_time) / 60.0
                if self.return_train_score:
                    score_str = f"(train={tr_score:.3f}, test={te_score:.3f})"
                else:
                    score_str = f"{te_score:.3f}"
                print(
                    f"[CV {fold_idx + 1}/{n_splits}] END {param_str};, "
                    f"score={score_str} total time= {total_min:.1f}min"
                )

        return {
            "params": dict(params),
            "fit_times": fit_times,
            "score_times": score_times,
            "test_scores": test_scores,
            "train_scores": train_scores,
            "mean_test_score": float(np.nanmean(test_scores)) if not np.all(np.isnan(test_scores)) else np.nan,
            "std_test_score": float(np.nanstd(test_scores)) if not np.all(np.isnan(test_scores)) else np.nan,
            "mean_train_score": float(np.nanmean(train_scores)) if not np.all(np.isnan(train_scores)) else np.nan,
            "std_train_score": float(np.nanstd(train_scores)) if not np.all(np.isnan(train_scores)) else np.nan,
        }

    def fit(self, X, y=None, groups=None, **fit_params):
        if not self.param_grid:
            raise ValueError("GASearchCV requires a non-empty param_grid.")
        for name, minimum in (("population_size", 2), ("generations", 0), ("tournament_size", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be a single integer >= {minimum}.")
        if self.tournament_size > self.population_size:
            raise ValueError("tournament_size must not exceed population_size.")
        for name in ("crossover_probability", "mutation_probability"):
            value = getattr(self, name)
            if not np.isscalar(value) or not 0 <= float(value) <= 1:
                raise ValueError(f"{name} must be a single probability between 0 and 1.")
        space = prepare_evolutionary_space(self.param_grid)
        rng = np.random.default_rng(self.random_state)

        if hasattr(self.cv, "split"):
            splits = list(self.cv.split(X, y, groups=groups))
        elif isinstance(self.cv, int):
            from sklearn.model_selection import KFold

            splits = list(KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state).split(X, y))
        else:
            splits = list(self.cv)

        self.n_splits_ = len(splits)
        maximize = str(self.criteria).lower() != "min"
        pop_size = int(self.population_size)
        n_generations = int(self.generations)

        history_records: List[Dict[str, Any]] = []
        cache: Dict[str, Dict[str, Any]] = {}

        def evaluate_or_Lookup(ind: Dict[str, Any]) -> Dict[str, Any]:
            key = self._canonical_key(ind)
            if key not in cache:
                cache[key] = self._evaluate_individual(ind, X, y, splits, fit_params)
                history_records.append(cache[key])
            elif hasattr(self, "_on_candidate_cache_hit"):
                self._on_candidate_cache_hit()
            return cache[key]

        population = [self._sample_individual(space, rng) for _ in range(pop_size)]
        pop_records = [evaluate_or_Lookup(ind) for ind in population]
        pop_scores = [rec["mean_test_score"] for rec in pop_records]

        for gen in range(1, n_generations + 1):
            if getattr(self, "search_exhausted_", False):
                break
            if self.verbose:
                best_gen = np.nanmax(pop_scores) if maximize else np.nanmin(pop_scores)
                print(f"Generation {gen}/{n_generations} - current best CV score: {best_gen:.4f}")

            offspring: List[Dict[str, Any]] = []
            while len(offspring) < pop_size:
                p1 = self._tournament_select(population, pop_scores, rng, maximize)
                p2 = self._tournament_select(population, pop_scores, rng, maximize)
                c1, c2 = self._crossover(p1, p2, rng)
                offspring.append(self._mutate(c1, space, rng))
                if len(offspring) < pop_size:
                    offspring.append(self._mutate(c2, space, rng))

            off_records = [evaluate_or_Lookup(ind) for ind in offspring]
            off_scores = [rec["mean_test_score"] for rec in off_records]

            if self.elitism:
                combined_pop = population + offspring
                combined_scores = pop_scores + off_scores

                def sort_key(idx: int):
                    s = combined_scores[idx]
                    if np.isnan(s):
                        return -np.inf if maximize else np.inf
                    return s if maximize else -s

                ranked_indices = sorted(range(len(combined_pop)), key=sort_key, reverse=True)
                next_pop: List[Dict[str, Any]] = []
                next_scores: List[float] = []
                seen_keys = set()
                for idx in ranked_indices:
                    k = self._canonical_key(combined_pop[idx])
                    if k not in seen_keys:
                        seen_keys.add(k)
                        next_pop.append(combined_pop[idx])
                        next_scores.append(combined_scores[idx])
                    if len(next_pop) == pop_size:
                        break
                while len(next_pop) < pop_size:
                    idx = ranked_indices[len(next_pop) % len(ranked_indices)]
                    next_pop.append(combined_pop[idx])
                    next_scores.append(combined_scores[idx])
                population, pop_scores = next_pop, next_scores
            else:
                population, pop_scores = offspring, off_scores

        self.cv_results_ = self._build_cv_results(history_records, maximize)
        self.best_index_ = int(np.argmin(self.cv_results_["rank_test_score"]))
        self.best_score_ = float(self.cv_results_["mean_test_score"][self.best_index_])
        self.best_params_ = dict(self.cv_results_["params"][self.best_index_])

        if self.refit:
            self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_)
            self.best_estimator_.fit(X, y, **fit_params)

        return self

    def _build_cv_results(self, records: List[Dict[str, Any]], maximize: bool) -> Dict[str, Any]:
        n_candidates = len(records)
        cv_results: Dict[str, Any] = {
            "mean_fit_time": np.array([np.mean(r["fit_times"]) for r in records], dtype=float),
            "std_fit_time": np.array([np.std(r["fit_times"]) for r in records], dtype=float),
            "mean_score_time": np.array([np.mean(r["score_times"]) for r in records], dtype=float),
            "std_score_time": np.array([np.std(r["score_times"]) for r in records], dtype=float),
        }
        param_keys = list(records[0]["params"].keys()) if records else []
        for k in param_keys:
            cv_results[f"param_{k}"] = [r["params"].get(k) for r in records]
        cv_results["params"] = [dict(r["params"]) for r in records]

        for fold_idx in range(self.n_splits_):
            cv_results[f"split{fold_idx}_test_score"] = np.array(
                [r["test_scores"][fold_idx] for r in records], dtype=float
            )
        mean_test = np.array([r["mean_test_score"] for r in records], dtype=float)
        std_test = np.array([r["std_test_score"] for r in records], dtype=float)
        cv_results["mean_test_score"] = mean_test
        cv_results["std_test_score"] = std_test

        order_vals = np.where(
            np.isnan(mean_test),
            np.inf,
            -mean_test if maximize else mean_test,
        )
        ranks = np.empty(n_candidates, dtype=np.int32)
        ranks[np.argsort(order_vals, kind="mergesort")] = np.arange(1, n_candidates + 1, dtype=np.int32)
        cv_results["rank_test_score"] = ranks

        if self.return_train_score:
            for fold_idx in range(self.n_splits_):
                cv_results[f"split{fold_idx}_train_score"] = np.array(
                    [r["train_scores"][fold_idx] for r in records], dtype=float
                )
            cv_results["mean_train_score"] = np.array([r["mean_train_score"] for r in records], dtype=float)
            cv_results["std_train_score"] = np.array([r["std_train_score"] for r in records], dtype=float)

        return cv_results

    def predict(self, X):
        check_is_fitted(self, ["best_estimator_"])
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self, ["best_estimator_"])
        return self.best_estimator_.predict_proba(X)

    def score(self, X, y=None):
        check_is_fitted(self, ["best_estimator_"])
        return self.best_estimator_.score(X, y)


def build_quick_results_df(search_or_cv_results: Any, holdout_score: Optional[float] = None) -> pd.DataFrame:
    """Build CV results DataFrame with best_score (train, test) and holdout_score appended at the bottom."""
    if hasattr(search_or_cv_results, "cv_results_"):
        cv_df = pd.DataFrame(search_or_cv_results.cv_results_).copy()
        best_idx = getattr(search_or_cv_results, "best_index_", None)
        best_test_score = getattr(search_or_cv_results, "best_score_", None)
    elif isinstance(search_or_cv_results, pd.DataFrame):
        cv_df = search_or_cv_results.copy()
        best_idx = None
        best_test_score = None
    else:
        cv_df = pd.DataFrame(search_or_cv_results).copy()
        best_idx = None
        best_test_score = None

    while len(cv_df.columns) < 3:
        cv_df[f"col_{len(cv_df.columns)}"] = np.nan

    if best_idx is None and not cv_df.empty:
        if "rank_test_score" in cv_df.columns and cv_df["rank_test_score"].notna().any():
            best_idx = int(cv_df["rank_test_score"].astype(float).idxmin())
        elif "mean_test_score" in cv_df.columns and cv_df["mean_test_score"].notna().any():
            best_idx = int(cv_df["mean_test_score"].astype(float).idxmax())

    if best_idx is not None and 0 <= int(best_idx) < len(cv_df):
        if "mean_train_score" in cv_df.columns:
            best_train_score = float(cv_df["mean_train_score"].iloc[int(best_idx)])
        elif "split0_train_score" in cv_df.columns:
            best_train_score = float(cv_df["split0_train_score"].iloc[int(best_idx)])
        else:
            best_train_score = np.nan
        if best_test_score is None and "mean_test_score" in cv_df.columns:
            best_test_score = float(cv_df["mean_test_score"].iloc[int(best_idx)])
    else:
        best_train_score = np.nan

    best_test_val = float(best_test_score) if best_test_score is not None else np.nan

    n_cols = len(cv_df.columns)
    best_row = ["best_score", best_train_score, best_test_val] + [np.nan] * (n_cols - 3)
    summary_rows = [best_row]

    if holdout_score is not None:
        holdout_row = ["holdout_score", float(holdout_score)] + [np.nan] * (n_cols - 2)
        summary_rows.append(holdout_row)

    summary_df = pd.DataFrame(summary_rows, columns=cv_df.columns)
    return pd.concat([cv_df, summary_df], ignore_index=True)


# ============================================================
# Sensor augmentation
# ============================================================


def augment_jitter(x, sigma=0.03, rng=None):
    if sigma == 0 or x.size == 0:
        return x.copy()
    rng = rng or np.random.default_rng()
    return x + rng.normal(loc=0.0, scale=sigma, size=x.shape)


def augment_gaussian_noise(x, std=0.05, rng=None):
    if std == 0 or x.size == 0:
        return x.copy()
    rng = rng or np.random.default_rng()
    ch_std = np.nanstd(x, axis=0, keepdims=True)
    ch_std = np.where(ch_std == 0, 1.0, ch_std)
    return x + rng.normal(loc=0.0, scale=std * ch_std, size=x.shape)


def augment_scaling(x, sigma=0.1, rng=None):
    if sigma == 0 or x.size == 0:
        return x.copy()
    rng = rng or np.random.default_rng()
    factor = rng.normal(loc=1.0, scale=sigma)
    return x * factor


def augment_time_shift(x, max_shift_frac=0.1, rng=None):
    if max_shift_frac <= 0 or len(x) <= 1:
        return x.copy()
    rng = rng or np.random.default_rng()
    T = x.shape[0]
    max_shift = min(T - 1, max(1, int(T * max_shift_frac)))
    shift = rng.integers(-max_shift, max_shift + 1)
    if shift == 0:
        return x.copy()
    out = np.empty_like(x)
    if shift > 0:
        out[:shift] = x[0]
        out[shift:] = x[:-shift]
    else:
        out[shift:] = x[-1]
        out[:shift] = x[-shift:]
    return out


def _normalize_crop_frac_range(value):
    if value is None:
        return (0.5, 0.9)
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed = value.strip("()[]")
            parts = [p.strip() for p in parsed.split(",") if p.strip()]
            if len(parts) != 2:
                raise ValueError(f"Invalid crop_frac_range value: {value!r}")
            parsed = tuple(float(p) for p in parts)
        if isinstance(parsed, (tuple, list, np.ndarray)) and len(parsed) == 2:
            return (float(parsed[0]), float(parsed[1]))
        raise ValueError(f"Invalid crop_frac_range value: {value!r}")
    if isinstance(value, np.ndarray):
        arr = value.tolist()
        if len(arr) == 2:
            return (float(arr[0]), float(arr[1]))
        raise ValueError(f"Invalid crop_frac_range value: {value!r}")
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return (float(value[0]), float(value[1]))
    raise ValueError(f"Invalid crop_frac_range value: {value!r}")


def augment_crop_resize(x, crop_frac_range=(0.5, 0.9), rng=None):
    crop_frac_range = _normalize_crop_frac_range(crop_frac_range)
    if len(x) <= 1 or crop_frac_range == (1.0, 1.0):
        return x.copy()
    rng = rng or np.random.default_rng()
    x = np.asarray(x, dtype=float)
    squeeze = False
    if x.ndim == 1:
        x = x[:, None]
        squeeze = True
    elif x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    T, C = x.shape
    lo, hi = crop_frac_range
    crop_frac = rng.uniform(lo, hi)
    crop_len = max(2, int(T * crop_frac))
    crop_len = min(crop_len, T)
    start = rng.integers(0, T - crop_len + 1)
    cropped = x[start : start + crop_len]
    old_idx = np.linspace(0, 1, crop_len)
    new_idx = np.linspace(0, 1, T)
    out = np.empty_like(x)
    for c in range(C):
        out[:, c] = np.interp(new_idx, old_idx, cropped[:, c])
    return out[:, 0] if squeeze else out


def augment_temporal_mask(x, mask_frac=0.1, num_masks=1, rng=None):
    if mask_frac <= 0 or num_masks <= 0 or len(x) == 0:
        return x.copy()
    rng = rng or np.random.default_rng()
    T = x.shape[0]
    mask_len = min(T, max(1, int(T * mask_frac)))
    out = x.copy()
    for _ in range(num_masks):
        start = rng.integers(0, T - mask_len + 1)
        out[start : start + mask_len] = 0.0
    return out


def augment_channel_dropout(x, drop_prob=0.1, rng=None):
    if drop_prob <= 0 or x.size == 0:
        return x.copy()
    rng = rng or np.random.default_rng()
    C = x.shape[1]
    mask = rng.random(C) >= drop_prob
    return x * mask[np.newaxis, :]


def augment_sensor_dropout(x, sensor_groups, drop_prob=0.3, rng=None):
    if drop_prob <= 0 or x.size == 0:
        return x.copy()
    rng = rng or np.random.default_rng()
    out = x.copy()
    for _name, col_idx in sensor_groups.items():
        if not col_idx:
            continue
        if rng.random() < drop_prob:
            out[:, col_idx] = 0.0
    return out


def augment_magnitude_warp(x, sigma=0.2, num_knots=4, rng=None):
    if sigma == 0 or num_knots <= 0 or len(x) <= 1:
        return x.copy()
    rng = rng or np.random.default_rng()
    if CubicSpline is None:
        return x
    T, C = x.shape
    num_knots = max(2, min(num_knots, T))
    knot_x = np.linspace(0, T - 1, num_knots)
    knot_y = rng.normal(loc=1.0, scale=sigma, size=(num_knots, C))
    cs = CubicSpline(knot_x, knot_y, extrapolate=True)
    warp = cs(np.arange(T))
    return x * warp


_AUGMENTATION_REGISTRY = {
    "jitter": augment_jitter,
    "gaussian_noise": augment_gaussian_noise,
    "scaling": augment_scaling,
    "time_shift": augment_time_shift,
    "crop_resize": augment_crop_resize,
    "temporal_mask": augment_temporal_mask,
    "channel_dropout": augment_channel_dropout,
    "sensor_dropout": augment_sensor_dropout,
    "magnitude_warp": augment_magnitude_warp,
}

ALL_AUGMENTATION_NAMES = tuple(_AUGMENTATION_REGISTRY.keys())

_SENSOR_PREFIXES = {
    "acc":  ("acc_", "lin_acc_"),
    "rot":  ("rot_", "rot6d_", "ang_vel_"),
    "tof":  ("tof_", "depth_"),
    "thm":  ("thm_", "thermal_", "temp_"),
    "stft": ("stft_",),
    "cwt": ("cwt_",),
}


def _detect_sensor_groups(columns):
    groups = {}
    for sensor, prefixes in _SENSOR_PREFIXES.items():
        idx = [i for i, c in enumerate(columns) if any(c.startswith(p) for p in prefixes)]
        groups[sensor] = idx
    return groups


def _coerce_augmentation_names(value):
    """Accept names as a sequence or as the JSON string produced for Bayesian search."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = [p.strip().strip("'\"") for p in text.strip("()[]").split(",") if p.strip()]
    if isinstance(value, str):
        value = [value]
    names = list(value)
    unknown = [n for n in names if n not in _AUGMENTATION_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown augmentation(s) {unknown}; expected a subset of {sorted(_AUGMENTATION_REGISTRY)}")
    return names

# ===========================================================================
# 10. Augmentation Adapter & Parent-Child Hierarchy
# ===========================================================================

try:
    import tsaug as _tsaug
    TSAUG_AVAILABLE = True
except Exception:
    _tsaug = None
    TSAUG_AVAILABLE = False


class TsaugAdapter:
    """Swappable augmentation backend delegating to ``tsaug`` with safe NumPy fallback."""

    archived_upstream: bool = True
    upstream_url: str = "https://github.com/arundo/tsaug"

    @staticmethod
    def available() -> bool:
        return TSAUG_AVAILABLE

    @staticmethod
    def add_noise(
        sub: np.ndarray,
        scale: float,
        *,
        distr: str = "gaussian",
        seed: int = 42,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        if scale <= 0:
            return sub.copy()
        if TSAUG_AVAILABLE:
            op = _tsaug.AddNoise(scale=float(scale), distr=distr, normalize=False, seed=int(seed))
            return op.augment(sub)
        gen = rng or np.random.default_rng(int(seed))
        if distr == "uniform":
            return sub + gen.uniform(-float(scale), float(scale), size=sub.shape)
        return sub + gen.normal(0.0, float(scale), size=sub.shape)

    @staticmethod
    def multiplicative_drift(
        sub: np.ndarray,
        max_drift: float,
        n_drift_points: int,
        *,
        seed: int = 42,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        if max_drift <= 0:
            return sub.copy()
        if TSAUG_AVAILABLE:
            op = _tsaug.Drift(
                max_drift=float(max_drift),
                n_drift_points=int(n_drift_points),
                kind="multiplicative",
                per_channel=True,
                normalize=False,
                seed=int(seed),
            )
            return op.augment(sub)
        gen = rng or np.random.default_rng(int(seed))
        m, length, channels = sub.shape
        t_grid = np.arange(length, dtype=np.float64)
        n_knots = max(2, int(n_drift_points) + 2)
        knot_t = np.linspace(0.0, float(length - 1), n_knots)
        knot_vals = 1.0 + gen.uniform(-float(max_drift), float(max_drift), size=(m, n_knots, channels))
        out = np.empty_like(sub)
        for i in range(m):
            for c in range(channels):
                out[i, :, c] = sub[i, :, c] * np.interp(t_grid, knot_t, knot_vals[i, :, c])
        return out

    @staticmethod
    def time_warp(
        sub: np.ndarray,
        n_speed_change: int,
        max_speed_ratio: float,
        *,
        seed: int = 42,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        if max_speed_ratio <= 1.0:
            return sub.copy()
        if TSAUG_AVAILABLE:
            op = _tsaug.TimeWarp(
                n_speed_change=int(n_speed_change),
                max_speed_ratio=float(max_speed_ratio),
                seed=int(seed),
            )
            return op.augment(sub)
        gen = rng or np.random.default_rng(int(seed))
        m, length, channels = sub.shape
        t_grid = np.arange(length, dtype=np.float64)
        n_anchors = max(2, int(n_speed_change) + 2)
        anchor_t = np.linspace(0.0, float(length - 1), n_anchors)
        out = np.empty_like(sub)
        for i in range(m):
            speeds = gen.uniform(1.0 / max_speed_ratio, max_speed_ratio, size=n_anchors - 1)
            cum = np.r_[0.0, np.cumsum(speeds)]
            cum = cum / max(cum[-1], 1e-9) * float(length - 1)
            tau = np.interp(t_grid, anchor_t, cum)
            for c in range(channels):
                out[i, :, c] = np.interp(tau, t_grid, sub[i, :, c])
        return out

    @staticmethod
    def crop_resize(
        sub: np.ndarray,
        crop_range: Tuple[float, float],
        *,
        seed: int = 42,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        lo, hi = float(crop_range[0]), float(crop_range[1])
        length = sub.shape[1]
        if hi >= 1.0 and lo >= 1.0:
            return sub.copy()
        if TSAUG_AVAILABLE:
            sizes = sorted({int(np.clip(round(f * length), 2, length)) for f in np.linspace(lo, hi, 6)})
            op = _tsaug.Crop(size=sizes, resize=length, seed=int(seed))
            return op.augment(sub)
        gen = rng or np.random.default_rng(int(seed))
        m, _, channels = sub.shape
        t_grid = np.arange(length, dtype=np.float64)
        out = np.empty_like(sub)
        for i in range(m):
            crop_len = int(np.clip(round(gen.uniform(lo, hi) * length), 2, length))
            start = int(gen.integers(0, max(1, length - crop_len + 1)))
            src_t = np.linspace(float(start), float(start + crop_len - 1), length)
            for c in range(channels):
                out[i, :, c] = np.interp(src_t, t_grid, sub[i, :, c])
        return out

    @staticmethod
    def temporal_dropout(
        sub: np.ndarray,
        n_masks: int,
        mask_frac: float,
        fill: Any = "ffill",
        *,
        seed: int = 42,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        if n_masks <= 0 or mask_frac <= 0:
            return sub.copy()
        length = sub.shape[1]
        size = max(1, int(round(float(mask_frac) * length / max(1, int(n_masks)))))
        fill_val = fill if isinstance(fill, str) else float(fill)
        if TSAUG_AVAILABLE:
            op = _tsaug.Dropout(p=float(mask_frac), size=[size], fill=fill_val, per_channel=False, seed=int(seed))
            return op.augment(sub)
        gen = rng or np.random.default_rng(int(seed))
        out = sub.copy()
        for i in range(sub.shape[0]):
            for _ in range(max(1, int(n_masks))):
                s = int(gen.integers(0, max(1, length - size + 1)))
                e = min(length, s + size)
                if fill == "ffill" and s > 0:
                    out[i, s:e, :] = out[i, s - 1 : s, :]
                elif isinstance(fill, (int, float, np.number)):
                    out[i, s:e, :] = float(fill)
                else:
                    out[i, s:e, :] = 0.0
        return out

    @staticmethod
    def channel_dropout(
        sub: np.ndarray,
        drop_prob: float,
        *,
        seed: int = 42,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        if drop_prob <= 0:
            return sub.copy()
        length = sub.shape[1]
        if TSAUG_AVAILABLE:
            op = _tsaug.Dropout(p=float(drop_prob), size=[length], fill=0.0, per_channel=True, seed=int(seed))
            return op.augment(sub)
        gen = rng or np.random.default_rng(int(seed))
        keep = (gen.random((sub.shape[0], 1, sub.shape[2])) >= float(drop_prob)).astype(sub.dtype)
        return sub * keep

    @staticmethod
    def protect_quaternion_norm(arr: np.ndarray, channel_names: Sequence[str]) -> np.ndarray:
        """Re-normalize quaternion channels ``(rot_w, rot_x, rot_y, rot_z)`` to unit norm."""
        q_names = ("rot_w", "rot_x", "rot_y", "rot_z")
        if not all(name in channel_names for name in q_names):
            return arr
        idx = [channel_names.index(name) for name in q_names]
        out = arr.copy()
        q = out[:, :, idx]
        norms = np.linalg.norm(q, axis=-1, keepdims=True)
        valid = norms > 1e-6
        out[:, :, idx] = np.where(valid, q / np.maximum(norms, 1e-6), q)
        return out


def shift_time(x: np.ndarray, max_shift_frac: float, rng: np.random.Generator) -> np.ndarray:
    """Per-sample circular-free shift by up to ``max_shift_frac * T`` samples (edge replicated).

    ``tsaug`` has no time-shift augmenter, so this helper is provided alongside ``TsaugAdapter``.
    """
    n, length = x.shape[0], x.shape[1]
    limit = int(round(float(max_shift_frac) * length))
    if limit <= 0:
        return x
    out = np.empty_like(x)
    shifts = rng.integers(-limit, limit + 1, size=n)
    base = np.arange(length)
    for i in range(n):
        out[i] = x[i, np.clip(base - shifts[i], 0, length - 1)]
    return out

# ---------------------------------------------------------------------------
# Abstract Parent Augmentor: BaseAugmentor (HoneycombBase)
# ---------------------------------------------------------------------------


class BaseAugmentor(HoneycombBase, ABC):
    """Shared parent Honeycomb cell for ``SensorAugmentor`` (base_utils) and ``TSAugmentor`` (utils)."""

    @staticmethod
    def _validate_probability(
        value: Any,
        name: str,
        *,
        strict_below_one: bool = False,
        exc_type: type = InvalidParams,
    ) -> float:
        try:
            prob = float(value)
        except (TypeError, ValueError) as exc:
            raise exc_type(f"{name} must be a finite probability in [0, 1]") from exc
        if not np.isfinite(prob) or prob < 0.0 or prob > 1.0:
            raise exc_type(f"{name} must be in [0, 1], got {value!r}")
        if strict_below_one and prob >= 1.0:
            raise exc_type(f"{name} must stay strictly below 1.0, got {value!r}")
        return prob

    @staticmethod
    def _validate_non_negative(
        value: Any,
        name: str,
        *,
        exc_type: type = InvalidParams,
    ) -> float:
        try:
            val = float(value)
        except (TypeError, ValueError) as exc:
            raise exc_type(f"{name} must be a non-negative finite number") from exc
        if not np.isfinite(val) or val < 0.0:
            raise exc_type(f"{name} must be >= 0, got {value!r}")
        return val

    @staticmethod
    def _make_rng(seed: Optional[int]) -> np.random.Generator:
        return np.random.default_rng(None if seed is None else int(seed))



class SensorAugmentor(BaseAugmentor):
    """Sequence augmentation with an explicit geometry boundary.

    ``transform`` accepts raw recordings and preserves all raw acceleration and
    rotation columns together. Generic component arithmetic/interpolation is
    not an orientation augmentation. The native multibranch path instead calls
    ``transform_features`` after cleaning, gravity removal, rotation derivatives
    and spectra, so dropout can zero IMU model features safely. A zero processed
    quaternion feature is a masked input, not an orientation for further math.
    """
    def __init__(
        self,
        augmentations: Optional[List[str]] = None,
        prob: float = 0.5,
        per_aug_prob: float = 0.5,
        jitter_sigma: float = 0.03,
        noise_std: float = 0.05,
        scaling_sigma: float = 0.1,
        time_shift_frac: float = 0.1,
        crop_frac_range=(0.5, 0.9),
        temporal_mask_frac: float = 0.1,
        temporal_num_masks: int = 1,
        channel_drop_prob: float = 0.1,
        sensor_drop_prob: float = 0.3,
        warp_sigma: float = 0.2,
        warp_num_knots: int = 4,
        sequence_col: str = "sequence_id",
        counter_col: str = "sequence_counter",
        seed: Optional[int] = 42,
        diagnostics: bool = False,
        diagnostic_max_sequences: int = 5,
        diagnostic_max_locations: int = 5,
    ):
        self.augmentations = augmentations
        self.prob = prob
        self.per_aug_prob = per_aug_prob
        self.jitter_sigma = jitter_sigma
        self.noise_std = noise_std
        self.scaling_sigma = scaling_sigma
        self.time_shift_frac = time_shift_frac
        self.crop_frac_range = crop_frac_range
        self.temporal_mask_frac = temporal_mask_frac
        self.temporal_num_masks = temporal_num_masks
        self.channel_drop_prob = channel_drop_prob
        self.sensor_drop_prob = sensor_drop_prob
        self.warp_sigma = warp_sigma
        self.warp_num_knots = warp_num_knots
        self.sequence_col = sequence_col
        self.counter_col = counter_col
        self.seed = seed
        self.diagnostics = diagnostics
        self.diagnostic_max_sequences = diagnostic_max_sequences
        self.diagnostic_max_locations = diagnostic_max_locations

    def fit(self, X, y=None):
        self._rng = np.random.default_rng(self.seed)
        self.diagnostics_ = []
        names = _coerce_augmentation_names(self.augmentations)
        self.aug_names_ = list(ALL_AUGMENTATION_NAMES if names is None else names)
        skip = {self.sequence_col, self.counter_col}
        self.sensor_cols_ = [
            c for c in X.columns
            if c not in skip
            and c.startswith(tuple(p for prefixes in _SENSOR_PREFIXES.values() for p in prefixes))
            and pd.api.types.is_numeric_dtype(X[c].dtype)
        ]
        self.protected_sensor_cols_ = [c for c in self.sensor_cols_
                                       if c.startswith(("acc_", "rot_", "rot6d_", "ang_vel_"))]
        self.mutable_sensor_cols_ = [c for c in self.sensor_cols_ if c not in self.protected_sensor_cols_]
        self.sensor_groups_ = _detect_sensor_groups(self.mutable_sensor_cols_)
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, "sensor_cols_")
        self.diagnose_input(X)
        return self._transform_frame(X, self.mutable_sensor_cols_, raw=True)

    def transform_features(self, X, feature_columns=None):
        """Augment already calculated features, never followed by geometry.

        The extractor supplies its actual feature columns, including arbitrary
        spectral prefixes. Metadata and numeric labels are not guessed as model
        features when this method is used independently.
        """
        check_is_fitted(self, "sensor_cols_")
        if feature_columns is None:
            prefixes = tuple(p for group in _SENSOR_PREFIXES.values() for p in group)
            feature_columns = [c for c in X if c.startswith(prefixes)]
        skip = {self.sequence_col, self.counter_col, "mask", "dt", "rot_dt"}
        columns = [c for c in feature_columns if c not in skip and c in X
                   and pd.api.types.is_numeric_dtype(X[c].dtype)]
        return self._transform_frame(X, columns, raw=False)

    def _diagnostics_enabled(self):
        return bool(self.diagnostics or getattr(self, "diagnostics_enabled_", False))

    def _validity(self, frame, columns, raw):
        """Only called in opt-in diagnostics; locations are bounded, data omitted."""
        limit = max(0, int(self.diagnostic_max_locations))
        values = frame[columns].to_numpy(dtype=float)
        bad = ~np.isfinite(values)

        def location(row):
            item = {"row_position": int(row), "index": str(frame.index[row])}
            if self.counter_col in frame:
                item["counter"] = str(frame[self.counter_col].iloc[row])
            return item

        nonfinite = []
        if limit and bad.any():
            for row in range(len(frame)):
                for col in np.flatnonzero(bad[row])[:limit - len(nonfinite)]:
                    nonfinite.append({**location(row), "column": columns[col]})
                if len(nonfinite) >= limit:
                    break
        result = {"nonfinite_count": int(bad.sum()), "nonfinite_locations": nonfinite,
                  "zero_quaternion_count": 0, "near_zero_quaternion_count": 0,
                  "quaternion_locations": []}
        qcols = ["rot_w", "rot_x", "rot_y", "rot_z"]
        if raw and all(c in frame for c in qcols):
            q = frame[qcols].to_numpy(dtype=float)
            norms = np.linalg.norm(q, axis=1)
            zero = np.isfinite(norms) & (norms <= 0)
            near = np.isfinite(norms) & (norms > 0) & (norms <= 1e-12)
            result["zero_quaternion_count"] = int(zero.sum())
            result["near_zero_quaternion_count"] = int(near.sum())
            result["quaternion_locations"] = [
                {**location(row), "norm": float(norms[row])}
                for row in np.flatnonzero(zero | near)[:limit]
            ]
        result["invalid"] = bool(result["nonfinite_count"] or result["zero_quaternion_count"])
        return result

    def _diagnostic_report(self, frame, columns, raw):
        validity = self._validity(frame, columns, raw)
        context = dict(getattr(self, "diagnostic_context_", {}))
        return {**context, "seed": self.seed,
                "rng_state": self._rng.bit_generator.state,
                "sequence_id": str(frame[self.sequence_col].iloc[0]),
                "stage": "raw" if raw else "features",
                "input_invalid": validity["invalid"], "input_validity": validity,
                "operations": [], "first_invalid_operation": "input" if validity["invalid"] else None,
                "validity": validity}

    def _save_diagnostic(self, report):
        if any(saved is report for saved in self.diagnostics_):
            return
        limit = max(0, int(self.diagnostic_max_sequences))
        if len(self.diagnostics_) < limit:
            self.diagnostics_.append(report)
        elif limit and report.get("first_invalid_operation") is not None:
            self.diagnostics_[-1] = report

    def _raise_invalid(self, report):
        error = InvalidExtractorParams(
            "Augmentation input was already invalid before augmentation"
            if report["input_invalid"] else
            f"Augmentation operation {report['first_invalid_operation']!r} created invalid data"
        )
        error.augmentation_diagnostics = report
        error.failure_stage = "augmentation"
        self._save_diagnostic(report)
        raise error

    def diagnose_input(self, X):
        """Optionally record invalid raw input before any augmentation operation."""
        if not self._diagnostics_enabled():
            return
        columns = [c for c in self.sensor_cols_ if c in X]
        for positions in X.groupby(self.sequence_col, sort=False).indices.values():
            grp = X.iloc[positions]
            report = self._diagnostic_report(grp, columns, raw=True)
            # Missing values may be intentionally interpolated by SignalCleaner.
            # Record them; the cleaner still decides whether they are recoverable.
            if report["validity"]["zero_quaternion_count"]:
                self._raise_invalid(report)
            if report["input_invalid"] or report["validity"]["near_zero_quaternion_count"]:
                self._save_diagnostic(report)

    def _transform_frame(self, X, columns, raw):
        if self.prob <= 0 or self.per_aug_prob <= 0 or not self.aug_names_ or not columns:
            return X
        out = X.copy()
        column_positions = out.columns.get_indexer(columns)
        diagnostics = self._diagnostics_enabled()
        # Positional indexing also works with duplicate caller index labels and
        # interleaved sequences; no operation crosses sequence boundaries.
        for positions in out.groupby(self.sequence_col, sort=False).indices.values():
            if self._rng.random() >= self.prob:
                continue
            grp = out.iloc[positions]
            vals = grp[columns].to_numpy(dtype=float)
            checked_columns = [c for c in self.sensor_cols_ if c in grp] if raw else columns
            report = self._diagnostic_report(grp, checked_columns, raw) if diagnostics else None
            if report and report["input_invalid"]:
                self._raise_invalid(report)
            for aug_name in self.aug_names_:
                if self._rng.random() >= self.per_aug_prob:
                    continue
                if report is not None:
                    report["operations"].append({"name": aug_name, "parameters": {}})
                try:
                    fn = _AUGMENTATION_REGISTRY[aug_name]
                    kwargs = self._build_kwargs(aug_name)
                    if aug_name == "sensor_dropout":
                        kwargs["sensor_groups"] = _detect_sensor_groups(columns)
                    if report is not None:
                        report["operations"][-1]["parameters"] = kwargs
                    updated = fn(vals, rng=self._rng, **kwargs)
                    if np.shape(updated) != np.shape(vals):
                        raise ValueError(f"{aug_name} changed sequence shape {np.shape(vals)} to {np.shape(updated)}")
                    vals = updated
                    if report is not None:
                        checked = grp.copy()
                        checked[columns] = vals
                        validity = self._validity(checked, checked_columns, raw)
                        report["validity"] = validity
                        if validity["invalid"]:
                            report["first_invalid_operation"] = aug_name
                            self._raise_invalid(report)
                except Exception as exc:
                    if report is not None:
                        report["first_invalid_operation"] = report["first_invalid_operation"] or aug_name
                        report["exception"] = f"{type(exc).__name__}: {exc}"
                        report["traceback"] = traceback.format_exc(limit=20)
                        self._save_diagnostic(report)
                        exc.augmentation_diagnostics = report
                    exc.failure_stage = "augmentation"
                    raise
            out.iloc[positions, column_positions] = vals
            if report is not None:
                self._save_diagnostic(report)
        return out

    @staticmethod
    def _coerce_crop_frac_range(value):
        return _normalize_crop_frac_range(value)

    def _build_kwargs(self, name):
        table = {
            "jitter":          {"sigma": self.jitter_sigma},
            "gaussian_noise":  {"std": self.noise_std},
            "scaling":         {"sigma": self.scaling_sigma},
            "time_shift":      {"max_shift_frac": self.time_shift_frac},
            "crop_resize":     {"crop_frac_range": self.crop_frac_range},
            "temporal_mask":   {"mask_frac": self.temporal_mask_frac, "num_masks": self.temporal_num_masks},
            "channel_dropout": {"drop_prob": self.channel_drop_prob},
            "sensor_dropout":  {"sensor_groups": self.sensor_groups_, "drop_prob": self.sensor_drop_prob},
            "magnitude_warp":  {"sigma": self.warp_sigma, "num_knots": self.warp_num_knots},
        }
        return table.get(name, {})

    def get_feature_names_out(self, input_features=None):
        return np.array(self.sensor_cols_)



# ---------------------------------------------------------------------------
# Dataset discovery and sequence-balanced sampling
# ---------------------------------------------------------------------------


def find_data_root(local_dir: str = "data", kaggle_root: str = "/kaggle/input") -> Path:
    """Locate the directory containing CMI ``train.csv`` (preferred) or ``sample.csv`` as a ``Path``."""
    search_starts: List[Path] = [Path.cwd()]
    try:
        search_starts.append(Path(__file__).resolve().parent)
    except Exception:
        pass
    try:
        frame = inspect.currentframe()
        while frame:
            fname = frame.f_globals.get("__file__")
            if fname:
                search_starts.append(Path(fname).resolve().parent)
            frame = frame.f_back
    except Exception:
        pass

    unique_starts: List[Path] = []
    for p in search_starts:
        if p not in unique_starts:
            unique_starts.append(p)

    exact_path = Path(local_dir)
    kaggle_path = Path(kaggle_root)

    # Pass 1: prefer full train.csv across local + Kaggle before falling back to sample.csv
    for required_file in ("train.csv", "sample.csv"):
        if exact_path.exists() and exact_path.is_dir() and (exact_path / required_file).exists():
            print(f"Using exact path: {exact_path.resolve()}")
            return exact_path
        for start_path in unique_starts:
            current = start_path.resolve()
            for parent in [current] + list(current.parents):
                target = parent / local_dir
                if target.exists() and target.is_dir() and (target / required_file).exists():
                    print(f"Found data folder: {target}")
                    return target
        if kaggle_path.exists():
            for candidate in kaggle_path.rglob(required_file):
                target_dir = candidate.parent
                print(f"Using Kaggle data folder: {target_dir}")
                return target_dir

    raise FileNotFoundError(
        f"Could not find the dataset.\n"
        f"Searched upwards from: {[str(p) for p in unique_starts]}\n"
        f"Looked for a folder named {local_dir!r} containing train.csv or sample.csv.\n"
        f"Current directory: {Path.cwd()}\n"
        f'Please ensure your CSV files are in a "data/" folder at your project root.'
    )


sample_balanced_split = sample_balanced_train_test_split

