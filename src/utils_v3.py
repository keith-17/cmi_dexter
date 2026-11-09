"""utils_v3 -- open-source multibranch stack for the CMI BFRB competition.

One consolidated module (``notebooks/multi-branch_version_3.ipynb`` imports only
this file).  Sections, in order:

  1. Optional open-source libraries   tsaug / tsfel / tsfresh / tsai / torch
  2. Time utilities                   rate handling, resampling, runs, Nyquist-aware bands
  3. NOOP descriptors                 retained reverse-engineered motion / regularity features,
                                      extended with the cr7_kimmitch ``ImuFeatureExtractor``
                                      (accel energy, gyro energy, jerk, ACF gait cadence)
  4. Sequence extractor               per-modality chunks that adapt to each sensor's sample rate
  5. Augmentation                     tsaug based, every knob an sklearn parameter
  6. Network                          tsai backbones per modality + fusion head + sklearn estimator
  7. Competition scorer               original formula + hidden-test (50 % ToF/thermopile null) simulation
  8. Search space                     ONE declarative dict -> grid / Bayesian / evolutionary views
  9. Search drivers, validation runner, permutation importance, synthetic data

Design notes
------------
* Rows of a CMI sequence share one clock (``row_sampling_rate``).  Each sensor has a
  native rate; in ``align_mode='time'`` it is resampled (anti-aliased) to its own
  target rate, so IMU (100 Hz) and ToF/thermopile (20 Hz) windows cover the same
  *seconds* with different numbers of samples.  Every frequency feature then uses the
  rate of the array it was computed from.
* Missing observations are never compacted into a faster clock: sequences are placed on
  a dense counter grid and absent rows are NaN / unobserved.
* ``NOOP`` features are off by default (all ``None``), exactly like the legacy extractor.
"""
from __future__ import annotations

import ast
import gc
import hashlib
import itertools
import json
import math
import time
import warnings
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import signal as sps
from scipy.spatial.transform import Rotation
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, ParameterGrid, ParameterSampler
from sklearn.preprocessing import LabelEncoder

import torch
from torch import nn

# ---------------------------------------------------------------------------
# 1. Optional open-source libraries (imported lazily where they are heavy)
# ---------------------------------------------------------------------------

try:  # augmentation
    import tsaug
    TSAUG_AVAILABLE = True
except Exception:  # pragma: no cover - depends on environment
    tsaug = None
    TSAUG_AVAILABLE = False

try:  # general time/spectral features
    import tsfel
    TSFEL_AVAILABLE = True
except Exception:  # pragma: no cover
    tsfel = None
    TSFEL_AVAILABLE = False

try:
    import tsfresh  # noqa: F401
    from tsfresh import extract_features as _tsfresh_extract
    TSFRESH_AVAILABLE = True
except Exception:  # pragma: no cover
    _tsfresh_extract = None
    TSFRESH_AVAILABLE = False

try:
    import importlib.util as _ilu
    TSAI_AVAILABLE = _ilu.find_spec("tsai") is not None
except Exception:  # pragma: no cover
    TSAI_AVAILABLE = False

try:
    from skopt import BayesSearchCV
    from skopt.space import Categorical as SkCategorical, Integer as SkInteger, Real as SkReal
    SKOPT_AVAILABLE = True
except Exception:  # pragma: no cover
    BayesSearchCV = SkCategorical = SkInteger = SkReal = None
    SKOPT_AVAILABLE = False


class InvalidParams(ValueError):
    """A hyperparameter combination that cannot produce a valid model/features."""


def library_status() -> Dict[str, bool]:
    """Which open-source building blocks are importable in this environment."""
    return {"torch": True, "tsai": TSAI_AVAILABLE, "tsaug": TSAUG_AVAILABLE, "tsfel": TSFEL_AVAILABLE,
            "tsfresh": TSFRESH_AVAILABLE, "scikit-optimize": SKOPT_AVAILABLE}


def _require(flag: bool, name: str, pip: Optional[str] = None):
    if not flag:
        raise ImportError(f"{name} is required for this option. Install with: pip install {pip or name}")


# ---------------------------------------------------------------------------
# 2. Time utilities
# ---------------------------------------------------------------------------

SEQ_COL = "sequence_id"
COUNTER_COL = "sequence_counter"
MODALITIES = ("imu", "thm", "tof")
GRAVITY = 9.80665
TOF_SHAPE = (5, 8, 8)           # sensors x rows x cols; 320 pixels, columns tof_{s}_v{p}
TOF_COLUMNS = [f"tof_{s}_v{p}" for s in range(1, 6) for p in range(64)]
THM_COLUMNS = [f"thm_{s}" for s in range(1, 6)]
ACC_COLUMNS = ["acc_x", "acc_y", "acc_z"]
ROT_COLUMNS = ["rot_w", "rot_x", "rot_y", "rot_z"]
SENSOR_PREFIXES = ("acc_", "rot_", "tof_", "thm_")


def positive_rate(value: Any, name: str = "sampling rate") -> float:
    try:
        rate = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidParams(f"{name} must be a finite positive number") from exc
    if not np.isfinite(rate) or rate <= 0:
        raise InvalidParams(f"{name} must be a finite positive number, got {value!r}")
    return rate


def rate_ratio(source: float, target: float, max_denominator: int = 1000) -> Fraction:
    """Exact-ish rational ``target / source`` used for polyphase resampling."""
    ratio = Fraction(positive_rate(target) / positive_rate(source)).limit_denominator(max_denominator)
    if ratio <= 0:
        raise InvalidParams(f"cannot resample {source} Hz -> {target} Hz")
    return ratio


def seconds_to_samples(seconds: float, rate: float, minimum: int = 1) -> int:
    """Window length in samples for a duration at a given rate (>= ``minimum``)."""
    return max(int(minimum), int(round(float(seconds) * positive_rate(rate))))


def nyquist(rate: float) -> float:
    return positive_rate(rate) / 2.0


def resample_array(x: np.ndarray, source_rate: float, target_rate: float, method: str = "poly") -> np.ndarray:
    """Resample ``x`` (time on axis 0, NaN-free) from ``source_rate`` to ``target_rate``.

    ``poly`` applies SciPy's anti-aliased polyphase filter (decimation is low-pass
    filtered, so a 100 Hz signal reduced to 20 Hz does not alias).  ``linear`` is a
    plain interpolation, ``subsample`` keeps the nearest source rows.
    """
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
    raise InvalidParams("resample method must be 'poly', 'linear' or 'subsample'")


def nearest_indices(n_source: int, source_rate: float, target_rate: float) -> np.ndarray:
    """Source row index for each output sample after a rate change (used for masks)."""
    ratio = rate_ratio(source_rate, target_rate)
    n_out = max(1, int(round(n_source * float(ratio))))
    return np.clip(np.round(np.arange(n_out) / float(ratio)).astype(int), 0, max(0, n_source - 1))


def infer_hold_rate(values: np.ndarray, row_rate: float, max_hold: int = 50) -> float:
    """Estimate a sensor's native rate from its sample-and-hold pattern.

    Sensors slower than the row clock repeat each reading for ``k`` rows.  The median
    length of constant runs gives ``k``; the native rate is ``row_rate / k``.  When rows
    change on every step the row rate itself is returned.
    """
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


def contiguous_runs(valid: np.ndarray, counter: np.ndarray) -> List[np.ndarray]:
    """Contiguous positions; neither missing rows nor counter gaps are joined."""
    positions = np.flatnonzero(valid)
    if not len(positions):
        return []
    cuts = np.flatnonzero((np.diff(positions) != 1) | (np.diff(counter[positions]) != 1)) + 1
    return np.split(positions, cuts)


def bounded_fill(values: np.ndarray, observed: np.ndarray, counter: np.ndarray, max_gap: int):
    """Linearly bridge interior gaps of at most ``max_gap`` rows (never edges)."""
    result = np.array(values, dtype=float, copy=True)
    valid = np.asarray(observed, dtype=bool).copy()
    filled = np.zeros(len(valid), dtype=bool)
    if max_gap:
        for gap in contiguous_runs(~valid, np.arange(len(valid))):
            first, last = int(gap[0]), int(gap[-1])
            if (len(gap) > max_gap or first == 0 or last == len(valid) - 1
                    or not valid[first - 1] or not valid[last + 1]
                    or np.any(np.diff(counter[first - 1:last + 2]) != 1)):
                continue
            fraction = np.arange(1, len(gap) + 1) / (len(gap) + 1)
            result[gap] = (result[first - 1][None, :] * (1 - fraction[:, None])
                           + result[last + 1][None, :] * fraction[:, None])
            valid[gap] = True
            filled[gap] = True
    return result, valid, filled


def detrend_values(values: np.ndarray, mode: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    centered = values - np.mean(values, axis=0)
    if mode == "linear" and len(values) > 1:
        t = np.arange(len(values), dtype=float)
        t -= t.mean()
        centered -= t[:, None] * (t @ centered / (t @ t))[None, :]
    return centered


def one_sided_psd(coefficients: np.ndarray, nfft: int) -> np.ndarray:
    """Convert one-sided FFT coefficients to a real-signal power density."""
    power = np.abs(coefficients) ** 2
    weights = np.full(power.shape[0], 2.0)
    weights[0] = 1.0
    if nfft % 2 == 0:
        weights[-1] = 1.0
    return power * weights[:, None]


def window_psd(values: np.ndarray, fs: float, detrend: str):
    """One-sided density (Hann window) summed over channels."""
    x = detrend_values(values, detrend)
    window = np.hanning(len(x) + 1)[:-1]  # periodic Hann, as SciPy's fftbins=True
    coefficients = np.fft.rfft(x * window[:, None], axis=0) / np.sqrt(fs * np.sum(window ** 2))
    density = one_sided_psd(coefficients, len(x)).sum(axis=1)
    return np.fft.rfftfreq(len(x), d=1 / fs), density


def resolve_power_bands(bands: Sequence[Tuple[float, float]], fs: float, n_samples: int,
                        min_cycles: float = 2.0, margin: float = 0.98):
    """Make requested frequency bands valid for *this* sensor's rate and window.

    Returns ``[(low, high, usable)]`` in the input order.  ``high`` is clipped to
    ``margin * Nyquist``; ``low`` is raised to the smallest frequency with ``min_cycles``
    cycles inside the window.  A band that collapses is kept but flagged unusable so
    feature schemas never change between sensors, windows or sample rates.
    """
    fs = positive_rate(fs)
    ceiling = margin * fs / 2.0
    floor = min_cycles * fs / max(1, int(n_samples))
    resolved = []
    for low, high in bands:
        low, high = float(low), float(high)
        if not (np.isfinite(low) and np.isfinite(high)) or not 0 <= low < high:
            raise InvalidParams(f"band ({low}, {high}) must satisfy 0 <= low < high")
        lo, hi = max(low, floor), min(high, ceiling)
        resolved.append((lo, hi, bool(lo < hi)))
    return resolved


def window_starts(n: int, window: int, stride: int, final_mode: str = "overlap",
                  max_windows: Optional[int] = None, cap_policy: str = "uniform") -> List[int]:
    """Start rows of the windows covering ``n`` samples.

    ``final_mode``: ``pad`` keeps a trailing partial window (padded later), ``overlap``
    aligns the last window with the end of the sequence, ``drop`` omits partial windows
    (a too-short sequence still yields one padded window).
    """
    if n <= window:
        return [0]
    starts = list(range(0, n - window + 1, max(1, stride)))
    last_full = starts[-1]
    if last_full + window < n:
        if final_mode == "pad":
            starts.append(starts[-1] + max(1, stride))
        elif final_mode == "overlap":
            starts.append(n - window)
        elif final_mode != "drop":
            raise InvalidParams("final_window_mode must be 'pad', 'overlap' or 'drop'")
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
            raise InvalidParams("window_cap_policy must be 'uniform', 'first' or 'last'")
    return starts


def _structured(value: Any) -> Any:
    """Accept JSON / Python-literal strings for list-like search values."""
    if isinstance(value, str):
        text = value.strip()
        if text[:1] in "[({":
            try:
                return json.loads(text)
            except (ValueError, TypeError):
                try:
                    return ast.literal_eval(text)
                except (ValueError, SyntaxError) as exc:
                    raise InvalidParams(f"cannot parse structured value {value!r}") from exc
    return value


def _pipe_modes(value: Any, allowed: Iterable[str], *, name: str, aliases: Optional[Mapping[str, Iterable[str]]] = None) -> set:
    """Parse ``None`` / ``'a|b'`` categorical mode strings."""
    if value is None:
        return set()
    if not isinstance(value, str):
        raise InvalidParams(f"{name} must be None or a '|'-separated mode string")
    if aliases and value in aliases:
        return set(aliases[value])
    allowed = set(allowed)
    parts = {p.strip() for p in value.split("|") if p.strip()}
    if not parts or not parts <= allowed:
        raise InvalidParams(f"Invalid {name}: {value!r}; expected a subset of {sorted(allowed)} or None")
    return parts


def angular_velocity_body(quaternions: np.ndarray, fs: float, valid: Optional[np.ndarray] = None) -> np.ndarray:
    """Body-frame angular velocity (rad/s) from consecutive ``(w, x, y, z)`` quaternions.

    Row ``t`` holds the rotation from ``t-1`` to ``t`` divided by ``1/fs``; row 0 and any
    row whose quaternion (or its predecessor) is invalid is NaN.
    """
    q = np.asarray(quaternions, dtype=float)
    n = len(q)
    out = np.full((n, 3), np.nan)
    if n < 2:
        return out
    ok = np.isfinite(q).all(axis=1) & (np.linalg.norm(np.nan_to_num(q), axis=1) > 0)
    if valid is not None:
        ok &= np.asarray(valid, dtype=bool)
    pairs = ok[1:] & ok[:-1]
    if not pairs.any():
        return out
    xyzw = q[:, [1, 2, 3, 0]]
    safe = np.where(ok[:, None], xyzw, np.array([0.0, 0.0, 0.0, 1.0]))
    rot = Rotation.from_quat(safe)
    delta = rot[:-1].inv() * rot[1:]
    vec = delta.as_rotvec() * float(fs)
    out[1:][pairs] = vec[pairs]
    return out


def fix_quaternion_continuity(q: np.ndarray) -> np.ndarray:
    """Flip signs so consecutive quaternions stay in the same hemisphere."""
    q = np.array(q, dtype=float, copy=True)
    finite = np.isfinite(q).all(axis=1)
    last = None
    for i in np.flatnonzero(finite):
        if last is not None and float(q[i] @ q[last]) < 0:
            q[i] = -q[i]
        last = i
    return q


def fill_missing_time(x: np.ndarray, fill_value: float = 0.0):
    """Linear interpolation inside a sequence, nearest at the edges; all-NaN -> ``fill_value``.

    Returns ``(filled, observed)`` where ``observed`` is the original finiteness mask.
    """
    x = np.asarray(x, dtype=float)
    observed = np.isfinite(x)
    out = x.copy()
    idx = np.arange(len(x))
    for c in range(x.shape[1]):
        ok = observed[:, c]
        if ok.all():
            continue
        if not ok.any():
            out[:, c] = fill_value
        else:
            out[:, c] = np.interp(idx, idx[ok], x[ok, c])
    return out, observed


# ---------------------------------------------------------------------------
# 3. NOOP descriptors (retained) + cr7_kimmitch analytics port
# ---------------------------------------------------------------------------
# The legacy modes/feature names are unchanged: acc_ac_rms, axis_rms, jerk_rms,
# rotation_mean, rotation_rms | peak_fraction, peak_neighbourhood_fraction,
# band_fractions, entropy | acf, acf_harmonic_checked | quality report/features.
# Additions inspired directly by the cr7_kimmitch (NOOP) repository:
#   - ImuFeatureExtractor: ``gyro_energy``, ``acf_cadence``
#   - PpgHr: ``acf_kimmitch`` (fundamental-period preference >= 0.85*peak + sub-lag parabolic fit)
#   - WakeMotionRefinement: ``posture_var`` (3D acceleration covariance trace)
#   - WorkoutDetector / WorkoutTypeClassifier / StepsEstimateEngine: ``motion_cv``
#     (3D vector-delta motion intensity mean, variance, and burstiness CV)
#   - SleepStagerV2 / SleepStager: ``jerk_floor`` (self-calibrated quiescent median jerk floor,
#     move/still duty-cycle fractions, and scale-free peak jerk gate ratio), ``dog_var``
#     (Walch multi-scale Difference-of-Gaussians variability), and spectral ``peakedness``
#   - Whoop5RawImu: ``acc_gyro_coupling`` (cross-modal accel-gyro correlation & energy ratio)
#   - StressIndex: ``baevsky_si`` (Baevsky histogram mode amplitude AMo%, range MxDMn, and SI)
#   - SkinTempDisplay / OpticalExperimentAnalysis: ``power_branches`` (thermal/ToF drift slope,
#     baseline deviation RMS, step-delta CV, and rate-adapted spectral descriptors at 20 Hz)

_LEGACY_NOOP_MOTION_ATOMS = (
    "acc_ac_rms", "axis_rms", "jerk_rms", "rotation_mean", "rotation_rms", "gyro_energy",
)
_KIMITCH_MOTION_ATOMS = (
    "posture_var", "motion_cv", "jerk_floor", "acc_gyro_coupling", "dog_var", "baevsky_si",
)
NOOP_MOTION_ATOMS = _LEGACY_NOOP_MOTION_ATOMS + _KIMITCH_MOTION_ATOMS
NOOP_SPECTRAL_ATOMS = (
    "peak_fraction", "peak_neighbourhood_fraction", "band_fractions", "entropy", "peakedness",
)
NOOP_PERIODICITY_MODES = (None, "acf", "acf_harmonic_checked", "acf_cadence", "acf_kimmitch")
NOOP_QUALITY_MODES = (None, "report", "features")
NOOP_SIGNALS = ("raw", "linear", "angular_speed")
NOOP_POWER_BRANCHES = ("imu", "thm", "tof")
_NOOP_MODE_PARAMETERS = ("motion_stats", "periodicity", "spectral_stats", "quality")
_NOOP_TUNING_PARAMETERS = (
    "signal", "band", "bands", "min_cycles", "min_samples", "max_gap_samples",
    "acf_harmonic_tolerance", "spectral_neighbourhood_hz", "detrend",
    "cadence_band", "min_cadence_strength", "power_branches", "nyquist_policy",
)


def noop_subset_modes(atoms: Sequence[str], aliases: Sequence[str] = (), max_combo_size: int = 6) -> List[Optional[str]]:
    """``[None, *aliases, '|'-joined subsets]`` -- the NOOP mode menu."""
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


NOOP_MOTION_MODES = noop_subset_modes(NOOP_MOTION_ATOMS, aliases=("basic", "kimmitch", "full"), max_combo_size=4)
NOOP_SPECTRAL_MODES = noop_subset_modes(NOOP_SPECTRAL_ATOMS, aliases=("all",), max_combo_size=5)


class NoopOptions(BaseEstimator):
    """Nested sklearn parameters; actual ``None`` is the canonical *off* value.

    ``signal`` selects the signed signal used for periodicity/spectral features.
    Motion statistics always describe raw acceleration (in its declared input units)
    or quaternion-derived angular speed (rad/s).  ``quality`` never removes windows.
    ``nyquist_policy='clip'`` adapts requested bands to each sensor's Nyquist
    frequency; ``'error'`` reproduces the legacy hard failure.
    """

    def __init__(
        self, motion_stats=None, periodicity=None, spectral_stats=None, quality=None,
        signal="raw", band=(0.5, 8.0), bands=((0.5, 2.0), (2.0, 4.0), (4.0, 8.0)),
        min_cycles=2.0, min_samples=16, max_gap_samples=0,
        acf_harmonic_tolerance=0.1, spectral_neighbourhood_hz=0.5, detrend="constant",
        cadence_band=(1.2, 3.5), min_cadence_strength=0.2, power_branches="imu",
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

    # -- mode parsing ------------------------------------------------------
    def mode_names(self):
        motion = _pipe_modes(
            self.motion_stats,
            NOOP_MOTION_ATOMS,
            name="noop__motion_stats",
            aliases={
                "basic": {"acc_ac_rms", "rotation_mean"},
                "kimmitch": {
                    "acc_ac_rms", "jerk_rms", "gyro_energy",
                    "posture_var", "motion_cv", "jerk_floor",
                    "acc_gyro_coupling", "dog_var", "baevsky_si",
                },
                "full": set(NOOP_MOTION_ATOMS),
            },
        )
        spectral = _pipe_modes(
            self.spectral_stats,
            NOOP_SPECTRAL_ATOMS,
            name="noop__spectral_stats",
            aliases={"all": set(NOOP_SPECTRAL_ATOMS)},
        )
        if self.periodicity not in NOOP_PERIODICITY_MODES:
            raise InvalidParams(f"noop__periodicity must be one of {NOOP_PERIODICITY_MODES}")
        if self.quality not in NOOP_QUALITY_MODES:
            raise InvalidParams(f"noop__quality must be one of {NOOP_QUALITY_MODES}")
        return motion, spectral

    def branch_names(self):
        return _pipe_modes(self.power_branches, NOOP_POWER_BRANCHES, name="noop__power_branches") or {"imu"}

    def enabled(self):
        return any(getattr(self, name) is not None for name in _NOOP_MODE_PARAMETERS)

    def validate(self, sampling_rate=None):
        motion, spectral = self.mode_names()
        if not self.enabled():
            return self
        signal_used = self.periodicity is not None or bool(spectral) or self.quality is not None
        if signal_used and self.signal not in NOOP_SIGNALS:
            raise InvalidParams("noop__signal must be raw, linear or angular_speed")
        if self.nyquist_policy not in {"clip", "error"}:
            raise InvalidParams("noop__nyquist_policy must be 'clip' or 'error'")
        self.branch_names()
        inactive = inactive_noop_parameters(self)
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
            lo, hi = self._bounds(self.cadence_band, None, "cadence_band")
            if lo <= 0:
                raise InvalidParams("noop__cadence_band needs a strictly positive lower bound")
            if not 0 <= float(self.min_cadence_strength) <= 1:
                raise InvalidParams("noop__min_cadence_strength must be in [0, 1]")
        if "peak_neighbourhood_fraction" in spectral:
            if not np.isfinite(self.spectral_neighbourhood_hz) or self.spectral_neighbourhood_hz <= 0:
                raise InvalidParams("noop__spectral_neighbourhood_hz must be finite and positive")
        return self

    def _bounds(self, value, fs, label="band"):
        value = _structured(value)
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise InvalidParams(f"NOOP {label} must contain (low, high) bounds")
        lo, hi = map(float, value)
        if not np.isfinite([lo, hi]).all() or not 0 <= lo < hi:
            raise InvalidParams(f"NOOP {label} must satisfy 0 <= low < high")
        if fs is not None and hi > fs / 2:
            if self.nyquist_policy == "error":
                raise InvalidParams(f"NOOP {label} must satisfy high <= Nyquist ({fs / 2:g} Hz)")
            hi = fs / 2  # adapt to this sensor's sample rate
            lo = min(lo, hi)
        return lo, hi

    def frequency_ranges(self, sampling_rate=None):
        fs = None if sampling_rate is None else positive_rate(sampling_rate, "NOOP sampling_rate")
        band = self._bounds(self.band, fs)
        if band[0] <= 0 and self.periodicity in {"acf", "acf_harmonic_checked"}:
            raise InvalidParams("Autocorrelation requires a strictly positive lower frequency")
        _, spectral = self.mode_names()
        bands = tuple(self._bounds(b, fs) for b in _structured(self.bands)) if "band_fractions" in spectral else ()
        return band, bands


def _noop_options_from(settings) -> NoopOptions:
    if isinstance(settings, NoopOptions) or type(settings).__name__ == "NoopOptions":
        return settings
    if settings is None:
        return NoopOptions()
    raise InvalidParams("noop must be None or NoopOptions")


def inactive_noop_parameters(settings) -> set:
    """Tuning parameters that cannot affect the output for the enabled modes."""
    options = _noop_options_from(settings)
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
    return {f"noop__{name}" for name in _NOOP_TUNING_PARAMETERS if name not in active}


def acf_estimate(values, fs, options: NoopOptions):
    """Strongest autocorrelation peak of the dominant axis inside ``options.band``."""
    x = detrend_values(values, options.detrend)
    axis = int(np.argmax(np.sum(x * x, axis=0)))
    x = x[:, axis]
    energy = float(x @ x)
    if energy <= np.finfo(float).eps * max(1, len(x)):
        return 0.0, 0.0, False
    (low, high), _ = options.frequency_ranges(fs)
    if low <= 0:
        return 0.0, 0.0, False
    first = max(1, int(np.ceil(fs / high)))
    last = min(len(x) - 2, int(np.floor(fs / low)), int(np.floor(len(x) / options.min_cycles)))
    if last < first:
        return 0.0, 0.0, False
    corr = np.correlate(x, x, mode="full")[len(x) - 1:] / energy
    peaks = [lag for lag in range(first, last + 1)
             if corr[lag] > corr[lag - 1] and corr[lag] >= corr[lag + 1] and corr[lag] > 0]
    if not peaks:
        return 0.0, 0.0, False
    lag = max(peaks, key=lambda pos: corr[pos])
    return fs / lag, float(np.clip(corr[lag], 0, 1)), True


def cadence_estimate(magnitude, fs, band=(1.2, 3.5), min_strength=0.2):
    """cr7_kimmitch ``ImuFeatureExtractor`` gait cadence.

    Normalised autocorrelation of the magnitude AC (mean removed) over the lags of
    ``band``.  Strength = ACF[lag] / ACF[0] (amplitude-scale free); the strongest lag
    gives the cadence, reported only when the strength reaches ``min_strength``.
    Returns ``(cadence_hz, strength, usable)``; ``cadence_hz == 0`` means "no rhythm".
    """
    ac = np.asarray(magnitude, dtype=float).reshape(-1)
    n = len(ac)
    if n < 8:
        return 0.0, 0.0, False
    ac = ac - ac.mean()
    ac0 = float(ac @ ac)
    if ac0 <= np.finfo(float).eps * n:
        return 0.0, 0.0, True  # a still sensor is a valid "no cadence" observation
    low, high = float(band[0]), min(float(band[1]), fs / 2.0)
    lo_lag = max(1, int(round(fs / high)))
    hi_lag = min(n - 1, int(round(fs / low)))
    if lo_lag >= hi_lag:
        return 0.0, 0.0, False
    corr = np.correlate(ac, ac, mode="full")[n - 1:]
    lags = np.arange(lo_lag, hi_lag + 1)
    strengths = corr[lags] / ac0
    k = int(np.argmax(strengths))
    best = float(strengths[k])
    cadence = fs / float(lags[k]) if best >= min_strength and best > 0 else 0.0
    return cadence, max(0.0, best), True


def kimmitch_acf_estimate(
    magnitude,
    fs: float,
    band: Tuple[float, float] = (1.2, 3.5),
    min_strength: float = 0.2,
    fundamental_ratio: float = 0.85,
) -> Tuple[float, float, float, float, bool]:
    """cr7_kimmitch ``PpgHr`` + ``ImuFeatureExtractor`` fundamental-preferred sub-lag ACF.

    1. Linearly detrends the signal to suppress slow posture/thermal drift.
    2. Computes normalized autocorrelation over ``band``.
    3. Prefers the **fundamental period** (smallest local-peak lag whose correlation is at
       least ``fundamental_ratio`` of the strongest peak in-band) so 2x/3x subharmonic lags
       do not steal the fundamental frequency.
    4. Refines the lag to sub-sample precision via 3-point parabolic interpolation
       ``delta = 0.5 * (y0 - y2) / (y0 - 2*y1 + y2)`` and measures peak concavity
       ``max(0, 2*y1 - y0 - y2)``.
    Returns ``(cadence_hz, sublag_hz, strength, concavity, usable)``.
    """
    raw = np.asarray(magnitude, dtype=float).reshape(-1)
    n = len(raw)
    if n < 8:
        return 0.0, 0.0, 0.0, 0.0, False
    ac = detrend_values(raw[:, None], "linear")[:, 0]
    ac0 = float(ac @ ac)
    if ac0 <= np.finfo(float).eps * n:
        return 0.0, 0.0, 0.0, 0.0, True
    low, high = float(band[0]), min(float(band[1]), fs / 2.0)
    lo_lag = max(1, int(round(fs / high)))
    hi_lag = min(n - 2, int(round(fs / low)))
    if lo_lag >= hi_lag:
        return 0.0, 0.0, 0.0, 0.0, False
    corr = np.correlate(ac, ac, mode="full")[n - 1:] / ac0
    peaks = [
        lag for lag in range(lo_lag, hi_lag + 1)
        if corr[lag] > corr[lag - 1] and corr[lag] >= corr[lag + 1] and corr[lag] > 0
    ]
    if peaks:
        max_peak = max(float(corr[p]) for p in peaks)
        threshold = float(fundamental_ratio) * max_peak
        chosen_lag = min(p for p in peaks if float(corr[p]) >= threshold)
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


def _gaussian_smooth_1d(x: np.ndarray, sigma_samples: float) -> np.ndarray:
    """1D Gaussian smoothing with edge reflection (cr7_kimmitch SleepStager DoG helper)."""
    x = np.asarray(x, dtype=float).reshape(-1)
    if len(x) < 3 or sigma_samples <= 0.25:
        return x.copy()
    radius = max(1, min(len(x) // 2, int(round(3.0 * float(sigma_samples)))))
    grid = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (grid / float(sigma_samples)) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(x, pad_width=radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _dog_variability_1d(x: np.ndarray, fs: float) -> float:
    """cr7_kimmitch ``SleepStager`` Walch Difference-of-Gaussians multi-scale variability."""
    x = np.asarray(x, dtype=float).reshape(-1)
    if len(x) < 4:
        return 0.0
    short_sigma = max(1.0, 0.12 * float(fs))
    long_sigma = max(short_sigma + 1.0, 0.65 * float(fs))
    diff = _gaussian_smooth_1d(x, short_sigma) - _gaussian_smooth_1d(x, long_sigma)
    return float(np.sqrt(np.mean(diff ** 2)))


def _baevsky_stress_metrics(x: np.ndarray, n_bins: int = 16) -> Tuple[float, float, float]:
    """cr7_kimmitch ``StressIndex`` Baevsky histogram Mode Amplitude (AMo%), Range (MxDMn), and SI."""
    x = np.asarray(x, dtype=float).reshape(-1)
    if len(x) < 4:
        return 0.0, 0.0, 0.0
    p2, p98 = np.percentile(x, [2.0, 98.0])
    mxdmn = float(max(0.0, p98 - p2))
    if mxdmn <= 1e-9:
        return 100.0, 0.0, 0.0
    clipped = x[(x >= p2) & (x <= p98)]
    if len(clipped) < 2:
        clipped = x
    counts, edges = np.histogram(clipped, bins=max(4, int(n_bins)), range=(float(p2), float(p98)))
    peak_bin = int(np.argmax(counts))
    amo_pct = 100.0 * float(counts[peak_bin]) / float(len(clipped))
    mo_center = 0.5 * float(edges[peak_bin] + edges[peak_bin + 1])
    scale_mo = max(abs(mo_center), float(np.std(clipped)), 1e-3)
    si = amo_pct / (2.0 * scale_mo * max(mxdmn, 1e-3))
    return float(amo_pct), float(mxdmn), float(si)


def _adder(result: dict):
    def add(name, value, supported):
        ok = bool(supported) and bool(np.isfinite(value))
        result[f"noop_{name}"] = float(value) if ok else 0.0
        result[f"noop_{name}_supported"] = float(ok)
    return add


def _spectral_block(add, selected, usable, fs, options: NoopOptions, spectral, prefix=""):
    """Peak fraction / neighbourhood / band fractions / entropy / peakedness for one signal at rate ``fs``."""
    band, bands = options.frequency_ranges(fs)
    if usable:
        frequencies, density = window_psd(selected, fs, options.detrend)
    else:
        frequencies, density = np.empty(0), np.empty(0)
    resolved_low = max(band[0], options.min_cycles * fs / max(1, len(selected)))
    in_band = (frequencies >= resolved_low) & (frequencies <= band[1])
    power = density[in_band]
    total = float(power.sum())
    supported = usable and len(power) >= 2 and total > np.finfo(float).tiny
    peak = int(np.argmax(power)) if supported else 0
    peak_frequency = float(frequencies[in_band][peak]) if supported else 0
    if "peak_fraction" in spectral:
        add(f"{prefix}spectral_peak_fraction", power[peak] / total if supported else 0, supported)
    if "peak_neighbourhood_fraction" in spectral:
        near = in_band & (np.abs(frequencies - peak_frequency) <= options.spectral_neighbourhood_hz)
        add(f"{prefix}spectral_peak_neighbourhood_fraction", density[near].sum() / total if supported else 0, supported)
    if "band_fractions" in spectral:
        for idx, (low, high) in enumerate(bands):
            inside = in_band & (frequencies >= low) & (frequencies < high)
            add(f"{prefix}spectral_band_{idx}_fraction", density[inside].sum() / total if supported else 0,
                supported and inside.any())
    if "entropy" in spectral:
        p = power[power > 0] / total if supported else np.empty(0)
        entropy = -np.sum(p * np.log2(p)) / np.log2(len(power)) if supported else 0
        add(f"{prefix}spectral_entropy_normalized", entropy, supported)
    if "peakedness" in spectral:
        p_norm = power / total if supported else np.empty(0)
        peakedness = float(np.sum(p_norm ** 2)) if supported else 0.0
        add(f"{prefix}spectral_peakedness", peakedness, supported)


def noop_imu_window(src: Mapping[str, np.ndarray], fs: float, options: NoopOptions):
    """NOOP summary of one IMU window at sample rate ``fs``.

    ``src`` keys: ``raw`` (T,3) acceleration, ``observed`` (T,3) bool, ``linear`` (T,3),
    ``rotation`` (T,3) body angular velocity in rad/s, ``rotation_observed`` (T,) bool
    (quaternion valid), ``counter`` (T,) row counter, ``dt`` (T,) seconds per row,
    optional ``row_present`` (T,) bool for the counter-gap quality figures.
    Returns ``(features, report)``.  Evaluates legacy NOOP + cr7_kimmitch descriptors at ``fs``.
    """
    motion, spectral = options.mode_names()
    min_samples = options.min_samples if motion or options.periodicity is not None or spectral else 16
    max_gap = options.max_gap_samples if options.periodicity is not None or spectral or options.quality is not None else 0
    counter = np.asarray(src["counter"], dtype=float)
    raw = np.asarray(src["raw"], dtype=float)
    observed_axes = np.asarray(src["observed"], dtype=bool)
    raw_observed = observed_axes.all(axis=1) & np.isfinite(raw).all(axis=1)
    q_observed = np.asarray(src["rotation_observed"], dtype=bool)
    contiguous = np.r_[False, np.diff(counter) == 1]
    rotation = np.asarray(src["rotation"], dtype=float)
    rotation_valid = (q_observed & np.r_[False, q_observed[:-1]] & contiguous & np.isfinite(rotation).all(axis=1))
    result: Dict[str, float] = {}
    add = _adder(result)

    def enough(valid):
        return int(np.sum(valid)) >= min_samples

    obs_raw = raw[raw_observed]
    if "acc_ac_rms" in motion:
        magnitude = np.linalg.norm(obs_raw, axis=1)
        add("acc_ac_rms", np.std(magnitude) if len(magnitude) else 0, enough(raw_observed))
    if "axis_rms" in motion:
        for axis, name in enumerate("xyz"):
            valid = observed_axes[:, axis] & np.isfinite(raw[:, axis])
            add(f"acc_{name}_rms", np.sqrt(np.mean(raw[valid, axis] ** 2)) if valid.any() else 0, enough(valid))
    if "jerk_rms" in motion:
        valid = raw_observed[1:] & raw_observed[:-1] & contiguous[1:]
        dt = np.asarray(src["dt"], dtype=float)[1:]
        valid &= np.isfinite(dt) & (dt > 0)
        jerk = np.diff(raw, axis=0)[valid] / dt[valid, None]
        add("jerk_rms", np.sqrt(np.mean(np.sum(jerk ** 2, axis=1))) if len(jerk) else 0, int(valid.sum()) >= min_samples - 1)
    speed = np.linalg.norm(rotation[rotation_valid], axis=1)
    if "rotation_mean" in motion:
        add("rotation_mean_rad_s", np.mean(speed) if len(speed) else 0, enough(rotation_valid))
    if "rotation_rms" in motion:
        add("rotation_rms_rad_s", np.sqrt(np.mean(speed ** 2)) if len(speed) else 0, enough(rotation_valid))
    if "gyro_energy" in motion:  # cr7_kimmitch: mean gyroscope magnitude, deg/s
        add("gyro_energy_dps", np.degrees(np.mean(speed)) if len(speed) else 0, enough(rotation_valid))
    if "posture_var" in motion:  # cr7_kimmitch WakeMotionRefinement: 3D covariance trace
        p_var = float(np.sum(np.var(obs_raw, axis=0))) if len(obs_raw) else 0.0
        add("posture_var", p_var, enough(raw_observed))
    if "motion_cv" in motion:  # cr7_kimmitch WorkoutDetector / WorkoutTypeClassifier: 3D vector-delta stats & CV
        pair_valid = raw_observed[1:] & raw_observed[:-1] & contiguous[1:]
        ok_pairs = int(pair_valid.sum()) >= max(2, min_samples - 1)
        deltas = np.linalg.norm(np.diff(raw, axis=0)[pair_valid], axis=1) if len(raw) > 1 and pair_valid.any() else np.empty(0)
        mean_i = float(np.mean(deltas)) if len(deltas) else 0.0
        var_i = float(np.var(deltas)) if len(deltas) else 0.0
        cv_i = float(np.sqrt(var_i) / mean_i) if mean_i > 1e-6 else 0.0
        add("motion_intensity_mean", mean_i, ok_pairs)
        add("motion_variance", var_i, ok_pairs)
        add("motion_cv", cv_i, ok_pairs)
    if "jerk_floor" in motion:  # cr7_kimmitch SleepStagerV2: quiescent jerk floor, move/still duty-cycle & gate ratio
        pair_valid = raw_observed[1:] & raw_observed[:-1] & contiguous[1:]
        dt = np.asarray(src["dt"], dtype=float)[1:]
        pair_valid &= np.isfinite(dt) & (dt > 0)
        ok_jerk = int(pair_valid.sum()) >= max(2, min_samples - 1)
        if len(raw) > 1 and pair_valid.any():
            j_mag = np.linalg.norm(np.diff(raw, axis=0)[pair_valid] / dt[pair_valid, None], axis=1)
            j_med = float(np.median(j_mag))
            j_scale = max(j_med, 1e-4)
            move_frac = float(np.mean(j_mag > 2.0 * j_scale))
            still_frac = float(np.mean(j_mag <= 1.25 * j_scale))
            gate_ratio = float(np.max(j_mag) / j_scale)
        else:
            j_med = move_frac = still_frac = gate_ratio = 0.0
        add("jerk_floor", j_med, ok_jerk)
        add("move_fraction", move_frac, ok_jerk)
        add("still_fraction", still_frac, ok_jerk)
        add("jerk_gate_ratio", gate_ratio, ok_jerk)
    if "acc_gyro_coupling" in motion:  # cr7_kimmitch Whoop5RawImu: cross-modal accel-gyro correlation & energy ratio
        joint = raw_observed & rotation_valid
        ok_joint = enough(joint)
        if ok_joint:
            acc_m = np.linalg.norm(raw[joint], axis=1)
            acc_ac = np.abs(acc_m - np.mean(acc_m))
            rot_m = np.linalg.norm(rotation[joint], axis=1)
            std_a, std_g = float(np.std(acc_ac)), float(np.std(rot_m))
            corr_ag = float(np.clip(np.corrcoef(acc_ac, rot_m)[0, 1], -1.0, 1.0)) if (std_a > 1e-6 and std_g > 1e-6) else 0.0
            ratio_ag = float(np.sqrt(np.mean(rot_m ** 2)) / (float(np.std(acc_m)) + 1e-4))
        else:
            corr_ag = ratio_ag = 0.0
        add("acc_gyro_corr", corr_ag, ok_joint)
        add("gyro_to_acc_ratio", ratio_ag, ok_joint)
    if "dog_var" in motion:  # cr7_kimmitch SleepStager: Walch Difference-of-Gaussians multi-scale variability
        ok_dog = enough(raw_observed)
        dog_val = _dog_variability_1d(np.linalg.norm(obs_raw, axis=1), fs) if ok_dog else 0.0
        add("dog_variability", dog_val, ok_dog)
    if "baevsky_si" in motion:  # cr7_kimmitch StressIndex: Baevsky AMo%, MxDMn, and centralization SI
        ok_b = enough(raw_observed)
        amo_pct, mxdmn, si_val = _baevsky_stress_metrics(np.linalg.norm(obs_raw, axis=1)) if ok_b else (0.0, 0.0, 0.0)
        add("baevsky_amo_pct", amo_pct, ok_b)
        add("baevsky_mxdmn", mxdmn, ok_b)
        add("baevsky_si", si_val, ok_b)

    if options.signal == "raw":
        signal, observed = raw, raw_observed
    elif options.signal == "linear":
        signal = np.asarray(src["linear"], dtype=float)
        observed = raw_observed & q_observed & np.isfinite(signal).all(axis=1)
    else:
        signal = np.linalg.norm(rotation, axis=1)[:, None]
        observed = rotation_valid
    signal, valid, imputed = bounded_fill(signal, observed, counter, max_gap)
    runs = contiguous_runs(valid, counter)
    longest = max(runs, key=len) if runs else np.empty(0, dtype=int)
    selected = signal[longest]
    usable = len(selected) >= min_samples
    if options.periodicity in {"acf", "acf_harmonic_checked"}:
        frequency, strength, supported = acf_estimate(selected, fs, options) if usable else (0, 0, False)
        add("acf_frequency_hz", frequency, supported)
        add("acf_strength", strength, supported)
        if options.periodicity == "acf_harmonic_checked":
            checked = supported and selected.shape[1] == 3
            magnitude_frequency, _, magnitude_supported = (
                acf_estimate(np.linalg.norm(selected, axis=1)[:, None], fs, options) if checked else (0, 0, False))
            ambiguous = (magnitude_supported and abs(magnitude_frequency / frequency - 2) <= 2 * options.acf_harmonic_tolerance)
            add("acf_magnitude_harmonic", float(ambiguous), checked and magnitude_supported)
    elif options.periodicity == "acf_cadence":
        magnitude = np.linalg.norm(selected, axis=1) if len(selected) else np.empty(0)
        hz, strength, supported = cadence_estimate(
            magnitude, fs, _structured(options.cadence_band), float(options.min_cadence_strength)
        ) if usable else (0.0, 0.0, False)
        add("cadence_hz", hz, supported and hz > 0)
        add("cadence_strength", strength, supported)
    elif options.periodicity == "acf_kimmitch":
        magnitude = np.linalg.norm(selected, axis=1) if len(selected) else np.empty(0)
        hz, sublag_hz, strength, concavity, supported = kimmitch_acf_estimate(
            magnitude, fs, _structured(options.cadence_band), float(options.min_cadence_strength)
        ) if usable else (0.0, 0.0, 0.0, 0.0, False)
        add("cadence_hz", hz, supported and hz > 0)
        add("cadence_sublag_hz", sublag_hz, supported and sublag_hz > 0)
        add("cadence_strength", strength, supported)
        add("acf_sublag_concavity", concavity, supported)
    if spectral and "imu" in options.branch_names():
        _spectral_block(add, selected, usable, fs, options, spectral)
    present = np.asarray(src.get("row_present", np.ones(len(counter), dtype=bool)), dtype=bool)
    gaps = int(np.sum(np.diff(present.astype(int)) == -1) + (0 if not len(present) or present[0] else 1))
    report = {
        "real_samples": int(len(counter)), "observed_samples": int(observed.sum()),
        "observed_fraction": float(observed.mean()) if len(counter) else 0.0,
        "imputed_samples": int(imputed.sum()), "longest_contiguous_samples": int(len(longest)),
        "counter_gap_count": gaps, "source": options.signal,
    }
    if options.quality == "features":
        add("observed_fraction", report["observed_fraction"], len(counter) > 0)
        add("imputed_fraction", imputed.sum() / max(1, len(counter)), len(counter) > 0)
        add("longest_contiguous_fraction", len(longest) / max(1, len(counter)), len(counter) > 0)
        add("counter_gap_fraction", float((~present).sum()) / max(1, len(counter)), len(counter) > 0)
    return result, report


def noop_power_window(values: np.ndarray, observed: np.ndarray, fs: float, options: NoopOptions, prefix: str):
    """Rate-adapted NOOP descriptors for a non-IMU branch (``thm`` / ``tof``) at its own rate ``fs``.

    ``values`` is the branch's pooled signal ``(T, C)``; ``observed`` marks real samples.
    When ``motion_stats`` is active, computes ``cr7_kimmitch`` thermal/optical drift slope,
    baseline deviation RMS, and step-delta burstiness CV at ``fs``.  When ``spectral_stats``
    is active, computes the rate-adapted spectral descriptors at ``fs``.
    """
    motion, spectral = options.mode_names()
    result: Dict[str, float] = {}
    if not motion and not spectral:
        return result
    add = _adder(result)
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    observed = np.asarray(observed, dtype=bool) & np.isfinite(values).all(axis=1)
    counter = np.arange(len(values), dtype=float)
    signal, valid, _ = bounded_fill(values, observed, counter, options.max_gap_samples)
    runs = contiguous_runs(valid, counter)
    longest = max(runs, key=len) if runs else np.empty(0, dtype=int)
    selected = signal[longest]
    usable = len(selected) >= max(4, min(int(options.min_samples), int(round(0.5 * fs))))
    if motion:
        series = selected[:, 0] if len(selected) else np.empty(0)
        if usable and len(series) >= 2:
            t_sec = np.arange(len(series), dtype=float) / float(fs)
            t_centered = t_sec - t_sec.mean()
            s_centered = series - series.mean()
            denom = float(t_centered @ t_centered)
            slope = float((t_centered @ s_centered) / denom) if denom > 1e-12 else 0.0
            dev_rms = float(np.std(series))
            diffs = np.abs(np.diff(series))
            mean_d = float(np.mean(diffs)) if len(diffs) else 0.0
            cv_d = float(np.std(diffs) / mean_d) if mean_d > 1e-6 else 0.0
        else:
            slope = dev_rms = cv_d = 0.0
        add(f"{prefix}_drift_slope", slope, usable)
        add(f"{prefix}_dev_rms", dev_rms, usable)
        add(f"{prefix}_motion_cv", cv_d, usable)
    if spectral:
        _spectral_block(add, selected, usable, fs, options, spectral, prefix=f"{prefix}_")
    return result


# ---------------------------------------------------------------------------
# 4. Sequence extractor (per-modality chunks, sample-rate adaptive)
# ---------------------------------------------------------------------------

STATIC_SOURCES = ("acc_mag", "lin_mag", "ang_speed", "thm_mean", "tof_mean")
TSFEL_DOMAINS = ("statistical", "temporal", "spectral")
# Compact default set: cheap enough to run for every candidate, still covers the
# time-domain and spectral basics.  Any tsfel feature name can be requested instead.
TSFEL_DEFAULT_FEATURES = (
    "Mean", "Standard deviation", "Skewness", "Kurtosis", "Interquartile range", "Max", "Min",
    "Zero crossing rate", "Mean absolute diff", "Entropy", "Slope", "Peak to peak distance",
    "Max power spectrum", "Spectral centroid", "Spectral entropy", "Fundamental frequency",
)
TSFRESH_PROFILES = ("minimal", "efficient")


class MultiModalSequenceExtractor(BaseEstimator):
    """Turn raw CMI sensor rows into per-modality window tensors + static features.

    Every sensor keeps its **own sample rate**.  ``window_seconds`` is a duration, so a
    2.5 s window is 250 samples of IMU at 100 Hz but 50 samples of ToF/thermopile at
    20 Hz; band edges, tsfel ``fs``, tsfresh FFT coefficients and NOOP spectra are all
    derived from the rate of the array they describe.

    ``align_mode='time'``: rows share the ``row_sampling_rate`` clock and each sensor is
    anti-alias resampled to its target rate.  ``'rows'``: one row is one native sample of
    every sensor (legacy behaviour); only target-rate resampling is applied.

    ``transform_chunks`` returns a dict with ``branches`` (``imu``/``thm``/``tof`` float32
    ``(N, T_m, C_m)``), ``masks``, ``present`` ``(N, 3)``, ``static`` / ``static_names``,
    ``noop_features`` / ``noop_feature_names``, ``sequence_ids``, ``window_index``.
    """

    def __init__(
        self, noop=None,
        # --- rates (Hz) ---
        row_sampling_rate=100.0, align_mode="time", resample_method="poly",
        imu_native_sampling_rate=100.0, rot_native_sampling_rate=100.0,
        thm_native_sampling_rate=20.0, tof_native_sampling_rate=20.0,
        imu_target_sampling_rate=None, thm_target_sampling_rate=None, tof_target_sampling_rate=None,
        # --- windows (seconds) ---
        window_seconds=2.5, stride_ratio=0.75, final_window_mode="overlap",
        max_windows_per_sequence=200, window_cap_policy="uniform",
        # --- IMU ---
        acc_modes="raw|linear", use_acc_magnitude=True, use_linear_acc_magnitude=True,
        rotation_modes="quaternion|angular_velocity", fix_quaternion_sign=True,
        linear_acc_mode="baseline", use_highpass_fallback=True, highpass_hz=0.3,
        # --- thermopile / ToF ---
        thm_modes="centered_diff", tof_modes="pooled_stats|sensor_stats", tof_fill_mode="far_500",
        # --- static (tsfel / tsfresh) ---
        static_backend="tsfel", static_channels="acc_mag|ang_speed|thm_mean|tof_mean",
        tsfel_domains="statistical|temporal|spectral", tsfel_features=None,
        tsfresh_profile="minimal", tsfresh_fft_bands=((0.5, 2.0), (2.0, 4.0), (4.0, 8.0)),
        static_clip=10.0, sequence_col=SEQ_COL, counter_col=COUNTER_COL,
    ):
        self.noop = noop
        self.row_sampling_rate = row_sampling_rate
        self.align_mode = align_mode
        self.resample_method = resample_method
        self.imu_native_sampling_rate = imu_native_sampling_rate
        self.rot_native_sampling_rate = rot_native_sampling_rate
        self.thm_native_sampling_rate = thm_native_sampling_rate
        self.tof_native_sampling_rate = tof_native_sampling_rate
        self.imu_target_sampling_rate = imu_target_sampling_rate
        self.thm_target_sampling_rate = thm_target_sampling_rate
        self.tof_target_sampling_rate = tof_target_sampling_rate
        self.window_seconds = window_seconds
        self.stride_ratio = stride_ratio
        self.final_window_mode = final_window_mode
        self.max_windows_per_sequence = max_windows_per_sequence
        self.window_cap_policy = window_cap_policy
        self.acc_modes = acc_modes
        self.use_acc_magnitude = use_acc_magnitude
        self.use_linear_acc_magnitude = use_linear_acc_magnitude
        self.rotation_modes = rotation_modes
        self.fix_quaternion_sign = fix_quaternion_sign
        self.linear_acc_mode = linear_acc_mode
        self.use_highpass_fallback = use_highpass_fallback
        self.highpass_hz = highpass_hz
        self.thm_modes = thm_modes
        self.tof_modes = tof_modes
        self.tof_fill_mode = tof_fill_mode
        self.static_backend = static_backend
        self.static_channels = static_channels
        self.tsfel_domains = tsfel_domains
        self.tsfel_features = tsfel_features
        self.tsfresh_profile = tsfresh_profile
        self.tsfresh_fft_bands = tsfresh_fft_bands
        self.static_clip = static_clip
        self.sequence_col = sequence_col
        self.counter_col = counter_col

    # -- sklearn plumbing: ``noop__*`` parameters configure a NoopOptions --------
    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        if deep and self.noop is None:
            params.update({f"noop__{k}": v for k, v in NoopOptions().get_params(deep=True).items()})
        return params

    def set_params(self, **params):
        if any(name.startswith("noop__") for name in params) and params.get("noop", self.noop) is None:
            params = dict(params)
            params["noop"] = NoopOptions()
        return super().set_params(**params)

    # -- configuration helpers --------------------------------------------------
    def noop_options(self) -> Optional[NoopOptions]:
        if self.noop is None:
            return None
        if not (isinstance(self.noop, NoopOptions) or type(self.noop).__name__ == "NoopOptions"):
            raise InvalidParams("noop must be None or NoopOptions")
        return self.noop if self.noop.enabled() else None

    def branch_rates(self) -> Dict[str, float]:
        """Sample rate (Hz) of each enabled branch array."""
        rates = {"imu": positive_rate(self.imu_target_sampling_rate or self.imu_native_sampling_rate, "imu rate")}
        if self.thm_modes is not None:
            rates["thm"] = positive_rate(self.thm_target_sampling_rate or self.thm_native_sampling_rate, "thm rate")
        if self.tof_modes is not None:
            rates["tof"] = positive_rate(self.tof_target_sampling_rate or self.tof_native_sampling_rate, "tof rate")
        return rates

    def branch_names(self) -> List[str]:
        return list(self.branch_rates())

    def window_lengths(self) -> Dict[str, int]:
        return {name: seconds_to_samples(self.window_seconds, rate) for name, rate in self.branch_rates().items()}

    def _modes(self):
        acc = _pipe_modes(self.acc_modes, {"raw", "linear", "jerk"}, name="acc_modes")
        rot = _pipe_modes(self.rotation_modes, {"quaternion", "angular_velocity", "euler"}, name="rotation_modes")
        thm = _pipe_modes(self.thm_modes, {"raw", "centered", "diff", "centered_diff", "pooled_stats"}, name="thm_modes")
        tof = _pipe_modes(self.tof_modes, {"raw", "pooled_stats", "sensor_stats"}, name="tof_modes")
        static = _pipe_modes(self.static_channels, STATIC_SOURCES, name="static_channels")
        return acc, rot, thm, tof, static

    def validate(self):
        self._modes()
        if self.align_mode not in {"time", "rows"}:
            raise InvalidParams("align_mode must be 'time' or 'rows'")
        if self.static_backend not in {"tsfel", "tsfresh", "none", None}:
            raise InvalidParams("static_backend must be 'tsfel', 'tsfresh' or 'none'")
        if self.tof_fill_mode not in {"far_500", "far_255", "zero", "nan_interpolate"}:
            raise InvalidParams("tof_fill_mode must be 'far_500', 'far_255', 'zero' or 'nan_interpolate'")
        if self.tsfresh_profile not in TSFRESH_PROFILES:
            raise InvalidParams(f"tsfresh_profile must be one of {TSFRESH_PROFILES}")
        if not 0 < float(self.stride_ratio) <= 1:
            raise InvalidParams("stride_ratio must be in (0, 1]")
        if float(self.window_seconds) <= 0:
            raise InvalidParams("window_seconds must be positive")
        rates = self.branch_rates()
        for name in ("imu", "rot", "thm", "tof"):
            positive_rate(getattr(self, f"{name}_native_sampling_rate"), f"{name} native rate")
        row = positive_rate(self.row_sampling_rate, "row_sampling_rate")
        if self.align_mode == "time":
            for name in ("imu", "rot", "thm", "tof"):
                if getattr(self, f"{name}_native_sampling_rate") > row * (1 + 1e-9):
                    raise InvalidParams(f"{name} native rate exceeds the row clock; raise row_sampling_rate")
        for name, rate in rates.items():
            if self.window_lengths()[name] < 4:
                raise InvalidParams(f"window_seconds is too short for the {name} branch at {rate:g} Hz")
        opts = self.noop_options()
        if opts is not None:
            opts.validate(rates["imu"])
            for branch in opts.branch_names() - {"imu"}:
                if branch not in rates:
                    raise InvalidParams(f"noop__power_branches includes disabled branch {branch!r}")
        return self

    def _imu_plan(self):
        acc, rot, _, _, static = self._modes()
        opts = self.noop_options()
        motion, spectral = opts.mode_names() if opts else (set(), set())
        signal_used = bool(opts) and (opts.periodicity is not None or bool(spectral) or opts.quality is not None)
        need_linear = "linear" in acc or (signal_used and opts.signal == "linear") or "lin_mag" in static
        need_ang = ("angular_velocity" in rot or "ang_speed" in static
                    or bool(motion & {"rotation_mean", "rotation_rms", "gyro_energy"})
                    or (signal_used and opts.signal == "angular_speed"))
        names = ["acc_x", "acc_y", "acc_z"]
        if self.use_acc_magnitude:
            names.append("acc_mag")
        if need_linear:
            names += ["lin_x", "lin_y", "lin_z"]
            if self.use_linear_acc_magnitude:
                names.append("lin_mag")
        if "jerk" in acc:
            names += ["jerk_x", "jerk_y", "jerk_z"]
        if "quaternion" in rot:
            names += ["rot_w", "rot_x", "rot_y", "rot_z"]
        if need_ang:
            names += ["ang_x", "ang_y", "ang_z"]
        if "euler" in rot:
            names += ["euler_x", "euler_y", "euler_z"]
        return names, need_linear, need_ang

    def channel_names(self) -> Dict[str, List[str]]:
        _, _, thm, tof, _ = self._modes()
        names = {"imu": self._imu_plan()[0]}
        if self.thm_modes is not None:
            cols = []
            if "raw" in thm:
                cols += [f"thm_raw_{i}" for i in range(1, 6)]
            if "centered" in thm or "centered_diff" in thm:
                cols += [f"thm_centered_{i}" for i in range(1, 6)]
            if "diff" in thm or "centered_diff" in thm:
                cols += [f"thm_diff_{i}" for i in range(1, 6)]
            if "pooled_stats" in thm:
                cols += ["thm_std", "thm_min", "thm_max"]
            names["thm"] = cols + ["thm_mean"]
        if self.tof_modes is not None:
            cols = [f"tof_raw_{i}" for i in range(320)] if "raw" in tof else []
            cols.append("tof_mean")
            if "pooled_stats" in tof:
                cols += ["tof_std", "tof_min", "tof_max", "tof_valid_fraction"]
            if "sensor_stats" in tof:
                cols += [f"tof_sensor_mean_{i}" for i in range(1, 6)] + [f"tof_sensor_std_{i}" for i in range(1, 6)]
            names["tof"] = cols
        return names

    # -- fitting ---------------------------------------------------------------
    def fit(self, X, y=None):
        self.validate()
        chunks = self.featurize(self.raw_chunks(X))
        self.fit_scalers(chunks)
        return self

    def transform_chunks(self, X):
        return self.scale(self.featurize(self.raw_chunks(X)))

    # -- raw windows -----------------------------------------------------------
    def raw_chunks(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Per-modality window tensors from raw rows (no static features, unscaled)."""
        self.validate()
        if self.sequence_col not in X:
            raise ValueError(f"X must contain {self.sequence_col!r}")
        names = self.channel_names()
        rates = self.branch_rates()
        wins = self.window_lengths()
        groups = X.groupby(self.sequence_col, sort=False).indices
        acc = {name: [] for name in rates}
        mask = {name: [] for name in rates}
        extra = {"obs_axes": [], "q_obs": [], "row_present": [], "obs_thm": [], "obs_tof": []}
        present, seq_ids, win_idx = [], [], []
        fs_imu = rates["imu"]
        for seq_id, positions in groups.items():
            prepared = self._prepare_sequence(X.iloc[positions])
            n_imu = len(prepared["imu"])
            starts = window_starts(
                n_imu, wins["imu"], max(1, int(round(wins["imu"] * float(self.stride_ratio)))),
                self.final_window_mode, self.max_windows_per_sequence, self.window_cap_policy)
            for k, s_imu in enumerate(starts):
                t0 = s_imu / fs_imu
                flags = []
                for name in rates:
                    arr = prepared[name]
                    start = int(round(t0 * rates[name]))
                    seg = arr[start:start + wins[name]]
                    out = np.zeros((wins[name], arr.shape[1]), dtype=np.float32)
                    out[:len(seg)] = seg
                    m = np.zeros(wins[name], dtype=bool)
                    m[:len(seg)] = True
                    acc[name].append(out)
                    mask[name].append(m)
                for key, src_key, name in (("obs_axes", "obs_axes", "imu"), ("q_obs", "q_obs", "imu"),
                                           ("row_present", "row_present", "imu")):
                    arr = prepared[src_key]
                    start = int(round(t0 * rates[name]))
                    seg = arr[start:start + wins[name]]
                    out = np.zeros((wins[name],) + arr.shape[1:], dtype=bool)
                    out[:len(seg)] = seg
                    extra[key].append(out)
                for key, name in (("obs_thm", "thm"), ("obs_tof", "tof")):
                    if name in rates:
                        arr = prepared[key]
                        start = int(round(t0 * rates[name]))
                        seg = arr[start:start + wins[name]]
                        out = np.zeros(wins[name], dtype=bool)
                        out[:len(seg)] = seg
                        extra[key].append(out)
                flags = [bool(prepared["present"]["imu"]),
                         bool(prepared["present"]["thm"]) and "thm" in rates,
                         bool(prepared["present"]["tof"]) and "tof" in rates]
                present.append(flags)
                seq_ids.append(seq_id)
                win_idx.append(k)
        chunks = {
            "branches": {n: np.stack(v) for n, v in acc.items()},
            "masks": {n: np.stack(v) for n, v in mask.items()},
            "present": np.asarray(present, dtype=bool).reshape(-1, 3),
            "sequence_ids": np.asarray(seq_ids), "window_index": np.asarray(win_idx),
            "channel_names": names, "rates": dict(rates),
            "obs_axes": np.stack(extra["obs_axes"]), "q_obs": np.stack(extra["q_obs"]),
            "row_present": np.stack(extra["row_present"]),
        }
        for key in ("obs_thm", "obs_tof"):
            if extra[key]:
                chunks[key] = np.stack(extra[key])
        # a modality that is absent for a window carries zeros, never stale values
        for j, name in enumerate(MODALITIES):
            if name in chunks["branches"]:
                absent = ~chunks["present"][:, j]
                chunks["branches"][name][absent] = 0.0
        return chunks

    def _prepare_sequence(self, frame: pd.DataFrame) -> Dict[str, Any]:
        """One sequence -> arrays at each branch rate plus observation masks."""
        rates = self.branch_rates()
        row = positive_rate(self.row_sampling_rate)
        counter = (pd.to_numeric(frame[self.counter_col], errors="coerce").to_numpy(dtype=float)
                   if self.counter_col in frame else np.arange(len(frame), dtype=float))
        if not np.isfinite(counter).all():
            counter = np.arange(len(frame), dtype=float)
        order = np.argsort(counter, kind="stable")
        counter = counter[order].astype(np.int64)
        keep = np.r_[np.diff(counter) != 0, True]  # keep the last duplicate
        order, counter = order[keep], counter[keep]
        pos = counter - counter.min()
        if pos.max() > 20 * len(pos) + 1000:  # implausible counters: fall back to row rank
            pos = np.arange(len(pos))
        n_rows = int(pos.max()) + 1
        row_present = np.zeros(n_rows, dtype=bool)
        row_present[pos] = True

        def dense(columns):
            out = np.full((n_rows, len(columns)), np.nan)
            cols = [c for c in columns if c in frame]
            if cols:
                idx = [columns.index(c) for c in cols]
                out[np.ix_(pos, idx)] = frame[cols].to_numpy(dtype=float)[order]
            return out

        def to_rate(values, native, target, observed_rows):
            """values (n_rows, C) NaN-filled -> (resampled, observed at the output rate)."""
            src_rate = row if self.align_mode == "time" else native
            filled, _ = fill_missing_time(values)
            out = resample_array(filled, src_rate, target, self.resample_method)
            idx = nearest_indices(len(filled), src_rate, target)[:len(out)]
            if len(idx) < len(out):
                idx = np.r_[idx, np.full(len(out) - len(idx), len(filled) - 1)]
            return out, observed_rows[idx]

        fs_i = rates["imu"]
        names, need_linear, need_ang = self._imu_plan()
        acc_rows = dense(ACC_COLUMNS)
        acc_obs_rows = np.isfinite(acc_rows)
        acc_rate_native = float(self.imu_native_sampling_rate)
        acc, _ = to_rate(acc_rows, acc_rate_native, fs_i, np.ones(n_rows, dtype=bool))
        n_imu = len(acc)
        idx_acc = nearest_indices(n_rows, row if self.align_mode == "time" else acc_rate_native, fs_i)[:n_imu]
        obs_axes = acc_obs_rows[np.clip(idx_acc, 0, n_rows - 1)] if n_imu else np.zeros((0, 3), dtype=bool)
        present_row = row_present[np.clip(idx_acc, 0, n_rows - 1)]
        quats_rows = dense(ROT_COLUMNS)
        q_valid_rows = np.isfinite(quats_rows).all(axis=1) & (np.linalg.norm(np.nan_to_num(quats_rows), axis=1) > 0)
        if self.fix_quaternion_sign:
            quats_rows = fix_quaternion_continuity(quats_rows)
        rot_native = float(self.rot_native_sampling_rate)
        q_fill, _ = fill_missing_time(np.where(q_valid_rows[:, None], quats_rows, np.nan), fill_value=0.0)
        q_res, _ = to_rate(q_fill, rot_native, fs_i, np.ones(n_rows, dtype=bool))
        norms = np.linalg.norm(q_res, axis=1, keepdims=True)
        q_res = np.where(norms > 1e-9, q_res / np.maximum(norms, 1e-9), 0.0)
        idx_rot = nearest_indices(n_rows, row if self.align_mode == "time" else rot_native, fs_i)[:n_imu]
        q_obs = q_valid_rows[np.clip(idx_rot, 0, n_rows - 1)] if n_imu else np.zeros(0, dtype=bool)
        columns: Dict[str, np.ndarray] = {}
        for i, a in enumerate("xyz"):
            columns[f"acc_{a}"] = acc[:, i]
        columns["acc_mag"] = np.linalg.norm(acc, axis=1)
        if need_linear and n_imu:
            lin = self._linear_acceleration(acc, q_res, q_obs, fs_i)
            for i, a in enumerate("xyz"):
                columns[f"lin_{a}"] = lin[:, i]
            columns["lin_mag"] = np.linalg.norm(lin, axis=1)
        if n_imu:
            jerk = np.vstack([np.zeros((1, 3)), np.diff(acc, axis=0) * fs_i])
            for i, a in enumerate("xyz"):
                columns[f"jerk_{a}"] = jerk[:, i]
            for i, a in enumerate("wxyz"):
                columns[f"rot_{a}"] = q_res[:, i]
            if need_ang:
                ang = np.nan_to_num(angular_velocity_body(q_res, fs_i, q_obs), nan=0.0)
                for i, a in enumerate("xyz"):
                    columns[f"ang_{a}"] = ang[:, i]
            if any(n.startswith("euler") for n in names):
                good = np.linalg.norm(q_res, axis=1) > 0
                eul = np.zeros((n_imu, 3))
                if good.any():
                    eul[good] = Rotation.from_quat(q_res[good][:, [1, 2, 3, 0]]).as_euler("xyz")
                for i, a in enumerate("xyz"):
                    columns[f"euler_{a}"] = eul[:, i]
        imu = (np.stack([columns[n] for n in names], axis=1) if n_imu else np.zeros((0, len(names))))
        prepared: Dict[str, Any] = {
            "imu": imu, "obs_axes": obs_axes, "q_obs": q_obs, "row_present": present_row,
            "present": {"imu": bool(acc_obs_rows.any()), "thm": False, "tof": False},
        }
        if "thm" in rates:
            thm_rows = dense(THM_COLUMNS)
            thm_obs_rows = np.isfinite(thm_rows).all(axis=1)
            prepared["present"]["thm"] = bool(np.isfinite(thm_rows).any())
            prepared["thm"], prepared["obs_thm"] = self._thermopile(thm_rows, thm_obs_rows, rates["thm"], to_rate)
        if "tof" in rates:
            tof_rows = dense(TOF_COLUMNS)
            tof_rows[tof_rows < 0] = np.nan  # -1 means "no return"
            prepared["present"]["tof"] = bool(np.isfinite(tof_rows).any())
            prepared["tof"], prepared["obs_tof"] = self._tof(tof_rows, rates["tof"], to_rate)
        return prepared

    def _linear_acceleration(self, acc, quats, q_obs, fs):
        """Gravity-removed acceleration: rotate gravity into the sensor frame, else high-pass."""
        lin = np.empty_like(acc)
        have_q = q_obs.all() if len(q_obs) else False
        if self.linear_acc_mode in {"baseline", "auto", "quaternion"} and have_q:
            rot = Rotation.from_quat(quats[:, [1, 2, 3, 0]])
            gravity = rot.inv().apply(np.array([0.0, 0.0, GRAVITY]))
            return acc - gravity
        if (self.linear_acc_mode == "highpass" or self.use_highpass_fallback) and len(acc) > 18 and 0 < self.highpass_hz < fs / 2:
            sos = sps.butter(2, float(self.highpass_hz), btype="highpass", fs=fs, output="sos")
            return sps.sosfiltfilt(sos, acc, axis=0)
        lin[:] = acc - acc.mean(axis=0, keepdims=True)
        return lin

    def _thermopile(self, rows, obs_rows, target, to_rate):
        _, _, thm, _, _ = self._modes()
        native = float(self.thm_native_sampling_rate)
        mean = np.nanmean(np.where(np.isfinite(rows), rows, np.nan), axis=1) if np.isfinite(rows).any() else np.full(len(rows), np.nan)
        base, obs = to_rate(rows, native, target, obs_rows)
        mean_r, _ = to_rate(mean[:, None], native, target, obs_rows)
        if not np.isfinite(rows).any():
            base = np.zeros_like(base)
            mean_r = np.zeros_like(mean_r)
        centered = base - mean_r
        diff = np.vstack([np.zeros((1, base.shape[1])), np.diff(base, axis=0)]) if len(base) else base
        cdiff = np.vstack([np.zeros((1, base.shape[1])), np.diff(centered, axis=0)]) if len(base) else base
        parts = []
        if "raw" in thm:
            parts.append(base)
        if "centered" in thm or "centered_diff" in thm:
            parts.append(centered)
        if "diff" in thm or "centered_diff" in thm:
            parts.append(cdiff if "centered_diff" in thm and "diff" not in thm else diff)
        if "pooled_stats" in thm:
            p_stats = np.stack([
                np.std(base, axis=1),
                np.min(base, axis=1),
                np.max(base, axis=1),
            ], axis=1) if len(base) else np.zeros((0, 3))
            parts.append(p_stats)
        parts.append(mean_r)
        return np.concatenate(parts, axis=1), obs

    def _tof(self, rows, target, to_rate):
        _, _, _, tof, _ = self._modes()
        native = float(self.tof_native_sampling_rate)
        valid = np.isfinite(rows)
        any_valid = valid.any(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sensors = rows.reshape(len(rows), TOF_SHAPE[0], -1)
            pooled = np.stack([np.nanmean(rows, axis=1), np.nanstd(rows, axis=1), np.nanmin(rows, axis=1),
                               np.nanmax(rows, axis=1), valid.mean(axis=1)], axis=1)
            s_mean = np.nanmean(sensors, axis=2)
            s_std = np.nanstd(sensors, axis=2)
        pooled[~any_valid, :4] = np.nan
        pooled[:, 4] = np.where(any_valid | True, pooled[:, 4], 0.0)
        s_mean[~valid.reshape(len(rows), TOF_SHAPE[0], -1).any(axis=2)] = np.nan
        s_std[~valid.reshape(len(rows), TOF_SHAPE[0], -1).any(axis=2)] = np.nan
        parts = []
        if "raw" in tof:
            if self.tof_fill_mode == "far_500":
                raw = np.where(valid, rows, 500.0)
            elif self.tof_fill_mode == "far_255":
                raw = np.where(valid, rows, 255.0)
            elif self.tof_fill_mode == "zero":
                raw = np.where(valid, rows, 0.0)
            else:
                raw = rows
            parts.append(raw)
        stats = [pooled[:, :1]]
        if "pooled_stats" in tof:
            stats.append(pooled[:, 1:])
        if "sensor_stats" in tof:
            stats += [s_mean, s_std]
        parts.append(np.concatenate(stats, axis=1))
        values = np.concatenate(parts, axis=1)
        out, obs = to_rate(values, native, target, any_valid)
        if not valid.any():
            out = np.zeros_like(out)
        return out, obs

    # -- static features (tsfel / tsfresh + NOOP) -------------------------------
    def _sources(self, chunks) -> Dict[str, Dict[str, np.ndarray]]:
        """Named 1-D series per chunk that static features are computed from."""
        _, _, _, _, wanted = self._modes()
        names = chunks["channel_names"]
        rates = chunks["rates"]
        imu = chunks["branches"]["imu"]
        imu_mask = chunks["masks"]["imu"]
        obs_imu = chunks["obs_axes"].all(axis=-1) & imu_mask
        sources: Dict[str, Dict[str, np.ndarray]] = {}

        def channel(branch, label):
            return chunks["branches"][branch][:, :, names[branch].index(label)]

        def vector(prefix):
            idx = [names["imu"].index(f"{prefix}_{a}") for a in "xyz"]
            return imu[:, :, idx]

        if "acc_mag" in wanted:
            sources["acc_mag"] = {"x": np.linalg.norm(vector("acc"), axis=-1), "fs": rates["imu"], "obs": obs_imu,
                                  "mask": imu_mask, "branch": 0}
        if "lin_mag" in wanted and "lin_x" in names["imu"]:
            sources["lin_mag"] = {"x": np.linalg.norm(vector("lin"), axis=-1), "fs": rates["imu"], "obs": obs_imu,
                                  "mask": imu_mask, "branch": 0}
        if "ang_speed" in wanted and "ang_x" in names["imu"]:
            sources["ang_speed"] = {"x": np.linalg.norm(vector("ang"), axis=-1), "fs": rates["imu"],
                                    "obs": chunks["q_obs"] & imu_mask, "mask": imu_mask, "branch": 0}
        if "thm_mean" in wanted and "thm" in rates:
            sources["thm_mean"] = {"x": channel("thm", "thm_mean"), "fs": rates["thm"],
                                   "obs": chunks["obs_thm"] & chunks["masks"]["thm"],
                                   "mask": chunks["masks"]["thm"], "branch": 1}
        if "tof_mean" in wanted and "tof" in rates:
            sources["tof_mean"] = {"x": channel("tof", "tof_mean"), "fs": rates["tof"],
                                   "obs": chunks["obs_tof"] & chunks["masks"]["tof"],
                                   "mask": chunks["masks"]["tof"], "branch": 2}
        return sources

    def tsfel_config(self):
        """tsfel feature dictionary for the requested domains / feature names."""
        _require(TSFEL_AVAILABLE, "tsfel")
        domains = _pipe_modes(self.tsfel_domains, TSFEL_DOMAINS, name="tsfel_domains")
        wanted = None if self.tsfel_features is None else set(_structured(self.tsfel_features) or ())
        if self.tsfel_features is None:
            wanted = set(TSFEL_DEFAULT_FEATURES)
        elif isinstance(self.tsfel_features, str) and "|" in self.tsfel_features:
            wanted = set(self.tsfel_features.split("|"))
        elif isinstance(self.tsfel_features, str):
            wanted = {self.tsfel_features}
        cfg: Dict[str, Any] = {}
        for domain in domains:
            full = tsfel.get_features_by_domain(domain)
            for feature, spec in full.get(domain, {}).items():
                if feature in wanted:
                    cfg.setdefault(domain, {})[feature] = dict(spec, use="yes")
        if not cfg:
            raise InvalidParams("no tsfel features selected for the requested domains")
        return cfg

    def _tsfel_features(self, sources):
        cfg = self.tsfel_config()
        columns: Dict[str, np.ndarray] = {}
        n = len(next(iter(sources.values()))["x"]) if sources else 0
        for src, info in sources.items():
            lengths = info["mask"].sum(axis=1)
            rows: List[Optional[np.ndarray]] = [None] * n
            for i in range(n):
                L = int(lengths[i])
                if L < 8 or not info["obs"][i, :L].any():
                    continue
                frame = pd.DataFrame({src: np.asarray(info["x"][i, :L], dtype=float)})
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    out = tsfel.time_series_features_extractor(cfg, frame, fs=float(info["fs"]), verbose=0)
                rows[i] = out.iloc[0]
            template = next((r for r in rows if r is not None), None)
            if template is None:
                template = self._tsfel_template(cfg, src, info)
            for name in template.index:
                columns[f"tsfel__{name}"] = np.array(
                    [float(r[name]) if r is not None and name in r.index else 0.0 for r in rows])
        return columns

    def _tsfel_template(self, cfg, src, info):
        probe = pd.DataFrame({src: np.sin(np.arange(64) / 3.0)})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return tsfel.time_series_features_extractor(cfg, probe, fs=float(info["fs"]), verbose=0).iloc[0]

    def tsfresh_settings(self, fs: float, n: int):
        """tsfresh settings whose FFT / autocorrelation indices follow the sensor's rate."""
        bands = _structured(self.tsfresh_fft_bands)
        coeffs = []
        for lo, hi in bands:
            centre = 0.5 * (float(lo) + float(hi))
            k = int(round(centre * n / float(fs)))
            if 0 < k <= n // 2:
                coeffs.append(k)
        settings: Dict[str, Any] = {
            "mean": None, "standard_deviation": None, "skewness": None, "kurtosis": None,
            "minimum": None, "maximum": None, "root_mean_square": None,
            "mean_abs_change": None, "absolute_sum_of_changes": None,
            "fft_aggregated": [{"aggtype": a} for a in ("centroid", "variance", "skew", "kurtosis")],
        }
        if coeffs:
            settings["fft_coefficient"] = [{"coeff": k, "attr": "abs"} for k in sorted(set(coeffs))]
        if self.tsfresh_profile == "efficient":
            lags = sorted({max(1, int(round(float(fs) / f))) for f in (1.0, 2.0, 4.0) if float(fs) / f < n})
            settings.update({"autocorrelation": [{"lag": lag} for lag in lags],
                             "approximate_entropy": [{"m": 2, "r": 0.2}],
                             "number_peaks": [{"n": 1}, {"n": 3}]})
        return settings

    def _tsfresh_features(self, sources):
        _require(TSFRESH_AVAILABLE, "tsfresh")
        frames, kind_settings = [], {}
        n_chunks = len(next(iter(sources.values()))["x"]) if sources else 0
        for src, info in sources.items():
            lengths = info["mask"].sum(axis=1)
            for i in range(n_chunks):
                L = int(lengths[i])
                if L < 8 or not info["obs"][i, :L].any():
                    continue
                frames.append(pd.DataFrame({"id": i, "time": np.arange(L), "kind": src,
                                            "value": np.asarray(info["x"][i, :L], dtype=float)}))
            kind_settings[src] = self.tsfresh_settings(info["fs"], info["x"].shape[1])
        columns: Dict[str, np.ndarray] = {}
        if not frames:
            return columns
        long = pd.concat(frames, ignore_index=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = _tsfresh_extract(long, column_id="id", column_sort="time", column_kind="kind",
                                   column_value="value", kind_to_fc_parameters=kind_settings,
                                   disable_progressbar=True, n_jobs=0)
        out = out.reindex(range(n_chunks)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        for name in out.columns:
            columns[f"tsfresh__{name}"] = out[name].to_numpy(dtype=float)
        return columns

    def _noop_features(self, chunks):
        opts = self.noop_options()
        if opts is None:
            return {}, None
        rates = chunks["rates"]
        names = chunks["channel_names"]["imu"]
        imu, imu_mask = chunks["branches"]["imu"], chunks["masks"]["imu"]
        acc_idx = [names.index(f"acc_{a}") for a in "xyz"]
        lin_idx = [names.index(f"lin_{a}") for a in "xyz"] if "lin_x" in names else None
        ang_idx = [names.index(f"ang_{a}") for a in "xyz"] if "ang_x" in names else None
        fs = rates["imu"]
        reports, rows, schema = [], [], None
        power = opts.branch_names() - {"imu"}
        motion, spectral = opts.mode_names()
        for i in range(len(imu)):
            L = int(imu_mask[i].sum())
            window = imu[i, :L].astype(float)
            present = chunks["present"][i, 0]
            nan3 = np.full((L, 3), np.nan)
            src = {
                "raw": window[:, acc_idx], "observed": chunks["obs_axes"][i, :L] & present,
                "linear": window[:, lin_idx] if lin_idx else nan3,
                "rotation": window[:, ang_idx] if ang_idx else nan3,
                "rotation_observed": chunks["q_obs"][i, :L] & present,
                "counter": np.arange(L, dtype=float), "dt": np.full(L, 1.0 / fs),
                "row_present": chunks["row_present"][i, :L],
            }
            feats, report = noop_imu_window(src, fs, opts)
            if motion or spectral:
                for j, branch in ((1, "thm"), (2, "tof")):
                    if branch in power and branch in rates:
                        label = f"{branch}_mean"
                        length = int(chunks["masks"][branch][i].sum())
                        series = chunks["branches"][branch][i, :length, names_index(chunks, branch, label)][:, None]
                        observed = chunks[f"obs_{branch}"][i, :length] & chunks["present"][i, j]
                        feats.update(noop_power_window(series.astype(float), observed, rates[branch], opts, branch))
            keys = list(feats)
            if schema is None:
                schema = keys
            elif schema != keys:
                raise AssertionError("NOOP summary schema changed between windows")
            report["sequence_id"] = str(chunks["sequence_ids"][i])
            rows.append([feats[k] for k in keys])
            reports.append(report)
        if schema is None:
            schema = self._noop_schema_probe(chunks)
        arr = np.asarray(rows, dtype=np.float32).reshape(len(rows), len(schema))
        return {name: arr[:, j] for j, name in enumerate(schema)}, (reports if opts.quality is not None else None)

    def _noop_schema_probe(self, chunks):
        opts = self.noop_options()
        fs = chunks["rates"]["imu"]
        L = 32
        src = {"raw": np.zeros((L, 3)), "observed": np.zeros((L, 3), bool), "linear": np.zeros((L, 3)),
               "rotation": np.zeros((L, 3)), "rotation_observed": np.zeros(L, bool),
               "counter": np.arange(L, dtype=float), "dt": np.full(L, 1 / fs)}
        feats, _ = noop_imu_window(src, fs, opts)
        motion, spectral = opts.mode_names()
        if motion or spectral:
            for branch in opts.branch_names() - {"imu"}:
                if branch in chunks["rates"]:
                    feats.update(noop_power_window(np.zeros((L, 1)), np.zeros(L, bool),
                                                   chunks["rates"][branch], opts, branch))
        return list(feats)

    def featurize(self, chunks: Dict[str, Any]) -> Dict[str, Any]:
        """Attach static features (tsfel/tsfresh + NOOP + presence flags) to raw chunks."""
        n = len(chunks["sequence_ids"])
        columns: Dict[str, np.ndarray] = {}
        backend = self.static_backend
        if backend in {"tsfel", "tsfresh"} and n:
            sources = self._sources(chunks)
            if sources:
                columns.update(self._tsfel_features(sources) if backend == "tsfel" else self._tsfresh_features(sources))
        generic = list(columns)
        noop_cols, reports = self._noop_features(chunks)
        columns.update(noop_cols)
        for j, name in enumerate(MODALITIES):
            columns[f"present_{name}"] = chunks["present"][:, j].astype(np.float32)
        expected = getattr(self, "static_names_", None)
        names = list(expected) if expected is not None else list(columns)
        matrix = np.zeros((n, len(names)), dtype=np.float32)
        for j, name in enumerate(names):
            if name in columns:
                matrix[:, j] = np.nan_to_num(np.asarray(columns[name], dtype=np.float64), nan=0.0,
                                             posinf=0.0, neginf=0.0).astype(np.float32)
        out = dict(chunks)
        out.update(static=matrix, static_names=names, noop_feature_names=list(noop_cols),
                   generic_feature_names=generic)
        if reports is not None:
            out["noop_quality"] = reports
        return out

    # -- scaling -----------------------------------------------------------------
    def fit_scalers(self, chunks: Dict[str, Any]):
        """Per-channel mean/std over present, real (unpadded) samples; static column z-scores."""
        self.channel_names_ = {k: list(v) for k, v in chunks["channel_names"].items()}
        self.rates_ = dict(chunks["rates"])
        self.scalers_ = {}
        for j, name in enumerate(MODALITIES):
            if name not in chunks["branches"]:
                continue
            arr, mask = chunks["branches"][name], chunks["masks"][name] & chunks["present"][:, j][:, None]
            flat = arr[mask]
            if len(flat):
                mean, std = flat.mean(axis=0), flat.std(axis=0)
            else:
                mean, std = np.zeros(arr.shape[2]), np.ones(arr.shape[2])
            self.scalers_[name] = (mean.astype(np.float32), np.where(std > 1e-6, std, 1.0).astype(np.float32))
        static = chunks["static"]
        mean = static.mean(axis=0) if len(static) else np.zeros(static.shape[1])
        std = static.std(axis=0) if len(static) else np.ones(static.shape[1])
        self.static_scaler_ = (mean.astype(np.float32), np.where(std > 1e-6, std, 1.0).astype(np.float32))
        self.static_names_ = list(chunks["static_names"])
        self.noop_feature_names_ = list(chunks["noop_feature_names"])
        return self

    def scale(self, chunks: Dict[str, Any]) -> Dict[str, Any]:
        """Standardise branch tensors and static features; absent modalities stay exactly zero."""
        out = dict(chunks)
        branches = {}
        for j, name in enumerate(MODALITIES):
            if name not in chunks["branches"]:
                continue
            mean, std = self.scalers_[name]
            arr = (chunks["branches"][name] - mean) / std
            arr = np.clip(arr, -self.static_clip * 5, self.static_clip * 5)
            arr = arr * chunks["masks"][name][:, :, None] * chunks["present"][:, j][:, None, None]
            branches[name] = arr.astype(np.float32)
        out["branches"] = branches
        mean, std = self.static_scaler_
        static = np.clip((chunks["static"] - mean) / std, -float(self.static_clip), float(self.static_clip))
        out["static"] = static.astype(np.float32)
        return out

    def select_finetuning_sequences(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Return the valid sequence dataframe for holdout/internal-validation evaluation."""
        return X.copy()



def names_index(chunks, branch, label):
    return chunks["channel_names"][branch].index(label)


# ---------------------------------------------------------------------------
# 5. Augmentation (tsaug)
# ---------------------------------------------------------------------------

AUGMENTATIONS = ("jitter", "noise", "scaling", "magnitude_warp", "time_warp", "time_shift", "crop",
                 "temporal_mask", "channel_drop")
_TIMING_OPS = {"time_warp", "time_shift", "crop"}


def shift_time(x: np.ndarray, max_shift_frac: float, rng: np.random.Generator) -> np.ndarray:
    """Per-sample circular-free shift by up to ``max_shift_frac * T`` samples (edge replicated).

    tsaug has no time-shift augmenter, so this is the one operation implemented here.
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


class TSAugmentor(BaseEstimator):
    """Training-time augmentation of per-modality window tensors using ``tsaug``.

    Every knob is an sklearn parameter, so grid / Bayesian / evolutionary searches tune
    them directly.  ``prob`` is the chance a window is augmented at all; ``per_aug_prob``
    the chance each enabled operation fires inside an augmented window.

    ============================ ======================================================
    parameter                    tsaug operation
    ============================ ======================================================
    ``jitter_sigma``             ``AddNoise(scale, normalize=False)`` (absolute units)
    ``noise_std``                ``AddNoise`` relative to each channel's std
    ``warp_sigma/warp_num_knots`` ``Drift(kind='multiplicative')`` (smooth magnitude warp)
    ``time_warp_*``              ``TimeWarp(n_speed_change, max_speed_ratio)``
    ``crop_frac_range``          ``Crop(size=..., resize=T)`` (crop + stretch back)
    ``temporal_num_masks/frac``  ``Dropout(p, size, fill)`` over all channels
    ``channel_drop_prob``        ``Dropout(per_channel=True)`` over whole windows
    ``time_shift_frac``          :func:`shift_time` (no tsaug equivalent)
    ``scaling_sigma``            per-window gain ``N(1, sigma)`` (no tsaug equivalent)
    ``sensor_drop_prob`` /       whole thermopile / ToF branch removed (hidden-test
    ``sensor_drop_joint_prob``   simulation); applied per mini-batch during training
    ============================ ======================================================

    ``n_copies`` augmented replicas of the training windows are created each fit and their
    static (tsfel / NOOP) features are recomputed from the augmented signal.  Timing
    operations are skipped for windows that contain unobserved samples so that the
    observation masks stay valid.
    """

    def __init__(
        self, prob=0.0, per_aug_prob=0.5, augmentations=None, n_copies=1,
        jitter_sigma=0.0, noise_std=0.0, scaling_sigma=0.0,
        warp_sigma=0.0, warp_num_knots=4,
        time_warp_max_speed_ratio=1.0, time_warp_n_speed_change=3,
        time_shift_frac=0.0, crop_frac_range=(1.0, 1.0),
        temporal_num_masks=0, temporal_mask_frac=0.0, temporal_mask_fill="ffill",
        channel_drop_prob=0.0, sensor_drop_prob=0.0, sensor_drop_joint_prob=0.0,
        random_state=None,
    ):
        self.prob = prob
        self.per_aug_prob = per_aug_prob
        self.augmentations = augmentations
        self.n_copies = n_copies
        self.jitter_sigma = jitter_sigma
        self.noise_std = noise_std
        self.scaling_sigma = scaling_sigma
        self.warp_sigma = warp_sigma
        self.warp_num_knots = warp_num_knots
        self.time_warp_max_speed_ratio = time_warp_max_speed_ratio
        self.time_warp_n_speed_change = time_warp_n_speed_change
        self.time_shift_frac = time_shift_frac
        self.crop_frac_range = crop_frac_range
        self.temporal_num_masks = temporal_num_masks
        self.temporal_mask_frac = temporal_mask_frac
        self.temporal_mask_fill = temporal_mask_fill
        self.channel_drop_prob = channel_drop_prob
        self.sensor_drop_prob = sensor_drop_prob
        self.sensor_drop_joint_prob = sensor_drop_joint_prob
        self.random_state = random_state

    # -- configuration -------------------------------------------------------
    def crop_range(self) -> Tuple[float, float]:
        value = _structured(self.crop_frac_range)
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise InvalidParams("crop_frac_range must be (low, high)")
        lo, hi = float(value[0]), float(value[1])
        if not 0 < lo <= hi <= 1:
            raise InvalidParams("crop_frac_range must satisfy 0 < low <= high <= 1")
        return lo, hi

    def allowed_ops(self) -> set:
        value = _structured(self.augmentations)
        if value is None:
            return set(AUGMENTATIONS)
        if isinstance(value, str):
            value = value.split("|")
        unknown = set(value) - set(AUGMENTATIONS)
        if unknown:
            raise InvalidParams(f"unknown augmentations {sorted(unknown)}; choose from {AUGMENTATIONS}")
        return set(value)

    def validate(self):
        for name in ("prob", "per_aug_prob", "channel_drop_prob", "sensor_drop_prob", "sensor_drop_joint_prob"):
            if not 0 <= float(getattr(self, name)) <= 1:
                raise InvalidParams(f"{name} must be in [0, 1]")
        if self.sensor_drop_prob >= 1 or self.channel_drop_prob >= 1:
            raise InvalidParams("removal probabilities must stay below 1")
        for name in ("jitter_sigma", "noise_std", "scaling_sigma", "warp_sigma", "time_shift_frac", "temporal_mask_frac"):
            if float(getattr(self, name)) < 0 or not np.isfinite(float(getattr(self, name))):
                raise InvalidParams(f"{name} must be a finite non-negative number")
        if float(self.time_warp_max_speed_ratio) < 1:
            raise InvalidParams("time_warp_max_speed_ratio must be >= 1 (1 disables time warping)")
        if int(self.warp_num_knots) < 1 or int(self.time_warp_n_speed_change) < 1:
            raise InvalidParams("warp knot / speed-change counts must be >= 1")
        if int(self.n_copies) < 0:
            raise InvalidParams("n_copies must be >= 0")
        self.crop_range()
        self.allowed_ops()
        return self

    def active_ops(self) -> List[str]:
        allowed = self.allowed_ops()
        lo, hi = self.crop_range()
        strengths = {
            "jitter": self.jitter_sigma > 0, "noise": self.noise_std > 0, "scaling": self.scaling_sigma > 0,
            "magnitude_warp": self.warp_sigma > 0,
            "time_warp": self.time_warp_max_speed_ratio > 1, "time_shift": self.time_shift_frac > 0,
            "crop": hi < 1, "temporal_mask": self.temporal_num_masks > 0 and self.temporal_mask_frac > 0,
            "channel_drop": self.channel_drop_prob > 0,
        }
        return [op for op in AUGMENTATIONS if op in allowed and strengths[op]]

    def enabled(self) -> bool:
        return (self.n_copies > 0 and self.prob > 0 and self.per_aug_prob > 0 and bool(self.active_ops()))

    # -- operations ------------------------------------------------------------
    def _tsaug_op(self, op: str, length: int, seed: int):
        _require(TSAUG_AVAILABLE, "tsaug")
        if op == "jitter":
            return tsaug.AddNoise(scale=float(self.jitter_sigma), normalize=False, seed=seed)
        if op == "noise":
            return tsaug.AddNoise(scale=float(self.noise_std), normalize=False, seed=seed)
        if op == "magnitude_warp":
            return tsaug.Drift(max_drift=float(self.warp_sigma), n_drift_points=int(self.warp_num_knots),
                               kind="multiplicative", per_channel=True, normalize=False, seed=seed)
        if op == "time_warp":
            return tsaug.TimeWarp(n_speed_change=int(self.time_warp_n_speed_change),
                                  max_speed_ratio=float(self.time_warp_max_speed_ratio), seed=seed)
        if op == "crop":
            lo, hi = self.crop_range()
            sizes = sorted({int(np.clip(round(f * length), 2, length)) for f in np.linspace(lo, hi, 6)})
            return tsaug.Crop(size=sizes, resize=length, seed=seed)
        if op == "temporal_mask":
            n_masks = max(1, int(self.temporal_num_masks))
            size = max(1, int(round(float(self.temporal_mask_frac) * length / n_masks)))
            fill = self.temporal_mask_fill if isinstance(self.temporal_mask_fill, str) else float(self.temporal_mask_fill)
            return tsaug.Dropout(p=float(self.temporal_mask_frac), size=[size], fill=fill, per_channel=False, seed=seed)
        if op == "channel_drop":
            return tsaug.Dropout(p=float(self.channel_drop_prob), size=[length], fill=0.0, per_channel=True, seed=seed)
        raise InvalidParams(f"operation {op!r} has no tsaug mapping")

    def augment_array(self, x, rng, gate, protect=None, reference_std=None):
        """Augment ``x`` ``(N, T, C)`` -- all samples share the same length ``T``.

        ``gate`` ``(N,)`` marks windows selected for augmentation; ``protect`` marks windows
        whose timing must not change; ``reference_std`` ``(C,)`` scales ``noise_std``.
        """
        x = np.asarray(x, dtype=np.float64)
        n, length, channels = x.shape
        out = x.copy()
        protect = np.zeros(n, dtype=bool) if protect is None else np.asarray(protect, dtype=bool)
        ref = np.ones(channels) if reference_std is None else np.where(np.asarray(reference_std) > 1e-9, reference_std, 1.0)
        for op in self.active_ops():
            fire = gate & (rng.random(n) < float(self.per_aug_prob))
            if op in _TIMING_OPS:
                fire &= ~protect
            idx = np.flatnonzero(fire)
            if not len(idx) or length < 4:
                continue
            sub = out[idx]
            seed = int(rng.integers(0, 2 ** 31 - 1))
            if op == "time_shift":
                sub = shift_time(sub, float(self.time_shift_frac), rng)
            elif op == "scaling":
                sub = sub * rng.normal(1.0, float(self.scaling_sigma), size=(len(idx), 1, channels))
            elif op == "noise":
                sub = self._tsaug_op(op, length, seed).augment(sub / ref) * ref
            else:
                sub = self._tsaug_op(op, length, seed).augment(sub)
            out[idx] = sub
        return out.astype(x.dtype if x.dtype == np.float32 else np.float64)

    def augment_chunks(self, chunks: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
        """Return a copy of raw (unscaled) chunks with augmented branch tensors."""
        self.validate()
        out = dict(chunks)
        n = len(chunks["sequence_ids"])
        gate = rng.random(n) < float(self.prob)
        branches = {}
        obs = {"imu": chunks["obs_axes"].all(axis=-1), "thm": chunks.get("obs_thm"), "tof": chunks.get("obs_tof")}
        for j, name in enumerate(MODALITIES):
            if name not in chunks["branches"]:
                continue
            arr, mask = chunks["branches"][name], chunks["masks"][name]
            lengths = mask.sum(axis=1)
            real = arr[mask & chunks["present"][:, j][:, None]]
            ref_std = real.std(axis=0) if len(real) else np.ones(arr.shape[2])
            observed = obs[name]
            partial = ((~observed) & mask).any(axis=1) if observed is not None else np.zeros(n, dtype=bool)
            new = arr.copy()
            usable = gate & chunks["present"][:, j]
            for length in np.unique(lengths):
                members = np.flatnonzero((lengths == length) & usable)
                if not len(members) or length < 4:
                    continue
                aug = self.augment_array(arr[members, :length], rng, np.ones(len(members), dtype=bool),
                                         protect=partial[members], reference_std=ref_std)
                new[members, :length] = aug
            branches[name] = new.astype(np.float32)
        out["branches"] = branches
        return out

    def sample_modality_keep(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """``(n, 3)`` bool keep-matrix for [imu, thm, tof]; thm/tof drop independently or jointly."""
        keep = np.ones((n, 3), dtype=bool)
        if self.sensor_drop_joint_prob > 0:
            joint = rng.random(n) < float(self.sensor_drop_joint_prob)
            keep[joint, 1:] = False
        if self.sensor_drop_prob > 0:
            keep[:, 1] &= rng.random(n) >= float(self.sensor_drop_prob)
            keep[:, 2] &= rng.random(n) >= float(self.sensor_drop_prob)
        return keep


# ---------------------------------------------------------------------------
# 6. Network (tsai backbones per modality + spatial ToF + fusion + estimator)
# ---------------------------------------------------------------------------

TSAI_ARCHS = (
    "InceptionTimePlus", "ResNetPlus", "FCNPlus", "TCN",
    "TSTPlus", "GRUPlus", "LSTMPlus", "XCMPlus", "Conv1d",
)
FUSION_TYPES = ("concat", "attention", "gated")
POOLING_MODES = ("default", "concat_pool", "attention")
TOF_SPATIAL_MODES = ("auto", "conv2d", "flat")
ACTIVATIONS = ("relu", "gelu", "silu", "elu")
OPTIMIZERS = ("adamw", "adam", "sgd")
LOSS_NAMES = ("cross_entropy", "focal")
LR_SCHEDULERS = ("cosine", "plateau", "onecycle", "none")
SEQUENCE_AGGREGATIONS = ("mean_proba", "last_window", "majority_vote")

_TSAI_MODEL_CACHE: Optional[Dict[str, Any]] = None


def _get_tsai_classes() -> Dict[str, Any]:
    """Lazily import tsai model classes so module import stays fast."""
    global _TSAI_MODEL_CACHE
    if _TSAI_MODEL_CACHE is not None:
        return _TSAI_MODEL_CACHE
    _require(TSAI_AVAILABLE, "tsai")
    from tsai.models.all import (
        FCNPlus, GRUPlus, InceptionTimePlus, LSTMPlus,
        ResNetPlus, TCN, TSTPlus, XCMPlus,
    )
    _TSAI_MODEL_CACHE = {
        "InceptionTimePlus": InceptionTimePlus,
        "ResNetPlus": ResNetPlus,
        "FCNPlus": FCNPlus,
        "TCN": TCN,
        "TSTPlus": TSTPlus,
        "GRUPlus": GRUPlus,
        "LSTMPlus": LSTMPlus,
        "XCMPlus": XCMPlus,
    }
    return _TSAI_MODEL_CACHE


def _make_activation(name: str) -> nn.Module:
    key = str(name).lower()
    if key == "relu":
        return nn.ReLU()
    if key == "gelu":
        return nn.GELU()
    if key == "silu":
        return nn.SiLU()
    if key == "elu":
        return nn.ELU()
    raise InvalidParams(f"activation must be one of {ACTIVATIONS}, got {name!r}")


class _Conv1dBackbone(nn.Module):
    """Lightweight residual 1-D CNN backbone expecting (B, C_in, T) and returning (B, C_out)."""

    def __init__(self, c_in: int, c_out: int, nf: int = 64, depth: int = 3, ks: int = 7,
                 conv_dropout: float = 0.0, fc_dropout: float = 0.0, use_bn: bool = True,
                 concat_pool: bool = False, act: str = "gelu"):
        super().__init__()
        nf = max(4, int(nf))
        depth = max(1, int(depth))
        ks = max(3, int(ks)) | 1
        blocks: List[nn.Module] = []
        in_ch = int(c_in)
        for i in range(depth):
            out_ch = nf if i == 0 else nf
            block = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=ks, padding=ks // 2, bias=not use_bn),
                nn.BatchNorm1d(out_ch) if use_bn else nn.Identity(),
                _make_activation(act),
                nn.Dropout(float(conv_dropout)) if conv_dropout > 0 else nn.Identity(),
            )
            blocks.append(block)
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)
        self.concat_pool = bool(concat_pool)
        pool_dim = nf * 2 if self.concat_pool else nf
        self.head = nn.Sequential(
            nn.Dropout(float(fc_dropout)) if fc_dropout > 0 else nn.Identity(),
            nn.Linear(pool_dim, int(c_out)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for block in self.blocks:
            out = block(h)
            h = out + h if out.shape == h.shape else out
        if self.concat_pool:
            pooled = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=-1)
        else:
            pooled = h.mean(dim=-1)
        return self.head(pooled)


def build_tsai_backbone(
    arch: str, c_in: int, c_out: int, seq_len: int, *,
    filters: int = 64, depth: int = 3, kernel_size: int = 7,
    dropout: float = 0.1, conv_dropout: float = 0.0,
    num_heads: int = 4, use_batchnorm: bool = True,
    pooling_mode: str = "default", activation: str = "gelu",
) -> nn.Module:
    """Instantiate a ``tsai`` temporal backbone that maps ``(B, C_in, T) -> (B, C_out)``."""
    if arch not in TSAI_ARCHS:
        raise InvalidParams(f"arch must be one of {TSAI_ARCHS}, got {arch!r}")
    c_in = max(1, int(c_in))
    c_out = max(2, int(c_out))
    seq_len = max(4, int(seq_len))
    nf = max(4, int(filters))
    depth = max(1, int(depth))
    ks = max(3, int(kernel_size))
    fc_drop = float(dropout)
    c_drop = float(conv_dropout)
    concat_pool = (pooling_mode == "concat_pool")

    if arch == "Conv1d" or not TSAI_AVAILABLE:
        if arch != "Conv1d":
            _require(TSAI_AVAILABLE, "tsai")
        return _Conv1dBackbone(
            c_in, c_out, nf=nf, depth=depth, ks=ks,
            conv_dropout=c_drop, fc_dropout=fc_drop,
            use_bn=bool(use_batchnorm), concat_pool=concat_pool, act=activation,
        )

    classes = _get_tsai_classes()
    cls = classes[arch]
    if arch == "InceptionTimePlus":
        return cls(
            c_in, c_out, seq_len=seq_len, nf=nf, depth=depth, ks=ks,
            conv_dropout=c_drop, fc_dropout=fc_drop, bn=bool(use_batchnorm),
            concat_pool=concat_pool,
        )
    if arch == "ResNetPlus":
        return cls(
            c_in, c_out, seq_len=seq_len, nf=nf,
            fc_dropout=fc_drop, concat_pool=concat_pool,
        )
    if arch == "FCNPlus":
        layers = [nf, nf * 2, nf]
        kss = [max(3, (ks - 2 * i) | 1) for i in range(3)]
        return cls(
            c_in, c_out, layers=layers, kss=kss,
            use_bn=bool(use_batchnorm), fc_dropout=fc_drop, residual=True,
        )
    if arch == "TCN":
        layers = [nf] * depth
        return cls(c_in, c_out, layers=layers, ks=max(2, ks), conv_dropout=c_drop, fc_dropout=fc_drop)
    if arch == "TSTPlus":
        heads = max(1, int(num_heads))
        d_model = max(heads * 2, (nf // heads) * heads)
        return cls(
            c_in, c_out, seq_len=seq_len, max_seq_len=max(seq_len, 512),
            n_layers=depth, d_model=d_model, n_heads=heads, d_ff=d_model * 2,
            dropout=c_drop, fc_dropout=fc_drop, bn=bool(use_batchnorm),
            concat_pool=concat_pool,
        )
    if arch in {"GRUPlus", "LSTMPlus"}:
        return cls(
            c_in, c_out, seq_len=seq_len, hidden_size=nf, n_layers=depth,
            rnn_dropout=c_drop if depth > 1 else 0.0, bidirectional=True,
            fc_dropout=fc_drop, bn=bool(use_batchnorm),
        )
    if arch == "XCMPlus":
        return cls(
            c_in, c_out, seq_len=seq_len, nf=nf,
            fc_dropout=fc_drop, bn=bool(use_batchnorm), concat_pool=concat_pool,
        )
    raise InvalidParams(f"Unsupported architecture {arch!r}")


class ToFSpatialEncoder(nn.Module):
    """2-D spatial encoder over the 5 Time-of-Flight 8x8 pixel grids per timestep.

    When raw ToF pixels are enabled (320 channels = 5 sensors x 8 x 8), reshapes the
    first 320 channels of each timestep into ``(B * T, 5, 8, 8)``, applies 2-D spatial
    convolutions across the 8x8 sensor grids, and concatenates the pooled spatial
    features with any remaining ToF summary channels.
    """

    def __init__(self, in_channels: int, spatial_filters: int = 32, spatial_kernel: int = 3,
                 use_batchnorm: bool = True, activation: str = "gelu"):
        super().__init__()
        self.raw_pixels = int(np.prod(TOF_SHAPE))  # 320 = 5 * 8 * 8
        if in_channels < self.raw_pixels:
            raise ValueError(f"ToFSpatialEncoder requires at least {self.raw_pixels} channels, got {in_channels}")
        self.extra_channels = int(in_channels) - self.raw_pixels
        sf = max(4, int(spatial_filters))
        k = max(1, int(spatial_kernel))
        pad = k // 2
        self.conv2d = nn.Sequential(
            nn.Conv2d(TOF_SHAPE[0], sf, kernel_size=k, padding=pad, bias=not use_batchnorm),
            nn.BatchNorm2d(sf) if use_batchnorm else nn.Identity(),
            _make_activation(activation),
            nn.Conv2d(sf, sf, kernel_size=k, padding=pad, bias=not use_batchnorm),
            nn.BatchNorm2d(sf) if use_batchnorm else nn.Identity(),
            _make_activation(activation),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.out_channels = sf + self.extra_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        b, t, _ = x.shape
        raw = x[:, :, :self.raw_pixels].contiguous().view(b * t, TOF_SHAPE[0], TOF_SHAPE[1], TOF_SHAPE[2])
        spatial = self.conv2d(raw).view(b, t, -1)
        if self.extra_channels > 0:
            return torch.cat([spatial, x[:, :, self.raw_pixels:]], dim=-1)
        return spatial


class TemporalAttentionPool(nn.Module):
    """Masked additive attention over timesteps ``(B, T, C) -> (B, out_dim)``."""

    def __init__(self, in_channels: int, attn_units: int, out_dim: int, activation: str = "gelu"):
        super().__init__()
        u = max(8, int(attn_units))
        self.score = nn.Sequential(
            nn.Linear(int(in_channels), u),
            nn.Tanh(),
            nn.Linear(u, 1, bias=False),
        )
        self.proj = nn.Sequential(
            nn.Linear(int(in_channels), int(out_dim)),
            _make_activation(activation),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C), mask: (B, T) bool
        logits = self.score(x).squeeze(-1)
        valid = mask.bool()
        any_valid = valid.any(dim=1, keepdim=True)
        safe_mask = torch.where(any_valid, valid, torch.ones_like(valid))
        logits = logits.masked_fill(~safe_mask, -1e9)
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)
        pooled = (x * weights).sum(dim=1)
        pooled = pooled * any_valid.to(x.dtype)
        return self.proj(pooled)


class ModalityBranch(nn.Module):
    """Per-modality temporal branch combining optional spatial encoding, tsai backbone, and attention."""

    def __init__(
        self, name: str, in_channels: int, seq_len: int, out_dim: int, *,
        arch: str = "InceptionTimePlus", filters: int = 64, depth: int = 3,
        kernel_size: int = 7, dropout: float = 0.1, conv_dropout: float = 0.0,
        num_heads: int = 4, use_batchnorm: bool = True, pooling_mode: str = "default",
        attention_units: int = 64, tof_spatial_mode: str = "auto",
        tof_spatial_filters: int = 32, tof_spatial_kernel: int = 3,
        activation: str = "gelu",
    ):
        super().__init__()
        self.name = name
        self.spatial: Optional[ToFSpatialEncoder] = None
        eff_channels = int(in_channels)
        use_spatial = (
            name == "tof"
            and tof_spatial_mode in {"auto", "conv2d"}
            and in_channels >= int(np.prod(TOF_SHAPE))
            and (tof_spatial_mode == "conv2d" or in_channels >= int(np.prod(TOF_SHAPE)))
        )
        if name == "tof" and tof_spatial_mode == "conv2d" and in_channels < int(np.prod(TOF_SHAPE)):
            # Graceful fallback when raw ToF pixels were not extracted
            use_spatial = False
        if use_spatial:
            self.spatial = ToFSpatialEncoder(
                in_channels, spatial_filters=tof_spatial_filters,
                spatial_kernel=tof_spatial_kernel, use_batchnorm=use_batchnorm,
                activation=activation,
            )
            eff_channels = self.spatial.out_channels

        self.backbone = build_tsai_backbone(
            arch, eff_channels, out_dim, seq_len,
            filters=filters, depth=depth, kernel_size=kernel_size,
            dropout=dropout, conv_dropout=conv_dropout, num_heads=num_heads,
            use_batchnorm=use_batchnorm, pooling_mode=pooling_mode, activation=activation,
        )
        self.attn_pool: Optional[TemporalAttentionPool] = None
        if pooling_mode == "attention":
            self.attn_pool = TemporalAttentionPool(eff_channels, attention_units, out_dim, activation=activation)
        self.norm = nn.LayerNorm(int(out_dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C), mask: (B, T), present: (B,)
        m = mask.unsqueeze(-1).to(x.dtype)
        x = x * m
        if self.spatial is not None:
            x = self.spatial(x) * m
        # tsai expects (B, C, T)
        out = self.backbone(x.transpose(1, 2))
        if self.attn_pool is not None:
            out = out + self.attn_pool(x, mask)
        out = self.norm(out)
        return out * present.unsqueeze(-1).to(out.dtype)


class MultiBranchTSAINet(nn.Module):
    """Multimodal tsai network fusing IMU (100 Hz), Thermopile (20 Hz), ToF (20 Hz), and Static features."""

    def __init__(
        self, branch_specs: Mapping[str, Tuple[int, int]], static_dim: int, n_classes: int, *,
        arch: str = "InceptionTimePlus",
        imu_arch: Optional[str] = None, thm_arch: Optional[str] = None, tof_arch: Optional[str] = None,
        branch_filters: int = 64, branch_depth: int = 3, branch_kernel_size: int = 7,
        branch_dropout: float = 0.1, conv_dropout: float = 0.0, branch_units: int = 128,
        pooling_mode: str = "default", attention_units: int = 64,
        tof_spatial_mode: str = "auto", tof_spatial_filters: int = 32, tof_spatial_kernel: int = 3,
        static_units: int = 64, static_layers: int = 2, static_dropout: float = 0.1,
        fusion_type: str = "attention", fusion_units: int = 128, fusion_dropout: float = 0.1,
        attention_heads: int = 4, dense_units: int = 128, dense_layers: int = 2,
        dropout: float = 0.2, activation: str = "gelu", use_batchnorm: bool = True,
    ):
        super().__init__()
        self.branch_order = [name for name in MODALITIES if name in branch_specs]
        arch_map = {
            "imu": imu_arch or arch,
            "thm": thm_arch or arch,
            "tof": tof_arch or arch,
        }
        b_units = max(8, int(branch_units))
        self.branches = nn.ModuleDict()
        for name in self.branch_order:
            seq_len, n_ch = branch_specs[name]
            self.branches[name] = ModalityBranch(
                name, n_ch, seq_len, b_units,
                arch=arch_map[name], filters=branch_filters, depth=branch_depth,
                kernel_size=branch_kernel_size, dropout=branch_dropout,
                conv_dropout=conv_dropout, num_heads=attention_heads,
                use_batchnorm=use_batchnorm, pooling_mode=pooling_mode,
                attention_units=attention_units, tof_spatial_mode=tof_spatial_mode,
                tof_spatial_filters=tof_spatial_filters, tof_spatial_kernel=tof_spatial_kernel,
                activation=activation,
            )

        self.static_dim = int(static_dim)
        s_units = max(8, int(static_units)) if self.static_dim > 0 else 0
        if self.static_dim > 0 and int(static_layers) > 0:
            s_blocks: List[nn.Module] = []
            in_d = self.static_dim
            for _ in range(int(static_layers)):
                s_blocks.extend([
                    nn.Linear(in_d, s_units),
                    nn.LayerNorm(s_units),
                    _make_activation(activation),
                    nn.Dropout(float(static_dropout)) if static_dropout > 0 else nn.Identity(),
                ])
                in_d = s_units
            self.static_net: Optional[nn.Module] = nn.Sequential(*s_blocks)
            self.static_out_dim = s_units
        else:
            self.static_net = None
            self.static_out_dim = 0

        self.fusion_type = str(fusion_type)
        f_units = max(8, int(fusion_units))
        n_temporal = len(self.branch_order)
        has_static = self.static_out_dim > 0
        n_tokens = n_temporal + (1 if has_static else 0)
        concat_dim = n_temporal * b_units + self.static_out_dim + 3

        if self.fusion_type == "attention":
            heads = max(1, int(attention_heads))
            while f_units % heads != 0 and heads > 1:
                heads -= 1
            self.token_projs = nn.ModuleDict({
                name: nn.Linear(b_units, f_units) for name in self.branch_order
            })
            if has_static:
                self.token_projs["static"] = nn.Linear(self.static_out_dim, f_units)
            self.modality_emb = nn.Parameter(torch.randn(1, n_tokens, f_units) * 0.02)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=f_units, num_heads=heads,
                dropout=float(fusion_dropout), batch_first=True,
            )
            self.fusion_norm = nn.LayerNorm(f_units)
            self.fusion_proj = nn.Sequential(
                nn.Linear(f_units + 3, f_units),
                _make_activation(activation),
                nn.Dropout(float(fusion_dropout)) if fusion_dropout > 0 else nn.Identity(),
            )
        elif self.fusion_type == "gated":
            self.gate_net = nn.Sequential(
                nn.Linear(concat_dim, f_units),
                _make_activation(activation),
                nn.Linear(f_units, n_tokens),
                nn.Sigmoid(),
            )
            self.token_projs = nn.ModuleDict({
                name: nn.Linear(b_units, f_units) for name in self.branch_order
            })
            if has_static:
                self.token_projs["static"] = nn.Linear(self.static_out_dim, f_units)
            self.fusion_proj = nn.Sequential(
                nn.LayerNorm(f_units),
                _make_activation(activation),
                nn.Dropout(float(fusion_dropout)) if fusion_dropout > 0 else nn.Identity(),
            )
        elif self.fusion_type == "concat":
            self.fusion_proj = nn.Sequential(
                nn.Linear(concat_dim, f_units),
                nn.LayerNorm(f_units),
                _make_activation(activation),
                nn.Dropout(float(fusion_dropout)) if fusion_dropout > 0 else nn.Identity(),
            )
        else:
            raise InvalidParams(f"fusion_type must be one of {FUSION_TYPES}, got {fusion_type!r}")

        head_blocks: List[nn.Module] = []
        in_h = f_units
        d_units = max(8, int(dense_units))
        for layer_idx in range(max(1, int(dense_layers))):
            out_h = d_units if layer_idx == 0 else max(8, d_units // (2 if layer_idx > 1 else 1))
            head_blocks.extend([
                nn.Linear(in_h, out_h, bias=not use_batchnorm),
                nn.BatchNorm1d(out_h) if use_batchnorm else nn.LayerNorm(out_h),
                _make_activation(activation),
                nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            ])
            in_h = out_h
        head_blocks.append(nn.Linear(in_h, int(n_classes)))
        self.classifier = nn.Sequential(*head_blocks)

    def forward(
        self, branches: Mapping[str, torch.Tensor], masks: Mapping[str, torch.Tensor],
        present: torch.Tensor, static: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mod_index = {"imu": 0, "thm": 1, "tof": 2}
        embs: Dict[str, torch.Tensor] = {}
        pres_flags: List[torch.Tensor] = []
        for name in self.branch_order:
            j = mod_index[name]
            p_j = present[:, j].bool()
            embs[name] = self.branches[name](branches[name], masks[name], p_j)
            pres_flags.append(p_j)

        static_emb: Optional[torch.Tensor] = None
        if self.static_net is not None and static is not None and static.shape[1] > 0:
            static_emb = self.static_net(static)

        pres_float = present.to(next(iter(embs.values())).dtype)
        if self.fusion_type == "concat":
            parts = [embs[n] for n in self.branch_order]
            if static_emb is not None:
                parts.append(static_emb)
            parts.append(pres_float)
            fused = self.fusion_proj(torch.cat(parts, dim=-1))
        elif self.fusion_type == "attention":
            tokens = [self.token_projs[n](embs[n]) for n in self.branch_order]
            token_present = list(pres_flags)
            if static_emb is not None:
                tokens.append(self.token_projs["static"](static_emb))
                token_present.append(torch.ones_like(pres_flags[0], dtype=torch.bool))
            seq = torch.stack(tokens, dim=1) + self.modality_emb
            pad_mask = ~torch.stack(token_present, dim=1)
            # Ensure at least the first token is unmasked per row to avoid NaN softmax
            all_masked = pad_mask.all(dim=1)
            if all_masked.any():
                pad_mask = pad_mask.clone()
                pad_mask[all_masked, 0] = False
            attn_out, _ = self.cross_attn(seq, seq, seq, key_padding_mask=pad_mask, need_weights=False)
            seq = self.fusion_norm(seq + attn_out)
            valid_tok = (~pad_mask).unsqueeze(-1).to(seq.dtype)
            pooled = (seq * valid_tok).sum(dim=1) / valid_tok.sum(dim=1).clamp_min(1.0)
            fused = self.fusion_proj(torch.cat([pooled, pres_float], dim=-1))
        else:  # gated
            raw_parts = [embs[n] for n in self.branch_order]
            tokens = [self.token_projs[n](embs[n]) for n in self.branch_order]
            token_present = list(pres_flags)
            if static_emb is not None:
                raw_parts.append(static_emb)
                tokens.append(self.token_projs["static"](static_emb))
                token_present.append(torch.ones_like(pres_flags[0], dtype=torch.bool))
            raw_parts.append(pres_float)
            gates = self.gate_net(torch.cat(raw_parts, dim=-1))
            valid_tok = torch.stack(token_present, dim=1).to(gates.dtype)
            gates = gates * valid_tok
            stacked = torch.stack(tokens, dim=1)
            gated_sum = (stacked * gates.unsqueeze(-1)).sum(dim=1) / gates.sum(dim=1, keepdim=True).clamp_min(1e-6)
            fused = self.fusion_proj(gated_sum)

        return self.classifier(fused)


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional class weights and label smoothing."""

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("weight", weight)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(
            logits, targets, reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce).clamp(1e-7, 1.0 - 1e-7)
        focal = ((1.0 - pt) ** self.gamma) * ce
        if self.weight is not None:
            focal = focal * self.weight[targets]
        return focal.mean()


def _add_select_finetuning_sequences():
    """Attach ``select_finetuning_sequences`` to ``MultiModalSequenceExtractor`` for notebook compatibility."""
    if not hasattr(MultiModalSequenceExtractor, "select_finetuning_sequences"):
        def select_finetuning_sequences(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> pd.DataFrame:
            return X
        MultiModalSequenceExtractor.select_finetuning_sequences = select_finetuning_sequences  # type: ignore[attr-defined]


_add_select_finetuning_sequences()


class MultiBranchTSAIClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible multimodal time-series classifier built on ``tsai``, ``tsaug``, and ``tsfel``/``tsfresh`` + ``NOOP``.

    Forwards ``extractor__*`` (including ``extractor__noop__*``) and ``augmentor__*`` via
    ``get_params`` / ``set_params`` so a single ``Pipeline([('estimator', clf)])`` exposes
    every hyperparameter in one unified namespace.
    """

    def __init__(
        self,
        extractor: Optional[MultiModalSequenceExtractor] = None,
        augmentor: Optional[TSAugmentor] = None,
        # --- tsai backbone & branch architecture ---
        arch: str = "InceptionTimePlus",
        imu_arch: Optional[str] = None,
        thm_arch: Optional[str] = None,
        tof_arch: Optional[str] = None,
        branch_filters: int = 64,
        branch_depth: int = 3,
        branch_kernel_size: int = 7,
        branch_dropout: float = 0.1,
        conv_dropout: float = 0.0,
        branch_units: int = 128,
        pooling_mode: str = "default",
        attention_units: int = 64,
        tof_spatial_mode: str = "auto",
        tof_spatial_filters: int = 32,
        tof_spatial_kernel: int = 3,
        # --- static branch & multimodal fusion ---
        static_units: int = 64,
        static_layers: int = 2,
        static_dropout: float = 0.1,
        fusion_type: str = "attention",
        fusion_units: int = 128,
        fusion_dropout: float = 0.1,
        attention_heads: int = 4,
        # --- classification head ---
        dense_units: int = 128,
        dense_layers: int = 2,
        dropout: float = 0.2,
        activation: str = "gelu",
        use_batchnorm: bool = True,
        # --- optimization & regularization ---
        epochs: int = 40,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer_name: str = "adamw",
        momentum: float = 0.9,
        loss_name: str = "cross_entropy",
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
        class_weight_mode: Optional[str] = "balanced",
        use_lr_scheduler: bool = True,
        lr_scheduler_type: str = "cosine",
        lr_factor: float = 0.5,
        lr_patience: int = 5,
        min_lr: float = 1e-6,
        cosine_t_max: int = 40,
        warmup_epochs: int = 0,
        gradient_clip_norm: Optional[float] = 1.0,
        early_stopping_patience: int = 10,
        restore_best_weights: bool = True,
        # --- validation & evaluation ---
        validation_split: float = 0.15,
        validation_group_column: Optional[str] = None,
        sequence_aggregation: str = "mean_proba",
        primary_target: str = "bfrb",
        invalid_candidate_policy: str = "penalize",
        invalid_candidate_score: float = -1.0,
        candidate_failure_log: Optional[str] = None,
        verbose: int = 0,
        random_state: int = 42,
        device: str = "auto",
    ):
        self.extractor = extractor
        self.augmentor = augmentor
        self.arch = arch
        self.imu_arch = imu_arch
        self.thm_arch = thm_arch
        self.tof_arch = tof_arch
        self.branch_filters = branch_filters
        self.branch_depth = branch_depth
        self.branch_kernel_size = branch_kernel_size
        self.branch_dropout = branch_dropout
        self.conv_dropout = conv_dropout
        self.branch_units = branch_units
        self.pooling_mode = pooling_mode
        self.attention_units = attention_units
        self.tof_spatial_mode = tof_spatial_mode
        self.tof_spatial_filters = tof_spatial_filters
        self.tof_spatial_kernel = tof_spatial_kernel
        self.static_units = static_units
        self.static_layers = static_layers
        self.static_dropout = static_dropout
        self.fusion_type = fusion_type
        self.fusion_units = fusion_units
        self.fusion_dropout = fusion_dropout
        self.attention_heads = attention_heads
        self.dense_units = dense_units
        self.dense_layers = dense_layers
        self.dropout = dropout
        self.activation = activation
        self.use_batchnorm = use_batchnorm
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.momentum = momentum
        self.loss_name = loss_name
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.class_weight_mode = class_weight_mode
        self.use_lr_scheduler = use_lr_scheduler
        self.lr_scheduler_type = lr_scheduler_type
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.min_lr = min_lr
        self.cosine_t_max = cosine_t_max
        self.warmup_epochs = warmup_epochs
        self.gradient_clip_norm = gradient_clip_norm
        self.early_stopping_patience = early_stopping_patience
        self.restore_best_weights = restore_best_weights
        self.validation_split = validation_split
        self.validation_group_column = validation_group_column
        self.sequence_aggregation = sequence_aggregation
        self.primary_target = primary_target
        self.invalid_candidate_policy = invalid_candidate_policy
        self.invalid_candidate_score = invalid_candidate_score
        self.candidate_failure_log = candidate_failure_log
        self.verbose = verbose
        self.random_state = random_state
        self.device = device

    # -- nested parameter plumbing --------------------------------------------
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        params = super().get_params(deep=deep)
        if deep:
            if self.extractor is None:
                ext = MultiModalSequenceExtractor(noop=NoopOptions())
                params.update({f"extractor__{k}": v for k, v in ext.get_params(deep=True).items()})
            if self.augmentor is None:
                aug = TSAugmentor()
                params.update({f"augmentor__{k}": v for k, v in aug.get_params(deep=True).items()})
        return params

    def set_params(self, **params):
        params = dict(params)
        if any(k.startswith("extractor__") for k in params) and params.get("extractor", self.extractor) is None:
            params["extractor"] = MultiModalSequenceExtractor(noop=NoopOptions())
        if any(k.startswith("augmentor__") for k in params) and params.get("augmentor", self.augmentor) is None:
            params["augmentor"] = TSAugmentor()
        for key, val in list(params.items()):
            if isinstance(val, str) and (val.startswith("[") or val.startswith("{") or val.startswith("(")):
                params[key] = _structured(val)
        return super().set_params(**params)

    # -- validation -----------------------------------------------------------
    def _build_subcomponents(self) -> Tuple[MultiModalSequenceExtractor, TSAugmentor]:
        ext = clone(self.extractor) if self.extractor is not None else MultiModalSequenceExtractor(noop=NoopOptions())
        if ext.noop is None:
            ext.noop = NoopOptions()
        aug = clone(self.augmentor) if self.augmentor is not None else TSAugmentor(random_state=self.random_state)
        ext.validate()
        aug.validate()
        return ext, aug

    def validate(self):
        self._build_subcomponents()
        for name, val in (("arch", self.arch), ("imu_arch", self.imu_arch),
                          ("thm_arch", self.thm_arch), ("tof_arch", self.tof_arch)):
            if val is not None and val not in TSAI_ARCHS:
                raise InvalidParams(f"{name} must be one of {TSAI_ARCHS}, got {val!r}")
        if self.fusion_type not in FUSION_TYPES:
            raise InvalidParams(f"fusion_type must be one of {FUSION_TYPES}")
        if self.pooling_mode not in POOLING_MODES:
            raise InvalidParams(f"pooling_mode must be one of {POOLING_MODES}")
        if self.tof_spatial_mode not in TOF_SPATIAL_MODES:
            raise InvalidParams(f"tof_spatial_mode must be one of {TOF_SPATIAL_MODES}")
        if str(self.activation).lower() not in ACTIVATIONS:
            raise InvalidParams(f"activation must be one of {ACTIVATIONS}")
        if str(self.optimizer_name).lower() not in OPTIMIZERS:
            raise InvalidParams(f"optimizer_name must be one of {OPTIMIZERS}")
        if str(self.loss_name).lower() not in LOSS_NAMES:
            raise InvalidParams(f"loss_name must be one of {LOSS_NAMES}")
        if str(self.lr_scheduler_type).lower() not in LR_SCHEDULERS:
            raise InvalidParams(f"lr_scheduler_type must be one of {LR_SCHEDULERS}")
        if self.sequence_aggregation not in SEQUENCE_AGGREGATIONS:
            raise InvalidParams(f"sequence_aggregation must be one of {SEQUENCE_AGGREGATIONS}")
        if self.class_weight_mode not in {None, "none", "balanced"}:
            raise InvalidParams("class_weight_mode must be None, 'none', or 'balanced'")
        if int(self.epochs) < 1 or int(self.batch_size) < 1:
            raise InvalidParams("epochs and batch_size must be >= 1")
        if float(self.learning_rate) <= 0 or not np.isfinite(float(self.learning_rate)):
            raise InvalidParams("learning_rate must be positive and finite")
        if float(self.weight_decay) < 0:
            raise InvalidParams("weight_decay must be >= 0")
        for drop_name in ("branch_dropout", "conv_dropout", "static_dropout", "fusion_dropout", "dropout"):
            val = float(getattr(self, drop_name))
            if not 0.0 <= val < 1.0:
                raise InvalidParams(f"{drop_name} must be in [0, 1)")
        if not 0.0 <= float(self.label_smoothing) < 1.0:
            raise InvalidParams("label_smoothing must be in [0, 1)")
        if not 0.0 <= float(self.validation_split) < 0.9:
            raise InvalidParams("validation_split must be in [0, 0.9)")
        return self

    def _resolve_device(self) -> torch.device:
        if self.device in {None, "auto"}:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    # -- sequence label extraction & internal validation split ----------------
    def _extract_sequence_labels(self, X: pd.DataFrame, y: Any) -> Tuple[pd.Series, Optional[pd.Series]]:
        seq_col = self.extractor.sequence_col if self.extractor is not None else SEQ_COL
        seq_order = pd.Index(X[seq_col].astype(str).unique(), name=seq_col)
        group_series: Optional[pd.Series] = None
        g_col = self.validation_group_column

        if isinstance(y, pd.DataFrame):
            target_col = self.primary_target if self.primary_target in y.columns else [
                c for c in y.columns if c not in {seq_col, "subject", "is_target"}
            ][0]
            if seq_col in y.columns:
                grouped = y.groupby(y[seq_col].astype(str), sort=False)[target_col].first()
                labels = grouped.reindex(seq_order)
                if g_col and g_col in y.columns:
                    group_series = y.groupby(y[seq_col].astype(str), sort=False)[g_col].first().reindex(seq_order)
            elif len(y) == len(X):
                tmp = pd.DataFrame({seq_col: X[seq_col].astype(str).to_numpy(), target_col: y[target_col].to_numpy()})
                labels = tmp.groupby(seq_col, sort=False)[target_col].first().reindex(seq_order)
            else:
                labels = pd.Series(y[target_col].to_numpy(), index=seq_order[:len(y)])
        elif isinstance(y, pd.Series):
            if len(y) == len(X):
                tmp = pd.DataFrame({seq_col: X[seq_col].astype(str).to_numpy(), "y": y.to_numpy()})
                labels = tmp.groupby(seq_col, sort=False)["y"].first().reindex(seq_order)
            elif y.index.isin(seq_order).all():
                labels = y.reindex(seq_order)
            else:
                labels = pd.Series(y.to_numpy(), index=seq_order[:len(y)])
        else:
            arr = np.asarray(y)
            if len(arr) == len(X):
                tmp = pd.DataFrame({seq_col: X[seq_col].astype(str).to_numpy(), "y": arr})
                labels = tmp.groupby(seq_col, sort=False)["y"].first().reindex(seq_order)
            else:
                labels = pd.Series(arr, index=seq_order[:len(arr)])

        if group_series is None and g_col and g_col in X.columns:
            group_series = X.groupby(X[seq_col].astype(str), sort=False)[g_col].first().reindex(seq_order)
        if labels.isna().any():
            raise ValueError("Missing sequence labels for one or more sequences in X")
        return labels.astype(str), group_series

    def _split_validation_sequences(
        self, X: pd.DataFrame, y: Any
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        seq_col = self.extractor_.sequence_col
        labels, groups = self._extract_sequence_labels(X, y)
        seq_ids = labels.index
        val_frac = float(self.validation_split)
        if val_frac <= 0 or len(seq_ids) < 2:
            self.train_sequence_ids_ = seq_ids
            self.preprocessing_sequence_ids_ = seq_ids
            self.validation_sequence_ids_ = pd.Index([], name=seq_col)
            return X, X.iloc[0:0].copy(), labels, labels.iloc[0:0]

        if groups is not None and groups.nunique() >= 2:
            splitter = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=int(self.random_state))
            fit_idx, val_idx = next(splitter.split(seq_ids, labels, groups=groups))
        else:
            rng = np.random.default_rng(int(self.random_state))
            perm = rng.permutation(len(seq_ids))
            n_val = max(1, min(len(seq_ids) - 1, int(round(len(seq_ids) * val_frac))))
            val_idx = np.sort(perm[:n_val])
            fit_idx = np.sort(perm[n_val:])

        fit_seqs = seq_ids[fit_idx]
        val_seqs = seq_ids[val_idx]
        self.train_sequence_ids_ = fit_seqs
        self.preprocessing_sequence_ids_ = fit_seqs
        self.validation_sequence_ids_ = val_seqs

        seq_str = X[seq_col].astype(str)
        X_fit = X.loc[seq_str.isin(fit_seqs)].copy()
        X_val = X.loc[seq_str.isin(val_seqs)].copy()
        return X_fit, X_val, labels.loc[fit_seqs], labels.loc[val_seqs]

    # -- fit / predict / score ------------------------------------------------
    def _record_failure(self, exc: Exception):
        self.fit_failed_ = True
        self.fit_error_ = f"{type(exc).__name__}: {exc}"
        if self.candidate_failure_log:
            try:
                path = Path(self.candidate_failure_log)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"error": self.fit_error_, "time": time.time()}) + "\n")
            except Exception:
                pass
        warnings.warn(f"Candidate skipped: {self.fit_error_}", RuntimeWarning)

    def fit(self, X: pd.DataFrame, y: Any):
        self.fit_failed_ = False
        self.fit_error_ = None
        try:
            return self._fit_impl(X, y)
        except Exception as exc:
            if self.invalid_candidate_policy == "raise":
                raise
            self._record_failure(exc)
            return self

    def _fit_impl(self, X: pd.DataFrame, y: Any):
        self.validate()
        seed = int(self.random_state or 42)
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.extractor_, self.augmentor_ = self._build_subcomponents()
        X_fit, X_val, y_fit_seq, y_val_seq = self._split_validation_sequences(X, y)

        self.le_ = LabelEncoder()
        self.le_.fit(pd.concat([y_fit_seq, y_val_seq], axis=0))
        self.classes_ = self.le_.classes_
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise InvalidParams("Training partition must contain at least 2 distinct target classes")

        raw_fit = self.extractor_.raw_chunks(X_fit)
        feat_fit = self.extractor_.featurize(raw_fit)
        self.extractor_.fit_scalers(feat_fit)
        train_chunks = self.extractor_.scale(feat_fit)
        self.noop_feature_names_ = list(self.extractor_.noop_feature_names_)
        self.static_feature_names_ = list(self.extractor_.static_names_)

        # Augment training chunks if enabled and recompute static features on augmented signals
        if self.augmentor_.enabled():
            replicas = [train_chunks]
            for _ in range(int(self.augmentor_.n_copies)):
                aug_raw = self.augmentor_.augment_chunks(raw_fit, rng)
                aug_feat = self.extractor_.featurize(aug_raw)
                replicas.append(self.extractor_.scale(aug_feat))
            train_chunks = _concat_chunk_dicts(replicas)

        val_chunks = self.extractor_.transform_chunks(X_val) if len(X_val) else None

        y_train_win = self.le_.transform(
            pd.Series(train_chunks["sequence_ids"].astype(str)).map(y_fit_seq).to_numpy()
        )
        y_val_win = (
            self.le_.transform(pd.Series(val_chunks["sequence_ids"].astype(str)).map(y_val_seq).to_numpy())
            if val_chunks is not None and len(val_chunks["sequence_ids"]) else None
        )

        branch_specs = {
            name: (arr.shape[1], arr.shape[2])
            for name, arr in train_chunks["branches"].items()
        }
        static_dim = train_chunks["static"].shape[1]
        self.device_ = self._resolve_device()

        self.model_ = MultiBranchTSAINet(
            branch_specs, static_dim, n_classes,
            arch=self.arch, imu_arch=self.imu_arch, thm_arch=self.thm_arch, tof_arch=self.tof_arch,
            branch_filters=self.branch_filters, branch_depth=self.branch_depth,
            branch_kernel_size=self.branch_kernel_size, branch_dropout=self.branch_dropout,
            conv_dropout=self.conv_dropout, branch_units=self.branch_units,
            pooling_mode=self.pooling_mode, attention_units=self.attention_units,
            tof_spatial_mode=self.tof_spatial_mode, tof_spatial_filters=self.tof_spatial_filters,
            tof_spatial_kernel=self.tof_spatial_kernel, static_units=self.static_units,
            static_layers=self.static_layers, static_dropout=self.static_dropout,
            fusion_type=self.fusion_type, fusion_units=self.fusion_units,
            fusion_dropout=self.fusion_dropout, attention_heads=self.attention_heads,
            dense_units=self.dense_units, dense_layers=self.dense_layers,
            dropout=self.dropout, activation=self.activation, use_batchnorm=self.use_batchnorm,
        ).to(self.device_)

        class_weights_t: Optional[torch.Tensor] = None
        if self.class_weight_mode == "balanced":
            counts = np.bincount(y_train_win, minlength=n_classes).astype(np.float64)
            weights = np.where(counts > 0, len(y_train_win) / (n_classes * np.maximum(counts, 1.0)), 1.0)
            class_weights_t = torch.tensor(weights, dtype=torch.float32, device=self.device_)

        if str(self.loss_name).lower() == "focal":
            criterion: nn.Module = FocalLoss(
                gamma=self.focal_gamma, weight=class_weights_t,
                label_smoothing=self.label_smoothing,
            )
        else:
            criterion = nn.CrossEntropyLoss(
                weight=class_weights_t, label_smoothing=float(self.label_smoothing),
            )

        opt_name = str(self.optimizer_name).lower()
        if opt_name == "adamw":
            optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                self.model_.parameters(), lr=float(self.learning_rate), weight_decay=float(self.weight_decay),
            )
        elif opt_name == "adam":
            optimizer = torch.optim.Adam(
                self.model_.parameters(), lr=float(self.learning_rate), weight_decay=float(self.weight_decay),
            )
        else:
            optimizer = torch.optim.SGD(
                self.model_.parameters(), lr=float(self.learning_rate),
                momentum=float(self.momentum), weight_decay=float(self.weight_decay),
            )

        n_samples = len(y_train_win)
        batch_size = max(1, int(self.batch_size))
        steps_per_epoch = max(1, int(math.ceil(n_samples / batch_size)))
        total_epochs = max(1, int(self.epochs))
        sched_type = str(self.lr_scheduler_type).lower() if self.use_lr_scheduler else "none"
        scheduler: Any = None
        if sched_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, int(self.cosine_t_max or total_epochs)), eta_min=float(self.min_lr),
            )
        elif sched_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=float(self.lr_factor),
                patience=max(1, int(self.lr_patience)), min_lr=float(self.min_lr),
            )
        elif sched_type == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=float(self.learning_rate),
                epochs=total_epochs, steps_per_epoch=steps_per_epoch,
            )

        self.history_ = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
        best_val_loss = float("inf")
        best_state: Optional[Dict[str, torch.Tensor]] = None
        self.best_epoch_ = 1
        patience_left = int(self.early_stopping_patience)
        present_col_idx = {
            name: train_chunks["static_names"].index(f"present_{name}")
            for name in MODALITIES
            if f"present_{name}" in train_chunks["static_names"]
        }

        for epoch in range(1, total_epochs + 1):
            self.model_.train()
            if int(self.warmup_epochs) > 0 and epoch <= int(self.warmup_epochs) and sched_type != "onecycle":
                warm_lr = float(self.learning_rate) * (epoch / float(self.warmup_epochs))
                for pg in optimizer.param_groups:
                    pg["lr"] = warm_lr

            order = rng.permutation(n_samples)
            epoch_loss = 0.0
            correct = 0
            total = 0

            for start in range(0, n_samples, batch_size):
                idx = order[start:start + batch_size]
                # Batch-Norm requires >= 2 samples in training mode
                if len(idx) == 1 and n_samples > 1:
                    idx = np.r_[idx, order[0]]
                b_branches, b_masks, b_present, b_static = self._slice_batch(
                    train_chunks, idx, rng=rng, apply_sensor_drop=True, present_col_idx=present_col_idx,
                )
                b_y = torch.as_tensor(y_train_win[idx], dtype=torch.long, device=self.device_)

                optimizer.zero_grad(set_to_none=True)
                logits = self.model_(b_branches, b_masks, b_present, b_static)
                loss = criterion(logits, b_y)
                loss.backward()
                if self.gradient_clip_norm is not None and float(self.gradient_clip_norm) > 0:
                    nn.utils.clip_grad_norm_(self.model_.parameters(), float(self.gradient_clip_norm))
                optimizer.step()
                if sched_type == "onecycle" and scheduler is not None:
                    scheduler.step()

                bs = len(idx)
                epoch_loss += float(loss.detach().cpu().item()) * bs
                preds = logits.detach().argmax(dim=-1)
                correct += int((preds == b_y).sum().cpu().item())
                total += bs

            train_loss = epoch_loss / max(1, total)
            train_acc = correct / max(1, total)

            if val_chunks is not None and y_val_win is not None and len(y_val_win):
                val_loss, val_acc = self._evaluate_chunks(val_chunks, y_val_win, criterion)
            else:
                val_loss, val_acc = train_loss, train_acc

            self.history_["loss"].append(float(train_loss))
            self.history_["val_loss"].append(float(val_loss))
            self.history_["accuracy"].append(float(train_acc))
            self.history_["val_accuracy"].append(float(val_acc))

            if sched_type == "cosine" and scheduler is not None and epoch > int(self.warmup_epochs):
                scheduler.step()
            elif sched_type == "plateau" and scheduler is not None:
                scheduler.step(val_loss)

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                self.best_epoch_ = epoch
                if self.restore_best_weights:
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
                patience_left = int(self.early_stopping_patience)
            elif int(self.early_stopping_patience) > 0:
                patience_left -= 1
                if patience_left <= 0:
                    break

        self.epochs_run_ = len(self.history_["loss"])
        if self.restore_best_weights and best_state is not None:
            self.model_.load_state_dict({k: v.to(self.device_) for k, v in best_state.items()})
        self.model_.eval()
        return self

    def _slice_batch(
        self, chunks: Mapping[str, Any], idx: np.ndarray, *,
        rng: Optional[np.random.Generator] = None, apply_sensor_drop: bool = False,
        present_col_idx: Optional[Mapping[str, int]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        present = chunks["present"][idx].copy()
        static = chunks["static"][idx].copy()
        branches_np = {k: v[idx] for k, v in chunks["branches"].items()}

        if apply_sensor_drop and rng is not None and (
            self.augmentor_.sensor_drop_prob > 0 or self.augmentor_.sensor_drop_joint_prob > 0
        ):
            keep = self.augmentor_.sample_modality_keep(len(idx), rng)
            present &= keep
            for j, name in ((1, "thm"), (2, "tof")):
                dropped = ~keep[:, j]
                if dropped.any():
                    if name in branches_np:
                        branches_np[name] = branches_np[name].copy()
                        branches_np[name][dropped] = 0.0
                    if present_col_idx and name in present_col_idx:
                        static[dropped, present_col_idx[name]] = 0.0

        b_branches = {
            k: torch.as_tensor(v, dtype=torch.float32, device=self.device_)
            for k, v in branches_np.items()
        }
        b_masks = {
            k: torch.as_tensor(v[idx], dtype=torch.bool, device=self.device_)
            for k, v in chunks["masks"].items()
        }
        b_present = torch.as_tensor(present, dtype=torch.bool, device=self.device_)
        b_static = torch.as_tensor(static, dtype=torch.float32, device=self.device_)
        return b_branches, b_masks, b_present, b_static

    def _evaluate_chunks(self, chunks: Mapping[str, Any], y_win: np.ndarray, criterion: nn.Module) -> Tuple[float, float]:
        self.model_.eval()
        n = len(y_win)
        bs = max(1, int(self.batch_size))
        total_loss = 0.0
        correct = 0
        with torch.no_grad():
            for start in range(0, n, bs):
                idx = np.arange(start, min(n, start + bs))
                b_branches, b_masks, b_present, b_static = self._slice_batch(chunks, idx)
                b_y = torch.as_tensor(y_win[idx], dtype=torch.long, device=self.device_)
                logits = self.model_(b_branches, b_masks, b_present, b_static)
                loss = criterion(logits, b_y)
                total_loss += float(loss.cpu().item()) * len(idx)
                correct += int((logits.argmax(dim=-1) == b_y).sum().cpu().item())
        return total_loss / max(1, n), correct / max(1, n)

    def get_history_dict(self) -> Dict[str, List[float]]:
        return dict(getattr(self, "history_", {}))

    def predict_window_proba(self, chunks: Mapping[str, Any]) -> np.ndarray:
        """Predict class probabilities ``(N_windows, n_classes)`` for pre-extracted chunks."""
        if getattr(self, "fit_failed_", False) or not hasattr(self, "model_"):
            raise RuntimeError(f"Model was not fitted successfully: {getattr(self, 'fit_error_', 'unfitted')}")
        self.model_.eval()
        n = len(chunks["sequence_ids"])
        if n == 0:
            return np.zeros((0, len(self.classes_)), dtype=np.float32)
        bs = max(1, int(self.batch_size))
        out = []
        with torch.no_grad():
            for start in range(0, n, bs):
                idx = np.arange(start, min(n, start + bs))
                b_branches, b_masks, b_present, b_static = self._slice_batch(chunks, idx)
                logits = self.model_(b_branches, b_masks, b_present, b_static)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                out.append(probs)
        return np.vstack(out).astype(np.float32)

    def aggregate_window_proba(
        self, window_proba: np.ndarray, sequence_ids: np.ndarray,
        seq_order: Optional[Sequence[str]] = None,
    ) -> Tuple[pd.Index, np.ndarray]:
        """Aggregate window probabilities into one probability vector per sequence."""
        seq_str = pd.Series(sequence_ids.astype(str))
        if seq_order is None:
            unique_seqs = pd.Index(seq_str.unique(), name=self.extractor_.sequence_col)
        else:
            unique_seqs = pd.Index([str(s) for s in seq_order], name=self.extractor_.sequence_col)
        n_classes = window_proba.shape[1]
        agg = np.zeros((len(unique_seqs), n_classes), dtype=np.float32)
        groups = seq_str.groupby(seq_str, sort=False).indices
        for i, seq_id in enumerate(unique_seqs):
            pos = groups.get(str(seq_id))
            if pos is None or len(pos) == 0:
                agg[i] = 1.0 / max(1, n_classes)
                continue
            sub = window_proba[pos]
            if self.sequence_aggregation == "last_window":
                agg[i] = sub[-1]
            elif self.sequence_aggregation == "majority_vote":
                votes = np.bincount(sub.argmax(axis=1), minlength=n_classes).astype(np.float32)
                agg[i] = (votes + 1e-3 * sub.mean(axis=0)) / (votes.sum() + 1e-3)
            else:
                agg[i] = sub.mean(axis=0)
        return unique_seqs, agg

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        seq_col = self.extractor_.sequence_col
        seq_order = X[seq_col].astype(str).unique()
        chunks = self.extractor_.transform_chunks(X)
        win_proba = self.predict_window_proba(chunks)
        _, seq_proba = self.aggregate_window_proba(win_proba, chunks["sequence_ids"], seq_order=seq_order)
        return seq_proba

    def predict(self, X: pd.DataFrame) -> pd.Series:
        seq_col = self.extractor_.sequence_col
        seq_order = pd.Index(X[seq_col].astype(str).unique(), name=seq_col)
        proba = self.predict_proba(X)
        labels = self.le_.inverse_transform(proba.argmax(axis=1))
        return pd.Series(labels, index=seq_order, name=self.primary_target)

    def score(self, X: pd.DataFrame, y: Any, sample_weight=None) -> float:
        if getattr(self, "fit_failed_", False):
            return float(self.invalid_candidate_score)
        try:
            preds = self.predict(X)
            labels, _ = self._extract_sequence_labels(X, y)
            labels = labels.reindex(preds.index)
            if self.primary_target == "bfrb":
                return float(competition_score(labels, preds)["competition_score"])
            return float(f1_score(labels, preds, average="macro", zero_division=0))
        except Exception as exc:
            if self.invalid_candidate_policy == "raise":
                raise
            warnings.warn(f"Candidate scoring failed: {exc}", RuntimeWarning)
            return float(self.invalid_candidate_score)


def _concat_chunk_dicts(dicts: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if len(dicts) == 1:
        return dict(dicts[0])
    first = dicts[0]
    out = dict(first)
    out["branches"] = {
        k: np.concatenate([d["branches"][k] for d in dicts], axis=0)
        for k in first["branches"]
    }
    out["masks"] = {
        k: np.concatenate([d["masks"][k] for d in dicts], axis=0)
        for k in first["masks"]
    }
    for key in ("present", "static", "sequence_ids", "window_index", "obs_axes", "q_obs", "row_present"):
        if key in first:
            out[key] = np.concatenate([d[key] for d in dicts], axis=0)
    for key in ("obs_thm", "obs_tof"):
        if key in first:
            out[key] = np.concatenate([d[key] for d in dicts], axis=0)
    return out


# Alias for backward compatibility
AdvancedMultiBranchSequenceClassifier = MultiBranchTSAIClassifier


# ---------------------------------------------------------------------------
# 7. Competition Scorer (Original + 50% Hidden-Test Nullification Simulation)
# ---------------------------------------------------------------------------

def competition_score(
    y_true_gesture: Any, y_pred: Any,
    y_true_binary: Optional[Any] = None,
    target_only_macro: bool = True,
) -> Dict[str, float]:
    """Compute the exact CMI competition metric: ``(binary_f1 + gesture_f1) / 2.0``.

    * ``binary_f1`` is the binary F1 for BFRB (``!= 'non_bfrb'``) vs everyday (``'non_bfrb'``).
    * ``gesture_f1`` is the macro F1 across the target BFRB gesture classes.
    """
    if isinstance(y_true_gesture, pd.Series) and isinstance(y_pred, pd.Series) and y_true_gesture.index.equals(y_pred.index):
        y_true_s = y_true_gesture.astype(str)
        y_pred_s = y_pred.astype(str)
    elif isinstance(y_true_gesture, pd.Series) and isinstance(y_pred, pd.Series) and set(y_pred.index).issubset(set(y_true_gesture.index)):
        y_pred_s = y_pred.astype(str)
        y_true_s = y_true_gesture.reindex(y_pred.index).astype(str)
    else:
        y_true_s = pd.Series(np.asarray(y_true_gesture)).astype(str)
        y_pred_s = pd.Series(np.asarray(y_pred)).astype(str)

    pred_binary = (y_pred_s != "non_bfrb").astype(int)
    if y_true_binary is not None:
        true_binary = pd.Series(np.asarray(y_true_binary), index=y_true_s.index).astype(int)
    else:
        true_binary = (y_true_s != "non_bfrb").astype(int)

    binary_f1 = float(f1_score(true_binary, pred_binary, average="binary", pos_label=1, zero_division=0))

    if target_only_macro and (true_binary == 1).any():
        mask = true_binary == 1
        target_classes = np.unique(y_true_s[mask])
        gesture_f1 = float(f1_score(
            y_true_s[mask], y_pred_s[mask],
            labels=target_classes, average="macro", zero_division=0,
        ))
    else:
        gesture_f1 = float(f1_score(y_true_s, y_pred_s, average="macro", zero_division=0))

    comp = (binary_f1 + gesture_f1) / 2.0
    return {
        "competition_score": float(comp),
        "binary_f1": float(binary_f1),
        "gesture_f1": float(gesture_f1),
    }


def simulate_hidden_test_nullification(
    X_df: pd.DataFrame,
    null_fraction: float = 0.5,
    random_state: int = 42,
    sequence_col: str = SEQ_COL,
) -> pd.DataFrame:
    """Simulate the Kaggle hidden test set by nullifying Thermopile and ToF for 50% of sequences.

    Deterministically selects ``round(null_fraction * n_sequences)`` sequences in ``X_df``
    and replaces all ``thm_*`` and ``tof_*`` columns for those sequences with ``np.nan``.
    """
    if not 0.0 <= float(null_fraction) <= 1.0:
        raise ValueError("null_fraction must be in [0, 1]")
    out = X_df.copy()
    seq_ids = pd.Index(out[sequence_col].astype(str).unique())
    n_seq = len(seq_ids)
    n_null = int(round(n_seq * float(null_fraction)))
    if 0 < float(null_fraction) < 1.0 and n_seq >= 2:
        n_null = max(1, min(n_seq - 1, n_null))
    rng = np.random.default_rng(int(random_state))
    chosen_idx = np.sort(rng.choice(n_seq, size=n_null, replace=False)) if n_null > 0 else np.empty(0, dtype=int)
    null_seqs = set(seq_ids[chosen_idx])
    sensor_cols = [c for c in out.columns if c.startswith("thm_") or c.startswith("tof_")]
    if null_seqs and sensor_cols:
        mask = out[sequence_col].astype(str).isin(null_seqs)
        out.loc[mask, sensor_cols] = np.nan
    out.attrs["nullified_sequence_ids"] = sorted(null_seqs)
    out.attrs["null_fraction"] = float(null_fraction)
    return out


def _sequence_level_truth(y_df: Any, target_col: str, seq_index: Optional[pd.Index] = None) -> pd.DataFrame:
    if isinstance(y_df, pd.DataFrame):
        if SEQ_COL in y_df.columns:
            seq_y = y_df.groupby(y_df[SEQ_COL].astype(str), sort=False).first()
        else:
            seq_y = y_df.copy()
            seq_y.index = seq_y.index.astype(str)
    elif isinstance(y_df, pd.Series):
        seq_y = pd.DataFrame({target_col: y_df.astype(str)})
        seq_y.index = seq_y.index.astype(str)
    else:
        arr = np.asarray(y_df)
        idx = seq_index if seq_index is not None else pd.Index([str(i) for i in range(len(arr))])
        seq_y = pd.DataFrame({target_col: arr}, index=idx.astype(str))
    if seq_index is not None:
        seq_y = seq_y.reindex(seq_index.astype(str))
    return seq_y


def evaluate_holdout(
    y_test_df: Any,
    y_pred: Any,
    target_col: str = "bfrb",
    verbose: bool = True,
    *,
    model: Optional[Any] = None,
    X_test: Optional[pd.DataFrame] = None,
    simulate_hidden_test: bool = True,
    null_fraction: float = 0.5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Evaluate holdout predictions with both the normal competition score and the 50%-nullified hidden-test simulation."""
    pred_index = y_pred.index if isinstance(y_pred, pd.Series) else None
    seq_y = _sequence_level_truth(y_test_df, target_col, seq_index=pred_index)
    y_true = seq_y[target_col].astype(str)
    y_pred_s = pd.Series(np.asarray(y_pred), index=y_true.index, name=target_col).astype(str)
    y_true_bin = seq_y["is_target"].astype(int) if "is_target" in seq_y.columns else None

    macro_f1 = float(f1_score(y_true, y_pred_s, average="macro", zero_division=0))
    results: Dict[str, Any] = {"macro_f1": macro_f1}

    if target_col == "bfrb":
        comp = competition_score(y_true, y_pred_s, y_true_binary=y_true_bin)
        results.update(comp)
    else:
        results.update({
            "competition_score": macro_f1,
            "binary_f1": macro_f1,
            "gesture_f1": macro_f1,
        })

    if simulate_hidden_test and model is not None and X_test is not None and len(X_test):
        X_null = simulate_hidden_test_nullification(
            X_test, null_fraction=null_fraction, random_state=random_state,
        )
        y_pred_null = model.predict(X_null)
        seq_y_null = _sequence_level_truth(
            y_test_df, target_col,
            seq_index=y_pred_null.index if isinstance(y_pred_null, pd.Series) else pred_index,
        )
        y_true_null = seq_y_null[target_col].astype(str)
        y_pred_null_s = pd.Series(np.asarray(y_pred_null), index=y_true_null.index, name=target_col).astype(str)
        y_bin_null = seq_y_null["is_target"].astype(int) if "is_target" in seq_y_null.columns else None
        hidden_macro = float(f1_score(y_true_null, y_pred_null_s, average="macro", zero_division=0))
        if target_col == "bfrb":
            hidden_comp = competition_score(y_true_null, y_pred_null_s, y_true_binary=y_bin_null)
        else:
            hidden_comp = {
                "competition_score": hidden_macro,
                "binary_f1": hidden_macro,
                "gesture_f1": hidden_macro,
            }
        results.update({
            "hidden_test_competition_score": float(hidden_comp["competition_score"]),
            "hidden_test_binary_f1": float(hidden_comp["binary_f1"]),
            "hidden_test_gesture_f1": float(hidden_comp["gesture_f1"]),
            "hidden_test_macro_f1": float(hidden_macro),
            "nullified_sequence_ids": list(X_null.attrs.get("nullified_sequence_ids", [])),
            "nullified_sequence_count": len(X_null.attrs.get("nullified_sequence_ids", [])),
            "total_holdout_sequences": int(X_test[SEQ_COL].nunique()) if SEQ_COL in X_test else len(y_true_null),
            "hidden_test_null_fraction": float(null_fraction),
        })

    if verbose:
        print(f"Holdout Normal Competition Score: {results['competition_score']:.4f} "
              f"(Binary F1: {results['binary_f1']:.4f}, Gesture Macro F1: {results['gesture_f1']:.4f}, "
              f"Overall Macro F1: {results['macro_f1']:.4f})")
        if "hidden_test_competition_score" in results:
            print(f"Holdout Hidden-Test Competition Score (50% ToF/Thermopile Nullified, "
                  f"{results['nullified_sequence_count']}/{results['total_holdout_sequences']} seqs): "
                  f"{results['hidden_test_competition_score']:.4f} "
                  f"(Binary F1: {results['hidden_test_binary_f1']:.4f}, "
                  f"Gesture Macro F1: {results['hidden_test_gesture_f1']:.4f})")
        print(classification_report(y_true, y_pred_s, zero_division=0))

    return results


def make_competition_scorer(
    target_col: str = "bfrb",
    simulate_hidden_test: bool = False,
    null_fraction: float = 0.5,
    random_state: int = 42,
):
    """Build an sklearn scorer callable ``(estimator, X, y) -> float``."""
    def _scorer(estimator, X, y, sample_weight=None) -> float:
        est = estimator.named_steps["estimator"] if hasattr(estimator, "named_steps") and "estimator" in estimator.named_steps else estimator
        if getattr(est, "fit_failed_", False):
            return float(getattr(est, "invalid_candidate_score", -1.0))
        try:
            X_eval = (
                simulate_hidden_test_nullification(X, null_fraction=null_fraction, random_state=random_state)
                if simulate_hidden_test else X
            )
            preds = estimator.predict(X_eval)
            seq_y = _sequence_level_truth(y, target_col, seq_index=preds.index if isinstance(preds, pd.Series) else None)
            y_true = seq_y[target_col].astype(str)
            y_bin = seq_y["is_target"].astype(int) if "is_target" in seq_y.columns else None
            if target_col == "bfrb":
                return float(competition_score(y_true, preds, y_true_binary=y_bin)["competition_score"])
            return float(f1_score(y_true, preds, average="macro", zero_division=0))
        except Exception as exc:
            if getattr(est, "invalid_candidate_policy", "penalize") == "raise":
                raise
            warnings.warn(f"Scorer failed: {exc}", RuntimeWarning)
            return float(getattr(est, "invalid_candidate_score", -1.0))
    return _scorer


def candidate_score(estimator, X, y, sample_weight=None) -> float:
    """Default CV scorer delegating to the fitted estimator's competition score."""
    est = estimator.named_steps["estimator"] if hasattr(estimator, "named_steps") and "estimator" in estimator.named_steps else estimator
    return float(est.score(X, y, sample_weight=sample_weight))


# ---------------------------------------------------------------------------
# 8. Search Space (ONE Unified Declarative Block -> Grid / Bayes / Evo / Best)
# ---------------------------------------------------------------------------

class ECat:
    """Categorical parameter descriptor for evolutionary search."""

    def __init__(self, choices: Sequence[Any]):
        self.choices = list(choices)
        if not self.choices:
            raise ValueError("ECat requires at least one choice")

    def sample(self, rng: np.random.Generator):
        idx = int(rng.integers(0, len(self.choices)))
        return self.choices[idx]

    def mutate(self, current_val: Any, rng: np.random.Generator):
        if len(self.choices) <= 1:
            return self.choices[0]
        other = [c for c in self.choices if c != current_val]
        if not other:
            return self.sample(rng)
        return other[int(rng.integers(0, len(other)))]

    def __repr__(self) -> str:
        return f"ECat({self.choices!r})"


class EInt:
    """Integer range descriptor for evolutionary search."""

    def __init__(self, low: int, high: int, step_scale: float = 0.2):
        self.low = int(low)
        self.high = int(high)
        if self.low > self.high:
            raise ValueError("EInt requires low <= high")
        self.step_scale = float(step_scale)

    def sample(self, rng: np.random.Generator) -> int:
        if self.low == self.high:
            return self.low
        return int(rng.integers(self.low, self.high + 1))

    def mutate(self, current_val: Any, rng: np.random.Generator) -> int:
        if self.low == self.high:
            return self.low
        span = max(1, int(round((self.high - self.low) * self.step_scale)))
        delta = int(rng.integers(-span, span + 1))
        if delta == 0:
            delta = 1 if rng.random() < 0.5 else -1
        return int(np.clip(int(current_val) + delta, self.low, self.high))

    def __repr__(self) -> str:
        return f"EInt({self.low}, {self.high})"


class EFloat:
    """Continuous float range descriptor for evolutionary search."""

    def __init__(
        self,
        low: float,
        high: float,
        distribution: str = "uniform",
        sigma_frac: float = 0.15,
        prior: Optional[str] = None,
    ):
        self.low = float(low)
        self.high = float(high)
        if self.low > self.high:
            raise ValueError("EFloat requires low <= high")
        eff_dist = prior if prior is not None else distribution
        if eff_dist not in {"uniform", "log-uniform"}:
            raise ValueError("EFloat distribution must be 'uniform' or 'log-uniform'")
        if eff_dist == "log-uniform" and self.low <= 0:
            raise ValueError("log-uniform EFloat requires low > 0")
        self.distribution = eff_dist
        self.sigma_frac = float(sigma_frac)

    def sample(self, rng: np.random.Generator) -> float:
        if self.low == self.high:
            return self.low
        if self.distribution == "log-uniform":
            return float(np.exp(rng.uniform(np.log(self.low), np.log(self.high))))
        return float(rng.uniform(self.low, self.high))

    def mutate(self, current_val: Any, rng: np.random.Generator) -> float:
        if self.low == self.high:
            return self.low
        val = float(current_val)
        if self.distribution == "log-uniform":
            lo, hi = np.log(self.low), np.log(self.high)
            cur = np.clip(np.log(max(val, self.low)), lo, hi)
            mutated = cur + rng.normal(0.0, (hi - lo) * self.sigma_frac)
            return float(np.exp(np.clip(mutated, lo, hi)))
        mutated = val + rng.normal(0.0, (self.high - self.low) * self.sigma_frac)
        return float(np.clip(mutated, self.low, self.high))

    def __repr__(self) -> str:
        return f"EFloat({self.low}, {self.high}, {self.distribution!r})"


def _encode_skopt_item(val: Any) -> Any:
    """Encode non-scalar objects (tuples, lists, dicts) as JSON strings for skopt Categorical."""
    if isinstance(val, (tuple, list, dict)):
        return json.dumps(val)
    return val


def _decode_skopt_params(params: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, np.generic):
            v = v.item()
        out[k] = _structured(v)
    return out


class ParamSpec:
    """Single-source hyperparameter declaration for Best, Grid, Bayesian, and Evolutionary modes.

    Parameters
    ----------
    default : Any
        Value placed in ``BEST_PARAMS``.
    grid : Sequence[Any]
        Candidate list placed in ``GRID_PARAM_SPACE``.
    kind : {'cat', 'int', 'float', 'log_float'}
        Dimension kind for Bayesian (``skopt``) and Evolutionary (``ECat``/``EInt``/``EFloat``) spaces.
    bounds : Optional[Tuple[float, float]]
        Explicit ``(low, high)`` range for ``int``, ``float``, or ``log_float`` dimensions.
        Defaults to ``(min(grid), max(grid))`` when omitted.
    choices : Optional[Sequence[Any]]
        Explicit categorical choices for Bayesian/Evolutionary spaces when different from ``grid``.
    """

    def __init__(
        self,
        default: Any,
        grid: Sequence[Any],
        kind: str = "cat",
        bounds: Optional[Tuple[Any, Any]] = None,
        choices: Optional[Sequence[Any]] = None,
    ):
        if kind not in {"cat", "int", "float", "log_float"}:
            raise ValueError(f"ParamSpec kind must be 'cat', 'int', 'float', or 'log_float'; got {kind!r}")
        self.default = default
        self.grid = list(grid)
        if not self.grid:
            raise ValueError("ParamSpec grid must contain at least one value")
        self.kind = kind
        self.bounds = bounds
        self.choices = list(choices) if choices is not None else list(self.grid)

    def _numeric_bounds(self) -> Tuple[float, float]:
        if self.bounds is not None:
            return float(self.bounds[0]), float(self.bounds[1])
        nums = [float(x) for x in self.grid if x is not None]
        return min(nums), max(nums)

    def to_grid(self) -> List[Any]:
        return list(self.grid)

    def to_evo(self):
        if self.kind == "cat":
            return ECat(self.choices)
        lo, hi = self._numeric_bounds()
        if lo == hi:
            return ECat([int(lo) if self.kind == "int" else float(lo)])
        if self.kind == "int":
            return EInt(int(round(lo)), int(round(hi)))
        dist = "log-uniform" if self.kind == "log_float" else "uniform"
        return EFloat(float(lo), float(hi), distribution=dist)

    def to_skopt(self, skopt_available: bool = SKOPT_AVAILABLE):
        if not skopt_available or SkCategorical is None:
            return None
        if self.kind == "cat":
            encoded = [_encode_skopt_item(c) for c in self.choices]
            return SkCategorical(encoded)
        lo, hi = self._numeric_bounds()
        if lo == hi:
            return SkCategorical([int(round(lo)) if self.kind == "int" else float(lo)])
        if self.kind == "int":
            return SkInteger(int(round(lo)), int(round(hi)))
        prior = "log-uniform" if self.kind == "log_float" else "uniform"
        return SkReal(float(lo), float(hi), prior=prior)


PS = Spec = ParamSpec


def compile_unified_search_space(
    unified_space: Mapping[str, Any],
    *,
    skopt_available: bool = SKOPT_AVAILABLE,
) -> Tuple[Dict[str, Any], Dict[str, List[Any]], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Compile ONE consolidated ``UNIFIED_PARAM_SPACE`` dictionary into all search views.

    Returns ``(BEST_PARAMS, GRID_PARAM_SPACE, BAYESIAN_PARAM_SPACE, EVOLUTIONARY_PARAM_SPACE, EVOLUTIONARY_SEARCH_SPACE)``.
    """
    best_params: Dict[str, Any] = {}
    grid_space: Dict[str, List[Any]] = {}
    bayes_space: Dict[str, Any] = {}
    evo_space: Dict[str, Any] = {}

    for key, raw_spec in unified_space.items():
        if isinstance(raw_spec, ParamSpec):
            spec = raw_spec
        elif isinstance(raw_spec, Mapping):
            spec = ParamSpec(
                default=raw_spec["default"],
                grid=raw_spec.get("grid", [raw_spec["default"]]),
                kind=raw_spec.get("kind", "cat"),
                bounds=raw_spec.get("bounds"),
                choices=raw_spec.get("choices"),
            )
        elif isinstance(raw_spec, (list, tuple)):
            spec = ParamSpec(default=raw_spec[0], grid=list(raw_spec), kind="cat")
        else:
            spec = ParamSpec(default=raw_spec, grid=[raw_spec], kind="cat")

        best_params[key] = spec.default
        grid_space[key] = spec.to_grid()
        evo_space[key] = spec.to_evo()
        if skopt_available and SkCategorical is not None:
            b_dim = spec.to_skopt(skopt_available=True)
            if b_dim is not None:
                bayes_space[key] = b_dim

    return best_params, grid_space, bayes_space, evo_space, dict(evo_space)


def prepare_bayesian_space(space: Mapping[str, Any], skopt_available: bool = SKOPT_AVAILABLE) -> Dict[str, Any]:
    if not skopt_available or SkCategorical is None:
        return {}
    out: Dict[str, Any] = {}
    for k, v in space.items():
        if isinstance(v, ParamSpec):
            out[k] = v.to_skopt(skopt_available=True)
        elif isinstance(v, (list, tuple)):
            out[k] = SkCategorical([_encode_skopt_item(x) for x in v])
        else:
            out[k] = v
    return out


def prepare_evolutionary_space(space: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in space.items():
        if isinstance(v, ParamSpec):
            out[k] = v.to_evo()
        elif isinstance(v, (ECat, EInt, EFloat)):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = ECat(v)
        else:
            out[k] = ECat([v])
    return out


def inactive_parameters_for_candidate(params: Mapping[str, Any]) -> set:
    """Return the set of full parameter keys that have no effect under ``params``."""
    p = {
        (k[len("estimator__"):] if k.startswith("estimator__") else k): _structured(v)
        for k, v in params.items()
    }
    prefix = "estimator__" if any(k.startswith("estimator__") for k in params) else ""
    inactive: set = set()

    # 1. NOOP inactive tuning parameters
    noop_kwargs = {
        k[len("extractor__noop__"):]: v
        for k, v in p.items()
        if k.startswith("extractor__noop__")
    }
    try:
        noop_opts = NoopOptions(**noop_kwargs)
        for name in inactive_noop_parameters(noop_opts):
            inactive.add(f"{prefix}extractor__{name}")
    except Exception:
        pass

    # 2. Static backend (tsfel vs tsfresh vs none)
    backend = p.get("extractor__static_backend", "tsfel")
    if backend in {"none", None}:
        for name in ("static_channels", "tsfel_domains", "tsfel_features", "tsfresh_profile", "tsfresh_fft_bands"):
            inactive.add(f"{prefix}extractor__{name}")
    elif backend == "tsfel":
        for name in ("tsfresh_profile", "tsfresh_fft_bands"):
            inactive.add(f"{prefix}extractor__{name}")
    elif backend == "tsfresh":
        for name in ("tsfel_domains", "tsfel_features"):
            inactive.add(f"{prefix}extractor__{name}")

    # 3. Modality disables
    if p.get("extractor__thm_modes", "centered_diff") is None:
        inactive.add(f"{prefix}extractor__thm_target_sampling_rate")
        inactive.add(f"{prefix}thm_arch")
    tof_modes = p.get("extractor__tof_modes", "pooled_stats|sensor_stats")
    if tof_modes is None:
        for name in ("extractor__tof_target_sampling_rate", "extractor__tof_fill_mode",
                     "tof_arch", "tof_spatial_mode", "tof_spatial_filters", "tof_spatial_kernel"):
            inactive.add(f"{prefix}{name}")
    else:
        try:
            parsed_tof = _pipe_modes(tof_modes, {"raw", "pooled_stats", "sensor_stats"}, name="tof_modes")
            if "raw" not in parsed_tof or p.get("tof_spatial_mode", "auto") == "flat":
                inactive.add(f"{prefix}tof_spatial_filters")
                inactive.add(f"{prefix}tof_spatial_kernel")
        except Exception:
            pass

    # 4. Augmentation inactive knobs
    aug_prob = float(p.get("augmentor__prob", 0.0) or 0.0)
    per_prob = float(p.get("augmentor__per_aug_prob", 0.5) or 0.0)
    n_copies = int(p.get("augmentor__n_copies", 1) or 0)
    if aug_prob <= 0 or per_prob <= 0 or n_copies <= 0:
        for name in (
            "per_aug_prob", "augmentations", "n_copies",
            "jitter_sigma", "noise_std", "scaling_sigma",
            "warp_sigma", "warp_num_knots",
            "time_warp_max_speed_ratio", "time_warp_n_speed_change",
            "time_shift_frac", "crop_frac_range",
            "temporal_num_masks", "temporal_mask_frac", "temporal_mask_fill",
            "channel_drop_prob",
        ):
            if name != "prob":
                inactive.add(f"{prefix}augmentor__{name}")
    else:
        try:
            aug = TSAugmentor(**{
                k[len("augmentor__"):]: v for k, v in p.items() if k.startswith("augmentor__")
            })
            allowed = aug.allowed_ops()
            op_params = {
                "jitter": ("jitter_sigma",),
                "noise": ("noise_std",),
                "scaling": ("scaling_sigma",),
                "magnitude_warp": ("warp_sigma", "warp_num_knots"),
                "time_warp": ("time_warp_max_speed_ratio", "time_warp_n_speed_change"),
                "time_shift": ("time_shift_frac",),
                "crop": ("crop_frac_range",),
                "temporal_mask": ("temporal_num_masks", "temporal_mask_frac", "temporal_mask_fill"),
                "channel_drop": ("channel_drop_prob",),
            }
            for op, fields in op_params.items():
                if op not in allowed:
                    for f in fields:
                        inactive.add(f"{prefix}augmentor__{f}")
            if float(p.get("augmentor__warp_sigma", 0.0) or 0.0) <= 0:
                inactive.add(f"{prefix}augmentor__warp_num_knots")
            if float(p.get("augmentor__time_warp_max_speed_ratio", 1.0) or 1.0) <= 1.0:
                inactive.add(f"{prefix}augmentor__time_warp_n_speed_change")
            if int(p.get("augmentor__temporal_num_masks", 0) or 0) <= 0 or float(p.get("augmentor__temporal_mask_frac", 0.0) or 0.0) <= 0:
                inactive.add(f"{prefix}augmentor__temporal_mask_fill")
        except Exception:
            pass

    # 5. Network / optimizer / scheduler inactive knobs
    if p.get("pooling_mode", "default") != "attention":
        inactive.add(f"{prefix}attention_units")
    if str(p.get("optimizer_name", "adamw")).lower() != "sgd":
        inactive.add(f"{prefix}momentum")
    if str(p.get("loss_name", "cross_entropy")).lower() != "focal":
        inactive.add(f"{prefix}focal_gamma")
    use_sched = bool(p.get("use_lr_scheduler", True))
    sched_type = str(p.get("lr_scheduler_type", "cosine")).lower() if use_sched else "none"
    if not use_sched or sched_type == "none":
        for name in ("lr_scheduler_type", "lr_factor", "lr_patience", "min_lr", "cosine_t_max", "warmup_epochs"):
            if not use_sched or name != "lr_scheduler_type":
                inactive.add(f"{prefix}{name}")
    elif sched_type == "cosine":
        inactive.add(f"{prefix}lr_factor")
        inactive.add(f"{prefix}lr_patience")
    elif sched_type == "plateau":
        inactive.add(f"{prefix}cosine_t_max")
    elif sched_type == "onecycle":
        for name in ("lr_factor", "lr_patience", "min_lr", "cosine_t_max", "warmup_epochs"):
            inactive.add(f"{prefix}{name}")
    if int(p.get("early_stopping_patience", 10) or 0) <= 0:
        inactive.add(f"{prefix}restore_best_weights")

    return inactive


def candidate_error(params: Mapping[str, Any], base_params: Optional[Mapping[str, Any]] = None) -> Optional[str]:
    """Return an error string if ``params`` is invalid, or ``None`` if valid."""
    merged = dict(base_params or {})
    merged.update(_decode_skopt_params(params))
    clf_kwargs = {
        (k[len("estimator__"):] if k.startswith("estimator__") else k): v
        for k, v in merged.items()
        if k not in {"estimator", "steps", "memory", "verbose"} and not k.startswith("estimator__extractor") and not k.startswith("estimator__augmentor")
    }
    try:
        clf = MultiBranchTSAIClassifier(
            extractor=MultiModalSequenceExtractor(noop=NoopOptions()),
            augmentor=TSAugmentor(),
        )
        valid_keys = set(clf.get_params(deep=True))
        filtered = {
            (k[len("estimator__"):] if k.startswith("estimator__") else k): v
            for k, v in merged.items()
            if (k[len("estimator__"):] if k.startswith("estimator__") else k) in valid_keys
        }
        clf.set_params(**filtered)
        clf.validate()
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def _canonical_candidate_key(params: Mapping[str, Any], base_params: Optional[Mapping[str, Any]] = None) -> str:
    merged = dict(base_params or {})
    merged.update(_decode_skopt_params(params))
    inactive = inactive_parameters_for_candidate(merged)
    active_items = {}
    for k in sorted(params.keys()):
        if k in inactive:
            active_items[k] = "__INACTIVE__"
        else:
            val = _structured(params[k])
            active_items[k] = val
    return json.dumps(active_items, sort_keys=True, default=str)


def deduplicate_param_grid(
    param_grid: Any,
    base_params: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, List[Any]]]:
    """Expand, validate, and deduplicate a parameter grid by pruning inactive parameter duplicates."""
    raw_candidates = list(ParameterGrid(param_grid))
    seen: set = set()
    unique: List[Dict[str, List[Any]]] = []
    for cand in raw_candidates:
        if candidate_error(cand, base_params=base_params) is not None:
            continue
        sig = _canonical_candidate_key(cand, base_params=base_params)
        if sig in seen:
            continue
        seen.add(sig)
        unique.append({k: [v] for k, v in cand.items()})
    if not unique and raw_candidates:
        # Keep first candidate if all were filtered so GridSearchCV still runs and logs failure
        first = raw_candidates[0]
        unique.append({k: [v] for k, v in first.items()})
    return unique


# ---------------------------------------------------------------------------
# 9. Search Drivers, Validation Runner, Permutation Importance, Synthetic Data
# ---------------------------------------------------------------------------

def refit_valid_candidate(cv_results: Any, X: Optional[pd.DataFrame] = None, y: Optional[Any] = None) -> Any:
    """Select the best candidate index whose mean_test_score is finite and above -1.

    When called as ``refit_valid_candidate(cv_results)`` (e.g. ``GridSearchCV(refit=refit_valid_candidate)``),
    returns the integer candidate index. When called as ``refit_valid_candidate(search, X, y)``,
    refits ``search.best_estimator_`` on ``(X, y)`` and returns ``True`` if a valid candidate succeeded.
    """
    if X is not None and hasattr(cv_results, "cv_results_"):
        search = cv_results
        res = search.cv_results_
        scores = np.asarray(res["mean_test_score"], dtype=float)
        valid = np.flatnonzero(np.isfinite(scores) & (scores > -1.0 + 1e-9))
        search.refit_errors_ = []
        for idx in sorted(valid, key=lambda i: scores[i], reverse=True):
            cand_params = res["params"][int(idx)]
            try:
                t0 = time.time()
                model = clone(search.estimator).set_params(**cand_params)
                model.fit(X, y)
                est = model.named_steps["estimator"] if hasattr(model, "named_steps") else model
                if getattr(est, "fit_failed_", False):
                    raise RuntimeError(getattr(est, "fit_error_", "fit failed"))
                search.best_index_ = int(idx)
                search.best_params_ = dict(cand_params)
                search.best_score_ = float(scores[int(idx)])
                search.best_estimator_ = model
                search.refit_time_ = time.time() - t0
                return True
            except Exception as exc:
                search.refit_errors_.append({"candidate": int(idx), "error": f"{type(exc).__name__}: {exc}"})
        return False

    scores = np.asarray(cv_results["mean_test_score"], dtype=float)
    valid = np.flatnonzero(np.isfinite(scores) & (scores > -1.0 + 1e-9))
    if len(valid):
        return int(valid[int(np.argmax(scores[valid]))])
    finite = np.flatnonzero(np.isfinite(scores))
    if len(finite):
        return int(finite[int(np.argmax(scores[finite]))])
    return 0


def _evaluate_candidate_cv(
    estimator: Any,
    params: Mapping[str, Any],
    X: pd.DataFrame,
    y: Any,
    groups: Optional[Any],
    cv: Any,
    scoring: Any,
    error_score: float = -1.0,
) -> Tuple[List[float], float]:
    decoded = _decode_skopt_params(params)
    err = candidate_error(decoded, base_params=estimator.get_params(deep=True))
    if err is not None:
        warnings.warn(f"Candidate skipped before CV ({err})", RuntimeWarning)
        splits = list(cv.split(X, y, groups))
        return [float(error_score)] * len(splits), 0.0

    fold_scores: List[float] = []
    t0 = time.time()
    scorer = scoring if callable(scoring) else candidate_score
    for train_idx, val_idx in cv.split(X, y, groups):
        X_tr = X.iloc[train_idx]
        X_va = X.iloc[val_idx]
        y_tr = y.iloc[train_idx] if isinstance(y, (pd.DataFrame, pd.Series)) and len(y) == len(X) else y
        y_va = y.iloc[val_idx] if isinstance(y, (pd.DataFrame, pd.Series)) and len(y) == len(X) else y
        model = clone(estimator).set_params(**decoded)
        try:
            model.fit(X_tr, y_tr)
            sc = float(scorer(model, X_va, y_va))
        except Exception as exc:
            warnings.warn(f"Fold evaluation failed: {exc}", RuntimeWarning)
            sc = float(error_score)
        fold_scores.append(sc)
    return fold_scores, time.time() - t0


class GASearchCV(BaseEstimator):
    """Genetic / Evolutionary hyperparameter search with hierarchical inactive-parameter caching."""

    def __init__(
        self,
        estimator: Any,
        param_space: Optional[Mapping[str, Any]] = None,
        cv: Any = None,
        scoring: Any = None,
        population_size: int = 10,
        generations: int = 2,
        tournament_size: int = 3,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.2,
        elitism: bool = True,
        refit: Any = True,
        random_state: int = 42,
        verbose: int = 1,
        error_score: float = -1.0,
        n_jobs: int = 1,
        *,
        param_grid: Optional[Mapping[str, Any]] = None,
        crossover_probability: Optional[float] = None,
        mutation_probability: Optional[float] = None,
        return_train_score: bool = True,
    ):
        self.estimator = estimator
        self.param_space = param_space if param_space is not None else param_grid
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_probability if crossover_probability is not None else crossover_prob
        self.mutation_prob = mutation_probability if mutation_probability is not None else mutation_prob
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.elitism = elitism
        self.refit = refit
        self.random_state = random_state
        self.verbose = verbose
        self.error_score = error_score
        self.n_jobs = n_jobs
        self.return_train_score = return_train_score

    def _sample_individual(self, space: Mapping[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
        base = self.estimator.get_params(deep=True)
        for _ in range(25):
            cand = {k: v.sample(rng) for k, v in space.items()}
            if candidate_error(cand, base_params=base) is None:
                return cand
        return {k: v.sample(rng) for k, v in space.items()}

    def _crossover(self, p1: Mapping[str, Any], p2: Mapping[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
        return {k: (p1[k] if rng.random() < 0.5 else p2[k]) for k in p1}

    def _mutate(self, ind: Mapping[str, Any], space: Mapping[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
        out = dict(ind)
        m_prob = float(self.mutation_probability if self.mutation_probability is not None else self.mutation_prob)
        for k, desc in space.items():
            if rng.random() < m_prob:
                out[k] = desc.mutate(out[k], rng)
        return out

    def fit(self, X: pd.DataFrame, y: Any, groups: Optional[Any] = None):
        rng = np.random.default_rng(int(self.random_state or 42))
        raw_space = self.param_space if self.param_space is not None else self.param_grid
        space = prepare_evolutionary_space(raw_space)
        cv = self.cv or GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=int(self.random_state or 42))
        base_params = self.estimator.get_params(deep=True)

        pop_size = max(2, int(self.population_size))
        n_gen = max(0, int(self.generations))
        population = [self._sample_individual(space, rng) for _ in range(pop_size)]

        cache: Dict[str, Tuple[List[float], float]] = {}
        all_params: List[Dict[str, Any]] = []
        all_fold_scores: List[List[float]] = []
        all_times: List[float] = []

        def evaluate_pop(pop: List[Dict[str, Any]]) -> List[float]:
            means: List[float] = []
            for cand in pop:
                sig = _canonical_candidate_key(cand, base_params=base_params)
                if sig in cache:
                    fold_scores, elapsed = cache[sig]
                else:
                    fold_scores, elapsed = _evaluate_candidate_cv(
                        self.estimator, cand, X, y, groups, cv, self.scoring,
                        error_score=float(self.error_score),
                    )
                    cache[sig] = (fold_scores, elapsed)
                decoded = _decode_skopt_params(cand)
                all_params.append(decoded)
                all_fold_scores.append(fold_scores)
                all_times.append(elapsed)
                means.append(float(np.mean(fold_scores)))
            return means

        fitness = evaluate_pop(population)
        c_prob = float(self.crossover_probability if self.crossover_probability is not None else self.crossover_prob)
        for _ in range(n_gen):
            next_pop: List[Dict[str, Any]] = []
            if self.elitism and len(population):
                best_i = int(np.argmax(fitness))
                next_pop.append(dict(population[best_i]))
            while len(next_pop) < pop_size:
                t_size = min(len(population), max(2, int(self.tournament_size)))
                idx1 = rng.choice(len(population), size=t_size, replace=False)
                p1 = population[int(idx1[int(np.argmax([fitness[i] for i in idx1]))])]
                idx2 = rng.choice(len(population), size=t_size, replace=False)
                p2 = population[int(idx2[int(np.argmax([fitness[i] for i in idx2]))])]
                child = self._crossover(p1, p2, rng) if rng.random() < c_prob else dict(p1)
                child = self._mutate(child, space, rng)
                next_pop.append(child)
            population = next_pop
            fitness = evaluate_pop(population)

        mean_scores = np.array([float(np.mean(s)) for s in all_fold_scores], dtype=float)
        std_scores = np.array([float(np.std(s)) for s in all_fold_scores], dtype=float)
        ranks = (len(mean_scores) - np.argsort(np.argsort(mean_scores))).astype(int)
        n_splits = len(all_fold_scores[0]) if all_fold_scores else 1
        self.cv_results_: Dict[str, Any] = {
            "params": all_params,
            "mean_test_score": mean_scores,
            "std_test_score": std_scores,
            "rank_test_score": ranks,
            "mean_fit_time": np.array(all_times, dtype=float),
        }
        for split_i in range(n_splits):
            self.cv_results_[f"split{split_i}_test_score"] = np.array(
                [s[split_i] for s in all_fold_scores], dtype=float,
            )

        best_idx = refit_valid_candidate(self.cv_results_) if callable(self.refit) else int(np.argmax(mean_scores))
        self.best_index_ = int(best_idx)
        self.best_params_ = dict(all_params[self.best_index_])
        self.best_score_ = float(mean_scores[self.best_index_])

        if self.refit:
            self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_)
            self.best_estimator_.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        return self.best_estimator_.predict(X)

    def score(self, X: pd.DataFrame, y: Any):
        return self.best_estimator_.score(X, y)


HierarchicalGASearchCV = GASearchCV


class HierarchicalBayesSearchCV(BaseEstimator):
    """Bayesian optimization search wrapper supporting JSON-encoded tuple/list categories and hierarchical caching.

    Iteration 0 deterministically evaluates the clean baseline starting point:
    augmentations switched off (0.0), minimal regularization (weight_decay=0.0,
    label_smoothing=0.0), all dropouts set to 0.0, and the first option of each
    Categorical dimension (or explicit ``x0`` if provided).
    """

    def __init__(
        self,
        estimator: Any,
        search_spaces: Mapping[str, Any],
        n_iter: int = 10,
        cv: Any = None,
        scoring: Any = None,
        refit: Any = True,
        random_state: int = 42,
        verbose: int = 0,
        error_score: Any = -1.0,
        n_jobs: int = 1,
        *,
        optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        pre_dispatch: Any = "n_jobs",
        return_train_score: bool = True,
        x0: Optional[Mapping[str, Any]] = None,
    ):
        self.estimator = estimator
        self.search_spaces = search_spaces
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.refit = refit
        self.random_state = random_state
        self.verbose = verbose
        self.error_score = error_score
        self.n_jobs = n_jobs
        self.optimizer_kwargs = optimizer_kwargs
        self.pre_dispatch = pre_dispatch
        self.return_train_score = return_train_score
        self.x0 = x0

    @staticmethod
    def _default_starting_point(keys: Sequence[str], dimensions: Sequence[Any], base_params: Mapping[str, Any], user_x0: Optional[Mapping[str, Any]] = None) -> List[Any]:
        """Construct the iteration-0 point with augmentations off, minimal regularization, and dropout=0."""
        point: List[Any] = []
        for k, dim in zip(keys, dimensions):
            is_zero_baseline = (
                "augmentor__" in k
                or "dropout" in k
                or k.endswith(("weight_decay", "label_smoothing"))
            )
            if user_x0 is not None and k in user_x0:
                val = user_x0[k]
                if hasattr(dim, "categories"):
                    cats = list(dim.categories)
                    if val in cats:
                        point.append(val)
                        continue
                    enc = json.dumps(val, sort_keys=True)
                    if enc in cats:
                        point.append(enc)
                        continue
                point.append(val)
                continue
            if hasattr(dim, "categories"):
                cats = list(dim.categories)
                if is_zero_baseline:
                    if 0.0 in cats:
                        point.append(0.0)
                        continue
                    if 0 in cats:
                        point.append(0)
                        continue
                point.append(cats[0])
            else:
                low, high = getattr(dim, "low", 0.0), getattr(dim, "high", 1.0)
                is_int = isinstance(low, (int, np.integer)) and isinstance(high, (int, np.integer))
                if is_zero_baseline and low <= 0 <= high:
                    point.append(int(0) if is_int else 0.0)
                elif k in base_params and isinstance(base_params[k], (int, float, np.number)) and low <= float(base_params[k]) <= high:
                    point.append(int(round(float(base_params[k]))) if is_int else float(base_params[k]))
                else:
                    point.append(int(low) if is_int else float(low))
        return point

    def fit(self, X: pd.DataFrame, y: Any, groups: Optional[Any] = None):
        _require(SKOPT_AVAILABLE, "scikit-optimize")
        from skopt import Optimizer

        cv = self.cv or GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=int(self.random_state or 42))
        space_dict = prepare_bayesian_space(self.search_spaces, skopt_available=True)
        keys = list(space_dict.keys())
        dimensions = [space_dict[k] for k in keys]
        opt_kwargs = dict(self.optimizer_kwargs or {})
        opt_kwargs.setdefault("random_state", int(self.random_state or 42))
        opt = Optimizer(dimensions, **opt_kwargs)

        base_params = self.estimator.get_params(deep=True)
        cache: Dict[str, Tuple[List[float], float]] = {}
        all_params: List[Dict[str, Any]] = []
        all_fold_scores: List[List[float]] = []
        all_times: List[float] = []

        initial_point = self._default_starting_point(keys, dimensions, base_params, self.x0)
        for it in range(max(1, int(self.n_iter))):
            x_point = initial_point if it == 0 else opt.ask()
            raw_cand = dict(zip(keys, x_point))
            decoded = _decode_skopt_params(raw_cand)
            sig = _canonical_candidate_key(decoded, base_params=base_params)
            if sig in cache:
                fold_scores, elapsed = cache[sig]
            else:
                if self.error_score == "raise":
                    err = candidate_error(decoded, base_params=base_params)
                    if err is not None:
                        raise InvalidParams(err)
                fold_scores, elapsed = _evaluate_candidate_cv(
                    self.estimator, decoded, X, y, groups, cv, self.scoring,
                    error_score=-1.0 if self.error_score == "raise" else float(self.error_score),
                )
                cache[sig] = (fold_scores, elapsed)

            mean_sc = float(np.mean(fold_scores))
            opt.tell(x_point, -mean_sc)
            all_params.append(decoded)
            all_fold_scores.append(fold_scores)
            all_times.append(elapsed)

        mean_scores = np.array([float(np.mean(s)) for s in all_fold_scores], dtype=float)
        std_scores = np.array([float(np.std(s)) for s in all_fold_scores], dtype=float)
        ranks = (len(mean_scores) - np.argsort(np.argsort(mean_scores))).astype(int)
        n_splits = len(all_fold_scores[0]) if all_fold_scores else 1
        self.cv_results_: Dict[str, Any] = {
            "params": all_params,
            "mean_test_score": mean_scores,
            "std_test_score": std_scores,
            "rank_test_score": ranks,
            "mean_fit_time": np.array(all_times, dtype=float),
        }
        for split_i in range(n_splits):
            self.cv_results_[f"split{split_i}_test_score"] = np.array(
                [s[split_i] for s in all_fold_scores], dtype=float,
            )

        best_idx = refit_valid_candidate(self.cv_results_) if callable(self.refit) else int(np.argmax(mean_scores))
        self.best_index_ = int(best_idx)
        self.best_params_ = dict(all_params[self.best_index_])
        self.best_score_ = float(mean_scores[self.best_index_])

        if self.refit:
            self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_)
            self.best_estimator_.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        return self.best_estimator_.predict(X)

    def score(self, X: pd.DataFrame, y: Any):
        return self.best_estimator_.score(X, y)


# -- Data loading & validation protocol helpers ------------------------------

def find_data_root() -> Path:
    """Locate the directory containing CMI ``train.csv`` or ``sample.csv`` as a ``Path``."""
    candidates = [
        Path.cwd() / "data",
        Path.cwd().parent / "data",
        Path(__file__).resolve().parents[1] / "data",
        Path("/kaggle/input/cmi-detect-behavior-with-sensor-data"),
    ]
    for path in candidates:
        if path.is_dir() and ((path / "train.csv").is_file() or (path / "sample.csv").is_file()):
            return path
    if Path("/kaggle/input").is_dir():
        for root, _, files in Path("/kaggle/input").walk() if hasattr(Path("/kaggle/input"), "walk") else []:
            if "train.csv" in files or "sample.csv" in files:
                return Path(root)
    return Path(__file__).resolve().parents[1] / "data"


def sample_balanced_split(
    df: pd.DataFrame,
    target_col: str = "bfrb",
    fraction: float = 1.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample a sequence-balanced subset of ``df`` preserving class distribution."""
    if float(fraction) >= 1.0:
        return df.copy()
    rng = np.random.default_rng(int(random_state))
    seq_meta = df.groupby(SEQ_COL, sort=False)[target_col].first()
    chosen: List[str] = []
    for _, group in seq_meta.groupby(seq_meta):
        n_keep = max(1, int(round(len(group) * float(fraction))))
        idx = rng.choice(len(group), size=min(len(group), n_keep), replace=False)
        chosen.extend(group.index[np.sort(idx)].astype(str).tolist())
    return df[df[SEQ_COL].astype(str).isin(set(chosen))].copy()


def validation_protocols(space: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Optional[str]]]:
    """Expand ``VALIDATION_PARAM_SPACE`` into protocol dictionaries."""
    if not space:
        return [{"validation_mode": "sequence", "group_column": None}]
    modes = space.get("validation_mode", ["sequence"])
    if isinstance(modes, str):
        modes = [modes]
    protocols: List[Dict[str, Optional[str]]] = []
    for mode in modes:
        key = str(mode).strip().lower()
        if key == "sequence":
            protocols.append({"validation_mode": "sequence", "group_column": None})
        elif key == "subject":
            protocols.append({"validation_mode": "subject", "group_column": "subject"})
        else:
            raise ValueError(f"Unsupported validation_mode {mode!r}; choose 'sequence' or 'subject'")
    return protocols


def split_validation_protocol(
    df: pd.DataFrame,
    protocol: Mapping[str, Optional[str]],
    *,
    train_size: float = 0.8,
    n_splits: int = 3,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, Any]:
    """Split ``df`` into ``(X_train, X_test, groups, cv_object)`` according to ``protocol``."""
    mode = protocol.get("validation_mode", "sequence")
    if mode not in {"sequence", "subject"}:
        raise ValueError(f"Unsupported validation_mode {mode!r}")
    if mode == "subject" and "subject" not in df.columns:
        raise ValueError("subject validation_mode requires a 'subject' column in the dataframe")

    group_col = "subject" if mode == "subject" else SEQ_COL
    test_size = max(0.05, min(0.95, 1.0 - float(train_size)))

    seq_df = df.groupby(SEQ_COL, sort=False).first().reset_index()
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=int(random_state))
    tr_idx, te_idx = next(splitter.split(seq_df, groups=seq_df[group_col]))

    # If the dataset has both target and non-target domains, ensure both appear in train and test when possible
    if "is_target" in seq_df.columns and seq_df["is_target"].nunique() >= 2:
        for seed_offset in range(1, 20):
            if (seq_df.iloc[tr_idx]["is_target"].nunique() >= 2
                    and seq_df.iloc[te_idx]["is_target"].nunique() >= 2):
                break
            alt = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=int(random_state) + seed_offset)
            tr_idx, te_idx = next(alt.split(seq_df, groups=seq_df[group_col]))

    train_seqs = set(seq_df.iloc[tr_idx][SEQ_COL].astype(str))
    test_seqs = set(seq_df.iloc[te_idx][SEQ_COL].astype(str))

    X_train = df[df[SEQ_COL].astype(str).isin(train_seqs)].copy()
    X_test = df[df[SEQ_COL].astype(str).isin(test_seqs)].copy()
    groups = X_train[group_col]

    if int(n_splits) <= 1:
        cv_object: Any = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=int(random_state))
    else:
        n_unique_groups = X_train[group_col].nunique()
        eff_splits = max(2, min(int(n_splits), int(n_unique_groups)))
        cv_object = GroupKFold(n_splits=eff_splits)

    return X_train, X_test, groups, cv_object


def build_quick_results_df(search_or_results: Any) -> pd.DataFrame:
    """Convert ``search.cv_results_`` (or a search object) into a sorted summary DataFrame."""
    cv_res = getattr(search_or_results, "cv_results_", search_or_results)
    if not cv_res:
        return pd.DataFrame()
    df = pd.DataFrame(cv_res)
    if "rank_test_score" in df.columns:
        df = df.sort_values("rank_test_score", kind="stable").reset_index(drop=True)
    elif "mean_test_score" in df.columns:
        df = df.sort_values("mean_test_score", ascending=False, kind="stable").reset_index(drop=True)
    return df


def run_validation_searches(
    search: Any,
    experiment_df: pd.DataFrame,
    *,
    target_col: str = "bfrb",
    validation_space: Optional[Mapping[str, Any]] = None,
    validation_param_space: Optional[Mapping[str, Any]] = None,
    sensor_columns: Optional[Sequence[str]] = None,
    train_size: float = 0.8,
    n_splits: int = 3,
    random_state: int = 42,
    results_dir: Any = "results",
    experiment_name: str = "multibranch_v3",
    search_mode: str = "bayes",
    timestamp: Optional[str] = None,
    simulate_hidden_test: bool = True,
    hidden_test_null_fraction: float = 0.5,
    quick_save: bool = True,
) -> Dict[str, Any]:
    """Run hyperparameter search across configured validation protocols and evaluate normal + hidden-test holdout."""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = timestamp or time.strftime("%Y%m%d_%H%M%S")
    eff_vspace = validation_param_space if validation_param_space is not None else validation_space
    protocols = validation_protocols(eff_vspace)

    label_cols = [c for c in (SEQ_COL, "subject", "is_target", "gesture", "bfrb", "gesture_action", "gesture_position", "orientation")
                  if c in experiment_df.columns]
    if target_col not in label_cols and target_col in experiment_df.columns:
        label_cols.append(target_col)

    runs: List[Dict[str, Any]] = []
    protocol_results: List[Dict[str, Any]] = []
    all_experiment_rows: List[pd.DataFrame] = []
    last_run: Dict[str, Any] = {}

    for protocol in protocols:
        v_mode = protocol["validation_mode"]
        g_col = protocol["group_column"]
        X_train, X_test, groups, cv_object = split_validation_protocol(
            experiment_df, protocol,
            train_size=train_size, n_splits=n_splits, random_state=random_state,
        )
        y_train = X_train[label_cols].copy()
        y_test = X_test[label_cols].copy()

        run_search = clone(search)
        if hasattr(run_search, "cv"):
            run_search.cv = cv_object
        est = run_search.estimator
        if hasattr(est, "named_steps") and "estimator" in est.named_steps:
            est.set_params(estimator__validation_group_column=g_col, estimator__primary_target=target_col)
        elif hasattr(est, "validation_group_column"):
            est.set_params(validation_group_column=g_col, primary_target=target_col)

        run_search.fit(X_train, y_train, groups=groups)
        if getattr(run_search, "best_estimator_", None) is None:
            refit_valid_candidate(run_search, X_train, y_train)
        best_model = getattr(run_search, "best_estimator_", None)
        best_score = float(getattr(run_search, "best_score_", np.nan))
        best_params = dict(getattr(run_search, "best_params_", {}))

        eval_res: Dict[str, Any] = {}
        if best_model is not None and len(X_test):
            best_est = best_model.named_steps["estimator"] if hasattr(best_model, "named_steps") else best_model
            if not getattr(best_est, "fit_failed_", False):
                y_pred = best_model.predict(X_test)
                eval_res = evaluate_holdout(
                    y_test, y_pred, target_col=target_col, verbose=True,
                    model=best_model, X_test=X_test,
                    simulate_hidden_test=simulate_hidden_test,
                    null_fraction=hidden_test_null_fraction,
                    random_state=random_state,
                )

        cv_df = pd.DataFrame(run_search.cv_results_)
        cv_df["validation_mode"] = v_mode
        cv_df["search_mode"] = search_mode
        cv_df["best_cv_score"] = best_score
        cv_df["holdout_competition_score"] = eval_res.get("competition_score", np.nan)
        cv_df["holdout_hidden_test_score"] = eval_res.get("hidden_test_competition_score", np.nan)
        all_experiment_rows.append(cv_df)

        summary_row = {
            "validation_mode": v_mode,
            "group_column": g_col,
            "search_mode": search_mode,
            "best_cv_score": best_score,
            "holdout_score": eval_res.get("competition_score", np.nan),
            "holdout_competition_score": eval_res.get("competition_score", np.nan),
            "holdout_binary_f1": eval_res.get("binary_f1", np.nan),
            "holdout_gesture_f1": eval_res.get("gesture_f1", np.nan),
            "holdout_macro_f1": eval_res.get("macro_f1", np.nan),
            "holdout_hidden_test_competition_score": eval_res.get("hidden_test_competition_score", np.nan),
            "holdout_hidden_test_binary_f1": eval_res.get("hidden_test_binary_f1", np.nan),
            "holdout_hidden_test_gesture_f1": eval_res.get("hidden_test_gesture_f1", np.nan),
        }
        protocol_results.append(summary_row)
        run_entry = {
            "validation_mode": v_mode,
            "protocol": protocol,
            "search": run_search,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "groups": groups,
            "cv_object": cv_object,
            "best_score": best_score,
            "best_params": best_params,
            "eval_results": eval_res,
            "holdout_results": eval_res,
            "experiment_succeeded": best_model is not None and np.isfinite(best_score) and best_score > -1.0,
        }
        runs.append(run_entry)
        last_run = dict(run_entry)

    experiments_df = pd.concat(all_experiment_rows, ignore_index=True) if all_experiment_rows else pd.DataFrame()
    exp_path = out_dir / f"{experiment_name}_{ts}_experiments.csv"
    sum_path = out_dir / f"{experiment_name}_{ts}_summary.json"
    experiments_df.to_csv(exp_path, index=False)
    sum_path.write_text(
        json.dumps({
            "experiment_name": experiment_name,
            "target_col": target_col,
            "search_mode": search_mode,
            "timestamp": ts,
            "protocol_results": protocol_results,
        }, indent=2, default=str),
        encoding="utf-8",
    )

    return {
        "last_run": last_run,
        "runs": runs,
        "protocol_results": protocol_results,
        "results_dir": out_dir,
        "experiments_df": experiments_df,
        "experiments_path": str(exp_path),
        "summary_path": str(sum_path),
    }


# -- Trained-Model Sequence-Aligned Permutation Importance -------------------

def _sequence_layout(sequence_ids: np.ndarray, mask: np.ndarray):
    """Group window indices by sequence and stratify by (window_count, mask_signature)."""
    seq_str = np.asarray(sequence_ids, dtype=str)
    unique_seqs: List[str] = []
    rows_by_seq: List[np.ndarray] = []
    strata: Dict[Tuple[int, bytes], List[int]] = {}

    seen: Dict[str, List[int]] = {}
    for idx, s in enumerate(seq_str):
        if s not in seen:
            unique_seqs.append(s)
            seen[s] = []
        seen[s].append(idx)

    for s_idx, s in enumerate(unique_seqs):
        r = np.asarray(seen[s], dtype=int)
        rows_by_seq.append(r)
        key = (len(r), np.asarray(mask[r], dtype=bool).tobytes())
        strata.setdefault(key, []).append(s_idx)

    return unique_seqs, rows_by_seq, strata


def _donor_rows(
    rows_by_seq: List[np.ndarray],
    strata: Mapping[Tuple[int, bytes], List[int]],
    rng: np.random.Generator,
    n_total: int,
) -> Tuple[np.ndarray, int]:
    """Permute sequences within matching window-layout strata while preserving window order."""
    donor = np.arange(n_total, dtype=int)
    moved_sequences = 0
    for seq_indices in strata.values():
        if len(seq_indices) <= 1:
            continue
        k = len(seq_indices)
        shift = int(rng.integers(1, k))
        order = list(rng.permutation(k))
        # Derangement: ensure no sequence maps to itself when k >= 2
        if any(order[i] == i for i in range(k)):
            order = [(i + shift) % k for i in range(k)]
        for dst_pos, src_pos in enumerate(order):
            dst_rows = rows_by_seq[seq_indices[dst_pos]]
            src_rows = rows_by_seq[seq_indices[src_pos]]
            donor[dst_rows] = src_rows
            if dst_pos != src_pos:
                moved_sequences += 1
    return donor, moved_sequences


def save_permutation_importance(
    importance_df: pd.DataFrame,
    output_dir: Any,
    prefix: str = "permutation_importance",
) -> Dict[str, str]:
    """Save permutation importance table (.csv), metadata (.json), and bar plot (.png)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{prefix}.csv"
    json_path = out / f"{prefix}.json"
    png_path = out / f"{prefix}.png"

    importance_df.to_csv(csv_path, index=False)
    payload = {
        "metadata": {
            "scope": importance_df.attrs.get("scope", "saved internal validation sequences"),
            "baseline_macro_f1": float(importance_df.attrs.get("baseline_macro_f1", 0.0)),
            "baseline_competition_score": float(importance_df.attrs.get("baseline_competition_score", 0.0)),
            "permutable_fraction": float(importance_df.attrs.get("permutable_fraction", 0.0)),
            "repeats": int(importance_df.attrs.get("repeats", 1)),
        },
        "importance": importance_df.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.35 * len(importance_df))))
    ordered = importance_df.sort_values("importance_mean", ascending=True)
    labels = [f"{r['kind']}: {r['name']}" for _, r in ordered.iterrows()]
    ax.barh(labels, ordered["importance_mean"], xerr=ordered["importance_std"], color="#2b6cb0", alpha=0.85)
    ax.axvline(0.0, color="#4a5568", linestyle="--", linewidth=1)
    ax.set_xlabel("Drop in Macro F1 (higher = more important)")
    ax.set_title("Validation Permutation Importance (v3 Multimodal Stack)")
    fig.tight_layout()
    fig.savefig(png_path, dpi=140)
    plt.close(fig)

    paths = {"csv": str(csv_path), "json": str(json_path), "png": str(png_path), "plot": str(png_path)}
    importance_df.attrs["paths"] = paths
    return paths


def compute_permutation_importance(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.DataFrame,
    *,
    repeats: int = 5,
    random_state: int = 42,
    include_noop_individual: bool = True,
    require_internal_validation: bool = True,
    output_dir: Optional[Any] = None,
    prefix: str = "permutation_importance",
) -> pd.DataFrame:
    """Compute sequence-aligned permutation importance on the internal validation partition.

    Extracts chunks once, verifies partition hygiene (rejects training or holdout sequences
    when ``require_internal_validation=True``), permutes modality groups and individual
    NOOP/static features across sequences with matching window layouts, and always restores
    chunk arrays in a ``finally`` block.
    """
    if not isinstance(repeats, (int, np.integer)) or int(repeats) < 1:
        raise ValueError("repeats must be a positive integer")

    est = estimator.named_steps["estimator"] if hasattr(estimator, "named_steps") and "estimator" in estimator.named_steps else estimator
    extractor = getattr(est, "extractor_", None) or getattr(est, "extractor", None)
    seq_col = getattr(extractor, "sequence_col", SEQ_COL)
    target_col = getattr(est, "primary_target", "bfrb")

    seq_ids_in_x = pd.Index(X[seq_col].astype(str).unique())
    if require_internal_validation:
        val_ids = pd.Index(getattr(est, "validation_sequence_ids_", [])).astype(str)
        train_ids = pd.Index(getattr(est, "train_sequence_ids_", [])).astype(str)
        if not len(val_ids) or not set(seq_ids_in_x).issubset(set(val_ids)):
            raise ValueError("Provided X contains sequences outside that partition (validation_sequence_ids_)")
        if set(seq_ids_in_x) & set(train_ids):
            raise ValueError("Provided X contains training sequences; importance must use internal validation only")

    # Validate sequence labels
    if seq_col not in y.columns or target_col not in y.columns:
        raise ValueError(f"y must contain {seq_col!r} and {target_col!r}")
    y_grouped = y.groupby(y[seq_col].astype(str), sort=False)[target_col].nunique()
    if (y_grouped > 1).any():
        raise ValueError(f"Target {target_col!r} must be constant within each sequence")
    label_map = y.groupby(y[seq_col].astype(str), sort=False)[target_col].first()
    if not set(seq_ids_in_x).issubset(set(label_map.index)):
        raise ValueError("Missing labels for one or more sequences in X")
    y_true = label_map.reindex(seq_ids_in_x).astype(str)

    chunks = extractor.transform_chunks(X)
    c_seq_ids = np.asarray(chunks["sequence_ids"], dtype=str)
    primary_mask = chunks["masks"]["imu"] if "masks" in chunks and "imu" in chunks["masks"] else chunks["mask"]
    unique_seqs, rows_by_seq, strata = _sequence_layout(c_seq_ids, primary_mask)
    permutable_seqs = sum(len(v) for v in strata.values() if len(v) > 1)
    permutable_fraction = permutable_seqs / max(1, len(unique_seqs))

    def score_current_chunks() -> Tuple[float, float]:
        if hasattr(est, "predict_window_proba"):
            win_proba = est.predict_window_proba(chunks)
            _, seq_proba = est.aggregate_window_proba(win_proba, c_seq_ids, seq_order=seq_ids_in_x)
            classes = np.asarray(est.le_.classes_)
        else:
            branches = est._branches_from_chunks(chunks)
            inputs = est._model_inputs(branches, chunks["mask"])
            win_proba = est.model_.predict(inputs, verbose=0)
            classes = np.asarray(est.le_.classes_)
            seq_proba = np.vstack([
                win_proba[c_seq_ids == s].mean(axis=0) for s in seq_ids_in_x
            ])
        preds = pd.Series(classes[seq_proba.argmax(axis=1)], index=seq_ids_in_x)
        macro = float(f1_score(y_true, preds, average="macro", zero_division=0))
        comp = float(competition_score(y_true, preds)["competition_score"]) if target_col == "bfrb" else macro
        return macro, comp

    baseline_macro, baseline_comp = score_current_chunks()

    # Build permutation specs
    specs: List[Tuple[str, str, Any]] = []
    if "branches" in chunks:
        # v3 MultiModalSequenceExtractor chunk structure
        ch_names = chunks["channel_names"]
        st_names = list(chunks["static_names"])
        imu_names = ch_names.get("imu", [])
        acc_ch = [i for i, n in enumerate(imu_names) if n.startswith(("acc_", "lin_", "jerk_"))]
        rot_ch = [i for i, n in enumerate(imu_names) if n.startswith(("rot_", "ang_", "euler_"))]
        imu_st = [i for i, n in enumerate(st_names) if any(tok in n for tok in ("acc_mag", "lin_mag", "ang_speed", "noop_"))]
        thm_st = [i for i, n in enumerate(st_names) if "thm" in n]
        tof_st = [i for i, n in enumerate(st_names) if "tof" in n]
        gen_st = [i for i, n in enumerate(st_names) if n.startswith(("tsfel__", "tsfresh__"))]
        noop_st = [i for i, n in enumerate(st_names) if n.startswith("noop_")]

        specs.append(("group", "motion_coherent", {"branches": {"imu": list(range(len(imu_names)))}, "static": imu_st}))
        if acc_ch:
            specs.append(("group", "imu_acceleration", {"branches": {"imu": acc_ch}, "static": []}))
        if rot_ch:
            specs.append(("group", "imu_rotation", {"branches": {"imu": rot_ch}, "static": []}))
        if "thm" in chunks["branches"]:
            specs.append(("group", "thm", {"branches": {"thm": list(range(chunks["branches"]["thm"].shape[2]))}, "static": thm_st}))
        if "tof" in chunks["branches"]:
            specs.append(("group", "tof", {"branches": {"tof": list(range(chunks["branches"]["tof"].shape[2]))}, "static": tof_st}))
        if gen_st:
            specs.append(("group", "static_open_source", {"branches": {}, "static": gen_st}))
        if noop_st:
            specs.append(("group", "noop_all", {"branches": {}, "static": noop_st}))

        if include_noop_individual:
            for feat_name in chunks.get("noop_feature_names", []):
                if feat_name.endswith("_supported"):
                    continue
                cols = [st_names.index(feat_name)] if feat_name in st_names else []
                sup_name = f"{feat_name}_supported"
                if sup_name in st_names:
                    cols.append(st_names.index(sup_name))
                if cols:
                    specs.append(("noop_feature", feat_name, {"branches": {}, "static": cols}))
    else:
        # Legacy fixture compatibility
        f_names = list(chunks.get("feature_names", []))
        noop_names = list(chunks.get("noop_feature_names", []))
        acc_idx = [i for i, n in enumerate(f_names) if not n.startswith("tof")]
        tof_idx = [i for i, n in enumerate(f_names) if n.startswith("tof")]
        specs.append(("group", "motion_coherent", {"X": acc_idx, "noop": list(range(len(noop_names)))}))
        specs.append(("group", "tof", {"X": tof_idx, "noop": []}))
        if include_noop_individual:
            for j, name in enumerate(noop_names):
                specs.append(("noop_feature", name, {"X": [], "noop": [j]}))

    # Snapshot originals so they are always restored
    if "branches" in chunks:
        orig_branches = {k: v.copy() for k, v in chunks["branches"].items()}
        orig_static = chunks["static"].copy()
    else:
        orig_X = chunks["X"].copy()
        orig_noop = chunks["noop_features"].copy()

    records: List[Dict[str, Any]] = []
    try:
        for kind, name, target_spec in specs:
            rng = np.random.default_rng(int(random_state))
            drops: List[float] = []
            perm_scores: List[float] = []
            for _ in range(int(repeats)):
                donor, _ = _donor_rows(rows_by_seq, strata, rng, len(c_seq_ids))
                if "branches" in chunks:
                    for b_name, ch_idx in target_spec["branches"].items():
                        chunks["branches"][b_name][:, :, ch_idx] = orig_branches[b_name][donor][:, :, ch_idx]
                    if target_spec["static"]:
                        s_idx = target_spec["static"]
                        chunks["static"][:, s_idx] = orig_static[donor][:, s_idx]
                else:
                    if target_spec["X"]:
                        x_idx = target_spec["X"]
                        chunks["X"][:, :, x_idx] = orig_X[donor][:, :, x_idx]
                    if target_spec["noop"]:
                        n_idx = target_spec["noop"]
                        chunks["noop_features"][:, n_idx] = orig_noop[donor][:, n_idx]

                perm_macro, _ = score_current_chunks()
                perm_scores.append(perm_macro)
                drops.append(baseline_macro - perm_macro)

                # Restore after each repeat
                if "branches" in chunks:
                    for b_name in target_spec["branches"]:
                        chunks["branches"][b_name][:] = orig_branches[b_name]
                    if target_spec["static"]:
                        chunks["static"][:] = orig_static
                else:
                    chunks["X"][:] = orig_X
                    chunks["noop_features"][:] = orig_noop

            records.append({
                "kind": kind,
                "name": name,
                "importance_mean": float(np.mean(drops)),
                "importance_std": float(np.std(drops)),
                "permuted_macro_f1_mean": float(np.mean(perm_scores)),
                "baseline_macro_f1": float(baseline_macro),
                "n_sequences": int(len(unique_seqs)),
                "n_permutable_sequences": int(permutable_seqs),
            })
    finally:
        if "branches" in chunks:
            for b_name, arr in orig_branches.items():
                chunks["branches"][b_name][:] = arr
            chunks["static"][:] = orig_static
        else:
            chunks["X"][:] = orig_X
            chunks["noop_features"][:] = orig_noop

    result = pd.DataFrame(records).sort_values("importance_mean", ascending=False, kind="stable").reset_index(drop=True)
    result.attrs["baseline_macro_f1"] = float(baseline_macro)
    result.attrs["baseline_competition_score"] = float(baseline_comp)
    result.attrs["permutable_fraction"] = float(permutable_fraction)
    result.attrs["repeats"] = int(repeats)
    result.attrs["scope"] = "saved internal validation sequences"

    if output_dir is not None:
        save_permutation_importance(result, output_dir, prefix=prefix)
    return result


# -- Synthetic dataset builder for fast unit & integration testing -----------

def make_synthetic_cmi_df(
    n_subjects: int = 6,
    sequences_per_subject: int = 4,
    n_timesteps: int = 60,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create a realistic synthetic CMI multimodal dataset with IMU, Thermopile, and 5x8x8 ToF."""
    rng = np.random.default_rng(int(random_state))
    gestures = [
        ("Forehead - pull hairline", 1, "Forehead", "pull hairline"),
        ("Cheek - pinch skin", 1, "Cheek", "pinch skin"),
        ("Drink from bottle/cup", 0, "other", "other"),
        ("Wave hello", 0, "other", "other"),
    ]
    frames: List[pd.DataFrame] = []
    for subj_i in range(int(n_subjects)):
        subj = f"SUBJ_{subj_i:03d}"
        for seq_j in range(int(sequences_per_subject)):
            g_name, is_target, g_pos, g_act = gestures[seq_j % len(gestures)]
            bfrb = g_name if is_target == 1 else "non_bfrb"
            seq_id = f"{subj}_SEQ_{seq_j:02d}"
            t = np.arange(int(n_timesteps), dtype=float) / 100.0
            freq = 2.0 + 0.8 * (seq_j % len(gestures))
            amp = 1.5 if is_target == 1 else 0.5
            phase = rng.uniform(0, 2 * np.pi)

            acc_x = amp * np.sin(2 * np.pi * freq * t + phase) + rng.normal(0, 0.05, len(t))
            acc_y = amp * np.cos(2 * np.pi * freq * t + phase) + rng.normal(0, 0.05, len(t))
            acc_z = GRAVITY + 0.3 * amp * np.sin(4 * np.pi * freq * t) + rng.normal(0, 0.05, len(t))

            angle = 0.25 * amp * np.sin(2 * np.pi * freq * t)
            rot_w = np.cos(angle / 2.0)
            axis_v = np.sin(angle / 2.0) / np.sqrt(3.0)

            data: Dict[str, Any] = {
                SEQ_COL: seq_id,
                "subject": subj,
                COUNTER_COL: np.arange(int(n_timesteps), dtype=int),
                "is_target": int(is_target),
                "gesture": g_name,
                "bfrb": bfrb,
                "gesture_position": g_pos,
                "gesture_action": g_act,
                "orientation": "Seated Straight",
                "acc_x": acc_x,
                "acc_y": acc_y,
                "acc_z": acc_z,
                "rot_w": rot_w,
                "rot_x": axis_v,
                "rot_y": axis_v,
                "rot_z": axis_v,
            }
            # Thermopile and ToF are held across 5 rows (20 Hz on a 100 Hz row clock)
            t_hold = np.floor(np.arange(int(n_timesteps)) / 5.0) * (5.0 / 100.0)
            for s in range(1, 6):
                data[f"thm_{s}"] = 31.0 + 0.8 * is_target + 0.1 * s + 0.2 * np.sin(2 * np.pi * freq * t_hold)
            for s in range(1, 6):
                base_dist = 120.0 if is_target == 1 else 260.0
                for p in range(64):
                    col_vals = base_dist + s * 4.0 + (p % 8) * 1.5 + 10.0 * np.cos(2 * np.pi * freq * t_hold)
                    if p == 63:
                        col_vals = np.where(np.arange(int(n_timesteps)) % 15 == 0, -1.0, col_vals)
                    data[f"tof_{s}_v{p}"] = col_vals
            frames.append(pd.DataFrame(data))
    return pd.concat(frames, ignore_index=True)
