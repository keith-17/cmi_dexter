"""
base_utils_qwen.py

Cleaned and runnable Honeycomb / sequence feature extraction utilities.
"""

from __future__ import annotations

import ast
import json
from fractions import Fraction
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, make_scorer

warnings.filterwarnings("ignore")

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
    import pywt
except Exception:
    pywt = None

try:
    from skopt.space import Categorical
except Exception:
    Categorical = None

try:
    from scipy.interpolate import CubicSpline
except Exception:
    CubicSpline = None


class InvalidExtractorParams(ValueError):
    """Raised when extractor hyperparameters cannot produce valid features."""


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


class HoneycombBase(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None):
        return self

    def _parse_modes(self, modes_str: Optional[str]) -> List[str]:
        if not modes_str:
            return []
        return [
            m.strip().lower()
            for m in str(modes_str).split("|")
            if m.strip() and m.strip().lower() != "none"
        ]


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
                df[num_cols] = df.groupby(self.sequence_col, sort=False)[num_cols].transform(
                    lambda g: g.interpolate(method="linear", limit_direction="both").ffill().bfill()
                )
            elif self.interp_mode == "ffill":
                df[num_cols] = df.groupby(self.sequence_col, sort=False)[num_cols].transform(
                    lambda g: g.ffill().bfill()
                )
            else:
                raise InvalidExtractorParams("interp_mode must be 'linear' or 'ffill'")

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
        q = q / norms

        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        R = np.empty((len(q), 3, 3), dtype=float)
        R[:, 0, 0] = 1 - 2 * y**2 - 2 * z**2
        R[:, 0, 1] = 2 * x * y - 2 * w * z
        R[:, 0, 2] = 2 * x * z + 2 * w * y
        R[:, 1, 0] = 2 * x * y + 2 * w * z
        R[:, 1, 1] = 1 - 2 * x**2 - 2 * z**2
        R[:, 1, 2] = 2 * y * z - 2 * w * x
        R[:, 2, 0] = 2 * x * z - 2 * w * y
        R[:, 2, 1] = 2 * y * z + 2 * w * x
        R[:, 2, 2] = 1 - 2 * x**2 - 2 * y**2

        unit_scale = gravity_magnitude if self.acceleration_units == "g" else 1.0
        acc = acc * unit_scale
        gravity_world = np.array([0.0, 0.0, gravity_magnitude])
        if self.linear_acc_frame == "world":
            lin_acc = np.einsum("nij,nj->ni", R, acc) - gravity_world
        else:
            gravity_body = np.einsum("nji,j->ni", R, gravity_world)
            lin_acc = acc - gravity_body

        for i, col in enumerate(["lin_acc_x", "lin_acc_y", "lin_acc_z"]):
            df[col] = lin_acc[:, i]

        return df

    def _linear_acc_highpass(self, df: pd.DataFrame, acc_cols: List[str]) -> pd.DataFrame:
        if not acc_cols:
            return df

        for col in acc_cols:
            baseline = df.groupby(self.sequence_col, sort=False)[col].transform(
                lambda g: g.rolling(
                    window=self.window_size,
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
        df = X.copy()

        acc_cols = [
            c
            for c in df.columns
            if c.startswith("acc_")
            and not c.endswith(("_vel", "_disp", "_jerk", "_mag", "_dr_vel", "_dr_pos"))
        ]

        if self.motion_filter_mode in {"kalman", "extended_kalman"} and acc_cols:
            if self.motion_filter_mode == "extended_kalman":
                warnings.warn(
                    "'extended_kalman' is a deprecated alias for the scalar random-walk "
                    "Kalman filter; it is not an EKF.",
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
        smooth_alpha: Optional[float] = None,
        sequence_col: str = "sequence_id",
    ):
        self.acc_modes = acc_modes
        self.linear_acc_modes = linear_acc_modes
        self.use_acc_magnitude = use_acc_magnitude
        self.use_linear_acc_magnitude = use_linear_acc_magnitude
        self.window_size = window_size
        self.smooth_alpha = smooth_alpha
        self.sequence_col = sequence_col

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
        df = X.copy()
        parts = []
        if "dt" not in df.columns:
            df["dt"] = 1.0
        dt = df["dt"].astype(float)
        first_sample = df.groupby(self.sequence_col, sort=False).cumcount().eq(0)
        integration_dt = dt.mask(first_sample, 0.0)

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
                        lambda g: g.rolling(window=self.window_size, center=True, min_periods=1).mean()
                    )
                parts.append(out.add_suffix("_smooth"))
            elif mode == "velocity":
                vel = (
                    df[self.acc_cols_].mul(integration_dt, axis=0)
                    .groupby(df[self.sequence_col].values, sort=False).cumsum()
                )
                parts.append(vel.add_suffix("_raw_integrated_velocity"))
            elif mode == "displacement":
                vel = (
                    df[self.acc_cols_].mul(integration_dt, axis=0)
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
                    vel = (
                        lin.mul(integration_dt, axis=0)
                        .groupby(df[self.sequence_col].values, sort=False).cumsum()
                    )
                    parts.append(vel.add_suffix("_linear_integrated_velocity"))
                elif mode == "displacement":
                    vel = (
                        lin.mul(integration_dt, axis=0)
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
        df = X.copy()
        parts = []
        dt = df.get("rot_dt", df.get("dt", pd.Series(1.0, index=df.index))).astype(float)
        if not self.rot_cols_:
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
                w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
                roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
                pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
                yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
                parts.append(
                    pd.DataFrame(
                        {"rot_roll": roll, "rot_pitch": pitch, "rot_yaw": yaw},
                        index=df.index,
                    )
                )
            elif mode == "delta_euler":
                w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
                roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
                pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
                yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
                euler_df = pd.DataFrame(
                    {"rot_roll": roll, "rot_pitch": pitch, "rot_yaw": yaw},
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
                    # The relative increment R_previous.inv() * R_current is a
                    # local/body-frame rotation under the body-to-world convention.
                    # q is scalar-first wxyz; SciPy expects scalar-last xyzw.
                    rotations = Rotation.from_quat(q[positions][:, [1, 2, 3, 0]])
                    for idx, cur in enumerate(positions[1:], start=1):
                        local_increment = rotations[idx - 1].inv() * rotations[idx]
                        angular_velocity[cur] = local_increment.as_rotvec() / float(dt.iloc[cur])
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
            elif mode == "rot6d":
                w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
                R = np.empty((len(q), 3, 3), dtype=float)
                R[:, 0, 0] = 1 - 2 * y**2 - 2 * z**2
                R[:, 0, 1] = 2 * x * y - 2 * w * z
                R[:, 0, 2] = 2 * x * z + 2 * w * y
                R[:, 1, 0] = 2 * x * y + 2 * w * z
                R[:, 1, 1] = 1 - 2 * x**2 - 2 * z**2
                R[:, 1, 2] = 2 * y * z - 2 * w * x
                R[:, 2, 0] = 2 * x * z - 2 * w * y
                R[:, 2, 1] = 2 * y * z + 2 * w * x
                R[:, 2, 2] = 1 - 2 * x**2 - 2 * y**2
                # Column-major: first rotation-matrix column, then the second.
                rot6d = np.concatenate([R[:, :, 0], R[:, :, 1]], axis=1)
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
        sequence_col: str = "sequence_id",
        n_sensors: int = 5,
    ):
        self.tof_modes = tof_modes
        self.tof_fill_mode = tof_fill_mode
        self.sequence_col = sequence_col
        self.n_sensors = n_sensors

    def fit(self, X: pd.DataFrame, y=None):
        self.tof_modes_ = self._parse_modes(self.tof_modes)
        self.tof_cols_ = [c for c in X.columns if c.startswith("tof_")]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["tof_modes_", "tof_cols_"])
        if not self.tof_cols_:
            return pd.DataFrame(index=X.index)

        observed = X[self.tof_cols_].astype(float).replace(-1.0, np.nan)
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
        if not self.thm_cols_:
            return pd.DataFrame(index=X.index)

        raw = X[self.thm_cols_].astype(float).copy()
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
        window_seconds: Optional[float] = None,
        counter_col: str = "sequence_counter",
        sequence_col: str = "sequence_id",
        feature_prefix: str = "stft0",
        signal_columns: Optional[List[str]] = None,
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
        self.window_seconds = window_seconds
        self.counter_col = counter_col
        self.sequence_col = sequence_col
        self.feature_prefix = feature_prefix
        self.signal_columns = signal_columns

    def fit(self, X: pd.DataFrame, y=None):
        fs = _positive_rate(self.sampling_rate, "STFT sampling_rate")
        derived_suffixes = (
            "_vel", "_disp", "_jerk", "_mag", "_dr_vel", "_dr_pos",
            "_raw_integrated_velocity", "_raw_integrated_displacement",
            "_raw_dead_reckoning_velocity", "_raw_dead_reckoning_position",
        )
        candidates = [
            c for c in X.columns
            if c.startswith("acc_") and not c.endswith(derived_suffixes)
        ]
        self.acc_cols_ = candidates if self.signal_columns is None else [c for c in self.signal_columns if c in X.columns]
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
        nyquist = fs / 2.0
        for name, bounds in bands.items():
            if len(bounds) != 2:
                raise InvalidExtractorParams(f"STFT band {name!r} must have (low, high) bounds")
            low, high = map(float, bounds)
            if not np.isfinite([low, high]).all() or low < 0 or high <= low or high > nyquist:
                raise InvalidExtractorParams(
                    f"STFT band {name!r} must satisfy 0 <= low < high <= Nyquist ({nyquist:g} Hz)"
                )
            self.frequency_bands_[str(name)] = (low, high)
        return self

    def _compute_stft_features(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        if stft is None:
            raise ImportError("STFTExtractor requires SciPy.")
        signal = _signal_values(signal, "STFTExtractor")
        if signal is None:
            signal = np.zeros(0, dtype=float)
        observed_length = len(signal)
        fs = _positive_rate(self.sampling_rate, "STFT sampling_rate")
        nperseg = int(self.nperseg_)
        if len(signal) < nperseg:
            signal = np.pad(signal, (0, nperseg - len(signal)), mode='constant')
        scipy_scaling = "psd" if self.representation_ == "psd" else "spectrum"
        f, t, Zxx = stft(
            signal, fs=fs, window=self.window_type,
            nperseg=nperseg, noverlap=self.noverlap_, nfft=None, detrend=self.detrend,
            return_onesided=True, scaling=scipy_scaling, axis=-1,
            boundary="zeros", padded=True,
        )
        if self.representation_ == "magnitude":
            representation = np.abs(Zxx)
        else:
            if self.representation_ == "psd":
                representation = _one_sided_psd(Zxx, nperseg)
                coverage = _stft_frame_coverage(
                    t, observed_length, fs, nperseg, self.window_type
                )
                representation = np.divide(
                    representation,
                    coverage[None, :],
                    out=np.zeros_like(representation),
                    where=coverage[None, :] > 0,
                )
            else:
                representation = np.abs(Zxx) ** 2
                if len(f) > 1:
                    one_sided = np.full(len(f), 2.0)
                    one_sided[0] = 1.0
                    if nperseg % 2 == 0:
                        one_sided[-1] = 1.0
                    representation *= one_sided[:, None]

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
            for band_name, (f_low, f_high) in self.frequency_bands_.items():
                band_mask = (f >= f_low) & (f < f_high)
                band_power = (
                    float(np.mean(np.sum(representation[band_mask], axis=0) * dfreq))
                    if band_mask.any() else 0.0
                )
                features[f"stft_band_{band_name}_mean_power"] = band_power
                features[f"stft_band_{band_name}_power_fraction"] = (
                    band_power / mean_power if mean_power > 0 else 0.0
                )
                features[f"stft_band_{band_name}_resolved"] = float(band_mask.any())

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
        df = X.copy()
        parts = []
        for col in self.acc_cols_:
            feat_dict = {}
            for seq_id, group in df.groupby(self.sequence_col, sort=False):
                self._validate_uniform_counter(group)
                signal = group[col].to_numpy(dtype=float)
                seq_features = self._compute_stft_features(signal)
                for feat_name, value in seq_features.items():
                    feat_dict.setdefault(feat_name, []).append(value)
            seq_ids = df[self.sequence_col].values
            unique_seqs = pd.unique(seq_ids)
            for feat_name, values in feat_dict.items():
                seq_to_val = dict(zip(unique_seqs, values))
                df[f"{col}_{self.feature_prefix}_{feat_name}"] = [seq_to_val[sid] for sid in seq_ids]
            feat_cols = [f"{col}_{self.feature_prefix}_{k}" for k in feat_dict.keys()]
            parts.append(df[feat_cols])
        if not parts:
            return pd.DataFrame(index=df.index)
        return pd.concat(parts, axis=1)

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
        min_scale: Optional[float] = None,
        max_frequency: Optional[float] = None,
        counter_col: str = "sequence_counter",
        sequence_col: str = "sequence_id",
        feature_prefix: str = "cwt0",
        signal_columns: Optional[List[str]] = None,
    ):
        self.acc_modes = acc_modes
        self.wavelet = wavelet
        self.widths = widths
        self.max_scale = max_scale
        self.n_scales = n_scales
        self.use_log_scale = use_log_scale
        self.sampling_rate = sampling_rate
        self.min_scale = min_scale
        self.max_frequency = max_frequency
        self.counter_col = counter_col
        self.sequence_col = sequence_col
        self.feature_prefix = feature_prefix
        self.signal_columns = signal_columns

    def fit(self, X: pd.DataFrame, y=None):
        if pywt is None:
            raise ImportError("CWTExtractor requires PyWavelets.")
        fs = _positive_rate(self.sampling_rate, "CWT sampling_rate")
        derived_suffixes = (
            "_vel", "_disp", "_jerk", "_mag", "_dr_vel", "_dr_pos",
            "_raw_integrated_velocity", "_raw_integrated_displacement",
            "_raw_dead_reckoning_velocity", "_raw_dead_reckoning_position",
        )
        candidates = [
            c for c in X.columns
            if c.startswith("acc_") and not c.endswith(derived_suffixes)
        ]
        self.acc_cols_ = candidates if self.signal_columns is None else [c for c in self.signal_columns if c in X.columns]
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

    def _compute_cwt_features(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        if pywt is None:
            raise ImportError("CWTExtractor requires PyWavelets.")
        signal = _signal_values(signal, "CWTExtractor")
        if signal is None:
            return self._empty_features()
        fs = _positive_rate(self.sampling_rate, "CWT sampling_rate")
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
        df = X.copy()
        parts = []
        for col in self.acc_cols_:
            feat_dict = {}
            for seq_id, group in df.groupby(self.sequence_col, sort=False):
                self._validate_uniform_counter(group)
                signal = group[col].to_numpy(dtype=float)
                seq_features = self._compute_cwt_features(signal)
                for feat_name, value in seq_features.items():
                    feat_dict.setdefault(feat_name, []).append(value)
            seq_ids = df[self.sequence_col].values
            unique_seqs = pd.unique(seq_ids)
            for feat_name, values in feat_dict.items():
                seq_to_val = dict(zip(unique_seqs, values))
                df[f"{col}_{self.feature_prefix}_{feat_name}"] = [seq_to_val[sid] for sid in seq_ids]
            feat_cols = [f"{col}_{self.feature_prefix}_{k}" for k in feat_dict.keys()]
            parts.append(df[feat_cols])
        if not parts:
            return pd.DataFrame(index=df.index)
        return pd.concat(parts, axis=1)

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


class SequenceExtractor(HoneycombBase):
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
        smooth_alpha: Optional[float] = None,
        clip_value: Optional[float] = None,
        interp_mode: str = "linear",

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
        self.smooth_alpha = smooth_alpha
        self.clip_value = clip_value
        self.interp_mode = interp_mode

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

        self._rebuild_components()

    def set_params(self, **params):
        super().set_params(**params)
        self._rebuild_components()
        return self

    # ---------- helpers ----------

    def _build_stft_defaults(self) -> Dict[str, Any]:
        return dict(
            nperseg=self.stft_nperseg,
            noverlap=self.stft_noverlap,
            window_type=self.stft_window_type,
            scaling=self.stft_scaling,
            detrend=self.stft_detrend,
            use_log_scale=self.stft_use_log_scale,
            frequency_bands=_coerce_search_value(self.stft_frequency_bands),
            sampling_rate=self.imu_native_sampling_rate,
            window_seconds=self.stft_window_seconds,
            counter_col=self.counter_col,
            sequence_col=self.sequence_col,
        )

    def _build_cwt_defaults(self) -> Dict[str, Any]:
        return dict(
            wavelet=self.cwt_wavelet,
            max_scale=self.cwt_max_scale,
            n_scales=self.cwt_n_scales,
            use_log_scale=self.cwt_use_log_scale,
            min_scale=self.cwt_min_scale,
            max_frequency=self.cwt_max_frequency,
            sampling_rate=self.imu_native_sampling_rate,
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
        )

        self.motion_filter = MotionFilter(
            motion_filter_mode=self.motion_filter_mode,
            kalman_process_noise=self.kalman_process_noise,
            kalman_measurement_noise=self.kalman_measurement_noise,
            use_dead_reckoning=self.use_dead_reckoning,
            dead_reckoning_detrend=self.dead_reckoning_detrend,
            sequence_col=self.sequence_col,
        )

        self.imu = IMUExtractor(
            acc_modes=self.acc_modes,
            linear_acc_modes=self.linear_acc_modes,
            use_acc_magnitude=self.use_acc_magnitude,
            use_linear_acc_magnitude=self.use_linear_acc_magnitude,
            window_size=self.window_size,
            smooth_alpha=self.smooth_alpha,
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
            new_df = pd.DataFrame(index=range(max_len))
            new_df[self.sequence_col] = seq_id
            if self.counter_col in g.columns:
                counter = pd.to_numeric(g[self.counter_col], errors="coerce").to_numpy(dtype=float)
                if not np.isfinite(counter).all() or (len(counter) > 1 and np.any(np.diff(counter) <= 0)):
                    raise ValueError("Resampling requires strictly increasing finite sequence counters")
                if len(counter) > 1 and np.any(np.diff(counter) != 1):
                    raise ValueError(
                        "Polyphase resampling requires a uniform input grid; interpolate missing "
                        "counter positions before enabling resample_modalities"
                    )
                new_df[self.counter_col] = g[self.counter_col].to_numpy()
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
                vals = g[cols].to_numpy(dtype=float)
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
                    new_df[col] = vals[:, col_idx]
            if "dt" in g.columns:
                new_df["dt"] = g["dt"].to_numpy(dtype=float)
            else:
                new_df["dt"] = 1.0 / float(self.imu_native_sampling_rate)
            if "rot_dt" in g.columns:
                new_df["rot_dt"] = g["rot_dt"].to_numpy(dtype=float)
            else:
                new_df["rot_dt"] = 1.0 / float(self.rot_native_sampling_rate)
            new_df["mask"] = 1.0
            rows.append(new_df)
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
        combined = combined.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        combined = combined.loc[:, ~combined.columns.duplicated()]
        if self.sequence_col in processed.columns:
            combined[self.sequence_col] = processed[self.sequence_col].values
        elif self.sequence_col not in combined.columns:
            combined[self.sequence_col] = 0
        if add_context:
            feature_cols = [c for c in combined.columns if c not in self._non_feature_cols()]
            combined = self._append_global_context(combined, feature_cols)
        return combined

    def _append_global_context(self, out: pd.DataFrame, base_cols: List[str]) -> pd.DataFrame:
        out = out.copy()
        for col in base_cols:
            if col not in out.columns:
                continue
            out[f"{col}_global_mean"] = out.groupby(self.sequence_col, sort=False)[col].transform("mean")
            out[f"{col}_global_std"] = out.groupby(self.sequence_col, sort=False)[col].transform("std").fillna(0.0)
        return out

    def _preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.filter_problematic_sequences:
            X = self.problematic_filter_.transform(X)
            if X.empty:
                raise InvalidExtractorParams("Problematic-sequence filtering removed every input row.")
        cleaned = self.cleaner.transform(X)
        filtered = self.motion_filter.transform(cleaned)
        processed = self._maybe_resample(filtered)
        combined = self._combine_feature_outputs(processed, add_context=self.add_global_context)
        return combined

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

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None):
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
        ).fit(X)
        X_fit = self.problematic_filter_.transform(X) if self.filter_problematic_sequences else X
        if X_fit.empty:
            raise InvalidExtractorParams("Problematic-sequence filtering removed every training row.")

        cleaned = self.cleaner.fit_transform(X_fit)
        filtered = self.motion_filter.fit_transform(cleaned)
        processed = self._maybe_resample(filtered)

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
            combined_ctx = self._append_global_context(combined.copy(), self.base_feature_names_)
            self.feature_names_in_ = [c for c in combined_ctx.columns if c not in self._non_feature_cols()]
        else:
            self.feature_names_in_ = list(self.base_feature_names_)

        return self

    # ---------- transform ----------

    def transform(self, X: pd.DataFrame):
        check_is_fitted(self, ["base_feature_names_", "feature_names_in_"])
        if str(self.output_format).lower() == "frame":
            return self.transform_frame(X)
        return self.transform_chunks(X)

    def transform_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["base_feature_names_", "frame_feature_names_"])
        out = self._preprocess_features(X)
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
        out = self._preprocess_features(X)
        for col in self.feature_names_in_:
            if col not in out.columns:
                out[col] = 0.0

        grouped_sequences = list(out.groupby(self.sequence_col, sort=False))
        if not grouped_sequences:
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
                min(len(group), maxlen) if maxlen is not None else len(group)
                for _, group in grouped_sequences
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

        for seq_id, group in grouped_sequences:
            if len(group) == 0:
                continue
            sequences_total += 1
            arr = group[self.feature_names_in_].to_numpy(dtype=float)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            full_length = len(arr)
            arr = self._crop_sequence(arr)
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
                    padding = np.full((pad_length, arr.shape[1]), self.padding_value, dtype=float)
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

        return {
            "X": np.stack(sequences, axis=0).astype(np.float32),
            "mask": np.stack(masks, axis=0).astype(bool),
            "sequence_ids": np.array(seq_ids),
            "feature_names": list(self.feature_names_in_),
        }


# ---------------------------------------------------------------------------
# Random Forest sequence classifier (unchanged)
# ---------------------------------------------------------------------------


class RandomForestSequenceClassifier(BaseEstimator, ClassifierMixin):
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
            yy = y.copy()
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
            yy = y.copy()
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
    if search_mode != "bayesian":
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


# ============================================================
# Sensor augmentation
# ============================================================


def augment_jitter(x, sigma=0.03, rng=None):
    rng = rng or np.random.default_rng()
    return x + rng.normal(loc=0.0, scale=sigma, size=x.shape)


def augment_gaussian_noise(x, std=0.05, rng=None):
    rng = rng or np.random.default_rng()
    ch_std = np.nanstd(x, axis=0, keepdims=True)
    ch_std = np.where(ch_std == 0, 1.0, ch_std)
    return x + rng.normal(loc=0.0, scale=std * ch_std, size=x.shape)


def augment_scaling(x, sigma=0.1, rng=None):
    rng = rng or np.random.default_rng()
    factor = rng.normal(loc=1.0, scale=sigma)
    return x * factor


def augment_time_shift(x, max_shift_frac=0.1, rng=None):
    rng = rng or np.random.default_rng()
    T = x.shape[0]
    max_shift = max(1, int(T * max_shift_frac))
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
    rng = rng or np.random.default_rng()
    crop_frac_range = _normalize_crop_frac_range(crop_frac_range)
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
    rng = rng or np.random.default_rng()
    T = x.shape[0]
    mask_len = max(1, int(T * mask_frac))
    out = x.copy()
    for _ in range(num_masks):
        start = rng.integers(0, max(1, T - mask_len))
        out[start : start + mask_len] = 0.0
    return out


def augment_channel_dropout(x, drop_prob=0.1, rng=None):
    rng = rng or np.random.default_rng()
    C = x.shape[1]
    mask = rng.random(C) >= drop_prob
    return x * mask[np.newaxis, :]


def augment_sensor_dropout(x, sensor_groups, drop_prob=0.3, rng=None):
    rng = rng or np.random.default_rng()
    out = x.copy()
    for _name, col_idx in sensor_groups.items():
        if not col_idx:
            continue
        if rng.random() < drop_prob:
            out[:, col_idx] = 0.0
    return out


def augment_magnitude_warp(x, sigma=0.2, num_knots=4, rng=None):
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
    "acc":  ("acc_",),
    "rot":  ("rot_",),
    "tof":  ("tof_", "depth_"),
    "thm":  ("thm_", "thermal_", "temp_"),
}


def _detect_sensor_groups(columns):
    groups = {}
    for sensor, prefixes in _SENSOR_PREFIXES.items():
        idx = [i for i, c in enumerate(columns) if any(c.startswith(p) for p in prefixes)]
        groups[sensor] = idx
    return groups


class SensorAugmentor(BaseEstimator, TransformerMixin):
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

    def fit(self, X, y=None):
        self._rng = np.random.default_rng(self.seed)
        self.aug_names_ = list(
            self.augmentations if self.augmentations is not None else ALL_AUGMENTATION_NAMES
        )
        skip = {self.sequence_col, self.counter_col}
        self.sensor_cols_ = [
            c for c in X.columns
            if c not in skip and np.issubdtype(X[c].dtype, np.number)
        ]
        self.sensor_groups_ = _detect_sensor_groups(self.sensor_cols_)
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, "sensor_cols_")
        out = X.copy()
        for seq_id, grp in out.groupby(self.sequence_col, sort=False):
            if self._rng.random() > self.prob:
                continue
            idx = grp.index
            vals = grp[self.sensor_cols_].to_numpy(dtype=float)
            for aug_name in self.aug_names_:
                if self._rng.random() > self.per_aug_prob:
                    continue
                fn = _AUGMENTATION_REGISTRY[aug_name]
                kwargs = self._build_kwargs(aug_name)
                vals = fn(vals, rng=self._rng, **kwargs)
            out.loc[idx, self.sensor_cols_] = vals
        return out

    @staticmethod
    def _coerce_crop_frac_range(value):
        return _normalize_crop_frac_range(value)

    def _build_kwargs(self, name):
        crop_range = self._coerce_crop_frac_range(self.crop_frac_range)
        table = {
            "jitter":          {"sigma": self.jitter_sigma},
            "gaussian_noise":  {"std": self.noise_std},
            "scaling":         {"sigma": self.scaling_sigma},
            "time_shift":      {"max_shift_frac": self.time_shift_frac},
            "crop_resize":     {"crop_frac_range": crop_range},
            "temporal_mask":   {"mask_frac": self.temporal_mask_frac, "num_masks": self.temporal_num_masks},
            "channel_dropout": {"drop_prob": self.channel_drop_prob},
            "sensor_dropout":  {"sensor_groups": self.sensor_groups_, "drop_prob": self.sensor_drop_prob},
            "magnitude_warp":  {"sigma": self.warp_sigma, "num_knots": self.warp_num_knots},
        }
        return table.get(name, {})

    def get_feature_names_out(self, input_features=None):
        return np.array(self.sensor_cols_)