"""
base_utils_qwen.py

Cleaned and runnable Honeycomb / sequence feature extraction utilities.

Includes:
- SignalCleaner
- MotionFilter
- IMUExtractor
- RotationExtractor
- TOFExtractor
- ThermoExtractor
- SequenceExtractor
- RandomForestSequenceClassifier
- competition scoring utilities

This is designed so that:
- Deep learning models can continue using chunk output.
- Random Forest can use sequence-level frame output.
"""

from __future__ import annotations

import json
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
    from scipy.signal import resample
except Exception:
    resample = None

try:
    from skopt.space import Categorical
except Exception:
    Categorical = None


class InvalidExtractorParams(ValueError):
    """Raised when extractor hyperparameters cannot produce valid features."""


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
    """
    Fail fast on extractor configs that are known to break CV trials.

    Raises InvalidExtractorParams so GridSearchCV / BayesSearchCV can assign
    error_score and continue with the next candidate.
    """

    output = str(params.get("output_format", "frame")).lower()
    if for_frame_output and output not in {"frame"}:
        raise InvalidExtractorParams(
            f"Sequence frame classifiers require output_format='frame', got {output!r}"
        )

    if params.get("window_size") is not None:
        _positive_int(params.get("window_size"), name="window_size")

    if params.get("maxlen") is not None:
        _positive_int(params.get("maxlen"), name="maxlen")

    resample_modalities = bool(params.get("resample_modalities", False))

    rate_specs = [
        ("imu_native_sampling_rate", "imu_target_sampling_rate"),
        ("rot_native_sampling_rate", "rot_target_sampling_rate"),
        ("tof_native_sampling_rate", "tof_target_sampling_rate"),
        ("thm_native_sampling_rate", "thm_target_sampling_rate"),
    ]

    for native_key, target_key in rate_specs:
        native = _positive_int(params.get(native_key, 20), name=native_key)
        target = _positive_int(
            params.get(target_key, native),
            name=target_key,
        )

        if resample_modalities:
            if resample is None:
                raise InvalidExtractorParams(
                    "resample_modalities=True requires scipy.signal.resample"
                )

            ratio = target / float(native)
            if ratio > 25 or ratio < 0.04:
                raise InvalidExtractorParams(
                    f"{target_key}/{native_key} ratio {ratio:.3f} is outside safe bounds"
                )

    if not for_frame_output:
        chunk_window = params.get("chunk_window_size")
        chunk_stride = params.get("chunk_stride")
        if chunk_window is not None and chunk_stride is not None:
            win = _positive_int(chunk_window, name="chunk_window_size")
            stride = _positive_int(chunk_stride, name="chunk_stride")
            if stride > win:
                raise InvalidExtractorParams(
                    "chunk_stride must be <= chunk_window_size"
                )


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class HoneycombBase(BaseEstimator, TransformerMixin):
    """Parent class for all Honeycomb components."""

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


# ---------------------------------------------------------------------------
# Signal cleaning
# ---------------------------------------------------------------------------


class SignalCleaner(HoneycombBase):
    """
    Handles interpolation, dt computation, clipping, masking, and optional
    linear acceleration estimation.
    """

    def __init__(
        self,
        native_sampling_rate: int = 20,
        compute_dt: bool = True,
        clip_value: Optional[float] = None,
        interp_mode: str = "linear",
        linear_acc_mode: Optional[str] = None,
        use_highpass_fallback: bool = True,
        window_size: int = 5,
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
        self.sequence_col = sequence_col
        self.counter_col = counter_col

    def fit(self, X: pd.DataFrame, y=None):
        self.acc_cols_ = [c for c in X.columns if c.startswith("acc_")]
        self.rot_cols_ = [c for c in X.columns if c.startswith("rot_")]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [
            c
            for c in num_cols
            if c not in {self.sequence_col, self.counter_col}
        ]

        if num_cols:
            if self.interp_mode == "linear":
                df[num_cols] = df.groupby(self.sequence_col, sort=False)[num_cols].transform(
                    lambda g: g.interpolate(method="linear", limit_direction="both").ffill().bfill()
                )
            elif self.interp_mode == "ffill":
                df[num_cols] = df.groupby(self.sequence_col, sort=False)[num_cols].ffill().bfill()

        if self.compute_dt:
            if self.counter_col in df.columns:
                df[self.counter_col] = pd.to_numeric(df[self.counter_col], errors="coerce")
                df["dt"] = (
                    df.groupby(self.sequence_col, sort=False)[self.counter_col]
                    .diff()
                    .fillna(1.0)
                    / float(self.native_sampling_rate)
                )
            else:
                df["dt"] = 1.0 / float(self.native_sampling_rate)

            df["dt"] = (
                df["dt"]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(1.0 / float(self.native_sampling_rate))
                .clip(lower=1e-6)
            )
        else:
            df["dt"] = 1.0 / float(self.native_sampling_rate)

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

        df["mask"] = 1.0
        return df

    def _add_linear_acceleration(self, df: pd.DataFrame) -> pd.DataFrame:
        acc_cols = [c for c in ["acc_x", "acc_y", "acc_z"] if c in df.columns]
        if len(acc_cols) < 3:
            acc_cols = [c for c in self.acc_cols_ if c in df.columns][:3]

        rot_cols = [c for c in ["rot_w", "rot_x", "rot_y", "rot_z"] if c in df.columns]
        if len(rot_cols) < 4:
            rot_cols = [c for c in self.rot_cols_ if c in df.columns][:4]

        if len(acc_cols) != 3 or len(rot_cols) != 4:
            if self.use_highpass_fallback:
                return self._linear_acc_highpass(df, acc_cols)
            return df

        acc = df[acc_cols].to_numpy(dtype=float)
        q = df[rot_cols].to_numpy(dtype=float)

        acc = np.nan_to_num(acc, nan=0.0, posinf=0.0, neginf=0.0)
        q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)

        norms = np.linalg.norm(q, axis=1, keepdims=True)
        bad = (norms == 0) | ~np.isfinite(norms)
        norms[bad] = 1.0
        q = q / norms
        q[bad[:, 0]] = np.array([1.0, 0.0, 0.0, 0.0])

        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        R = np.zeros((len(q), 3, 3))
        R[:, 0, 0] = 1 - 2 * y**2 - 2 * z**2
        R[:, 0, 1] = 2 * x * y - 2 * w * z
        R[:, 0, 2] = 2 * x * z + 2 * w * y

        R[:, 1, 0] = 2 * x * y + 2 * w * z
        R[:, 1, 1] = 1 - 2 * x**2 - 2 * z**2
        R[:, 1, 2] = 2 * y * z - 2 * w * x

        R[:, 2, 0] = 2 * x * z - 2 * w * y
        R[:, 2, 1] = 2 * y * z + 2 * w * x
        R[:, 2, 2] = 1 - 2 * x**2 - 2 * y**2

        gravity = np.array([0.0, 0.0, 9.81])
        acc_world = np.einsum("nij,nj->ni", R, acc)
        lin_acc = acc_world - gravity

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
            df[f"lin_{col}"] = df[col] - baseline

        return df


# ---------------------------------------------------------------------------
# Motion filter / dead reckoning
# ---------------------------------------------------------------------------


class MotionFilter(HoneycombBase):
    """
    Applies optional Kalman filtering and/or groupwise dead reckoning to IMU streams.

    Important fix:
    - dead reckoning is now computed per sequence, not across sequence boundaries.
    """

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
            for col in acc_cols:
                df[col] = df.groupby(self.sequence_col, sort=False)[col].transform(
                    lambda g: self._kalman_filter_1d(g.to_numpy())
                )

        if self.use_dead_reckoning and acc_cols and "dt" in df.columns:
            for col in acc_cols:
                vel_col = f"{col}_dr_vel"
                pos_col = f"{col}_dr_pos"

                vel_inc = df[col] * df["dt"]
                df[vel_col] = vel_inc.groupby(df[self.sequence_col], sort=False).cumsum()

                if self.dead_reckoning_detrend:
                    df[vel_col] = df.groupby(self.sequence_col, sort=False)[vel_col].transform(
                        self._detrend_series
                    )

                pos_inc = df[vel_col] * df["dt"]
                df[pos_col] = pos_inc.groupby(df[self.sequence_col], sort=False).cumsum()

                if self.dead_reckoning_detrend:
                    df[pos_col] = df.groupby(self.sequence_col, sort=False)[pos_col].transform(
                        self._detrend_series
                    )

        return df


# ---------------------------------------------------------------------------
# IMU features
# ---------------------------------------------------------------------------


class IMUExtractor(HoneycombBase):
    """Extracts multi-domain accelerometer features."""

    def __init__(
        self,
        acc_modes: str = "raw",
        use_acc_magnitude: bool = False,
        use_linear_acc_magnitude: bool = False,
        window_size: int = 5,
        smooth_alpha: Optional[float] = None,
        sequence_col: str = "sequence_id",
    ):
        self.acc_modes = acc_modes
        self.use_acc_magnitude = use_acc_magnitude
        self.use_linear_acc_magnitude = use_linear_acc_magnitude
        self.window_size = window_size
        self.smooth_alpha = smooth_alpha
        self.sequence_col = sequence_col

    def fit(self, X: pd.DataFrame, y=None):
        self.acc_modes_ = self._parse_modes(self.acc_modes)
        self.acc_cols_ = [
            c
            for c in X.columns
            if c.startswith("acc_") and not c.startswith("lin_acc")
        ]
        self.lin_cols_ = [c for c in X.columns if c.startswith("lin_acc")]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["acc_modes_", "acc_cols_"])

        df = X.copy()
        parts = []

        if "dt" not in df.columns:
            df["dt"] = 1.0

        dt = df["dt"].astype(float)

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
                        lambda g: g.rolling(
                            window=self.window_size,
                            center=True,
                            min_periods=1,
                        ).mean()
                    )
                parts.append(out.add_suffix("_smooth"))

            elif mode == "velocity":
                vel = (
                    df[self.acc_cols_]
                    .mul(dt, axis=0)
                    .groupby(df[self.sequence_col].values, sort=False)
                    .cumsum()
                )
                parts.append(vel.add_suffix("_vel"))

            elif mode == "displacement":
                vel = (
                    df[self.acc_cols_]
                    .mul(dt, axis=0)
                    .groupby(df[self.sequence_col].values, sort=False)
                    .cumsum()
                )
                disp = (
                    vel.mul(dt, axis=0)
                    .groupby(df[self.sequence_col].values, sort=False)
                    .cumsum()
                )
                parts.append(disp.add_suffix("_disp"))

            elif mode == "jerk":
                jerk = (
                    df.groupby(self.sequence_col, sort=False)[self.acc_cols_]
                    .diff()
                    .div(dt, axis=0)
                    .fillna(0.0)
                )
                parts.append(jerk.add_suffix("_jerk"))

        if self.use_acc_magnitude and self.acc_cols_:
            mag = np.sqrt(df[self.acc_cols_].pow(2).sum(axis=1))
            parts.append(pd.DataFrame({"acc_mag": mag}, index=df.index))

        if self.use_linear_acc_magnitude and self.lin_cols_:
            mag = np.sqrt(df[self.lin_cols_].pow(2).sum(axis=1))
            parts.append(pd.DataFrame({"lin_acc_mag": mag}, index=df.index))

        if not parts:
            return pd.DataFrame(index=df.index)

        return pd.concat(parts, axis=1)


# ---------------------------------------------------------------------------
# Rotation features
# ---------------------------------------------------------------------------


class RotationExtractor(HoneycombBase):
    """Extracts multi-domain rotation features."""

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
        self.rot_cols_ = [c for c in X.columns if c.startswith("rot_")]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["rot_modes_", "rot_cols_"])

        df = X.copy()
        parts = []

        if "dt" not in df.columns:
            df["dt"] = 1.0

        dt = df["dt"].astype(float)

        if not self.rot_cols_:
            return pd.DataFrame(index=df.index)

        q = df[self.rot_cols_].to_numpy(dtype=float, copy=True)
        q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)

        norms = np.linalg.norm(q, axis=1, keepdims=True)
        bad = (norms == 0) | ~np.isfinite(norms)
        norms[bad] = 1.0
        q = q / norms
        q[bad[:, 0]] = np.array([1.0, 0.0, 0.0, 0.0])

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
                        {
                            "rot_roll": roll,
                            "rot_pitch": pitch,
                            "rot_yaw": yaw,
                        },
                        index=df.index,
                    )
                )

            elif mode == "delta_euler":
                w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

                roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
                pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
                yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

                euler_df = pd.DataFrame(
                    {
                        "rot_roll": roll,
                        "rot_pitch": pitch,
                        "rot_yaw": yaw,
                    },
                    index=df.index,
                )

                delta = (
                    euler_df.groupby(df[self.sequence_col].values, sort=False)
                    .diff()
                    .fillna(0.0)
                )
                parts.append(delta.add_suffix("_delta"))

            elif mode == "angular_velocity":
                ang_vel = (
                    df.groupby(self.sequence_col, sort=False)[self.rot_cols_]
                    .diff()
                    .div(dt, axis=0)
                    .fillna(0.0)
                )
                parts.append(ang_vel.add_suffix("_angvel"))

                ang_vel_mag = np.sqrt(ang_vel.pow(2).sum(axis=1))
                parts.append(pd.DataFrame({"ang_vel_mag": ang_vel_mag}, index=df.index))

            elif mode == "rot6d":
                w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

                R = np.zeros((len(q), 3, 3))

                R[:, 0, 0] = 1 - 2 * y**2 - 2 * z**2
                R[:, 0, 1] = 2 * x * y - 2 * w * z
                R[:, 0, 2] = 2 * x * z + 2 * w * y

                R[:, 1, 0] = 2 * x * y + 2 * w * z
                R[:, 1, 1] = 1 - 2 * x**2 - 2 * z**2
                R[:, 1, 2] = 2 * y * z - 2 * w * x

                rot6d = np.concatenate([R[:, :, 0], R[:, :, 1]], axis=1)
                cols = [
                    "rot6d_c1_x",
                    "rot6d_c1_y",
                    "rot6d_c1_z",
                    "rot6d_c2_x",
                    "rot6d_c2_y",
                    "rot6d_c2_z",
                ]
                parts.append(pd.DataFrame(rot6d, columns=cols, index=df.index))

        if not parts:
            return pd.DataFrame(index=df.index)

        return pd.concat(parts, axis=1)


# ---------------------------------------------------------------------------
# ToF features
# ---------------------------------------------------------------------------


class TOFExtractor(HoneycombBase):
    """Extracts Time-of-Flight features."""

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

        df = X[self.tof_cols_].astype(float).copy()
        df = df.replace(-1.0, np.nan)

        if self.tof_fill_mode == "nan_interpolate":
            df = df.groupby(X[self.sequence_col].values, sort=False).transform(
                lambda g: g.interpolate(method="linear", limit_direction="both").ffill().bfill()
            ).fillna(255.0)
        elif self.tof_fill_mode == "zero":
            df = df.fillna(0.0)
        elif self.tof_fill_mode == "far_255":
            df = df.fillna(255.0)
        elif self.tof_fill_mode == "far_500":
            df = df.fillna(500.0)
        else:
            df = df.fillna(0.0)

        parts = []

        sensor_map: Dict[str, List[str]] = {}
        for col in df.columns:
            parts_col = str(col).split("_")
            key = parts_col[1] if len(parts_col) > 1 else "all"
            sensor_map.setdefault(key, []).append(col)

        if not sensor_map:
            sensor_map = {"all": list(df.columns)}

        for mode in self.tof_modes_:
            if mode == "raw":
                parts.append(df.add_suffix("_raw"))

            elif mode == "sensor_stats":
                for sensor_key, cols in sensor_map.items():
                    if not cols:
                        continue

                    arr = df[cols].to_numpy(dtype=float)
                    parts.append(
                        pd.DataFrame(
                            {
                                f"tof_{sensor_key}_mean": np.nanmean(arr, axis=1),
                                f"tof_{sensor_key}_std": np.nanstd(arr, axis=1),
                                f"tof_{sensor_key}_min": np.nanmin(arr, axis=1),
                                f"tof_{sensor_key}_max": np.nanmax(arr, axis=1),
                            },
                            index=X.index,
                        )
                    )

            elif mode == "pooled_stats":
                arr = df.to_numpy(dtype=float)
                parts.append(
                    pd.DataFrame(
                        {
                            "tof_pooled_mean": np.nanmean(arr, axis=1),
                            "tof_pooled_std": np.nanstd(arr, axis=1),
                            "tof_pooled_min": np.nanmin(arr, axis=1),
                            "tof_pooled_max": np.nanmax(arr, axis=1),
                        },
                        index=X.index,
                    )
                )

            elif mode == "pooled":
                for sensor_key, cols in sensor_map.items():
                    if len(cols) != 64:
                        continue

                    arr = df[cols].to_numpy(dtype=float).reshape(-1, 8, 8)
                    pool = (
                        arr.reshape(-1, 4, 2, 4, 2)
                        .mean(axis=(2, 4))
                        .reshape(-1, 16)
                    )
                    parts.append(
                        pd.DataFrame(
                            pool,
                            columns=[f"tof_{sensor_key}_pool_{i}" for i in range(16)],
                            index=X.index,
                        )
                    )

        if not parts:
            return pd.DataFrame(index=X.index)

        return pd.concat(parts, axis=1)


# ---------------------------------------------------------------------------
# Thermopile features
# ---------------------------------------------------------------------------


class ThermoExtractor(HoneycombBase):
    """Extracts Thermopile features."""

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

        if not parts:
            return pd.DataFrame(index=X.index)

        return pd.concat(parts, axis=1)


# ---------------------------------------------------------------------------
# Sequence extractor
# ---------------------------------------------------------------------------


class SequenceExtractor(HoneycombBase):
    """
    Orchestrates cleaning, motion filtering, and multi-domain extraction.

    output_format:
      - "chunks": returns dict with X, mask, sequence_ids, feature_names
      - "frame": returns pandas DataFrame, one row per sequence

    Important fixes:
    - frame output is sequence-level and aligned by sequence_id.
    - chunk output includes mask.
    - resampling is optional and disabled by default.
    """

    def __init__(
        self,
        acc_modes: str = "raw|velocity|displacement|jerk",
        rotation_modes: str = "quaternion",
        tof_modes: str = "sensor_stats",
        thm_modes: str = "centered_diff",
        motion_filter_mode: Optional[str] = None,
        use_dead_reckoning: bool = False,
        dead_reckoning_detrend: bool = False,
        kalman_process_noise: float = 1e-3,
        kalman_measurement_noise: float = 1e-2,
        compute_dt: bool = True,
        window_size: int = 7,
        smooth_alpha: Optional[float] = None,
        clip_value: Optional[float] = None,
        interp_mode: str = "linear",
        maxlen: int = 160,
        padding_value: float = -999.0,
        sequence_col: str = "sequence_id",
        counter_col: str = "sequence_counter",
        imu_native_sampling_rate: int = 20,
        imu_target_sampling_rate: int = 20,
        rot_native_sampling_rate: int = 20,
        rot_target_sampling_rate: int = 20,
        tof_native_sampling_rate: int = 5,
        tof_target_sampling_rate: int = 5,
        thm_native_sampling_rate: int = 5,
        thm_target_sampling_rate: int = 5,
        chunk_window_size: Optional[int] = 128,
        chunk_stride: Optional[int] = 64,
        output_format: str = "chunks",
        frame_stats: str = "mean,std,min,max,last",
        add_global_context: bool = False,
        resample_modalities: bool = False,
    ):
        self.acc_modes = acc_modes
        self.rotation_modes = rotation_modes
        self.tof_modes = tof_modes
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
        self.output_format = output_format
        self.frame_stats = frame_stats
        self.add_global_context = add_global_context
        self.resample_modalities = resample_modalities

        self.cleaner = SignalCleaner(
            native_sampling_rate=self.imu_native_sampling_rate,
            compute_dt=self.compute_dt,
            clip_value=self.clip_value,
            interp_mode=self.interp_mode,
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
            window_size=self.window_size,
            smooth_alpha=self.smooth_alpha,
            sequence_col=self.sequence_col,
        )

        self.rotation = RotationExtractor(
            rotation_modes=self.rotation_modes,
            sequence_col=self.sequence_col,
        )

        self.tof = TOFExtractor(
            tof_modes=self.tof_modes,
            sequence_col=self.sequence_col,
        )

        self.thermo = ThermoExtractor(
            thm_modes=self.thm_modes,
            sequence_col=self.sequence_col,
        )

    # ----------------------------- helpers -----------------------------

    def _non_feature_cols(self) -> List[str]:
        cols = {self.sequence_col, self.counter_col, "dt", "mask"}
        return [c for c in cols if c is not None]

    def _parse_frame_stats(self) -> List[str]:
        known = {
            "mean",
            "std",
            "min",
            "max",
            "first",
            "last",
            "median",
            "rms",
            "abs_mean",
        }

        stats = [
            s.strip().lower()
            for s in str(self.frame_stats).split(",")
            if s.strip().lower() in known
        ]

        if not stats:
            stats = ["mean"]

        return stats

    def _maybe_resample(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optional resampling.

        Disabled by default because multimodal resampling is dangerous unless
        the sensor clock model is known exactly.
        """

        if not self.resample_modalities or resample is None:
            return df

        try:
            return self._maybe_resample_inner(df)
        except InvalidExtractorParams:
            raise
        except Exception as exc:
            raise InvalidExtractorParams(f"Multimodal resampling failed: {exc}") from exc

    def _maybe_resample_inner(self, df: pd.DataFrame) -> pd.DataFrame:
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

            for prefix, (native_rate, target_rate) in rate_map.items():
                cols = [
                    c
                    for c in g.columns
                    if c.startswith(prefix)
                    and c not in {self.sequence_col, self.counter_col, "dt", "mask"}
                ]

                if not cols:
                    continue

                native_rate = max(1, int(native_rate or 1))
                target_rate = max(1, int(target_rate or native_rate))

                target_len = int(round(len(g) * target_rate / native_rate))
                if target_len <= 0:
                    continue

                vals = g[cols].fillna(0.0).to_numpy(dtype=float)

                if target_len != len(g):
                    vals = resample(vals, target_len, axis=0)

                s = pd.DataFrame(vals, columns=cols)

                if len(s) != max_len:
                    old_index = np.linspace(0.0, 1.0, len(s))
                    new_index = np.linspace(0.0, 1.0, max_len)
                    s = s.copy()
                    s.index = pd.Index(old_index)
                    s = s.reindex(new_index)
                    s = s.interpolate(method="linear", limit_direction="both").ffill().bfill()

                if len(s) != max_len:
                    raise InvalidExtractorParams(
                        f"Resampling {prefix} columns failed to align to sequence length"
                    )

                for col in cols:
                    if col in s.columns:
                        new_df[col] = s[col].to_numpy()
                    else:
                        new_df[col] = 0.0

            if "dt" in g.columns:
                new_df["dt"] = float(np.nanmean(g["dt"].fillna(1.0 / 20.0)))
            else:
                new_df["dt"] = 1.0 / 20.0

            new_df["mask"] = 1.0
            rows.append(new_df)

        if not rows:
            return df

        return pd.concat(rows, ignore_index=True)

    def _combine_feature_outputs(self, processed: pd.DataFrame, add_context: bool = False) -> pd.DataFrame:
        outs = []

        for est in (self.imu, self.rotation, self.tof, self.thermo):
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
            feature_cols = [
                c
                for c in combined.columns
                if c not in self._non_feature_cols()
            ]
            combined = self._append_global_context(combined, feature_cols)

        return combined

    def _append_global_context(self, out: pd.DataFrame, base_cols: List[str]) -> pd.DataFrame:
        """
        Calculates sequence-level global mean and std for each base feature
        and appends them as new columns.
        """

        out = out.copy()

        for col in base_cols:
            if col not in out.columns:
                continue

            out[f"{col}_global_mean"] = out.groupby(self.sequence_col, sort=False)[col].transform("mean")
            out[f"{col}_global_std"] = out.groupby(self.sequence_col, sort=False)[col].transform("std").fillna(0.0)

        return out

    def _preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        cleaned = self.cleaner.transform(X)
        filtered = self.motion_filter.transform(cleaned)
        processed = self._maybe_resample(filtered)
        combined = self._combine_feature_outputs(processed, add_context=self.add_global_context)
        return combined

    def _stat_vector(self, arr: np.ndarray, stat: str) -> np.ndarray:
        if arr.size == 0:
            return np.zeros((arr.shape[1] if arr.ndim == 2 else 1,), dtype=float)

        if stat == "mean":
            return np.nanmean(arr, axis=0)

        if stat == "std":
            return np.nanstd(arr, axis=0)

        if stat == "min":
            return np.nanmin(arr, axis=0)

        if stat == "max":
            return np.nanmax(arr, axis=0)

        if stat == "first":
            return arr[0]

        if stat == "last":
            return arr[-1]

        if stat == "median":
            return np.nanmedian(arr, axis=0)

        if stat == "rms":
            return np.sqrt(np.nanmean(np.square(arr), axis=0))

        if stat == "abs_mean":
            return np.nanmean(np.abs(arr), axis=0)

        return np.nanmean(arr, axis=0)

    # ----------------------------- fit -----------------------------

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None):
        validate_sequence_extractor_params(
            self.get_params(),
            for_frame_output=str(self.output_format).lower() == "frame",
        )

        self.cleaner = SignalCleaner(
            native_sampling_rate=self.imu_native_sampling_rate,
            compute_dt=self.compute_dt,
            clip_value=self.clip_value,
            interp_mode=self.interp_mode,
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
            window_size=self.window_size,
            smooth_alpha=self.smooth_alpha,
            sequence_col=self.sequence_col,
        )

        self.rotation = RotationExtractor(
            rotation_modes=self.rotation_modes,
            sequence_col=self.sequence_col,
        )

        self.tof = TOFExtractor(
            tof_modes=self.tof_modes,
            sequence_col=self.sequence_col,
        )

        self.thermo = ThermoExtractor(
            thm_modes=self.thm_modes,
            sequence_col=self.sequence_col,
        )

        cleaned = self.cleaner.fit_transform(X)
        filtered = self.motion_filter.fit_transform(cleaned)
        processed = self._maybe_resample(filtered)

        self.imu.fit(processed)
        self.rotation.fit(processed)
        self.tof.fit(processed)
        self.thermo.fit(processed)

        combined = self._combine_feature_outputs(processed, add_context=False)

        feature_cols = [
            c
            for c in combined.columns
            if c not in self._non_feature_cols()
        ]

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
            self.feature_names_in_ = [
                c
                for c in combined_ctx.columns
                if c not in self._non_feature_cols()
            ]
        else:
            self.feature_names_in_ = list(self.base_feature_names_)

        return self

    # ----------------------------- transform -----------------------------

    def transform(self, X: pd.DataFrame):
        check_is_fitted(self, ["base_feature_names_", "feature_names_in_"])

        if str(self.output_format).lower() == "frame":
            return self.transform_frame(X)

        return self.transform_chunks(X)

    def transform_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a pandas DataFrame with one row per sequence.

        The index is sequence_id sorted ascending, which makes it deterministic
        for sklearn scorers and holdout evaluation.
        """

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

            stat_vectors = []
            for stat in self.frame_stats_:
                stat_vectors.append(self._stat_vector(arr, stat))

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

    def transform_chunks(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Returns chunked sequence tensors for deep models.

        Output dict:
        {
            "X": np.ndarray shape (n_chunks, window, n_features),
            "mask": np.ndarray shape (n_chunks, window),
            "sequence_ids": np.ndarray shape (n_chunks,),
            "feature_names": List[str]
        }
        """

        check_is_fitted(self, ["feature_names_in_"])

        out = self._preprocess_features(X)

        for col in self.feature_names_in_:
            if col not in out.columns:
                out[col] = 0.0

        sequences = []
        masks = []
        seq_ids = []

        for seq_id, g in out.groupby(self.sequence_col, sort=False):
            if len(g) == 0:
                continue

            arr = g[self.feature_names_in_].to_numpy(dtype=float)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

            L = len(arr)
            if L == 0:
                continue

            win = self.chunk_window_size
            if win is None or int(win) <= 0:
                win = L

            stride = self.chunk_stride
            if stride is None or int(stride) <= 0:
                stride = win

            if L <= win:
                pad_len = win - L
                if pad_len > 0:
                    pad = np.full((pad_len, arr.shape[1]), self.padding_value, dtype=float)
                    chunk = np.vstack([arr, pad])
                    mask = np.concatenate([np.ones(L, dtype=bool), np.zeros(pad_len, dtype=bool)])
                else:
                    chunk = arr[:win]
                    mask = np.ones(win, dtype=bool)

                sequences.append(chunk)
                masks.append(mask)
                seq_ids.append(seq_id)

            else:
                starts = np.arange(0, L - win + 1, stride)
                if len(starts) == 0 or starts[-1] + win < L:
                    starts = np.append(starts, L - win)

                for start in starts:
                    chunk = arr[start : start + win]
                    mask = np.ones(win, dtype=bool)

                    sequences.append(chunk)
                    masks.append(mask)
                    seq_ids.append(seq_id)

        if not sequences:
            raise ValueError("SequenceExtractor.transform_chunks produced no sequences.")

        return {
            "X": np.stack(sequences, axis=0).astype(np.float32),
            "mask": np.stack(masks, axis=0).astype(bool),
            "sequence_ids": np.array(seq_ids),
            "feature_names": list(self.feature_names_in_),
        }


# ---------------------------------------------------------------------------
# Random Forest sequence classifier
# ---------------------------------------------------------------------------


class RandomForestSequenceClassifier(BaseEstimator, ClassifierMixin):
    """
    Sequence-level Random Forest classifier.

    This estimator:
    - accepts raw row-level dataframe input,
    - uses SequenceExtractor to create one feature vector per sequence,
    - aligns labels to sequence_id,
    - fits RandomForestClassifier on sequence-level features,
    - returns sequence-level predictions indexed by sequence_id.

    This makes it usable in GridSearchCV / BayesSearchCV in a style similar to
    the multibranch Keras classifier.
    """

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
            output_format="frame",
            chunk_window_size=None,
            padding_value=0.0,
            add_global_context=False,
            frame_stats="mean,std,min,max,last",
            resample_modalities=False,
        )

    def _default_estimator(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=300,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )

    def _align_y(self, sequence_ids: pd.Index, y: Any) -> np.ndarray:
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

            y_seq = (
                yy.drop_duplicates(seq_col)
                .set_index(seq_col)[self.primary_target]
            )

            aligned = y_seq.reindex(sequence_ids)
            aligned = aligned.fillna(fill_value)
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

    def _validate_rf_extractor(self) -> None:
        if self.extractor_ is None:
            return

        if hasattr(self.extractor_, "output_format"):
            try:
                self.extractor_.set_params(output_format="frame")
            except Exception:
                pass

        validate_sequence_extractor_params(
            self.extractor_.get_params(),
            for_frame_output=True,
        )

    def _fit_inner(self, X: pd.DataFrame, y: Any) -> pd.DataFrame:
        self.extractor_.fit(X)

        if hasattr(self.extractor_, "transform_frame"):
            frame = self.extractor_.transform_frame(X)
        else:
            frame = self.extractor_.transform(X)

        if not isinstance(frame, pd.DataFrame):
            raise InvalidExtractorParams(
                "Extractor did not return a pandas DataFrame for frame output."
            )

        if frame.empty:
            raise InvalidExtractorParams(
                "Feature extraction produced zero sequences."
            )

        if not np.isfinite(frame.to_numpy(dtype=float, copy=False)).all():
            raise InvalidExtractorParams(
                "Feature extraction produced non-finite values."
            )

        y_aligned = self._align_y(frame.index, y)

        self.le_ = LabelEncoder()
        self.le_.fit(y_aligned)
        self.classes_ = self.le_.classes_

        y_enc = self.le_.transform(y_aligned)

        self.estimator_ = (
            clone(self.estimator)
            if self.estimator is not None
            else self._default_estimator()
        )

        self.estimator_.fit(frame, y_enc)
        return frame

    def fit(self, X: pd.DataFrame, y: Any = None, **fit_params):
        if y is None:
            raise ValueError("RandomForestSequenceClassifier requires y.")

        self.extractor_ = (
            clone(self.extractor)
            if self.extractor is not None
            else self._default_extractor()
        )

        self._validate_rf_extractor()

        try:
            self._fit_inner(X, y)
        except InvalidExtractorParams:
            raise
        except Exception as exc:
            raise InvalidExtractorParams(
                f"RandomForestSequenceClassifier fit failed: {exc}"
            ) from exc

        self.history_ = {
            "loss": [0.0],
            "accuracy": [1.0],
        }

        return self

    def _transform_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["extractor_"])

        try:
            if hasattr(self.extractor_, "transform_frame"):
                frame = self.extractor_.transform_frame(X)
            else:
                frame = self.extractor_.transform(X)
        except Exception as exc:
            raise InvalidExtractorParams(
                f"RandomForestSequenceClassifier transform failed: {exc}"
            ) from exc

        if not isinstance(frame, pd.DataFrame):
            raise InvalidExtractorParams(
                "Extractor did not return a pandas DataFrame for frame output."
            )

        if frame.empty:
            raise InvalidExtractorParams(
                "Feature extraction produced zero sequences."
            )

        return frame

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["estimator_", "le_"])

        frame = self._transform_frame(X)
        probs = self.estimator_.predict_proba(frame)

        return pd.DataFrame(
            probs,
            index=frame.index,
            columns=self.estimator_.classes_,
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        check_is_fitted(self, ["estimator_", "le_"])

        frame = self._transform_frame(X)
        pred_enc = self.estimator_.predict(frame)
        preds = self.le_.inverse_transform(pred_enc)

        return pd.Series(
            preds,
            index=frame.index,
            name=self.primary_target,
        ).sort_index()

    def score(self, X: pd.DataFrame, y: Any, sample_weight=None) -> float:
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
                preds_aligned = preds.reindex(y_seq[seq_col]).to_numpy()
            else:
                preds_aligned = np.asarray(preds)

            y_true = y_seq[self.primary_target].values

            if "is_target" in y_seq.columns:
                y_true_binary = y_seq["is_target"].astype(int).values
            else:
                y_true_binary = (y_true != "non_bfrb").astype(int)

            if self.primary_target == "bfrb":
                return competition_score(
                    y_true,
                    preds_aligned,
                    y_true_binary=y_true_binary,
                    target_only_macro=True,
                )

            return f1_score(
                y_true,
                preds_aligned,
                average="macro",
                zero_division=0,
            )

        y_arr = np.asarray(y)
        preds_arr = preds.to_numpy() if hasattr(preds, "to_numpy") else np.asarray(preds)

        return f1_score(
            y_arr,
            preds_arr,
            average="macro",
            zero_division=0,
        )

    def get_history_dict(self) -> Dict[str, List[float]]:
        return getattr(self, "history_", {})

    def summarize_model(self) -> None:
        if hasattr(self, "estimator_"):
            print(self.estimator_)
        else:
            print("Model is not fitted yet.")


# ---------------------------------------------------------------------------
# Competition scoring
# ---------------------------------------------------------------------------


def competition_score(
    y_true_gesture,
    y_pred,
    y_true_binary=None,
    target_only_macro: bool = True,
) -> float:
    """
    Competition metric:

        (Binary F1 + Macro F1) / 2

    Binary F1:
        target vs non-target

    Macro F1:
        gesture classes, by default computed on target sequences only.
    """

    y_true_gesture = np.asarray(y_true_gesture)
    y_pred = np.asarray(y_pred)

    if y_true_binary is None:
        y_true_binary = (y_true_gesture != "non_bfrb").astype(int)
    else:
        y_true_binary = np.asarray(y_true_binary).astype(int)

    y_pred_binary = (y_pred != "non_bfrb").astype(int)

    binary_f1 = f1_score(
        y_true_binary,
        y_pred_binary,
        zero_division=0,
    )

    if target_only_macro:
        mask = y_true_binary == 1
        if mask.sum() > 0:
            macro_f1 = f1_score(
                y_true_gesture[mask],
                y_pred[mask],
                average="macro",
                zero_division=0,
            )
        else:
            macro_f1 = 0.0
    else:
        macro_f1 = f1_score(
            y_true_gesture,
            y_pred,
            average="macro",
            zero_division=0,
        )

    return (binary_f1 + macro_f1) / 2.0


def make_competition_scorer(target_col: str = "bfrb"):
    """
    Sklearn CV scorer aligned with row-level y and sequence-level predictions.
    """

    seq_col = "sequence_id"

    def _score(y_true, y_pred):
        if isinstance(y_true, pd.DataFrame):
            if seq_col not in y_true.columns:
                if y_true.index.name == seq_col:
                    y_true = y_true.reset_index()
                else:
                    raise ValueError("y_true must contain sequence_id.")

            y_seq = (
                y_true.drop_duplicates(seq_col)
                .sort_values(seq_col)
            )

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

        return competition_score(
            y_true_gesture,
            y_pred,
            y_true_binary=y_true_binary,
            target_only_macro=True,
        )

    return make_scorer(_score)


competition_scorer = make_competition_scorer("bfrb")


def evaluate_holdout(
    y_test_df: pd.DataFrame,
    y_pred,
    target_col: str = "bfrb",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Final holdout evaluation.

    Returns dict with:
    - binary_f1
    - gesture_f1
    - competition_score
    - results_df
    """

    seq_col = "sequence_id"

    y_df = y_test_df.copy()

    if seq_col not in y_df.columns:
        if y_df.index.name == seq_col:
            y_df = y_df.reset_index()
        else:
            raise ValueError("y_test_df must contain sequence_id.")

    y_test_seq = (
        y_df.drop_duplicates(subset=[seq_col])
        .sort_values(seq_col)
        .reset_index(drop=True)
    )

    if isinstance(y_pred, pd.Series) and y_pred.index.name == seq_col:
        y_pred = y_pred.reindex(y_test_seq[seq_col]).to_numpy()
    else:
        y_pred = np.asarray(y_pred)

    if "is_target" in y_test_seq.columns:
        y_true_binary = y_test_seq["is_target"].astype(int).values
    else:
        y_true_binary = (y_test_seq[target_col] != "non_bfrb").astype(int)

    y_pred_binary = (y_pred != "non_bfrb").astype(int)

    binary_f1 = f1_score(
        y_true_binary,
        y_pred_binary,
        zero_division=0,
    )

    target_mask = y_true_binary == 1

    if target_mask.sum() > 0:
        gesture_f1 = f1_score(
            y_test_seq.loc[target_mask, target_col].values,
            y_pred[target_mask],
            average="macro",
            zero_division=0,
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
            print(
                classification_report(
                    y_test_seq.loc[target_mask, target_col].values,
                    y_pred[target_mask],
                    zero_division=0,
                )
            )

    results_df = pd.DataFrame(
        {
            "sequence_id": y_test_seq[seq_col].values,
            "is_target_true": y_true_binary,
            "is_target_pred": y_pred_binary,
            f"{target_col}_true": y_test_seq[target_col].values,
            f"{target_col}_pred": y_pred,
        }
    )

    return {
        "binary_f1": binary_f1,
        "gesture_f1": gesture_f1,
        "competition_score": comp_score,
        "results_df": results_df,
    }


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------


def prepare_multitask_param_space(param_space: Dict[str, Any], search_mode: str) -> Dict[str, Any]:
    """
    Kept for compatibility. Encodes dict parameters to JSON strings if needed.
    """

    if search_mode != "bayesian":
        return param_space

    out = {}

    for key, val in param_space.items():
        name = key.split("__")[-1]

        if (
            name in {"branch_filters", "branch_kernel_sizes", "branch_pool_sizes"}
            and isinstance(val, list)
            and val
            and isinstance(val[0], dict)
        ):
            out[key] = [json.dumps(d, sort_keys=True) for d in val]
        else:
            out[key] = val

    return out


def prepare_bayesian_space(param_space: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts lists and complex Categorical values into skopt-compatible spaces.
    """

    if Categorical is None:
        return param_space

    out = {}

    for key, space in param_space.items():
        if isinstance(space, Categorical):
            new_cats = []

            for cat in space.categories:
                if isinstance(cat, (tuple, list, dict)):
                    new_cats.append(json.dumps(cat, sort_keys=True))
                else:
                    new_cats.append(cat)

            out[key] = Categorical(new_cats)

        elif isinstance(space, list):
            out[key] = Categorical(space)

        else:
            out[key] = space

    return out