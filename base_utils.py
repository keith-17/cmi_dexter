"""
base_utils.py
The Hive: Advanced multi-domain sequence extraction with Kalman filtering,
dead reckoning, and pipe-separated feature modes.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from typing import Literal, Optional, List, Dict, Any, Tuple
import warnings
warnings.filterwarnings("ignore")

# --- Type Aliases ---
AccModeStr = str  # e.g., "raw|velocity|displacement|jerk"
RotModeStr = str  # e.g., "quaternion|rot6d|angular_velocity"
TOFMode = Literal["sensor_stats", "pooled_stats", "pooled_diff", "raw"]
THMMode = Literal["centered_diff", "diff", "centered", "raw"]
InterpMode = Literal["linear", "ffill", None]
MotionFilterMode = Literal[None, "kalman", "extended_kalman"]


class HoneycombBase(BaseEstimator, TransformerMixin):
    """Parent class for all Honeycomb components."""
    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> HoneycombBase:
        return self


class SensorCleaner(HoneycombBase):
    """Handles interpolation, dt computation, clipping, and masking."""
    def __init__(
        self,
        sampling_rate: int = 20,
        compute_dt: bool = True,
        clip_value: Optional[float] = None,
        interp_mode: InterpMode = "linear",
        sequence_col: str = "sequence_id",
        counter_col: str = "sequence_counter",
    ):
        self.sampling_rate = sampling_rate
        self.compute_dt = compute_dt
        self.clip_value = clip_value
        self.interp_mode = interp_mode
        self.sequence_col = sequence_col
        self.counter_col = counter_col

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [c for c in num_cols if c not in [self.sequence_col, self.counter_col]]

        if self.interp_mode == "linear":
            df[num_cols] = df.groupby(self.sequence_col, sort=False)[num_cols].transform(
                lambda g: g.interpolate(method="linear", limit_direction="both").ffill().bfill()
            )
        elif self.interp_mode == "ffill":
            df[num_cols] = df.groupby(self.sequence_col, sort=False)[num_cols].ffill().bfill()

        if self.compute_dt:
            if self.counter_col in df.columns:
                df["dt"] = df.groupby(self.sequence_col, sort=False)[self.counter_col].diff().fillna(1.0) / float(self.sampling_rate)
            else:
                df["dt"] = 1.0 / float(self.sampling_rate)
            df["dt"] = df["dt"].replace([np.inf, -np.inf], np.nan).fillna(1.0 / float(self.sampling_rate)).clip(lower=1e-6)
        else:
            df["dt"] = 1.0 / float(self.sampling_rate)

        if self.clip_value is not None:
            acc_cols = [c for c in df.columns if c.startswith("acc_") and not c.endswith(("_vel", "_disp", "_jerk", "_mag"))]
            if acc_cols:
                df[acc_cols] = df[acc_cols].clip(lower=-float(self.clip_value), upper=float(self.clip_value))

        df["mask"] = 1.0
        return df


class MotionFilter(HoneycombBase):
    """Applies Kalman filtering and/or Dead Reckoning to IMU streams."""
    def __init__(
        self,
        motion_filter_mode: MotionFilterMode = None,
        kalman_process_noise: float = 1e-3,
        kalman_measurement_noise: float = 1e-2,
        use_dead_reckoning: bool = False,
        dead_reckoning_use_quaternion: bool = False,
        dead_reckoning_detrend: bool = False,
        sequence_col: str = "sequence_id",
    ):
        self.motion_filter_mode = motion_filter_mode
        self.kalman_process_noise = kalman_process_noise
        self.kalman_measurement_noise = kalman_measurement_noise
        self.use_dead_reckoning = use_dead_reckoning
        self.dead_reckoning_use_quaternion = dead_reckoning_use_quaternion
        self.dead_reckoning_detrend = dead_reckoning_detrend
        self.sequence_col = sequence_col

    def _kalman_filter_1d(self, signal: np.ndarray, dt: np.ndarray) -> np.ndarray:
        Q = self.kalman_process_noise
        R = self.kalman_measurement_noise
        x, P = signal[0], 1.0
        out = np.empty_like(signal)
        for i in range(len(signal)):
            P_pred = P + Q
            K = P_pred / (P_pred + R)
            x = x + K * (signal[i] - x)
            P = (1 - K) * P_pred
            out[i] = x
        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        acc_cols = [c for c in df.columns if c.startswith("acc_") and c.endswith("_raw")]
        rot_cols = [c for c in df.columns if c.startswith("rot_") and c.endswith("_quat")]

        if self.motion_filter_mode in ("kalman", "extended_kalman") and acc_cols:
            for col in acc_cols:
                df[col] = df.groupby(self.sequence_col, sort=False).apply(
                    lambda g: pd.Series(self._kalman_filter_1d(g[col].values, g["dt"].values), index=g.index)
                ).droplevel(0)

        if self.use_dead_reckoning and acc_cols:
            dt = df["dt"].values
            for col in acc_cols:
                acc = df[col].values
                vel = np.cumsum(acc * dt)
                if self.dead_reckoning_detrend:
                    vel = vel - np.linspace(vel[0], vel[-1], len(vel))
                df[col.replace("_raw", "_dr_vel")] = vel
                pos = np.cumsum(vel * dt)
                if self.dead_reckoning_detrend:
                    pos = pos - np.linspace(pos[0], pos[-1], len(pos))
                df[col.replace("_raw", "_dr_pos")] = pos

        return df


class IMUExtractor(HoneycombBase):
    """Extracts multi-domain accelerometer and rotation features."""
    def __init__(
        self,
        acc_modes: AccModeStr = "raw",
        rotation_modes: RotModeStr = "quaternion",
        window_size: int = 5,
        smooth_alpha: Optional[float] = None,
        sequence_col: str = "sequence_id",
    ):
        self.acc_modes = acc_modes
        self.rotation_modes = rotation_modes
        self.window_size = window_size
        self.smooth_alpha = smooth_alpha
        self.sequence_col = sequence_col
        self.acc_modes_: List[str] = []
        self.rot_modes_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> IMUExtractor:
        self.acc_modes_ = self.acc_modes.split("|") if self.acc_modes else ["raw"]
        self.rot_modes_ = self.rotation_modes.split("|") if self.rotation_modes else ["quaternion"]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["acc_modes_", "rot_modes_"])
        df = X.copy()
        parts = []
        acc_base = [c for c in df.columns if c.startswith("acc_") and c.endswith("_raw")]
        rot_base = [c for c in df.columns if c.startswith("rot_") and c.endswith("_quat")]
        dt = df["dt"].astype(float)

        for mode in self.acc_modes_:
            if mode == "raw" and acc_base:
                parts.append(df[acc_base].add_suffix("_raw"))
            elif mode == "smoothed" and acc_base:
                if self.smooth_alpha is not None:
                    out = df.groupby(self.sequence_col, sort=False)[acc_base].transform(
                        lambda g: g.ewm(alpha=float(self.smooth_alpha), adjust=False).mean()
                    )
                else:
                    out = df.groupby(self.sequence_col, sort=False)[acc_base].transform(
                        lambda g: g.rolling(window=self.window_size, center=True, min_periods=1).mean()
                    )
                parts.append(out.add_suffix("_smooth"))
            elif mode == "velocity" and acc_base:
                vel = df[acc_base].mul(dt, axis=0).groupby(df[self.sequence_col].values, sort=False).cumsum()
                parts.append(vel.add_suffix("_vel"))
            elif mode == "displacement" and acc_base:
                vel = df[acc_base].mul(dt, axis=0).groupby(df[self.sequence_col].values, sort=False).cumsum()
                disp = vel.mul(dt, axis=0).groupby(df[self.sequence_col].values, sort=False).cumsum()
                parts.append(disp.add_suffix("_disp"))
            elif mode == "jerk" and acc_base:
                jerk = df.groupby(self.sequence_col, sort=False)[acc_base].diff().div(dt, axis=0).fillna(0.0)
                parts.append(jerk.add_suffix("_jerk"))
            elif mode == "acc_mag" and acc_base:
                mag = np.sqrt(df[acc_base].pow(2).sum(axis=1))
                parts.append(pd.DataFrame({"acc_mag": mag}, index=df.index))

        for mode in self.rot_modes_:
            if mode == "quaternion" and rot_base:
                parts.append(df[rot_base].add_suffix("_quat"))
            elif mode == "angular_velocity" and rot_base:
                ang_vel = df.groupby(self.sequence_col, sort=False)[rot_base].diff().div(dt, axis=0).fillna(0.0)
                parts.append(ang_vel.add_suffix("_angvel"))
            elif mode == "rot6d" and rot_base:
                # Simplified 6D rotation representation (first two cols of rotation matrix)
                # Placeholder: uses quaternion derivatives as proxy for speed
                parts.append(df[rot_base].iloc[:, :2].add_suffix("_rot6d"))

        return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)


class TOFExtractor(HoneycombBase):
    """Extracts Time-of-Flight features."""
    def __init__(
        self,
        tof_mode: TOFMode = "sensor_stats",
        sequence_col: str = "sequence_id",
        n_sensors: int = 5,
        grid_size: int = 8,
    ):
        self.tof_mode = tof_mode
        self.sequence_col = sequence_col
        self.n_sensors = n_sensors
        self.grid_size = grid_size
        self.tof_cols_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> TOFExtractor:
        self.tof_cols_ = [c for c in X.columns if c.startswith("tof_")]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["tof_cols_"])
        if not self.tof_cols_:
            return pd.DataFrame(index=X.index)
        df = X[self.tof_cols_].astype(float).copy().replace(-1.0, np.nan).fillna(255.0)

        if self.tof_mode == "raw":
            return df.add_suffix("_raw")
        elif self.tof_mode == "sensor_stats":
            parts = []
            for s in range(1, self.n_sensors + 1):
                cols = [c for c in df.columns if f"tof_{s}_" in c]
                if cols:
                    arr = df[cols].values
                    parts.append(pd.DataFrame({
                        f"tof_{s}_mean": arr.mean(axis=1),
                        f"tof_{s}_std": arr.std(axis=1),
                        f"tof_{s}_min": arr.min(axis=1),
                        f"tof_{s}_max": arr.max(axis=1),
                    }, index=X.index))
            return pd.concat(parts, axis=1)
        elif self.tof_mode == "pooled_stats":
            arr = df.values
            return pd.DataFrame({
                "tof_pooled_mean": arr.mean(axis=1),
                "tof_pooled_std": arr.std(axis=1),
            }, index=X.index)
        return df


class ThermoExtractor(HoneycombBase):
    """Extracts Thermopile features."""
    def __init__(self, thm_mode: THMMode = "centered_diff", sequence_col: str = "sequence_id"):
        self.thm_mode = thm_mode
        self.sequence_col = sequence_col
        self.thm_cols_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> ThermoExtractor:
        self.thm_cols_ = [c for c in X.columns if c.startswith("thm_")]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["thm_cols_"])
        if not self.thm_cols_:
            return pd.DataFrame(index=X.index)
        raw = X[self.thm_cols_].astype(float).copy()
        if self.thm_mode == "raw":
            return raw.add_suffix("_raw")
        elif self.thm_mode == "centered":
            means = raw.groupby(X[self.sequence_col].values, sort=False).transform("mean")
            return (raw - means).add_suffix("_centered")
        elif self.thm_mode == "centered_diff":
            means = raw.groupby(X[self.sequence_col].values, sort=False).transform("mean")
            centered = (raw - means).add_suffix("_centered")
            diff = centered.groupby(X[self.sequence_col].values, sort=False).diff().fillna(0.0).add_suffix("_diff")
            return pd.concat([centered, diff], axis=1)
        return raw


class AdvancedMultiDomainSequenceExtractor(HoneycombBase):
    """
    Orchestrates cleaning, motion filtering, and multi-domain extraction.
    Outputs a padded 3D numpy array: (n_sequences, maxlen, n_features).
    """
    def __init__(
        self,
        acc_modes: AccModeStr = "raw|velocity|displacement|jerk",
        rotation_modes: RotModeStr = "quaternion",
        tof_mode: TOFMode = "sensor_stats",
        thm_mode: THMMode = "centered_diff",
        motion_filter_mode: MotionFilterMode = None,
        kalman_process_noise: float = 1e-3,
        kalman_measurement_noise: float = 1e-2,
        use_dead_reckoning: bool = False,
        dead_reckoning_use_quaternion: bool = False,
        dead_reckoning_detrend: bool = False,
        sampling_rate: int = 20,
        compute_dt: bool = True,
        window_size: int = 7,
        smooth_alpha: Optional[float] = None,
        clip_value: Optional[float] = None,
        interp_mode: InterpMode = "linear",
        maxlen: int = 160,
        padding_value: float = -999.0,
        sequence_col: str = "sequence_id",
        counter_col: str = "sequence_counter",
    ):
        self.acc_modes = acc_modes
        self.rotation_modes = rotation_modes
        self.tof_mode = tof_mode
        self.thm_mode = thm_mode
        self.motion_filter_mode = motion_filter_mode
        self.kalman_process_noise = kalman_process_noise
        self.kalman_measurement_noise = kalman_measurement_noise
        self.use_dead_reckoning = use_dead_reckoning
        self.dead_reckoning_use_quaternion = dead_reckoning_use_quaternion
        self.dead_reckoning_detrend = dead_reckoning_detrend
        self.sampling_rate = sampling_rate
        self.compute_dt = compute_dt
        self.window_size = window_size
        self.smooth_alpha = smooth_alpha
        self.clip_value = clip_value
        self.interp_mode = interp_mode
        self.maxlen = maxlen
        self.padding_value = padding_value
        self.sequence_col = sequence_col
        self.counter_col = counter_col

        self.cleaner = SensorCleaner(sampling_rate, compute_dt, clip_value, interp_mode, sequence_col, counter_col)
        self.motion_filter = MotionFilter(motion_filter_mode, kalman_process_noise, kalman_measurement_noise,
                                          use_dead_reckoning, dead_reckoning_use_quaternion, dead_reckoning_detrend, sequence_col)
        self.imu = IMUExtractor(acc_modes, rotation_modes, window_size, smooth_alpha, sequence_col)
        self.tof = TOFExtractor(tof_mode, sequence_col)
        self.thermo = ThermoExtractor(thm_mode, sequence_col)
        self.feature_names_in_: List[str] = []
        self.modality_slices_: Dict[str, slice] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> AdvancedMultiDomainSequenceExtractor:
        cleaned = self.cleaner.transform(X)
        filtered = self.motion_filter.transform(cleaned)
        self.imu.fit(filtered, y)
        self.tof.fit(filtered, y)
        self.thermo.fit(filtered, y)

        imu_out = self.imu.transform(filtered)
        tof_out = self.tof.transform(filtered)
        thm_out = self.thermo.transform(filtered)
        combined = pd.concat([imu_out, tof_out, thm_out], axis=1).fillna(0.0)
        self.feature_names_in_ = list(combined.columns)

        # Map modalities to column slices for multi-branch models
        acc_idx = [i for i, c in enumerate(self.feature_names_in_) if "acc" in c or "dr_" in c]
        rot_idx = [i for i, c in enumerate(self.feature_names_in_) if "rot" in c or "quat" in c or "angvel" in c]
        tof_idx = [i for i, c in enumerate(self.feature_names_in_) if "tof" in c]
        thm_idx = [i for i, c in enumerate(self.feature_names_in_) if "thm" in c]
        
        self.modality_slices_ = {
            "acc": slice(min(acc_idx), max(acc_idx)+1) if acc_idx else slice(0,0),
            "rot": slice(min(rot_idx), max(rot_idx)+1) if rot_idx else slice(0,0),
            "tof": slice(min(tof_idx), max(tof_idx)+1) if tof_idx else slice(0,0),
            "thm": slice(min(thm_idx), max(thm_idx)+1) if thm_idx else slice(0,0),
        }
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, ["feature_names_in_", "modality_slices_"])
        cleaned = self.cleaner.transform(X)
        filtered = self.motion_filter.transform(cleaned)
        out = pd.concat([
            self.imu.transform(filtered),
            self.tof.transform(filtered),
            self.thermo.transform(filtered)
        ], axis=1).fillna(0.0)

        sequences = []
        for _, g in out.groupby(self.sequence_col, sort=False):
            arr = g[self.feature_names_in_].to_numpy(dtype=np.float32)
            if len(arr) >= self.maxlen:
                arr = arr[:self.maxlen]
            else:
                pad = np.full((self.maxlen - len(arr), arr.shape[1]), self.padding_value, dtype=np.float32)
                arr = np.vstack([arr, pad])
            sequences.append(arr)
        return np.stack(sequences, axis=0)

# Backward compatibility alias
SequenceExtractor = AdvancedMultiDomainSequenceExtractor
HiveSequenceExtractor = AdvancedMultiDomainSequenceExtractor