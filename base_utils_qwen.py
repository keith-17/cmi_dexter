# base_utils.py
"""
base_utils.py
The Honeycomb: Modular temporal feature extraction and sequence padding.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from typing import Literal, Optional, List, Dict, Any, Tuple
import warnings
warnings.filterwarnings("ignore")
import json
from skopt.space import Categorical

# --- Type Aliases ---
AccModeStr = str  # e.g., "raw|velocity|displacement|jerk"
RotModeStr = str  # e.g., "quaternion|rot6d|angular_velocity"
TOFModeStr = str
THMModeStr = str
InterpMode = Literal["linear", "ffill", None]
MotionFilterMode = Literal[None, "kalman", "extended_kalman"]

class HoneycombBase(BaseEstimator, TransformerMixin):
    """Parent class for all Honeycomb components."""
    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> 'HoneycombBase':
        return self

    def _parse_modes(self, modes_str: Optional[str]) -> List[str]:
        if not modes_str:
            return []
        return [m.strip() for m in modes_str.split("|") if m.strip() and m.strip().lower() != "none"]

class SignalCleaner(HoneycombBase):
    """Handles interpolation, dt computation, clipping, and masking."""
    def __init__(
        self,
        sampling_rate: int = 20,
        compute_dt: bool = True,
        clip_value: Optional[float] = None,
        interp_mode: InterpMode = "linear",
        linear_acc_mode: Optional[Literal["baseline"]] = None,
        use_highpass_fallback: bool = True,
        window_size: int = 5,
        sequence_col: str = "sequence_id",
        counter_col: str = "sequence_counter",
    ):
        self.sampling_rate = sampling_rate
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

        if self.linear_acc_mode == "baseline":
            df = self._add_linear_acceleration(df)

        df["mask"] = 1.0
        return df

    def _add_linear_acceleration(self, df: pd.DataFrame) -> pd.DataFrame:
        acc_cols = [c for c in self.acc_cols_ if c in df.columns]
        rot_cols = [c for c in self.rot_cols_ if c in df.columns]
        
        if len(acc_cols) != 3 or len(rot_cols) != 4:
            if self.use_highpass_fallback:
                return self._linear_acc_highpass(df, acc_cols)
            return df

        acc = df[acc_cols].to_numpy(dtype=float)
        q = df[rot_cols].to_numpy(dtype=float)
        
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        bad = (norms == 0) | ~np.isfinite(norms)
        norms[bad] = 1.0
        q = q / norms
        q[bad[:, 0]] = np.array([1.0, 0.0, 0.0, 0.0])
        
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R = np.zeros((len(q), 3, 3))
        R[:, 0, 0] = 1 - 2*y**2 - 2*z**2
        R[:, 0, 1] = 2*x*y - 2*w*z
        R[:, 0, 2] = 2*x*z + 2*w*y
        R[:, 1, 0] = 2*x*y + 2*w*z
        R[:, 1, 1] = 1 - 2*x**2 - 2*z**2
        R[:, 1, 2] = 2*y*z - 2*w*x
        R[:, 2, 0] = 2*x*z - 2*w*y
        R[:, 2, 1] = 2*y*z + 2*w*x
        R[:, 2, 2] = 1 - 2*x**2 - 2*y**2
        
        gravity = np.array([0.0, 0.0, 9.81])
        acc_world = np.einsum('nij,nj->ni', R, acc)
        lin_acc = acc_world - gravity
        
        for i, col in enumerate(["lin_acc_x", "lin_acc_y", "lin_acc_z"]):
            df[col] = lin_acc[:, i]
        return df

    def _linear_acc_highpass(self, df: pd.DataFrame, acc_cols: List[str]) -> pd.DataFrame:
        for col in acc_cols:
            baseline = df.groupby(self.sequence_col, sort=False)[col].transform(
                lambda g: g.rolling(window=self.window_size, center=True, min_periods=1).mean()
            )
            df[f"lin_{col}"] = df[col] - baseline
        return df

class MotionFilter(HoneycombBase):
    """Applies Kalman filtering and/or Dead Reckoning to IMU streams."""
    def __init__(
        self,
        motion_filter_mode: MotionFilterMode = None,
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
        acc_cols = [c for c in df.columns if c.startswith("acc_") and not c.endswith(("_vel", "_disp", "_jerk", "_mag", "_dr_vel", "_dr_pos"))]
        
        if self.motion_filter_mode in ("kalman", "extended_kalman") and acc_cols:
            for col in acc_cols:
                df[col] = df.groupby(self.sequence_col, sort=False)[col].transform(
                    lambda g: self._kalman_filter_1d(g.to_numpy(dtype=float))
                )

        if self.use_dead_reckoning and acc_cols:
            dt = df["dt"].to_numpy(dtype=float)
            for col in acc_cols:
                acc = df[col].to_numpy(dtype=float)
                vel = np.cumsum(acc * dt)
                if self.dead_reckoning_detrend:
                    vel = vel - np.linspace(vel[0], vel[-1], len(vel))
                df[f"{col}_dr_vel"] = vel
                pos = np.cumsum(vel * dt)
                if self.dead_reckoning_detrend:
                    pos = pos - np.linspace(pos[0], pos[-1], len(pos))
                df[f"{col}_dr_pos"] = pos
        return df

class IMUExtractor(HoneycombBase):
    """Extracts multi-domain accelerometer features."""
    def __init__(
        self,
        acc_modes: AccModeStr = "raw",
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
        self.acc_cols_ = [c for c in X.columns if c.startswith("acc_") and not c.startswith("lin_acc")]
        self.lin_cols_ = [c for c in X.columns if c.startswith("lin_acc")]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["acc_modes_", "acc_cols_"])
        df = X.copy()
        parts = []
        dt = df["dt"].astype(float)

        for mode in self.acc_modes_:
            if mode == "raw" and self.acc_cols_:
                parts.append(df[self.acc_cols_].add_suffix("_raw"))
            elif mode == "smoothed" and self.acc_cols_:
                if self.smooth_alpha is not None:
                    out = df.groupby(self.sequence_col, sort=False)[self.acc_cols_].transform(
                        lambda g: g.ewm(alpha=float(self.smooth_alpha), adjust=False).mean()
                    )
                else:
                    out = df.groupby(self.sequence_col, sort=False)[self.acc_cols_].transform(
                        lambda g: g.rolling(window=self.window_size, center=True, min_periods=1).mean()
                    )
                parts.append(out.add_suffix("_smooth"))
            elif mode == "velocity" and self.acc_cols_:
                vel = df[self.acc_cols_].mul(dt, axis=0).groupby(df[self.sequence_col].values, sort=False).cumsum()
                parts.append(vel.add_suffix("_vel"))
            elif mode == "displacement" and self.acc_cols_:
                vel = df[self.acc_cols_].mul(dt, axis=0).groupby(df[self.sequence_col].values, sort=False).cumsum()
                disp = vel.mul(dt, axis=0).groupby(df[self.sequence_col].values, sort=False).cumsum()
                parts.append(disp.add_suffix("_disp"))
            elif mode == "jerk" and self.acc_cols_:
                jerk = df.groupby(self.sequence_col, sort=False)[self.acc_cols_].diff().div(dt, axis=0).fillna(0.0)
                parts.append(jerk.add_suffix("_jerk"))

        if self.use_acc_magnitude and self.acc_cols_:
            mag = np.sqrt(df[self.acc_cols_].pow(2).sum(axis=1))
            parts.append(pd.DataFrame({"acc_mag": mag}, index=df.index))
            
        if self.use_linear_acc_magnitude and self.lin_cols_:
            mag = np.sqrt(df[self.lin_cols_].pow(2).sum(axis=1))
            parts.append(pd.DataFrame({"lin_acc_mag": mag}, index=df.index))

        return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)

class RotationExtractor(HoneycombBase):
    """Extracts multi-domain rotation features."""
    def __init__(
        self,
        rotation_modes: RotModeStr = "quaternion",
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
        dt = df["dt"].astype(float)
        
        q = df[self.rot_cols_].to_numpy(dtype=float) if self.rot_cols_ else None
        if q is not None and self.fix_quaternion_sign:
            for seq_id, idx in df.groupby(self.sequence_col, sort=False).groups.items():
                pos = df.index.get_indexer(idx)
                for i in range(1, len(pos)):
                    if np.dot(q[pos[i-1]], q[pos[i]]) < 0:
                        q[pos[i]] *= -1.0

        for mode in self.rot_modes_:
            if mode == "quaternion" and q is not None:
                parts.append(pd.DataFrame(q, columns=[c + "_quat" for c in self.rot_cols_], index=df.index))
            elif mode == "euler" and q is not None:
                w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
                roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
                pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
                yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
                parts.append(pd.DataFrame({"rot_roll": roll, "rot_pitch": pitch, "rot_yaw": yaw}, index=df.index))
            elif mode == "delta_euler" and q is not None:
                w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
                roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
                pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
                yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
                euler_df = pd.DataFrame({"rot_roll": roll, "rot_pitch": pitch, "rot_yaw": yaw}, index=df.index)
                delta = euler_df.groupby(df[self.sequence_col].values, sort=False).diff().fillna(0.0)
                parts.append(delta.add_suffix("_delta"))
            elif mode == "angular_velocity" and q is not None:
                ang_vel = df.groupby(self.sequence_col, sort=False)[self.rot_cols_].diff().div(dt, axis=0).fillna(0.0)
                parts.append(ang_vel.add_suffix("_angvel"))
                ang_vel_mag = np.sqrt(ang_vel.pow(2).sum(axis=1))
                parts.append(pd.DataFrame({"ang_vel_mag": ang_vel_mag}, index=df.index))
            elif mode == "rot6d" and q is not None:
                w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
                R = np.zeros((len(q), 3, 3))
                R[:, 0, 0] = 1 - 2*y**2 - 2*z**2; R[:, 0, 1] = 2*x*y - 2*w*z; R[:, 0, 2] = 2*x*z + 2*w*y
                R[:, 1, 0] = 2*x*y + 2*w*z; R[:, 1, 1] = 1 - 2*x**2 - 2*z**2; R[:, 1, 2] = 2*y*z - 2*w*x
                rot6d = np.concatenate([R[:, :, 0], R[:, :, 1]], axis=1)
                cols = ["rot6d_c1_x", "rot6d_c1_y", "rot6d_c1_z", "rot6d_c2_x", "rot6d_c2_y", "rot6d_c2_z"]
                parts.append(pd.DataFrame(rot6d, columns=cols, index=df.index))

        return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)

class TOFExtractor(HoneycombBase):
    """Extracts Time-of-Flight features."""
    def __init__(
        self,
        tof_modes: TOFModeStr = "sensor_stats",
        tof_fill_mode: Literal["nan_interpolate", "zero", "far_255", "far_500"] = "far_255",
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
            
        df = X[self.tof_cols_].astype(float).copy().replace(-1.0, np.nan)
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

        parts = []
        for mode in self.tof_modes_:
            if mode == "raw":
                parts.append(df.add_suffix("_raw"))
            elif mode == "sensor_stats":
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
            elif mode == "pooled_stats":
                arr = df.values
                parts.append(pd.DataFrame({
                    "tof_pooled_mean": arr.mean(axis=1),
                    "tof_pooled_std": arr.std(axis=1),
                }, index=X.index))
            elif mode == "pooled":
                # Simple spatial pooling (e.g. 8x8 grid -> 2x2)
                for s in range(1, self.n_sensors + 1):
                    cols = [c for c in df.columns if f"tof_{s}_" in c]
                    if len(cols) == 64:
                        arr = df[cols].values.reshape(-1, 8, 8)
                        pool = arr.reshape(-1, 4, 2, 4, 2).mean(axis=(2, 4)).reshape(-1, 16)
                        parts.append(pd.DataFrame(pool, columns=[f"tof_{s}_pool_{i}" for i in range(16)], index=X.index))
                        
        return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=X.index)

class ThermoExtractor(HoneycombBase):
    """Extracts Thermopile features."""
    def __init__(self, thm_modes: THMModeStr = "centered_diff", sequence_col: str = "sequence_id"):
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
                parts.append(raw.groupby(X[self.sequence_col].values, sort=False).diff().fillna(0.0).add_suffix("_diff"))
            elif mode == "centered_diff":
                means = raw.groupby(X[self.sequence_col].values, sort=False).transform("mean")
                centered = raw - means
                diff = centered.groupby(X[self.sequence_col].values, sort=False).diff().fillna(0.0)
                parts.append(centered.add_suffix("_centered"))
                parts.append(diff.add_suffix("_centered_diff"))
                
        return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=X.index)


class SequenceExtractor(HoneycombBase):
    """
    Orchestrates cleaning, motion filtering, and multi-domain extraction.
    Outputs a padded 3D numpy array: (n_sequences, maxlen, n_features).

    Extended with Kalman and dead reckoning fine-tuning.
    """
    def __init__(
        self,
        acc_modes: AccModeStr = "raw|velocity|displacement|jerk",
        rotation_modes: RotModeStr = "quaternion",
        tof_modes: TOFModeStr = "sensor_stats",          # FIXED: plural
        thm_modes: THMModeStr = "centered_diff",         # FIXED: plural
        motion_filter_mode: MotionFilterMode = None,
        use_dead_reckoning: bool = False,
        dead_reckoning_detrend: bool = False,            # NEW
        kalman_process_noise: float = 1e-3,              # NEW
        kalman_measurement_noise: float = 1e-2,          # NEW
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
        self.tof_modes = tof_modes
        self.thm_modes = thm_modes
        self.motion_filter_mode = motion_filter_mode
        self.use_dead_reckoning = use_dead_reckoning
        self.dead_reckoning_detrend = dead_reckoning_detrend          # store
        self.kalman_process_noise = kalman_process_noise              # store
        self.kalman_measurement_noise = kalman_measurement_noise      # store
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

        # Create internal components with the new parameters
        self.cleaner = SignalCleaner(
            sampling_rate, compute_dt, clip_value, interp_mode,
            sequence_col=sequence_col, counter_col=counter_col,
            window_size=window_size
        )
        self.motion_filter = MotionFilter(
            motion_filter_mode,
            kalman_process_noise=kalman_process_noise,           # pass through
            kalman_measurement_noise=kalman_measurement_noise,
            use_dead_reckoning=use_dead_reckoning,
            dead_reckoning_detrend=dead_reckoning_detrend,       # pass through
            sequence_col=sequence_col
        )
        self.imu = IMUExtractor(
            acc_modes, window_size=window_size, smooth_alpha=smooth_alpha,
            sequence_col=sequence_col
        )
        self.rotation = RotationExtractor(
            rotation_modes, sequence_col=sequence_col
        )
        self.tof = TOFExtractor(
            tof_modes, sequence_col=sequence_col           # now tof_modes is plural
        )
        self.thermo = ThermoExtractor(
            thm_modes, sequence_col=sequence_col           # now thm_modes is plural
        )
        self.feature_names_in_: List[str] = []
        self.base_feature_names_: List[str] = []

    @staticmethod
    def _global_context_feature_names(base_cols: List[str]) -> List[str]:
        return (
            list(base_cols)
            + [f"{c}_global_mean" for c in base_cols]
            + [f"{c}_global_std" for c in base_cols]
        )

    def _append_global_context(self, df: pd.DataFrame, base_cols: List[str]) -> pd.DataFrame:
        out = df.copy()
        grouped = out.groupby(self.sequence_col, sort=False)
        for col in base_cols:
            out[f"{col}_global_mean"] = grouped[col].transform("mean")
            out[f"{col}_global_std"] = grouped[col].transform("std").fillna(0.0)
        return out

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> 'SequenceExtractor':
        cleaned = self.cleaner.fit_transform(X)
        filtered = self.motion_filter.fit_transform(cleaned)
        self.imu.fit(filtered)
        self.rotation.fit(filtered)
        self.tof.fit(filtered)
        self.thermo.fit(filtered)

        imu_out = self.imu.transform(filtered)
        rot_out = self.rotation.transform(filtered)
        tof_out = self.tof.transform(filtered)
        thm_out = self.thermo.transform(filtered)

        combined = pd.concat([imu_out, rot_out, tof_out, thm_out], axis=1).fillna(0.0)
        base_cols = list(combined.columns)
        self.base_feature_names_ = base_cols
        self.feature_names_in_ = (
            base_cols
            + [f"{c}_global_mean" for c in base_cols]
            + [f"{c}_global_std" for c in base_cols]
        )
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, ["feature_names_in_", "base_feature_names_"])

        cleaned = self.cleaner.transform(X)
        filtered = self.motion_filter.transform(cleaned)

        out = pd.concat([
            self.imu.transform(filtered),
            self.rotation.transform(filtered),
            self.tof.transform(filtered),
            self.thermo.transform(filtered),
        ], axis=1).fillna(0.0)

        out[self.sequence_col] = X[self.sequence_col].values
        if self.counter_col in X.columns:
            out[self.counter_col] = X[self.counter_col].values

        out = self._append_global_context(out, self.base_feature_names_)

        sequences = []
        for _, g in out.groupby(self.sequence_col, sort=False):
            if self.counter_col in g.columns:
                g = g.sort_values(self.counter_col)
            arr = g[self.feature_names_in_].to_numpy(dtype=np.float32)

            if len(arr) >= self.maxlen:
                arr = arr[:self.maxlen]
            else:
                pad = np.full((self.maxlen - len(arr), arr.shape[1]), self.padding_value, dtype=np.float32)
                arr = np.vstack([arr, pad])

            sequences.append(arr)

        return np.stack(sequences, axis=0)


def prepare_multitask_param_space(param_space: dict, search_mode: str) -> dict:
    """Encodes dict parameters to JSON strings for skopt Bayesian optimization."""
    if search_mode != "bayesian":
        return param_space
    out = {}
    for key, val in param_space.items():
        name = key.split("__")[-1]
        if name in {"branch_filters", "branch_kernel_sizes", "branch_pool_sizes"} and isinstance(val, list) and val and isinstance(val[0], dict):
            out[key] = [json.dumps(d, sort_keys=True) for d in val]
        else:
            out[key] = val
    return out


def prepare_bayesian_space(param_space):
    """Convert any non-scalar Categorical category (tuple, list, dict) to a JSON string."""
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
        else:
            out[key] = space
    return out