from __future__ import annotations
from typing import Literal, Optional, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import warnings

warnings.filterwarnings('ignore', '.*mask.*Conv1D.*')

AccMode = Optional[Literal["raw", "smoothed", "velocity", "displacement", "jerk"]]
LinearAccMode = Optional[Literal["baseline"]]
InterpMode = Optional[Literal["ffill", "linear"]]
StandardizeMode = Optional[Literal["mean_std"]]
RotationMode = Optional[Literal[
    "quaternion",
    "euler",
    "delta_euler",
    "angular_velocity",
    "rot6d",
]]
TOFMode = Optional[Literal["raw", "pooled", "pooled_diff", "sensor_stats", "pooled_stats"]]
TOFFillMode = Literal["nan_interpolate", "zero", "far_255", "far_500"]
THMMode = Optional[Literal["raw", "diff", "centered", "centered_diff"]]


class SignalCleaner:
    def __init__(
        self,
        sampling_rate: int = 20,
        compute_dt: bool = True,
        clip_value: Optional[float] = None,
        interp_mode: InterpMode = None,
        linear_acc_mode: LinearAccMode = None,
        use_highpass_fallback: bool = True,
        window_size: int = 5,
        sequence_col: str = "sequence_id",
        counter_col: str = "sequence_counter",
        acc_cols: Optional[List[str]] = None,
        rot_cols: Optional[List[str]] = None,
    ) -> None:
        self.sampling_rate = sampling_rate
        self.compute_dt = compute_dt
        self.clip_value = clip_value
        self.interp_mode = interp_mode
        self.linear_acc_mode = linear_acc_mode
        self.use_highpass_fallback = use_highpass_fallback
        self.window_size = window_size
        self.sequence_col = sequence_col
        self.counter_col = counter_col
        self.acc_cols = acc_cols
        self.rot_cols = rot_cols

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "SignalCleaner":
        self.acc_cols_ = self.acc_cols or [c for c in ["acc_x", "acc_y", "acc_z"] if c in X.columns]
        self.rot_cols_ = self.rot_cols or [c for c in ["rot_w", "rot_x", "rot_y", "rot_z"] if c in X.columns]
        
        if len(self.acc_cols_) != 3:
            raise ValueError(f"Expected 3 accelerometer columns, got: {self.acc_cols_}")
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "acc_cols_"):
            raise RuntimeError("SignalCleaner must be fitted first.")
        
        df = X.copy()
        
        if self.interp_mode is not None:
            df = self._interpolate(df)
        
        if self.compute_dt:
            df = self._add_dt(df)
        else:
            df["dt"] = 1.0 / float(self.sampling_rate)
        
        if self.clip_value is not None:
            df[self.acc_cols_] = df[self.acc_cols_].clip(
                lower=-float(self.clip_value),
                upper=float(self.clip_value),
            )
        
        if self.linear_acc_mode == "baseline":
            df = self._add_linear_acceleration(df)
        elif self.linear_acc_mode is not None:
            raise ValueError(f"Unknown linear_acc_mode: {self.linear_acc_mode}")
        
        df["mask"] = 1.0
        
        return df

    def _interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        parts = []
        
        for _, g in df.groupby(self.sequence_col, sort=False):
            g = g.copy()
            num_cols = g.select_dtypes(include=[np.number]).columns.tolist()
            num_cols = [c for c in num_cols if c not in [self.sequence_col]]
            
            if self.interp_mode == "ffill":
                g[num_cols] = g[num_cols].ffill().bfill()
            elif self.interp_mode == "linear":
                g[num_cols] = (
                    g[num_cols]
                    .interpolate(method="linear", limit_direction="both")
                    .ffill()
                    .bfill()
                )
            else:
                raise ValueError(f"Unknown interp_mode: {self.interp_mode}")
            
            parts.append(g)
        
        return pd.concat(parts, axis=0, ignore_index=True)

    def _add_dt(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        if self.counter_col in df.columns:
            df["dt"] = (
                df.groupby(self.sequence_col, sort=False)[self.counter_col]
                .diff()
                .fillna(1.0)
                / float(self.sampling_rate)
            )
        else:
            df["dt"] = 1.0 / float(self.sampling_rate)
        
        df["dt"] = df["dt"].replace([np.inf, -np.inf], np.nan)
        df["dt"] = df["dt"].fillna(1.0 / float(self.sampling_rate))
        df["dt"] = df["dt"].clip(lower=1e-6)
        
        return df

    def _add_linear_acceleration(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        has_rot = set(self.rot_cols_).issubset(df.columns)
        
        if has_rot:
            lin_acc = self._linear_acc_from_quaternion(df)
        elif self.use_highpass_fallback:
            lin_acc = self._linear_acc_highpass(df)
        else:
            raise ValueError(
                "linear_acc_mode='baseline' needs rotation columns, "
                "or use_highpass_fallback=True."
            )
        
        df["lin_acc_x"] = lin_acc[:, 0]
        df["lin_acc_y"] = lin_acc[:, 1]
        df["lin_acc_z"] = lin_acc[:, 2]
        
        return df

    def _linear_acc_highpass(self, df: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(df), 3), dtype=float)
        
        for _, idx in df.groupby(self.sequence_col, sort=False).groups.items():
            acc = df.loc[idx, self.acc_cols_].astype(float)
            baseline = acc.rolling(
                window=self.window_size,
                center=True,
                min_periods=1,
            ).mean()
            out[df.index.get_indexer(idx), :] = (acc - baseline).to_numpy()
        
        return out

    def _linear_acc_from_quaternion(self, df: pd.DataFrame) -> np.ndarray:
        acc = df[self.acc_cols_].astype(float).to_numpy()
        q = df[self.rot_cols_].astype(float).to_numpy()
        
        bad_q = ~np.isfinite(q).all(axis=1)
        q = self._normalise_quaternion(q)
        
        acc_world = self._rotate_sensor_to_world(acc, q)
        lin_acc = acc_world - np.array([0.0, 0.0, 9.81])
        
        if bad_q.any() and self.use_highpass_fallback:
            fallback = self._linear_acc_highpass(df)
            lin_acc[bad_q] = fallback[bad_q]
        
        return lin_acc

    @staticmethod
    def _normalise_quaternion(q: np.ndarray) -> np.ndarray:
        q = q.astype(float).copy()
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        bad = (~np.isfinite(norms)) | (norms == 0.0)
        norms[bad] = 1.0
        q = q / norms
        q[bad[:, 0]] = np.array([1.0, 0.0, 0.0, 0.0])
        return q

    @staticmethod
    def _rotate_sensor_to_world(v: np.ndarray, q: np.ndarray) -> np.ndarray:
        w = q[:, 0]
        xyz = q[:, 1:4]
        t = 2.0 * np.cross(xyz, v)
        return v + w[:, None] * t + np.cross(xyz, t)


class IMUExtractor:
    def __init__(
        self,
        acc_mode: AccMode = "raw",
        linear_acc_mode: LinearAccMode = None,
        use_acc_magnitude: bool = False,
        use_linear_acc_magnitude: bool = False,
        window_size: int = 5,
        smooth_alpha: Optional[float] = None,
        sequence_col: str = "sequence_id",
        acc_cols: Optional[List[str]] = None,
    ) -> None:
        self.acc_mode = acc_mode
        self.linear_acc_mode = linear_acc_mode
        self.use_acc_magnitude = use_acc_magnitude
        self.use_linear_acc_magnitude = use_linear_acc_magnitude
        self.window_size = window_size
        self.smooth_alpha = smooth_alpha
        self.sequence_col = sequence_col
        self.acc_cols = acc_cols

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "IMUExtractor":
        self.acc_cols_ = self.acc_cols or [c for c in ["acc_x", "acc_y", "acc_z"] if c in X.columns]
        if len(self.acc_cols_) != 3:
            raise ValueError(f"Expected 3 accelerometer columns, got: {self.acc_cols_}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "acc_cols_"):
            raise RuntimeError("IMUExtractor must be fitted first.")
        
        parts = []
        
        if self.acc_mode is None:
            pass
        elif self.acc_mode == "raw":
            parts.append(self._raw_acc(X))
        elif self.acc_mode == "smoothed":
            parts.append(self._smoothed_acc(X))
        elif self.acc_mode == "velocity":
            parts.append(self._velocity(X))
        elif self.acc_mode == "displacement":
            parts.append(self._displacement(X))
        elif self.acc_mode == "jerk":
            parts.append(self._jerk(X))
        else:
            raise ValueError(f"Unknown acc_mode: {self.acc_mode}")
        
        if self.linear_acc_mode == "baseline":
            parts.append(self._linear_acc(X))
        elif self.linear_acc_mode is not None:
            raise ValueError(f"Unknown linear_acc_mode: {self.linear_acc_mode}")
        
        if self.use_acc_magnitude:
            parts.append(self._acc_magnitude(X))
        
        if self.use_linear_acc_magnitude:
            parts.append(self._linear_acc_magnitude(X))
        
        if not parts:
            return pd.DataFrame(index=X.index)
        
        return pd.concat(parts, axis=1)

    def _raw_acc(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.acc_cols_].copy()

    def _linear_acc(self, X: pd.DataFrame) -> pd.DataFrame:
        lin_cols = ["lin_acc_x", "lin_acc_y", "lin_acc_z"]
        if not set(lin_cols).issubset(X.columns):
            raise ValueError("linear_acc_mode='baseline' needs SignalCleaner to create lin_acc_x/y/z.")
        return X[lin_cols].copy()

    def _smoothed_acc(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X[self.acc_cols_].astype(float)
        if self.smooth_alpha is not None:
            out = df.groupby(X[self.sequence_col].values, sort=False).transform(
                lambda g: g.ewm(alpha=float(self.smooth_alpha), adjust=False).mean()
            )
        else:
            out = df.groupby(X[self.sequence_col].values, sort=False).transform(
                lambda g: g.rolling(
                    window=self.window_size,
                    center=True,
                    min_periods=1,
                ).mean()
            )
        out.columns = [f"{c}_smooth" for c in self.acc_cols_]
        return out

    def _velocity(self, X: pd.DataFrame) -> pd.DataFrame:
        acc = X[self.acc_cols_].astype(float)
        dt = X["dt"].astype(float)
        out = acc.mul(dt, axis=0)
        out = out.groupby(X[self.sequence_col].values, sort=False).cumsum()
        out.columns = ["acc_vel_x", "acc_vel_y", "acc_vel_z"]
        return out

    def _displacement(self, X: pd.DataFrame) -> pd.DataFrame:
        vel = self._velocity(X)
        dt = X["dt"].astype(float)
        out = vel.mul(dt, axis=0)
        out = out.groupby(X[self.sequence_col].values, sort=False).cumsum()
        out.columns = ["acc_disp_x", "acc_disp_y", "acc_disp_z"]
        return out

    def _jerk(self, X: pd.DataFrame) -> pd.DataFrame:
        acc = X[self.acc_cols_].astype(float)
        dt = X["dt"].replace(0.0, np.nan).astype(float)
        out = acc.groupby(X[self.sequence_col].values, sort=False).diff()
        out = out.div(dt, axis=0).fillna(0.0)
        out.columns = ["acc_jerk_x", "acc_jerk_y", "acc_jerk_z"]
        return out

    def _acc_magnitude(self, X: pd.DataFrame) -> pd.DataFrame:
        acc = X[self.acc_cols_].astype(float)
        mag = np.sqrt(np.sum(acc.to_numpy() ** 2, axis=1))
        return pd.DataFrame({"acc_mag": mag}, index=X.index)

    def _linear_acc_magnitude(self, X: pd.DataFrame) -> pd.DataFrame:
        lin_cols = ["lin_acc_x", "lin_acc_y", "lin_acc_z"]
        if self.linear_acc_mode != 'baseline':
            if self.use_linear_acc_magnitude:
                print(f"Warning: use_linear_acc_magnitude=True but linear_acc_mode={self.linear_acc_mode}")
                return pd.DataFrame({"lin_acc_mag": np.zeros(len(X))}, index=X.index)
        
        if not set(lin_cols).issubset(X.columns):
            if self.use_linear_acc_magnitude:
                print(f"Warning: linear acceleration columns not found")
                return pd.DataFrame({"lin_acc_mag": np.zeros(len(X))}, index=X.index)
        
        lin = X[lin_cols].astype(float)
        mag = np.sqrt(np.sum(lin.to_numpy() ** 2, axis=1))
        return pd.DataFrame({"lin_acc_mag": mag}, index=X.index)


class RotationExtractor:
    def __init__(
        self,
        rotation_mode: RotationMode = None,
        sequence_col: str = "sequence_id",
        counter_col: str = "sequence_counter",
        dt_col: str = "dt",
        sampling_rate: int = 20,
        rot_cols: Optional[List[str]] = None,
        fix_quaternion_sign: bool = True,
    ) -> None:
        self.rotation_mode = rotation_mode
        self.sequence_col = sequence_col
        self.counter_col = counter_col
        self.dt_col = dt_col
        self.sampling_rate = sampling_rate
        self.rot_cols = rot_cols
        self.fix_quaternion_sign = fix_quaternion_sign

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "RotationExtractor":
        self.rot_cols_ = self.rot_cols or [c for c in ["rot_w", "rot_x", "rot_y", "rot_z"] if c in X.columns]
        if len(self.rot_cols_) != 4:
            raise ValueError(f"Expected 4 quaternion columns, got: {self.rot_cols_}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "rot_cols_"):
            raise RuntimeError("RotationExtractor must be fitted first.")
        
        if self.rotation_mode is None:
            return pd.DataFrame(index=X.index)
        
        if self.rotation_mode == "quaternion":
            return self._quaternion(X)
        if self.rotation_mode == "euler":
            return self._euler(X)
        if self.rotation_mode == "delta_euler":
            return self._delta_euler(X)
        if self.rotation_mode == "angular_velocity":
            return self._angular_velocity(X)
        if self.rotation_mode == "rot6d":
            return self._rot6d(X)
        
        raise ValueError(f"Unknown rotation_mode: {self.rotation_mode}")

    def _get_dt(self, X: pd.DataFrame) -> np.ndarray:
        if self.dt_col in X.columns:
            dt = X[self.dt_col].astype(float).to_numpy()
        elif self.counter_col in X.columns:
            dt = (X.groupby(self.sequence_col, sort=False)[self.counter_col].diff().fillna(1.0) / float(self.sampling_rate)).astype(float).to_numpy()
        else:
            dt = np.full(len(X), 1.0 / float(self.sampling_rate), dtype=float)
        dt = np.nan_to_num(dt, nan=1.0 / float(self.sampling_rate), posinf=1.0 / float(self.sampling_rate), neginf=1.0 / float(self.sampling_rate))
        return np.maximum(dt, 1e-6)

    def _get_quaternion(self, X: pd.DataFrame) -> np.ndarray:
        q = X[self.rot_cols_].astype(float).to_numpy()
        q = self._normalise_quaternion(q)
        if self.fix_quaternion_sign:
            q = self._fix_sign_continuity(q, X[self.sequence_col].to_numpy())
        return q

    def _quaternion(self, X: pd.DataFrame) -> pd.DataFrame:
        q = self._get_quaternion(X)
        return pd.DataFrame(q, columns=["rot_w_norm", "rot_x_norm", "rot_y_norm", "rot_z_norm"], index=X.index)

    def _euler(self, X: pd.DataFrame) -> pd.DataFrame:
        euler = self._quaternion_to_euler(self._get_quaternion(X))
        return pd.DataFrame(euler, columns=["rot_roll", "rot_pitch", "rot_yaw"], index=X.index)

    def _delta_euler(self, X: pd.DataFrame) -> pd.DataFrame:
        euler_df = self._euler(X)
        out = euler_df.groupby(X[self.sequence_col].values, sort=False).diff().fillna(0.0)
        out.columns = ["delta_rot_roll", "delta_rot_pitch", "delta_rot_yaw"]
        return out

    def _angular_velocity(self, X: pd.DataFrame) -> pd.DataFrame:
        q = self._get_quaternion(X)
        dt = self._get_dt(X)
        out = np.zeros((len(X), 3), dtype=float)
        
        for _, idx in X.groupby(self.sequence_col, sort=False).groups.items():
            pos = X.index.get_indexer(idx)
            if len(pos) <= 1:
                continue
            q_prev = q[pos[:-1]]
            q_curr = q[pos[1:]]
            q_rel = self._quaternion_multiply(q_curr, self._quaternion_conjugate(q_prev))
            q_rel = self._normalise_quaternion(q_rel)
            q_rel[q_rel[:, 0] < 0] *= -1.0
            w = np.clip(q_rel[:, 0], -1.0, 1.0)
            xyz = q_rel[:, 1:4]
            angle = 2.0 * np.arctan2(np.linalg.norm(xyz, axis=1), w)
            small = angle < 1e-8
            axis = np.zeros_like(xyz)
            norms = np.linalg.norm(xyz, axis=1)
            axis[~small] = xyz[~small] / norms[~small, None]
            safe_dt = dt[pos[1:]]
            out[pos[1:], :] = axis * (angle / safe_dt)[:, None]
        
        return pd.DataFrame(out, columns=["ang_vel_x", "ang_vel_y", "ang_vel_z"], index=X.index)

    def _rot6d(self, X: pd.DataFrame) -> pd.DataFrame:
        R = self._quaternion_to_rotation_matrix(self._get_quaternion(X))
        rot6d = np.concatenate([R[:, :, 0], R[:, :, 1]], axis=1)
        return pd.DataFrame(rot6d, columns=["rot6d_c1_x", "rot6d_c1_y", "rot6d_c1_z", "rot6d_c2_x", "rot6d_c2_y", "rot6d_c2_z"], index=X.index)

    @staticmethod
    def _normalise_quaternion(q: np.ndarray) -> np.ndarray:
        q = q.astype(float).copy()
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        bad = (~np.isfinite(norms)) | (norms == 0.0)
        norms[bad] = 1.0
        q = q / norms
        q[bad[:, 0]] = np.array([1.0, 0.0, 0.0, 0.0])
        return q

    @staticmethod
    def _fix_sign_continuity(q: np.ndarray, seq_ids: np.ndarray) -> np.ndarray:
        q = q.copy()
        for seq_id in pd.unique(seq_ids):
            pos = np.flatnonzero(seq_ids == seq_id)
            for i in range(1, len(pos)):
                if np.dot(q[pos[i - 1]], q[pos[i]]) < 0:
                    q[pos[i]] *= -1.0
        return q

    @staticmethod
    def _quaternion_conjugate(q: np.ndarray) -> np.ndarray:
        out = q.copy()
        out[:, 1:] *= -1.0
        return out

    @staticmethod
    def _quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        return np.column_stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    @staticmethod
    def _quaternion_to_euler(q: np.ndarray) -> np.ndarray:
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        roll = np.arctan2(2.0 * (w*x + y*z), 1.0 - 2.0 * (x*x + y*y))
        pitch = np.arcsin(np.clip(2.0 * (w*y - z*x), -1.0, 1.0))
        yaw = np.arctan2(2.0 * (w*z + x*y), 1.0 - 2.0 * (y*y + z*z))
        return np.column_stack([roll, pitch, yaw])

    @staticmethod
    def _quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R = np.zeros((len(q), 3, 3), dtype=float)
        R[:, 0, 0] = 1.0 - 2.0 * (y*y + z*z)
        R[:, 0, 1] = 2.0 * (x*y - z*w)
        R[:, 0, 2] = 2.0 * (x*z + y*w)
        R[:, 1, 0] = 2.0 * (x*y + z*w)
        R[:, 1, 1] = 1.0 - 2.0 * (x*x + z*z)
        R[:, 1, 2] = 2.0 * (y*z - x*w)
        R[:, 2, 0] = 2.0 * (x*z - y*w)
        R[:, 2, 1] = 2.0 * (y*z + x*w)
        R[:, 2, 2] = 1.0 - 2.0 * (x*x + y*y)
        return R


class TOFExtractor:
    def __init__(
        self,
        tof_mode: TOFMode = None,
        tof_fill_mode: TOFFillMode = "far_255",
        invalid_value: float = -1.0,
        sequence_col: str = "sequence_id",
        n_sensors: int = 5,
        grid_size: int = 8,
        tof_cols: Optional[List[str]] = None,
    ) -> None:
        self.tof_mode = tof_mode
        self.tof_fill_mode = tof_fill_mode
        self.invalid_value = invalid_value
        self.sequence_col = sequence_col
        self.n_sensors = n_sensors
        self.grid_size = grid_size
        self.tof_cols = tof_cols

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "TOFExtractor":
        self.tof_cols_ = self.tof_cols or [
            f"tof_{sensor}_v{pixel}"
            for sensor in range(1, self.n_sensors + 1)
            for pixel in range(self.grid_size * self.grid_size)
            if f"tof_{sensor}_v{pixel}" in X.columns
        ]
        if self.tof_mode is not None and len(self.tof_cols_) == 0:
            raise ValueError("tof_mode is set but no TOF columns were found.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "tof_cols_"):
            raise RuntimeError("TOFExtractor must be fitted first.")
        
        if self.tof_mode is None:
            return pd.DataFrame(index=X.index)
        
        if self.tof_mode == "raw":
            return self._raw(X)
        if self.tof_mode == "pooled":
            return self._pooled(X)
        if self.tof_mode == "pooled_diff":
            pooled = self._pooled(X)
            diff = self._diff(pooled, X)
            diff.columns = [f"{c}_diff" for c in pooled.columns]
            return pd.concat([pooled, diff], axis=1)
        if self.tof_mode == "sensor_stats":
            return self._sensor_stats(X)
        if self.tof_mode == "pooled_stats":
            return pd.concat([self._pooled(X), self._sensor_stats(X)], axis=1)
        
        raise ValueError(f"Unknown tof_mode: {self.tof_mode}")

    def _clean_df(self, X: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        df = X[cols].astype(float).copy()
        df = df.replace(self.invalid_value, np.nan)
        
        if self.tof_fill_mode == "nan_interpolate":
            df[self.sequence_col] = X[self.sequence_col].values
            df[cols] = df.groupby(self.sequence_col, sort=False)[cols].transform(
                lambda g: g.interpolate(method="linear", limit_direction="both").ffill().bfill()
            )
            df = df.drop(columns=[self.sequence_col])
            return df.fillna(255.0)
        if self.tof_fill_mode == "zero":
            return df.fillna(0.0)
        if self.tof_fill_mode == "far_255":
            return df.fillna(255.0)
        if self.tof_fill_mode == "far_500":
            return df.fillna(500.0)
        raise ValueError(f"Unknown tof_fill_mode: {self.tof_fill_mode}")

    def _sensor_cols(self, sensor: int) -> List[str]:
        return [f"tof_{sensor}_v{pixel}" for pixel in range(self.grid_size * self.grid_size)]

    def _sensor_grid(self, X: pd.DataFrame, sensor: int) -> np.ndarray:
        cols = self._sensor_cols(sensor)
        missing = [c for c in cols if c not in X.columns]
        if missing:
            raise ValueError(f"Missing TOF columns for sensor {sensor}: {missing[:5]}")
        clean = self._clean_df(X, cols)
        return clean.to_numpy(dtype=float).reshape(len(X), self.grid_size, self.grid_size)

    def _raw(self, X: pd.DataFrame) -> pd.DataFrame:
        clean = self._clean_df(X, self.tof_cols_)
        clean.columns = [f"tof_raw_{i}" for i in range(clean.shape[1])]
        return clean

    def _pooled(self, X: pd.DataFrame) -> pd.DataFrame:
        regions = [
            (0, 3, 0, 3), (0, 3, 3, 5), (0, 3, 5, 8),
            (3, 5, 0, 3), (3, 5, 3, 5), (3, 5, 5, 8),
            (5, 8, 0, 3), (5, 8, 3, 5), (5, 8, 5, 8),
        ]
        parts = []
        for sensor in range(1, self.n_sensors + 1):
            grid = self._sensor_grid(X, sensor)
            data = {}
            for region_id, (r0, r1, c0, c1) in enumerate(regions):
                patch = grid[:, r0:r1, c0:c1]
                data[f"tof_{sensor}_pool_{region_id}"] = patch.mean(axis=(1, 2))
            parts.append(pd.DataFrame(data, index=X.index))
        return pd.concat(parts, axis=1)

    def _sensor_stats(self, X: pd.DataFrame) -> pd.DataFrame:
        data = {}
        for sensor in range(1, self.n_sensors + 1):
            cols = self._sensor_cols(sensor)
            clean = self._clean_df(X, cols)
            arr = clean.to_numpy(dtype=float)
            raw = X[cols].astype(float).to_numpy()
            valid = raw != self.invalid_value
            data[f"tof_{sensor}_mean"] = arr.mean(axis=1)
            data[f"tof_{sensor}_std"] = arr.std(axis=1)
            data[f"tof_{sensor}_min"] = arr.min(axis=1)
            data[f"tof_{sensor}_max"] = arr.max(axis=1)
            data[f"tof_{sensor}_valid_ratio"] = valid.mean(axis=1)
        return pd.DataFrame(data, index=X.index)

    def _diff(self, df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
        return df.groupby(X[self.sequence_col].values, sort=False).diff().fillna(0.0)


class ThermoExtractor:
    def __init__(
        self,
        thm_mode: THMMode = None,
        sequence_col: str = "sequence_id",
        thm_cols: Optional[List[str]] = None,
    ) -> None:
        self.thm_mode = thm_mode
        self.sequence_col = sequence_col
        self.thm_cols = thm_cols

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "ThermoExtractor":
        self.thm_cols_ = self.thm_cols or [c for c in X.columns if c.startswith("thm_")]
        if self.thm_mode is not None and len(self.thm_cols_) == 0:
            raise ValueError("thm_mode is set but no thermopile columns were found.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "thm_cols_"):
            raise RuntimeError("ThermoExtractor must be fitted first.")
        
        if self.thm_mode is None:
            return pd.DataFrame(index=X.index)
        
        if self.thm_mode == "raw":
            return self._raw(X)
        if self.thm_mode == "diff":
            raw = self._raw(X)
            diff = self._diff(raw, X)
            diff.columns = [f"{c}_diff" for c in raw.columns]
            return diff
        if self.thm_mode == "centered":
            return self._centered(X)
        if self.thm_mode == "centered_diff":
            centered = self._centered(X)
            diff = self._diff(centered, X)
            diff.columns = [f"{c}_diff" for c in centered.columns]
            return pd.concat([centered, diff], axis=1)
        
        raise ValueError(f"Unknown thm_mode: {self.thm_mode}")

    def _raw(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X[self.thm_cols_].astype(float).copy()
        out.columns = [f"{c}_raw" for c in self.thm_cols_]
        return out

    def _centered(self, X: pd.DataFrame) -> pd.DataFrame:
        raw = X[self.thm_cols_].astype(float)
        means = raw.groupby(X[self.sequence_col].values, sort=False).transform("mean")
        out = raw - means
        out.columns = [f"{c}_centered" for c in self.thm_cols_]
        return out

    def _diff(self, df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
        return df.groupby(X[self.sequence_col].values, sort=False).diff().fillna(0.0)


class SequenceExtractor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        acc_mode: AccMode = "raw",
        linear_acc_mode: LinearAccMode = None,
        use_acc_magnitude: bool = False,
        use_linear_acc_magnitude: bool = False,
        sampling_rate: int = 20,
        compute_dt: bool = True,
        clip_value: Optional[float] = None,
        interp_mode: InterpMode = None,
        use_highpass_fallback: bool = True,
        window_size: int = 5,
        smooth_alpha: Optional[float] = None,
        standardize: StandardizeMode = None,
        include_mask: bool = False,
        sequence_col: str = "sequence_id",
        counter_col: str = "sequence_counter",
        acc_cols: Optional[List[str]] = None,
        rot_cols: Optional[List[str]] = None,
        rotation_mode: RotationMode = None,
        fix_quaternion_sign: bool = True,
        tof_mode: TOFMode = None,
        tof_fill_mode: TOFFillMode = "far_255",
        thm_mode: THMMode = None,
        tof_cols: Optional[List[str]] = None,
        thm_cols: Optional[List[str]] = None,
    ) -> None:
        self.acc_mode = acc_mode
        self.linear_acc_mode = linear_acc_mode
        self.use_acc_magnitude = use_acc_magnitude
        self.use_linear_acc_magnitude = use_linear_acc_magnitude
        self.sampling_rate = sampling_rate
        self.compute_dt = compute_dt
        self.clip_value = clip_value
        self.interp_mode = interp_mode
        self.use_highpass_fallback = use_highpass_fallback
        self.window_size = window_size
        self.smooth_alpha = smooth_alpha
        self.standardize = standardize
        self.include_mask = include_mask
        self.sequence_col = sequence_col
        self.counter_col = counter_col
        self.acc_cols = acc_cols
        self.rot_cols = rot_cols
        self.rotation_mode = rotation_mode
        self.fix_quaternion_sign = fix_quaternion_sign
        self.tof_mode = tof_mode
        self.tof_fill_mode = tof_fill_mode
        self.thm_mode = thm_mode
        self.tof_cols = tof_cols
        self.thm_cols = thm_cols

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "SequenceExtractor":
        self.cleaner_ = SignalCleaner(
            sampling_rate=self.sampling_rate,
            compute_dt=self.compute_dt,
            clip_value=self.clip_value,
            interp_mode=self.interp_mode,
            linear_acc_mode=self.linear_acc_mode,
            use_highpass_fallback=self.use_highpass_fallback,
            window_size=self.window_size,
            sequence_col=self.sequence_col,
            counter_col=self.counter_col,
            acc_cols=self.acc_cols,
            rot_cols=self.rot_cols,
        )
        self.cleaner_.fit(X, y)
        cleaned = self.cleaner_.transform(X)
        
        self.imu_extractor_ = IMUExtractor(
            acc_mode=self.acc_mode,
            linear_acc_mode=self.linear_acc_mode,
            use_acc_magnitude=self.use_acc_magnitude,
            use_linear_acc_magnitude=self.use_linear_acc_magnitude,
            window_size=self.window_size,
            smooth_alpha=self.smooth_alpha,
            sequence_col=self.sequence_col,
            acc_cols=self.cleaner_.acc_cols_,
        )
        self.imu_extractor_.fit(cleaned, y)
        
        self.rotation_extractor_ = RotationExtractor(
            rotation_mode=self.rotation_mode,
            sequence_col=self.sequence_col,
            counter_col=self.counter_col,
            dt_col="dt",
            sampling_rate=self.sampling_rate,
            rot_cols=self.cleaner_.rot_cols_,
            fix_quaternion_sign=self.fix_quaternion_sign,
        )
        self.rotation_extractor_.fit(cleaned, y)
        
        self.tof_extractor_ = TOFExtractor(
            tof_mode=self.tof_mode,
            tof_fill_mode=self.tof_fill_mode,
            sequence_col=self.sequence_col,
            tof_cols=self.tof_cols,
        )
        self.tof_extractor_.fit(cleaned, y)
        
        self.thermo_extractor_ = ThermoExtractor(
            thm_mode=self.thm_mode,
            sequence_col=self.sequence_col,
            thm_cols=self.thm_cols,
        )
        self.thermo_extractor_.fit(cleaned, y)
        
        imu_sample = self.imu_extractor_.transform(cleaned)
        rot_sample = self.rotation_extractor_.transform(cleaned)
        tof_sample = self.tof_extractor_.transform(cleaned)
        thm_sample = self.thermo_extractor_.transform(cleaned)
        
        sample = pd.concat([imu_sample, rot_sample, tof_sample, thm_sample], axis=1)
        
        if self.include_mask:
            sample["mask"] = cleaned["mask"].to_numpy()
        
        self.feature_names_in_ = list(sample.columns)
        
        if self.standardize == "mean_std" and not sample.empty:
            self.mean_ = sample.mean(axis=0)
            self.std_ = sample.std(axis=0).replace(0.0, 1.0)
        else:
            self.mean_ = None
            self.std_ = None
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["cleaner_", "imu_extractor_", "rotation_extractor_", "tof_extractor_", "thermo_extractor_", "feature_names_in_"])
        
        cleaned = self.cleaner_.transform(X)
        imu_out = self.imu_extractor_.transform(cleaned)
        rot_out = self.rotation_extractor_.transform(cleaned)
        tof_out = self.tof_extractor_.transform(cleaned)
        thm_out = self.thermo_extractor_.transform(cleaned)
        
        out = pd.concat([imu_out, rot_out, tof_out, thm_out], axis=1)
        
        if self.include_mask:
            out["mask"] = cleaned["mask"].to_numpy()
        
        if self.standardize == "mean_std" and not out.empty:
            out = (out - self.mean_) / self.std_
        
        out = out.fillna(0.0)
        out.index = cleaned[self.sequence_col].to_numpy()
        out.columns = [str(c) for c in out.columns]
        
        return out
    

def correct_handedness_and_quaternions(
    df: pd.DataFrame, 
    demo_df: pd.DataFrame,
    rot_cols: List[str] = ["rot_w", "rot_x", "rot_y", "rot_z"]
) -> pd.DataFrame:
    """Apply handedness correction to accelerometer and quaternion data."""
    from scipy.spatial.transform import Rotation as R
    
    df = df.copy()
    df["handedness"] = df["subject"].map(demo_df.set_index("subject")["handedness"])
    left_handed_mask = df["handedness"].eq(0)
    
    # Mirror acc_x
    df.loc[left_handed_mask, "acc_x"] *= -1.0
    
    # Quaternion correction
    if left_handed_mask.any():
        q = df.loc[left_handed_mask, rot_cols].to_numpy(dtype=float)
        q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
        norm = np.linalg.norm(q, axis=1, keepdims=True)
        bad = norm.squeeze() == 0.0
        if bad.any():
            q[bad] = np.array([1.0, 0.0, 0.0, 0.0])
            norm = np.linalg.norm(q, axis=1, keepdims=True)
        q = q / norm
        
        # wxyz -> xyzw for scipy
        q_xyzw = q[:, [1, 2, 3, 0]]
        euler_xyz = R.from_quat(q_xyzw).as_euler("xyz", degrees=False)
        # Flip pitch and yaw, keep roll
        euler_xyz[:, [1, 2]] *= -1.0
        q_xyzw_fixed = R.from_euler("xyz", euler_xyz, degrees=False).as_quat()
        q_wxyz_fixed = q_xyzw_fixed[:, [3, 0, 1, 2]]
        df.loc[left_handed_mask, rot_cols] = q_wxyz_fixed
    
    return df

