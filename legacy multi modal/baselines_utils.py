"""
baselines_utils.py
==================
Baseline feature extraction and classifier pipelines for the CMI Kaggle competition.

Feature modes
-------------
tabular_simple       : per-sequence statistical summary (mean/std/min/max/energy per channel)
tabular_honeycomb    : richer tabular features — acc, rotation (euler/delta), thm, tof pooled stats
temporal_honeycomb   : (N_seq, maxlen, C) tensor with channel engineering for ROCKET / 1D CNN
temporal_raw         : raw sensor columns padded to (N_seq, maxlen, C)

Classifier names
----------------
dummy                : majority-class baseline
rf                   : RandomForestClassifier inside ManyToOneWrapper
ridge_rocket         : RidgeRocketClassifier (MiniRocket + Ridge)
cnn                  : Keras1DCNNClassifier (temporal 3-D input)

Public API
----------
make_baseline_pipeline(feature_mode, classifier_name, target_col, ...) -> Pipeline
build_feature_extractor(feature_mode, **feature_kwargs)                 -> transformer
build_classifier(classifier_name, target_col, **classifier_kwargs)      -> estimator
competition_scorer        (for GridSearchCV / BayesSearchCV scoring=)
evaluate_holdout(y_true, y_pred, target_col, verbose)
plot_training_curves(history)
"""

from __future__ import annotations

import warnings
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.validation import check_is_fitted

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── optional tensorflow ───────────────────────────────────────────────────────
try:
    import os; os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
except ImportError:
    HAS_TF = False

# ── optional sktime/MiniRocket ─────────────────────────────────────────────────
try:
    from sktime.transformations.panel.rocket import MiniRocket
    HAS_ROCKET = True
except ImportError:
    HAS_ROCKET = False


# =============================================================================
# SENSOR COLUMN DEFINITIONS
# =============================================================================

_ACC_COLS  = ["acc_x", "acc_y", "acc_z"]
_ROT_COLS  = ["rot_w", "rot_x", "rot_y", "rot_z"]
_THM_COLS  = ["thm_1", "thm_2", "thm_3", "thm_4", "thm_5"]
_TOF_COLS  = [f"tof_{s}_v{p}" for s in range(1, 6) for p in range(64)]


# =============================================================================
# HELPERS
# =============================================================================

def _quat_to_euler(df: pd.DataFrame) -> pd.DataFrame:
    q = df.astype(float)
    w, x, y, z = q.iloc[:, 0], q.iloc[:, 1], q.iloc[:, 2], q.iloc[:, 3]
    roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
    yaw   = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
    return pd.DataFrame({"rot_roll": roll.values, "rot_pitch": pitch.values, "rot_yaw": yaw.values},
                        index=df.index)


def _time_stats(arr: np.ndarray, prefix: str) -> dict:
    """Per-column time-domain statistics."""
    feats: dict = {}
    for i, col in enumerate(arr.T):
        col = col[np.isfinite(col)]
        if len(col) == 0:
            for s in ("mean", "std", "min", "max", "rms", "energy", "ptp"):
                feats[f"{prefix}_{i}_{s}"] = 0.0
            continue
        feats[f"{prefix}_{i}_mean"]   = np.mean(col)
        feats[f"{prefix}_{i}_std"]    = np.std(col)
        feats[f"{prefix}_{i}_min"]    = np.min(col)
        feats[f"{prefix}_{i}_max"]    = np.max(col)
        feats[f"{prefix}_{i}_rms"]    = np.sqrt(np.mean(col**2))
        feats[f"{prefix}_{i}_energy"] = np.sum(col**2)
        feats[f"{prefix}_{i}_ptp"]    = np.ptp(col)
    return feats


# =============================================================================
# TABULAR FEATURE EXTRACTORS
# =============================================================================

class SimpleTabularExtractor(BaseEstimator, TransformerMixin):
    """
    Per-sequence statistical summary (mean/std/min/max/energy) over all
    numeric sensor channels.  One row per sequence_id.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        sensor_cols = [c for c in X.columns
                       if c in _ACC_COLS + _ROT_COLS + _THM_COLS
                       or c.startswith("tof_")]
        rows = []
        for seq_id, grp in X.groupby("sequence_id", sort=False):
            vals = grp[sensor_cols].replace(-1.0, np.nan).astype(float)
            feats: dict = {"sequence_id": seq_id}
            for col in sensor_cols:
                v = vals[col].dropna().values
                if len(v) == 0:
                    feats[f"{col}_mean"] = 0.0; feats[f"{col}_std"] = 0.0
                    feats[f"{col}_min"]  = 0.0; feats[f"{col}_max"] = 0.0
                    feats[f"{col}_energy"] = 0.0
                else:
                    feats[f"{col}_mean"]   = np.mean(v)
                    feats[f"{col}_std"]    = np.std(v)
                    feats[f"{col}_min"]    = np.min(v)
                    feats[f"{col}_max"]    = np.max(v)
                    feats[f"{col}_energy"] = np.sum(v**2)
            rows.append(feats)
        return pd.DataFrame(rows).set_index("sequence_id").fillna(0.0)


class HoneycombTabularExtractor(BaseEstimator, TransformerMixin):
    """
    Richer tabular feature extractor inspired by the competition honeycomb approach.

    Parameters
    ----------
    acc_modes          : pipe-separated string, subset of {raw, velocity, smoothed, jerk}
    rotation_modes     : pipe-separated string, subset of {quaternion, euler, angular_velocity}
    tof_modes          : pipe-separated string, subset of {sensor_stats, pooled_stats, pooled_diff}
    thm_modes          : pipe-separated string, subset of {centered_diff, diff, centered, raw}
    sampling_rate      : expected Hz (used for jerk normalisation)
    window_size        : rolling smoothing window
    clip_value         : clip sensor values to ±clip_value (None = no clip)
    interp_mode        : 'linear' or 'ffill' for NaN filling
    motion_filter_mode : None | 'kalman' | 'extended_kalman'  (kalman = simple EWM smoother)
    use_dead_reckoning : integrate acc to velocity (appends extra stats)
    """

    def __init__(
        self,
        acc_modes: str = "raw",
        rotation_modes: str = "quaternion",
        tof_modes: str = "pooled_stats",
        thm_modes: str = "centered_diff",
        sampling_rate: int = 100,
        window_size: int = 20,
        clip_value: Optional[float] = None,
        interp_mode: str = "linear",
        motion_filter_mode: Optional[str] = None,
        use_dead_reckoning: bool = False,
    ):
        self.acc_modes          = acc_modes
        self.rotation_modes     = rotation_modes
        self.tof_modes          = tof_modes
        self.thm_modes          = thm_modes
        self.sampling_rate      = sampling_rate
        self.window_size        = window_size
        self.clip_value         = clip_value
        self.interp_mode        = interp_mode
        self.motion_filter_mode = motion_filter_mode
        self.use_dead_reckoning = use_dead_reckoning

    def fit(self, X, y=None):
        return self

    # ── main transform ─────────────────────────────────────────────────────────
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for seq_id, grp in X.groupby("sequence_id", sort=False):
            feats: dict = {"sequence_id": seq_id}
            feats.update(self._acc_features(grp))
            feats.update(self._rotation_features(grp))
            feats.update(self._thm_features(grp))
            feats.update(self._tof_features(grp))
            rows.append(feats)
        return pd.DataFrame(rows).set_index("sequence_id").fillna(0.0)

    # ── accelerometer ──────────────────────────────────────────────────────────
    def _acc_features(self, grp: pd.DataFrame) -> dict:
        avail = [c for c in _ACC_COLS if c in grp.columns]
        if not avail:
            return {}
        acc = grp[avail].astype(float).copy()
        if self.clip_value is not None:
            acc = acc.clip(-self.clip_value, self.clip_value)
        acc = self._fill(acc)
        if self.motion_filter_mode == "kalman":
            acc = acc.ewm(span=max(2, self.window_size // 4), adjust=False).mean()

        feats: dict = {}
        modes = [m.strip() for m in self.acc_modes.split("|")]

        if "raw" in modes or "smoothed" in modes:
            smoothed = acc.rolling(self.window_size, center=True, min_periods=1).mean()
            src = smoothed if "smoothed" in modes else acc
            feats.update(_time_stats(src.values, "acc"))

        if "velocity" in modes or "jerk" in modes:
            dt = 1.0 / max(1, self.sampling_rate)
            vel = acc.cumsum() * dt
            if "velocity" in modes:
                feats.update(_time_stats(vel.values, "acc_vel"))
            if "jerk" in modes:
                jerk = acc.diff().fillna(0) / dt
                feats.update(_time_stats(jerk.values, "acc_jerk"))

        if self.use_dead_reckoning:
            dt = 1.0 / max(1, self.sampling_rate)
            vel  = acc.cumsum() * dt
            disp = vel.cumsum() * dt
            feats.update(_time_stats(disp.values, "acc_disp"))

        return feats

    # ── rotation ───────────────────────────────────────────────────────────────
    def _rotation_features(self, grp: pd.DataFrame) -> dict:
        avail = [c for c in _ROT_COLS if c in grp.columns]
        if not avail:
            return {}
        rot_raw = grp[avail].astype(float).copy()
        rot_raw = self._fill(rot_raw)

        euler = _quat_to_euler(rot_raw)
        # per-sequence unwrap
        for col in euler.columns:
            euler[col] = np.unwrap(euler[col].values)

        feats: dict = {}
        modes = [m.strip() for m in self.rotation_modes.split("|")]

        if "quaternion" in modes:
            feats.update(_time_stats(rot_raw.values, "rot_q"))

        if "euler" in modes:
            feats.update(_time_stats(euler.values, "rot_e"))

        if "angular_velocity" in modes:
            ang_vel = euler.diff().fillna(0)
            feats.update(_time_stats(ang_vel.values, "rot_av"))

        return feats

    # ── thermopile ─────────────────────────────────────────────────────────────
    def _thm_features(self, grp: pd.DataFrame) -> dict:
        avail = [c for c in _THM_COLS if c in grp.columns]
        if not avail:
            return {}
        thm = grp[avail].astype(float).copy()
        thm = self._fill(thm)

        feats: dict = {}
        modes = [m.strip() for m in self.thm_modes.split("|")]

        if "raw" in modes or "centered" in modes:
            src = thm.sub(thm.mean(axis=1), axis=0) if "centered" in modes else thm
            feats.update(_time_stats(src.values, "thm"))

        if "diff" in modes:
            feats.update(_time_stats(thm.diff().fillna(0).values, "thm_diff"))

        if "centered_diff" in modes:
            cd = thm.sub(thm.mean(axis=1), axis=0).diff().fillna(0)
            feats.update(_time_stats(cd.values, "thm_cd"))

        return feats

    # ── time-of-flight ─────────────────────────────────────────────────────────
    def _tof_features(self, grp: pd.DataFrame) -> dict:
        feats: dict = {}
        modes = [m.strip() for m in self.tof_modes.split("|")]

        for sensor_id in range(1, 6):
            s_cols = [c for c in _TOF_COLS if c.startswith(f"tof_{sensor_id}_") and c in grp.columns]
            if not s_cols:
                continue
            raw = grp[s_cols].astype(float).replace(-1.0, np.nan)
            raw = raw.mask((raw <= 0) | (raw >= 4000), np.nan)
            vals = raw.values  # (T, 64)

            if "sensor_stats" in modes or "pooled_stats" in modes:
                flat = vals.flatten()
                valid = flat[np.isfinite(flat)]
                if len(valid) > 0:
                    pref = f"tof{sensor_id}"
                    feats[f"{pref}_mean"]   = np.nanmean(vals)
                    feats[f"{pref}_std"]    = np.nanstd(vals)
                    feats[f"{pref}_min"]    = np.nanmin(vals)
                    feats[f"{pref}_max"]    = np.nanmax(vals)
                    feats[f"{pref}_valid_ratio"] = np.isfinite(vals).mean()
                    # temporal mean frame → center vs border
                    mean_frame = np.nanmean(vals.reshape(-1, 64), axis=0)
                    mean_frame = np.nan_to_num(mean_frame, nan=4000.0).reshape(8, 8)
                    feats[f"{pref}_center_mean"] = mean_frame[2:6, 2:6].mean()
                    feats[f"{pref}_edge_mean"]   = (mean_frame.sum() - mean_frame[2:6, 2:6].sum()) / 48

            if "pooled_diff" in modes:
                row_means = np.nanmean(vals, axis=1)
                if len(row_means) > 1:
                    diffs = np.diff(row_means)
                    feats[f"tof{sensor_id}_diff_mean"] = np.mean(diffs)
                    feats[f"tof{sensor_id}_diff_std"]  = np.std(diffs)

        return feats

    # ── utility ────────────────────────────────────────────────────────────────
    def _fill(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.interp_mode == "linear":
            return df.interpolate(method="linear", limit_direction="both").fillna(0.0)
        return df.ffill().bfill().fillna(0.0)


# =============================================================================
# TEMPORAL (SEQUENCE TENSOR) EXTRACTOR — for ROCKET and 1D CNN
# =============================================================================

class SequenceTensorExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts per-timestep feature tensors and pads to (N_seq, maxlen, C).

    Parameters match the param spaces in the notebook (extractor__ prefix):
      acc_modes, rotation_modes, tof_modes, thm_modes, sampling_rate,
      maxlen, window_size, clip_value, interp_mode, motion_filter_mode,
      use_dead_reckoning, dead_reckoning_detrend,
      kalman_process_noise, kalman_measurement_noise
    """

    def __init__(
        self,
        acc_modes: str = "raw",
        rotation_modes: str = "quaternion",
        tof_modes: str = "pooled_stats",
        thm_modes: str = "centered_diff",
        sampling_rate: int = 100,
        maxlen: int = 120,
        window_size: int = 20,
        clip_value: Optional[float] = None,
        interp_mode: str = "linear",
        motion_filter_mode: Optional[str] = None,
        use_dead_reckoning: bool = False,
        dead_reckoning_detrend: bool = False,
        kalman_process_noise: float = 1e-3,
        kalman_measurement_noise: float = 1e-1,
        padding_value: float = 0.0,
    ):
        self.acc_modes              = acc_modes
        self.rotation_modes         = rotation_modes
        self.tof_modes              = tof_modes
        self.thm_modes              = thm_modes
        self.sampling_rate          = sampling_rate
        self.maxlen                 = maxlen
        self.window_size            = window_size
        self.clip_value             = clip_value
        self.interp_mode            = interp_mode
        self.motion_filter_mode     = motion_filter_mode
        self.use_dead_reckoning     = use_dead_reckoning
        self.dead_reckoning_detrend = dead_reckoning_detrend
        self.kalman_process_noise   = kalman_process_noise
        self.kalman_measurement_noise = kalman_measurement_noise
        self.padding_value          = padding_value

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        """Return dict with keys 'X' (n_seq, maxlen, C), 'sequence_ids', 'lengths'."""
        seq_ids_all = X["sequence_id"].unique()
        tensors, seq_ids_out, lengths = [], [], []

        for seq_id in seq_ids_all:
            grp = X[X["sequence_id"] == seq_id]
            channels = self._build_channels(grp)  # (T, C)
            T = min(len(channels), self.maxlen)
            lengths.append(len(channels))
            seq_ids_out.append(seq_id)
            tensors.append(channels[:T])

        n_seq = len(tensors)
        if n_seq == 0:
            return {"X": np.zeros((0, self.maxlen, 1), dtype=np.float32),
                    "sequence_ids": np.array([]), "lengths": np.array([])}

        C = tensors[0].shape[1]
        out = np.full((n_seq, self.maxlen, C), self.padding_value, dtype=np.float32)
        for i, arr in enumerate(tensors):
            T = arr.shape[0]
            out[i, :T, :] = arr.astype(np.float32)

        return {
            "X": out,
            "sequence_ids": np.array(seq_ids_out),
            "lengths": np.array(lengths),
        }

    # ── channel builder ────────────────────────────────────────────────────────
    def _build_channels(self, grp: pd.DataFrame) -> np.ndarray:
        parts = []

        # ACC
        avail_acc = [c for c in _ACC_COLS if c in grp.columns]
        if avail_acc:
            acc = grp[avail_acc].astype(float).copy()
            if self.clip_value is not None:
                acc = acc.clip(-self.clip_value, self.clip_value)
            acc = self._fill(acc)
            if self.motion_filter_mode == "kalman":
                span = max(2, int(1.0 / max(self.kalman_process_noise, 1e-6)))
                acc = acc.ewm(span=min(span, len(acc)), adjust=False).mean()

            modes = [m.strip() for m in self.acc_modes.split("|")]
            if "raw" in modes:
                parts.append(acc.values)
            if "smoothed" in modes:
                parts.append(acc.rolling(self.window_size, center=True, min_periods=1).mean().values)
            if "velocity" in modes:
                parts.append((acc.cumsum() / max(1, self.sampling_rate)).values)
            if "jerk" in modes:
                parts.append((acc.diff().fillna(0) * max(1, self.sampling_rate)).values)
            if self.use_dead_reckoning:
                dt = 1.0 / max(1, self.sampling_rate)
                vel  = acc.cumsum() * dt
                disp = vel.cumsum() * dt
                if self.dead_reckoning_detrend:
                    disp = disp - disp.mean()
                parts.append(disp.values)

        # ROTATION
        avail_rot = [c for c in _ROT_COLS if c in grp.columns]
        if avail_rot:
            rot_raw = grp[avail_rot].astype(float)
            rot_raw = self._fill(rot_raw)
            euler = _quat_to_euler(rot_raw)
            for col in euler.columns:
                euler[col] = np.unwrap(euler[col].values)

            modes = [m.strip() for m in self.rotation_modes.split("|")]
            if "quaternion" in modes:
                parts.append(rot_raw.values)
            if "euler" in modes:
                parts.append(euler.values)
            if "angular_velocity" in modes:
                parts.append(euler.diff().fillna(0).values)

        # THERMOPILE
        avail_thm = [c for c in _THM_COLS if c in grp.columns]
        if avail_thm:
            thm = grp[avail_thm].astype(float)
            thm = self._fill(thm)

            modes = [m.strip() for m in self.thm_modes.split("|")]
            if "raw" in modes:
                parts.append(thm.values)
            if "centered" in modes:
                parts.append(thm.sub(thm.mean(axis=1), axis=0).values)
            if "diff" in modes:
                parts.append(thm.diff().fillna(0).values)
            if "centered_diff" in modes:
                parts.append(thm.sub(thm.mean(axis=1), axis=0).diff().fillna(0).values)

        # TOF — pooled per-sensor summary per timestep
        modes = [m.strip() for m in self.tof_modes.split("|")]
        if "sensor_stats" in modes or "pooled_stats" in modes:
            tof_parts = []
            for sensor_id in range(1, 6):
                s_cols = [c for c in _TOF_COLS if c.startswith(f"tof_{sensor_id}_") and c in grp.columns]
                if not s_cols:
                    continue
                raw = grp[s_cols].astype(float).replace(-1.0, np.nan)
                raw = raw.mask((raw <= 0) | (raw >= 4000), np.nan)
                tof_parts.append(np.nanmean(raw.values, axis=1, keepdims=True))
                tof_parts.append(np.nanstd(raw.values, axis=1, keepdims=True))
                tof_parts.append(np.nanmin(raw.values, axis=1, keepdims=True))
            if tof_parts:
                tof_arr = np.concatenate(tof_parts, axis=1)
                parts.append(np.nan_to_num(tof_arr, nan=0.0))

        if not parts:
            return np.zeros((len(grp), 1), dtype=np.float32)
        return np.concatenate(parts, axis=1).astype(np.float32)

    def _fill(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.interp_mode == "linear":
            return df.interpolate(method="linear", limit_direction="both").fillna(0.0)
        return df.ffill().bfill().fillna(0.0)


# =============================================================================
# MANY-TO-ONE WRAPPER  (tabular: sequence_id-indexed DataFrame)
# =============================================================================

class ManyToOneWrapper(ClassifierMixin, BaseEstimator):
    """
    Collapses row-level y DataFrame to one label per sequence and delegates
    to an sklearn estimator.  X must be a DataFrame indexed by sequence_id.

    The inner estimator is exposed as ``base_estimator`` so that GridSearchCV /
    BayesSearchCV param grids can use the key pattern
    ``classifier__base_estimator__<param>``.
    """
    _estimator_type = "classifier"

    def __init__(self, base_estimator, target: str = "bfrb"):
        self.base_estimator = base_estimator
        self.target         = target

    def _collapse_y(self, X, y):
        target_map = (
            y.drop_duplicates("sequence_id")
             .set_index("sequence_id")[self.target]
        )
        y_seq = pd.Series(X.index.map(target_map), index=X.index)
        valid = y_seq.notna()
        return X.loc[valid], y_seq.loc[valid]

    def fit(self, X, y):
        X_ok, y_seq = self._collapse_y(X, y)
        if len(y_seq) == 0:
            raise ValueError("No valid labels after alignment.")
        self.estimator_ = clone(self.base_estimator)
        self.estimator_.fit(X_ok, y_seq)
        self.classes_ = self.estimator_.classes_ if hasattr(self.estimator_, "classes_") else np.unique(y_seq)
        return self

    def predict(self, X):
        check_is_fitted(self, ["estimator_"])
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self, ["estimator_"])
        return self.estimator_.predict_proba(X)

    def score(self, X, y):
        _, y_seq = self._collapse_y(X, y)
        return np.mean(self.predict(X.loc[y_seq.index]) == y_seq.values)


# =============================================================================
# MANY-TO-ONE WRAPPER — TEMPORAL (dict input from SequenceTensorExtractor)
# =============================================================================

class ManyToOneWrapperTemporal(ClassifierMixin, BaseEstimator):
    """
    Wraps RidgeRocketClassifier / Keras1DCNNClassifier for dict-input tensors.
    Inner estimator as ``base_estimator`` for param grid: classifier__base_estimator__<param>.
    """
    _estimator_type = "classifier"

    def __init__(self, base_estimator, target: str = "bfrb"):
        self.base_estimator = base_estimator
        self.target         = target

    def _collapse_y(self, seq_ids, y):
        target_map = (
            y.drop_duplicates("sequence_id")
             .set_index("sequence_id")[self.target]
        )
        y_seq = pd.Series(seq_ids).map(target_map)
        return y_seq

    def _filter(self, X_dict, mask):
        mask = np.asarray(mask)
        return {
            "X":            X_dict["X"][mask],
            "sequence_ids": np.asarray(X_dict["sequence_ids"])[mask],
            "lengths":      np.asarray(X_dict["lengths"])[mask],
        }

    def fit(self, X, y):
        seq_ids = np.asarray(X["sequence_ids"])
        y_seq   = self._collapse_y(seq_ids, y)
        valid   = ~y_seq.isna().values
        if not valid.all():
            X = self._filter(X, valid)
            y_seq = y_seq[valid].reset_index(drop=True)
        if len(y_seq) == 0:
            raise ValueError("No valid labels after alignment.")
        self.estimator_ = clone(self.base_estimator)
        self.estimator_.fit(X, y_seq.values)
        if hasattr(self.estimator_, "classes_"):
            self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        check_is_fitted(self, ["estimator_"])
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self, ["estimator_"])
        return self.estimator_.predict_proba(X)

    def score(self, X, y):
        seq_ids = np.asarray(X["sequence_ids"])
        y_seq   = self._collapse_y(seq_ids, y)
        valid   = ~y_seq.isna().values
        X_ok    = self._filter(X, valid)
        y_ok    = y_seq[valid].reset_index(drop=True)
        return np.mean(self.predict(X_ok) == y_ok.values)


# =============================================================================
# ROCKET CLASSIFIER
# =============================================================================

class RidgeRocketClassifier(ClassifierMixin, BaseEstimator):
    """MiniRocket + StandardScaler + RidgeClassifier."""
    _estimator_type = "classifier"

    def __init__(
        self,
        num_kernels: int = 1000,
        alpha: float = 1.0,
        feature_selection_percentile: Optional[int] = None,
        class_weight: Optional[str] = "balanced",
        random_state: int = 42,
    ):
        self.num_kernels                 = num_kernels
        self.alpha                       = alpha
        self.feature_selection_percentile = feature_selection_percentile
        self.class_weight                = class_weight
        self.random_state                = random_state

    def _extract(self, X, fit: bool = False):
        from sklearn.linear_model import RidgeClassifier
        arr = X["X"] if isinstance(X, dict) else X
        X_r = np.transpose(arr, (0, 2, 1))  # (N, C, T) for sktime
        if fit:
            if not HAS_ROCKET:
                raise ImportError("sktime not installed — cannot use RidgeRocketClassifier.")
            self.rocket_   = MiniRocket(num_kernels=self.num_kernels, random_state=self.random_state)
            self.rocket_.fit(X_r)
            feats = self.rocket_.transform(X_r)
            self.scaler_   = StandardScaler()
            feats = self.scaler_.fit_transform(feats)
            if self.feature_selection_percentile is not None:
                self._selector = SelectPercentile(f_classif, percentile=self.feature_selection_percentile)
                feats = self._selector.fit_transform(feats, self._fit_labels_)
            else:
                self._selector = None
        else:
            check_is_fitted(self, ["rocket_"])
            feats = self.rocket_.transform(X_r)
            feats = self.scaler_.transform(feats)
            if self._selector is not None:
                feats = self._selector.transform(feats)
        return feats

    def fit(self, X, y):
        from sklearn.linear_model import RidgeClassifier
        self._fit_labels_ = np.asarray(y)
        feats = self._extract(X, fit=True)
        self.clf_ = RidgeClassifier(alpha=self.alpha, class_weight=self.class_weight)
        self.clf_.fit(feats, y)
        self.classes_ = self.clf_.classes_
        return self

    def predict(self, X):
        return self.clf_.predict(self._extract(X, fit=False))

    def predict_proba(self, X):
        dec = self.clf_.decision_function(self._extract(X, fit=False))
        if dec.ndim == 1:
            dec = np.column_stack([-dec, dec])
        exp = np.exp(dec - dec.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    def score(self, X, y):
        return np.mean(self.predict(X) == np.asarray(y))

    def get_params(self, deep=True):
        return {
            "num_kernels":                 self.num_kernels,
            "alpha":                       self.alpha,
            "feature_selection_percentile": self.feature_selection_percentile,
            "class_weight":                self.class_weight,
            "random_state":                self.random_state,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        # reset cached state so CV refits cleanly
        for attr in ("rocket_", "scaler_", "_selector", "clf_", "_fit_labels_"):
            self.__dict__.pop(attr, None)
        return self


# =============================================================================
# 1D CNN CLASSIFIER
# =============================================================================

class Keras1DCNNClassifier(ClassifierMixin, BaseEstimator):
    """
    Multi-block 1D CNN for temporal tensors (dict input).

    Hyperparameter strings (compatible with param grid):
      filters  : "32-64" | "64-128" | "128-256"
      kernels  : "3-3"   | "5-3"    | "7-5-3"
      pools    : "none"  | "2"      | "2-2"
    """
    _estimator_type = "classifier"

    def __init__(
        self,
        filters: str = "32-64",
        kernels: str = "3-3",
        pools: str = "2",
        dropout: float = 0.3,
        spatial_dropout: float = 0.1,
        l2_reg: float = 1e-4,
        learning_rate: float = 5e-3,
        batch_size: int = 32,
        epochs: int = 20,
        patience: int = 10,
        verbose: int = 0,
        random_state: int = 42,
    ):
        self.filters        = filters
        self.kernels        = kernels
        self.pools          = pools
        self.dropout        = dropout
        self.spatial_dropout = spatial_dropout
        self.l2_reg         = l2_reg
        self.learning_rate  = learning_rate
        self.batch_size     = batch_size
        self.epochs         = epochs
        self.patience       = patience
        self.verbose        = verbose
        self.random_state   = random_state

    def _parse_int_list(self, s: str):
        return [int(x) for x in str(s).split("-") if x.strip()]

    def _build(self, input_shape, n_classes):
        reg      = keras.regularizers.l2(self.l2_reg)
        f_list   = self._parse_int_list(self.filters)
        k_list   = self._parse_int_list(self.kernels)
        p_list   = self._parse_int_list(self.pools) if self.pools != "none" else []

        inputs = keras.Input(shape=input_shape)
        x = inputs

        for i, (f, k) in enumerate(zip(f_list, k_list + [k_list[-1]] * len(f_list))):
            x = layers.Conv1D(f, k, padding="same", activation="relu",
                              kernel_regularizer=reg)(x)
            x = layers.BatchNormalization()(x)
            if self.spatial_dropout > 0:
                x = layers.SpatialDropout1D(self.spatial_dropout)(x)
            if i < len(p_list):
                x = layers.MaxPooling1D(p_list[i])(x)

        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation="relu", kernel_regularizer=reg)(x)
        x = layers.Dropout(self.dropout)(x)
        outputs = layers.Dense(n_classes, activation="softmax")(x)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, X, y):
        if not HAS_TF:
            raise ImportError("tensorflow not installed.")
        arr = X["X"] if isinstance(X, dict) else np.asarray(X)
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(np.asarray(y))
        self.classes_ = self.le_.classes_

        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)

        self.model_ = self._build((arr.shape[1], arr.shape[2]), len(self.classes_))

        weights = compute_class_weight("balanced", classes=np.unique(y_enc), y=y_enc)
        class_weight = dict(enumerate(weights))

        callbacks = [keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=self.patience, restore_best_weights=True
        )]
        self.history_ = self.model_.fit(
            arr, y_enc,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.15,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=self.verbose,
        )
        return self

    def predict(self, X):
        arr = X["X"] if isinstance(X, dict) else np.asarray(X)
        proba = self.model_.predict(arr, verbose=0)
        return self.le_.inverse_transform(np.argmax(proba, axis=1))

    def predict_proba(self, X):
        arr = X["X"] if isinstance(X, dict) else np.asarray(X)
        return self.model_.predict(arr, verbose=0)

    def score(self, X, y):
        return np.mean(self.predict(X) == np.asarray(y))


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def build_feature_extractor(
    feature_mode: str,
    **feature_kwargs,
) -> BaseEstimator:
    """
    Return the appropriate transformer for feature_mode.

    feature_mode options
    --------------------
    tabular_simple       -> SimpleTabularExtractor
    tabular_honeycomb    -> HoneycombTabularExtractor
    temporal_honeycomb   -> SequenceTensorExtractor
    temporal_raw         -> SequenceTensorExtractor (acc raw only, no engineering)
    """
    mode = feature_mode.lower().strip()
    if mode == "tabular_simple":
        return SimpleTabularExtractor()
    elif mode == "tabular_honeycomb":
        return HoneycombTabularExtractor(**feature_kwargs)
    elif mode in ("temporal_honeycomb", "temporal_raw"):
        kw = dict(feature_kwargs)
        if mode == "temporal_raw":
            kw.setdefault("acc_modes", "raw")
            kw.setdefault("rotation_modes", "quaternion")
            kw.setdefault("tof_modes", "pooled_stats")
            kw.setdefault("thm_modes", "raw")
        return SequenceTensorExtractor(**kw)
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode!r}. "
                         f"Choose from: tabular_simple, tabular_honeycomb, "
                         f"temporal_honeycomb, temporal_raw")


def build_classifier(
    classifier_name: str,
    target_col: str = "bfrb",
    random_state: int = 42,
    **classifier_kwargs,
) -> BaseEstimator:
    """
    Build a sklearn-compatible classifier wrapped for sequence-level prediction.

    classifier_name options
    -----------------------
    dummy        -> DummyClassifier (majority class)
    rf           -> RandomForestClassifier via ManyToOneWrapper
    ridge_rocket -> RidgeRocketClassifier via ManyToOneWrapperTemporal
    cnn          -> Keras1DCNNClassifier via ManyToOneWrapperTemporal
    """
    name = classifier_name.lower().strip()

    if name == "dummy":
        base = DummyClassifier(strategy="most_frequent", random_state=random_state)
        return ManyToOneWrapper(base_estimator=base, target=target_col)

    elif name == "rf":
        kw = dict(n_estimators=300, max_depth=None, class_weight="balanced",
                  n_jobs=-1, random_state=random_state)
        kw.update(classifier_kwargs)
        base = RandomForestClassifier(**kw)
        return ManyToOneWrapper(base_estimator=base, target=target_col)

    elif name == "ridge_rocket":
        kw = dict(num_kernels=500, alpha=1.0, class_weight="balanced",
                  random_state=random_state)
        kw.update(classifier_kwargs)
        base = RidgeRocketClassifier(**kw)
        return ManyToOneWrapperTemporal(base_estimator=base, target=target_col)

    elif name == "cnn":
        if not HAS_TF:
            raise ImportError("tensorflow not installed — cannot use 'cnn' classifier.")
        kw = dict(epochs=20, verbose=0, random_state=random_state)
        kw.update(classifier_kwargs)
        base = Keras1DCNNClassifier(**kw)
        return ManyToOneWrapperTemporal(base_estimator=base, target=target_col)

    else:
        raise ValueError(f"Unknown classifier_name: {classifier_name!r}. "
                         f"Choose from: dummy, rf, ridge_rocket, cnn")


def make_baseline_pipeline(
    feature_mode: str,
    classifier_name: str,
    target_col: str = "bfrb",
    feature_kwargs: Optional[Dict[str, Any]] = None,
    classifier_kwargs: Optional[Dict[str, Any]] = None,
    augment: bool = False,
    augment_kwargs: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> Pipeline:
    """
    Assemble a two-step sklearn Pipeline:  extractor  →  classifier.

    The classifier step is named 'classifier' and the extractor 'extractor'
    so that param grids use keys like 'extractor__acc_modes' and
    'classifier__base_estimator__n_estimators'.

    Parameters
    ----------
    feature_mode       : see build_feature_extractor
    classifier_name    : see build_classifier
    target_col         : label column in y DataFrame (default 'bfrb')
    feature_kwargs     : dict forwarded to the extractor constructor
    classifier_kwargs  : dict forwarded to the classifier constructor
    augment            : (tabular only) apply Gaussian noise augmentation
    augment_kwargs     : kwargs for TabularAugmenter
    random_state       : passed to both extractor and classifier
    """
    feature_kwargs    = feature_kwargs    or {}
    classifier_kwargs = classifier_kwargs or {}

    extractor  = build_feature_extractor(feature_mode, **feature_kwargs)
    classifier = build_classifier(classifier_name, target_col=target_col,
                                  random_state=random_state, **classifier_kwargs)

    steps = [("extractor", extractor)]

    if augment and classifier_name in ("dummy", "rf"):
        augmenter = TabularAugmenter(**(augment_kwargs or {}))
        steps.append(("augmenter", augmenter))

    steps.append(("classifier", classifier))
    return Pipeline(steps)


# =============================================================================
# TABULAR AUGMENTER
# =============================================================================

class TabularAugmenter(BaseEstimator, TransformerMixin):
    """Light Gaussian noise augmentation for tabular features (RF only)."""

    def __init__(self, use_gaussian_noise: bool = True, noise_std: float = 0.01):
        self.use_gaussian_noise = use_gaussian_noise
        self.noise_std          = noise_std

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not self.use_gaussian_noise:
            return X
        noise = np.random.normal(0, self.noise_std, size=X.shape)
        if isinstance(X, pd.DataFrame):
            return X + noise
        return X + noise


# =============================================================================
# SCORING AND EVALUATION
# =============================================================================

def competition_score(y_true: pd.DataFrame, y_pred: np.ndarray,
                      target_col: str = "bfrb") -> float:
    """
    Competition F1-macro averaged over is_target and gesture predictions.
    """
    # collapse y_true to sequence level
    y_seq = (
        y_true.drop_duplicates("sequence_id")
              .set_index("sequence_id")
    )

    # align predictions to sequence_ids
    if hasattr(y_pred, "__len__") and len(y_pred) == len(y_seq):
        pred_series = pd.Series(y_pred, index=y_seq.index, name=target_col)
    else:
        pred_series = pd.Series(y_pred, name=target_col)

    true_labels = y_seq[target_col].values
    pred_labels = pred_series.values

    is_target_true = (true_labels != "non_bfrb").astype(int)
    is_target_pred = (pred_labels != "non_bfrb").astype(int)

    bfrb_f1      = f1_score(is_target_true, is_target_pred, average="binary", zero_division=0)
    gesture_mask = is_target_true == 1
    if gesture_mask.sum() > 0:
        gesture_f1 = f1_score(
            true_labels[gesture_mask],
            pred_labels[gesture_mask],
            average="macro",
            zero_division=0,
        )
    else:
        gesture_f1 = 0.0

    return 0.5 * bfrb_f1 + 0.5 * gesture_f1


from sklearn.metrics import make_scorer
competition_scorer = make_scorer(
    competition_score,
    greater_is_better=True,
    response_method="predict",
)


def evaluate_holdout(
    y_true: pd.DataFrame,
    y_pred: np.ndarray,
    target_col: str = "bfrb",
    verbose: bool = False,
) -> dict:
    """
    Evaluate predictions on holdout data.

    Parameters
    ----------
    y_true     : raw holdout DataFrame (has sequence_id and target_col columns)
    y_pred     : predictions from pipeline.predict(X_test), length = n_unique_sequences
    target_col : default 'bfrb'
    verbose    : print classification report

    Returns
    -------
    dict with keys: competition_score, bfrb_f1, gesture_f1, report
    """
    y_seq = (
        y_true.drop_duplicates("sequence_id")
              .set_index("sequence_id")
    )
    true_labels = y_seq[target_col].values
    pred_labels = np.asarray(y_pred)

    if len(pred_labels) != len(true_labels):
        raise ValueError(
            f"Length mismatch: y_pred has {len(pred_labels)} entries "
            f"but holdout has {len(true_labels)} unique sequences."
        )

    is_target_true = (true_labels != "non_bfrb").astype(int)
    is_target_pred = (pred_labels != "non_bfrb").astype(int)

    bfrb_f1 = f1_score(is_target_true, is_target_pred, average="binary", zero_division=0)

    gesture_mask = is_target_true == 1
    if gesture_mask.sum() > 0:
        gesture_f1 = f1_score(
            true_labels[gesture_mask],
            pred_labels[gesture_mask],
            average="macro",
            zero_division=0,
        )
        report = classification_report(true_labels[gesture_mask], pred_labels[gesture_mask], zero_division=0)
    else:
        gesture_f1 = 0.0
        report = "(no target sequences in holdout)"

    score = 0.5 * bfrb_f1 + 0.5 * gesture_f1

    if verbose:
        print(f"\nCompetition score : {score:.4f}")
        print(f"  BFRB detection F1: {bfrb_f1:.4f}")
        print(f"  Gesture F1-macro : {gesture_f1:.4f}")
        print("\nGesture classification report (Target sequences only):")
        print(report)

    return {
        "competition_score": score,
        "bfrb_f1":           bfrb_f1,
        "gesture_f1":        gesture_f1,
        "report":            report,
    }


def plot_training_curves(history) -> None:
    """Plot Keras training curves (loss + accuracy) if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — cannot plot training curves.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ["loss", "accuracy"]):
        if metric in history.history:
            ax.plot(history.history[metric],          label="train")
            val_key = f"val_{metric}"
            if val_key in history.history:
                ax.plot(history.history[val_key],     label="val")
            ax.set_title(metric.capitalize())
            ax.set_xlabel("Epoch")
            ax.legend()
    plt.tight_layout()
    plt.show()
