from __future__ import annotations
from typing import Literal, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import LabelEncoder
import keras
from tensorflow.keras import layers
import tensorflow as tf
from sklearn.metrics import f1_score, make_scorer
import warnings
warnings.filterwarnings('ignore', '.*mask.*Conv1D.*')
import json


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
        acc_cols: Optional[list[str]] = None,
        rot_cols: Optional[list[str]] = None,
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
        acc_cols: Optional[list[str]] = None,
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

        return pd.DataFrame(
            {"acc_mag": mag},
            index=X.index,
        )

    def _linear_acc_magnitude(self, X: pd.DataFrame) -> pd.DataFrame:
        lin_cols = ["lin_acc_x", "lin_acc_y", "lin_acc_z"]

        # Don't raise error, just return empty or use fallback
        if self.linear_acc_mode != 'baseline':
            if self.use_linear_acc_magnitude:
                print(f"Warning: use_linear_acc_magnitude=True but linear_acc_mode={self.linear_acc_mode}")
                # Return zeros instead of failing
                return pd.DataFrame(
                    {"lin_acc_mag": np.zeros(len(X))},
                    index=X.index,
                )

        if not set(lin_cols).issubset(X.columns):
            if self.use_linear_acc_magnitude:
                print(f"Warning: linear acceleration columns not found")
                return pd.DataFrame(
                    {"lin_acc_mag": np.zeros(len(X))},
                    index=X.index,
                )

        lin = X[lin_cols].astype(float)
        mag = np.sqrt(np.sum(lin.to_numpy() ** 2, axis=1))

        return pd.DataFrame(
            {"lin_acc_mag": mag},
            index=X.index,
        )


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
        acc_cols: Optional[list[str]] = None,
        rot_cols: Optional[list[str]] = None,
        rotation_mode: RotationMode = None,
        fix_quaternion_sign: bool = True,
        tof_mode: TOFMode = None,
        tof_fill_mode: TOFFillMode = "far_255",
        thm_mode: THMMode = None,
        tof_cols: Optional[list[str]] = None,
        thm_cols: Optional[list[str]] = None,
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
        check_is_fitted(
            self,
            [
                "cleaner_",
                "imu_extractor_",
                "rotation_extractor_",
                "tof_extractor_",
                "thermo_extractor_",
                "feature_names_in_",
            ],
        )

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


class KerasCNN1DSequenceClassifier(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"

    def __init__(
        self,
        target="gesture_action",
        maxlen=64,
        padding_value=-999.0,
        conv_filters="64",
        kernel_sizes="5",
        pool_sizes="none",
        use_batch_norm=True,
        spatial_dropout=0.0,
        dense_units="64",
        dropout=0.2,
        learning_rate=5e-4,
        batch_size=32,
        epochs=80,
        patience=12,
        verbose=0,
        random_state=42,
    ):
        self.target = target
        self.maxlen = maxlen
        self.padding_value = padding_value
        self.conv_filters = conv_filters
        self.kernel_sizes = kernel_sizes
        self.pool_sizes = pool_sizes
        self.use_batch_norm = use_batch_norm
        self.spatial_dropout = spatial_dropout
        self.dense_units = dense_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.random_state = random_state

    def _to_tuple(self, value):
        if value is None:
            return ()

        if isinstance(value, str):
            if value == "none":
                return ()

            out = []
            for part in value.split("-"):
                if part == "none":
                    out.append(None)
                else:
                    out.append(int(part))

            return tuple(out)

        if isinstance(value, tuple):
            return value

        if isinstance(value, list):
            return tuple(value)

        return (value,)

    def _align(self, value, n: int) -> tuple:
        value = self._to_tuple(value)

        if len(value) == 0:
            return (None,) * n

        if len(value) == n:
            return value

        if len(value) == 1:
            return value * n

        if len(value) < n:
            return value + (value[-1],) * (n - len(value))

        return value[:n]

    def _pad(self, X):
        grouped = list(X.groupby(level=0, sort=False))

        n_seq = len(grouped)
        n_feat = X.shape[1]

        out = np.full(
            (n_seq, self.maxlen, n_feat),
            self.padding_value,
            dtype=np.float32,
        )

        seq_ids = []

        for i, (sid, g) in enumerate(grouped):
            arr = g.to_numpy(dtype=np.float32)
            length = min(len(arr), self.maxlen)

            out[i, :length] = arr[:length]
            seq_ids.append(sid)

        return out, pd.Series(seq_ids, name="sequence_id")

    def _collapse_y(self, seq_ids, y):
        if isinstance(y, pd.DataFrame):
            if "sequence_id" not in y.columns:
                raise ValueError("y dataframe must contain sequence_id.")

            if self.target not in y.columns:
                raise ValueError(f"y dataframe must contain target column: {self.target}")

            target_map = (
                y.drop_duplicates("sequence_id")
                .set_index("sequence_id")[self.target]
            )

            y_seq = seq_ids.map(target_map)

        else:
            y_seq = pd.Series(y).reset_index(drop=True)

            if len(y_seq) != len(seq_ids):
                raise ValueError(
                    "If y is not a dataframe, it must already be one label per sequence."
                )

        if y_seq.isna().any():
            missing = seq_ids[y_seq.isna()].head(10).tolist()
            raise ValueError(f"Missing labels for sequence_ids: {missing}")

        return y_seq.reset_index(drop=True)

    def _build(self, shape, n_classes):
        tf.keras.backend.clear_session()
        keras.utils.set_random_seed(self.random_state)

        conv_filters = self._to_tuple(self.conv_filters)
        n_layers = len(conv_filters)

        if n_layers == 0:
            raise ValueError("conv_filters cannot be empty.")

        kernel_sizes = self._align(self.kernel_sizes, n_layers)
        pool_sizes = self._align(self.pool_sizes, n_layers)
        dense_units = self._to_tuple(self.dense_units)

        inp = keras.Input(shape=shape)
        x = inp

        for filters, kernel_size, pool_size in zip(conv_filters, kernel_sizes, pool_sizes):
            x = layers.Conv1D(
                filters=int(filters),
                kernel_size=int(kernel_size),
                padding="same",
                activation="relu",
            )(x)

            if self.use_batch_norm:
                x = layers.BatchNormalization()(x)

            if float(self.spatial_dropout) > 0:
                x = layers.SpatialDropout1D(
                    rate=float(self.spatial_dropout)
                )(x)

            if pool_size is not None:
                x = layers.MaxPooling1D(
                    pool_size=int(pool_size)
                )(x)

        x = layers.GlobalAveragePooling1D()(x)

        for units in dense_units:
            x = layers.Dense(
                int(units),
                activation="relu",
            )(x)

            if float(self.dropout) > 0:
                x = layers.Dropout(
                    rate=float(self.dropout)
                )(x)

        out = layers.Dense(
            n_classes,
            activation="softmax",
        )(x)

        model = keras.Model(inp, out)

        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=float(self.learning_rate)
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def fit(self, X, y):
        X_pad, seq_ids = self._pad(X)
        y_seq = self._collapse_y(seq_ids, y)

        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y_seq)
        self.classes_ = self.le_.classes_

        self.model_ = self._build(
            shape=(X_pad.shape[1], X_pad.shape[2]),
            n_classes=len(self.classes_),
        )

        self.history_ = self.model_.fit(
            X_pad,
            y_enc,
            batch_size=int(self.batch_size),
            epochs=int(self.epochs),
            validation_split=0.15,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    patience=int(self.patience),
                    restore_best_weights=True,
                )
            ],
            verbose=int(self.verbose),
            shuffle=True,
        )

        return self

    def predict_proba(self, X):
        X_pad, _ = self._pad(X)
        return self.model_.predict(X_pad, verbose=0)

    def predict(self, X):
        proba = self.predict_proba(X)
        pred_idx = np.argmax(proba, axis=1)
        return self.le_.inverse_transform(pred_idx)

    def score(self, X, y):
        _, seq_ids = self._pad(X)
        y_seq = self._collapse_y(seq_ids, y)
        preds = self.predict(X)
        return f1_score(y_seq.to_numpy(), preds, average="macro")


class SequenceScorer:
    def __init__(self, metric: str = 'f1_macro', sequence_col: str = 'sequence_id', target_col: str = 'gesture') -> None:
        self.metric = metric
        self.sequence_col = sequence_col
        self.target_col = target_col

    def __call__(self, y_true: pd.DataFrame | pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
        if isinstance(y_true, pd.DataFrame):
            y_true = y_true.drop_duplicates(self.sequence_col).reset_index(drop=True)[self.target_col]
        else:
            y_true = pd.Series(y_true).reset_index(drop=True)

        if self.metric == 'f1_macro':
            return f1_score(y_true, y_pred, average='macro')
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")


class RotationExtractor:
    def __init__(
        self,
        rotation_mode: RotationMode = None,
        sequence_col: str = "sequence_id",
        counter_col: str = "sequence_counter",
        dt_col: str = "dt",
        sampling_rate: int = 20,
        rot_cols: Optional[list[str]] = None,
        fix_quaternion_sign: bool = True,
    ) -> None:
        self.rotation_mode = rotation_mode
        self.sequence_col = sequence_col
        self.counter_col = counter_col
        self.dt_col = dt_col
        self.sampling_rate = sampling_rate
        self.rot_cols = rot_cols
        self.fix_quaternion_sign = fix_quaternion_sign

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.DataFrame] = None,
    ) -> "RotationExtractor":
        self.rot_cols_ = self.rot_cols or [
            c for c in ["rot_w", "rot_x", "rot_y", "rot_z"]
            if c in X.columns
        ]

        if len(self.rot_cols_) != 4:
            raise ValueError(f"Expected 4 quaternion columns, got: {self.rot_cols_}")

        return self

    def transform(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
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

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def _get_dt(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        if self.dt_col in X.columns:
            dt = X[self.dt_col].astype(float).to_numpy()

        elif self.counter_col in X.columns:
            dt = (
                X.groupby(self.sequence_col, sort=False)[self.counter_col]
                .diff()
                .fillna(1.0)
                / float(self.sampling_rate)
            ).astype(float).to_numpy()

        else:
            dt = np.full(
                len(X),
                1.0 / float(self.sampling_rate),
                dtype=float,
            )

        dt = np.nan_to_num(
            dt,
            nan=1.0 / float(self.sampling_rate),
            posinf=1.0 / float(self.sampling_rate),
            neginf=1.0 / float(self.sampling_rate),
        )

        return np.maximum(dt, 1e-6)

    def _get_quaternion(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        q = X[self.rot_cols_].astype(float).to_numpy()
        q = self._normalise_quaternion(q)

        if self.fix_quaternion_sign:
            q = self._fix_sign_continuity(
                q,
                X[self.sequence_col].to_numpy(),
            )

        return q

    def _quaternion(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        q = self._get_quaternion(X)

        return pd.DataFrame(
            q,
            columns=[
                "rot_w_norm",
                "rot_x_norm",
                "rot_y_norm",
                "rot_z_norm",
            ],
            index=X.index,
        )

    def _euler(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        euler = self._quaternion_to_euler(
            self._get_quaternion(X)
        )

        return pd.DataFrame(
            euler,
            columns=[
                "rot_roll",
                "rot_pitch",
                "rot_yaw",
            ],
            index=X.index,
        )

    def _delta_euler(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        euler_df = self._euler(X)

        out = (
            euler_df
            .groupby(X[self.sequence_col].values, sort=False)
            .diff()
            .fillna(0.0)
        )

        out.columns = [
            "delta_rot_roll",
            "delta_rot_pitch",
            "delta_rot_yaw",
        ]

        return out

    def _angular_velocity(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        q = self._get_quaternion(X)
        dt = self._get_dt(X)

        out = np.zeros((len(X), 3), dtype=float)

        for _, idx in X.groupby(self.sequence_col, sort=False).groups.items():
            pos = X.index.get_indexer(idx)

            if len(pos) <= 1:
                continue

            q_prev = q[pos[:-1]]
            q_curr = q[pos[1:]]

            q_rel = self._quaternion_multiply(
                q_curr,
                self._quaternion_conjugate(q_prev),
            )

            q_rel = self._normalise_quaternion(q_rel)

            q_rel[q_rel[:, 0] < 0] *= -1.0

            w = np.clip(q_rel[:, 0], -1.0, 1.0)
            xyz = q_rel[:, 1:4]

            angle = 2.0 * np.arctan2(
                np.linalg.norm(xyz, axis=1),
                w,
            )

            small = angle < 1e-8

            axis = np.zeros_like(xyz)
            norms = np.linalg.norm(xyz, axis=1)
            axis[~small] = xyz[~small] / norms[~small, None]

            safe_dt = dt[pos[1:]]

            out[pos[1:], :] = axis * (angle / safe_dt)[:, None]

        return pd.DataFrame(
            out,
            columns=[
                "ang_vel_x",
                "ang_vel_y",
                "ang_vel_z",
            ],
            index=X.index,
        )

    def _rot6d(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        R = self._quaternion_to_rotation_matrix(
            self._get_quaternion(X)
        )

        rot6d = np.concatenate(
            [
                R[:, :, 0],
                R[:, :, 1],
            ],
            axis=1,
        )

        return pd.DataFrame(
            rot6d,
            columns=[
                "rot6d_c1_x",
                "rot6d_c1_y",
                "rot6d_c1_z",
                "rot6d_c2_x",
                "rot6d_c2_y",
                "rot6d_c2_z",
            ],
            index=X.index,
        )

    @staticmethod
    def _normalise_quaternion(
        q: np.ndarray,
    ) -> np.ndarray:
        q = q.astype(float).copy()

        norms = np.linalg.norm(q, axis=1, keepdims=True)
        bad = (~np.isfinite(norms)) | (norms == 0.0)

        norms[bad] = 1.0
        q = q / norms

        q[bad[:, 0]] = np.array([1.0, 0.0, 0.0, 0.0])

        return q

    @staticmethod
    def _fix_sign_continuity(
        q: np.ndarray,
        seq_ids: np.ndarray,
    ) -> np.ndarray:
        q = q.copy()

        for seq_id in pd.unique(seq_ids):
            pos = np.flatnonzero(seq_ids == seq_id)

            for i in range(1, len(pos)):
                if np.dot(q[pos[i - 1]], q[pos[i]]) < 0:
                    q[pos[i]] *= -1.0

        return q

    @staticmethod
    def _quaternion_conjugate(
        q: np.ndarray,
    ) -> np.ndarray:
        out = q.copy()
        out[:, 1:] *= -1.0
        return out

    @staticmethod
    def _quaternion_multiply(
        q1: np.ndarray,
        q2: np.ndarray,
    ) -> np.ndarray:
        w1 = q1[:, 0]
        x1 = q1[:, 1]
        y1 = q1[:, 2]
        z1 = q1[:, 3]

        w2 = q2[:, 0]
        x2 = q2[:, 1]
        y2 = q2[:, 2]
        z2 = q2[:, 3]

        return np.column_stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    @staticmethod
    def _quaternion_to_euler(
        q: np.ndarray,
    ) -> np.ndarray:
        w = q[:, 0]
        x = q[:, 1]
        y = q[:, 2]
        z = q[:, 3]

        roll = np.arctan2(
            2.0 * (w*x + y*z),
            1.0 - 2.0 * (x*x + y*y),
        )

        pitch = np.arcsin(
            np.clip(
                2.0 * (w*y - z*x),
                -1.0,
                1.0,
            )
        )

        yaw = np.arctan2(
            2.0 * (w*z + x*y),
            1.0 - 2.0 * (y*y + z*z),
        )

        return np.column_stack([
            roll,
            pitch,
            yaw,
        ])

    @staticmethod
    def _quaternion_to_rotation_matrix(
        q: np.ndarray,
    ) -> np.ndarray:
        w = q[:, 0]
        x = q[:, 1]
        y = q[:, 2]
        z = q[:, 3]

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
        tof_cols: Optional[list[str]] = None,
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

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def _clean_df(self, X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        df = X[cols].astype(float).copy()
        df = df.replace(self.invalid_value, np.nan)

        if self.tof_fill_mode == "nan_interpolate":
            df[self.sequence_col] = X[self.sequence_col].values
            df[cols] = (
                df.groupby(self.sequence_col, sort=False)[cols]
                .transform(lambda g: g.interpolate(method="linear", limit_direction="both").ffill().bfill())
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

    def _sensor_cols(self, sensor: int) -> list[str]:
        cols = [
            f"tof_{sensor}_v{pixel}"
            for pixel in range(self.grid_size * self.grid_size)
        ]

        if len(cols) != self.grid_size * self.grid_size:
            raise ValueError(f"Bad TOF column setup for sensor {sensor}.")

        return cols

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
            (0, 3, 0, 3),
            (0, 3, 3, 5),
            (0, 3, 5, 8),
            (3, 5, 0, 3),
            (3, 5, 3, 5),
            (3, 5, 5, 8),
            (5, 8, 0, 3),
            (5, 8, 3, 5),
            (5, 8, 5, 8),
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
        return (
            df.groupby(X[self.sequence_col].values, sort=False)
            .diff()
            .fillna(0.0)
        )


class ThermoExtractor:
    def __init__(
        self,
        thm_mode: THMMode = None,
        sequence_col: str = "sequence_id",
        thm_cols: Optional[list[str]] = None,
    ) -> None:
        self.thm_mode = thm_mode
        self.sequence_col = sequence_col
        self.thm_cols = thm_cols

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "ThermoExtractor":
        self.thm_cols_ = self.thm_cols or [
            c for c in X.columns
            if c.startswith("thm_")
        ]

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

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

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
        return (
            df.groupby(X[self.sequence_col].values, sort=False)
            .diff()
            .fillna(0.0)
        )


# ----------------------------------------------------------------------------
# Multi‐Branch 1D CNN classifier with optional Self‐Attention / BiGRU fusion
# ----------------------------------------------------------------------------
class KerasMultiBranchClassifier(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"

    def __init__(
            self,
            target: str = "gesture_action",
            maxlen: int = 64,
            padding_value: float = -999.0,

            # which columns go to which branch
            branch_config: dict | None = None,

            # per‑branch Conv1D architecture strings (hyphen‑separated)
            branch_filters: dict | None = None,
            branch_kernel_sizes: dict | None = None,
            branch_pool_sizes: dict | None = None,

            # fusion after branch concatenation
            fusion_mode: str = "attention",
            attention_heads: int = 4,
            gru_units: int = 128,

            # common to all branches
            use_batch_norm: bool = True,
            spatial_dropout: float = 0.1,

            # classification head
            dense_units: str = "64",
            dropout: float = 0.3,
            learning_rate: float = 5e-4,
            batch_size: int = 32,
            epochs: int = 80,
            patience: int = 12,
            verbose: int = 0,
            random_state: int = 42,
    ):
        self.target = target
        self.maxlen = maxlen
        self.padding_value = padding_value

        self.branch_config = branch_config
        self.branch_filters = branch_filters
        self.branch_kernel_sizes = branch_kernel_sizes
        self.branch_pool_sizes = branch_pool_sizes

        self.fusion_mode = fusion_mode
        self.attention_heads = attention_heads
        self.gru_units = gru_units

        self.use_batch_norm = use_batch_norm
        self.spatial_dropout = spatial_dropout

        self.dense_units = dense_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _to_tuple(self, value):
        if value is None:
            return ()
        if isinstance(value, str):
            if value == "none":
                return ()
            parts = value.split("-")
            out = []
            for p in parts:
                out.append(None if p == "none" else int(p))
            return tuple(out)
        if isinstance(value, tuple):
            return value
        if isinstance(value, list):
            return tuple(value)
        return (value,)

    def _align(self, value, n: int) -> tuple:
        value = self._to_tuple(value)
        if len(value) == 0:
            return (None,) * n
        if len(value) == n:
            return value
        if len(value) == 1:
            return value * n
        if len(value) < n:
            return value + (value[-1],) * (n - len(value))
        return value[:n]

    # ------------------------------------------------------------------
    # Default branch definitions
    # ------------------------------------------------------------------
    @staticmethod
    def _default_branch_config() -> dict:
        return {
            "acc": ["acc_", "lin_acc_"],
            "rot": ["rot_", "delta_rot_", "ang_vel_", "rot6d_"],
            "tof": ["tof_"],
            "thm": ["thm_"],
        }

    @staticmethod
    def _default_branch_filters() -> dict:
        return {
            "acc": "64-128",
            "rot": "32-64",
            "tof": "32",
            "thm": "16",
        }

    @staticmethod
    def _default_branch_kernel_sizes() -> dict:
        return {
            "acc": "3-3",
            "rot": "3-3",
            "tof": "3",
            "thm": "3",
        }

    @staticmethod
    def _default_branch_pool_sizes() -> dict:
        return {
            "acc": "none-none",
            "rot": "none-none",
            "tof": "none",
            "thm": "none",
        }

    # ------------------------------------------------------------------
    # Helper to safely merge dict params (FIX FOR BAYESIAN)
    # ------------------------------------------------------------------
    def _get_branch_param(self, param_dict: dict | None, branch_name: str, default: str) -> str:
        """Safely extract branch parameter, handling None and missing keys."""
        if param_dict is None:
            return default
        if isinstance(param_dict, dict):
            return param_dict.get(branch_name, default)
        # If it's something else (rare), return default
        return default

    # ------------------------------------------------------------------
    # Determine column indices for each branch
    # ------------------------------------------------------------------
    def _get_branch_indices(self, all_columns: pd.Index):
        config = self.branch_config or self._default_branch_config()
        branch_order = []
        branch_indices = {}
        remaining = set(range(len(all_columns)))

        for name, prefixes in config.items():
            idxs = []
            for i, col in enumerate(all_columns):
                if any(str(col).startswith(p) for p in prefixes):
                    idxs.append(i)
            if idxs:
                branch_order.append(name)
                branch_indices[name] = sorted(idxs)
                remaining -= set(idxs)

        if remaining:
            branch_order.append("other")
            branch_indices["other"] = sorted(remaining)

        return branch_order, branch_indices

    # ------------------------------------------------------------------
    # Pad sequences
    # ------------------------------------------------------------------
    def _pad(self, X):
        if hasattr(X, 'groupby') and hasattr(X.index, 'levels') and len(X.index.levels) > 1:
            grouped = list(X.groupby(level=0, sort=False))
        elif hasattr(X, 'groupby'):
            grouped = list(X.groupby(X.index, sort=False))
        else:
            grouped = [(i, X) for i in range(len(X))]

        n_seq = len(grouped)
        n_feat = X.shape[1] if hasattr(X, 'shape') else 1

        out = np.full(
            (n_seq, self.maxlen, n_feat),
            self.padding_value,
            dtype=np.float32,
        )

        seq_ids = []

        for i, (sid, g) in enumerate(grouped):
            if hasattr(g, 'to_numpy'):
                arr = g.to_numpy(dtype=np.float32)
            else:
                arr = np.array(g, dtype=np.float32)

            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)

            length = min(len(arr), self.maxlen)
            out[i, :length, :] = arr[:length, :]
            seq_ids.append(sid)

        return out, pd.Series(seq_ids, name="sequence_id")

    # ------------------------------------------------------------------
    # Collapse y
    # ------------------------------------------------------------------
    def _collapse_y(self, seq_ids, y):
        if isinstance(y, pd.DataFrame):
            if "sequence_id" not in y.columns:
                raise ValueError("y dataframe must contain sequence_id.")
            if self.target not in y.columns:
                raise ValueError(f"y dataframe must contain target column: {self.target}")
            target_map = (
                y.drop_duplicates("sequence_id")
                    .set_index("sequence_id")[self.target]
            )
            y_seq = seq_ids.map(target_map)
        else:
            y_seq = pd.Series(y).reset_index(drop=True)
            if len(y_seq) != len(seq_ids):
                raise ValueError(
                    "If y is not a dataframe, it must already be one label per sequence."
                )

        if y_seq.isna().any():
            missing = seq_ids[y_seq.isna()].head(10).tolist()
            raise ValueError(f"Missing labels for sequence_ids: {missing}")

        return y_seq.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Build Keras model
    # ------------------------------------------------------------------
    def _build(self, n_features: int, n_classes: int):
        tf.keras.backend.clear_session()
        keras.utils.set_random_seed(self.random_state)

        branch_filters_cfg = self.branch_filters if self.branch_filters is not None else self._default_branch_filters()
        branch_kernels_cfg = self.branch_kernel_sizes if self.branch_kernel_sizes is not None else self._default_branch_kernel_sizes()
        branch_pools_cfg = self.branch_pool_sizes if self.branch_pool_sizes is not None else self._default_branch_pool_sizes()

        inp = keras.Input(shape=(self.maxlen, n_features), name="input")

        branch_outs = []
        for br_name in self.branch_order_:
            idxs = self.branch_indices_[br_name]

            br_x = layers.Lambda(
                lambda t, i=idxs: tf.gather(t, i, axis=-1),
                name=f"branch_{br_name}_slice",
            )(inp)

            f_str = self._get_branch_param(branch_filters_cfg, br_name, "32")
            k_str = self._get_branch_param(branch_kernels_cfg, br_name, "3")
            p_str = self._get_branch_param(branch_pools_cfg, br_name, "none")

            filters = self._to_tuple(f_str)
            n = len(filters) if filters else 1
            if n == 0:
                filters = (32,)
                n = 1

            kernels = self._align(k_str, n)
            pools = self._align(p_str, n)

            x = br_x
            for f, k, p in zip(filters, kernels, pools):
                x = layers.Conv1D(
                    filters=int(f),
                    kernel_size=int(k),
                    padding="same",
                    activation="relu",
                )(x)

                if self.use_batch_norm:
                    x = layers.BatchNormalization()(x)

                if float(self.spatial_dropout) > 0:
                    x = layers.SpatialDropout1D(rate=float(self.spatial_dropout))(x)

                if p is not None:
                    x = layers.MaxPooling1D(pool_size=int(p))(x)

            branch_outs.append(x)

        # Fusion
        if len(branch_outs) == 1:
            x = branch_outs[0]
            branches_pooled = False
        else:
            time_dims = [b.shape[1] for b in branch_outs]
            if len(set(time_dims)) > 1:
                # branches have mismatched time dims due to different pool configs
                # collapse each to (batch, filters) before concat
                branch_outs = [
                    layers.GlobalAveragePooling1D(name=f"branch_pool_{name}")(b)
                    for name, b in zip(self.branch_order_, branch_outs)
                ]
                x = layers.Concatenate(axis=-1, name="branch_concat")(branch_outs)
                branches_pooled = True
            else:
                x = layers.Concatenate(axis=-1, name="branch_concat")(branch_outs)
                branches_pooled = False

        if branches_pooled:
            # x is (batch, total_filters) — reshape to sequence of 1 for attention/gru
            if self.fusion_mode == "attention":
                x = layers.Reshape((1, x.shape[-1]))(x)
                attn = layers.MultiHeadAttention(
                    num_heads=int(self.attention_heads),
                    key_dim=max(1, x.shape[-1] // int(self.attention_heads)),
                    name="fusion_attn",
                )(x, x)
                x = layers.Add()([x, attn])
                x = layers.LayerNormalization()(x)
                x = layers.Flatten()(x)
            elif self.fusion_mode == "bigru":
                # treat each branch vector as a token: (batch, n_branches, filters)
                stacked = [
                    layers.Reshape((1, b.shape[-1]))(b)
                    for b in branch_outs
                ]
                x = layers.Concatenate(axis=1, name="branch_tokens")(stacked)
                x = layers.Bidirectional(
                    layers.GRU(int(self.gru_units), return_sequences=False),
                    name="fusion_bigru",
                )(x)
            # fusion_mode == "none": x stays as flat concat
        else:
            # x is (batch, time, features) — normal temporal fusion
            if self.fusion_mode == "attention":
                attn = layers.MultiHeadAttention(
                    num_heads=int(self.attention_heads),
                    key_dim=max(1, x.shape[-1] // int(self.attention_heads)),
                    name="fusion_attn",
                )(x, x)
                x = layers.Add()([x, attn])
                x = layers.LayerNormalization()(x)
            elif self.fusion_mode == "bigru":
                x = layers.Bidirectional(
                    layers.GRU(int(self.gru_units), return_sequences=True),
                    name="fusion_bigru",
                )(x)

            x = layers.GlobalAveragePooling1D(name="global_pool")(x)

        # Classification head
        dense_units = self._to_tuple(self.dense_units)
        for units in dense_units:
            x = layers.Dense(int(units), activation="relu")(x)
            if float(self.dropout) > 0:
                x = layers.Dropout(rate=float(self.dropout))(x)

        out = layers.Dense(n_classes, activation="softmax", name="output")(x)

        model = keras.Model(inp, out)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=float(self.learning_rate)),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    # ------------------------------------------------------------------
    # fit / predict / predict_proba / score
    # ------------------------------------------------------------------
    def fit(self, X, y):
        # Determine branch splits from column names
        if hasattr(X, "columns"):
            self.branch_order_, self.branch_indices_ = self._get_branch_indices(X.columns)
        else:
            self.branch_order_ = ["all"]
            self.branch_indices_ = {"all": list(range(X.shape[1])) if hasattr(X, 'shape') else [0]}

        X_pad, seq_ids = self._pad(X)
        y_seq = self._collapse_y(seq_ids, y)

        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y_seq)
        self.classes_ = self.le_.classes_

        self.model_ = self._build(
            n_features=X_pad.shape[2],
            n_classes=len(self.classes_),
        )

        self.history_ = self.model_.fit(
            X_pad,
            y_enc,
            batch_size=int(self.batch_size),
            epochs=int(self.epochs),
            validation_split=0.15,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    patience=int(self.patience),
                    restore_best_weights=True,
                )
            ],
            verbose=int(self.verbose),
            shuffle=True,
        )
        return self

    def predict_proba(self, X):
        X_pad, _ = self._pad(X)
        return self.model_.predict(X_pad, verbose=0)

    def predict(self, X):
        proba = self.predict_proba(X)
        pred_idx = np.argmax(proba, axis=1)
        return self.le_.inverse_transform(pred_idx)

    def score(self, X, y):
        X_pad, seq_ids = self._pad(X)
        y_seq = self._collapse_y(seq_ids, y)
        proba = self.model_.predict(X_pad, verbose=0)
        pred_idx = np.argmax(proba, axis=1)
        preds = self.le_.inverse_transform(pred_idx)
        return f1_score(y_seq.to_numpy(), preds, average="macro")

    def set_params(self, **params):
        """Override set_params to auto-decode JSON-encoded branch dicts."""
        import json
        BRANCH_DICT_PARAMS = {"branch_filters", "branch_kernel_sizes", "branch_pool_sizes"}
        decoded = {}
        for k, v in params.items():
            if k in BRANCH_DICT_PARAMS and isinstance(v, str):
                try:
                    decoded[k] = json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    decoded[k] = v
            else:
                decoded[k] = v
        return super().set_params(**decoded)


def prepare_multibranch_param_space(param_space, search_mode, Categorical=None, ECat=None):
    """
    Encode dict-valued branch params (branch_filters, branch_kernel_sizes,
    branch_pool_sizes) as JSON strings so BayesSearchCV can handle them.

    For 'bayesian' mode: wraps lists of dicts in skopt Categorical as JSON strings.
    For 'evolutionary' mode: wraps in sklearn_genetic ECat as JSON strings.
    For 'grid'/'random' mode: leaves as-is (plain lists of dicts work fine).

    The KerasMultiBranchClassifier.set_params() must decode these back via
    _decode_branch_params().
    """
    BRANCH_DICT_PARAMS = {"branch_filters", "branch_kernel_sizes", "branch_pool_sizes"}

    if search_mode not in ("bayesian", "evolutionary"):
        return param_space  # grid/random: dicts in lists work natively

    out = {}
    for key, val in param_space.items():
        # Check if this is a branch dict param (key ends with one of our special names)
        param_name = key.split("__")[-1]
        if param_name in BRANCH_DICT_PARAMS and isinstance(val, list) and val and isinstance(val[0], dict):
            encoded = [json.dumps(d, sort_keys=True) for d in val]
            if search_mode == "bayesian":
                out[key] = Categorical(encoded)
            elif search_mode == "evolutionary":
                out[key] = ECat(encoded)
        else:
            out[key] = val
    return out