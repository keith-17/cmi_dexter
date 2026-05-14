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
        fusion_mode: str = "attention",   # "attention", "bigru", "none"
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
    # Helpers – identical to your existing KerasCNN1DSequenceClassifier
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
        # keep same temporal length for fusion
        return {
            "acc": "none-none",
            "rot": "none-none",
            "tof": "none",
            "thm": "none",
        }

    # ------------------------------------------------------------------
    # Determine column indices for each branch from DataFrame columns
    # ------------------------------------------------------------------
    def _get_branch_indices(self, all_columns: pd.Index):
        config = self.branch_config or self._default_branch_config()
        branch_order = []
        branch_indices = {}   # name -> list of int positions
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
        # if any columns left unmatched, assign them to an "other" branch
        if remaining:
            branch_order.append("other")
            branch_indices["other"] = sorted(remaining)
        return branch_order, branch_indices

    # ------------------------------------------------------------------
    # Pad (identical to your existing classifier)
    # ------------------------------------------------------------------
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
            out[i, :length, :] = arr[:length, :]
            seq_ids.append(sid)
        return out, pd.Series(seq_ids, name="sequence_id")

    # ------------------------------------------------------------------
    # Collapse y (identical)
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

        # --- branch configuration ---
        branch_filters_cfg = self.branch_filters or self._default_branch_filters()
        branch_kernels_cfg = self.branch_kernel_sizes or self._default_branch_kernel_sizes()
        branch_pools_cfg = self.branch_pool_sizes or self._default_branch_pool_sizes()

        inp = keras.Input(shape=(self.maxlen, n_features), name="input")

        # --- process each branch ---
        branch_outs = []
        for br_name in self.branch_order_:
            idxs = self.branch_indices_[br_name]
            # extract branch features
            br_x = layers.Lambda(
                lambda t, i=idxs: tf.gather(t, i, axis=-1),
                name=f"branch_{br_name}_slice",
            )(inp)

            # parse architecture for this branch
            f_str = branch_filters_cfg.get(br_name, "32")
            k_str = branch_kernels_cfg.get(br_name, "3")
            p_str = branch_pools_cfg.get(br_name, "none")

            filters = self._to_tuple(f_str)
            n = len(filters)
            kernels = self._align(k_str, n)
            pools = self._align(p_str, n)

            # apply Conv1D blocks
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

            # ensure output time dimension matches by projecting with a 1x1 if needed (to same length)
            # but pooling may change length – okay only if all branches have identical pooling factors.
            branch_outs.append(x)

        # --- fusion ---
        if len(branch_outs) == 1:
            x = branch_outs[0]
        else:
            x = layers.Concatenate(axis=-1, name="branch_concat")(branch_outs)

        if self.fusion_mode == "attention":
            # simple self‑attention with residual connection
            attn = layers.MultiHeadAttention(
                num_heads=int(self.attention_heads),
                key_dim=x.shape[-1] // int(self.attention_heads),
                name="fusion_attn",
            )(x, x)
            x = layers.Add()([x, attn])
            x = layers.LayerNormalization()(x)

        elif self.fusion_mode == "bigru":
            x = layers.Bidirectional(
                layers.GRU(int(self.gru_units), return_sequences=True),
                name="fusion_bigru",
            )(x)

        elif self.fusion_mode == "none":
            pass
        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")

        # --- global pooling & classification head ---
        x = layers.GlobalAveragePooling1D(name="global_pool")(x)

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
        # determine branch splits from column names
        if hasattr(X, "columns"):
            self.branch_order_, self.branch_indices_ = self._get_branch_indices(X.columns)
        else:
            # fallback: treat all features as one "all" branch
            self.branch_order_ = ["all"]
            self.branch_indices_ = {"all": list(range(X.shape[1]))}

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
        _, seq_ids = self._pad(X)
        y_seq = self._collapse_y(seq_ids, y)
        preds = self.predict(X)
        return f1_score(y_seq.to_numpy(), preds, average="macro")

# =============================================================================
# Orientation correction + training-time augmentation extensions
# =============================================================================

class SensorOrientationCorrector(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        handedness_lookup: Optional[dict] = None,
        handedness_col: str = "handedness",
        subject_col: str = "subject",
        correct_handedness: bool = True,
        left_handed_value: int = 0,
        correct_upside_down: bool = True,
        upside_down_subjects: Optional[list[str]] = None,
        rot_cols: Optional[list[str]] = None,
        acc_cols: Optional[list[str]] = None,
    ) -> None:
        self.handedness_lookup = handedness_lookup
        self.handedness_col = handedness_col
        self.subject_col = subject_col
        self.correct_handedness = correct_handedness
        self.left_handed_value = left_handed_value
        self.correct_upside_down = correct_upside_down
        self.upside_down_subjects = upside_down_subjects
        self.rot_cols = rot_cols
        self.acc_cols = acc_cols

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "SensorOrientationCorrector":
        self.rot_cols_ = self.rot_cols or ["rot_w", "rot_x", "rot_y", "rot_z"]
        self.acc_cols_ = self.acc_cols or ["acc_x", "acc_y", "acc_z"]
        self.upside_down_subjects_ = self.upside_down_subjects or ["SUBJ_019262", "SUBJ_045235"]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        if self.correct_handedness:
            if self.handedness_col in df.columns:
                handedness = df[self.handedness_col]
            elif self.handedness_lookup is not None and self.subject_col in df.columns:
                handedness = df[self.subject_col].map(self.handedness_lookup)
            else:
                handedness = pd.Series(False, index=df.index)

            left_mask = handedness.eq(self.left_handed_value)

            if left_mask.any() and "acc_x" in df.columns:
                df.loc[left_mask, "acc_x"] *= -1.0

            if left_mask.any() and set(self.rot_cols_).issubset(df.columns):
                q = df.loc[left_mask, self.rot_cols_].to_numpy(dtype=float)
                q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
                norm = np.linalg.norm(q, axis=1, keepdims=True)
                bad = (~np.isfinite(norm.squeeze())) | (norm.squeeze() == 0.0)
                q[bad] = np.array([1.0, 0.0, 0.0, 0.0])
                norm = np.linalg.norm(q, axis=1, keepdims=True)
                q = q / norm

                w = q[:, 0]
                x = q[:, 1]
                y = q[:, 2]
                z = q[:, 3]

                roll = np.arctan2(
                    2.0 * (w * x + y * z),
                    1.0 - 2.0 * (x * x + y * y),
                )
                pitch = np.arcsin(
                    np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
                )
                yaw = np.arctan2(
                    2.0 * (w * z + x * y),
                    1.0 - 2.0 * (y * y + z * z),
                )

                pitch *= -1.0
                yaw *= -1.0

                cr = np.cos(roll / 2.0)
                sr = np.sin(roll / 2.0)
                cp = np.cos(pitch / 2.0)
                sp = np.sin(pitch / 2.0)
                cy = np.cos(yaw / 2.0)
                sy = np.sin(yaw / 2.0)

                q_fixed = np.column_stack([
                    cr * cp * cy + sr * sp * sy,
                    sr * cp * cy - cr * sp * sy,
                    cr * sp * cy + sr * cp * sy,
                    cr * cp * sy - sr * sp * cy,
                ])
                norm = np.linalg.norm(q_fixed, axis=1, keepdims=True)
                bad = (~np.isfinite(norm.squeeze())) | (norm.squeeze() == 0.0)
                q_fixed[bad] = np.array([1.0, 0.0, 0.0, 0.0])
                norm = np.linalg.norm(q_fixed, axis=1, keepdims=True)
                q_fixed = q_fixed / norm

                df.loc[left_mask, self.rot_cols_] = q_fixed

            if left_mask.any() and {"ang_vel_x", "ang_vel_y", "ang_vel_z"}.issubset(df.columns):
                df.loc[left_mask, ["ang_vel_y", "ang_vel_z"]] *= -1.0

        if self.correct_upside_down and self.subject_col in df.columns:
            upside_down_mask = df[self.subject_col].isin(self.upside_down_subjects_)

            if upside_down_mask.any() and set(self.acc_cols_).issubset(df.columns):
                df.loc[upside_down_mask, self.acc_cols_] *= -1.0

            if upside_down_mask.any() and set(self.rot_cols_).issubset(df.columns):
                df.loc[upside_down_mask, ["rot_x", "rot_y", "rot_z"]] *= -1.0

            if upside_down_mask.any() and {"ang_vel_x", "ang_vel_y", "ang_vel_z"}.issubset(df.columns):
                df.loc[upside_down_mask, ["ang_vel_x", "ang_vel_y", "ang_vel_z"]] *= -1.0

        if self.handedness_col in df.columns:
            df = df.drop(columns=[self.handedness_col])

        return df


class KerasAugmentedMultiBranchClassifier(KerasMultiBranchClassifier):
    def __init__(
        self,
        target: str = "gesture_action",
        maxlen: int = 64,
        padding_value: float = -999.0,
        branch_config: dict | None = None,
        branch_filters: dict | None = None,
        branch_kernel_sizes: dict | None = None,
        branch_pool_sizes: dict | None = None,
        fusion_mode: str = "attention",
        attention_heads: int = 4,
        gru_units: int = 128,
        use_batch_norm: bool = True,
        spatial_dropout: float = 0.1,
        dense_units: str = "64",
        dropout: float = 0.3,
        learning_rate: float = 5e-4,
        batch_size: int = 32,
        epochs: int = 80,
        patience: int = 12,
        verbose: int = 0,
        random_state: int = 42,
        validation_fraction: float = 0.15,
        use_mixup: bool = False,
        mixup_alpha: float = 0.4,
        mixup_size: float = 1.0,
        mixup_group_col: Optional[str] = None,
        use_time_shift: bool = False,
        time_shift_max_pct: float = 0.25,
        use_time_stretch: bool = False,
        time_stretch_min_rate: float = 0.5,
        time_stretch_max_rate: float = 1.5,
        use_noise: bool = False,
        noise_std: float = 0.01,
        use_magnitude_scaling: bool = False,
        magnitude_scale_min: float = 0.9,
        magnitude_scale_max: float = 1.1,
        use_time_mask: bool = False,
        time_mask_ratio: float = 0.1,
        use_channel_dropout: bool = False,
        channel_dropout_prob: float = 0.05,
        use_modality_dropout: bool = False,
        modality_dropout_prob: float = 0.1,
        use_tof_dropout: bool = False,
        tof_dropout_prob: float = 0.1,
        use_quat_sign_flip: bool = False,
        use_cutmix: bool = False,
        cutmix_prob: float = 0.25,
        cutmix_ratio: float = 0.3,
        use_frequency_filter: bool = False,
        freq_keep_min: float = 0.1,
        freq_keep_max: float = 0.9,
    ):
        super().__init__(
            target=target,
            maxlen=maxlen,
            padding_value=padding_value,
            branch_config=branch_config,
            branch_filters=branch_filters,
            branch_kernel_sizes=branch_kernel_sizes,
            branch_pool_sizes=branch_pool_sizes,
            fusion_mode=fusion_mode,
            attention_heads=attention_heads,
            gru_units=gru_units,
            use_batch_norm=use_batch_norm,
            spatial_dropout=spatial_dropout,
            dense_units=dense_units,
            dropout=dropout,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            verbose=verbose,
            random_state=random_state,
        )
        self.validation_fraction = validation_fraction
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.mixup_size = mixup_size
        self.mixup_group_col = mixup_group_col
        self.use_time_shift = use_time_shift
        self.time_shift_max_pct = time_shift_max_pct
        self.use_time_stretch = use_time_stretch
        self.time_stretch_min_rate = time_stretch_min_rate
        self.time_stretch_max_rate = time_stretch_max_rate
        self.use_noise = use_noise
        self.noise_std = noise_std
        self.use_magnitude_scaling = use_magnitude_scaling
        self.magnitude_scale_min = magnitude_scale_min
        self.magnitude_scale_max = magnitude_scale_max
        self.use_time_mask = use_time_mask
        self.time_mask_ratio = time_mask_ratio
        self.use_channel_dropout = use_channel_dropout
        self.channel_dropout_prob = channel_dropout_prob
        self.use_modality_dropout = use_modality_dropout
        self.modality_dropout_prob = modality_dropout_prob
        self.use_tof_dropout = use_tof_dropout
        self.tof_dropout_prob = tof_dropout_prob
        self.use_quat_sign_flip = use_quat_sign_flip
        self.use_cutmix = use_cutmix
        self.cutmix_prob = cutmix_prob
        self.cutmix_ratio = cutmix_ratio
        self.use_frequency_filter = use_frequency_filter
        self.freq_keep_min = freq_keep_min
        self.freq_keep_max = freq_keep_max

    def _manual_split(self, X_pad: np.ndarray, y_enc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(int(self.random_state))
        n = X_pad.shape[0]
        idx = rng.permutation(n)
        n_val = max(1, int(float(self.validation_fraction) * n)) if n > 1 else 0
        if n_val == 0:
            return X_pad, y_enc, X_pad, y_enc
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        return X_pad[train_idx], y_enc[train_idx], X_pad[val_idx], y_enc[val_idx]

    def _feature_indices(self, prefixes: tuple[str, ...]) -> list[int]:
        if not hasattr(self, "feature_columns_"):
            return []
        return [i for i, c in enumerate(self.feature_columns_) if str(c).startswith(prefixes)]

    def _augment_X_only(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        X_aug = X.copy().astype(np.float32)
        n, t, f = X_aug.shape
        valid = X_aug != float(self.padding_value)

        if self.use_time_shift:
            max_shift = int(round(float(self.time_shift_max_pct) * t))
            if max_shift > 0:
                for i in range(n):
                    shift = int(rng.integers(-max_shift, max_shift + 1))
                    if shift > 0:
                        X_aug[i, shift:, :] = X_aug[i, :-shift, :]
                        X_aug[i, :shift, :] = self.padding_value
                    elif shift < 0:
                        s = abs(shift)
                        X_aug[i, :-s, :] = X_aug[i, s:, :]
                        X_aug[i, -s:, :] = self.padding_value

        if self.use_time_stretch:
            stretched = np.full_like(X_aug, float(self.padding_value))
            base_x = np.arange(t, dtype=np.float32)
            for i in range(n):
                rate = float(rng.uniform(float(self.time_stretch_min_rate), float(self.time_stretch_max_rate)))
                source_x = np.clip(base_x * rate, 0, t - 1)
                for j in range(f):
                    stretched[i, :, j] = np.interp(source_x, base_x, X_aug[i, :, j]).astype(np.float32)
            X_aug = stretched

        if self.use_magnitude_scaling:
            scale = rng.uniform(float(self.magnitude_scale_min), float(self.magnitude_scale_max), size=(n, 1, 1)).astype(np.float32)
            X_aug = np.where(X_aug == float(self.padding_value), X_aug, X_aug * scale)

        if self.use_noise:
            noise = rng.normal(0.0, float(self.noise_std), size=X_aug.shape).astype(np.float32)
            X_aug = np.where(X_aug == float(self.padding_value), X_aug, X_aug + noise)

        if self.use_time_mask:
            n_mask = max(1, int(round(float(self.time_mask_ratio) * t)))
            for i in range(n):
                start = int(rng.integers(0, max(1, t - n_mask + 1)))
                X_aug[i, start:start + n_mask, :] = 0.0

        if self.use_channel_dropout:
            drop = rng.random((n, 1, f)) < float(self.channel_dropout_prob)
            X_aug = np.where(drop & valid, 0.0, X_aug)

        if self.use_tof_dropout:
            idxs = self._feature_indices(("tof_",))
            if idxs:
                drop = rng.random((n, 1, len(idxs))) < float(self.tof_dropout_prob)
                X_aug[:, :, idxs] = np.where(drop & valid[:, :, idxs], 0.0, X_aug[:, :, idxs])

        if self.use_modality_dropout:
            modality_prefixes = [("acc_", "lin_acc_"), ("rot_", "delta_rot_", "ang_vel_", "rot6d_"), ("tof_",), ("thm_",)]
            for prefixes in modality_prefixes:
                idxs = self._feature_indices(prefixes)
                if idxs:
                    drop_seq = rng.random(n) < float(self.modality_dropout_prob)
                    X_aug[drop_seq, :, :][:, :, idxs] = 0.0
                    for row in np.flatnonzero(drop_seq):
                        X_aug[row, :, idxs] = 0.0

        if self.use_frequency_filter:
            keep_min = float(np.clip(self.freq_keep_min, 0.0, 1.0))
            keep_max = float(np.clip(self.freq_keep_max, keep_min, 1.0))
            fft = np.fft.rfft(np.where(X_aug == float(self.padding_value), 0.0, X_aug), axis=1)
            n_freq = fft.shape[1]
            low = int(round(keep_min * n_freq))
            high = max(low + 1, int(round(keep_max * n_freq)))
            mask = np.zeros(n_freq, dtype=bool)
            mask[low:high] = True
            fft[:, ~mask, :] = 0.0
            X_aug = np.fft.irfft(fft, n=t, axis=1).astype(np.float32)

        return X_aug.astype(np.float32)

    def _apply_quat_sign_flip(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idxs = self._feature_indices(("rot_w_norm", "rot_x_norm", "rot_y_norm", "rot_z_norm"))
        if not self.use_quat_sign_flip or not idxs:
            return X, y
        X_flip = X.copy()
        X_flip[:, :, idxs] *= -1.0
        return np.concatenate([X, X_flip], axis=0), np.concatenate([y, y], axis=0)

    def _apply_cutmix(self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        if not self.use_cutmix or X.shape[0] < 2:
            return X, y
        n, t, _ = X.shape
        X_new = X.copy()
        classes = np.unique(y)
        for cls in classes:
            idx = np.flatnonzero(y == cls)
            if len(idx) < 2:
                continue
            for i in idx:
                if rng.random() > float(self.cutmix_prob):
                    continue
                j = int(rng.choice(idx))
                length = max(1, int(round(float(self.cutmix_ratio) * t)))
                start = int(rng.integers(0, max(1, t - length + 1)))
                X_new[i, start:start + length, :] = X[j, start:start + length, :]
        return X_new.astype(np.float32), y

    def _apply_mixup(self, X: np.ndarray, y: np.ndarray, n_classes: int, groups: Optional[pd.Series], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        y_onehot = keras.utils.to_categorical(y, num_classes=n_classes).astype(np.float32)
        if not self.use_mixup or X.shape[0] < 2:
            return X.astype(np.float32), y_onehot

        n = X.shape[0]
        n_mix = max(1, int(round(float(self.mixup_size) * n)))
        idx_a = rng.integers(0, n, size=n_mix)
        idx_b = rng.integers(0, n, size=n_mix)

        if groups is not None:
            g = pd.Series(groups).reset_index(drop=True)
            for m, ia in enumerate(idx_a):
                candidates = np.flatnonzero(g.to_numpy() == g.iloc[int(ia)])
                if len(candidates) > 0:
                    idx_b[m] = int(rng.choice(candidates))

        lam = rng.beta(float(self.mixup_alpha), float(self.mixup_alpha), size=n_mix).astype(np.float32)
        lam_x = lam.reshape(-1, 1, 1)
        lam_y = lam.reshape(-1, 1)
        X_mix = lam_x * X[idx_a] + (1.0 - lam_x) * X[idx_b]
        y_mix = lam_y * y_onehot[idx_a] + (1.0 - lam_y) * y_onehot[idx_b]
        return np.concatenate([X, X_mix], axis=0).astype(np.float32), np.concatenate([y_onehot, y_mix], axis=0).astype(np.float32)

    def fit(self, X, y):
        if hasattr(X, "columns"):
            self.feature_columns_ = list(X.columns)
            self.branch_order_, self.branch_indices_ = self._get_branch_indices(X.columns)
        else:
            self.feature_columns_ = [str(i) for i in range(X.shape[1])] if hasattr(X, "shape") else ["0"]
            self.branch_order_ = ["all"]
            self.branch_indices_ = {"all": list(range(X.shape[1])) if hasattr(X, "shape") else [0]}

        X_pad, seq_ids = self._pad(X)
        y_seq = self._collapse_y(seq_ids, y)

        mixup_groups = None
        if self.mixup_group_col is not None and isinstance(y, pd.DataFrame) and self.mixup_group_col in y.columns:
            group_map = y.drop_duplicates("sequence_id").set_index("sequence_id")[self.mixup_group_col]
            mixup_groups = seq_ids.map(group_map).reset_index(drop=True)

        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y_seq)
        self.classes_ = self.le_.classes_
        n_classes = len(self.classes_)

        X_train, y_train, X_val, y_val = self._manual_split(X_pad, y_enc)

        if mixup_groups is not None:
            train_seq_ids = seq_ids.iloc[:len(y_enc)].reset_index(drop=True)
            # approximate group split order after _manual_split is not preserved; keep no grouping if uncertain
            mixup_groups_train = None
        else:
            mixup_groups_train = None

        rng = np.random.default_rng(int(self.random_state))
        X_train = self._augment_X_only(X_train, rng)
        X_train, y_train = self._apply_cutmix(X_train, y_train, rng)
        X_train, y_train = self._apply_quat_sign_flip(X_train, y_train)

        self.model_ = self._build(n_features=X_pad.shape[2], n_classes=n_classes)

        if self.use_mixup:
            X_train_final, y_train_final = self._apply_mixup(X_train, y_train, n_classes, mixup_groups_train, rng)
            y_val_final = keras.utils.to_categorical(y_val, num_classes=n_classes).astype(np.float32)
            self.model_.compile(
                optimizer=keras.optimizers.Adam(learning_rate=float(self.learning_rate)),
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )
        else:
            X_train_final = X_train
            y_train_final = y_train
            y_val_final = y_val

        self.history_ = self.model_.fit(
            X_train_final,
            y_train_final,
            batch_size=int(self.batch_size),
            epochs=int(self.epochs),
            validation_data=(X_val, y_val_final),
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
