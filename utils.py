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

        if not set(lin_cols).issubset(X.columns):
            raise ValueError("use_linear_acc_magnitude=True needs linear_acc_mode='baseline'.")

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

        sample = self.imu_extractor_.transform(cleaned)

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
        check_is_fitted(self, ["cleaner_", "imu_extractor_", "feature_names_in_"])

        cleaned = self.cleaner_.transform(X)
        out = self.imu_extractor_.transform(cleaned)

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