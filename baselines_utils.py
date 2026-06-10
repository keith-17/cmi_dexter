"""
baselines_utils.py
Sklearn pipelines for baseline models: tabular (Dummy, RF, MiniRocket) and temporal (1D CNN).

Feature extraction reuses Honeycomb components from base_utils_qwen.
Competition scoring lives in base_utils_qwen.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.linear_model import (
    LogisticRegression,
    PassiveAggressiveClassifier,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.validation import check_is_fitted
from sktime.transformations.panel.rocket import MiniRocket

from base_utils_qwen import (
    HoneycombBase,
    IMUExtractor,
    MotionFilter,
    RotationExtractor,
    SequenceExtractor,
    SignalCleaner,
    ThermoExtractor,
    TOFExtractor,
    competition_score,
    competition_scorer,
    evaluate_holdout,
    make_competition_scorer,
)

HAS_TF = None  # lazy: resolved on first CNN use


def _ensure_tf():
    global HAS_TF, tf, keras, layers, regularizers
    if HAS_TF is not None:
        return HAS_TF
    try:
        import tensorflow as _tf
        from tensorflow import keras as _keras
        from tensorflow.keras import layers as _layers, regularizers as _regularizers

        tf = _tf
        keras = _keras
        layers = _layers
        regularizers = _regularizers
        HAS_TF = True
    except ImportError:
        HAS_TF = False
    return HAS_TF

# Re-export for notebook convenience
__all__ = [
    "TabularAugmentor",
    "HoneycombTabularExtractor",
    "TabularSequenceExtractor",
    "SequenceTensorExtractor",
    "SequencePadder",
    "SequenceLevelClassifier",
    "BaseRocketClassifier",
    "RidgeRocketClassifier",
    "LogisticRocketClassifier",
    "SGDRocketClassifier",
    "KerasTemporalCNNClassifier",
    "build_feature_extractor",
    "build_classifier",
    "make_baseline_pipeline",
    "competition_score",
    "competition_scorer",
    "make_competition_scorer",
    "evaluate_holdout",
]

META_COLS = {
    "sequence_id",
    "subject",
    "sequence_counter",
    "sequence_type",
    "gesture",
    "is_target",
    "bfrb",
    "handedness",
    "orientation",
    "behavior",
    "phase",
    "adult_child",
    "age",
    "sex",
    "height_cm",
    "shoulder_to_wrist_cm",
    "elbow_to_wrist_cm",
    "gesture_action",
    "gesture_position",
}


# ==============================================================================
# DATA AUGMENTATION
# ==============================================================================
class TabularAugmentor(BaseEstimator, TransformerMixin):
    """Lightweight augmentation on tabular sequence features (train-time only via pipeline)."""

    def __init__(
        self,
        use_gaussian_noise: bool = False,
        noise_std: float = 0.01,
        use_scaling: bool = False,
        scale_min: float = 0.9,
        scale_max: float = 1.1,
        random_state: int = 42,
    ):
        self.use_gaussian_noise = use_gaussian_noise
        self.noise_std = noise_std
        self.use_scaling = use_scaling
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rng = np.random.default_rng(self.random_state)
        out = X.copy() if isinstance(X, pd.DataFrame) else np.array(X, copy=True)

        if self.use_gaussian_noise:
            noise = rng.normal(0, self.noise_std, out.shape)
            out = out + noise

        if self.use_scaling:
            scale = rng.uniform(self.scale_min, self.scale_max, (out.shape[0], 1))
            out = out * scale

        return out


# ==============================================================================
# TABULAR FEATURE EXTRACTION
# ==============================================================================
class HoneycombTabularExtractor(HoneycombBase):
    """
    Multi-stream Honeycomb extraction aggregated to one row per sequence.
    Uses IMU / rotation / TOF / thermo pipelines then mean/std/max/min stats.
    """

    def __init__(
        self,
        acc_modes: str = "raw|velocity|jerk",
        rotation_modes: str = "quaternion|angular_velocity",
        tof_modes: str = "sensor_stats",
        thm_modes: str = "centered_diff",
        motion_filter_mode=None,
        sampling_rate: int = 20,
        window_size: int = 7,
        clip_value: Optional[float] = None,
        interp_mode: str = "linear",
        agg_funcs: Tuple[str, ...] = ("mean", "std", "max", "min"),
        sequence_col: str = "sequence_id",
        counter_col: str = "sequence_counter",
    ):
        self.acc_modes = acc_modes
        self.rotation_modes = rotation_modes
        self.tof_modes = tof_modes
        self.thm_modes = thm_modes
        self.motion_filter_mode = motion_filter_mode
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.clip_value = clip_value
        self.interp_mode = interp_mode
        self.agg_funcs = agg_funcs
        self.sequence_col = sequence_col
        self.counter_col = counter_col

        self.cleaner = SignalCleaner(
            sampling_rate=sampling_rate,
            clip_value=clip_value,
            interp_mode=interp_mode,
            sequence_col=sequence_col,
            counter_col=counter_col,
            window_size=window_size,
        )
        self.motion_filter = MotionFilter(
            motion_filter_mode=motion_filter_mode, sequence_col=sequence_col
        )
        self.imu = IMUExtractor(acc_modes=acc_modes, window_size=window_size, sequence_col=sequence_col)
        self.rotation = RotationExtractor(rotation_modes=rotation_modes, sequence_col=sequence_col)
        self.tof = TOFExtractor(tof_modes=tof_modes, sequence_col=sequence_col)
        self.thermo = ThermoExtractor(thm_modes=thm_modes, sequence_col=sequence_col)

    def fit(self, X: pd.DataFrame, y=None):
        cleaned = self.cleaner.fit_transform(X)
        filtered = self.motion_filter.fit_transform(cleaned)
        for comp in (self.imu, self.rotation, self.tof, self.thermo):
            comp.fit(filtered)
        sample = pd.concat(
            [
                self.imu.transform(filtered),
                self.rotation.transform(filtered),
                self.tof.transform(filtered),
                self.thermo.transform(filtered),
            ],
            axis=1,
        ).fillna(0.0)
        self.feature_cols_ = list(sample.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["feature_cols_"])
        cleaned = self.cleaner.transform(X)
        filtered = self.motion_filter.transform(cleaned)
        ts = pd.concat(
            [
                self.imu.transform(filtered),
                self.rotation.transform(filtered),
                self.tof.transform(filtered),
                self.thermo.transform(filtered),
            ],
            axis=1,
        ).fillna(0.0)
        ts[self.sequence_col] = X[self.sequence_col].values

        grouped = ts.groupby(self.sequence_col, sort=False)[self.feature_cols_].agg(list(self.agg_funcs))
        grouped.columns = ["_".join(c) for c in grouped.columns]
        grouped.index.name = self.sequence_col
        return grouped.fillna(0.0)


class TabularSequenceExtractor(BaseEstimator, TransformerMixin):
    """
    Simple tabular aggregation (legacy-compatible) or Honeycomb-backed extraction.
    """
    def __init__(
        self,
        mode: str = "simple",
        agg_funcs: Tuple[str, ...] = ("mean", "std", "max", "min"),
        sequence_col: str = "sequence_id",
        honeycomb_kwargs: Optional[dict] = None,
    ):
        self.mode = mode
        self.agg_funcs = agg_funcs
        self.sequence_col = sequence_col
        # CRITICAL FIX: Store exactly what was passed. Do NOT use `or {}` here!
        self.honeycomb_kwargs = honeycomb_kwargs 

    def fit(self, X, y=None):
        if self.mode == "honeycomb":
            # Handle the None case here instead of in __init__
            hc_kwargs = self.honeycomb_kwargs if self.honeycomb_kwargs is not None else {}
            
            self.inner_ = HoneycombTabularExtractor(
                agg_funcs=self.agg_funcs,
                sequence_col=self.sequence_col,
                **hc_kwargs,
            )
            self.inner_.fit(X, y)
            self.feature_cols_ = self.inner_.feature_cols_
        else:
            num_cols = [
                c
                for c in X.columns
                if c not in META_COLS and pd.api.types.is_numeric_dtype(X[c])
            ]
            self.feature_cols_ = num_cols
        return self

    def transform(self, X):
        check_is_fitted(self, ["feature_cols_"])
        if self.mode == "honeycomb":
            return self.inner_.transform(X)
            
        grouped = X.groupby(self.sequence_col, sort=False)[self.feature_cols_].agg(list(self.agg_funcs))
        grouped.columns = ["_".join(c) for c in grouped.columns]
        grouped.index.name = self.sequence_col
        return grouped.fillna(0.0)


# ==============================================================================
# TEMPORAL FEATURE EXTRACTION
# ==============================================================================
class SequenceTensorExtractor(BaseEstimator, TransformerMixin):
    """Wraps Honeycomb SequenceExtractor; returns dict for temporal classifiers / MiniRocket."""

    def __init__(self, sequence_col: str = "sequence_id", **extractor_kwargs):
        self.sequence_col = sequence_col
        self.extractor_kwargs = extractor_kwargs
        self.extractor_ = SequenceExtractor(sequence_col=sequence_col, **extractor_kwargs)

    def fit(self, X, y=None):
        self.extractor_.fit(X, y)
        self.n_features_in_ = len(self.extractor_.feature_names_in_)
        return self

    def transform(self, X) -> dict:
        check_is_fitted(self, ["n_features_in_"])
        arr = self.extractor_.transform(X)
        seq_ids = self._sequence_order(X)
        return {"X": arr, "sequence_ids": np.array(seq_ids), "lengths": np.full(len(seq_ids), arr.shape[1])}

    def _sequence_order(self, X: pd.DataFrame) -> List:
        return [sid for sid, _ in X.groupby(self.sequence_col, sort=False)]


class SequencePadder(BaseEstimator, TransformerMixin):
    """Pad per-timestep numeric columns into 3D tensors (n_seq, maxlen, n_features)."""

    def __init__(
        self,
        maxlen: int = 160,
        padding_value: float = -999.0,
        dtype=np.float32,
        sequence_col: str = "sequence_id",
        counter_col: str = "sequence_counter",
        feature_cols: Optional[List[str]] = None,
    ):
        self.maxlen = maxlen
        self.padding_value = padding_value
        self.dtype = dtype
        self.sequence_col = sequence_col
        self.counter_col = counter_col
        self.feature_cols = feature_cols

    def fit(self, X, y=None):
        if self.feature_cols is None:
            self.feature_cols_ = [
                c
                for c in X.columns
                if c not in META_COLS and pd.api.types.is_numeric_dtype(X[c])
            ]
        else:
            self.feature_cols_ = list(self.feature_cols)
        self.n_features_in_ = len(self.feature_cols_)
        return self

    def transform(self, X) -> dict:
        check_is_fitted(self, ["feature_cols_", "n_features_in_"])
        seq_ids = []
        lengths = []
        tensors = []

        for seq_id, grp in X.groupby(self.sequence_col, sort=False):
            if self.counter_col in grp.columns:
                grp = grp.sort_values(self.counter_col)
            arr = grp[self.feature_cols_].to_numpy(dtype=self.dtype)
            seq_len = len(arr)
            if seq_len >= self.maxlen:
                padded = arr[: self.maxlen]
            else:
                pad = np.full(
                    (self.maxlen - seq_len, arr.shape[1]),
                    self.padding_value,
                    dtype=self.dtype,
                )
                padded = np.vstack([arr, pad])
            tensors.append(padded)
            seq_ids.append(seq_id)
            lengths.append(seq_len)

        return {
            "X": np.stack(tensors, axis=0),
            "sequence_ids": np.array(seq_ids),
            "lengths": np.array(lengths),
        }


# ==============================================================================
# SEQUENCE-LEVEL CLASSIFIER WRAPPER
# ==============================================================================
class SequenceLevelClassifier(BaseEstimator, ClassifierMixin):
    """Wraps sklearn / keras classifiers for sequence-level targets and dict inputs."""

    def __init__(self, base_estimator=None, target_col: str = "bfrb"):
        self.base_estimator = base_estimator
        self.target_col = target_col

    def _resolve_inputs(self, X) -> Tuple[Any, np.ndarray]:
        if isinstance(X, dict):
            return X["X"], np.asarray(X["sequence_ids"])
        if isinstance(X, pd.DataFrame):
            return X, X.index.to_numpy()
        return X, np.arange(len(X))

    def _collapse_y(self, seq_ids: np.ndarray, y) -> pd.Series:
        if isinstance(y, pd.DataFrame):
            y_map = y.drop_duplicates("sequence_id").set_index("sequence_id")[self.target_col]
            return pd.Series(seq_ids).map(y_map)
        return pd.Series(y).reset_index(drop=True)

    def _filter_valid(self, X_data, y_seq: pd.Series):
        valid_mask = y_seq.notna().to_numpy()
        if isinstance(X_data, pd.DataFrame):
            X_data = X_data.values   # Convert to numpy array to avoid index issues
        X_train = X_data[valid_mask]
        y_out = y_seq.loc[valid_mask].values
        return X_train, y_out

    def fit(self, X, y):
        self.estimator_ = (
            clone(self.base_estimator)
            if self.base_estimator is not None
            else DummyClassifier(strategy="most_frequent")
        )
        X_data, seq_ids = self._resolve_inputs(X)
        y_seq = self._collapse_y(seq_ids, y)
        X_train, y_train = self._filter_valid(X_data, y_seq)

        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y_train)
        self.classes_ = self.le_.classes_
        self.estimator_.fit(X_train, y_enc)
        return self

    def predict(self, X):
        check_is_fitted(self, ["estimator_", "le_"])
        X_data, _ = self._resolve_inputs(X)
        preds_enc = self.estimator_.predict(X_data)
        return self.le_.inverse_transform(preds_enc)

    def score(self, X, y):
        X_data, seq_ids = self._resolve_inputs(X)
        y_seq = self._collapse_y(seq_ids, y)
        _, y_test = self._filter_valid(X_data, y_seq)
        preds = self.predict(X)
        if isinstance(X, dict):
            valid_mask = pd.Series(seq_ids).map(
                y.drop_duplicates("sequence_id").set_index("sequence_id")[self.target_col]
            ).notna().to_numpy()
            preds = preds[valid_mask]
        elif isinstance(X, pd.DataFrame):
            valid_mask = y_seq.notna().to_numpy()
            preds = preds[valid_mask] if len(preds) == len(valid_mask) else preds
        return competition_score(y_test, preds)


# ==============================================================================
# MINIROCKET CLASSIFIERS (with regularisation knobs)
# ==============================================================================
class BaseRocketClassifier(ClassifierMixin, BaseEstimator, ABC):
    _estimator_type = "classifier"

    def __init__(
        self,
        num_kernels: int = 1000,
        random_state: int = 42,
        feature_selection_percentile: Optional[float] = None,
    ):
        self.num_kernels = num_kernels
        self.random_state = random_state
        self.feature_selection_percentile = feature_selection_percentile
        self.rocket = None
        self.scaler = None
        self.selector = None
        self.classifier = None

    def _extract_array(self, X):
        if isinstance(X, dict):
            return X["X"]
        return X

    def _extract_rocket_features(self, X, fit: bool = False):
        X_arr = self._extract_array(X)
        if X_arr.ndim == 2:
            return X_arr

        X_rocket = np.transpose(X_arr, (0, 2, 1))
        if fit or self.rocket is None:
            self.rocket = MiniRocket(num_kernels=self.num_kernels, random_state=self.random_state)
            X_transform = self.rocket.fit_transform(X_rocket)
            self.scaler = StandardScaler()
            return self.scaler.fit_transform(X_transform)

        X_transform = self.rocket.transform(X_rocket)
        return self.scaler.transform(X_transform)

    @abstractmethod
    def _get_classifier(self):
        raise NotImplementedError

    def fit(self, X, y):
        X_scaled = self._extract_rocket_features(X, fit=True)
        if self.feature_selection_percentile is not None:
            self.selector = SelectPercentile(
                score_func=f_classif, percentile=self.feature_selection_percentile
            )
            X_scaled = self.selector.fit_transform(X_scaled, y)
        else:
            self.selector = None
        self.classifier = self._get_classifier()
        self.classifier.fit(X_scaled, y)
        return self

    def predict(self, X):
        X_scaled = self._extract_rocket_features(X, fit=False)
        if self.selector is not None:
            X_scaled = self.selector.transform(X_scaled)
        return self.classifier.predict(X_scaled)

    def get_params(self, deep=True):
        return {
            "num_kernels": self.num_kernels,
            "random_state": self.random_state,
            "feature_selection_percentile": self.feature_selection_percentile,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        self.rocket = None
        self.scaler = None
        self.selector = None
        self.classifier = None
        return self


class RidgeRocketClassifier(BaseRocketClassifier):
    """MiniRocket + RidgeClassifier (L2 regularisation via alpha)."""

    def __init__(
        self,
        num_kernels: int = 1000,
        random_state: int = 42,
        feature_selection_percentile: Optional[float] = None,
        alpha: float = 1.0,
        class_weight: Optional[str] = "balanced",
    ):
        super().__init__(num_kernels, random_state, feature_selection_percentile)
        self.alpha = alpha
        self.class_weight = class_weight

    def _get_classifier(self):
        return RidgeClassifier(
            alpha=self.alpha,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )

    def get_params(self, deep=True):
        params = super().get_params(deep)
        params.update({"alpha": self.alpha, "class_weight": self.class_weight})
        return params


class LogisticRocketClassifier(BaseRocketClassifier):
    """MiniRocket + LogisticRegression (L2 via C=1/alpha)."""

    def __init__(
        self,
        num_kernels: int = 1000,
        random_state: int = 42,
        feature_selection_percentile: Optional[float] = None,
        C: float = 1.0,
        penalty: str = "l2",
        solver: str = "lbfgs",
        max_iter: int = 1000,
        class_weight: Optional[str] = "balanced",
    ):
        super().__init__(num_kernels, random_state, feature_selection_percentile)
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.class_weight = class_weight

    def _get_classifier(self):
        return LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            solver=self.solver,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=-1,
        )

    def get_params(self, deep=True):
        params = super().get_params(deep)
        params.update(
            {
                "C": self.C,
                "penalty": self.penalty,
                "solver": self.solver,
                "max_iter": self.max_iter,
                "class_weight": self.class_weight,
            }
        )
        return params


class SGDRocketClassifier(BaseRocketClassifier):
    """MiniRocket + SGDClassifier (elastic net / L1 / L2)."""

    def __init__(
        self,
        num_kernels: int = 1000,
        random_state: int = 42,
        feature_selection_percentile: Optional[float] = None,
        alpha: float = 1e-4,
        penalty: str = "l2",
        loss: str = "log_loss",
        max_iter: int = 1000,
        class_weight: Optional[str] = "balanced",
    ):
        super().__init__(num_kernels, random_state, feature_selection_percentile)
        self.alpha = alpha
        self.penalty = penalty
        self.loss = loss
        self.max_iter = max_iter
        self.class_weight = class_weight

    def _get_classifier(self):
        return SGDClassifier(
            loss=self.loss,
            penalty=self.penalty,
            alpha=self.alpha,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=-1,
        )

    def get_params(self, deep=True):
        params = super().get_params(deep)
        params.update(
            {
                "alpha": self.alpha,
                "penalty": self.penalty,
                "loss": self.loss,
                "max_iter": self.max_iter,
                "class_weight": self.class_weight,
            }
        )
        return params


# ==============================================================================
# TEMPORAL 1D CNN CLASSIFIER
# ==============================================================================
class KerasTemporalCNNClassifier(ClassifierMixin, BaseEstimator):
    """1D CNN on padded temporal tensors (dict['X'] or ndarray)."""

    _estimator_type = "classifier"

    def __init__(
        self,
        filters: str = "64-128",
        kernels: str = "3-3",
        pools: str = "none",
        dropout: float = 0.3,
        spatial_dropout: float = 0.1,
        l2_reg: float = 1e-4,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 50,
        patience: int = 10,
        validation_split: float = 0.15,
        class_weight_mode: str = "balanced",
        mask_value: float = -999.0,
        verbose: int = 0,
        random_state: int = 42,
    ):
        self.filters = filters
        self.kernels = kernels
        self.pools = pools
        self.dropout = dropout
        self.spatial_dropout = spatial_dropout
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.validation_split = validation_split
        self.class_weight_mode = class_weight_mode
        self.mask_value = mask_value
        self.verbose = verbose
        self.random_state = random_state

    @staticmethod
    def _parse_tuple(val: str) -> Tuple[int, ...]:
        if not val or val.lower() == "none":
            return ()
        return tuple(int(p) for p in val.split("-") if p.lower() != "none")

    def _extract_array(self, X):
        return X["X"] if isinstance(X, dict) else X

    def _build_model(self, input_shape, n_classes):
        if not _ensure_tf():
            raise ImportError("tensorflow is required for KerasTemporalCNNClassifier")

        l2 = regularizers.l2(self.l2_reg)
        inputs = keras.Input(shape=input_shape)
        x = layers.Lambda(
            lambda t, mv=self.mask_value: t * tf.cast(
                tf.reduce_any(t != mv, axis=-1, keepdims=True), tf.float32
            )
        )(inputs)

        f_list = self._parse_tuple(self.filters)
        k_list = self._parse_tuple(self.kernels)
        p_list = self._parse_tuple(self.pools)
        for i, (f, k) in enumerate(zip(f_list, k_list)):
            x = layers.Conv1D(f, k, padding="same", activation="relu", kernel_regularizer=l2)(x)
            x = layers.BatchNormalization()(x)
            if self.spatial_dropout > 0:
                x = layers.SpatialDropout1D(rate=self.spatial_dropout)(x)
            if i < len(p_list) and p_list[i] > 1:
                x = layers.MaxPooling1D(p_list[i])(x)

        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(128, activation="relu", kernel_regularizer=l2)(x)
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
        if not _ensure_tf():
            raise ImportError("tensorflow is required for KerasTemporalCNNClassifier")

        X_arr = self._extract_array(X)
        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(pd.Series(y))
        self.classes_ = self.label_encoder_.classes_

        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)

        self.model_ = self._build_model((X_arr.shape[1], X_arr.shape[2]), len(self.classes_))
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=self.patience, restore_best_weights=True
            )
        ]

        class_weight = None
        if self.class_weight_mode == "balanced":
            weights = compute_class_weight("balanced", classes=np.unique(y_enc), y=y_enc)
            class_weight = dict(enumerate(weights))

        self.history_ = self.model_.fit(
            X_arr,
            y_enc,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=self.verbose,
            shuffle=True,
        )
        return self

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.label_encoder_.inverse_transform(np.argmax(proba, axis=1))

    def predict_proba(self, X):
        X_arr = self._extract_array(X)
        return self.model_.predict(X_arr, verbose=0)


# ==============================================================================
# FACTORY HELPERS
# ==============================================================================
def build_feature_extractor(
    feature_mode: str = "tabular_simple",
    **kwargs,
) -> BaseEstimator:
    """
    feature_mode:
      - tabular_simple: raw numeric agg stats
      - tabular_honeycomb: multi-domain Honeycomb agg stats
      - temporal_honeycomb: Honeycomb SequenceExtractor -> dict
      - temporal_raw: raw column padding -> dict
    """
    honeycomb_keys = {
        "acc_modes", "rotation_modes", "tof_modes", "thm_modes",
        "motion_filter_mode", "sampling_rate", "window_size", "clip_value",
        "interp_mode", "use_dead_reckoning",
        "dead_reckoning_detrend", "kalman_process_noise", "kalman_measurement_noise",
        "compute_dt", "smooth_alpha", "counter_col", "sequence_col",
    }
    tabular_keys = {"agg_funcs", "sequence_col", "mode", "honeycomb_kwargs"}
    if feature_mode == "tabular_simple":
        simple_kw = {k: v for k, v in kwargs.items() if k in tabular_keys or k not in honeycomb_keys}
        return TabularSequenceExtractor(mode="simple", **simple_kw)
    if feature_mode == "tabular_honeycomb":
        hc_kw = {k: v for k, v in kwargs.items() if k in honeycomb_keys}
        other = {k: v for k, v in kwargs.items() if k in tabular_keys}
        return TabularSequenceExtractor(mode="honeycomb", honeycomb_kwargs=hc_kw, **other)
    if feature_mode == "temporal_honeycomb":
        return SequenceTensorExtractor(**kwargs)
    if feature_mode == "temporal_raw":
        raw_kw = {k: v for k, v in kwargs.items() if k not in honeycomb_keys or k in {"maxlen", "padding_value", "sequence_col", "counter_col", "feature_cols"}}
        return SequencePadder(**raw_kw)
    raise ValueError(f"Unknown feature_mode: {feature_mode}")


def build_classifier(
    classifier_name: str = "rf",
    random_state: int = 42,
    **kwargs,
) -> BaseEstimator:
    """Build a base estimator for SequenceLevelClassifier."""
    name = classifier_name.lower()
    if name == "dummy":
        return DummyClassifier(strategy=kwargs.get("strategy", "most_frequent"))
    if name in ("rf", "random_forest"):
        return RandomForestClassifier(random_state=random_state, n_jobs=-1, **kwargs)
    if name in ("ridge_rocket", "rocket", "minirocket"):
        return RidgeRocketClassifier(random_state=random_state, **kwargs)
    if name == "logistic_rocket":
        return LogisticRocketClassifier(random_state=random_state, **kwargs)
    if name == "sgd_rocket":
        return SGDRocketClassifier(random_state=random_state, **kwargs)
    if name in ("cnn", "temporal_cnn", "1dcnn"):
        return KerasTemporalCNNClassifier(random_state=random_state, **kwargs)
    raise ValueError(f"Unknown classifier_name: {classifier_name}")


def make_baseline_pipeline(
    feature_mode: str = "tabular_simple",
    classifier_name: str = "rf",
    target_col: str = "bfrb",
    augment: bool = False,
    augment_kwargs: Optional[dict] = None,
    feature_kwargs: Optional[dict] = None,
    classifier_kwargs: Optional[dict] = None,
    random_state: int = 42,
):
    """Assemble sklearn Pipeline with optional tabular augmentation."""
    from sklearn.pipeline import Pipeline

    feature_kwargs = feature_kwargs or {}
    classifier_kwargs = classifier_kwargs or {}
    steps = [("extractor", build_feature_extractor(feature_mode, **feature_kwargs))]

    if augment and feature_mode.startswith("tabular"):
        steps.append(
            (
                "augment",
                TabularAugmentor(random_state=random_state, **(augment_kwargs or {})),
            )
        )

    steps.append(
        (
            "classifier",
            SequenceLevelClassifier(
                base_estimator=build_classifier(
                    classifier_name, random_state=random_state, **classifier_kwargs
                ),
                target_col=target_col,
            ),
        )
    )
    return Pipeline(steps)
