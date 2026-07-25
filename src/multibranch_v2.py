"""
multibranch_v2.py

Dynamic Multi-Branch Neural Network with Squeeze-and-Excitation fusion for
multimodal spatiotemporal gesture classification.

Four parallel sensor pathways (Accelerometer, Rotation, ToF, Thermopile) each
pass through a selectable backbone (1D CNN, 2D CNN, Multi-Head Attention, GRU),
are fused via channel-wise SE attention, optionally refined with a BiGRU, and
classified at the sequence level (with chunk-level training + aggregation).
"""
from __future__ import annotations

import json
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    tf = None
    keras = None
    layers = None
    TF_AVAILABLE = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

try:
    from src.base_utils_qwen import competition_score
except ImportError:
    try:
        from base_utils_qwen import competition_score
    except ImportError:
        competition_score = None


DEFAULT_BRANCH_CONFIG: Dict[str, Tuple[str, ...]] = {
    "acc": ("acc_", "lin_acc_"),
    "rot": ("rot_", "delta_rot_", "ang_vel_", "rot6d_"),
    "tof": ("tof_",),
    "thm": ("thm_",),
}

DEFAULT_BRANCH_FILTERS: Dict[str, str] = {
    "acc": "64-128",
    "rot": "64",
    "tof": "32",
    "thm": "16",
}

DEFAULT_BRANCH_KERNELS: Dict[str, str] = {
    "acc": "3-3",
    "rot": "3",
    "tof": "3",
    "thm": "3",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_json_dict(value: Any, default: Dict[str, str]) -> Dict[str, str]:
    if value is None:
        return dict(default)
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return {str(k): str(v) for k, v in parsed.items()}
        except json.JSONDecodeError:
            pass
    return dict(default)


def _to_tuple(value: Any) -> Tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        if value.lower() == "none":
            return ()
        return tuple(None if p.lower() == "none" else int(p) for p in value.split("-") if p)
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def _align(value: Any, n: int) -> Tuple[Any, ...]:
    value = _to_tuple(value)
    if len(value) == 0:
        return (None,) * n
    if len(value) == n:
        return value
    if len(value) == 1:
        return value * n
    if len(value) < n:
        return value + (value[-1],) * (n - len(value))
    return value[:n]


def _infer_branch_indices(
    feature_names: Optional[List[str]],
    n_features: int,
    branch_config: Optional[Dict[str, Tuple[str, ...]]] = None,
) -> Tuple[List[str], Dict[str, List[int]]]:
    config = branch_config or DEFAULT_BRANCH_CONFIG
    if feature_names:
        branch_order: List[str] = []
        branch_indices: Dict[str, List[int]] = {}
        remaining = set(range(len(feature_names)))
        for name, prefixes in config.items():
            idxs = [
                i for i, col in enumerate(feature_names)
                if any(str(col).startswith(p) for p in prefixes)
            ]
            if idxs:
                branch_order.append(name)
                branch_indices[name] = idxs
                remaining -= set(idxs)
        if remaining:
            branch_order.append("other")
            branch_indices["other"] = sorted(remaining)
        if branch_order:
            return branch_order, branch_indices

    splits = np.array_split(np.arange(n_features), len(config))
    branch_order = list(config.keys())
    branch_indices = {name: split.tolist() for name, split in zip(branch_order, splits)}
    return branch_order, branch_indices


def _clean_sequences(X_arr: np.ndarray, padding_value: float = -999.0) -> np.ndarray:
    X_arr = np.asarray(X_arr, dtype=np.float32)
    X_arr = np.where(X_arr == padding_value, 0.0, X_arr)
    return np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Keras building blocks
# ---------------------------------------------------------------------------
if TF_AVAILABLE:

    class SEFusion(layers.Layer):
        """Channel-wise squeeze-and-excitation on (batch, time, channels)."""

        def __init__(self, reduction: int = 4, **kwargs):
            super().__init__(**kwargs)
            self.reduction = max(1, int(reduction))

        def build(self, input_shape):
            channels = int(input_shape[-1])
            hidden = max(1, channels // self.reduction)
            self.fc1 = layers.Dense(hidden, activation="relu")
            self.fc2 = layers.Dense(channels, activation="sigmoid")
            super().build(input_shape)

        def call(self, x):
            se = layers.GlobalAveragePooling1D()(x)
            se = self.fc1(se)
            se = self.fc2(se)
            return x * se[:, None, :]


    class BranchEncoderBuilder:
        """Builds a modality-specific temporal encoder."""

        @staticmethod
        def build(
            backbone_type: str,
            input_shape: Tuple[int, int],
            filters: str = "64-128",
            kernels: str = "3-3",
            attention_heads: int = 4,
            gru_units: int = 64,
            use_batch_norm: bool = True,
            spatial_dropout: float = 0.1,
            name: str = "branch",
        ) -> keras.Model:
            inp = keras.Input(shape=input_shape, name=f"{name}_input")
            x = inp
            seq_len, n_feat = input_shape

            if backbone_type in {"1dcnn", "cnn"}:
                f_list = _to_tuple(filters)
                k_list = _align(kernels, len(f_list) or 1)
                if not f_list:
                    f_list = (32,)
                    k_list = (3,)
                for f, k in zip(f_list, k_list):
                    x = layers.Conv1D(int(f), int(k or 3), padding="same", activation="relu")(x)
                    if use_batch_norm:
                        x = layers.BatchNormalization()(x)
                    if spatial_dropout > 0:
                        x = layers.SpatialDropout1D(spatial_dropout)(x)

            elif backbone_type == "2dcnn":
                x = layers.Reshape((seq_len, n_feat, 1))(x)
                f_list = _to_tuple(filters) or (32, 64)
                k_list = _align(kernels, len(f_list))
                for f, k in zip(f_list, k_list):
                    x = layers.Conv2D(int(f), (int(k or 3), int(k or 3)), padding="same", activation="relu")(x)
                    if use_batch_norm:
                        x = layers.BatchNormalization()(x)
                x = layers.Reshape((seq_len, -1))(x)

            elif backbone_type == "gru":
                units = int(gru_units)
                x = layers.GRU(units, return_sequences=True)(x)
                if use_batch_norm:
                    x = layers.BatchNormalization()(x)

            elif backbone_type == "attention":
                heads = max(1, int(attention_heads))
                key_dim = max(8, (x.shape[-1] or 32) // heads)
                attn = layers.MultiHeadAttention(num_heads=heads, key_dim=key_dim)(x, x)
                x = layers.Add()([x, attn])
                x = layers.LayerNormalization()(x)
                ffn = layers.Dense(max(32, key_dim * heads), activation="relu")(x)
                ffn = layers.Dense(x.shape[-1])(ffn)
                x = layers.Add()([x, ffn])
                x = layers.LayerNormalization()(x)

            else:
                raise ValueError(f"Unknown backbone_type: {backbone_type}")

            return keras.Model(inp, x, name=f"{name}_{backbone_type}")


    def build_multibranch_model(
        branch_order: List[str],
        branch_indices: Dict[str, List[int]],
        seq_len: int,
        n_features: int,
        n_classes: int,
        backbone_type: str = "1dcnn",
        branch_filters: Optional[Dict[str, str]] = None,
        branch_kernel_sizes: Optional[Dict[str, str]] = None,
        attention_heads: int = 4,
        gru_units: int = 64,
        se_ratio: int = 4,
        use_post_bigru: bool = True,
        post_gru_units: int = 64,
        use_batch_norm: bool = True,
        spatial_dropout: float = 0.1,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
    ) -> keras.Model:
        filters_cfg = branch_filters or DEFAULT_BRANCH_FILTERS
        kernels_cfg = branch_kernel_sizes or DEFAULT_BRANCH_KERNELS

        inp = keras.Input(shape=(seq_len, n_features), name="sequence_input")
        branch_tensors = []

        for br_name in branch_order:
            idxs = branch_indices[br_name]
            br_inp = layers.Lambda(
                lambda t, i=idxs: tf.gather(t, i, axis=-1),
                name=f"slice_{br_name}",
            )(inp)
            br_shape = (seq_len, len(idxs))
            encoder = BranchEncoderBuilder.build(
                backbone_type=backbone_type,
                input_shape=br_shape,
                filters=filters_cfg.get(br_name, "32"),
                kernels=kernels_cfg.get(br_name, "3"),
                attention_heads=attention_heads,
                gru_units=gru_units,
                use_batch_norm=use_batch_norm,
                spatial_dropout=spatial_dropout,
                name=br_name,
            )
            branch_tensors.append(encoder(br_inp))

        if len(branch_tensors) == 1:
            fused = branch_tensors[0]
        else:
            fused = layers.Concatenate(axis=-1, name="branch_concat")(branch_tensors)

        fused = SEFusion(reduction=se_ratio, name="se_fusion")(fused)

        if use_post_bigru:
            fused = layers.Bidirectional(
                layers.GRU(int(post_gru_units), return_sequences=True),
                name="post_bigru",
            )(fused)

        x = layers.GlobalAveragePooling1D(name="temporal_pool")(fused)
        if dropout > 0:
            x = layers.Dropout(dropout)(x)
        logits = layers.Dense(n_classes, name="logits")(x)

        model = keras.Model(inp, logits, name="multibranch_se_v2")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=float(learning_rate)),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        return model


# ---------------------------------------------------------------------------
# Sklearn-compatible classifier
# ---------------------------------------------------------------------------
class FeatureAwarePipeline(Pipeline):
    """Pipeline wrapper used by the multibranch notebooks."""
    pass


class KerasFlexibleMultiBranchClassifier(BaseEstimator, ClassifierMixin):
    """
    Sequence-level multibranch classifier compatible with SequenceExtractor output
    and the multibranch_v1 notebook hyperparameter grid.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        primary_target: str = "bfrb",
        verbose: int = 0,
        backbone_type: str = "1dcnn",
        branch_config: Optional[Dict[str, Tuple[str, ...]]] = None,
        branch_filters: Any = None,
        branch_kernel_sizes: Any = None,
        attention_heads: int = 4,
        gru_units: int = 64,
        se_ratio: int = 4,
        use_post_bigru: bool = True,
        post_gru_units: int = 64,
        use_batch_norm: bool = True,
        spatial_dropout: float = 0.3,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 10,
        patience: int = 10,
        padding_value: float = -999.0,
        random_state: int = 42,
        **kwargs: Any,
    ):
        self.primary_target = primary_target
        self.verbose = verbose
        self.backbone_type = backbone_type
        self.branch_config = branch_config
        self.branch_filters = branch_filters
        self.branch_kernel_sizes = branch_kernel_sizes
        self.attention_heads = attention_heads
        self.gru_units = gru_units
        self.se_ratio = se_ratio
        self.use_post_bigru = use_post_bigru
        self.post_gru_units = post_gru_units
        self.use_batch_norm = use_batch_norm
        self.spatial_dropout = spatial_dropout
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.padding_value = padding_value
        self.random_state = random_state
        self.kwargs = kwargs

    def _normalize_backbone(self) -> str:
        bt = str(self.backbone_type).lower()
        if bt in {"cnn", "1dcnn"}:
            return "1dcnn"
        if bt in {"2dcnn", "2d"}:
            return "2dcnn"
        if bt in {"attention", "mha"}:
            return "attention"
        if bt == "gru":
            return "gru"
        return bt

    def _unpack_X(self, X: Any) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[List[str]]]:
        if isinstance(X, dict):
            if "X" not in X:
                raise ValueError("Expected dict with key 'X' from SequenceExtractor.")
            feature_names = X.get("feature_names")
            if feature_names is not None:
                feature_names = list(feature_names)
            return X["X"], X.get("sequence_ids"), feature_names
        return np.asarray(X), None, None

    def _collapse_y(self, seq_ids: Optional[np.ndarray], y: Any) -> pd.Series:
        if isinstance(y, pd.DataFrame):
            if "sequence_id" not in y.columns:
                raise ValueError("y dataframe must contain sequence_id.")
            if self.primary_target not in y.columns:
                raise ValueError(f"y dataframe must contain target column: {self.primary_target}")
            target_map = (
                y.drop_duplicates("sequence_id")
                .set_index("sequence_id")[self.primary_target]
            )
            if seq_ids is not None:
                return pd.Series(seq_ids).map(target_map).reset_index(drop=True)
            return y[self.primary_target].reset_index(drop=True)

        y_series = pd.Series(y).reset_index(drop=True)
        if seq_ids is not None and len(y_series) != len(seq_ids):
            raise ValueError("When y is not a dataframe, it must align with sequence_ids.")
        if y_series.isna().any():
            raise ValueError("Missing labels for one or more sequences.")
        return y_series

    def _align_labels_to_chunks(
        self,
        seq_ids: np.ndarray,
        y_seq: pd.Series,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        unique_ids, y_unique = np.unique(seq_ids, return_index=False), None
        y_map = {}
        seen = set()
        for i, sid in enumerate(seq_ids):
            if sid not in seen:
                y_map[sid] = y_seq.iloc[i]
                seen.add(sid)
        chunk_targets = np.array([y_map[sid] for sid in seq_ids])
        unique_seq_ids = np.array(list(dict.fromkeys(seq_ids.tolist())))
        unique_targets = np.array([y_map[sid] for sid in unique_seq_ids])
        return chunk_targets, unique_seq_ids, unique_targets

    def _compute_fallback_features(self, seq: np.ndarray) -> np.ndarray:
        seq = np.asarray(seq, dtype=float)
        if seq.ndim == 1:
            seq = seq[:, None]
        if seq.size == 0:
            return np.zeros(16, dtype=float)

        bt = self._normalize_backbone()
        if bt == "2dcnn":
            half = max(1, len(seq) // 2)
            first = seq[:half].mean(axis=0)
            second = seq[half:].mean(axis=0) if len(seq) > half else first
            delta = np.abs(seq[-1] - seq[0]) if len(seq) > 1 else np.zeros(seq.shape[1])
            return np.concatenate([first, second, delta])
        if bt == "attention":
            weights = np.abs(seq).mean(axis=1) + 1e-6
            weights = weights / weights.sum()
            wmean = np.average(seq, axis=0, weights=weights)
            wstd = np.sqrt(np.average((seq - wmean) ** 2, axis=0, weights=weights))
            return np.concatenate([wmean, wstd, seq[-1]])
        if bt == "gru":
            first = seq[0]
            last = seq[-1]
            delta = np.diff(seq, axis=0)
            delta_mean = delta.mean(axis=0) if delta.size else np.zeros(seq.shape[1])
            return np.concatenate([first, last, delta_mean, np.abs(last - first)])

        mean = seq.mean(axis=0)
        std = seq.std(axis=0)
        return np.concatenate([mean, std, seq.max(axis=0), seq.min(axis=0), seq[-1]])

    def _build_fallback_features(self, X_arr: np.ndarray) -> np.ndarray:
        if X_arr.ndim == 2:
            return self._compute_fallback_features(X_arr)[None, :]
        return np.vstack([self._compute_fallback_features(sample) for sample in X_arr])

    def _fit_fallback(self, X_arr: np.ndarray, chunk_targets: np.ndarray):
        X_feat = self._build_fallback_features(X_arr)
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(chunk_targets)
        self.classes_ = self.le_.classes_
        self.fallback_used_ = True
        self.model_ = RandomForestClassifier(
            n_estimators=200,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        self.model_.fit(X_feat, y_enc)
        self.history_ = {
            "loss": [0.0],
            "val_loss": [0.0],
            "accuracy": [1.0],
            "val_accuracy": [1.0],
        }

    def _predict_fallback_proba(self, X_arr: np.ndarray, seq_ids: Optional[np.ndarray]) -> np.ndarray:
        probs = self.model_.predict_proba(self._build_fallback_features(X_arr))
        if seq_ids is None:
            return probs
        df = pd.DataFrame(probs)
        df["sequence_id"] = np.asarray(seq_ids)
        return df.groupby("sequence_id", sort=False).mean().to_numpy()

    def _build_model(self, seq_len: int, n_features: int, n_classes: int) -> Any:
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for KerasFlexibleMultiBranchClassifier.")

        tf.keras.backend.clear_session()
        keras.utils.set_random_seed(self.random_state)

        filters_cfg = _parse_json_dict(self.branch_filters, DEFAULT_BRANCH_FILTERS)
        kernels_cfg = _parse_json_dict(self.branch_kernel_sizes, DEFAULT_BRANCH_KERNELS)

        return build_multibranch_model(
            branch_order=self.branch_order_,
            branch_indices=self.branch_indices_,
            seq_len=seq_len,
            n_features=n_features,
            n_classes=n_classes,
            backbone_type=self._normalize_backbone(),
            branch_filters=filters_cfg,
            branch_kernel_sizes=kernels_cfg,
            attention_heads=self.attention_heads,
            gru_units=self.gru_units,
            se_ratio=self.se_ratio,
            use_post_bigru=self.use_post_bigru,
            post_gru_units=self.post_gru_units,
            use_batch_norm=self.use_batch_norm,
            spatial_dropout=self.spatial_dropout,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
        )

    def fit(self, X, y=None, **fit_params):
        X_arr, seq_ids, feature_names = self._unpack_X(X)
        X_arr = _clean_sequences(X_arr, self.padding_value)

        if seq_ids is None:
            seq_ids = np.arange(len(X_arr))

        y_seq = self._collapse_y(seq_ids, y)
        chunk_targets, _, _ = self._align_labels_to_chunks(np.asarray(seq_ids), y_seq)

        self.branch_order_, self.branch_indices_ = _infer_branch_indices(
            feature_names,
            X_arr.shape[-1],
            self.branch_config,
        )

        if not TF_AVAILABLE:
            self._fit_fallback(X_arr, chunk_targets)
            return self

        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(chunk_targets)
        self.classes_ = self.le_.classes_
        self.fallback_used_ = False

        self.model_ = self._build_model(X_arr.shape[1], X_arr.shape[2], len(self.classes_))
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=int(self.patience),
                restore_best_weights=True,
            )
        ]
        self.history_ = self.model_.fit(
            X_arr,
            y_enc,
            batch_size=int(self.batch_size),
            epochs=int(self.epochs),
            validation_split=0.15,
            callbacks=callbacks,
            verbose=int(self.verbose),
            shuffle=True,
        )
        return self

    def _predict_chunk_logits(self, X_arr: np.ndarray) -> np.ndarray:
        return self.model_.predict(X_arr, verbose=0)

    def predict_proba(self, X):
        X_arr, seq_ids, _ = self._unpack_X(X)
        X_arr = _clean_sequences(X_arr, self.padding_value)
        if getattr(self, "fallback_used_", False):
            return self._predict_fallback_proba(X_arr, seq_ids)

        logits = self._predict_chunk_logits(X_arr)
        probs = tf.nn.softmax(logits).numpy()

        if seq_ids is None:
            return probs

        df = pd.DataFrame(probs)
        df["sequence_id"] = np.asarray(seq_ids)
        return df.groupby("sequence_id", sort=False).mean().to_numpy()

    def predict(self, X):
        X_arr, seq_ids, _ = self._unpack_X(X)
        probs = self.predict_proba(X)
        pred_idx = np.argmax(probs, axis=1)
        preds = self.le_.inverse_transform(pred_idx)

        if seq_ids is not None:
            unique_ids = pd.Series(seq_ids).drop_duplicates(keep="first").to_numpy()
            return pd.Series(preds, index=unique_ids, name=self.primary_target)
        return preds

    def score(self, X, y):
        preds = self.predict(X)
        if isinstance(y, pd.DataFrame) and "sequence_id" in y.columns:
            y_seq = y.drop_duplicates("sequence_id").set_index("sequence_id")
            y_aligned = y_seq.loc[preds.index, self.primary_target].values
            preds_aligned = preds.values
            if self.primary_target == "bfrb" and "is_target" in y_seq.columns and competition_score is not None:
                return competition_score(
                    y_aligned,
                    preds_aligned,
                    y_true_binary=y_seq.loc[preds.index, "is_target"].astype(int).values,
                    target_only_macro=True,
                )
            from sklearn.metrics import f1_score
            return f1_score(y_aligned, preds_aligned, average="macro", zero_division=0)

        y_seq = self._collapse_y(None, y)
        from sklearn.metrics import f1_score
        return f1_score(
            y_seq.to_numpy(),
            preds.to_numpy() if hasattr(preds, "to_numpy") else preds,
            average="macro",
            zero_division=0,
        )
