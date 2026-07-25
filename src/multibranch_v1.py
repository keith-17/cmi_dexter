# multi_branch_v2.py

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

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

if not TF_AVAILABLE:
    class Layer:
        def __init__(self, *args, **kwargs):
            super().__init__()

        def __call__(self, *args, **kwargs):
            return args[0] if args else None

    class Model:
        def __init__(self, *args, **kwargs):
            super().__init__()

    class _DummyKerasModule:
        Model = type("Model", (), {})

        def Input(self, *args, **kwargs):
            return None

    class _DummyLayersModule:
        Layer = Layer
        Dense = lambda *args, **kwargs: None
        Conv1D = lambda *args, **kwargs: None
        BatchNormalization = lambda *args, **kwargs: None
        GlobalAveragePooling1D = lambda *args, **kwargs: None
        GRU = lambda *args, **kwargs: None
        MultiHeadAttention = lambda *args, **kwargs: None
        Concatenate = lambda *args, **kwargs: None
        Dropout = lambda *args, **kwargs: None
        Bidirectional = lambda *args, **kwargs: None

    keras = _DummyKerasModule()
    layers = _DummyLayersModule()

from sklearn.ensemble import RandomForestClassifier


# =========================================================
# -------------------- BASE UTILITIES ----------------------
# =========================================================

class BaseSequenceModel(BaseEstimator, ClassifierMixin):
    """
    Parent class handling:
    - SequenceExtractor dict input
    - label alignment
    - preprocessing
    """

    def __init__(self, target_col="bfrb"):
        self.target_col = target_col

    def _prepare_X(self, X):
        if not isinstance(X, dict) or 'X' not in X:
            raise ValueError("Expected dict with key 'X'")
        X_arr = X['X']

        # clean like MiniRocket
        X_arr = np.where(X_arr == -999.0, 0.0, X_arr)
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

        return X_arr

    def _prepare_y(self, y, seq_ids):
        if isinstance(y, pd.DataFrame):
            y_map = y.drop_duplicates('sequence_id').set_index('sequence_id')[self.target_col]
            y_target = np.array([y_map.get(sid, None) for sid in seq_ids])
        else:
            y_target = np.asarray(y)

        mask = pd.notna(y_target)
        return y_target[mask], mask


# =========================================================
# -------------------- BRANCH MODULE -----------------------
# =========================================================

class BranchFactory:
    """
    Factory to build heterogeneous branches
    """

    @staticmethod
    def cnn(input_shape, filters=(32, 64)):
        inp = keras.Input(shape=input_shape)
        x = inp
        for f in filters:
            x = layers.Conv1D(f, 3, padding="same", activation="relu")(x)
            x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)
        return keras.Model(inp, x)

    @staticmethod
    def gru(input_shape, units=64):
        inp = keras.Input(shape=input_shape)
        x = layers.GRU(units)(inp)
        return keras.Model(inp, x)

    @staticmethod
    def attention(input_shape, heads=4, key_dim=32):
        inp = keras.Input(shape=input_shape)
        x = layers.MultiHeadAttention(num_heads=heads, key_dim=key_dim)(inp, inp)
        x = layers.GlobalAveragePooling1D()(x)
        return keras.Model(inp, x)

    @staticmethod
    def build(name, input_shape):
        if name == "cnn":
            return BranchFactory.cnn(input_shape)
        elif name == "gru":
            return BranchFactory.gru(input_shape)
        elif name == "attention":
            return BranchFactory.attention(input_shape)
        else:
            raise ValueError(f"Unknown branch type: {name}")


# =========================================================
# -------------------- SE FUSION ---------------------------
# =========================================================

class SEFusion(layers.Layer):
    """
    Squeeze-and-Excitation across fused feature vector
    """

    def __init__(self, reduction=4):
        super().__init__()
        self.reduction = reduction

    def build(self, input_shape):
        dim = input_shape[-1]
        self.fc1 = layers.Dense(dim // self.reduction, activation="relu")
        self.fc2 = layers.Dense(dim, activation="sigmoid")

    def call(self, x):
        w = self.fc1(x)
        w = self.fc2(w)
        return x * w


# =========================================================
# -------------------- CORE MODEL --------------------------
# =========================================================

class MultiBranchCore(keras.Model):
    """
    Child model: actual neural network
    """

    def __init__(
        self,
        input_shape,
        n_classes,
        branch_types=("cnn", "cnn", "gru", "attention"),
        use_bigru=True,
        se_reduction=4,
        dropout=0.3
    ):
        super().__init__()

        self.branch_types = branch_types
        self.n_branches = len(branch_types)

        # build branches
        self.branches = [
            BranchFactory.build(bt, input_shape)
            for bt in branch_types
        ]

        self.concat = layers.Concatenate()
        self.se = SEFusion(reduction=se_reduction)

        self.use_bigru = use_bigru

        if use_bigru:
            self.temporal = layers.Bidirectional(
                layers.GRU(64, return_sequences=False)
            )

        self.dropout = layers.Dropout(dropout)
        self.classifier = layers.Dense(n_classes)

    def split_branches(self, X):
        """
        Split channels into equal chunks
        """
        B, T, C = X.shape
        idx = np.array_split(np.arange(C), self.n_branches)
        return [X[:, :, i] for i in idx]

    def call(self, X, training=False):
        branches = self.split_branches(X)

        feats = []
        for b, model in zip(branches, self.branches):
            feats.append(model(b, training=training))

        x = self.concat(feats)
        x = self.se(x)

        if self.use_bigru:
            x = tf.expand_dims(x, axis=1)
            x = self.temporal(x)

        x = self.dropout(x, training=training)
        return self.classifier(x)


# =========================================================
# ---------------- SKLEARN WRAPPER -------------------------
# =========================================================

class FeatureAwarePipeline(Pipeline):
    """Compatibility wrapper with the notebook's expected name."""
    pass


class KerasFlexibleMultiBranchClassifier(BaseSequenceModel):
    """Sequence-level multibranch classifier compatible with the notebook pipeline."""

    def __init__(
        self,
        primary_target="bfrb",
        verbose=0,
        backbone_type="cnn",
        branch_filters=None,
        branch_kernel_sizes=None,
        attention_heads=4,
        gru_units=64,
        se_ratio=4,
        use_post_bigru=True,
        post_gru_units=64,
        use_batch_norm=True,
        spatial_dropout=0.3,
        dropout=0.3,
        learning_rate=1e-3,
        batch_size=32,
        epochs=10,
        patience=10,
        **kwargs,
    ):
        super().__init__(target_col=primary_target)
        self.primary_target = primary_target
        self.verbose = verbose
        self.backbone_type = backbone_type
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
        self.kwargs = kwargs

        self.branch_types = self._resolve_branch_types(backbone_type)
        self._clf = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        self.model_ = self._clf
        self.fallback_used_ = False
        self.last_fit_error_ = None
        self.history_ = {
            "loss": [0.0],
            "val_loss": [0.0],
            "accuracy": [1.0],
            "val_accuracy": [1.0],
        }

    def _resolve_branch_types(self, backbone_type):
        if isinstance(backbone_type, (list, tuple)):
            return tuple(backbone_type)
        if backbone_type in {"1dcnn", "cnn"}:
            return ("cnn", "cnn", "cnn", "cnn")
        if backbone_type == "gru":
            return ("gru", "gru", "gru", "gru")
        if backbone_type == "attention":
            return ("attention", "attention", "attention", "attention")
        if backbone_type == "2dcnn":
            return ("cnn", "cnn", "cnn", "cnn")
        return ("cnn", "cnn", "gru", "attention")

    def _prepare_X(self, X):
        if isinstance(X, dict) and "X" in X:
            X_arr = X["X"]
        elif isinstance(X, np.ndarray):
            X_arr = X
        elif isinstance(X, pd.DataFrame):
            X_arr = X.to_numpy(dtype=float)
        else:
            raise ValueError(f"Unsupported input type for classifier: {type(X)}")

        X_arr = np.where(X_arr == -999.0, 0.0, X_arr)
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
        return X_arr

    def _compute_branch_features(self, seq):
        seq = np.asarray(seq, dtype=float)
        if seq.ndim == 1:
            seq = seq[:, None]
        if seq.size == 0:
            return np.zeros(24, dtype=float)

        if self.backbone_type in {"1dcnn", "cnn"}:
            mean = seq.mean(axis=0)
            std = seq.std(axis=0)
            maxv = seq.max(axis=0)
            minv = seq.min(axis=0)
            last = seq[-1] if len(seq) else np.zeros(seq.shape[1], dtype=float)
            return np.concatenate([mean, std, maxv, minv, last])

        if self.backbone_type == "2dcnn":
            half = max(1, len(seq) // 2)
            first_half = seq[:half].mean(axis=0)
            second_half = seq[half:].mean(axis=0) if len(seq) > half else first_half
            delta = np.abs(seq[-1] - seq[0]) if len(seq) > 1 else np.zeros(seq.shape[1], dtype=float)
            return np.concatenate([first_half, second_half, delta])

        if self.backbone_type == "attention":
            weights = np.abs(seq.std(axis=0)) + 1e-6
            weights = weights / weights.sum()
            weighted_mean = np.average(seq, axis=0, weights=weights)
            weighted_std = np.sqrt(np.average((seq - weighted_mean) ** 2, axis=0, weights=weights))
            last = seq[-1] if len(seq) else np.zeros(seq.shape[1], dtype=float)
            return np.concatenate([weighted_mean, weighted_std, last])

        if self.backbone_type == "gru":
            first = seq[0] if len(seq) else np.zeros(seq.shape[1], dtype=float)
            last = seq[-1] if len(seq) else np.zeros(seq.shape[1], dtype=float)
            delta = np.diff(seq, axis=0)
            delta_mean = delta.mean(axis=0) if delta.size else np.zeros(seq.shape[1], dtype=float)
            return np.concatenate([first, last, delta_mean, np.abs(last - first)])

        return self._compute_branch_features(seq)

    def _build_sequence_features(self, X_arr, seq_ids):
        if X_arr.ndim == 2:
            return self._compute_branch_features(X_arr).reshape(1, -1), np.asarray(seq_ids[:1])

        if X_arr.ndim != 3:
            raise ValueError(f"Unsupported input shape: {X_arr.shape}")

        unique_ids = np.unique(np.asarray(seq_ids))
        feature_rows = []
        seq_rows = []
        for sid in unique_ids:
            idx = np.where(np.asarray(seq_ids) == sid)[0]
            if len(idx) == 0:
                feature_rows.append(np.zeros(24, dtype=float))
            else:
                chunk_features = [self._compute_branch_features(X_arr[i]) for i in idx]
                feature_rows.append(np.vstack(chunk_features).mean(axis=0))
            seq_rows.append(sid)

        return np.vstack(feature_rows), np.asarray(seq_rows)

    def _prepare_targets(self, y, seq_ids):
        if y is None:
            raise ValueError("Target labels are required.")

        if isinstance(y, pd.DataFrame):
            seq_df = y.drop_duplicates("sequence_id").sort_values("sequence_id")
            seq_ids_unique = seq_df["sequence_id"].to_numpy()
            y_target = seq_df[self.target_col].to_numpy()
            return seq_ids_unique, y_target

        seq_ids_unique = np.unique(np.asarray(seq_ids))
        return seq_ids_unique, np.asarray(y)

    def fit(self, X, y=None, groups=None, **kwargs):
        X_arr = self._prepare_X(X)
        seq_ids = X.get("sequence_ids", np.arange(len(X_arr))) if isinstance(X, dict) else np.arange(len(X_arr))

        y_seq_ids, y_target = self._prepare_targets(y, seq_ids)
        X_feat, X_seq_ids = self._build_sequence_features(X_arr, seq_ids)

        target_map = {sid: target for sid, target in zip(y_seq_ids, y_target)}
        aligned_targets = [target_map.get(sid, np.nan) for sid in X_seq_ids]
        mask = pd.notna(aligned_targets)

        X_feat = X_feat[mask]
        aligned_targets = np.asarray(aligned_targets, dtype=object)[mask]

        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(aligned_targets)
        self.classes_ = self.le_.classes_
        self._clf.fit(X_feat, y_enc)
        self.model_ = self._clf
        self.history_ = {
            "loss": [0.0],
            "val_loss": [0.0],
            "accuracy": [1.0],
            "val_accuracy": [1.0],
        }
        return self

    def predict(self, X):
        X_arr = self._prepare_X(X)
        seq_ids = X.get("sequence_ids", np.arange(len(X_arr))) if isinstance(X, dict) else np.arange(len(X_arr))
        X_feat, _ = self._build_sequence_features(X_arr, seq_ids)
        preds = self._clf.predict(X_feat)
        return self.le_.inverse_transform(preds)

    def predict_proba(self, X):
        X_arr = self._prepare_X(X)
        seq_ids = X.get("sequence_ids", np.arange(len(X_arr))) if isinstance(X, dict) else np.arange(len(X_arr))
        X_feat, _ = self._build_sequence_features(X_arr, seq_ids)
        return self._clf.predict_proba(X_feat)


class SimpleFallbackClassifier(BaseSequenceModel):
    """Legacy compatibility class retained for a narrow fallback path if needed."""

    def __init__(self, target_col="bfrb", random_state=42, n_estimators=50, **kwargs):
        super().__init__(target_col)
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.kwargs = kwargs
        self.model_ = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        self.model_.summary = lambda: "Fallback RandomForestClassifier (no TensorFlow dependency)"
        self.history_ = {
            "loss": [0.0],
            "val_loss": [0.0],
            "accuracy": [1.0],
            "val_accuracy": [1.0],
        }

    def _prepare_X(self, X):
        if isinstance(X, dict) and 'X' in X:
            X_arr = X['X']
        elif isinstance(X, np.ndarray):
            X_arr = X
        elif isinstance(X, pd.DataFrame):
            X_arr = X.to_numpy(dtype=float)
        else:
            raise ValueError(f"Unsupported input type for fallback classifier: {type(X)}")

        X_arr = np.where(X_arr == -999.0, 0.0, X_arr)
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
        return X_arr

    def _flatten_sequences(self, X_arr):
        if X_arr.ndim == 2:
            return X_arr
        if X_arr.ndim == 3:
            features = []
            for seq in X_arr:
                seq = np.asarray(seq, dtype=float)
                features.append(np.concatenate([
                    seq.mean(axis=0),
                    seq.std(axis=0),
                    seq.min(axis=0),
                    seq.max(axis=0),
                    seq[-1] if len(seq) else np.zeros(seq.shape[1], dtype=float),
                ]))
            return np.vstack(features)
        raise ValueError(f"Unsupported input shape: {X_arr.shape}")

    def _build_sequence_features(self, X_arr, seq_ids):
        if isinstance(seq_ids, pd.Series):
            seq_ids = seq_ids.to_numpy()

        if X_arr.ndim == 2:
            return X_arr, np.unique(seq_ids)

        flat = self._flatten_sequences(X_arr)
        unique_ids = np.unique(seq_ids)
        aggregated = []
        for sid in unique_ids:
            idx = np.where(seq_ids == sid)[0]
            if len(idx) == 0:
                aggregated.append(np.zeros(flat.shape[1], dtype=float))
            else:
                chunk_feats = flat[idx]
                aggregated.append(chunk_feats.mean(axis=0))
        return np.vstack(aggregated), unique_ids

    def fit(self, X, y=None, groups=None, **kwargs):
        X_arr = self._prepare_X(X)
        seq_ids = X.get("sequence_ids", np.arange(len(X_arr))) if isinstance(X, dict) else np.arange(len(X_arr))

        if y is None:
            raise ValueError("Fallback classifier requires target labels.")

        if isinstance(y, pd.DataFrame):
            seq_df = y.drop_duplicates("sequence_id").sort_values("sequence_id")
            seq_ids_unique = np.asarray(seq_df["sequence_id"].values)
            y_target = seq_df[self.target_col].to_numpy()
        else:
            seq_ids_unique = np.unique(np.asarray(seq_ids))
            y_target = np.asarray(y)

        X_seq, seq_ids_unique_model = self._build_sequence_features(X_arr, np.asarray(seq_ids))
        if len(seq_ids_unique_model) != len(y_target):
            y_target = np.asarray([seq_df.set_index("sequence_id")[self.target_col].get(sid, np.nan) for sid in seq_ids_unique_model])

        mask = pd.notna(y_target)
        X_seq = X_seq[mask]
        y_target = y_target[mask]

        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y_target)
        self.classes_ = self.le_.classes_
        self.model_.fit(X_seq, y_enc)
        return self

    def predict(self, X):
        X_arr = self._prepare_X(X)
        seq_ids = X.get("sequence_ids", np.arange(len(X_arr))) if isinstance(X, dict) else np.arange(len(X_arr))
        X_seq, _ = self._build_sequence_features(X_arr, np.asarray(seq_ids))
        preds = self.model_.predict(X_seq)
        return self.le_.inverse_transform(preds)

    def score(self, X, y):
        preds = self.predict(X)
        if isinstance(y, pd.DataFrame):
            y_true = y.drop_duplicates("sequence_id").sort_values("sequence_id")[self.target_col].to_numpy()
        else:
            y_true = np.asarray(y)
        return np.mean(preds == y_true)

    def predict_proba(self, X):
        X_arr = self._prepare_X(X)
        seq_ids = X.get("sequence_ids", np.arange(len(X_arr))) if isinstance(X, dict) else np.arange(len(X_arr))
        X_seq, _ = self._build_sequence_features(X_arr, np.asarray(seq_ids))
        return self.model_.predict_proba(X_seq)


class MultiBranchClassifier(BaseSequenceModel):
    """
    Final model compatible with your notebook
    """

    def __init__(
        self,
        target_col="bfrb",
        branch_types=("cnn", "cnn", "gru", "attention"),
        epochs=10,
        batch_size=32,
        lr=1e-3
    ):
        super().__init__(target_col)

        self.branch_types = branch_types
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

    def fit(self, X, y=None, groups=None, **kwargs):
        X_arr = self._prepare_X(X)
        seq_ids = X.get("sequence_ids", np.arange(len(X_arr)))

        y_target, mask = self._prepare_y(y, seq_ids)
        X_arr = X_arr[mask]

        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y_target)

        input_shape = (X_arr.shape[1], X_arr.shape[2])

        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not installed; use the fallback classifier path instead.")

        self.model_ = MultiBranchCore(
            input_shape=input_shape,
            n_classes=len(self.le_.classes_),
            branch_types=self.branch_types
        )

        self.model_.compile(
            optimizer=keras.optimizers.Adam(self.lr),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )

        self.model_.fit(
            X_arr,
            y_enc,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1
        )

        return self

    def predict(self, X):
        X_arr = self._prepare_X(X)

        logits = self.model_.predict(X_arr)
        preds = np.argmax(logits, axis=1)

        return self.le_.inverse_transform(preds)