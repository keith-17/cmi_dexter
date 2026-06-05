"""
utils_proto.py
Corrected Prototypical Network – works with pipeline (3D numpy array) or DataFrame.
Compatible with base_utils.py SequenceExtractor.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import f1_score, make_scorer
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Custom Scorer for Pipeline
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Custom Scorer for Pipeline
# ----------------------------------------------------------------------
def make_competition_scorer(target_col='bfrb'):
    """
    Custom scorer that handles the mismatch between row-level y_true
    (from CV splitter) and sequence-level y_pred (from the model).
    """

    def _score(y_true, y_pred):
        # y_true is a row-level DataFrame. We need sequence-level labels.
        if isinstance(y_true, pd.DataFrame):
            y_true_seq = y_true.drop_duplicates(subset=['sequence_id']).sort_values('sequence_id')
            y_true_binary = y_true_seq['is_target'].astype(int).values
            y_true_gesture = y_true_seq[target_col].values
        else:
            # Fallback if y_true is somehow already sequence-level
            y_true_binary = np.array(y_true)
            y_true_gesture = np.array(y_true)

        y_pred = np.array(y_pred)

        # Binary F1 (target vs non-target)
        # The model predicts 'non_bfrb' for non-targets
        y_pred_binary = (y_pred != 'non_bfrb').astype(int)
        binary_f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

        # Gesture Macro F1 (only target sequences)
        target_mask = y_true_binary == 1
        if target_mask.sum() > 0:
            gesture_f1 = f1_score(
                y_true_gesture[target_mask],
                y_pred[target_mask],
                average='macro',
                zero_division=0
            )
        else:
            gesture_f1 = 0.0

        return (binary_f1 + gesture_f1) / 2

    return make_scorer(_score, response_method='predict')


# Default scorer for the 'bfrb' column
competition_scorer = make_competition_scorer('bfrb')

# Default scorer for the 'bfrb' column
competition_scorer = make_competition_scorer('bfrb')

# ----------------------------------------------------------------------
# Encoder builder
# ----------------------------------------------------------------------
def build_encoder(
    input_shape: Tuple[int, int],  # (maxlen, n_features)
    conv_filters: str = "64-128",
    kernel_sizes: str = "5-3",
    pool_sizes: str = "none",
    use_batch_norm: bool = True,
    spatial_dropout: float = 0.1,
    dense_units: str = "64",
    dropout: float = 0.3,
    embedding_dim: int = 128,
) -> keras.Model:
    tf.keras.backend.clear_session()
    
    def to_tuple(val):
        if isinstance(val, str):
            if val == "none":
                return ()
            return tuple(None if p == "none" else int(p) for p in val.split("-"))
        return (val,) if not isinstance(val, tuple) else val

    def align(val, n):
        t = to_tuple(val)
        if len(t) == 0:
            return (None,) * n
        if len(t) == n:
            return t
        if len(t) == 1:
            return t * n
        if len(t) < n:
            return t + (t[-1],) * (n - len(t))
        return t[:n]

    inp = keras.Input(shape=input_shape, name="input")
    x = inp

    filters_list = to_tuple(conv_filters)
    n_layers = max(1, len(filters_list))
    kernels = align(kernel_sizes, n_layers)
    pools = align(pool_sizes, n_layers)

    for f, k, p in zip(filters_list, kernels, pools):
        x = layers.Conv1D(int(f), int(k), padding="same", activation="relu")(x)
        if use_batch_norm:
            x = layers.BatchNormalization()(x)
        if spatial_dropout > 0:
            x = layers.SpatialDropout1D(spatial_dropout)(x)
        if p is not None:
            x = layers.MaxPooling1D(int(p))(x)

    x = layers.GlobalAveragePooling1D()(x)

    dense_list = to_tuple(dense_units)
    for units in dense_list:
        x = layers.Dense(int(units), activation="relu")(x)
        if dropout > 0:
            x = layers.Dropout(dropout)(x)

    embedding = layers.Dense(embedding_dim, name="embedding")(x)
    embedding = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(embedding)
    return keras.Model(inp, embedding, name="encoder")


# ----------------------------------------------------------------------
# Single‑head prototypical network
# ----------------------------------------------------------------------
class BinaryPlusGesturePrototypicalNetwork(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"
    
    def __init__(
        self,
        gesture_column: str = "bfrb",
        n_way: int = 20,
        n_support: int = 5,
        n_query: int = 15,
        backbone_type: str = "1dcnn",
        conv_filters: str = "64-128",
        kernel_sizes: str = "5-3",
        pool_sizes: str = "none",
        use_batch_norm: bool = True,
        spatial_dropout: float = 0.1,
        dense_units: str = "64",
        dropout: float = 0.3,
        embedding_dim: int = 128,
        distance: str = "euclidean",
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = 15,
        maxlen: int = 160,
        padding_value: float = -999.0,
        verbose: int = 0,
        random_state: int = 42,
    ):
        self.gesture_column = gesture_column
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.backbone_type = backbone_type
        self.conv_filters = conv_filters
        self.kernel_sizes = kernel_sizes
        self.pool_sizes = pool_sizes
        self.use_batch_norm = use_batch_norm
        self.spatial_dropout = spatial_dropout
        self.dense_units = dense_units
        self.dropout = dropout
        self.embedding_dim = embedding_dim
        self.distance = distance
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.maxlen = maxlen
        self.padding_value = padding_value
        self.verbose = verbose
        self.random_state = random_state

    def get_params(self, deep=True):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    # ------------------------------------------------------------------
    # Accept both DataFrame (row‑level) and pre‑padded numpy array
    # ------------------------------------------------------------------
    def _prepare_X(self, X):
        """Convert X to padded numpy array (n_seq, maxlen, n_feat) if needed."""
        if isinstance(X, np.ndarray) and X.ndim == 3:
            return X, None
        elif isinstance(X, pd.DataFrame):
            grouped = list(X.groupby(level=0, sort=False))
            n_seq = len(grouped)
            n_feat = X.shape[1]
            out = np.full((n_seq, self.maxlen, n_feat), self.padding_value, dtype=np.float32)
            seq_ids = []
            for i, (sid, g) in enumerate(grouped):
                arr = g.to_numpy(dtype=np.float32)
                length = min(len(arr), self.maxlen)
                out[i, :length] = arr[:length]
                seq_ids.append(sid)
            return out, pd.Series(seq_ids, name="sequence_id")
        else:
            raise TypeError("X must be a DataFrame (row‑level) or 3D numpy array (padded sequences).")

    def _prepare_y(self, y, seq_ids=None):
        """Convert y to 1D label array (one per sequence)."""
        if isinstance(y, pd.DataFrame):
            if seq_ids is not None:
                y_df = y.drop_duplicates("sequence_id").set_index("sequence_id")
                y_combined = []
                for sid in seq_ids:
                    row = y_df.loc[sid]
                    if row["is_target"]:
                        y_combined.append(row[self.gesture_column])
                    else:
                        y_combined.append("non_bfrb") # FIXED: was "non_target"
                return pd.Series(y_combined)
            else:
                if "sequence_id" in y.columns:
                    y_df = y.drop_duplicates("sequence_id").sort_values("sequence_id")
                    y_combined = []
                    for _, row in y_df.iterrows():
                        if row["is_target"]:
                            y_combined.append(row[self.gesture_column])
                        else:
                            y_combined.append("non_bfrb") # FIXED: was "non_target"
                    return pd.Series(y_combined)
                else:
                    y_combined = []
                    for _, row in y.iterrows():
                        if "is_target" in row and row["is_target"]:
                            y_combined.append(row.get(self.gesture_column, "non_bfrb"))
                        else:
                            y_combined.append("non_bfrb") # FIXED: was "non_target"
                    return pd.Series(y_combined)
        else:
            return pd.Series(y)

    # ------------------------------------------------------------------
    # Episode sampling
    # ------------------------------------------------------------------
    def _sample_episode(
        self, X: np.ndarray, y_enc: np.ndarray, class_indices: Dict[int, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        available = [c for c, idx in class_indices.items() if len(idx) >= self.n_support + self.n_query]
        if len(available) < self.n_way:
            available = list(class_indices.keys())
        episode_classes = np.random.choice(available, self.n_way, replace=False)

        support_x, support_y = [], []
        query_x, query_y = [], []

        for local_label, cls in enumerate(episode_classes):
            pool = class_indices[cls].copy()
            np.random.shuffle(pool)
            need = self.n_support + self.n_query
            if len(pool) < need:
                pool = np.tile(pool, (need + len(pool) - 1) // len(pool))[:need]

            support_idx = pool[:self.n_support]
            query_idx = pool[self.n_support:self.n_support + self.n_query]

            support_x.append(X[support_idx])
            support_y.extend([local_label] * self.n_support)
            query_x.append(X[query_idx])
            query_y.extend([local_label] * self.n_query)

        support_x = np.concatenate(support_x, axis=0)
        query_x = np.concatenate(query_x, axis=0)
        support_y = np.array(support_y, dtype=np.int32)
        query_y = np.array(query_y, dtype=np.int32)
        return support_x, support_y, query_x, query_y

    # ------------------------------------------------------------------
    # Prototypical loss
    # ------------------------------------------------------------------
    @staticmethod
    def prototypical_loss(
        support_emb: tf.Tensor, support_labels: tf.Tensor,
        query_emb: tf.Tensor, query_labels: tf.Tensor,
        n_way: int, distance: str
    ) -> tf.Tensor:
        prototypes = tf.stack([
            tf.reduce_mean(tf.boolean_mask(support_emb, tf.equal(support_labels, c)), axis=0)
            for c in range(n_way)
        ])
        if distance == "euclidean":
            dist = tf.reduce_sum((tf.expand_dims(query_emb, 1) - tf.expand_dims(prototypes, 0)) ** 2, axis=2)
        else:
            q_norm = tf.math.l2_normalize(query_emb, axis=-1)
            p_norm = tf.math.l2_normalize(prototypes, axis=-1)
            dist = 1.0 - tf.matmul(q_norm, p_norm, transpose_b=True)
        log_p = tf.nn.log_softmax(-dist, axis=-1)
        loss = -tf.reduce_mean(tf.gather(log_p, query_labels, batch_dims=1))
        return loss

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X, y):
        X_pad, seq_ids = self._prepare_X(X)
        y_series = self._prepare_y(y, seq_ids)

        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(y_series)
        self.classes_ = self.label_encoder_.classes_

        class_indices = {c: np.where(y_enc == c)[0] for c in range(len(self.classes_))}
        n_way_actual = min(self.n_way, len(self.classes_))

        # Dynamically use X_pad.shape[1] to match the sequence length output by SequenceExtractor
        input_shape = (X_pad.shape[1], X_pad.shape[2])
        self.encoder_ = build_encoder(
            input_shape,
            conv_filters=self.conv_filters,
            kernel_sizes=self.kernel_sizes,
            pool_sizes=self.pool_sizes,
            use_batch_norm=self.use_batch_norm,
            spatial_dropout=self.spatial_dropout,
            dense_units=self.dense_units,
            dropout=self.dropout,
            embedding_dim=self.embedding_dim,
        )
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)

        best_loss = np.inf
        patience_counter = 0
        steps_per_epoch = max(1, len(X_pad) // self.batch_size)

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for _ in range(steps_per_epoch):
                sup_x, sup_y, qry_x, qry_y = self._sample_episode(X_pad, y_enc, class_indices)
                with tf.GradientTape() as tape:
                    sup_emb = self.encoder_(sup_x, training=True)
                    qry_emb = self.encoder_(qry_x, training=True)
                    loss = self.prototypical_loss(sup_emb, sup_y, qry_emb, qry_y, n_way_actual, self.distance)
                grads = tape.gradient(loss, self.encoder_.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.encoder_.trainable_variables))
                epoch_loss += loss.numpy()
            epoch_loss /= steps_per_epoch
            if self.verbose:
                print(f"Epoch {epoch+1}/{self.epochs} - loss: {epoch_loss:.4f}")

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break

        all_emb = self.encoder_.predict(X_pad, verbose=0)
        self.prototypes_ = np.array([
            np.mean(all_emb[y_enc == c], axis=0) for c in range(len(self.classes_))
        ])
        norms = np.linalg.norm(self.prototypes_, axis=1, keepdims=True)
        self.prototypes_ = self.prototypes_ / (norms + 1e-8)
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def fit_support(self, X_support, y_support):
        X_pad, seq_ids = self._prepare_X(X_support)
        y_series = self._prepare_y(y_support, seq_ids)
        y_enc = self.label_encoder_.transform(y_series)
        emb = self.encoder_.predict(X_pad, verbose=0)
        self.prototypes_ = np.array([
            np.mean(emb[y_enc == c], axis=0) for c in range(len(self.classes_))
        ])
        norms = np.linalg.norm(self.prototypes_, axis=1, keepdims=True)
        self.prototypes_ = self.prototypes_ / (norms + 1e-8)
        return self

    def predict(self, X):
        check_is_fitted(self, ["encoder_", "label_encoder_", "prototypes_"])
        X_pad, _ = self._prepare_X(X)
        emb = self.encoder_.predict(X_pad, verbose=0)
        if self.distance == "euclidean":
            dist = np.sum((emb[:, np.newaxis, :] - self.prototypes_[np.newaxis, :, :]) ** 2, axis=2)
        else:
            q_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
            p_norm = self.prototypes_ / (np.linalg.norm(self.prototypes_, axis=1, keepdims=True) + 1e-8)
            dist = 1.0 - np.dot(q_norm, p_norm.T)
        pred_idx = np.argmin(dist, axis=1)
        return self.label_encoder_.inverse_transform(pred_idx)

    def predict_proba(self, X):
        check_is_fitted(self, ["encoder_", "prototypes_"])
        X_pad, _ = self._prepare_X(X)
        emb = self.encoder_.predict(X_pad, verbose=0)
        if self.distance == "euclidean":
            dist = np.sum((emb[:, np.newaxis, :] - self.prototypes_[np.newaxis, :, :]) ** 2, axis=2)
        else:
            q_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
            p_norm = self.prototypes_ / (np.linalg.norm(self.prototypes_, axis=1, keepdims=True) + 1e-8)
            dist = 1.0 - np.dot(q_norm, p_norm.T)
        neg_d = -dist
        neg_d -= neg_d.max(axis=1, keepdims=True)
        probs = np.exp(neg_d) / np.exp(neg_d).sum(axis=1, keepdims=True)
        return probs

    def score(self, X, y):
        y_pred = self.predict(X)
        if isinstance(y, pd.DataFrame):
            y_seq = y.drop_duplicates(subset=['sequence_id']).sort_values('sequence_id')
            y_true_binary = y_seq['is_target'].astype(int).values
            y_true_gesture = y_seq[self.gesture_column].values
        else:
            y_true_binary = np.array(y)
            y_true_gesture = np.array(y)

        y_pred_binary = (np.array(y_pred) != 'non_bfrb').astype(int)
        binary_f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

        target_mask = y_true_binary == 1
        if target_mask.sum() > 0:
            gesture_f1 = f1_score(
                y_true_gesture[target_mask],
                np.array(y_pred)[target_mask],
                average='macro',
                zero_division=0
            )
        else:
            gesture_f1 = 0.0

        return (binary_f1 + gesture_f1) / 2


# ----------------------------------------------------------------------
# Multi‑head version (minimal)
# ----------------------------------------------------------------------
class DynamicMultiHeadPrototypicalNetwork(BinaryPlusGesturePrototypicalNetwork):
    def __init__(self, heads=None, primary_target="gesture", **kwargs):
        super().__init__(gesture_column=primary_target, **kwargs)
        self.heads = heads or ["gesture_action", "orientation", "gesture_position"]
        self.primary_target = primary_target
        
    def predict(self, X):
        return super().predict(X)