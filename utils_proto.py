"""
utils_proto.py
Dynamic Multi-Head Prototypical Network with Multi-Branch Fusion,
Aggressive Augmentations, and Skopt-Compatible Dict Parameters.
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
from typing import Tuple, Dict, Any, List, Optional
import json
import warnings
warnings.filterwarnings("ignore")

def prepare_multitask_param_space(param_space: Dict[str, Any], search_mode: str) -> Dict[str, Any]:
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


class DynamicMultiHeadPrototypicalNetwork(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"

    def __init__(
        self,
        gesture_target: str = "gesture_action",
        orientation_target: str = "orientation",
        phase_target: str = "phase",
        n_way: int = 9,
        n_support: int = 5,
        n_query: int = 15,
        branch_filters: Optional[Dict[str, str]] = None,
        branch_kernel_sizes: Optional[Dict[str, str]] = None,
        branch_pool_sizes: Optional[Dict[str, str]] = None,
        fusion_mode: str = "attention",
        attention_heads: int = 4,
        gru_units: int = 170,
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
        validation_fraction: float = 0.15,
        uncertainty_weighting: bool = False,
        fixed_gesture_weight: float = 1.0,
        fixed_orientation_weight: float = 0.5,
        fixed_phase_weight: float = 0.5,
        use_mixup: bool = False, mixup_alpha: float = 0.4, mixup_size: float = 1.0, mixup_prob: float = 0.5,
        use_time_shift: bool = False, max_shift_pct: float = 0.15,
        use_time_stretch: bool = False, time_stretch_min_rate: float = 0.8, time_stretch_max_rate: float = 1.2,
        use_gaussian_noise: bool = False, noise_std: float = 0.01,
        use_magnitude_scaling: bool = False, magnitude_scale_min: float = 0.75, magnitude_scale_max: float = 1.5,
        use_time_mask: bool = False, time_mask_ratio: float = 0.1,
        use_channel_dropout: bool = False, channel_dropout_prob: float = 0.1,
        use_modality_dropout: bool = False, drop_acc_prob: float = 0.2, drop_rot_prob: float = 0.2, drop_tof_prob: float = 0.2, drop_thm_prob: float = 0.2,
        verbose: int = 0,
        random_state: int = 42,
    ):
        self.gesture_target = gesture_target
        self.orientation_target = orientation_target
        self.phase_target = phase_target
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.branch_filters = branch_filters or {"acc": "64-128", "rot": "32-64", "tof": "32", "thm": "16"}
        self.branch_kernel_sizes = branch_kernel_sizes or {"acc": "3-3", "rot": "3-3", "tof": "3", "thm": "3"}
        self.branch_pool_sizes = branch_pool_sizes or {"acc": "none-none", "rot": "none-none", "tof": "none", "thm": "none"}
        self.fusion_mode = fusion_mode
        self.attention_heads = attention_heads
        self.gru_units = gru_units
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
        self.validation_fraction = validation_fraction
        self.uncertainty_weighting = uncertainty_weighting
        self.fixed_gesture_weight = fixed_gesture_weight
        self.fixed_orientation_weight = fixed_orientation_weight
        self.fixed_phase_weight = fixed_phase_weight
        self.use_mixup, self.mixup_alpha, self.mixup_size, self.mixup_prob = use_mixup, mixup_alpha, mixup_size, mixup_prob
        self.use_time_shift, self.max_shift_pct = use_time_shift, max_shift_pct
        self.use_time_stretch, self.time_stretch_min_rate, self.time_stretch_max_rate = use_time_stretch, time_stretch_min_rate, time_stretch_max_rate
        self.use_gaussian_noise, self.noise_std = use_gaussian_noise, noise_std
        self.use_magnitude_scaling, self.magnitude_scale_min, self.magnitude_scale_max = use_magnitude_scaling, magnitude_scale_min, magnitude_scale_max
        self.use_time_mask, self.time_mask_ratio = use_time_mask, time_mask_ratio
        self.use_channel_dropout, self.channel_dropout_prob = use_channel_dropout, channel_dropout_prob
        self.use_modality_dropout, self.drop_acc_prob, self.drop_rot_prob, self.drop_tof_prob, self.drop_thm_prob = use_modality_dropout, drop_acc_prob, drop_rot_prob, drop_tof_prob, drop_thm_prob
        self.verbose = verbose
        self.random_state = random_state

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def set_params(self, **params: Any) -> DynamicMultiHeadPrototypicalNetwork:
        for k, v in params.items():
            if k in {"branch_filters", "branch_kernel_sizes", "branch_pool_sizes"} and isinstance(v, str):
                setattr(self, k, json.loads(v))
            else:
                setattr(self, k, v)
        return self

    def _to_tuple(self, val: Any) -> Tuple[Any, ...]:
        if isinstance(val, str):
            return () if val == "none" else tuple(None if p == "none" else int(p) for p in val.split("-"))
        return (val,) if not isinstance(val, tuple) else val

    def _build_branch(self, inp: tf.Tensor, filters: str, kernels: str, pools: str, name: str) -> tf.Tensor:
        x = inp
        f_list = self._to_tuple(filters)
        k_list = self._to_tuple(kernels)
        p_list = self._to_tuple(pools)
        n = max(1, len(f_list))
        k_list = (k_list * n)[:n] if len(k_list) < n else k_list[:n]
        p_list = (p_list * n)[:n] if len(p_list) < n else p_list[:n]

        for f, k, p in zip(f_list, k_list, p_list):
            x = layers.Conv1D(int(f), int(k), padding="same", activation="relu", name=f"{name}_conv")(x)
            if self.use_batch_norm:
                x = layers.BatchNormalization(name=f"{name}_bn")(x)
            if self.spatial_dropout > 0:
                x = layers.SpatialDropout1D(self.spatial_dropout)(x)
            if p is not None:
                x = layers.MaxPooling1D(int(p), name=f"{name}_pool")(x)
        return x

    def _build_model(self, input_shape: Tuple[int, int], modality_slices: Dict[str, slice]) -> keras.Model:
        tf.keras.backend.clear_session()
        inp = keras.Input(shape=input_shape, name="main_input")
        branches = []
        for mod, sl in modality_slices.items():
            if sl.start < sl.stop:
                branch_inp = layers.Lambda(lambda x, s=sl: x[:, :, s], name=f"{mod}_slice")(inp)
                branches.append(self._build_branch(
                    branch_inp,
                    self.branch_filters.get(mod, "32"),
                    self.branch_kernel_sizes.get(mod, "3"),
                    self.branch_pool_sizes.get(mod, "none"),
                    mod
                ))
        
        if not branches:
            branches.append(inp)
            
        if len(branches) > 1:
            if self.fusion_mode == "attention":
                fused = layers.Concatenate()(branches)
                fused = layers.MultiHeadAttention(self.attention_heads, key_dim=32)(fused, fused)
            elif self.fusion_mode == "bigru":
                fused = layers.Concatenate()(branches)
                fused = layers.Bidirectional(layers.GRU(self.gru_units, return_sequences=True))(fused)
            else:
                fused = layers.Concatenate()(branches)
        else:
            fused = branches[0]

        x = layers.GlobalAveragePooling1D()(fused)
        for u in self._to_tuple(self.dense_units):
            x = layers.Dense(int(u), activation="relu")(x)
            if self.dropout > 0:
                x = layers.Dropout(self.dropout)(x)

        emb = layers.Dense(self.embedding_dim, name="embedding")(x)
        emb = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(emb)
        return keras.Model(inp, emb, name="proto_encoder")

    def _augment_batch(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.use_gaussian_noise and np.random.rand() < 0.5:
            X += np.random.normal(0, self.noise_std, X.shape).astype(np.float32)
        if self.use_magnitude_scaling and np.random.rand() < 0.5:
            scale = np.random.uniform(self.magnitude_scale_min, self.magnitude_scale_max)
            X *= scale
        if self.use_time_shift and np.random.rand() < 0.5:
            shift = int(X.shape[1] * np.random.uniform(-self.max_shift_pct, self.max_shift_pct))
            X = np.roll(X, shift, axis=1)
        if self.use_mixup and np.random.rand() < self.mixup_prob:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            idx = np.random.permutation(len(X))
            X = lam * X + (1 - lam) * X[idx]
            y = lam * y + (1 - lam) * y[idx]  # Soft labels for mixup
        return X, y

    def _sample_episode(self, X: np.ndarray, y_enc: np.ndarray, class_indices: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        available = [c for c, idx in class_indices.items() if len(idx) >= self.n_support + self.n_query]
        if len(available) < self.n_way:
            available = list(class_indices.keys())
        episode_classes = np.random.choice(available, min(self.n_way, len(available)), replace=False)
        sup_x, sup_y, qry_x, qry_y = [], [], [], []
        for local_lbl, cls in enumerate(episode_classes):
            pool = class_indices[cls].copy()
            np.random.shuffle(pool)
            need = self.n_support + self.n_query
            if len(pool) < need:
                pool = np.tile(pool, (need + len(pool) - 1) // len(pool))[:need]
            sup_x.append(X[pool[:self.n_support]])
            sup_y.extend([local_lbl] * self.n_support)
            qry_x.append(X[pool[self.n_support:self.n_support + self.n_query]])
            qry_y.extend([local_lbl] * self.n_query)
        return np.concatenate(sup_x), np.array(sup_y, dtype=np.int32), np.concatenate(qry_x), np.array(qry_y, dtype=np.int32)

    @staticmethod
    def _proto_loss(sup_emb: tf.Tensor, sup_y: tf.Tensor, qry_emb: tf.Tensor, qry_y: tf.Tensor, n_way: int, dist: str) -> tf.Tensor:
        protos = tf.stack([tf.reduce_mean(tf.boolean_mask(sup_emb, tf.equal(sup_y, c)), axis=0) for c in range(n_way)])
        if dist == "euclidean":
            d = tf.reduce_sum((tf.expand_dims(qry_emb, 1) - tf.expand_dims(protos, 0)) ** 2, axis=2)
        else:
            d = 1.0 - tf.matmul(tf.math.l2_normalize(qry_emb, axis=-1), tf.math.l2_normalize(protos, axis=-1), transpose_b=True)
        log_p = tf.nn.log_softmax(-d, axis=-1)
        return -tf.reduce_mean(tf.gather(log_p, qry_y, batch_dims=1))

    def fit(self, X: np.ndarray, y: pd.DataFrame) -> DynamicMultiHeadPrototypicalNetwork:
        if not isinstance(X, np.ndarray) or X.ndim != 3:
            raise TypeError("X must be a 3D numpy array from AdvancedMultiDomainSequenceExtractor.")
        
        y_seq = y.drop_duplicates("sequence_id").sort_values("sequence_id") if "sequence_id" in y.columns else y
        target_col = self.gesture_target
        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(y_seq[target_col])
        self.classes_ = self.label_encoder_.classes_
        class_indices = {c: np.where(y_enc == c)[0] for c in range(len(self.classes_))}

        # Extract modality slices from pipeline if available
        mod_slices = {"acc": slice(0, X.shape[2])}  # Fallback
        if hasattr(self, "modality_slices_"):
            mod_slices = self.modality_slices_

        self.encoder_ = self._build_model((X.shape[1], X.shape[2]), mod_slices)
        opt = keras.optimizers.Adam(self.learning_rate)
        best_loss, patience_cnt = np.inf, 0
        steps = max(1, len(X) // self.batch_size)

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for _ in range(steps):
                sup_x, sup_y, qry_x, qry_y = self._sample_episode(X, y_enc, class_indices)
                sup_x, sup_y = self._augment_batch(sup_x, sup_y)
                qry_x, qry_y = self._augment_batch(qry_x, qry_y)
                with tf.GradientTape() as tape:
                    loss = self._proto_loss(self.encoder_(sup_x, training=True), sup_y,
                                            self.encoder_(qry_x, training=True), qry_y, len(np.unique(sup_y)), self.distance)
                opt.apply_gradients(zip(tape.gradient(loss, self.encoder_.trainable_variables), self.encoder_.trainable_variables))
                epoch_loss += loss.numpy()
            epoch_loss /= steps
            if self.verbose: print(f"Epoch {epoch+1}/{self.epochs} - loss: {epoch_loss:.4f}")
            if epoch_loss < best_loss: best_loss, patience_cnt = epoch_loss, 0
            else:
                patience_cnt += 1
                if patience_cnt >= self.patience:
                    if self.verbose: print(f"Early stopping at epoch {epoch+1}")
                    break

        all_emb = self.encoder_.predict(X, verbose=0)
        self.prototypes_ = np.array([np.mean(all_emb[y_enc == c], axis=0) for c in range(len(self.classes_))])
        self.prototypes_ /= (np.linalg.norm(self.prototypes_, axis=1, keepdims=True) + 1e-8)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["encoder_", "label_encoder_", "prototypes_"])
        emb = self.encoder_.predict(X, verbose=0)
        if self.distance == "euclidean":
            dist = np.sum((emb[:, None, :] - self.prototypes_[None, :, :]) ** 2, axis=2)
        else:
            dist = 1.0 - np.dot(emb / (np.linalg.norm(emb, axis=1, keepdims=True)+1e-8), self.prototypes_.T)
        return self.label_encoder_.inverse_transform(np.argmin(dist, axis=1))

    def score(self, X: np.ndarray, y: pd.DataFrame) -> float:
        y_pred = self.predict(X)
        y_true = y.drop_duplicates("sequence_id").sort_values("sequence_id")[self.gesture_target].values if "sequence_id" in y.columns else y
        return f1_score(y_true, y_pred, average="macro", zero_division=0)