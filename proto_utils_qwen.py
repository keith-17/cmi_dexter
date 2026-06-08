# proto_utils.py
"""
proto_utils.py
Prototypical Networks with Multi-Branch Fusion, Temporal Augmentations, 
and Dynamic Backbone Selection (1D CNN, 2D CNN, Attention, LSTM).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, make_scorer
from typing import Tuple, Dict, Any, List, Optional
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Temporal Augmentations
# ---------------------------------------------------------------------------
class TemporalAugmentor:
    """Applies resource-efficient temporal augmentations to 3D tensors (batch, time, features)."""
    def __init__(
        self,
        use_mixup: bool = False, mixup_alpha: float = 0.4, mixup_prob: float = 0.5,
        use_time_shift: bool = False, max_shift_pct: float = 0.15,
        use_time_stretch: bool = False, time_stretch_min: float = 0.8, time_stretch_max: float = 1.2,
        use_gaussian_noise: bool = False, noise_std: float = 0.01,
        use_magnitude_scaling: bool = False, mag_min: float = 0.9, mag_max: float = 1.1,
        use_time_mask: bool = False, time_mask_ratio: float = 0.1,
        use_channel_dropout: bool = False, channel_drop_prob: float = 0.1,
        use_quaternion_flip: bool = False, quat_flip_prob: float = 0.5,
        use_freq_filter: bool = False, freq_keep_low: float = 0.1, freq_keep_high: float = 0.9
    ):
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob
        self.use_time_shift = use_time_shift
        self.max_shift_pct = max_shift_pct
        self.use_time_stretch = use_time_stretch
        self.time_stretch_min = time_stretch_min
        self.time_stretch_max = time_stretch_max
        self.use_gaussian_noise = use_gaussian_noise
        self.noise_std = noise_std
        self.use_magnitude_scaling = use_magnitude_scaling
        self.mag_min = mag_min
        self.mag_max = mag_max
        self.use_time_mask = use_time_mask
        self.time_mask_ratio = time_mask_ratio
        self.use_channel_dropout = use_channel_dropout
        self.channel_drop_prob = channel_drop_prob
        self.use_quaternion_flip = use_quaternion_flip
        self.quat_flip_prob = quat_flip_prob
        self.use_freq_filter = use_freq_filter
        self.freq_keep_low = freq_keep_low
        self.freq_keep_high = freq_keep_high

    def __call__(self, X: np.ndarray, y: Optional[np.ndarray] = None, padding_value: float = -999.0) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        X_aug = X.copy()
        y_aug = y.copy() if y is not None else None
        valid = X_aug != padding_value
        n, t, f = X_aug.shape

        if self.use_gaussian_noise and np.random.rand() < 0.5:
            noise = np.random.normal(0, self.noise_std, X_aug.shape).astype(np.float32)
            X_aug = np.where(valid, X_aug + noise, X_aug)

        if self.use_magnitude_scaling and np.random.rand() < 0.5:
            scale = np.random.uniform(self.mag_min, self.mag_max, (n, 1, 1)).astype(np.float32)
            X_aug = np.where(valid, X_aug * scale, X_aug)

        if self.use_time_shift and np.random.rand() < 0.5:
            max_shift = max(1, int(t * self.max_shift_pct))
            for i in range(n):
                shift = np.random.randint(-max_shift, max_shift + 1)
                X_aug[i] = np.roll(X_aug[i], shift, axis=0)

        if self.use_time_mask and np.random.rand() < 0.5:
            mask_len = max(1, int(t * self.time_mask_ratio))
            for i in range(n):
                start = np.random.randint(0, max(1, t - mask_len + 1))
                X_aug[i, start:start + mask_len, :] = padding_value

        if self.use_channel_dropout and np.random.rand() < 0.5:
            ch_mask = np.random.random((n, 1, f)) < self.channel_drop_prob
            X_aug = np.where(ch_mask & valid, padding_value, X_aug)

        if self.use_mixup and y_aug is not None and np.random.rand() < self.mixup_prob:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            idx = np.random.permutation(n)
            X_aug = lam * X_aug + (1 - lam) * X_aug[idx]
            # For MixUp, we assume soft labels or just keep hard labels for prototypical sampling
            # In episodic training, we usually just mix the embeddings or inputs.
            
        return X_aug, y_aug

# ---------------------------------------------------------------------------
# Backbone Builder (1D CNN, 2D CNN, Attention, LSTM)
# ---------------------------------------------------------------------------
class BackboneBuilder:
    @staticmethod
    def _parse_tuple(val: str) -> Tuple[int, ...]:
        if not val or val.lower() == "none": return ()
        return tuple(int(p) for p in val.split("-") if p.lower() != "none")

    @staticmethod
    def build(
        backbone_type: str,
        input_shape: Tuple[int, int],
        branch_slices: Optional[Dict[str, slice]] = None,
        filters: str = "64-128",
        kernels: str = "3-3",
        pools: str = "none",
        lstm_units: int = 128,
        attention_heads: int = 4,
        embed_dim: int = 128,
        dropout: float = 0.2
    ) -> tf.keras.Model:
        inp = layers.Input(shape=input_shape, name="main_input")
        
        # Branch processing (Honeycomb structure)
        branches = []
        if branch_slices:
            for mod, sl in branch_slices.items():
                if sl.start < sl.stop:
                    branch_inp = layers.Lambda(lambda x, s=sl: x[:, :, s], name=f"{mod}_slice")(inp)
                    branches.append(branch_inp)
        
        if not branches:
            branches.append(inp)
            
        x = layers.Concatenate()(branches) if len(branches) > 1 else branches[0]

        if backbone_type == "1dcnn":
            f_list = BackboneBuilder._parse_tuple(filters)
            k_list = BackboneBuilder._parse_tuple(kernels)
            p_list = BackboneBuilder._parse_tuple(pools)
            for i, (f, k) in enumerate(zip(f_list, k_list)):
                x = layers.Conv1D(f, k, padding="same", activation="relu")(x)
                x = layers.BatchNormalization()(x)
                if i < len(p_list) and p_list[i] > 1:
                    x = layers.MaxPooling1D(p_list[i])(x)
            x = layers.GlobalAveragePooling1D()(x)
            
        elif backbone_type == "2dcnn":
            # Treat (time, features) as (height, width) with 1 channel
            x = layers.Reshape((input_shape[0], input_shape[1], 1))(x)
            f_list = BackboneBuilder._parse_tuple(filters)
            k_list = BackboneBuilder._parse_tuple(kernels)
            for f, k in zip(f_list, k_list):
                x = layers.Conv2D(f, (k, k), padding="same", activation="relu")(x)
                x = layers.BatchNormalization()(x)
                x = layers.MaxPooling2D((2, 2))(x)
            x = layers.GlobalAveragePooling2D()(x)
            
        elif backbone_type == "lstm":
            x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(x)
            x = layers.GlobalAveragePooling1D()(x)
            
        elif backbone_type == "attention":
            attn = layers.MultiHeadAttention(num_heads=attention_heads, key_dim=32)(x, x)
            x = layers.Add()([x, attn])
            x = layers.LayerNormalization()(x)
            x = layers.GlobalAveragePooling1D()(x)
            
        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}")

        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        emb = layers.Dense(embed_dim, name="embedding")(x)
        emb = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(emb)
        
        return models.Model(inp, emb, name=f"{backbone_type}_encoder")

# ---------------------------------------------------------------------------
# Prototypical Base Class
# ---------------------------------------------------------------------------
class PrototypicalBase(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        n_way: int = 9,
        n_support: int = 5,
        n_query: int = 15,
        backbone_type: str = "1dcnn",
        filters: str = "64-128",
        kernels: str = "3-3",
        pools: str = "none",
        lstm_units: int = 128,
        attention_heads: int = 4,
        embed_dim: int = 128,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = 15,
        augmentor: Optional[TemporalAugmentor] = None,
        padding_value: float = -999.0,
        verbose: int = 0,
        random_state: int = 42,
    ):
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.backbone_type = backbone_type
        self.filters = filters
        self.kernels = kernels
        self.pools = pools
        self.lstm_units = lstm_units
        self.attention_heads = attention_heads
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.augmentor = augmentor or TemporalAugmentor()
        self.padding_value = padding_value
        self.verbose = verbose
        self.random_state = random_state

    def _sample_episode(self, X: np.ndarray, y_enc: np.ndarray, class_indices: Dict[int, np.ndarray]):
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
    @tf.function
    def _proto_loss(sup_emb: tf.Tensor, sup_y: tf.Tensor, qry_emb: tf.Tensor, qry_y: tf.Tensor, n_way: int):
        protos = tf.stack([tf.reduce_mean(tf.boolean_mask(sup_emb, tf.equal(sup_y, c)), axis=0) for c in range(n_way)])
        d = tf.reduce_sum((tf.expand_dims(qry_emb, 1) - tf.expand_dims(protos, 0)) ** 2, axis=2)
        log_p = tf.nn.log_softmax(-d, axis=-1)
        loss = -tf.reduce_mean(tf.gather(log_p, qry_y, batch_dims=1))
        preds = tf.argmax(-d, axis=-1, output_type=tf.int32)
        acc = tf.reduce_mean(tf.cast(tf.equal(preds, qry_y), tf.float32))
        return loss, acc

# ---------------------------------------------------------------------------
# Single Head Prototypical Network
# ---------------------------------------------------------------------------
class SingleHeadPrototypicalNetwork(PrototypicalBase):
    def __init__(self, target: str = "bfrb", **kwargs):
        super().__init__(**kwargs)
        self.target = target

    def fit(self, X: np.ndarray, y: pd.DataFrame):
        tf.keras.backend.clear_session()
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)

        y_seq = y.drop_duplicates("sequence_id").sort_values("sequence_id") if "sequence_id" in y.columns else y
        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(y_seq[self.target])
        self.classes_ = self.label_encoder_.classes_
        class_indices = {c: np.where(y_enc == c)[0] for c in range(len(self.classes_))}

        self.encoder_ = BackboneBuilder.build(
            self.backbone_type, (X.shape[1], X.shape[2]),
            filters=self.filters, kernels=self.kernels, pools=self.pools,
            lstm_units=self.lstm_units, attention_heads=self.attention_heads,
            embed_dim=self.embed_dim, dropout=self.dropout
        )
        
        opt = optimizers.Adam(self.learning_rate)
        best_loss, patience_cnt = np.inf, 0
        steps = max(1, len(X) // self.batch_size)

        for epoch in range(self.epochs):
            epoch_loss, epoch_acc = 0.0, 0.0
            for _ in range(steps):
                sup_x, sup_y, qry_x, qry_y = self._sample_episode(X, y_enc, class_indices)
                sup_x, _ = self.augmentor(sup_x, padding_value=self.padding_value)
                qry_x, _ = self.augmentor(qry_x, padding_value=self.padding_value)
                
                with tf.GradientTape() as tape:
                    sup_emb = self.encoder_(sup_x, training=True)
                    qry_emb = self.encoder_(qry_x, training=True)
                    loss, acc = self._proto_loss(sup_emb, sup_y, qry_emb, qry_y, len(np.unique(sup_y)))
                    
                grads = tape.gradient(loss, self.encoder_.trainable_variables)
                opt.apply_gradients(zip(grads, self.encoder_.trainable_variables))
                epoch_loss += loss.numpy()
                epoch_acc += acc.numpy()
                
            epoch_loss /= steps
            epoch_acc /= steps
            if self.verbose: print(f"Epoch {epoch+1}/{self.epochs} - loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f}")
            
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
        dist = np.sum((emb[:, None, :] - self.prototypes_[None, :, :]) ** 2, axis=2)
        return self.label_encoder_.inverse_transform(np.argmin(dist, axis=1))

# ---------------------------------------------------------------------------
# Multi Head Prototypical Network
# ---------------------------------------------------------------------------
class MultiHeadPrototypicalNetwork(PrototypicalBase):
    def __init__(self, primary_target: str = "bfrb", sub_heads: Optional[List[str]] = None, **kwargs):
        super().__init__(**kwargs)
        self.primary_target = primary_target
        self.sub_heads = sub_heads or ["gesture_action", "orientation", "phase"]

    def fit(self, X: np.ndarray, y: pd.DataFrame):
        # For simplicity in episodic training, we sample based on the primary target
        # but compute prototypes for all heads.
        tf.keras.backend.clear_session()
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)

        y_seq = y.drop_duplicates("sequence_id").sort_values("sequence_id")
        self.primary_encoder_ = LabelEncoder()
        y_primary = self.primary_encoder_.fit_transform(y_seq[self.primary_target])
        self.primary_classes_ = self.primary_encoder_.classes_
        
        self.sub_encoders_ = {}
        self.sub_prototypes_ = {}
        for head in self.sub_heads:
            if head in y_seq.columns:
                le = LabelEncoder()
                le.fit(y_seq[head])
                self.sub_encoders_[head] = le

        class_indices = {c: np.where(y_primary == c)[0] for c in range(len(self.primary_classes_))}

        self.encoder_ = BackboneBuilder.build(
            self.backbone_type, (X.shape[1], X.shape[2]),
            filters=self.filters, kernels=self.kernels, pools=self.pools,
            lstm_units=self.lstm_units, attention_heads=self.attention_heads,
            embed_dim=self.embed_dim, dropout=self.dropout
        )
        
        opt = optimizers.Adam(self.learning_rate)
        best_loss, patience_cnt = np.inf, 0
        steps = max(1, len(X) // self.batch_size)

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for _ in range(steps):
                sup_x, sup_y, qry_x, qry_y = self._sample_episode(X, y_primary, class_indices)
                sup_x, _ = self.augmentor(sup_x, padding_value=self.padding_value)
                qry_x, _ = self.augmentor(qry_x, padding_value=self.padding_value)
                
                with tf.GradientTape() as tape:
                    sup_emb = self.encoder_(sup_x, training=True)
                    qry_emb = self.encoder_(qry_x, training=True)
                    loss, _ = self._proto_loss(sup_emb, sup_y, qry_emb, qry_y, len(np.unique(sup_y)))
                    
                grads = tape.gradient(loss, self.encoder_.trainable_variables)
                opt.apply_gradients(zip(grads, self.encoder_.trainable_variables))
                epoch_loss += loss.numpy()
                
            epoch_loss /= steps
            if self.verbose: print(f"Epoch {epoch+1}/{self.epochs} - loss: {epoch_loss:.4f}")
            
            if epoch_loss < best_loss: best_loss, patience_cnt = epoch_loss, 0
            else:
                patience_cnt += 1
                if patience_cnt >= self.patience: break

        all_emb = self.encoder_.predict(X, verbose=0)
        self.prototypes_ = np.array([np.mean(all_emb[y_primary == c], axis=0) for c in range(len(self.primary_classes_))])
        self.prototypes_ /= (np.linalg.norm(self.prototypes_, axis=1, keepdims=True) + 1e-8)
        
        # Compute sub-head prototypes
        for head, le in self.sub_encoders_.items():
            y_head = le.transform(y_seq[head])
            protos = np.array([np.mean(all_emb[y_head == c], axis=0) for c in range(len(le.classes_))])
            protos /= (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8)
            self.sub_prototypes_[head] = protos
            
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["encoder_", "primary_encoder_", "prototypes_"])
        emb = self.encoder_.predict(X, verbose=0)
        dist = np.sum((emb[:, None, :] - self.prototypes_[None, :, :]) ** 2, axis=2)
        return self.primary_encoder_.inverse_transform(np.argmin(dist, axis=1))

# Aliases for notebook compatibility
BinaryPlusGesturePrototypicalNetwork = SingleHeadPrototypicalNetwork
DynamicMultiHeadPrototypicalNetwork = MultiHeadPrototypicalNetwork

# ---------------------------------------------------------------------------
# Custom Scorer for Pipeline
# ---------------------------------------------------------------------------
def make_competition_scorer(target_col='bfrb'):
    """
    Custom scorer that handles the mismatch between row-level y_true
    (from CV splitter) and sequence-level y_pred (from the model).
    Evaluates Binary F1 + Macro F1 (with non-targets collapsed).
    """
    def _score(y_true, y_pred):
        if isinstance(y_true, pd.DataFrame):
            y_true_seq = y_true.drop_duplicates(subset=['sequence_id']).sort_values('sequence_id')
            y_true_binary = y_true_seq['is_target'].astype(int).values
            y_true_gesture = y_true_seq[target_col].values
        else:
            y_true_binary = np.array(y_true)
            y_true_gesture = np.array(y_true)

        y_pred = np.array(y_pred)

        # Binary F1 (target vs non-target)
        y_pred_binary = (y_pred != 'non_bfrb').astype(int)
        binary_f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

        # Macro F1 (non-targets are already collapsed into 'non_bfrb' in y_pred and y_true_gesture)
        macro_f1 = f1_score(y_true_gesture, y_pred, average='macro', zero_division=0)

        return (binary_f1 + macro_f1) / 2

    return make_scorer(_score, response_method='predict')

competition_scorer = make_competition_scorer('bfrb')