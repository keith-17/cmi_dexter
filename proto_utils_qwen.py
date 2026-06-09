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
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, make_scorer
from typing import Tuple, Dict, Any, List, Optional
import warnings
warnings.filterwarnings("ignore")
from sklearn.utils.validation import check_is_fitted
import json

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
        dropout: float = 0.2,
        spatial_dropout: float = 0.1,
        l2_reg: float = 1e-4,
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

        l2 = regularizers.l2(l2_reg)
        use_cnn_reg = backbone_type == "1dcnn"

        if backbone_type == "1dcnn":
            f_list = BackboneBuilder._parse_tuple(filters)
            k_list = BackboneBuilder._parse_tuple(kernels)
            p_list = BackboneBuilder._parse_tuple(pools)
            for i, (f, k) in enumerate(zip(f_list, k_list)):
                x = layers.Conv1D(f, k, padding="same", activation="relu", kernel_regularizer=l2)(x)
                x = layers.BatchNormalization()(x)
                if spatial_dropout > 0:
                    x = layers.SpatialDropout1D(rate=spatial_dropout)(x)
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

        dense_l2 = l2 if use_cnn_reg else None
        x = layers.Dense(128, activation="relu", kernel_regularizer=dense_l2)(x)
        x = layers.Dropout(dropout)(x)
        emb = layers.Dense(embed_dim, name="embedding", kernel_regularizer=dense_l2)(x)
        emb = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(emb)
        
        return models.Model(inp, emb, name=f"{backbone_type}_encoder")


# ============================================================================
# Prototypical Base Class (with full augmentation parameters)
# ============================================================================
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
            spatial_dropout: float = 0.1,
            l2_reg: float = 1e-4,
            learning_rate: float = 1e-3,
            batch_size: int = 32,
            epochs: int = 100,
            patience: int = 15,
            validation_split: float = 0.1,
            padding_value: float = -999.0,
            verbose: int = 1,
            random_state: int = 42,
            temperature: float = 0.1,
            # Augmentation parameters
            use_mixup: bool = False,
            mixup_alpha: float = 0.4,
            mixup_prob: float = 0.5,
            use_time_shift: bool = False,
            max_shift_pct: float = 0.15,
            use_time_stretch: bool = False,
            time_stretch_min: float = 0.8,
            time_stretch_max: float = 1.2,
            use_gaussian_noise: bool = False,
            noise_std: float = 0.01,
            use_magnitude_scaling: bool = False,
            mag_min: float = 0.9,
            mag_max: float = 1.1,
            use_time_mask: bool = False,
            time_mask_ratio: float = 0.1,
            use_channel_dropout: bool = False,
            channel_drop_prob: float = 0.1,
            use_quaternion_flip: bool = False,
            quat_flip_prob: float = 0.5,
            use_freq_filter: bool = False,
            freq_keep_low: float = 0.1,
            freq_keep_high: float = 0.9,
    ):
        # Basic parameters
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
        self.spatial_dropout = spatial_dropout
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.validation_split = validation_split
        self.padding_value = padding_value
        self.verbose = verbose
        self.random_state = random_state
        self.temperature = temperature

        # Augmentation parameters
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

        # Create the augmentor internally
        self.augmentor = TemporalAugmentor(
            use_mixup=use_mixup,
            mixup_alpha=mixup_alpha,
            mixup_prob=mixup_prob,
            use_time_shift=use_time_shift,
            max_shift_pct=max_shift_pct,
            use_time_stretch=use_time_stretch,
            time_stretch_min=time_stretch_min,
            time_stretch_max=time_stretch_max,
            use_gaussian_noise=use_gaussian_noise,
            noise_std=noise_std,
            use_magnitude_scaling=use_magnitude_scaling,
            mag_min=mag_min,
            mag_max=mag_max,
            use_time_mask=use_time_mask,
            time_mask_ratio=time_mask_ratio,
            use_channel_dropout=use_channel_dropout,
            channel_drop_prob=channel_drop_prob,
            use_quaternion_flip=use_quaternion_flip,
            quat_flip_prob=quat_flip_prob,
            use_freq_filter=use_freq_filter,
            freq_keep_low=freq_keep_low,
            freq_keep_high=freq_keep_high
        )

    def _sample_episode(
            self,
            X: np.ndarray,
            y_enc: np.ndarray,
            class_indices: Dict[int, np.ndarray],
            return_indices: bool = False,
    ):
        available = [c for c, idx in class_indices.items() if len(idx) >= self.n_support + self.n_query]
        if len(available) < self.n_way:
            available = [c for c, idx in class_indices.items() if len(idx) > 0]
        if not available:
            raise ValueError(
                "No classes with samples available for episode sampling "
                f"(n_support={self.n_support}, n_query={self.n_query})."
            )

        episode_classes = np.random.choice(available, min(self.n_way, len(available)), replace=False)
        sup_x, sup_y, qry_x, qry_y = [], [], [], []
        sup_idx, qry_idx = [], []

        for local_lbl, cls in enumerate(episode_classes):
            pool = np.asarray(class_indices[cls], dtype=np.int32).copy()
            np.random.shuffle(pool)
            need = self.n_support + self.n_query
            if len(pool) == 0:
                continue
            if len(pool) < need:
                pool = np.tile(pool, (need + len(pool) - 1) // len(pool))[:need]

            support_pool = pool[:self.n_support]
            query_pool = pool[self.n_support:self.n_support + self.n_query]
            sup_x.append(X[support_pool])
            sup_y.extend([local_lbl] * self.n_support)
            qry_x.append(X[query_pool])
            qry_y.extend([local_lbl] * self.n_query)
            if return_indices:
                sup_idx.extend(support_pool.tolist())
                qry_idx.extend(query_pool.tolist())

        sup_x = np.concatenate(sup_x)
        qry_x = np.concatenate(qry_x)
        sup_y = np.array(sup_y, dtype=np.int32)
        qry_y = np.array(qry_y, dtype=np.int32)
        if return_indices:
            return sup_x, sup_y, qry_x, qry_y, np.array(sup_idx, dtype=np.int32), np.array(qry_idx, dtype=np.int32)
        return sup_x, sup_y, qry_x, qry_y

    @staticmethod
    def _remap_episode_labels(labels: np.ndarray) -> Tuple[np.ndarray, int]:
        unique = np.sort(np.unique(labels))
        remap = {int(v): i for i, v in enumerate(unique)}
        local = np.array([remap[int(v)] for v in labels], dtype=np.int32)
        return local, len(unique)

    @staticmethod
    @tf.function
    def _proto_loss(
            sup_emb: tf.Tensor,
            sup_y: tf.Tensor,
            qry_emb: tf.Tensor,
            qry_y: tf.Tensor,
            n_way: int,
            temperature: float = 0.1,
    ):
        protos = tf.stack([tf.reduce_mean(tf.boolean_mask(sup_emb, tf.equal(sup_y, c)), axis=0) for c in range(n_way)])
        d = tf.reduce_sum((tf.expand_dims(qry_emb, 1) - tf.expand_dims(protos, 0)) ** 2, axis=2)
        log_p = tf.nn.log_softmax(-d / temperature, axis=-1)
        loss = -tf.reduce_mean(tf.gather(log_p, qry_y, batch_dims=1))
        preds = tf.argmax(-d, axis=-1, output_type=tf.int32)
        acc = tf.reduce_mean(tf.cast(tf.equal(preds, qry_y), tf.float32))
        return loss, acc


# ============================================================================
# Single Head Prototypical Network
# ============================================================================
class SingleHeadPrototypicalNetwork(PrototypicalBase):
    def __init__(
            self,
            target: str = "bfrb",
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
            spatial_dropout: float = 0.1,
            l2_reg: float = 1e-4,
            learning_rate: float = 1e-3,
            batch_size: int = 32,
            epochs: int = 100,
            patience: int = 15,
            validation_split: float = 0.1,
            padding_value: float = -999.0,
            verbose: int = 1,
            random_state: int = 42,
            temperature: float = 0.1,
            # Augmentation parameters
            use_mixup: bool = False,
            mixup_alpha: float = 0.4,
            mixup_prob: float = 0.5,
            use_time_shift: bool = False,
            max_shift_pct: float = 0.15,
            use_time_stretch: bool = False,
            time_stretch_min: float = 0.8,
            time_stretch_max: float = 1.2,
            use_gaussian_noise: bool = False,
            noise_std: float = 0.01,
            use_magnitude_scaling: bool = False,
            mag_min: float = 0.9,
            mag_max: float = 1.1,
            use_time_mask: bool = False,
            time_mask_ratio: float = 0.1,
            use_channel_dropout: bool = False,
            channel_drop_prob: float = 0.1,
            use_quaternion_flip: bool = False,
            quat_flip_prob: float = 0.5,
            use_freq_filter: bool = False,
            freq_keep_low: float = 0.1,
            freq_keep_high: float = 0.9,
    ):
        self.target = target
        super().__init__(
            n_way=n_way,
            n_support=n_support,
            n_query=n_query,
            backbone_type=backbone_type,
            filters=filters,
            kernels=kernels,
            pools=pools,
            lstm_units=lstm_units,
            attention_heads=attention_heads,
            embed_dim=embed_dim,
            dropout=dropout,
            spatial_dropout=spatial_dropout,
            l2_reg=l2_reg,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            validation_split=validation_split,
            padding_value=padding_value,
            verbose=verbose,
            random_state=random_state,
            temperature=temperature,
            use_mixup=use_mixup,
            mixup_alpha=mixup_alpha,
            mixup_prob=mixup_prob,
            use_time_shift=use_time_shift,
            max_shift_pct=max_shift_pct,
            use_time_stretch=use_time_stretch,
            time_stretch_min=time_stretch_min,
            time_stretch_max=time_stretch_max,
            use_gaussian_noise=use_gaussian_noise,
            noise_std=noise_std,
            use_magnitude_scaling=use_magnitude_scaling,
            mag_min=mag_min,
            mag_max=mag_max,
            use_time_mask=use_time_mask,
            time_mask_ratio=time_mask_ratio,
            use_channel_dropout=use_channel_dropout,
            channel_drop_prob=channel_drop_prob,
            use_quaternion_flip=use_quaternion_flip,
            quat_flip_prob=quat_flip_prob,
            use_freq_filter=use_freq_filter,
            freq_keep_low=freq_keep_low,
            freq_keep_high=freq_keep_high,
        )

    def fit(self, X: np.ndarray, y: pd.DataFrame):
        tf.keras.backend.clear_session()
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)

        # Prepare sequence-level labels
        if "sequence_id" in y.columns:
            y_seq = y.drop_duplicates("sequence_id").sort_values("sequence_id")
            # Get unique sequences from X (since X is already grouped by sequence)
            unique_seqs = np.unique([i for i in range(len(X))])  # X is (n_sequences, maxlen, features)
            n_seq = len(unique_seqs)
        else:
            y_seq = y
            n_seq = len(X)

        # Split indices for train/validation
        indices = np.arange(n_seq)
        np.random.shuffle(indices)
        val_size = int(n_seq * self.validation_split)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        # Encode labels
        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(y_seq[self.target])
        self.classes_ = self.label_encoder_.classes_

        # Get class indices for training sequences only
        train_class_indices = {}
        for c in range(len(self.classes_)):
            train_class_indices[c] = [i for i in train_indices if y_enc[i] == c]

        # Build encoder
        self.encoder_ = BackboneBuilder.build(
            self.backbone_type,
            (X.shape[1], X.shape[2]),
            filters=self.filters,
            kernels=self.kernels,
            pools=self.pools,
            lstm_units=self.lstm_units,
            attention_heads=self.attention_heads,
            embed_dim=self.embed_dim,
            dropout=self.dropout,
            spatial_dropout=self.spatial_dropout,
            l2_reg=self.l2_reg,
        )

        opt = optimizers.Adam(self.learning_rate)
        best_val_loss = np.inf
        patience_cnt = 0
        steps = max(1, len(train_indices) // self.batch_size)

        # Store history
        self.history_ = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

        if int(self.verbose) > 0:
            print(
                f"Training for up to {self.epochs} epochs "
                f"({steps} steps/epoch, {len(train_indices)} train / {len(val_indices)} val sequences)...",
                flush=True,
            )

        for epoch in range(self.epochs):
            # Training phase
            train_loss = 0.0
            train_acc = 0.0

            for _ in range(steps):
                sup_x, sup_y, qry_x, qry_y = self._sample_episode(X, y_enc, train_class_indices)
                sup_x, _ = self.augmentor(sup_x, padding_value=self.padding_value)
                qry_x, _ = self.augmentor(qry_x, padding_value=self.padding_value)

                with tf.GradientTape() as tape:
                    sup_emb = self.encoder_(sup_x, training=True)
                    qry_emb = self.encoder_(qry_x, training=True)
                    loss, acc = self._proto_loss(
                        sup_emb, sup_y, qry_emb, qry_y, len(np.unique(sup_y)), self.temperature
                    )

                grads = tape.gradient(loss, self.encoder_.trainable_variables)
                opt.apply_gradients(zip(grads, self.encoder_.trainable_variables))
                train_loss += loss.numpy()
                train_acc += acc.numpy()

            train_loss /= steps
            train_acc /= steps

            # Validation phase (if we have validation data)
            val_loss = 0.0
            val_acc = 0.0
            val_ran = False

            if len(val_indices) > 0:
                val_class_indices = {}
                for c in range(len(self.classes_)):
                    val_class_indices[c] = [i for i in val_indices if y_enc[i] == c]

                if any(len(idx) > 0 for idx in val_class_indices.values()):
                    val_ran = True
                    val_steps = max(1, len(val_indices) // self.batch_size)
                    for _ in range(val_steps):
                        sup_x, sup_y, qry_x, qry_y = self._sample_episode(X, y_enc, val_class_indices)
                        sup_x, _ = self.augmentor(sup_x, padding_value=self.padding_value)
                        qry_x, _ = self.augmentor(qry_x, padding_value=self.padding_value)

                        sup_emb = self.encoder_(sup_x, training=False)
                        qry_emb = self.encoder_(qry_x, training=False)
                        loss, acc = self._proto_loss(
                            sup_emb, sup_y, qry_emb, qry_y, len(np.unique(sup_y)), self.temperature
                        )
                        val_loss += loss.numpy()
                        val_acc += acc.numpy()

                    val_loss /= val_steps
                    val_acc /= val_steps

            # Store history
            self.history_['train_loss'].append(train_loss)
            self.history_['train_acc'].append(train_acc)
            self.history_['val_loss'].append(val_loss)
            self.history_['val_acc'].append(val_acc)

            if int(self.verbose) > 0:
                msg = (
                    f"Epoch {epoch + 1:3d}/{self.epochs}"
                    f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}"
                )
                if val_ran:
                    msg += f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
                else:
                    msg += "  val_loss=N/A  val_acc=N/A"
                print(msg, flush=True)

            # Early stopping
            monitor_loss = val_loss if val_ran else train_loss
            if monitor_loss < best_val_loss:
                best_val_loss = monitor_loss
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= self.patience:
                    if int(self.verbose) > 0:
                        print(f"Early stopping at epoch {epoch + 1}", flush=True)
                    break

        # Compute final prototypes using all training data
        all_emb = self.encoder_.predict(X[train_indices] if len(val_indices) > 0 else X, verbose=0)
        y_train_enc = y_enc[train_indices] if len(val_indices) > 0 else y_enc
        self.prototypes_ = np.array(
            [np.mean(all_emb[y_train_enc == c], axis=0) for c in range(len(self.classes_))]
        )
        self.prototypes_ /= np.linalg.norm(self.prototypes_, axis=1, keepdims=True) + 1e-8
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, ["encoder_", "label_encoder_", "prototypes_"])
        emb = self.encoder_.predict(X, verbose=0)
        dist = np.sum((emb[:, None, :] - self.prototypes_[None, :, :]) ** 2, axis=2)
        return self.label_encoder_.inverse_transform(np.argmin(dist, axis=1))


# ============================================================================
# Multi Head Prototypical Network
# ============================================================================
class MultiHeadPrototypicalNetwork(PrototypicalBase):
    def __init__(
            self,
            primary_target: str = "bfrb",
            sub_heads: Optional[List[str]] = None,
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
            spatial_dropout: float = 0.1,
            l2_reg: float = 1e-4,
            learning_rate: float = 1e-3,
            batch_size: int = 32,
            epochs: int = 100,
            patience: int = 15,
            validation_split: float = 0.1,
            padding_value: float = -999.0,
            verbose: int = 1,
            random_state: int = 42,
            temperature: float = 0.1,
            # Augmentation parameters
            use_mixup: bool = False,
            mixup_alpha: float = 0.4,
            mixup_prob: float = 0.5,
            use_time_shift: bool = False,
            max_shift_pct: float = 0.15,
            use_time_stretch: bool = False,
            time_stretch_min: float = 0.8,
            time_stretch_max: float = 1.2,
            use_gaussian_noise: bool = False,
            noise_std: float = 0.01,
            use_magnitude_scaling: bool = False,
            mag_min: float = 0.9,
            mag_max: float = 1.1,
            use_time_mask: bool = False,
            time_mask_ratio: float = 0.1,
            use_channel_dropout: bool = False,
            channel_drop_prob: float = 0.1,
            use_quaternion_flip: bool = False,
            quat_flip_prob: float = 0.5,
            use_freq_filter: bool = False,
            freq_keep_low: float = 0.1,
            freq_keep_high: float = 0.9,
            uncertainty_weighting: bool = True,
            fixed_loss_weights: Optional[Dict[str, float]] = None,
    ):
        if isinstance(sub_heads, str) and sub_heads.startswith(('[', '{')):
            try:
                sub_heads = json.loads(sub_heads)
            except:
                pass
        self.primary_target = primary_target
        self.sub_heads = sub_heads or ["gesture_action", "orientation", "phase"]
        self.uncertainty_weighting = uncertainty_weighting
        self.fixed_loss_weights = fixed_loss_weights
        super().__init__(
            n_way=n_way,
            n_support=n_support,
            n_query=n_query,
            backbone_type=backbone_type,
            filters=filters,
            kernels=kernels,
            pools=pools,
            lstm_units=lstm_units,
            attention_heads=attention_heads,
            embed_dim=embed_dim,
            dropout=dropout,
            spatial_dropout=spatial_dropout,
            l2_reg=l2_reg,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            validation_split=validation_split,
            padding_value=padding_value,
            verbose=verbose,
            random_state=random_state,
            temperature=temperature,
            use_mixup=use_mixup,
            mixup_alpha=mixup_alpha,
            mixup_prob=mixup_prob,
            use_time_shift=use_time_shift,
            max_shift_pct=max_shift_pct,
            use_time_stretch=use_time_stretch,
            time_stretch_min=time_stretch_min,
            time_stretch_max=time_stretch_max,
            use_gaussian_noise=use_gaussian_noise,
            noise_std=noise_std,
            use_magnitude_scaling=use_magnitude_scaling,
            mag_min=mag_min,
            mag_max=mag_max,
            use_time_mask=use_time_mask,
            time_mask_ratio=time_mask_ratio,
            use_channel_dropout=use_channel_dropout,
            channel_drop_prob=channel_drop_prob,
            use_quaternion_flip=use_quaternion_flip,
            quat_flip_prob=quat_flip_prob,
            use_freq_filter=use_freq_filter,
            freq_keep_low=freq_keep_low,
            freq_keep_high=freq_keep_high,
        )

    def _head_episode_labels(
            self,
            support_global: np.ndarray,
            query_global: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        unique = np.sort(np.unique(np.concatenate([support_global, query_global])))
        remap = {int(v): i for i, v in enumerate(unique)}
        sup_local = np.array([remap[int(v)] for v in support_global], dtype=np.int32)
        qry_local = np.array([remap[int(v)] for v in query_global], dtype=np.int32)
        return sup_local, qry_local, len(unique)

    def _multitask_proto_losses(
            self,
            sup_emb: tf.Tensor,
            qry_emb: tf.Tensor,
            sup_idx: np.ndarray,
            qry_idx: np.ndarray,
            sup_y_primary: np.ndarray,
            qry_y_primary: np.ndarray,
    ) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
        head_losses: Dict[str, tf.Tensor] = {}
        head_accs: Dict[str, tf.Tensor] = {}

        n_way_primary = len(np.unique(sup_y_primary))
        loss_p, acc_p = self._proto_loss(
            sup_emb, sup_y_primary, qry_emb, qry_y_primary, n_way_primary, self.temperature
        )
        head_losses["primary"] = loss_p
        head_accs["primary"] = acc_p

        for head, y_enc in self.y_encodings_.items():
            if head == "primary":
                continue
            sup_y_h, qry_y_h, n_way_h = self._head_episode_labels(y_enc[sup_idx], y_enc[qry_idx])
            if n_way_h < 1:
                continue
            loss_h, acc_h = self._proto_loss(
                sup_emb, sup_y_h, qry_emb, qry_y_h, n_way_h, self.temperature
            )
            head_losses[head] = loss_h
            head_accs[head] = acc_h

        return head_losses, head_accs

    def _effective_loss_weights(self) -> Dict[str, float]:
        weights = {"primary": 1.0}
        if self.fixed_loss_weights:
            weights.update(self.fixed_loss_weights)
        for head in self.sub_heads:
            weights.setdefault(head, 0.5)
        return weights

    def _combine_head_losses(self, head_losses: Dict[str, tf.Tensor]) -> tf.Tensor:
        if self.uncertainty_weighting:
            total = tf.constant(0.0, dtype=tf.float32)
            for head, loss in head_losses.items():
                log_var = self.log_vars_[head]
                total += tf.exp(-log_var) * loss + log_var
            return total

        weights = self._effective_loss_weights()
        total = tf.constant(0.0, dtype=tf.float32)
        for head, loss in head_losses.items():
            weight = float(weights.get(head, 0.5))
            total += weight * loss
        return total

    def _episode_step(
            self,
            X: np.ndarray,
            class_indices: Dict[int, List[int]],
            training: bool,
    ) -> Tuple[float, float]:
        sup_x, sup_y, qry_x, qry_y, sup_idx, qry_idx = self._sample_episode(
            X, self.y_encodings_["primary"], class_indices, return_indices=True
        )
        sup_x, _ = self.augmentor(sup_x, padding_value=self.padding_value)
        qry_x, _ = self.augmentor(qry_x, padding_value=self.padding_value)

        if training:
            with tf.GradientTape() as tape:
                sup_emb = self.encoder_(sup_x, training=True)
                qry_emb = self.encoder_(qry_x, training=True)
                head_losses, head_accs = self._multitask_proto_losses(
                    sup_emb, qry_emb, sup_idx, qry_idx, sup_y, qry_y
                )
                total_loss = self._combine_head_losses(head_losses)
            trainable_vars = self.encoder_.trainable_variables + list(self.log_vars_.values())
            grads = tape.gradient(total_loss, trainable_vars)
            self.optimizer_.apply_gradients(zip(grads, trainable_vars))
            return float(total_loss.numpy()), float(head_accs["primary"].numpy())

        sup_emb = self.encoder_(sup_x, training=False)
        qry_emb = self.encoder_(qry_x, training=False)
        head_losses, head_accs = self._multitask_proto_losses(
            sup_emb, qry_emb, sup_idx, qry_idx, sup_y, qry_y
        )
        total_loss = self._combine_head_losses(head_losses)
        return float(total_loss.numpy()), float(head_accs["primary"].numpy())

    def fit(self, X: np.ndarray, y: pd.DataFrame):
        tf.keras.backend.clear_session()
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)

        # Prepare sequence-level labels
        y_seq = y.drop_duplicates("sequence_id").sort_values("sequence_id")
        n_seq = len(y_seq)

        # Split indices for train/validation
        indices = np.arange(n_seq)
        np.random.shuffle(indices)
        val_size = int(n_seq * self.validation_split)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        # Primary target encoder
        self.primary_encoder_ = LabelEncoder()
        y_primary = self.primary_encoder_.fit_transform(y_seq[self.primary_target])
        self.primary_classes_ = self.primary_encoder_.classes_

        # Sub-head encoders and per-sequence encodings for all heads
        self.sub_encoders_ = {}
        self.sub_prototypes_ = {}
        self.y_encodings_ = {"primary": y_primary.astype(np.int32)}
        for head in self.sub_heads:
            if head in y_seq.columns:
                le = LabelEncoder()
                le.fit(y_seq[head])
                self.sub_encoders_[head] = le
                self.y_encodings_[head] = le.transform(y_seq[head]).astype(np.int32)

        self.log_vars_ = {
            head: tf.Variable(0.0, trainable=True, dtype=tf.float32, name=f"log_var_{head}")
            for head in self.y_encodings_
        }

        # Get class indices for training sequences
        train_class_indices = {}
        for c in range(len(self.primary_classes_)):
            train_class_indices[c] = [i for i in train_indices if y_primary[i] == c]

        # Build encoder
        self.encoder_ = BackboneBuilder.build(
            self.backbone_type,
            (X.shape[1], X.shape[2]),
            filters=self.filters,
            kernels=self.kernels,
            pools=self.pools,
            lstm_units=self.lstm_units,
            attention_heads=self.attention_heads,
            embed_dim=self.embed_dim,
            dropout=self.dropout,
            spatial_dropout=self.spatial_dropout,
            l2_reg=self.l2_reg,
        )

        self.optimizer_ = optimizers.Adam(self.learning_rate)
        best_val_loss = np.inf
        patience_cnt = 0
        steps = max(1, len(train_indices) // self.batch_size)

        # Store history
        self.history_ = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
        }

        active_heads = [h for h in self.y_encodings_]
        if int(self.verbose) > 0:
            print(
                f"Training for up to {self.epochs} epochs "
                f"({steps} steps/epoch, {len(train_indices)} train / {len(val_indices)} val sequences, "
                f"heads={active_heads})...",
                flush=True,
            )

        for epoch in range(self.epochs):
            # Training phase
            train_loss = 0.0
            train_acc = 0.0
            for _ in range(steps):
                step_loss, step_acc = self._episode_step(X, train_class_indices, training=True)
                train_loss += step_loss
                train_acc += step_acc

            train_loss /= steps
            train_acc /= steps

            # Validation phase
            val_loss = 0.0
            val_acc = 0.0
            val_ran = False
            if len(val_indices) > 0:
                val_class_indices = {}
                for c in range(len(self.primary_classes_)):
                    val_class_indices[c] = [i for i in val_indices if y_primary[i] == c]

                if any(len(idx) > 0 for idx in val_class_indices.values()):
                    val_ran = True
                    val_steps = max(1, len(val_indices) // self.batch_size)
                    for _ in range(val_steps):
                        step_loss, step_acc = self._episode_step(X, val_class_indices, training=False)
                        val_loss += step_loss
                        val_acc += step_acc

                    val_loss /= val_steps
                    val_acc /= val_steps

            # Store history
            self.history_['train_loss'].append(train_loss)
            self.history_['train_acc'].append(train_acc)
            self.history_['val_loss'].append(val_loss)
            self.history_['val_acc'].append(val_acc)

            if int(self.verbose) > 0:
                msg = (
                    f"Epoch {epoch + 1:3d}/{self.epochs}"
                    f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}"
                )
                if val_ran:
                    msg += f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
                else:
                    msg += "  val_loss=N/A  val_acc=N/A"
                print(msg, flush=True)

            # Early stopping
            monitor_loss = val_loss if val_ran else train_loss
            if monitor_loss < best_val_loss:
                best_val_loss = monitor_loss
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= self.patience:
                    if int(self.verbose) > 0:
                        print(f"Early stopping at epoch {epoch + 1}", flush=True)
                    break

        # Compute primary prototypes using all training data
        all_emb = self.encoder_.predict(X[train_indices] if len(val_indices) > 0 else X, verbose=0)
        y_train_primary = y_primary[train_indices] if len(val_indices) > 0 else y_primary
        self.prototypes_ = np.array(
            [np.mean(all_emb[y_train_primary == c], axis=0) for c in range(len(self.primary_classes_))]
        )
        self.prototypes_ /= np.linalg.norm(self.prototypes_, axis=1, keepdims=True) + 1e-8

        # Compute sub-head prototypes
        for head, le in self.sub_encoders_.items():
            y_head = le.transform(y_seq[head])
            y_train_head = y_head[train_indices] if len(val_indices) > 0 else y_head
            protos = np.array([np.mean(all_emb[y_train_head == c], axis=0) for c in range(len(le.classes_))])
            protos /= np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8
            self.sub_prototypes_[head] = protos

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, ["encoder_", "primary_encoder_", "prototypes_"])
        emb = self.encoder_.predict(X, verbose=0)
        dist = np.sum((emb[:, None, :] - self.prototypes_[None, :, :]) ** 2, axis=2)
        return self.primary_encoder_.inverse_transform(np.argmin(dist, axis=1))
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