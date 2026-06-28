"""
proto_utils_v4.py
DEFINITIVE V4: All Augmentations, All Backbones, Multi-Head, SupCon, Phase-Aware, Modality Dropout.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict, Any, List, Optional
import warnings
warnings.filterwarnings("ignore")
from sklearn.utils.validation import check_is_fitted
from scipy.signal import resample
from scipy.ndimage import zoom
import inspect

# Import your base extractor
# ============================================================================
# 2. EXHAUSTIVE TEMPORAL AUGMENTATIONS (All 11 from proto_utils_qwen)
# ============================================================================
class TemporalAugmentor:
    def __init__(self, use_mixup=False, mixup_alpha=0.4, mixup_prob=0.5,
                 use_time_shift=False, max_shift_pct=0.15,
                 use_time_stretch=False, time_stretch_min=0.8, time_stretch_max=1.2,
                 use_gaussian_noise=False, noise_std=0.01,
                 use_magnitude_scaling=False, mag_min=0.9, mag_max=1.1,
                 use_time_mask=False, time_mask_ratio=0.1,
                 use_channel_dropout=False, channel_drop_prob=0.1,
                 use_quaternion_flip=False, quat_flip_prob=0.5,
                 use_freq_filter=False, freq_keep_low=0.1, freq_keep_high=0.9):
        self.use_mixup, self.mixup_alpha, self.mixup_prob = use_mixup, mixup_alpha, mixup_prob
        self.use_time_shift, self.max_shift_pct = use_time_shift, max_shift_pct
        self.use_time_stretch, self.time_stretch_min, self.time_stretch_max = use_time_stretch, time_stretch_min, time_stretch_max
        self.use_gaussian_noise, self.noise_std = use_gaussian_noise, noise_std
        self.use_magnitude_scaling, self.mag_min, self.mag_max = use_magnitude_scaling, mag_min, mag_max
        self.use_time_mask, self.time_mask_ratio = use_time_mask, time_mask_ratio
        self.use_channel_dropout, self.channel_drop_prob = use_channel_dropout, channel_drop_prob
        self.use_quaternion_flip, self.quat_flip_prob = use_quaternion_flip, quat_flip_prob
        self.use_freq_filter, self.freq_keep_low, self.freq_keep_high = use_freq_filter, freq_keep_low, freq_keep_high

    def __call__(self, X: np.ndarray, y: Optional[np.ndarray] = None, padding_value: float = -999.0):
        X_aug, y_aug = X.copy(), y.copy() if y is not None else None
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

        if self.use_time_stretch and np.random.rand() < 0.5:
            factor = np.random.uniform(self.time_stretch_min, self.time_stretch_max)
            for i in range(n):
                valid_mask = X_aug[i, :, 0] != padding_value
                if valid_mask.sum() > 2:
                    valid_data = X_aug[i, valid_mask]
                    new_len = max(2, int(valid_data.shape[0] * factor))
                    stretched = resample(valid_data, new_len, axis=0)
                    X_aug[i] = padding_value
                    fill_len = min(new_len, t)
                    X_aug[i, :fill_len] = stretched[:fill_len]

        if self.use_time_mask and np.random.rand() < 0.5:
            mask_len = max(1, int(t * self.time_mask_ratio))
            for i in range(n):
                start = np.random.randint(0, max(1, t - mask_len + 1))
                X_aug[i, start:start + mask_len, :] = padding_value

        if self.use_channel_dropout and np.random.rand() < 0.5:
            ch_mask = np.random.random((n, 1, f)) < self.channel_drop_prob
            X_aug = np.where(ch_mask & valid, padding_value, X_aug)

        if self.use_quaternion_flip and np.random.rand() < 0.5:
            # Apply random sign flips to simulate quaternion/rotation inversions
            flip_mask = np.random.choice([-1.0, 1.0], size=(n, 1, f), p=[self.quat_flip_prob, 1.0 - self.quat_flip_prob])
            X_aug = np.where(valid, X_aug * flip_mask, X_aug)

        if self.use_freq_filter and np.random.rand() < 0.5:
            for i in range(n):
                for j in range(f):
                    if X_aug[i, 0, j] != padding_value:
                        fft = np.fft.rfft(X_aug[i, :, j])
                        freqs = np.fft.rfftfreq(t)
                        mask = (freqs >= self.freq_keep_low) & (freqs <= self.freq_keep_high)
                        fft[~mask] = 0
                        X_aug[i, :, j] = np.fft.irfft(fft, n=t)

        if self.use_mixup and y_aug is not None and np.random.rand() < self.mixup_prob:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            idx = np.random.permutation(n)
            X_aug = lam * X_aug + (1 - lam) * X_aug[idx]
            
        return X_aug, y_aug

# ============================================================================
# 3. V4 CUSTOM LAYERS
# ============================================================================
class PhaseAwareAttentionPooling(layers.Layer):
    def __init__(self, use_phase_attention=False, **kwargs):
        super().__init__(**kwargs)
        self.use_phase_attention = use_phase_attention
        if self.use_phase_attention:
            self.attention = layers.MultiHeadAttention(num_heads=4, key_dim=32)
            self.norm = layers.LayerNormalization()
        
    def call(self, x, mask=None):
        if not self.use_phase_attention:
            if mask is not None:
                mask = tf.cast(mask, tf.float32)
                mask = tf.expand_dims(mask, -1)
                x = x * mask
                return tf.reduce_sum(x, axis=1) / (tf.reduce_sum(mask, axis=1) + 1e-9)
            return tf.reduce_mean(x, axis=1)
            
        seq_len = tf.shape(x)[1]
        time_bias = tf.linspace(0.0, 5.0, seq_len) 
        time_bias = tf.reshape(time_bias, (1, seq_len, 1))
        attn_out = self.attention(x, x)
        x = self.norm(x + attn_out)
        x = x * tf.exp(time_bias) 
        
        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            mask = tf.expand_dims(mask, -1)
            x = x * mask
            return tf.reduce_sum(x, axis=1) / (tf.reduce_sum(mask, axis=1) + 1e-9)
        return tf.reduce_mean(x, axis=1)

class ModalityDropout(layers.Layer):
    def __init__(self, dropout_prob=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dropout_prob = dropout_prob

    def call(self, inputs, training=None):
        if self.dropout_prob > 0.0 and training:
            keep_prob = 1.0 - self.dropout_prob
            uniform = tf.random.uniform((tf.shape(inputs)[0], 1, tf.shape(inputs)[2]), minval=0.0, maxval=1.0)
            mask = tf.cast(uniform < keep_prob, inputs.dtype)
            return inputs * mask
        return inputs

class AttentionPrototypeComputation(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attention = layers.MultiHeadAttention(num_heads=4, key_dim=32)
        self.norm = layers.LayerNormalization()
        
    def call(self, support_embeddings):
        # MultiHeadAttention expects (batch, seq_len, features). support_embeddings is
        # (n_support_total, embed_dim) -- treat it as a single sequence of n_support_total
        # tokens so each support embedding can attend to the others.
        x = tf.expand_dims(support_embeddings, axis=0)
        attn_out = self.attention(x, x)
        out = self.norm(x + attn_out)
        return tf.squeeze(out, axis=0)

# ============================================================================
# 4. EXHAUSTIVE BACKBONE BUILDER (1D, 2D, LSTM, Attention, Conv4, ResNet12/18)
# ============================================================================
class BackboneBuilder:
    @staticmethod
    def _parse_tuple(val: str) -> Tuple[int, ...]:
        if not val or val.lower() == "none": return ()
        return tuple(int(p) for p in val.split("-") if p.lower() != "none")

    @staticmethod
    def build(
        backbone_type: str, input_shape: Tuple[int, int],
        filters: str = "64-128", kernels: str = "3-3", pools: str = "none",
        lstm_units: int = 128, attention_heads: int = 4, embed_dim: int = 128,
        dropout: float = 0.2, spatial_dropout: float = 0.1, l2_reg: float = 1e-4,
        use_phase_attention: bool = False, modality_dropout_prob: float = 0.0,
    ) -> tf.keras.Model:
        inp = layers.Input(shape=input_shape, name="main_input")
        mask_inp = layers.Input(shape=(input_shape[0],), dtype=tf.bool, name="mask_input")
        
        x = inp
        if modality_dropout_prob > 0.0:
            x = ModalityDropout(dropout_prob=modality_dropout_prob)(x)

        l2 = regularizers.l2(l2_reg)

        # --- 2D VISION BACKBONES ---
        if backbone_type in ['conv4', 'resnet12', 'resnet18', '2dcnn']:
            x = layers.Reshape((input_shape[0], input_shape[1], 1))(x)
            f_list = BackboneBuilder._parse_tuple(filters) or (64, 64, 64, 64)
            k_list = BackboneBuilder._parse_tuple(kernels) or (3,) * len(f_list)
            p_list = BackboneBuilder._parse_tuple(pools) or (2,) * len(f_list)
            # Pad k_list/p_list to match f_list length if the grid gave a shorter spec.
            if len(k_list) < len(f_list): k_list = k_list + (k_list[-1],) * (len(f_list) - len(k_list))
            if len(p_list) < len(f_list): p_list = p_list + (p_list[-1],) * (len(f_list) - len(p_list))

            if backbone_type == 'conv4':
                for f, k, p in zip(f_list, k_list, p_list):
                    x = layers.Conv2D(f, (k, k), padding='same', kernel_regularizer=l2)(x)
                    x = layers.BatchNormalization()(x)
                    x = layers.ReLU()(x)
                    x = layers.MaxPooling2D((p, p), padding='same')(x)
            elif backbone_type == 'resnet12':
                for f, k, p in zip(f_list, k_list, p_list):
                    skip = layers.Conv2D(f, (1, 1), padding='same')(x)
                    skip = layers.BatchNormalization()(skip)
                    for _ in range(3):
                        x = layers.Conv2D(f, (k, k), padding='same', kernel_regularizer=l2)(x)
                        x = layers.BatchNormalization()(x)
                        x = layers.ReLU()(x)
                    x = layers.Add()([x, skip])
                    x = layers.ReLU()(x)
                    x = layers.MaxPooling2D((p, p), padding='same')(x)
            elif backbone_type in ['resnet18', '2dcnn']:
                stem_k = k_list[0]
                x = layers.Conv2D(f_list[0], (stem_k, stem_k), padding='same', kernel_regularizer=l2)(x)
                x = layers.BatchNormalization()(x)
                x = layers.ReLU()(x)
                def basic_block(inputs, filters_, kernel_, strides=1):
                    skip = inputs
                    if strides != 1 or inputs.shape[-1] != filters_:
                        skip = layers.Conv2D(filters_, (1, 1), strides=strides, padding='same')(skip)
                        skip = layers.BatchNormalization()(skip)
                    out = layers.Conv2D(filters_, (kernel_, kernel_), strides=strides, padding='same', kernel_regularizer=l2)(inputs)
                    out = layers.BatchNormalization()(out)
                    out = layers.ReLU()(out)
                    out = layers.Conv2D(filters_, (kernel_, kernel_), strides=1, padding='same', kernel_regularizer=l2)(out)
                    out = layers.BatchNormalization()(out)
                    out = layers.Add()([out, skip])
                    return layers.ReLU()(out)
                for stage_idx, (f, k) in enumerate(zip(f_list, k_list)):
                    strides = 1 if stage_idx == 0 else 2
                    x = basic_block(x, f, k, strides=strides)
                    x = basic_block(x, f, k)

            # Masked global average pooling over the (downsampled) time axis, so that
            # zero-padded timesteps don't get averaged in as if they were real signal,
            # and so mask_inp is always connected to the output graph (required by
            # Keras 3's functional Model, which rejects unused Input tensors).
            time_steps_now = x.shape[1]
            pool_factor = input_shape[0] // time_steps_now if time_steps_now else 1
            mask_f = layers.Lambda(lambda m: tf.cast(m, tf.float32))(mask_inp)
            mask_f = layers.Reshape((-1, 1))(mask_f)
            if pool_factor > 1:
                mask_f = layers.MaxPooling1D(pool_factor)(mask_f)
            mask_2d = layers.Lambda(lambda m: tf.cast(tf.squeeze(m, axis=-1) > 0.5, tf.float32))(mask_f)

            def masked_avg_pool_2d(args):
                feat, m = args
                m = tf.reshape(m, (tf.shape(m)[0], tf.shape(feat)[1], 1, 1))
                feat = feat * m
                summed = tf.reduce_sum(feat, axis=[1, 2])
                count = tf.reduce_sum(m, axis=[1, 2]) * tf.cast(tf.shape(feat)[2], tf.float32)
                return summed / (count + 1e-9)
            x = layers.Lambda(masked_avg_pool_2d)([x, mask_2d])



        # --- 1D / SEQUENTIAL BACKBONES ---
        else:
            if backbone_type == '1dcnn':
                f_list = BackboneBuilder._parse_tuple(filters)
                k_list = BackboneBuilder._parse_tuple(kernels)
                p_list = BackboneBuilder._parse_tuple(pools)
                pooled_mask = mask_inp
                for i, (f, k) in enumerate(zip(f_list, k_list)):
                    x = layers.Conv1D(f, k, padding="same", activation="relu", kernel_regularizer=l2)(x)
                    x = layers.BatchNormalization()(x)
                    if spatial_dropout > 0: x = layers.SpatialDropout1D(rate=spatial_dropout)(x)
                    if i < len(p_list) and p_list[i] > 1:
                        pool_size = p_list[i]
                        x = layers.MaxPooling1D(pool_size)(x)
                        # Keep the mask's temporal resolution in sync with the feature map:
                        # a pooled timestep is "valid" if any of the timesteps it covers was valid.
                        mask_f = layers.Reshape((-1, 1))(layers.Lambda(lambda m: tf.cast(m, tf.float32))(pooled_mask))
                        mask_f = layers.MaxPooling1D(pool_size)(mask_f)
                        pooled_mask = layers.Lambda(lambda m: tf.cast(tf.squeeze(m, axis=-1) > 0.5, tf.bool))(mask_f)
                x = PhaseAwareAttentionPooling(use_phase_attention=use_phase_attention)(x, mask=pooled_mask)
            elif backbone_type == 'lstm':
                x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(x)
                x = PhaseAwareAttentionPooling(use_phase_attention=use_phase_attention)(x, mask=mask_inp)
            elif backbone_type == 'attention':
                attn = layers.MultiHeadAttention(num_heads=attention_heads, key_dim=32)(x, x)
                x = layers.Add()([x, attn])
                x = layers.LayerNormalization()(x)
                x = PhaseAwareAttentionPooling(use_phase_attention=use_phase_attention)(x, mask=mask_inp)
            else:
                raise ValueError(f"Unknown backbone_type: {backbone_type}")

        dense_l2 = l2 if backbone_type == '1dcnn' else None
        x = layers.Dense(128, activation="relu", kernel_regularizer=dense_l2)(x)
        x = layers.Dropout(dropout)(x)
        emb = layers.Dense(embed_dim, name="embedding", kernel_regularizer=dense_l2)(x)
        emb = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(emb)
        return models.Model([inp, mask_inp], emb, name=f"{backbone_type}_encoder")

# ============================================================================
# 5. HYBRID LOSS FUNCTIONS (Proto + SupCon + MultiHead)
# ============================================================================
def hybrid_prototypical_loss(sup_emb, sup_y, qry_emb, qry_y, n_way, temperature, supcon_weight):
    protos = tf.stack([tf.reduce_mean(tf.boolean_mask(sup_emb, tf.equal(sup_y, c)), axis=0) for c in range(n_way)])
    d = tf.reduce_sum((tf.expand_dims(qry_emb, 1) - tf.expand_dims(protos, 0)) ** 2, axis=2)
    log_p = tf.nn.log_softmax(-d / temperature, axis=-1)
    proto_loss = -tf.reduce_mean(tf.gather(log_p, qry_y, batch_dims=1))
    preds = tf.argmax(-d, axis=-1, output_type=tf.int32)
    acc = tf.reduce_mean(tf.cast(tf.equal(preds, qry_y), tf.float32))
    
    sup_emb_norm = tf.math.l2_normalize(sup_emb, axis=1)
    sim_matrix = tf.matmul(sup_emb_norm, sup_emb_norm, transpose_b=True) / temperature
    labels = tf.cast(sup_y, tf.int32)
    labels = tf.reshape(labels, [-1, 1])
    mask = tf.cast(tf.equal(labels, tf.transpose(labels)), tf.float32)
    logits_mask = tf.ones_like(mask) - tf.eye(tf.shape(sup_y)[0])
    mask = mask * logits_mask
    logits_max = tf.reduce_max(sim_matrix, axis=1, keepdims=True)
    logits = sim_matrix - logits_max
    exp_logits = tf.exp(logits) * logits_mask
    log_prob = logits - tf.math.log(tf.reduce_sum(exp_logits, axis=1, keepdims=True) + 1e-12)
    positives_count = tf.reduce_sum(mask, axis=1)
    valid_samples = tf.cast(positives_count > 0, tf.float32)
    mean_log_prob = tf.reduce_sum(mask * log_prob, axis=1) / (positives_count + 1e-12)
    supcon_loss = -tf.reduce_sum(mean_log_prob * valid_samples) / (tf.reduce_sum(valid_samples) + 1e-12)
    
    total_loss = proto_loss + supcon_weight * supcon_loss
    return total_loss, acc, proto_loss, supcon_loss

def multitask_hybrid_prototypical_loss(sup_emb, qry_emb, sup_y_dict, qry_y_dict, n_way_dict, 
                                       temperature, supcon_weight, log_vars, uncertainty_weighting, fixed_weights):
    total_loss = 0.0
    head_losses = {}
    head_accs = {}
    for head, sup_y in sup_y_dict.items():
        qry_y = qry_y_dict[head]
        n_way = n_way_dict[head]
        loss, acc, _, _ = hybrid_prototypical_loss(sup_emb, sup_y, qry_emb, qry_y, n_way, temperature, supcon_weight)
        head_losses[head] = loss
        head_accs[head] = acc
        
    if uncertainty_weighting:
        for head, loss in head_losses.items():
            log_var = log_vars[head]
            total_loss += tf.exp(-log_var) * loss + log_var
    else:
        for head, loss in head_losses.items():
            w = fixed_weights.get(head, 1.0) if fixed_weights else 1.0
            total_loss += w * loss
            
    primary_acc = head_accs.get('primary', head_accs[list(head_accs.keys())[0]])
    return total_loss, primary_acc, head_losses

# ============================================================================
# 6. V4 PROTOTYPICAL NETWORKS (Single & Multi-Head)
# ============================================================================
class V4PrototypicalNetwork(ClassifierMixin, BaseEstimator):
    def __init__(self, target="bfrb", n_way=9, n_support=20, n_query=20,
                 backbone_type="1dcnn", filters="64-128", kernels="3-3", pools="none",
                 lstm_units=128, attention_heads=4, embed_dim=128,
                 dropout=0.2, spatial_dropout=0.1, l2_reg=1e-4,
                 learning_rate=1e-3, batch_size=32, epochs=50, patience=10,
                 validation_split=0.1, padding_value=-999.0, verbose=1, random_state=42,
                 temperature=0.1, supcon_weight=0.1,
                 use_phase_attention=False, modality_dropout_prob=0.0, use_attention_prototypes=False,
                 use_mixup=False, mixup_alpha=0.4, mixup_prob=0.5,
                 use_time_shift=False, max_shift_pct=0.15,
                 use_time_stretch=False, time_stretch_min=0.8, time_stretch_max=1.2,
                 use_gaussian_noise=False, noise_std=0.01,
                 use_magnitude_scaling=False, mag_min=0.9, mag_max=1.1,
                 use_time_mask=False, time_mask_ratio=0.1,
                 use_channel_dropout=False, channel_drop_prob=0.1,
                 use_quaternion_flip=False, quat_flip_prob=0.5,
                 use_freq_filter=False, freq_keep_low=0.1, freq_keep_high=0.9):
        locals_dict = locals()
        locals_dict.pop('self')
        for k, v in locals_dict.items(): setattr(self, k, v)
        augmentor_params = inspect.signature(TemporalAugmentor).parameters
        self.augmentor = TemporalAugmentor(**{k: v for k, v in locals_dict.items() if k in augmentor_params})
        
    def _unpack_X(self, X):
        if isinstance(X, dict): return X['X'], X.get('sequence_ids')
        return X, None

    def _sample_episode(self, X, y_enc, class_indices):
        available = [c for c, idx in class_indices.items() if len(idx) >= self.n_support + self.n_query]
        if len(available) < self.n_way: available = [c for c, idx in class_indices.items() if len(idx) > 0]
        if not available: raise ValueError("No classes with enough samples.")
        episode_classes = np.random.choice(available, min(self.n_way, len(available)), replace=False)
        sup_x, sup_y, qry_x, qry_y, sup_idx, qry_idx = [], [], [], [], [], []
        for local_lbl, cls in enumerate(episode_classes):
            pool = np.asarray(class_indices[cls], dtype=np.int32).copy()
            np.random.shuffle(pool)
            need = self.n_support + self.n_query
            if len(pool) < need: pool = np.tile(pool, (need + len(pool) - 1) // len(pool))[:need]
            sup_x.append(X[pool[:self.n_support]]); sup_y.extend([local_lbl] * self.n_support); sup_idx.extend(pool[:self.n_support].tolist())
            qry_x.append(X[pool[self.n_support:need]]); qry_y.extend([local_lbl] * self.n_query); qry_idx.extend(pool[self.n_support:need].tolist())
        return np.concatenate(sup_x), np.array(sup_y, dtype=np.int32), np.concatenate(qry_x), np.array(qry_y, dtype=np.int32), np.array(sup_idx), np.array(qry_idx)

    def fit(self, X: dict, y: pd.DataFrame):
        tf.keras.backend.clear_session()
        tf.random.set_seed(self.random_state); np.random.seed(self.random_state)
        X_arr, seq_ids = self._unpack_X(X)
        y_seq = y.drop_duplicates("sequence_id").sort_values("sequence_id")
        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(y_seq[self.target])
        self.classes_ = self.label_encoder_.classes_
        indices = np.arange(len(y_enc)); np.random.shuffle(indices)
        val_size = int(len(y_enc) * self.validation_split)
        val_indices = indices[:val_size]; train_indices = indices[val_size:]
        train_class_indices = {c: [i for i in train_indices if y_enc[i] == c] for c in range(len(self.classes_))}
        self.encoder_ = BackboneBuilder.build(self.backbone_type, (X_arr.shape[1], X_arr.shape[2]), self.filters, self.kernels, self.pools, self.lstm_units, self.attention_heads, self.embed_dim, self.dropout, self.spatial_dropout, self.l2_reg, self.use_phase_attention, self.modality_dropout_prob)
        self.attn_proto_ = AttentionPrototypeComputation() if self.use_attention_prototypes else None
        opt = optimizers.Adam(self.learning_rate)
        best_val_loss, patience_cnt, steps = np.inf, 0, max(1, len(train_indices) // self.batch_size)
        self.history_ = {'train_loss': [], 'val_loss': []}
        for epoch in range(self.epochs):
            train_loss = 0.0
            for _ in range(steps):
                sup_x, sup_y, qry_x, qry_y, _, _ = self._sample_episode(X_arr, y_enc, train_class_indices)
                sup_x, _ = self.augmentor(sup_x, padding_value=self.padding_value)
                qry_x, _ = self.augmentor(qry_x, padding_value=self.padding_value)
                sup_x = tf.convert_to_tensor(sup_x, dtype=tf.float32)
                qry_x = tf.convert_to_tensor(qry_x, dtype=tf.float32)
                sup_mask = tf.reduce_any(tf.not_equal(sup_x, self.padding_value), axis=-1)
                qry_mask = tf.reduce_any(tf.not_equal(qry_x, self.padding_value), axis=-1)
                with tf.GradientTape() as tape:
                    sup_emb = self.encoder_([sup_x, sup_mask], training=True)
                    qry_emb = self.encoder_([qry_x, qry_mask], training=True)
                    if self.use_attention_prototypes: sup_emb = self.attn_proto_(sup_emb)
                    loss, _, _, _ = hybrid_prototypical_loss(sup_emb, sup_y, qry_emb, qry_y, len(np.unique(sup_y)), self.temperature, self.supcon_weight)
                grads = tape.gradient(loss, self.encoder_.trainable_variables + (self.attn_proto_.trainable_variables if self.attn_proto_ else []))
                opt.apply_gradients(zip(grads, self.encoder_.trainable_variables + (self.attn_proto_.trainable_variables if self.attn_proto_ else [])))
                train_loss += loss.numpy()
            self.history_['train_loss'].append(train_loss/steps)
            if self.verbose > 0: print(f"Epoch {epoch+1}/{self.epochs} | Loss: {train_loss/steps:.4f}")
        all_mask = tf.reduce_any(tf.not_equal(X_arr[train_indices], self.padding_value), axis=-1)
        all_emb = self.encoder_.predict([X_arr[train_indices], all_mask], verbose=0)
        y_train_enc = y_enc[train_indices]
        self.prototypes_ = np.array([np.mean(all_emb[y_train_enc == c], axis=0) for c in range(len(self.classes_))])
        self.prototypes_ /= np.linalg.norm(self.prototypes_, axis=1, keepdims=True) + 1e-8
        return self

    def predict(self, X: dict) -> pd.Series:
        check_is_fitted(self, ["encoder_", "label_encoder_", "prototypes_"])
        X_arr, seq_ids = self._unpack_X(X)
        mask = tf.reduce_any(tf.not_equal(X_arr, self.padding_value), axis=-1)
        emb = self.encoder_.predict([X_arr, mask], verbose=0)
        dist = np.sum((emb[:, None, :] - self.prototypes_[None, :, :]) ** 2, axis=2)
        preds = self.label_encoder_.inverse_transform(np.argmin(dist, axis=1))
        if seq_ids is not None:
            df = pd.DataFrame({'seq_id': seq_ids, 'pred': preds})
            agg_preds = df.groupby('seq_id').agg(lambda x: x.value_counts().index[0])
            return pd.Series(agg_preds['pred'].values, index=agg_preds.index, name=self.target)
        return preds

class V4MultiHeadPrototypicalNetwork(V4PrototypicalNetwork):
    def __init__(self, primary_target="bfrb", sub_heads=None, uncertainty_weighting=True, fixed_loss_weights=None, **kwargs):
        super().__init__(target=primary_target, **kwargs)
        self.primary_target = primary_target
        self.sub_heads = sub_heads or []
        self.uncertainty_weighting = uncertainty_weighting
        self.fixed_loss_weights = fixed_loss_weights

    def fit(self, X: dict, y: pd.DataFrame):
        tf.keras.backend.clear_session()
        tf.random.set_seed(self.random_state); np.random.seed(self.random_state)
        X_arr, seq_ids = self._unpack_X(X)
        y_seq = y.drop_duplicates("sequence_id").sort_values("sequence_id")
        
        self.primary_encoder_ = LabelEncoder()
        y_primary = self.primary_encoder_.fit_transform(y_seq[self.primary_target])
        self.primary_classes_ = self.primary_encoder_.classes_
        
        self.sub_encoders_ = {}
        self.y_encodings_ = {'primary': y_primary.astype(np.int32)}
        for head in self.sub_heads:
            if head in y_seq.columns:
                le = LabelEncoder(); le.fit(y_seq[head])
                self.sub_encoders_[head] = le
                self.y_encodings_[head] = le.transform(y_seq[head]).astype(np.int32)
                
        self.log_vars_ = {head: tf.Variable(0.0, trainable=True, dtype=tf.float32, name=f"log_var_{head}") for head in self.y_encodings_}
        
        indices = np.arange(len(y_primary)); np.random.shuffle(indices)
        val_size = int(len(y_primary) * self.validation_split)
        val_indices = indices[:val_size]; train_indices = indices[val_size:]
        train_class_indices = {c: [i for i in train_indices if y_primary[i] == c] for c in range(len(self.primary_classes_))}
        
        self.encoder_ = BackboneBuilder.build(self.backbone_type, (X_arr.shape[1], X_arr.shape[2]), self.filters, self.kernels, self.pools, self.lstm_units, self.attention_heads, self.embed_dim, self.dropout, self.spatial_dropout, self.l2_reg, self.use_phase_attention, self.modality_dropout_prob)
        self.attn_proto_ = AttentionPrototypeComputation() if self.use_attention_prototypes else None
        opt = optimizers.Adam(self.learning_rate)
        steps = max(1, len(train_indices) // self.batch_size)
        self.history_ = {'train_loss': []}
        
        for epoch in range(self.epochs):
            train_loss = 0.0
            for _ in range(steps):
                sup_x, sup_y, qry_x, qry_y, sup_idx, qry_idx = self._sample_episode(X_arr, y_primary, train_class_indices)
                sup_x, _ = self.augmentor(sup_x, padding_value=self.padding_value)
                qry_x, _ = self.augmentor(qry_x, padding_value=self.padding_value)
                sup_x = tf.convert_to_tensor(sup_x, dtype=tf.float32)
                qry_x = tf.convert_to_tensor(qry_x, dtype=tf.float32)
                
                sup_y_dict, qry_y_dict, n_way_dict = {}, {}, {}
                for head, y_enc in self.y_encodings_.items():
                    s_y, q_y = y_enc[sup_idx], y_enc[qry_idx]
                    unique_classes = np.unique(np.concatenate([s_y, q_y]))
                    remap = {c: i for i, c in enumerate(unique_classes)}
                    sup_y_dict[head] = np.array([remap[c] for c in s_y], dtype=np.int32)
                    qry_y_dict[head] = np.array([remap[c] for c in q_y], dtype=np.int32)
                    n_way_dict[head] = len(unique_classes)
                    
                sup_mask = tf.reduce_any(tf.not_equal(sup_x, self.padding_value), axis=-1)
                qry_mask = tf.reduce_any(tf.not_equal(qry_x, self.padding_value), axis=-1)
                
                with tf.GradientTape() as tape:
                    sup_emb = self.encoder_([sup_x, sup_mask], training=True)
                    qry_emb = self.encoder_([qry_x, qry_mask], training=True)
                    if self.use_attention_prototypes: sup_emb = self.attn_proto_(sup_emb)
                    loss, _, _ = multitask_hybrid_prototypical_loss(sup_emb, qry_emb, sup_y_dict, qry_y_dict, n_way_dict, self.temperature, self.supcon_weight, self.log_vars_, self.uncertainty_weighting, self.fixed_loss_weights)
                
                trainable_vars = self.encoder_.trainable_variables + (self.attn_proto_.trainable_variables if self.attn_proto_ else [])
                if self.uncertainty_weighting: trainable_vars += list(self.log_vars_.values())
                grads = tape.gradient(loss, trainable_vars)
                opt.apply_gradients(zip(grads, trainable_vars))
                train_loss += loss.numpy()
            self.history_['train_loss'].append(train_loss/steps)
            if self.verbose > 0: print(f"Epoch {epoch+1}/{self.epochs} | MultiHead Loss: {train_loss/steps:.4f}")
            
        all_mask = tf.reduce_any(tf.not_equal(X_arr[train_indices], self.padding_value), axis=-1)
        all_emb = self.encoder_.predict([X_arr[train_indices], all_mask], verbose=0)
        y_train_primary = y_primary[train_indices]
        self.prototypes_ = np.array([np.mean(all_emb[y_train_primary == c], axis=0) for c in range(len(self.primary_classes_))])
        self.prototypes_ /= np.linalg.norm(self.prototypes_, axis=1, keepdims=True) + 1e-8
        return self

    def predict(self, X: dict) -> pd.Series:
        check_is_fitted(self, ["encoder_", "primary_encoder_", "prototypes_"])
        X_arr, seq_ids = self._unpack_X(X)
        mask = tf.reduce_any(tf.not_equal(X_arr, self.padding_value), axis=-1)
        emb = self.encoder_.predict([X_arr, mask], verbose=0)
        dist = np.sum((emb[:, None, :] - self.prototypes_[None, :, :]) ** 2, axis=2)
        preds = self.primary_encoder_.inverse_transform(np.argmin(dist, axis=1))
        if seq_ids is not None:
            df = pd.DataFrame({'seq_id': seq_ids, 'pred': preds})
            agg_preds = df.groupby('seq_id').agg(lambda x: x.value_counts().index[0])
            return pd.Series(agg_preds['pred'].values, index=agg_preds.index, name=self.primary_target)
        return preds