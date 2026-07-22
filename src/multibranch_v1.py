"""
multibranch_v2.py
Flexible Multi-Branch Architecture (1DCNN / 2DCNN / Attention / GRU)
with SE-Attention Fusion, proper padding masking, and configurable
per-sensor branch routing.

Changes vs multibranch_v1.py, and why:

1. Branch routing is now CONFIGURABLE (branch_keyword_map), not hardcoded to
   CMI's acc/rot/tof/thm prefixes. Pass your own {branch_name: [substrings]}
   for a different sensor layout; defaults to the old CMI mapping so existing
   code keeps working unchanged.
2. Masking is now EXPLICIT and threaded through every layer by hand, following
   the pattern proven in utils_siamese_contrastive.py. The old approach relied
   on Keras' automatic mask propagation via a top-level Masking() layer, but
   Lambda (used for branch slicing) and Conv1D/Conv2D do NOT propagate masks
   (supports_masking=False), so the mask silently died the moment it entered
   any branch. Every conv/attention/pooling step below now re-applies the
   mask explicitly instead of hoping Keras carries it along.
3. `y` is accepted as either a bare label array OR the metadata DataFrame
   (sequence_id/is_target/target) that a competition-style CV scorer needs,
   matching the `_collapse_y`-style contract in utils_siamese_contrastive.py.
4. predict()/predict_proba() align predictions back to their sequence_id via
   an explicit map (like single_minirocket.py's `y_map.get(sid)` pattern)
   and return a pandas Series indexed by sequence_id, so downstream alignment
   with a metadata DataFrame can never silently go out of order.
5. The idx late-binding closure bug, the dynamic-reshape bug in the 2dcnn
   backbone, and the feature_names_/feature_names_in_ mismatch from v1 are
   still fixed here.
"""
from __future__ import annotations
import json
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, Conv2D, BatchNormalization, Activation,
    Concatenate, Multiply, Reshape, GlobalAveragePooling1D,
    Bidirectional, GRU, Dense, Dropout, SpatialDropout1D,
    Lambda, MultiHeadAttention, LayerNormalization, Add, Attention,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")


# Default branch routing, matching the CMI acc/rot/tof/thm sensor layout.
# Override via `branch_keyword_map=` on the classifier for a different
# sensor/channel layout instead of editing this file.
DEFAULT_BRANCH_KEYWORD_MAP: Dict[str, List[str]] = {
    "acc": ["acc", "lin_acc", "jerk", "vel", "disp", "mag", "dr_vel", "dr_pos"],
    "rot": ["rot", "quat", "euler", "ang", "6d"],
    "tof": ["tof"],
    "thm": ["thm"],
}


def get_branch_indices(feature_names, branch_keyword_map: Dict[str, List[str]] = None):
    """
    Maps feature names to branch indices using a configurable keyword map.
    First keyword group to match (in dict insertion order) wins per feature,
    matching the original if/elif precedence behaviour.
    """
    branch_keyword_map = branch_keyword_map or DEFAULT_BRANCH_KEYWORD_MAP
    config = {b: [] for b in branch_keyword_map}
    if feature_names is None:
        return config
    for i, name in enumerate(feature_names):
        name_lower = str(name).lower()
        for branch, keywords in branch_keyword_map.items():
            if any(k in name_lower for k in keywords):
                config[branch].append(i)
                break
    return config


class KerasFlexibleMultiBranchClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        backbone_type: str = "1dcnn",  # "1dcnn", "2dcnn", "attention", "gru"
        maxlen: int = 160,
        padding_value: float = -999.0,
        branch_keyword_map: Optional[Dict[str, List[str]]] = None,
        branch_filters: dict | str = None,
        branch_kernel_sizes: dict | str = None,
        attention_heads: int = 4,
        ff_dim: int = 128,
        gru_units: int = 128,
        se_ratio: int = 4,
        use_post_bigru: bool = True,
        post_gru_units: int = 128,
        dropout: float = 0.3,
        spatial_dropout: float = 0.1,
        use_batch_norm: bool = True,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 50,
        patience: int = 10,
        primary_target: str = "gesture",
        verbose: int = 0,
        random_state: int = 42
    ):
        self.backbone_type = backbone_type
        self.maxlen = maxlen
        self.padding_value = padding_value
        # NOTE: default_factory-style dict defaults are intentionally re-created
        # per-instance below rather than as a mutable default arg.
        self.branch_keyword_map = branch_keyword_map or dict(DEFAULT_BRANCH_KEYWORD_MAP)
        self.branch_filters = branch_filters or {"acc": "64-128", "rot": "64", "tof": "32", "thm": "16"}
        self.branch_kernel_sizes = branch_kernel_sizes or {"acc": "5-5", "rot": "5", "tof": "3", "thm": "3"}
        self.attention_heads = attention_heads
        self.ff_dim = ff_dim
        self.gru_units = gru_units
        self.se_ratio = se_ratio
        self.use_post_bigru = use_post_bigru
        self.post_gru_units = post_gru_units
        self.dropout = dropout
        self.spatial_dropout = spatial_dropout
        self.use_batch_norm = use_batch_norm
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.primary_target = primary_target
        self.verbose = verbose
        self.random_state = random_state

    def _parse_dict_param(self, param):
        if isinstance(param, str):
            try:
                return json.loads(param.replace("'", '"'))
            except Exception:
                return eval(param)
        return param

    # ------------------------------------------------------------------
    # Branch backbones. Every path re-applies `mask_f` (float, shape
    # (batch, seq_len, 1)) after any op that can leak signal into padded
    # timesteps, following the pattern in utils_siamese_contrastive.py.
    # `is_real` (bool, shape (batch, seq_len)) is passed to mask-aware
    # Keras layers (GRU, GlobalAveragePooling1D, Attention) directly.
    # ------------------------------------------------------------------
    def _build_branch(self, x, name, is_real, mask_f, attn_mask):
        if self.backbone_type == "1dcnn":
            filters_dict = self._parse_dict_param(self.branch_filters)
            kernels_dict = self._parse_dict_param(self.branch_kernel_sizes)
            filters_list = [int(f) for f in filters_dict.get(name, "64-128").split("-")]
            kernels_list = [int(k) for k in kernels_dict.get(name, "5-5").split("-")]
            if self.spatial_dropout > 0:
                x = SpatialDropout1D(self.spatial_dropout)(x)
            for f, k in zip(filters_list, kernels_list):
                x = Conv1D(f, k, padding="same")(x)
                # 'same' padding lets a padded neighbour leak one timestep
                # into the real region at sequence boundaries; re-zero after
                # every conv/BN so BatchNorm's running stats aren't polluted
                # by padding either.
                x = Multiply()([x, mask_f])
                if self.use_batch_norm:
                    x = BatchNormalization()(x)
                    x = Multiply()([x, mask_f])
                x = Activation("relu")(x)
            return x

        elif self.backbone_type == "2dcnn":
            x = Multiply()([x, mask_f])
            x = Lambda(lambda t: tf.expand_dims(t, axis=-1))(x)
            x = Conv2D(32, (3, 3), padding="same", activation="relu")(x)
            x = Multiply()([x, mask_f[..., None]])
            if self.use_batch_norm:
                x = BatchNormalization()(x)
                x = Multiply()([x, mask_f[..., None]])
            x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
            x = Multiply()([x, mask_f[..., None]])
            # Static Reshape (not a dynamic tf.shape() reshape) so the
            # channel dim stays known and _se_block's channels//se_ratio
            # doesn't see None.
            time_dim, feat_dim, ch_dim = x.shape[1], x.shape[2], x.shape[3]
            x = Reshape((time_dim, feat_dim * ch_dim))(x)
            x = Multiply()([x, mask_f])
            return x

        elif self.backbone_type == "attention":
            attn_out = MultiHeadAttention(
                num_heads=self.attention_heads, key_dim=x.shape[-1]
            )(x, x, attention_mask=attn_mask)
            attn_out = Multiply()([attn_out, mask_f])
            attn_out = Dropout(self.dropout)(attn_out)
            attn_out = Add()([x, attn_out])
            attn_out = LayerNormalization()(attn_out)
            attn_out = Multiply()([attn_out, mask_f])
            ff = Dense(self.ff_dim, activation="relu")(attn_out)
            ff = Dense(x.shape[-1])(ff)
            ff = Multiply()([ff, mask_f])
            ff = Dropout(self.dropout)(ff)
            out = Add()([attn_out, ff])
            out = LayerNormalization()(out)
            out = Multiply()([out, mask_f])
            return out

        elif self.backbone_type == "gru":
            # mask=is_real makes the (bidirectional) GRU skip padded
            # timesteps in both directions instead of running over -999.0.
            x = Bidirectional(GRU(self.gru_units, return_sequences=True))(x, mask=is_real)
            x = Multiply()([x, mask_f])
            return x

    def _se_block(self, x, is_real):
        channels = x.shape[-1]
        se = GlobalAveragePooling1D()(x, mask=is_real)
        se = Dense(max(1, channels // self.se_ratio), activation="relu")(se)
        se = Dense(channels, activation="sigmoid")(se)
        se = Reshape((1, channels))(se)
        return Multiply()([x, se])

    def _build_model(self, seq_len, total_features, branch_indices, num_classes):
        inputs = Input(shape=(seq_len, total_features), name="main_input")

        # Compute the padding mask ONCE from the raw input (a timestep is
        # either fully padded or fully real across ALL sensors, since
        # padding is applied per-timestep by the extractor), then thread it
        # through every branch and the fusion stage explicitly. This is the
        # utils_siamese_contrastive.py pattern: Lambda/Conv1D/Conv2D don't
        # propagate Keras' automatic mask, so a top-level Masking() layer
        # alone (v1's approach) silently stops working the moment any
        # branch slices or convolves the input.
        is_real = Lambda(
            lambda t: tf.reduce_any(tf.not_equal(t, self.padding_value), axis=-1),
            name="padding_mask",
        )(inputs)  # (batch, seq_len) bool
        mask_f = Lambda(
            lambda m: tf.cast(m, tf.float32)[..., None], name="padding_mask_float"
        )(is_real)  # (batch, seq_len, 1)
        attn_mask = Lambda(
            lambda m: m[:, :, None] & m[:, None, :],
            output_shape=lambda s: (s[1], s[1]),
            name="attention_mask",
        )(is_real)  # (batch, seq_len, seq_len) bool, for MultiHeadAttention

        branch_tensors = []
        for b_name in branch_indices:
            if len(branch_indices[b_name]) > 0:
                idx = branch_indices[b_name]
                # idx bound as a default arg to avoid the late-binding
                # closure bug: without this, every branch's Lambda re-reads
                # the loop variable at CALL time (not build time), so all
                # branches would silently gather the LAST branch's indices.
                b_tensor = Lambda(
                    lambda x, idx=idx: tf.gather(x, indices=idx, axis=-1),
                    name=f"{b_name}_slice",
                )(inputs)
                b_out = self._build_branch(b_tensor, b_name, is_real, mask_f, attn_mask)
                branch_tensors.append(b_out)

        if len(branch_tensors) > 1:
            merged = Concatenate(axis=-1, name="concat_branches")(branch_tensors)
        elif branch_tensors:
            merged = branch_tensors[0]
        else:
            # No branch matched anything in branch_keyword_map: fall back to
            # the raw (masked) input rather than silently building nothing.
            merged = Multiply()([inputs, mask_f])

        merged_se = self._se_block(merged, is_real)

        if self.use_post_bigru:
            merged_se = Bidirectional(GRU(self.post_gru_units, return_sequences=True))(
                merged_se, mask=is_real
            )
            merged_se = Multiply()([merged_se, mask_f])

        # Keras' Attention layer takes mask as [query_mask, value_mask].
        att = Attention()([merged_se, merged_se], mask=[is_real, is_real])
        att = Multiply()([att, mask_f])

        pooled = GlobalAveragePooling1D()(att, mask=is_real)
        pooled = Dropout(self.dropout)(pooled)
        out_main = Dense(num_classes, activation="softmax", name="head_primary")(pooled)

        model = Model(inputs=inputs, outputs=out_main)
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model

    # ------------------------------------------------------------------
    # y handling: accept either a bare label array, or the metadata
    # DataFrame (sequence_id/is_target/target) a competition-style CV
    # scorer needs — mirroring utils_siamese_contrastive.py's _collapse_y.
    # ------------------------------------------------------------------
    def _collapse_y(self, y):
        if isinstance(y, pd.DataFrame):
            if self.primary_target not in y.columns:
                raise ValueError(f"y DataFrame must contain target column: {self.primary_target}")
            return y[self.primary_target].values
        return np.asarray(y)

    def fit(self, X, y, feature_names=None, **kwargs):
        tf.random.set_seed(self.random_state)
        if isinstance(X, dict):
            X_arr = X['X']
        else:
            X_arr = X

        y_labels = self._collapse_y(y)

        self.seq_len_ = X_arr.shape[1]
        self.total_features_ = X_arr.shape[-1]
        self.branch_indices_ = get_branch_indices(feature_names, self.branch_keyword_map)
        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(y_labels)
        num_classes = len(self.label_encoder_.classes_)

        self.model_ = self._build_model(self.seq_len_, self.total_features_, self.branch_indices_, num_classes)

        mask = np.any(X_arr != self.padding_value, axis=-1).astype(np.float32)
        sample_weights = np.mean(mask, axis=-1)
        cbs = [
            EarlyStopping(monitor="val_loss", patience=self.patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)
        ]

        history = self.model_.fit(
            X_arr, y_enc,
            sample_weight=sample_weights,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.2,
            callbacks=cbs,
            verbose=self.verbose
        )
        self.history_ = history.history
        return self

    def predict_proba(self, X):
        """
        Returns (probs, unique_seq_ids). If X came from SequenceExtractor's
        dict output, chunk-level rows are mean-pooled back to one row per
        sequence_id (mirroring utils_siamese_contrastive.py). If X has no
        sequence_ids, returns (probs, None) at chunk/row granularity.
        """
        check_is_fitted(self, "model_")
        if isinstance(X, dict):
            X_arr = X['X']
            seq_ids = X.get('sequence_ids', None)
        else:
            X_arr = X
            seq_ids = None

        proba = self.model_.predict(X_arr, verbose=0)

        if seq_ids is not None and len(seq_ids) > 0:
            df = pd.DataFrame(proba)
            df["__seq_id__"] = np.asarray(seq_ids)
            grouped = df.groupby("__seq_id__", sort=True).mean()
            return grouped.to_numpy(), grouped.index.to_numpy()

        return proba, None

    def predict(self, X):
        """Returns a pandas Series indexed by sequence_id when available,
        so alignment against a metadata DataFrame can never silently drift
        out of order (matches single_minirocket.py's y_map.get(sid) idea:
        always resolve by explicit key, never by position)."""
        proba, seq_ids = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        preds = self.label_encoder_.inverse_transform(indices)
        if seq_ids is not None:
            return pd.Series(preds, index=seq_ids, name=self.primary_target)
        return preds


class FeatureAwarePipeline(Pipeline):
    """Custom Pipeline that passes feature_names from the Extractor to the Classifier."""
    def fit(self, X, y, **fit_params):
        Xt = X
        for name, transform in self.steps[:-1]:
            Xt = transform.fit_transform(Xt, y)
            # SequenceExtractor stores this as `feature_names_in_`, not
            # `feature_names_` — check both so branch routing actually
            # receives names instead of silently defaulting to empty branches.
            names = getattr(transform, 'feature_names_in_', None) or getattr(transform, 'feature_names_', None)
            if names is not None:
                fit_params[f'{self.steps[-1][0]}__feature_names'] = names

        final_name, final_estimator = self.steps[-1]
        final_params = {k.replace(f'{final_name}__', ''): v for k, v in fit_params.items() if k.startswith(f'{final_name}__')}
        final_estimator.fit(Xt, y, **final_params)
        return self