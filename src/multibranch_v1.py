"""
sota_multibranch_utils.py
Flexible Multi-Branch Architecture (1DCNN / 2DCNN / Attention / GRU) 
with SE-Attention Fusion and Multi-Task Heads.
Designed to integrate seamlessly with base_utils_qwen.SequenceExtractor.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv1D, Conv2D, BatchNormalization, Activation, 
                                     Concatenate, Multiply, Reshape, GlobalAveragePooling1D, 
                                     Bidirectional, GRU, Dense, Dropout, SpatialDropout1D, 
                                     Lambda, MultiHeadAttention, LayerNormalization, Add, Attention)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline
import json
import warnings
warnings.filterwarnings("ignore")

def get_branch_indices(feature_names):
    """Dynamically maps feature names to branch indices based on physical prefixes."""
    config = {"acc": [], "rot": [], "tof": [], "thm": []}
    if feature_names is None: return config
    for i, name in enumerate(feature_names):
        name_lower = str(name).lower()
        if any(k in name_lower for k in ["acc", "lin_acc", "jerk", "vel", "disp", "mag", "dr_vel", "dr_pos"]):
            config["acc"].append(i)
        elif any(k in name_lower for k in ["rot", "quat", "euler", "ang", "6d"]):
            config["rot"].append(i)
        elif "tof" in name_lower:
            config["tof"].append(i)
        elif "thm" in name_lower:
            config["thm"].append(i)
    return config

class KerasFlexibleMultiBranchClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        backbone_type: str = "1dcnn", # "1dcnn", "2dcnn", "attention", "gru"
        maxlen: int = 160,
        padding_value: float = -999.0,
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
            try: return json.loads(param.replace("'", '"'))
            except: return eval(param)
        return param

    def _build_branch(self, x, name):
        if self.backbone_type == "1dcnn":
            filters_dict = self._parse_dict_param(self.branch_filters)
            kernels_dict = self._parse_dict_param(self.branch_kernel_sizes)
            filters_list = [int(f) for f in filters_dict.get(name, "64-128").split("-")]
            kernels_list = [int(k) for k in kernels_dict.get(name, "5-5").split("-")]
            if self.spatial_dropout > 0:
                x = SpatialDropout1D(self.spatial_dropout)(x)
            for f, k in zip(filters_list, kernels_list):
                x = Conv1D(f, k, padding="same")(x)
                if self.use_batch_norm: x = BatchNormalization()(x)
                x = Activation("relu")(x)
            return x
        elif self.backbone_type == "2dcnn":
            # Reshape (Batch, Time, Channels) -> (Batch, Time, Channels, 1)
            x = Lambda(lambda t: tf.expand_dims(t, axis=-1))(x)
            x = Conv2D(32, (3, 3), padding="same", activation="relu")(x)
            if self.use_batch_norm: x = BatchNormalization()(x)
            x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
            # FIXED: Use dynamic shape (tf.shape(t)[1]) instead of hardcoded self.maxlen
            x = Lambda(lambda t: tf.reshape(t, (tf.shape(t)[0], tf.shape(t)[1], -1)))(x)
            return x
        elif self.backbone_type == "attention":
            attn_out = MultiHeadAttention(num_heads=self.attention_heads, key_dim=x.shape[-1])(x, x)
            attn_out = Dropout(self.dropout)(attn_out)
            attn_out = Add()([x, attn_out])
            attn_out = LayerNormalization()(attn_out)
            ff = Dense(self.ff_dim, activation="relu")(attn_out)
            ff = Dense(x.shape[-1])(ff)
            ff = Dropout(self.dropout)(ff)
            out = Add()([attn_out, ff])
            out = LayerNormalization()(out)
            return out
        elif self.backbone_type == "gru":
            x = GRU(self.gru_units, return_sequences=True)(x)
            return x

    def _se_block(self, x, name):
        channels = x.shape[-1]
        se = GlobalAveragePooling1D()(x)
        se = Dense(max(1, channels // self.se_ratio), activation="relu")(se)
        se = Dense(channels, activation="sigmoid")(se)
        se = Reshape((1, channels))(se)
        return Multiply()([x, se])

    def _build_model(self, seq_len, total_features, branch_indices, num_classes):
        # FIXED: Use dynamic seq_len instead of hardcoded self.maxlen
        inputs = Input(shape=(seq_len, total_features), name="main_input")
        branch_tensors = []
        for b_name in ["acc", "rot", "tof", "thm"]:
            if b_name in branch_indices and len(branch_indices[b_name]) > 0:
                idx = branch_indices[b_name]
                b_tensor = Lambda(lambda x: tf.gather(x, indices=idx, axis=-1), name=f"{b_name}_slice")(inputs)
                b_out = self._build_branch(b_tensor, b_name)
                branch_tensors.append(b_out)
        if len(branch_tensors) > 1:
            merged = Concatenate(axis=-1, name="concat_branches")(branch_tensors)
        else:
            merged = branch_tensors[0] if branch_tensors else inputs
        merged_se = self._se_block(merged, name="fusion")
        if self.use_post_bigru:
            merged_se = Bidirectional(GRU(self.post_gru_units, return_sequences=True))(merged_se)
        att = Attention()([merged_se, merged_se])
        pooled = GlobalAveragePooling1D()(att)
        pooled = Dropout(self.dropout)(pooled)
        out_main = Dense(num_classes, activation="softmax", name="head_primary")(pooled)
        model = Model(inputs=inputs, outputs=out_main)
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def fit(self, X, y, feature_names=None, **kwargs):
        tf.random.set_seed(self.random_state)
        # Handle dict output from SequenceExtractor
        if isinstance(X, dict):
            X_arr = X['X']
        else:
            X_arr = X
        
        # FIXED: Dynamically infer sequence length from the actual input data
        self.seq_len_ = X_arr.shape[1]
        self.total_features_ = X_arr.shape[-1]
        self.branch_indices_ = get_branch_indices(feature_names)
        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(y)
        num_classes = len(self.label_encoder_.classes_)
        
        # Pass the dynamically inferred seq_len_ to _build_model
        self.model_ = self._build_model(self.seq_len_, self.total_features_, self.branch_indices_, num_classes)
        
        # Masking for Padding
        mask = np.any(X_arr != self.padding_value, axis=-1).astype(np.float32)
        sample_weights = np.mean(mask, axis=-1)
        cbs = [
            EarlyStopping(monitor="val_loss", patience=self.patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)
        ]
        self.model_.fit(
            X_arr, y_enc,
            sample_weight=sample_weights,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.2,
            callbacks=cbs,
            verbose=self.verbose
        )
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        if isinstance(X, dict): X_arr = X['X']
        else: X_arr = X
        return self.model_.predict(X_arr, verbose=0)

    def predict(self, X):
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.label_encoder_.inverse_transform(indices)

class FeatureAwarePipeline(Pipeline):
    """Custom Pipeline that passes feature_names from the Extractor to the Classifier."""
    def fit(self, X, y, **fit_params):
        Xt = X
        for name, transform in self.steps[:-1]:
            Xt = transform.fit_transform(Xt, y)
            if hasattr(transform, 'feature_names_'):
                fit_params[f'{self.steps[-1][0]}__feature_names'] = transform.feature_names_
        
        final_name, final_estimator = self.steps[-1]
        # Pass extracted params to the final estimator
        final_params = {k.replace(f'{final_name}__', ''): v for k, v in fit_params.items() if k.startswith(f'{final_name}__')}
        final_estimator.fit(Xt, y, **final_params)
        return self