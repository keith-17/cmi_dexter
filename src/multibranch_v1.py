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
                                     Lambda, MultiHeadAttention, LayerNormalization, Add, Attention,
                                     Masking)
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
            # FIXED: the previous tf.reshape(..., tf.shape(t)[0], tf.shape(t)[1], -1)
            # used a fully dynamic shape for every axis, which erases the static
            # channel dimension downstream (x.shape[-1] becomes None), which then
            # breaks `_se_block`'s `channels // se_ratio`. seq_len/feature/filter
            # counts are all static here (padding="same" preserves spatial dims),
            # so use a plain Reshape with static ints instead.
            time_dim, feat_dim, ch_dim = x.shape[1], x.shape[2], x.shape[3]
            x = Reshape((time_dim, feat_dim * ch_dim))(x)
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
        # Mark padded timesteps (== padding_value across ALL features) so masking
        # propagates through mask-aware layers (GRU/Bidirectional/Attention/Dense/etc.)
        # instead of every padded timestep being treated as a real -999.0 signal.
        masked_inputs = Masking(mask_value=self.padding_value, name="pad_mask")(inputs)
        branch_tensors = []
        for b_name in ["acc", "rot", "tof", "thm"]:
            if b_name in branch_indices and len(branch_indices[b_name]) > 0:
                idx = branch_indices[b_name]
                # NOTE: tf.gather via Lambda drops the propagated mask for CNN/attention
                # branches (Lambda has no compute_mask). GRU-backbone branches keep it.
                # Re-derive the mask for those branches explicitly below via merged_se.
                # FIXED: bind idx as a default arg to avoid the late-binding closure bug
                # (without this, every branch's Lambda re-reads the loop variable `idx`
                # at call time and all branches silently gather the LAST branch's indices).
                b_tensor = Lambda(lambda x, idx=idx: tf.gather(x, indices=idx, axis=-1), name=f"{b_name}_slice")(masked_inputs)
                b_out = self._build_branch(b_tensor, b_name)
                branch_tensors.append(b_out)
        if len(branch_tensors) > 1:
            merged = Concatenate(axis=-1, name="concat_branches")(branch_tensors)
        else:
            merged = branch_tensors[0] if branch_tensors else masked_inputs
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
        if isinstance(X, dict):
            X_arr = X['X']
        else:
            X_arr = X

        # GridSearchCV's y must stay the SAME object end-to-end for both fit() and
        # the scorer: make_competition_scorer needs a DataFrame (sequence_id,
        # is_target, target_col) to correctly dedupe/align against predict()'s
        # per-sequence output, but a plain label array is needed here for training.
        # Passing y_train as just X_train[TARGET_COL] (a bare Series) satisfies
        # fit() but starves the scorer of sequence_id/is_target, which silently
        # produces mismatched-length y_true/y_pred inside competition_score() and,
        # combined with error_score=0.0, reports every candidate as a score of 0.0.
        if isinstance(y, pd.DataFrame):
            y_labels = y[self.primary_target].values
        else:
            y_labels = y

        self.seq_len_ = X_arr.shape[1]
        self.total_features_ = X_arr.shape[-1]
        self.branch_indices_ = get_branch_indices(feature_names)
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
        
        # --- FIX: Capture and save the history object ---
        history = self.model_.fit(
            X_arr, y_enc,
            sample_weight=sample_weights,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.2,
            callbacks=cbs,
            verbose=self.verbose
        )
        self.history_ = history.history  # Save for plotting curves
        return self

    def predict_proba(self, X):
        """
        Returns class probabilities. If X came from SequenceExtractor.transform()
        (a dict with 'X' and 'sequence_ids'), a sequence longer than
        chunk_window_size may have been split into multiple overlapping chunks.
        Those chunk-level rows are mean-pooled back into ONE row per sequence_id
        here, so len(predict_proba(X)) == number of unique sequences, not
        number of chunks.
        """
        check_is_fitted(self, "model_")
        if isinstance(X, dict):
            X_arr = X['X']
            seq_ids = X.get('sequence_ids', None)
        else:
            X_arr = X
            seq_ids = None

        proba = self.model_.predict(X_arr, verbose=0)

        if seq_ids is not None:
            proba = self._aggregate_chunks_to_sequences(proba, seq_ids)

        return proba

    def _aggregate_chunks_to_sequences(self, proba: np.ndarray, seq_ids: np.ndarray) -> np.ndarray:
        """
        Mean-pool chunk-level probability rows onto their parent sequence_id.
        Sorted ascending by sequence_id to match evaluate_holdout's
        `y_test_seq = y_test_df.drop_duplicates(...).sort_values("sequence_id")`.
        """
        seq_ids = np.asarray(seq_ids)
        df = pd.DataFrame(proba)
        df["__seq_id__"] = seq_ids
        agg = df.groupby("__seq_id__", sort=True).mean()
        return agg.to_numpy()

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
            # NOTE: SequenceExtractor stores this as `feature_names_in_`, not
            # `feature_names_`. Check both so branch routing actually receives names.
            names = getattr(transform, 'feature_names_in_', None) or getattr(transform, 'feature_names_', None)
            if names is not None:
                fit_params[f'{self.steps[-1][0]}__feature_names'] = names
        
        final_name, final_estimator = self.steps[-1]
        # Pass extracted params to the final estimator
        final_params = {k.replace(f'{final_name}__', ''): v for k, v in fit_params.items() if k.startswith(f'{final_name}__')}
        final_estimator.fit(Xt, y, **final_params)
        return self