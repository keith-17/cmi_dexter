"""
multibranch_v2.py
Dynamic Multi-Branch Neural Network with Squeeze-and-Excitation (SE) Fusion
for multimodal spatiotemporal gesture classification.

Inspired by: proto_utils_qwen.py, utils_siamese_contrastive.py, single_minirocket.py
Compatible with: multibranch_v1.ipynb (drop-in replacement)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from typing import Tuple, Dict, Any, List, Optional, Union
import warnings
import json

warnings.filterwarnings("ignore")


# ==============================================================================
# UTILITY HELPERS
# ==============================================================================

def _parse_tuple(val: str) -> Tuple[int, ...]:
    """Parse a dash-separated string like '64-128-256' into a tuple of ints."""
    if val is None or (isinstance(val, str) and val.lower() == "none"):
        return ()
    if isinstance(val, (list, tuple)):
        return tuple(int(v) for v in val)
    return tuple(int(p) for p in str(val).split("-") if p.strip().lower() != "none")


def _parse_branch_dict(val: Union[str, dict], default: str = "64") -> Dict[str, Tuple[int, ...]]:
    """Parse branch config that may be a JSON string or dict of dash-separated values."""
    if isinstance(val, str):
        try:
            val = json.loads(val.replace("'", '"'))
        except (json.JSONDecodeError, ValueError):
            return {"acc": _parse_tuple(val), "rot": _parse_tuple(val),
                    "tof": _parse_tuple(val), "thm": _parse_tuple(val)}
    if isinstance(val, dict):
        return {k: _parse_tuple(v) for k, v in val.items()}
    return {"acc": _parse_tuple(default), "rot": _parse_tuple(default),
            "tof": _parse_tuple(default), "thm": _parse_tuple(default)}


# ==============================================================================
# SQUEEZE-AND-EXCITATION BLOCK
# ==============================================================================

class SqueezeExcitation(layers.Layer):
    """Channel-wise Squeeze-and-Excitation attention block."""

    def __init__(self, se_ratio: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.se_ratio = max(se_ratio, 1)

    def build(self, input_shape):
        channels = input_shape[-1]
        reduced = max(channels // self.se_ratio, 4)
        self.squeeze = layers.GlobalAveragePooling1D()
        self.excite = models.Sequential([
            layers.Dense(reduced, activation="relu", name="se_fc1"),
            layers.Dense(channels, activation="sigmoid", name="se_fc2"),
        ])
        super().build(input_shape)

    def call(self, x, training=None):
        se = self.squeeze(x)
        se = self.excite(se)
        return x * se[:, tf.newaxis, :]

    def get_config(self):
        config = super().get_config()
        config.update({"se_ratio": self.se_ratio})
        return config


# ==============================================================================
# BACKBONE BUILDER (per-branch heterogeneous processing)
# ==============================================================================

class BackboneBuilder:
    """
    Builds a per-branch backbone: 1D CNN, 2D CNN, Multi-Head Attention, or GRU.
    Each branch can have a different architecture.
    """

    @staticmethod
    def build(
        inp: tf.Tensor,
        backbone_type: str = "1dcnn",
        filters: Tuple[int, ...] = (64, 128),
        kernel_sizes: Tuple[int, ...] = (3, 3),
        dropout: float = 0.2,
        spatial_dropout: float = 0.1,
        l2_reg: float = 1e-4,
        use_batch_norm: bool = True,
        se_ratio: int = 4,
        lstm_units: int = 128,
        attention_heads: int = 4,
        embed_dim: int = 128,
        branch_name: str = "branch",
    ) -> tf.Tensor:
        """
        Build a backbone sub-network for one sensor modality branch.

        Parameters
        ----------
        inp : tf.Tensor
            Input tensor of shape (batch, timesteps, features).
        backbone_type : str
            One of '1dcnn', '2dcnn', 'attention', 'gru'.
        filters : tuple of int
            Filter depths for CNN layers.
        kernel_sizes : tuple of int
            Kernel sizes for CNN layers.
        dropout : float
            Dense/dropout rate.
        spatial_dropout : float
            Spatial dropout rate (zeroes entire feature channels).
        l2_reg : float
            L2 regularization strength.
        use_batch_norm : bool
            Whether to apply BatchNormalization after conv layers.
        se_ratio : int
            SE reduction ratio.
        lstm_units : int
            Units for GRU backbone.
        attention_heads : int
            Number of attention heads.
        embed_dim : int
            Embedding dimension for attention backbone.
        branch_name : str
            Name prefix for layers.

        Returns
        -------
        tf.Tensor
            Output tensor of shape (batch, timesteps, out_channels).
        """
        x = inp
        l2 = regularizers.l2(l2_reg) if l2_reg > 0 else None
        btype = backbone_type.lower().strip()

        if btype == "1dcnn":
            n_layers = max(len(filters), 1)
            for i in range(n_layers):
                f = filters[i] if i < len(filters) else filters[-1]
                k = kernel_sizes[i] if i < len(kernel_sizes) else kernel_sizes[-1]
                x = layers.Conv1D(
                    f, k, padding="same", activation=None,
                    kernel_regularizer=l2,
                    name=f"{branch_name}_conv1d_{i}"
                )(x)
                if use_batch_norm:
                    x = layers.BatchNormalization(name=f"{branch_name}_bn_{i}")(x)
                x = layers.Activation("relu", name=f"{branch_name}_relu_{i}")(x)
                if spatial_dropout > 0:
                    x = layers.SpatialDropout1D(
                        spatial_dropout, name=f"{branch_name}_spdrop_{i}"
                    )(x)
                # SE after every conv block
                x = SqueezeExcitation(se_ratio=se_ratio, name=f"{branch_name}_se_{i}")(x)

        elif btype == "2dcnn":
            # Treat (timesteps, features) as a 2D image with 1 channel
            x = layers.Reshape(
                target_shape=(-1, tf.shape(x)[-1], 1),
                name=f"{branch_name}_reshape2d"
            )(x)
            n_layers = max(len(filters), 1)
            for i in range(n_layers):
                f = filters[i] if i < len(filters) else filters[-1]
                k = kernel_sizes[i] if i < len(kernel_sizes) else kernel_sizes[-1]
                x = layers.Conv2D(
                    f, (k, k), padding="same", activation=None,
                    kernel_regularizer=l2,
                    name=f"{branch_name}_conv2d_{i}"
                )(x)
                if use_batch_norm:
                    x = layers.BatchNormalization(name=f"{branch_name}_bn2d_{i}")(x)
                x = layers.Activation("relu", name=f"{branch_name}_relu2d_{i}")(x)
                if spatial_dropout > 0:
                    x = layers.SpatialDropout2D(
                        spatial_dropout, name=f"{branch_name}_spdrop2d_{i}"
                    )(x)
            # Collapse back to (batch, timesteps, features)
            shape = tf.shape(x)
            x = layers.Reshape(
                target_shape=(shape[1], shape[2] * shape[3]),
                name=f"{branch_name}_flatten2d"
            )(x)

        elif btype == "attention":
            # Multi-Head Self-Attention backbone
            x = layers.Dense(
                embed_dim, activation="relu", kernel_regularizer=l2,
                name=f"{branch_name}_attn_proj"
            )(x)
            x = layers.MultiHeadAttention(
                num_heads=attention_heads,
                key_dim=embed_dim // attention_heads,
                dropout=dropout,
                name=f"{branch_name}_mha"
            )(x, x)
            if use_batch_norm:
                x = layers.BatchNormalization(name=f"{branch_name}_attn_bn")(x)
            x = layers.Activation("relu", name=f"{branch_name}_attn_relu")(x)
            # Add SE on attention output
            x = SqueezeExcitation(se_ratio=se_ratio, name=f"{branch_name}_attn_se")(x)

        elif btype == "gru":
            x = layers.GRU(
                lstm_units, return_sequences=True,
                kernel_regularizer=l2,
                name=f"{branch_name}_gru"
            )(x)
            if use_batch_norm:
                x = layers.BatchNormalization(name=f"{branch_name}_gru_bn")(x)
            x = layers.Activation("relu", name=f"{branch_name}_gru_relu")(x)
            if spatial_dropout > 0:
                x = layers.SpatialDropout1D(
                    spatial_dropout, name=f"{branch_name}_gru_spdrop"
                )(x)
            x = SqueezeExcitation(se_ratio=se_ratio, name=f"{branch_name}_gru_se")(x)

        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}")

        return x


# ==============================================================================
# MULTI-BRANCH MODEL BUILDER
# ==============================================================================

def build_multibranch_model(
    input_shape: Tuple[int, int],
    n_classes: int,
    branch_slices: Dict[str, slice],
    branch_backbone_types: Dict[str, str],
    branch_filters: Dict[str, Tuple[int, ...]],
    branch_kernel_sizes: Dict[str, Tuple[int, ...]],
    dropout: float = 0.2,
    spatial_dropout: float = 0.1,
    l2_reg: float = 1e-4,
    use_batch_norm: bool = True,
    se_ratio: int = 4,
    use_post_bigru: bool = True,
    post_gru_units: int = 128,
    lstm_units: int = 128,
    attention_heads: int = 4,
    embed_dim: int = 128,
    class_weight_mode: str = "balanced",
) -> tf.keras.Model:
    """
    Build the full Dynamic Multi-Branch SE-Fusion model.

    Parameters
    ----------
    input_shape : (maxlen, n_features)
    n_classes : int
    branch_slices : dict mapping modality name -> slice object
    branch_backbone_types : dict mapping modality name -> backbone type string
    branch_filters : dict mapping modality name -> tuple of filter counts
    branch_kernel_sizes : dict mapping modality name -> tuple of kernel sizes
    dropout : float
    spatial_dropout : float
    l2_reg : float
    use_batch_norm : bool
    se_ratio : int
    use_post_bigru : bool
    post_gru_units : int
    lstm_units : int
    attention_heads : int
    embed_dim : int
    class_weight_mode : str

    Returns
    -------
    tf.keras.Model
    """
    maxlen, n_features = input_shape
    inp = layers.Input(shape=(maxlen, n_features), name="sensor_input")

    # ---- Dynamic branch slicing ----
    branch_outputs = []
    active_branches = []

    for mod_name, sl in branch_slices.items():
        if sl.start >= sl.stop:
            continue
        # Slice features for this modality
        branch_inp = layers.Lambda(
            lambda x, s=sl: x[:, :, s],
            name=f"{mod_name}_slice"
        )(inp)

        btype = branch_backbone_types.get(mod_name, "1dcnn")
        bfilters = branch_filters.get(mod_name, (64, 128))
        bkernels = branch_kernel_sizes.get(mod_name, (3, 3))

        branch_out = BackboneBuilder.build(
            branch_inp,
            backbone_type=btype,
            filters=bfilters,
            kernel_sizes=bkernels,
            dropout=dropout,
            spatial_dropout=spatial_dropout,
            l2_reg=l2_reg,
            use_batch_norm=use_batch_norm,
            se_ratio=se_ratio,
            lstm_units=lstm_units,
            attention_heads=attention_heads,
            embed_dim=embed_dim,
            branch_name=mod_name,
        )
        branch_outputs.append(branch_out)
        active_branches.append(mod_name)

    # ---- SE-Gated Fusion ----
    if len(branch_outputs) == 0:
        raise ValueError("No valid branches found. Check feature_names and branch_slices.")

    if len(branch_outputs) == 1:
        fused = branch_outputs[0]
    else:
        # Concatenate all branch outputs along feature axis
        concat = layers.Concatenate(name="branch_concat")(branch_outputs)

        # Global SE gate on the fused representation
        fused = SqueezeExcitation(se_ratio=se_ratio, name="fusion_se")(concat)
    # ---- Optional Post-Fusion Bidirectional GRU ----
    if use_post_bigru and post_gru_units > 0:
        fused = layers.Bidirectional(
            layers.GRU(
                post_gru_units, return_sequences=True,
                kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None,
                name="post_bigru"
            ),
            name="post_bigru_bidir"
        )(fused)
        fused = layers.BatchNormalization(name="post_bigru_bn")(fused)
        fused = layers.Activation("relu", name="post_bigru_relu")(fused)

    # ---- Global Pooling + Classification Head ----
    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(fused)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    x = layers.Dense(
        max(n_classes * 2, 64), activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None,
        name="head_dense"
    )(x)
    x = layers.Dropout(dropout * 0.5, name="head_dropout2")(x)
    out = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = models.Model(inputs=inp, outputs=out, name="MultiBranch_SE_Fusion_v2")
    return model


# ==============================================================================
# MAIN CLASSIFIER (sklearn-compatible)
# ==============================================================================

class KerasFlexibleMultiBranchClassifier(BaseEstimator, ClassifierMixin):
    """
    Dynamic Multi-Branch Neural Network with SE Fusion.

    Drop-in replacement for multibranch_v1.KerasFlexibleMultiBranchClassifier.
    Compatible with FeatureAwarePipeline and Bayesian/Grid search.

    Parameters
    ----------
    primary_target : str
        Target column name (e.g. 'bfrb').
    branch_backbone_types : dict or str
        Per-branch backbone selection. E.g. {"acc":"1dcnn","rot":"attention","tof":"gru","thm":"1dcnn"}
        If a single string, applied to all branches.
    branch_filters : dict or str
        Per-branch filter depths. E.g. {"acc":"64-64-128","rot":"64-64-128","thm":"16","tof":"64-64"}
    branch_kernel_sizes : dict or str
        Per-branch kernel sizes. E.g. {"acc":"3-3-3","rot":"3-3-3","thm":"3","tof":"3-3"}
    dropout : float
    spatial_dropout : float
    l2_reg : float
    use_batch_norm : bool
    se_ratio : int
        SE reduction ratio (channels // se_ratio).
    use_post_bigru : bool
    post_gru_units : int
    lstm_units : int
    attention_heads : int
    embed_dim : int
    learning_rate : float
    epochs : int
    batch_size : int
    patience : int
    validation_split : float
    padding_value : float
        Value used for padding (replaced with 0 internally).
    verbose : int
    random_state : int
    """

    def __init__(
        self,
        primary_target: str = "bfrb",
        branch_backbone_types: Union[str, dict] = "1dcnn",
        branch_filters: Union[str, dict] = "64-128",
        branch_kernel_sizes: Union[str, dict] = "3-3",
        dropout: float = 0.2,
        spatial_dropout: float = 0.1,
        l2_reg: float = 1e-4,
        use_batch_norm: bool = True,
        se_ratio: int = 4,
        use_post_bigru: bool = True,
        post_gru_units: int = 128,
        lstm_units: int = 128,
        attention_heads: int = 4,
        embed_dim: int = 128,
        learning_rate: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 15,
        validation_split: float = 0.15,
        padding_value: float = -999.0,
        verbose: int = 1,
        random_state: int = 42,
    ):
        self.primary_target = primary_target
        self.branch_backbone_types = branch_backbone_types
        self.branch_filters = branch_filters
        self.branch_kernel_sizes = branch_kernel_sizes
        self.dropout = dropout
        self.spatial_dropout = spatial_dropout
        self.l2_reg = l2_reg
        self.use_batch_norm = use_batch_norm
        self.se_ratio = se_ratio
        self.use_post_bigru = use_post_bigru
        self.post_gru_units = post_gru_units
        self.lstm_units = lstm_units
        self.attention_heads = attention_heads
        self.embed_dim = embed_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.validation_split = validation_split
        self.padding_value = padding_value
        self.verbose = verbose
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Feature name → branch slice mapping
    # ------------------------------------------------------------------
    MODALITY_PREFIXES = {
        "acc": ["acc_", "accelerometer_", "vel_", "velocity_", "disp_", "displacement_",
                "jerk_", "mag_acc", "mag_vel", "mag_disp", "mag_jerk"],
        "rot": ["rot_", "rotation_", "quat_", "quaternion_", "euler_", "gyro_rot"],
        "tof": ["tof_", "time_of_flight_", "distance_"],
        "thm": ["thm_", "thermopile_", "temp_"],
    }

    def _resolve_branch_slices(
        self, feature_names: Optional[List[str]], n_features: int
    ) -> Dict[str, slice]:
        """
        Map feature_names to per-modality slices.
        Falls back to equal quarter-splitting if names are unavailable.
        """
        if feature_names is None or len(feature_names) == 0:
            # Equal split fallback
            q = n_features // 4
            return {
                "acc": slice(0, q),
                "rot": slice(q, 2 * q),
                "tof": slice(2 * q, 3 * q),
                "thm": slice(3 * q, n_features),
            }

        slices = {}
        assigned = set()
        for mod, prefixes in self.MODALITY_PREFIXES.items():
            indices = [
                i for i, fn in enumerate(feature_names)
                if any(fn.lower().startswith(p) for p in prefixes)
            ]
            if indices:
                slices[mod] = slice(min(indices), max(indices) + 1)
                assigned.update(indices)

        # Assign remaining features to 'acc' (largest modality typically)
        remaining = [i for i in range(n_features) if i not in assigned]
        if remaining:
            if "acc" in slices:
                old = slices["acc"]
                slices["acc"] = slice(min(old.start, min(remaining)),
                                      max(old.stop, max(remaining) + 1))
            else:
                slices["acc"] = slice(min(remaining), max(remaining) + 1)

        # Ensure all 4 modalities exist (empty slice if missing)
        for mod in ["acc", "rot", "tof", "thm"]:
            if mod not in slices:
                slices[mod] = slice(0, 0)

        return slices

    def _resolve_backbone_types(self) -> Dict[str, str]:
        """Parse backbone type config into per-branch dict."""
        if isinstance(self.branch_backbone_types, str):
            try:
                d = json.loads(self.branch_backbone_types.replace("'", '"'))
                if isinstance(d, dict):
                    return d
            except (json.JSONDecodeError, ValueError):
                pass
            return {"acc": self.branch_backbone_types, "rot": self.branch_backbone_types,
                    "tof": self.branch_backbone_types, "thm": self.branch_backbone_types}
        if isinstance(self.branch_backbone_types, dict):
            return self.branch_backbone_types
        return {"acc": "1dcnn", "rot": "1dcnn", "tof": "1dcnn", "thm": "1dcnn"}

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    def _prepare_data(self, X: np.ndarray, y: pd.DataFrame):
        """
        Prepare sequence-level labels and clean the 3D array.
        Returns X_clean (n_seq, maxlen, n_feat), y_seq (DataFrame), y_enc (ndarray).
        """
        # Handle padding
        X_clean = np.where(X == self.padding_value, 0.0, X.copy())
        X_clean = np.nan_to_num(X_clean, nan=0.0, posinf=0.0, neginf=0.0)

        # Sequence-level labels
        if isinstance(y, pd.DataFrame) and "sequence_id" in y.columns:
            y_seq = y.drop_duplicates("sequence_id").sort_values("sequence_id").reset_index(drop=True)
        elif isinstance(y, pd.DataFrame):
            y_seq = y.reset_index(drop=True)
        else:
            y_seq = pd.DataFrame({self.primary_target: np.asarray(y)})

        # Ensure alignment: X has n_seq rows
        n_seq = X_clean.shape[0]
        if len(y_seq) != n_seq:
            # Truncate or pad labels to match
            y_seq = y_seq.iloc[:n_seq].reset_index(drop=True)

        # Encode target
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y_seq[self.primary_target].values)
        self.classes_ = self.le_.classes_

        return X_clean, y_seq, y_enc

    # ------------------------------------------------------------------
    # FIT
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: pd.DataFrame, feature_names: Optional[List[str]] = None, **kwargs):
        """
        Fit the multi-branch SE-fusion classifier.

        Parameters
        ----------
        X : np.ndarray, shape (n_sequences, maxlen, n_features)
        y : pd.DataFrame with at least the target column
        feature_names : list of str, optional
            Feature column names from SequenceExtractor.
        """
        tf.keras.backend.clear_session()
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)

        X_clean, y_seq, y_enc = self._prepare_data(X, y)
        n_seq, maxlen, n_features = X_clean.shape
        n_classes = len(self.classes_)

        # Resolve branch architecture
        self.branch_slices_ = self._resolve_branch_slices(feature_names, n_features)
        backbone_types = self._resolve_backbone_types()
        b_filters = _parse_branch_dict(self.branch_filters, "64-128")
        b_kernels = _parse_branch_dict(self.branch_kernel_sizes, "3-3")

        # Build model
        self.model_ = build_multibranch_model(
            input_shape=(maxlen, n_features),
            n_classes=n_classes,
            branch_slices=self.branch_slices_,
            branch_backbone_types=backbone_types,
            branch_filters=b_filters,
            branch_kernel_sizes=b_kernels,
            dropout=self.dropout,
            spatial_dropout=self.spatial_dropout,
            l2_reg=self.l2_reg,
            use_batch_norm=self.use_batch_norm,
            se_ratio=self.se_ratio,
            use_post_bigru=self.use_post_bigru,
            post_gru_units=self.post_gru_units,
            lstm_units=self.lstm_units,
            attention_heads=self.attention_heads,
            embed_dim=self.embed_dim,
        )

        # Compile
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        self.model_.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        if self.verbose > 0:
            self.model_.summary(print_fn=lambda s: None)  # suppress in CV
            print(f"[MultiBranch_v2] Branches: {list(self.branch_slices_.keys())}")
            print(f"[MultiBranch_v2] Slices: { {k: (v.start, v.stop) for k, v in self.branch_slices_.items()} }")
            print(f"[MultiBranch_v2] Backbones: {backbone_types}")
            print(f"[MultiBranch_v2] Classes ({n_classes}): {list(self.classes_)}")

        # Compute sample weights from padding mask (down-weight padded timesteps)
        mask = (X_clean != 0.0).any(axis=-1).astype(np.float32)  # (n_seq, maxlen)
        sample_weights = np.clip(np.mean(mask, axis=-1), 0.1, 1.0)

        # Callbacks
        cbs = [
            callbacks.EarlyStopping(
                monitor="val_loss", patience=self.patience,
                restore_best_weights=True, verbose=self.verbose
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=max(self.patience // 3, 3),
                min_lr=1e-6, verbose=self.verbose
            ),
        ]

        # Fit
        self.model_.fit(
            X_clean, y_enc,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=cbs,
            sample_weight=sample_weights,
            verbose=self.verbose,
        )

        # Store prototypes (class centroids in embedding space) for potential use
        self._prototypes = np.array([
            X_clean[y_enc == c].mean(axis=0) for c in range(n_classes)
        ])

        return self

    # ------------------------------------------------------------------
    # PREDICT
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        check_is_fitted(self, ["model_", "le_", "classes_"])
        X_clean = np.where(X == self.padding_value, 0.0, X.copy())
        X_clean = np.nan_to_num(X_clean, nan=0.0, posinf=0.0, neginf=0.0)
        probs = self.model_.predict(X_clean, verbose=0)
        pred_idx = np.argmax(probs, axis=1)
        return self.le_.inverse_transform(pred_idx)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        check_is_fitted(self, ["model_", "le_", "classes_"])
        X_clean = np.where(X == self.padding_value, 0.0, X.copy())
        X_clean = np.nan_to_num(X_clean, nan=0.0, posinf=0.0, neginf=0.0)
        return self.model_.predict(X_clean, verbose=0)

    # ------------------------------------------------------------------
    # SKLEARN COMPATIBILITY
    # ------------------------------------------------------------------
    def get_params(self, deep=True):
        return {
            "primary_target": self.primary_target,
            "branch_backbone_types": self.branch_backbone_types,
            "branch_filters": self.branch_filters,
            "branch_kernel_sizes": self.branch_kernel_sizes,
            "dropout": self.dropout,
            "spatial_dropout": self.spatial_dropout,
            "l2_reg": self.l2_reg,
            "use_batch_norm": self.use_batch_norm,
            "se_ratio": self.se_ratio,
            "use_post_bigru": self.use_post_bigru,
            "post_gru_units": self.post_gru_units,
            "lstm_units": self.lstm_units,
            "attention_heads": self.attention_heads,
            "embed_dim": self.embed_dim,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "validation_split": self.validation_split,
            "padding_value": self.padding_value,
            "verbose": self.verbose,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


# ==============================================================================
# FEATURE-AWARE PIPELINE (passes feature_names from extractor → classifier)
# ==============================================================================

class FeatureAwarePipeline:
    """
    A lightweight pipeline that:
      1. Fits/transforms via SequenceExtractor (or any transformer).
      2. Passes `feature_names` from the transformer to the classifier's fit().

    Compatible with sklearn's cross_val_score / GridSearchCV / BayesSearchCV
    via duck-typing (get_params, set_params, fit, predict, score).
    """

    def __init__(self, steps: List[Tuple[str, Any]]):
        self.steps = steps
        self._validate_steps()

    def _validate_steps(self):
        names = [name for name, _ in self.steps]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate step names in pipeline.")

    @property
    def named_steps(self) -> Dict[str, Any]:
        return {name: est for name, est in self.steps}

    def fit(self, X, y, **fit_params):
        Xt = X
        for i, (name, est) in enumerate(self.steps[:-1]):
            Xt = est.fit_transform(Xt, y)

        # Final estimator (classifier)
        clf_name, clf = self.steps[-1]
        # Try to extract feature_names from the last transformer
        feature_names = None
        if len(self.steps) > 1:
            last_transformer = self.steps[-2][1]
            if hasattr(last_transformer, "feature_names_"):
                feature_names = last_transformer.feature_names_
            elif hasattr(last_transformer, "get_feature_names_out"):
                try:
                    feature_names = list(last_transformer.get_feature_names_out())
                except Exception:
                    pass

        clf.fit(Xt, y, feature_names=feature_names)
        return self

    def predict(self, X):
        Xt = X
        for name, est in self.steps[:-1]:
            Xt = est.transform(Xt)
        return self.steps[-1][1].predict(Xt)

    def predict_proba(self, X):
        Xt = X
        for name, est in self.steps[:-1]:
            Xt = est.transform(Xt)
        return self.steps[-1][1].predict_proba(Xt)

    def score(self, X, y, **kwargs):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))

    def get_params(self, deep=True):
        params = {}
        for name, est in self.steps:
            params[name] = est
            if deep and hasattr(est, "get_params"):
                for k, v in est.get_params(deep=True).items():
                    params[f"{name}__{k}"] = v
        return params

    def set_params(self, **params):
        step_params = {}
        for key, val in params.items():
            if "__" in key:
                step_name, param_name = key.split("__", 1)
                step_params.setdefault(step_name, {})[param_name] = val
            else:
                # Direct step replacement
                for i, (name, _) in enumerate(self.steps):
                    if name == key:
                        self.steps[i] = (name, val)
                        break

        for step_name, p_dict in step_params.items():
            for name, est in self.steps:
                if name == step_name and hasattr(est, "set_params"):
                    est.set_params(**p_dict)
        return self


# ==============================================================================
# BAYESIAN / GRID SEARCH SPACE (for use with skopt / optuna)
# ==============================================================================

def prepare_bayesian_space_v2() -> Dict[str, Any]:
    """
    Returns a parameter grid/space compatible with BayesSearchCV or GridSearchCV.
    Prefixes: 'extractor__' for SequenceExtractor, 'classifier__' for the model.
    """
    from skopt.space import Real, Integer, Categorical

    space = {
        # ---- Extractor (signal processing) ----
        "extractor__acc_modes": Categorical([
            "raw|velocity|jerk",
            "smoothed|velocity|jerk",
            "raw|velocity|displacement|jerk",
        ]),
        "extractor__rotation_modes": Categorical(["quaternion", "quaternion|euler"]),
        "extractor__motion_filter_mode": Categorical([None, "extended_kalman"]),
        "extractor__chunk_window_size": Integer(120, 256),
        "extractor__chunk_stride": Integer(60, 160),
        "extractor__imu_target_sampling_rate": Categorical([50, 100]),

        # ---- Classifier (topology) ----
        "classifier__branch_backbone_types": Categorical([
            '{"acc":"1dcnn","rot":"1dcnn","tof":"1dcnn","thm":"1dcnn"}',
            '{"acc":"1dcnn","rot":"attention","tof":"gru","thm":"1dcnn"}',
            '{"acc":"1dcnn","rot":"1dcnn","tof":"1dcnn","thm":"gru"}',
        ]),
        "classifier__branch_filters": Categorical([
            '{"acc":"64-128","rot":"64-128","thm":"32","tof":"64-64"}',
            '{"acc":"64-64-128","rot":"64-64-128","thm":"16","tof":"64-64"}',
            '{"acc":"128-256","rot":"64-128","thm":"32-64","tof":"64-128"}',
        ]),
        "classifier__branch_kernel_sizes": Categorical([
            '{"acc":"3-3","rot":"3-3","thm":"3","tof":"3-3"}',
            '{"acc":"3-3-3","rot":"3-3-3","thm":"3","tof":"3-3"}',
            '{"acc":"5-3","rot":"3-3","thm":"5","tof":"3-3"}',
        ]),
        "classifier__dropout": Real(0.0, 0.4, prior="uniform"),
        "classifier__spatial_dropout": Real(0.0, 0.3, prior="uniform"),
        "classifier__l2_reg": Real(1e-5, 1e-3, prior="log-uniform"),
        "classifier__se_ratio": Integer(2, 8),
        "classifier__use_batch_norm": Categorical([True, False]),
        "classifier__use_post_bigru": Categorical([True, False]),
        "classifier__post_gru_units": Integer(64, 256),
        "classifier__learning_rate": Real(1e-4, 1e-2, prior="log-uniform"),
        "classifier__epochs": Integer(60, 150),
        "classifier__patience": Integer(8, 20),
        "classifier__batch_size": Categorical([16, 32, 64]),
    }
    return space


# ==============================================================================
# CONVENIENCE: Quick evaluation helper (mirrors base_utils_qwen.evaluate_holdout)
# ==============================================================================

def evaluate_multibranch(
    y_true_df: pd.DataFrame,
    y_pred: np.ndarray,
    target_col: str = "bfrb",
    verbose: bool = True,
) -> dict:
    """
    Evaluate predictions using the competition metric:
    (Binary F1 + Macro Gesture F1) / 2.
    """
    from sklearn.metrics import f1_score, classification_report

    if isinstance(y_true_df, pd.DataFrame) and "sequence_id" in y_true_df.columns:
        y_seq = (
            y_true_df.drop_duplicates("sequence_id")
            .sort_values("sequence_id")
            .reset_index(drop=True)
        )
    else:
        y_seq = y_true_df.reset_index(drop=True) if isinstance(y_true_df, pd.DataFrame) else pd.DataFrame(y_true_df)

    y_true_binary = y_seq["is_target"].values.astype(int) if "is_target" in y_seq.columns \
        else (y_seq[target_col].values != "non_bfrb").astype(int)
    y_pred = np.asarray(y_pred)
    y_pred_binary = (y_pred != "non_bfrb").astype(int)

    binary_f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    target_mask = y_true_binary == 1

    if target_mask.sum() > 0:
        gesture_f1 = f1_score(
            y_seq.loc[target_mask, target_col].values,
            y_pred[target_mask],
            average="macro",
            zero_division=0,
        )
    else:
        gesture_f1 = 0.0

    comp_score = (binary_f1 + gesture_f1) / 2.0

    if verbose:
        print("\n" + "=" * 60)
        print("MULTI-BRANCH v2 EVALUATION")
        print("=" * 60)
        print(f"  Binary F1 (non_bfrb vs bfrb): {binary_f1:.4f}")
        print(f"  BFRB Gesture Macro F1:        {gesture_f1:.4f}")
        print(f"  COMPETITION SCORE:            {comp_score:.4f}")
        if target_mask.sum() > 0:
            print("\n" + "-" * 40)
            print("  Gesture Classification Report")
            print("-" * 40)
            print(classification_report(
                y_seq.loc[target_mask, target_col].values,
                y_pred[target_mask],
                zero_division=0,
            ))

    return {
        "binary_f1": binary_f1,
        "gesture_f1": gesture_f1,
        "competition_score": comp_score,
    }