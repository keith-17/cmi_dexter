"""
multibranch_utils.py

Multi-branch (modality-specific) Keras classifier with every architecture,
optimizer, loss, and scheduler knob exposed for search.
"""

from __future__ import annotations

import json
import hashlib
import math
import re
import time
import traceback
import warnings
from collections import Counter
from contextlib import contextmanager
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

if __package__:
    from .base_utils import (
        GASearchCV,
        InvalidExtractorParams,
        SequenceExtractor,
        SensorAugmentor,
        inactive_sequence_parameters,
        sequence_feature_families,
        _coerce_augmentation_names,
        competition_score,
        prepare_bayesian_space,
        prepare_evolutionary_space,
        prepare_multitask_param_space,
        validate_sequence_extractor_params,
    )
else:
    from base_utils import (
        GASearchCV,
        InvalidExtractorParams,
        SequenceExtractor,
        SensorAugmentor,
        inactive_sequence_parameters,
        sequence_feature_families,
        _coerce_augmentation_names,
        competition_score,
        prepare_bayesian_space,
        prepare_evolutionary_space,
        prepare_multitask_param_space,
        validate_sequence_extractor_params,
    )

try:
    import tensorflow as tf
    from tensorflow.keras import callbacks, layers, models, optimizers, losses

    TENSORFLOW_AVAILABLE = True
except Exception:
    tf = None
    layers = models = optimizers = callbacks = losses = None
    TENSORFLOW_AVAILABLE = False


BRANCH_ORDER = ["acc", "derived_imu", "rotation", "tof", "thermo", "stft", "cwt"]
TIME_SERIES_BRANCHES = {"acc", "derived_imu", "rotation", "tof", "thermo"}
STATIC_BRANCHES = {"stft", "cwt"}

DEFAULT_BRANCH_FILTERS = {"acc": 64, "rotation": 64, "tof": 32, "thermo": 32, "stft": 32, "cwt": 32}
DEFAULT_BRANCH_KERNEL_SIZES = {"acc": 5, "rotation": 5, "tof": 3, "thermo": 3, "stft": 3, "cwt": 3}
DEFAULT_BRANCH_POOL_SIZES = {"acc": 2, "rotation": 2, "tof": 2, "thermo": 2, "stft": 2, "cwt": 2}


def _normalise_aux_heads(specs):
    """Validate serialisable sequence/window head declarations without mutating parameters."""
    specs = _json_maybe(specs)
    if specs is None:
        return []
    if not isinstance(specs, (list, tuple)):
        raise ValueError("custom_aux_heads must be a list of head dictionaries (or JSON).")
    result, names = [], {"predictions", "orientation_aux", "phase_aux"}
    allowed = {"name", "target", "task", "hidden_units", "activation", "dropout", "loss_weight", "window_size", "num_classes"}
    for raw in specs:
        if not isinstance(raw, dict) or set(raw) - allowed:
            raise ValueError("Unknown custom auxiliary head settings.")
        spec = dict(raw)
        name = spec.get("name", "")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) or name in names:
            raise ValueError("Auxiliary head names must be unique identifiers and not reserved.")
        names.add(name)
        if not isinstance(spec.get("target"), str) or not spec["target"]:
            raise ValueError(f"Auxiliary head {name} requires a target column.")
        spec.setdefault("task", "classification")
        if spec["task"] not in {"classification", "regression"}:
            raise ValueError("Auxiliary task must be classification or regression.")
        spec.setdefault("hidden_units", [32])
        if not isinstance(spec["hidden_units"], (list, tuple)) or any(
            isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1
            for v in spec["hidden_units"]
        ):
            raise ValueError("hidden_units must contain positive integers.")
        spec.setdefault("activation", "relu")
        spec.setdefault("dropout", 0.0)
        spec.setdefault("loss_weight", 1.0)
        if not np.isfinite(spec["dropout"]) or not 0 <= spec["dropout"] < 1:
            raise ValueError("Auxiliary dropout must be in [0, 1).")
        if not np.isfinite(spec["loss_weight"]) or spec["loss_weight"] < 0:
            raise ValueError("Auxiliary loss_weight must be finite and nonnegative.")
        window = spec.get("window_size")
        if window is not None and (isinstance(window, bool) or not isinstance(window, (int, np.integer)) or window < 1):
            raise ValueError("Auxiliary window_size must be a positive integer.")
        result.append(spec)
    return result


def _attention_pool_over_time(name: str, x):
    attention_logits = layers.Dense(1, activation="tanh", name=f"{name}_attention_score")(x)
    attention_logits = layers.Reshape((-1,), name=f"{name}_attention_flat")(attention_logits)
    attention_weights = layers.Softmax(name=f"{name}_attention_weights")(attention_logits)
    attention_weights = layers.Reshape((-1, 1), name=f"{name}_attention_reshape")(attention_weights)
    pooled = layers.Multiply(name=f"{name}_attention_pool")([x, attention_weights])
    return layers.Lambda(lambda t: tf.reduce_sum(t, axis=1), name=f"{name}_attention_sum")(pooled)


def _mean_pool_over_time(name: str, x):
    return layers.GlobalAveragePooling1D(name=f"{name}_mean_pool")(x)


def _max_pool_over_time(name: str, x):
    return layers.GlobalMaxPooling1D(name=f"{name}_max_pool")(x)


def _last_pool_over_time(name: str, x):
    return layers.Lambda(lambda t: t[:, -1, :], name=f"{name}_last_pool")(x)


def _masked_attention_pool(name: str, x, mask, units: int, attention_type: str):
    attention_type = str(attention_type).lower()
    if attention_type not in {"additive", "linear"}:
        raise ValueError("attention_type must be 'additive' or 'linear'.")
    if attention_type == "additive":
        logits = layers.Dense(units, activation="tanh", name=f"{name}_attention_hidden")(x)
        logits = layers.Dense(1, name=f"{name}_attention_logits")(logits)
    else:
        logits = layers.Dense(1, use_bias=False, name=f"{name}_attention_logits")(x)
    logits = layers.Lambda(
        lambda value: tf.squeeze(value, axis=-1),
        name=f"{name}_attention_flat",
    )(logits)
    logits = layers.Lambda(
        lambda values: tf.where(
            values[1] > 0,
            values[0],
            tf.cast(-1e9, values[0].dtype),
        ),
        name=f"{name}_attention_mask",
    )([logits, mask])
    weights = layers.Softmax(axis=1, name=f"{name}_attention_weights")(logits)
    weights = layers.Reshape((-1, 1), name=f"{name}_attention_reshape")(weights)
    return layers.Lambda(
        lambda values: tf.reduce_sum(values[0] * values[1], axis=1),
        name=f"{name}_attention_pool",
    )([x, weights])


def _positional_encoding(length: int, width: int) -> np.ndarray:
    positions = np.arange(length, dtype=np.float32)[:, None]
    dimensions = np.arange(width, dtype=np.float32)[None, :]
    angles = positions / np.power(10000.0, (2.0 * np.floor(dimensions / 2.0)) / width)
    encoding = np.where(dimensions.astype(int) % 2 == 0, np.sin(angles), np.cos(angles))
    return encoding[None, :, :].astype(np.float32)


def _transformer_encoder_block(
    x,
    mask,
    name: str,
    d_model: int,
    num_heads: int,
    ff_dim: int,
    dropout: float,
    attention_dropout: float,
    activation: str,
):
    normalized = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_attn_norm")(x)
    attention_mask = layers.Lambda(
        lambda value: tf.cast(value[:, None, :], tf.bool),
        name=f"{name}_attention_mask",
    )(mask)
    attended = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=attention_dropout,
        name=f"{name}_self_attention",
    )(normalized, normalized, attention_mask=attention_mask)
    attended = layers.Dropout(dropout, name=f"{name}_attn_dropout")(attended)
    x = layers.Add(name=f"{name}_attn_residual")([x, attended])

    normalized = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ffn_norm")(x)
    feed_forward = layers.Dense(ff_dim, activation=activation, name=f"{name}_ffn_expand")(normalized)
    feed_forward = layers.Dropout(dropout, name=f"{name}_ffn_dropout1")(feed_forward)
    feed_forward = layers.Dense(d_model, name=f"{name}_ffn_project")(feed_forward)
    feed_forward = layers.Dropout(dropout, name=f"{name}_ffn_dropout2")(feed_forward)
    return layers.Add(name=f"{name}_ffn_residual")([x, feed_forward])


def _phase_attention_pool(
    x,
    mask,
    n_phases: int,
    attention_units: int,
    temperature: float,
    fusion_mode: str,
):
    phase_logits = layers.Dense(n_phases, name="phase_logits")(x)
    phase_probabilities = layers.Softmax(axis=-1, name="phase_probabilities")(phase_logits)
    expanded_mask = layers.Reshape((-1, 1), name="phase_mask_expand")(mask)
    phase_probabilities = layers.Multiply(name="phase_masked_probabilities")(
        [phase_probabilities, expanded_mask]
    )
    attention_logits = layers.Dense(
        attention_units, activation="tanh", name="phase_attention_hidden"
    )(x)
    attention_logits = layers.Dense(n_phases, name="phase_attention_logits")(attention_logits)
    attention_logits = layers.Lambda(
        lambda value: value / temperature,
        name="phase_attention_temperature",
    )(attention_logits)
    attention_logits = layers.Permute((2, 1), name="phase_attention_transpose")(attention_logits)
    repeated_mask = layers.Lambda(
        lambda value: tf.repeat(value[:, None, :], repeats=n_phases, axis=1),
        name="phase_attention_mask",
    )(mask)
    attention_logits = layers.Lambda(
        lambda values: tf.where(
            values[1] > 0,
            values[0],
            tf.cast(-1e9, values[0].dtype),
        ),
        name="phase_attention_masked_logits",
    )([attention_logits, repeated_mask])
    attention_weights = layers.Softmax(axis=-1, name="phase_temporal_attention")(attention_logits)
    attention_weights = layers.Permute((2, 1), name="phase_attention_untranspose")(attention_weights)
    weights = layers.Multiply(name="phase_joint_weights")(
        [phase_probabilities, attention_weights]
    )
    weights = layers.Lambda(
        lambda value: value / tf.maximum(
            tf.reduce_sum(value, axis=1, keepdims=True),
            tf.keras.backend.epsilon(),
        ),
        name="phase_normalized_weights",
    )(weights)
    pooled = layers.Lambda(
        lambda values: tf.einsum("btd,btp->bpd", values[0], values[1]),
        name="phase_specific_pooling",
    )([x, weights])

    mode = str(fusion_mode).lower()
    if mode == "concat":
        fused = layers.Reshape((-1,), name="phase_concat")(pooled)
    elif mode == "sum":
        fused = layers.Lambda(lambda value: tf.reduce_sum(value, axis=1), name="phase_sum")(pooled)
    elif mode == "learned":
        summary = layers.Lambda(
            lambda value: tf.reduce_mean(value, axis=-1),
            name="phase_embedding_summary",
        )(pooled)
        phase_weights = layers.Softmax(axis=1, name="phase_fusion_weights")(
            layers.Dense(1, name="phase_fusion_logits")(summary)
        )
        phase_weights = layers.Reshape((-1, 1), name="phase_fusion_expand")(phase_weights)
        fused = layers.Lambda(
            lambda values: tf.reduce_sum(values[0] * values[1], axis=1),
            name="phase_learned_fusion",
        )([pooled, phase_weights])
    else:
        raise ValueError("phase_fusion_mode must be 'concat', 'sum', or 'learned'.")
    return fused, phase_probabilities


_POOLING_REGISTRY = {
    "attention": _attention_pool_over_time,
    "mean": _mean_pool_over_time,
    "max": _max_pool_over_time,
    "last": _last_pool_over_time,
}


def _make_activation(name: str):
    name = str(name).lower()
    if name == "relu":     return layers.Activation("relu")
    if name == "gelu":     return layers.Activation("gelu")
    if name == "swish":    return layers.Activation("swish")
    if name == "elu":      return layers.Activation("elu")
    if name == "leaky":    return layers.LeakyReLU(negative_slope=0.1)
    if name == "tanh":     return layers.Activation("tanh")
    raise ValueError(f"Unknown activation '{name}'")


def _json_maybe(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return value


def prepare_multibranch_bayesian_space(param_space: Dict[str, Any]) -> Dict[str, Any]:
    return prepare_bayesian_space(prepare_multitask_param_space(param_space, "bayesian"))


def prepare_multibranch_evolutionary_space(param_space: Dict[str, Any]) -> Dict[str, Any]:
    return prepare_evolutionary_space(prepare_multitask_param_space(param_space, "evolutionary"))


def group_feature_names(
    feature_names: Sequence[str],
    stft_prefixes: Sequence[str] = (),
    cwt_prefixes: Sequence[str] = (),
    separate_derived_imu: bool = False,
) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {name: [] for name in BRANCH_ORDER}
    stft_tags = [f"_{p}_" for p in stft_prefixes if p]
    cwt_tags = [f"_{p}_" for p in cwt_prefixes if p]
    for idx, name in enumerate(feature_names):
        tagged = f"_{name}_"
        if any(tag in tagged for tag in stft_tags):
            groups["stft"].append(idx)
        elif any(tag in tagged for tag in cwt_tags):
            groups["cwt"].append(idx)
        elif separate_derived_imu and _is_derived_imu_feature(name):
            groups["derived_imu"].append(idx)
        elif name.startswith("acc_") or name.startswith("lin_acc"):
            groups["acc"].append(idx)
        elif name.startswith("rot_") or name.startswith("ang_vel") or name.startswith("rot6d"):
            groups["rotation"].append(idx)
        elif name.startswith("tof_"):
            groups["tof"].append(idx)
        elif name.startswith("thm_"):
            groups["thermo"].append(idx)
        else:
            raise ValueError(
                "Unmapped feature name while building branches: "
                f"{name!r}. Update the branch prefixes to account for every feature."
            )
    return {name: idxs for name, idxs in groups.items() if idxs}


def _is_derived_imu_feature(name: str) -> bool:
    if name.startswith("lin_acc"):
        return True
    if not name.startswith("acc_"):
        return False
    return any(
        marker in name
        for marker in (
            "_jerk",
            "_raw_integrated_velocity",
            "_raw_integrated_displacement",
            "_linear_",
            "_mag",
            "_dr_vel",
            "_dr_pos",
        )
    )


def describe_branches(feature_names, groups):
    rows = []
    for name in BRANCH_ORDER:
        idxs = groups.get(name, [])
        if not idxs:
            continue
        example_cols = [feature_names[i] for i in idxs[:3]]
        rows.append({
            "branch": name,
            "n_features": len(idxs),
            "kind": "static (per-sequence)" if name in STATIC_BRANCHES else "time-series",
            "example_columns": ", ".join(example_cols) + (", ..." if len(idxs) > 3 else ""),
        })
    return pd.DataFrame(rows)


def _build_optimizer(
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    momentum: float,
    gradient_clip_norm: Optional[float],
    gradient_clip_value: Optional[float],
):
    name = str(optimizer_name).lower()
    if gradient_clip_norm is not None and gradient_clip_value is not None:
        raise ValueError("Set at most one of gradient_clip_norm and gradient_clip_value.")
    clip_kwargs = {}
    if gradient_clip_norm is not None:
        clip_kwargs["clipnorm"] = float(gradient_clip_norm)
    if gradient_clip_value is not None:
        clip_kwargs["clipvalue"] = float(gradient_clip_value)

    if name == "adamw":
        return optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay, **clip_kwargs
        )
    if name == "adam":
        return optimizers.Adam(learning_rate=learning_rate, **clip_kwargs)
    if name == "sgd":
        return optimizers.SGD(
            learning_rate=learning_rate, momentum=momentum, nesterov=True, **clip_kwargs
        )
    if name == "rmsprop":
        return optimizers.RMSprop(learning_rate=learning_rate, **clip_kwargs)
    if name == "adagrad":
        return optimizers.Adagrad(learning_rate=learning_rate, **clip_kwargs)
    if name == "adadelta":
        return optimizers.Adadelta(learning_rate=learning_rate, **clip_kwargs)
    if name == "nadam":
        return optimizers.Nadam(learning_rate=learning_rate, **clip_kwargs)
    raise ValueError(f"Unknown optimizer_name '{optimizer_name}'")


def _build_loss(loss_name: str, label_smoothing: float):
    name = str(loss_name).lower()
    if name == "cce":
        return losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
    if name == "sparse_cce":
        # Keras 3's SparseCategoricalCrossentropy does not accept
        # ``label_smoothing``.  Convert sparse labels to one-hot only when
        # smoothing is requested, preserving the native sparse path otherwise.
        if not label_smoothing:
            return losses.SparseCategoricalCrossentropy()

        def sparse_cce_with_smoothing(y_true, y_pred):
            y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
            y_true_oh = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
            return losses.categorical_crossentropy(
                y_true_oh,
                y_pred,
                label_smoothing=float(label_smoothing),
            )

        return sparse_cce_with_smoothing
    if name == "focal":
        # Basic focal-loss for sparse labels
        def sparse_focal(y_true, y_pred):
            y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
            y_true_oh = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
            gamma = 2.0
            eps = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
            cross_ent = -y_true_oh * tf.math.log(y_pred)
            weight = tf.pow(1.0 - y_pred, gamma)
            loss = tf.reduce_sum(weight * cross_ent, axis=-1)
            return tf.reduce_mean(loss)
        return sparse_focal
    raise ValueError(f"Unknown loss_name '{loss_name}'")


def _build_temporal_fusion_model(
    time_series_shapes,
    static_shapes,
    n_classes,
    architecture,
    sequence_length,
    branch_filters,
    branch_kernel_sizes,
    branch_num_conv_layers,
    branch_filter_mode,
    activation,
    use_batchnorm,
    conv_dropout,
    branch_dense_units,
    static_branch_units,
    static_dropout,
    dense_units,
    dropout,
    use_bigru,
    gru_units,
    gru_layers,
    gru_dropout,
    gru_recurrent_dropout,
    bidirectional_gru,
    attention_type,
    attention_units,
    cnn_depth,
    cnn_filters,
    cnn_kernel_size,
    cnn_dropout,
    use_residual,
    use_se,
    se_ratio,
    n_phases,
    phase_attention_units,
    phase_attention_temperature,
    phase_fusion_mode,
    branch_encoder_type,
    branch_cnn_filters,
    branch_cnn_kernel_size,
    branch_cnn_depth,
    branch_dropout,
    post_fusion_attention_type,
    post_fusion_attention_units,
    post_fusion_pooling,
    bert_num_layers,
    bert_d_model,
    bert_num_heads,
    bert_ff_dim,
    bert_dropout,
    bert_attention_dropout,
    bert_activation,
    bert_pooling,
    bert_use_cls_token,
    bert_positional_encoding,
    tof_encoder_type,
    tof_spatial_filters,
    tof_spatial_kernel_size,
    tof_spatial_depth,
    tof_gru_units,
    fusion_type,
    fusion_units,
    fusion_dropout,
    use_orientation_aux_head,
    use_phase_aux_head,
    orientation_num_classes,
    custom_aux_heads=None,
):
    """Build alternate temporal models while keeping the original builder path intact."""
    bert_positional_encoding = str(bert_positional_encoding).lower()
    bert_pooling = str(bert_pooling).lower()
    tof_encoder_type = str(tof_encoder_type).lower()
    if (architecture == "phase_attention" or use_phase_aux_head) and n_phases < 2:
        raise ValueError("n_phases must be >= 2.")
    if architecture == "phase_attention" and (
        not np.isfinite(phase_attention_temperature) or phase_attention_temperature <= 0
    ):
        raise ValueError("phase_attention_temperature must be positive.")
    if architecture == "bert" and (
        bert_d_model < 1 or bert_num_heads < 1 or bert_d_model % bert_num_heads
    ):
        raise ValueError("bert_d_model must be positive and divisible by bert_num_heads.")
    if architecture == "bert" and (bert_num_layers < 1 or bert_ff_dim < 1):
        raise ValueError("bert_num_layers and bert_ff_dim must be positive.")
    if architecture == "bert" and bert_pooling == "cls" and not bert_use_cls_token:
        raise ValueError("bert_pooling='cls' requires bert_use_cls_token=True.")
    if architecture == "bert" and bert_positional_encoding not in {"learned", "sinusoidal"}:
        raise ValueError("bert_positional_encoding must be 'learned' or 'sinusoidal'.")
    if architecture == "bert" and bert_pooling not in {"cls", "attention", "avg", "avg_max"}:
        raise ValueError("bert_pooling must be 'cls', 'attention', 'avg', or 'avg_max'.")
    if fusion_type not in {"concat", "sum", "attention"}:
        raise ValueError("fusion_type must be 'concat', 'sum', or 'attention'.")
    if use_se and int(se_ratio) < 1:
        raise ValueError("se_ratio must be a positive integer.")

    ts_names = list(time_series_shapes)
    static_names = list(static_shapes)
    inferred_length = next(iter(time_series_shapes.values()))[0] if time_series_shapes else sequence_length
    if ts_names and any(shape[0] != inferred_length for shape in time_series_shapes.values()):
        raise ValueError("All time-series branches must share the same timestep count.")
    inputs = []
    encoded = []
    input_by_name = {}
    mask_input = None
    if inferred_length is None or int(inferred_length) < 1:
        raise ValueError("A positive sequence_length is required for temporal models.")
    if (
        ts_names
        or architecture != "bigru_attention"
        or use_bigru
        or use_orientation_aux_head
        or use_phase_aux_head
        or custom_aux_heads
    ):
        inferred_length = int(inferred_length)
        mask_input = layers.Input(shape=(inferred_length,), name="valid_timestep_mask")
    model_mask_input = mask_input

    for name, shape in time_series_shapes.items():
        branch_input = layers.Input(shape=shape, name=f"{name}_input")
        inputs.append(branch_input)
        input_by_name[name] = branch_input
        x = branch_input
        if mask_input is not None:
            branch_mask = layers.Reshape((-1, 1), name=f"{name}_input_mask_expand")(mask_input)
            x = layers.Multiply(name=f"{name}_masked_input")([x, branch_mask])
        if name == "tof" and tof_encoder_type in {"spatial_cnn", "spatial_cnn_gru"}:
            if shape[-1] % 64:
                raise ValueError(
                    "ToF spatial encoders require raw 8x8 pixel groups; "
                    f"received {shape[-1]} features (expected a multiple of 64)."
                )
            channels = shape[-1] // 64
            x = layers.Reshape(
                (shape[0], 8, 8, channels),
                name="tof_grid_reshape",
            )(x)
            for layer_idx in range(max(1, int(tof_spatial_depth))):
                x = layers.TimeDistributed(
                    layers.Conv2D(
                        int(tof_spatial_filters),
                        int(tof_spatial_kernel_size),
                        padding="same",
                    ),
                    name=f"tof_spatial_conv{layer_idx + 1}",
                )(x)
                x = layers.TimeDistributed(
                    _make_activation(activation),
                    name=f"tof_spatial_activation{layer_idx + 1}",
                )(x)
            x = layers.TimeDistributed(
                layers.GlobalAveragePooling2D(),
                name="tof_spatial_pool",
            )(x)
            if tof_encoder_type == "spatial_cnn_gru":
                tof_mask = layers.Lambda(
                    lambda value: tf.cast(value, tf.bool),
                    name="tof_spatial_boolean_mask",
                )(mask_input)
                x = layers.Bidirectional(
                    layers.GRU(int(tof_gru_units), return_sequences=True),
                    name="tof_spatial_bigru",
                )(x, mask=tof_mask)
        elif name == "tof" and tof_encoder_type != "flatten_1d":
            raise ValueError(
                "tof_encoder_type must be 'flatten_1d', 'spatial_cnn', or 'spatial_cnn_gru'."
            )
        else:
            if branch_encoder_type not in {"conv1d", "cnn"}:
                raise ValueError("branch_encoder_type must be 'conv1d' or 'cnn'.")
            depth = max(1, int(branch_cnn_depth))
            for layer_idx in range(depth):
                shortcut = x
                x = layers.Conv1D(
                    int(branch_cnn_filters),
                    int(branch_cnn_kernel_size),
                    padding="same",
                    name=f"{name}_temporal_conv{layer_idx + 1}",
                )(x)
                if use_batchnorm:
                    x = layers.BatchNormalization(name=f"{name}_temporal_bn{layer_idx + 1}")(x)
                x = _make_activation(activation)(x)
                if branch_dropout:
                    x = layers.SpatialDropout1D(
                        branch_dropout,
                        name=f"{name}_temporal_dropout{layer_idx + 1}",
                    )(x)
                if use_residual:
                    if int(shortcut.shape[-1]) != int(branch_cnn_filters):
                        shortcut = layers.Conv1D(
                            int(branch_cnn_filters), 1, padding="same",
                            name=f"{name}_temporal_projection{layer_idx + 1}",
                        )(shortcut)
                    x = layers.Add(name=f"{name}_temporal_residual{layer_idx + 1}")([x, shortcut])
                if use_se:
                    gate = layers.GlobalAveragePooling1D(
                        name=f"{name}_se_pool{layer_idx + 1}"
                    )(x)
                    gate = layers.Dense(
                        max(1, int(branch_cnn_filters) // int(se_ratio)),
                        activation="relu",
                        name=f"{name}_se_reduce{layer_idx + 1}",
                    )(gate)
                    gate = layers.Dense(
                        int(branch_cnn_filters),
                        activation="sigmoid",
                        name=f"{name}_se_expand{layer_idx + 1}",
                    )(gate)
                    gate = layers.Reshape((1, int(branch_cnn_filters)))(gate)
                    x = layers.Multiply(name=f"{name}_se_gate{layer_idx + 1}")([x, gate])
        encoded.append(x)

    static_depths = _json_maybe(branch_num_conv_layers)
    for name, shape in static_shapes.items():
        branch_input = layers.Input(shape=(shape,), name=f"{name}_input")
        inputs.append(branch_input)
        input_by_name[name] = branch_input
        static_depth = static_depths.get(name, 1) if isinstance(static_depths, dict) else 1
        if isinstance(static_depth, bool) or not isinstance(static_depth, Integral) or static_depth <= 0:
            raise ValueError(f"Static branch depth for {name!r} must be a positive integer, got {static_depth!r}.")
        x = branch_input
        for layer_idx in range(static_depth - 1):
            x = layers.Dense(
                static_branch_units, activation="relu", name=f"{name}_static_hidden{layer_idx + 1}"
            )(x)
        x = layers.Dense(static_branch_units, activation="relu", name=f"{name}_static_dense")(x)
        if static_dropout:
            x = layers.Dropout(static_dropout, name=f"{name}_static_dropout")(x)
        if mask_input is not None:
            x = layers.RepeatVector(inferred_length, name=f"{name}_static_repeat")(x)
        else:
            x = layers.Reshape((1, static_branch_units), name=f"{name}_static_expand")(x)
        encoded.append(x)

    if not encoded:
        raise InvalidExtractorParams("No modality branches were produced; nothing to feed the model.")

    if mask_input is not None:
        if fusion_type == "concat":
            x = layers.Concatenate(axis=-1, name="temporal_branch_fusion")(encoded) if len(encoded) > 1 else encoded[0]
        else:
            projected = [
                layers.Dense(fusion_units, name=f"fusion_projection_{index}")(branch)
                for index, branch in enumerate(encoded)
            ]
            if fusion_type == "sum":
                x = layers.Add(name="temporal_branch_sum")(projected)
            else:
                stacked = layers.Lambda(
                    lambda values: tf.stack(values, axis=2),
                    name="temporal_branch_stack",
                )(projected)
                weights = layers.Softmax(axis=2, name="temporal_branch_attention")(
                    layers.Dense(1, name="temporal_branch_attention_logits")(stacked)
                )
                x = layers.Lambda(
                    lambda values: tf.reduce_sum(values[0] * values[1], axis=2),
                    name="temporal_branch_attention_pool",
                )([stacked, weights])
        x = layers.Dense(fusion_units, activation=activation, name="fusion_projection")(x)
        if fusion_dropout:
            x = layers.Dropout(fusion_dropout, name="fusion_dropout")(x)
        mask_expanded = layers.Reshape((-1, 1), name="fusion_mask_expand")(mask_input)
        x = layers.Multiply(name="fusion_masked_timesteps")([x, mask_expanded])
    else:
        pooled_static = [
            layers.GlobalAveragePooling1D(name=f"{name}_static_reduce")(branch)
            for name, branch in zip(static_names, encoded)
        ]
        x = layers.Concatenate(name="static_branch_fusion")(pooled_static) if len(pooled_static) > 1 else pooled_static[0]
        x = layers.Dense(fusion_units, activation=activation, name="fusion_projection")(x)

    phase_probabilities = None
    sequence_mask = mask_input
    if architecture == "bert":
        d_model = int(bert_d_model)
        x = layers.Dense(d_model, name="bert_input_projection")(x)
        if bert_use_cls_token:
            cls_indices = layers.Lambda(
                lambda value: tf.zeros((tf.shape(value)[0], 1), dtype=tf.int32),
                name="bert_cls_indices",
            )(x)
            cls = layers.Embedding(1, d_model, name="bert_cls_embedding")(cls_indices)
            x = layers.Concatenate(axis=1, name="bert_prepend_cls")([cls, x])
            sequence_mask = layers.Concatenate(axis=1, name="bert_cls_mask")(
                [layers.Lambda(lambda value: tf.ones((tf.shape(value)[0], 1), dtype=value.dtype))(mask_input), mask_input]
            )
        pos_length = int(x.shape[1])
        if bert_positional_encoding == "learned":
            positions = layers.Lambda(
                lambda value: tf.tile(tf.range(pos_length)[None, :], [tf.shape(value)[0], 1]),
                name="bert_position_indices",
            )(x)
            position_embedding = layers.Embedding(
                pos_length, d_model, name="bert_position_embedding"
            )(positions)
        else:
            constant_encoding = _positional_encoding(pos_length, d_model)
            position_embedding = layers.Lambda(
                lambda value: tf.cast(constant_encoding, value.dtype),
                name="bert_sinusoidal_positions",
            )(x)
        x = layers.Add(name="bert_add_positions")([x, position_embedding])
        for layer_idx in range(max(1, int(bert_num_layers))):
            x = _transformer_encoder_block(
                x, sequence_mask, f"bert_block{layer_idx + 1}", d_model,
                int(bert_num_heads), int(bert_ff_dim), float(bert_dropout),
                float(bert_attention_dropout), str(bert_activation),
            )
        if bert_pooling == "cls":
            if not bert_use_cls_token:
                raise ValueError("bert_pooling='cls' requires bert_use_cls_token=True.")
            pooled = layers.Lambda(lambda value: value[:, 0, :], name="bert_cls_pool")(x)
        elif bert_pooling == "attention":
            pooled = _masked_attention_pool(
                "bert", x, sequence_mask, int(attention_units), attention_type
            )
        elif bert_pooling == "avg":
            pooled = layers.Lambda(
                lambda values: tf.reduce_sum(values[0] * values[1][:, :, None], axis=1)
                / tf.maximum(tf.reduce_sum(values[1], axis=1, keepdims=True), 1.0),
                name="bert_masked_average_pool",
            )([x, sequence_mask])
        elif bert_pooling == "avg_max":
            avg = layers.Lambda(
                lambda values: tf.reduce_sum(values[0] * values[1][:, :, None], axis=1)
                / tf.maximum(tf.reduce_sum(values[1], axis=1, keepdims=True), 1.0),
                name="bert_masked_average_pool",
            )([x, sequence_mask])
            max_input = layers.Lambda(
                lambda values: tf.where(
                    values[1][:, :, None] > 0,
                    values[0],
                    tf.cast(-1e9, values[0].dtype),
                ),
                name="bert_masked_max_input",
            )([x, sequence_mask])
            pooled = layers.Concatenate(name="bert_avg_max_pool")(
                [avg, layers.GlobalMaxPooling1D(name="bert_max_pool")(max_input)]
            )
        else:
            raise ValueError("bert_pooling must be 'cls', 'attention', 'avg', or 'avg_max'.")
    else:
        if architecture == "phase_attention":
            if int(cnn_depth) < 1:
                raise ValueError("cnn_depth must be positive.")
            for layer_idx in range(int(cnn_depth)):
                shortcut = x
                x = layers.Conv1D(
                    int(cnn_filters), int(cnn_kernel_size), padding="same",
                    name=f"shared_temporal_conv{layer_idx + 1}",
                )(x)
                if use_batchnorm:
                    x = layers.BatchNormalization(name=f"shared_temporal_bn{layer_idx + 1}")(x)
                x = _make_activation(activation)(x)
                if cnn_dropout:
                    x = layers.SpatialDropout1D(cnn_dropout, name=f"shared_temporal_dropout{layer_idx + 1}")(x)
                if use_residual:
                    if int(shortcut.shape[-1]) != int(cnn_filters):
                        shortcut = layers.Conv1D(int(cnn_filters), 1, padding="same")(shortcut)
                    x = layers.Add(name=f"shared_temporal_residual{layer_idx + 1}")([x, shortcut])
            pooled, phase_probabilities = _phase_attention_pool(
            x, sequence_mask, int(n_phases), int(phase_attention_units),
            float(phase_attention_temperature), phase_fusion_mode,
            )
        elif architecture == "cnn_attention_pool":
            for layer_idx in range(max(1, int(cnn_depth))):
                x = layers.Conv1D(
                    int(cnn_filters), int(cnn_kernel_size), padding="same",
                    activation=activation, name=f"post_fusion_conv{layer_idx + 1}",
                )(x)
                if cnn_dropout:
                    x = layers.SpatialDropout1D(cnn_dropout, name=f"post_fusion_dropout{layer_idx + 1}")(x)
            pooling = str(post_fusion_pooling).lower()
            if pooling == "attention":
                pooled = _masked_attention_pool(
                    "post_fusion", x, sequence_mask,
                    int(post_fusion_attention_units), post_fusion_attention_type,
                )
            elif pooling in {"avg", "max", "avg_max", "attention_avg_max"}:
                avg = layers.Lambda(
                    lambda values: tf.reduce_sum(values[0] * values[1][:, :, None], axis=1)
                    / tf.maximum(tf.reduce_sum(values[1], axis=1, keepdims=True), 1.0),
                    name="post_fusion_average_pool",
                )([x, sequence_mask])
                max_input = layers.Lambda(
                    lambda values: tf.where(
                        values[1][:, :, None] > 0,
                        values[0],
                        tf.cast(-1e9, values[0].dtype),
                    ),
                    name="post_fusion_masked_max_input",
                )([x, sequence_mask])
                maximum = layers.GlobalMaxPooling1D(name="post_fusion_max_pool")(max_input)
                if pooling == "avg":
                    pooled = avg
                elif pooling == "max":
                    pooled = maximum
                elif pooling == "avg_max":
                    pooled = layers.Concatenate(name="post_fusion_avg_max")([avg, maximum])
                else:
                    attention = _masked_attention_pool(
                        "post_fusion", x, sequence_mask,
                        int(post_fusion_attention_units), post_fusion_attention_type,
                    )
                    pooled = layers.Concatenate(name="post_fusion_attention_avg_max")(
                        [attention, avg, maximum]
                    )
            else:
                raise ValueError(
                    "post_fusion_pooling must be 'attention', 'avg', 'max', 'avg_max', "
                    "or 'attention_avg_max'."
                )
        elif architecture == "bigru_attention":
            if use_bigru:
                gru_mask = layers.Lambda(
                    lambda value: tf.cast(value, tf.bool),
                    name="shared_gru_boolean_mask",
                )(sequence_mask)
                for layer_idx in range(max(1, int(gru_layers))):
                    gru = layers.GRU(
                        int(gru_units),
                        return_sequences=True,
                        dropout=float(gru_dropout),
                        recurrent_dropout=float(gru_recurrent_dropout),
                        name=f"shared_gru{layer_idx + 1}",
                    )
                    x = (
                        layers.Bidirectional(gru, name=f"shared_bigru{layer_idx + 1}")(
                            x, mask=gru_mask
                        )
                        if bidirectional_gru
                        else gru(x, mask=gru_mask)
                    )
            pooled = _masked_attention_pool(
                    "bigru", x, sequence_mask, int(attention_units), attention_type
            )
        else:
            raise ValueError(f"Unsupported temporal_architecture: {architecture!r}")

    head = layers.Dense(dense_units, activation="relu", name="head_dense")(pooled)
    head = layers.Dropout(dropout, name="head_dropout")(head)
    outputs = {"predictions": layers.Dense(n_classes, activation="softmax", name="predictions")(head)}
    auxiliary_x, auxiliary_mask = x, sequence_mask
    if architecture == "bert" and bert_use_cls_token:
        auxiliary_x = layers.Lambda(lambda value: value[:, 1:, :], name="aux_remove_cls")(x)
        auxiliary_mask = mask_input
    if use_orientation_aux_head:
        outputs["orientation_aux"] = layers.Dense(
            int(orientation_num_classes), activation="softmax", name="orientation_aux"
        )(head)
    if use_phase_aux_head:
        if phase_probabilities is None:
            if sequence_mask is None:
                raise ValueError("use_phase_aux_head requires time-series features.")
            phase_probabilities = layers.Softmax(axis=-1, name="phase_aux_probabilities")(
                layers.Dense(int(n_phases), name="phase_aux_logits")(auxiliary_x)
            )
        outputs["phase_aux"] = phase_probabilities
    for spec in custom_aux_heads or []:
        name = spec["name"]
        aux = pooled
        if spec.get("window_size") is not None:
            if sequence_mask is None or not time_series_shapes:
                raise ValueError("Window auxiliary heads require time-series inputs.")
            # Each window head sees its own final valid span of the shared temporal representation.
            # Positions are measured in extracted timesteps, not seconds.
            window = int(spec["window_size"])
            if window > int(inferred_length):
                raise ValueError(f"Auxiliary window {window} exceeds chunk length {inferred_length}.")
            def window_pool(values, window=window):
                values_x, mask = values
                positions = tf.cumsum(mask, axis=1)
                end = tf.reduce_sum(mask, axis=1, keepdims=True)
                selected = mask * tf.cast(positions > end - window, mask.dtype)
                return tf.reduce_sum(values_x * selected[:, :, None], axis=1) / tf.maximum(
                    tf.reduce_sum(selected, axis=1, keepdims=True), 1.0)
            aux = layers.Lambda(window_pool, name=f"{name}_window_pool")([auxiliary_x, auxiliary_mask])
        for idx, units in enumerate(spec["hidden_units"]):
            aux = layers.Dense(units, activation=spec["activation"], name=f"{name}_dense{idx}")(aux)
            aux = layers.Dropout(spec["dropout"], name=f"{name}_dropout{idx}")(aux)
        outputs[name] = layers.Dense(
            1 if spec["task"] == "regression" else spec["num_classes"],
            activation="linear" if spec["task"] == "regression" else "softmax", name=name,
        )(aux)
    if not use_orientation_aux_head and not use_phase_aux_head and not custom_aux_heads:
        outputs = outputs["predictions"]
    model_inputs = inputs + ([model_mask_input] if model_mask_input is not None else [])
    model = models.Model(
        inputs=model_inputs,
        outputs=outputs,
        name=f"multibranch_{architecture}",
    )
    # Non-output tensors let an optional pretraining decoder reuse the exact
    # temporal backbone without changing the ordinary classifier's graph.
    model._temporal_representation_tensor = auxiliary_x
    model._temporal_mask_tensor = auxiliary_mask
    return model


def build_multibranch_model(
    time_series_shapes: Dict[str, Tuple[int, int]],
    static_shapes: Dict[str, int],
    n_classes: int,
    branch_filters: Optional[Dict[str, int]] = None,
    branch_kernel_sizes: Optional[Dict[str, int]] = None,
    branch_pool_sizes: Optional[Dict[str, int]] = None,
    branch_num_conv_layers: Any = 2,
    branch_filter_mode: str = "custom",
    branch_dense_units: int = 64,
    static_branch_units: int = 32,
    static_dropout: float = 0.1,
    dense_units: int = 128,
    dropout: float = 0.3,
    conv_dropout: float = 0.1,
    learning_rate: float = 1e-3,
    use_batchnorm: bool = True,
    # ---- NEW ----
    activation: str = "relu",
    pooling_mode: str = "attention",
    optimizer_name: str = "adam",
    weight_decay: float = 0.0,
    momentum: float = 0.9,
    gradient_clip_norm: Optional[float] = None,
    gradient_clip_value: Optional[float] = None,
    loss_name: str = "sparse_cce",
    label_smoothing: float = 0.0,
    # ------------
    random_state: int = 42,
    temporal_architecture: str = "bigru_attention",
    use_bigru: bool = False,
    gru_units: int = 64,
    gru_layers: int = 1,
    gru_dropout: float = 0.0,
    gru_recurrent_dropout: float = 0.0,
    bidirectional_gru: bool = True,
    attention_type: str = "additive",
    attention_units: int = 64,
    cnn_depth: int = 2,
    cnn_filters: int = 64,
    cnn_kernel_size: int = 5,
    cnn_dropout: float = 0.1,
    use_residual: bool = False,
    use_se: bool = False,
    se_ratio: int = 8,
    n_phases: int = 3,
    phase_attention_units: int = 32,
    phase_attention_temperature: float = 1.0,
    phase_fusion_mode: str = "concat",
    branch_encoder_type: str = "conv1d",
    branch_cnn_filters: int = 32,
    branch_cnn_kernel_size: int = 3,
    branch_cnn_depth: int = 2,
    branch_dropout: float = 0.1,
    post_fusion_attention_type: str = "additive",
    post_fusion_attention_units: int = 64,
    post_fusion_pooling: str = "attention",
    bert_num_layers: int = 2,
    bert_d_model: int = 64,
    bert_num_heads: int = 4,
    bert_ff_dim: int = 128,
    bert_dropout: float = 0.1,
    bert_attention_dropout: float = 0.1,
    bert_activation: str = "gelu",
    bert_pooling: str = "attention",
    bert_use_cls_token: bool = False,
    bert_positional_encoding: str = "learned",
    tof_encoder_type: str = "flatten_1d",
    tof_spatial_filters: int = 16,
    tof_spatial_kernel_size: int = 3,
    tof_spatial_depth: int = 2,
    tof_gru_units: int = 32,
    fusion_type: str = "concat",
    fusion_units: int = 64,
    fusion_dropout: float = 0.1,
    use_orientation_aux_head: bool = False,
    use_phase_aux_head: bool = False,
    orientation_num_classes: int = 4,
    custom_aux_heads=None,
    orientation_loss_weight: float = 1.0,
    phase_loss_weight: float = 1.0,
    sequence_length: Optional[int] = None,
):
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("MultiBranchSequenceClassifier requires TensorFlow/Keras.")

    tf.random.set_seed(random_state)

    architecture = str(temporal_architecture).lower()
    if architecture not in {"bigru_attention", "phase_attention", "bert", "cnn_attention_pool"}:
        raise ValueError(
            "temporal_architecture must be 'bigru_attention', 'phase_attention', "
            "'bert', or 'cnn_attention_pool'."
        )
    custom_aux_heads = _normalise_aux_heads(custom_aux_heads)
    for enabled, weight in ((use_orientation_aux_head, orientation_loss_weight),
                            (use_phase_aux_head, phase_loss_weight)):
        if enabled and (not np.isfinite(weight) or weight < 0):
            raise ValueError("Auxiliary loss weights must be finite and nonnegative.")
    if architecture != "bigru_attention" or use_bigru or use_orientation_aux_head or use_phase_aux_head or custom_aux_heads:
        model = _build_temporal_fusion_model(
            time_series_shapes=time_series_shapes,
            static_shapes=static_shapes,
            n_classes=n_classes,
            architecture=architecture,
            sequence_length=sequence_length,
            branch_filters=branch_filters,
            branch_kernel_sizes=branch_kernel_sizes,
            branch_num_conv_layers=branch_num_conv_layers,
            branch_filter_mode=branch_filter_mode,
            activation=activation,
            use_batchnorm=use_batchnorm,
            conv_dropout=conv_dropout,
            branch_dense_units=branch_dense_units,
            static_branch_units=static_branch_units,
            static_dropout=static_dropout,
            dense_units=dense_units,
            dropout=dropout,
            use_bigru=use_bigru,
            gru_units=gru_units,
            gru_layers=gru_layers,
            gru_dropout=gru_dropout,
            gru_recurrent_dropout=gru_recurrent_dropout,
            bidirectional_gru=bidirectional_gru,
            attention_type=attention_type,
            attention_units=attention_units,
            cnn_depth=cnn_depth,
            cnn_filters=cnn_filters,
            cnn_kernel_size=cnn_kernel_size,
            cnn_dropout=cnn_dropout,
            use_residual=use_residual,
            use_se=use_se,
            se_ratio=se_ratio,
            n_phases=n_phases,
            phase_attention_units=phase_attention_units,
            phase_attention_temperature=phase_attention_temperature,
            phase_fusion_mode=phase_fusion_mode,
            branch_encoder_type=branch_encoder_type,
            branch_cnn_filters=branch_cnn_filters,
            branch_cnn_kernel_size=branch_cnn_kernel_size,
            branch_cnn_depth=branch_cnn_depth,
            branch_dropout=branch_dropout,
            post_fusion_attention_type=post_fusion_attention_type,
            post_fusion_attention_units=post_fusion_attention_units,
            post_fusion_pooling=post_fusion_pooling,
            bert_num_layers=bert_num_layers,
            bert_d_model=bert_d_model,
            bert_num_heads=bert_num_heads,
            bert_ff_dim=bert_ff_dim,
            bert_dropout=bert_dropout,
            bert_attention_dropout=bert_attention_dropout,
            bert_activation=bert_activation,
            bert_pooling=bert_pooling,
            bert_use_cls_token=bert_use_cls_token,
            bert_positional_encoding=bert_positional_encoding,
            tof_encoder_type=tof_encoder_type,
            tof_spatial_filters=tof_spatial_filters,
            tof_spatial_kernel_size=tof_spatial_kernel_size,
            tof_spatial_depth=tof_spatial_depth,
            tof_gru_units=tof_gru_units,
            fusion_type=fusion_type,
            fusion_units=fusion_units,
            fusion_dropout=fusion_dropout,
            use_orientation_aux_head=use_orientation_aux_head,
            use_phase_aux_head=use_phase_aux_head,
            orientation_num_classes=orientation_num_classes,
            custom_aux_heads=custom_aux_heads,
        )
        main_loss = _build_loss(loss_name=loss_name, label_smoothing=label_smoothing)
        has_auxiliary_outputs = use_orientation_aux_head or use_phase_aux_head or bool(custom_aux_heads)
        model_loss = (
            {
                "predictions": main_loss,
                **({"orientation_aux": "sparse_categorical_crossentropy"} if use_orientation_aux_head else {}),
                **({"phase_aux": "sparse_categorical_crossentropy"} if use_phase_aux_head else {}),
                **{h["name"]: ("mse" if h["task"] == "regression" else "sparse_categorical_crossentropy") for h in custom_aux_heads},
            }
            if has_auxiliary_outputs
            else main_loss
        )
        model.compile(
            optimizer=_build_optimizer(
                optimizer_name=optimizer_name,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                momentum=momentum,
                gradient_clip_norm=gradient_clip_norm,
                gradient_clip_value=gradient_clip_value,
            ),
            loss=model_loss,
            loss_weights=({
                "predictions": 1.0,
                **({"orientation_aux": orientation_loss_weight} if use_orientation_aux_head else {}),
                **({"phase_aux": phase_loss_weight} if use_phase_aux_head else {}),
                **{h["name"]: h["loss_weight"] for h in custom_aux_heads},
            } if has_auxiliary_outputs else None),
            metrics={"predictions": ["accuracy"]} if has_auxiliary_outputs else ["accuracy"],
        )
        return model

    branch_filters = _json_maybe(branch_filters) or DEFAULT_BRANCH_FILTERS
    branch_kernel_sizes = _json_maybe(branch_kernel_sizes) or DEFAULT_BRANCH_KERNEL_SIZES
    branch_pool_sizes = _json_maybe(branch_pool_sizes) or DEFAULT_BRANCH_POOL_SIZES
    branch_num_conv_layers = _json_maybe(branch_num_conv_layers)
    if isinstance(branch_num_conv_layers, dict):
        for name in static_shapes:
            if name in branch_num_conv_layers:
                depth = branch_num_conv_layers[name]
                if isinstance(depth, bool) or not isinstance(depth, Integral) or depth <= 0:
                    raise ValueError(f"Static branch depth for {name!r} must be a positive integer, got {depth!r}.")
        branch_num_conv_layers = {str(k): max(1, int(v)) for k, v in branch_num_conv_layers.items()}
    else:
        branch_num_conv_layers = max(1, int(branch_num_conv_layers))
    branch_filter_mode = str(branch_filter_mode).lower()
    if branch_filter_mode not in {"custom", "double", "constant"}:
        raise ValueError("branch_filter_mode must be 'custom', 'double', or 'constant'.")

    pooling_fn = _POOLING_REGISTRY.get(str(pooling_mode).lower())
    if pooling_fn is None:
        raise ValueError(f"Unknown pooling_mode '{pooling_mode}'. Use one of {list(_POOLING_REGISTRY)}")

    inputs = []
    branch_embeddings = []

    for name, shape in time_series_shapes.items():
        branch_in = layers.Input(shape=shape, name=f"{name}_input")
        inputs.append(branch_in)

        raw_filter_value = branch_filters.get(name, 32)
        if isinstance(raw_filter_value, (list, tuple, np.ndarray)):
            filter_values = [int(v) for v in raw_filter_value]
        else:
            filter_values = [int(raw_filter_value)]

        kernel_size = int(branch_kernel_sizes.get(name, 3))
        pool_size = int(branch_pool_sizes.get(name, 2))

        if isinstance(branch_num_conv_layers, dict):
            branch_depth = max(1, int(branch_num_conv_layers.get(name, 1)))
        else:
            branch_depth = branch_num_conv_layers

        x = branch_in
        for layer_idx in range(branch_depth):
            if branch_filter_mode == "double":
                layer_filters = int(raw_filter_value) * (2 ** layer_idx)
            elif branch_filter_mode == "constant":
                layer_filters = int(raw_filter_value)
            else:
                if len(filter_values) == 1:
                    layer_filters = int(filter_values[0])
                elif len(filter_values) > layer_idx:
                    layer_filters = int(filter_values[layer_idx])
                else:
                    layer_filters = int(filter_values[-1])
            x = layers.Conv1D(
                filters=layer_filters, kernel_size=kernel_size, padding="same",
                name=f"{name}_conv{layer_idx + 1}",
            )(x)
            if use_batchnorm:
                x = layers.BatchNormalization(name=f"{name}_bn{layer_idx + 1}")(x)
            x = _make_activation(activation)(x)
            if conv_dropout:
                x = layers.SpatialDropout1D(conv_dropout, name=f"{name}_drop{layer_idx + 1}")(x)
            x = layers.MaxPooling1D(pool_size=pool_size, padding="same", name=f"{name}_pool{layer_idx + 1}")(x)

        x = pooling_fn(name, x)
        x = layers.Dense(branch_dense_units, activation="relu", name=f"{name}_embed")(x)
        branch_embeddings.append(x)

    for name, dim in static_shapes.items():
        branch_in = layers.Input(shape=(dim,), name=f"{name}_input")
        inputs.append(branch_in)
        # Static entries specify Dense depth; absent entries retain the original
        # two-layer branch independently of the time-series convolution depth.
        static_depth = branch_num_conv_layers.get(name, 2) if isinstance(branch_num_conv_layers, dict) else 2
        x = branch_in
        for layer_idx in range(static_depth):
            final_layer = layer_idx == static_depth - 1
            layer_name = f"{name}_embed" if final_layer else f"{name}_dense{layer_idx + 1}"
            x = layers.Dense(static_branch_units, activation="relu", name=layer_name)(x)
            if not final_layer:
                suffix = "" if layer_idx == 0 else str(layer_idx + 1)
                if use_batchnorm:
                    x = layers.BatchNormalization(name=f"{name}_bn{suffix}")(x)
                if static_dropout:
                    x = layers.Dropout(static_dropout, name=f"{name}_dropout{suffix}")(x)
        branch_embeddings.append(x)

    if not branch_embeddings:
        raise InvalidExtractorParams("No modality branches were produced; nothing to feed the model.")

    merged = (
        layers.Concatenate(name="branch_concat")(branch_embeddings)
        if len(branch_embeddings) > 1
        else branch_embeddings[0]
    )
    head = layers.Dense(dense_units, activation="relu", name="head_dense")(merged)
    head = layers.Dropout(dropout, name="head_dropout")(head)
    outputs = layers.Dense(n_classes, activation="softmax", name="predictions")(head)

    model = models.Model(inputs=inputs, outputs=outputs, name="multibranch_sequence_classifier")
    opt = _build_optimizer(
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
        gradient_clip_norm=gradient_clip_norm,
        gradient_clip_value=gradient_clip_value,
    )
    loss = _build_loss(loss_name=loss_name, label_smoothing=label_smoothing)
    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
    return model


class MultiBranchSequenceClassifier(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"

    def __init__(
        self,
        primary_target: str = "bfrb",
        extractor: Optional[SequenceExtractor] = None,
        augmentor: Optional[BaseEstimator] = None,
        branch_filters: Optional[Dict[str, int]] = None,
        branch_kernel_sizes: Optional[Dict[str, int]] = None,
        branch_pool_sizes: Optional[Dict[str, int]] = None,
        branch_num_conv_layers: Any = 2,
        branch_filter_mode: str = "custom",
        branch_dense_units: int = 64,
        static_branch_units: int = 32,
        static_dropout: float = 0.1,
        dense_units: int = 128,
        dropout: float = 0.3,
        conv_dropout: float = 0.1,
        learning_rate: float = 1e-3,
        use_batchnorm: bool = True,

        # ---- model architecture (new) ----
        activation: str = "relu",
        pooling_mode: str = "attention",

        # ---- optimizer (new) ----
        optimizer_name: str = "adam",
        weight_decay: float = 0.0,
        momentum: float = 0.9,

        # ---- gradient clipping (new) ----
        gradient_clip_norm: Optional[float] = None,
        gradient_clip_value: Optional[float] = None,

        # ---- loss (new) ----
        loss_name: str = "sparse_cce",
        label_smoothing: float = 0.0,

        # ---- training loop (existing) ----
        epochs: int = 30,
        batch_size: int = 64,
        validation_split: float = 0.15,
        early_stopping_patience: int = 6,
        class_weight_mode: Optional[str] = "balanced",
        verbose: int = 0,
        random_state: int = 42,

        # ---- LR scheduler (new) ----
        use_lr_scheduler: bool = False,
        lr_scheduler_type: str = "plateau",           # 'plateau' or 'cosine'
        lr_factor: float = 0.5,
        lr_patience: int = 5,
        min_lr: float = 1e-6,
        cosine_t_max: int = 100,
        warmup_epochs: int = 0,
        temporal_architecture: str = "bigru_attention",
        use_bigru: bool = False,
        gru_units: int = 64,
        gru_layers: int = 1,
        gru_dropout: float = 0.0,
        gru_recurrent_dropout: float = 0.0,
        bidirectional_gru: bool = True,
        attention_type: str = "additive",
        attention_units: int = 64,
        cnn_depth: int = 2,
        cnn_filters: int = 64,
        cnn_kernel_size: int = 5,
        cnn_dropout: float = 0.1,
        use_residual: bool = False,
        use_se: bool = False,
        se_ratio: int = 8,
        n_phases: int = 3,
        phase_attention_units: int = 32,
        phase_attention_temperature: float = 1.0,
        phase_fusion_mode: str = "concat",
        branch_encoder_type: str = "conv1d",
        branch_cnn_filters: int = 32,
        branch_cnn_kernel_size: int = 3,
        branch_cnn_depth: int = 2,
        branch_dropout: float = 0.1,
        post_fusion_attention_type: str = "additive",
        post_fusion_attention_units: int = 64,
        post_fusion_pooling: str = "attention",
        bert_num_layers: int = 2,
        bert_d_model: int = 64,
        bert_num_heads: int = 4,
        bert_ff_dim: int = 128,
        bert_dropout: float = 0.1,
        bert_attention_dropout: float = 0.1,
        bert_activation: str = "gelu",
        bert_pooling: str = "attention",
        bert_use_cls_token: bool = False,
        bert_positional_encoding: str = "learned",
        tof_encoder_type: str = "flatten_1d",
        tof_spatial_filters: int = 16,
        tof_spatial_kernel_size: int = 3,
        tof_spatial_depth: int = 2,
        tof_gru_units: int = 32,
        fusion_type: str = "concat",
        fusion_units: int = 64,
        fusion_dropout: float = 0.1,
        use_orientation_aux_head: bool = False,
        use_phase_aux_head: bool = False,
        orientation_num_classes: int = 4,
        custom_aux_heads=None,
        orientation_loss_weight: float = 1.0,
        phase_loss_weight: float = 1.0,
        phase_label_provider=None,
        invalid_candidate_policy: str = "raise",
        invalid_candidate_score: float = -1.0,
        candidate_failure_log=None,
    ):
        self.primary_target = primary_target
        self.extractor = extractor
        self.augmentor = augmentor
        self.branch_filters = branch_filters
        self.branch_kernel_sizes = branch_kernel_sizes
        self.branch_pool_sizes = branch_pool_sizes
        self.branch_num_conv_layers = branch_num_conv_layers
        self.branch_filter_mode = branch_filter_mode
        self.branch_dense_units = branch_dense_units
        self.static_branch_units = static_branch_units
        self.static_dropout = static_dropout
        self.dense_units = dense_units
        self.dropout = dropout
        self.conv_dropout = conv_dropout
        self.learning_rate = learning_rate
        self.use_batchnorm = use_batchnorm

        self.activation = activation
        self.pooling_mode = pooling_mode

        self.optimizer_name = optimizer_name
        self.weight_decay = weight_decay
        self.momentum = momentum

        self.gradient_clip_norm = gradient_clip_norm
        self.gradient_clip_value = gradient_clip_value

        self.loss_name = loss_name
        self.label_smoothing = label_smoothing

        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.class_weight_mode = class_weight_mode
        self.verbose = verbose
        self.random_state = random_state

        self.use_lr_scheduler = use_lr_scheduler
        self.lr_scheduler_type = lr_scheduler_type
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.min_lr = min_lr
        self.cosine_t_max = cosine_t_max
        self.warmup_epochs = warmup_epochs
        self.temporal_architecture = temporal_architecture
        self.use_bigru = use_bigru
        self.gru_units = gru_units
        self.gru_layers = gru_layers
        self.gru_dropout = gru_dropout
        self.gru_recurrent_dropout = gru_recurrent_dropout
        self.bidirectional_gru = bidirectional_gru
        self.attention_type = attention_type
        self.attention_units = attention_units
        self.cnn_depth = cnn_depth
        self.cnn_filters = cnn_filters
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_dropout = cnn_dropout
        self.use_residual = use_residual
        self.use_se = use_se
        self.se_ratio = se_ratio
        self.n_phases = n_phases
        self.phase_attention_units = phase_attention_units
        self.phase_attention_temperature = phase_attention_temperature
        self.phase_fusion_mode = phase_fusion_mode
        self.branch_encoder_type = branch_encoder_type
        self.branch_cnn_filters = branch_cnn_filters
        self.branch_cnn_kernel_size = branch_cnn_kernel_size
        self.branch_cnn_depth = branch_cnn_depth
        self.branch_dropout = branch_dropout
        self.post_fusion_attention_type = post_fusion_attention_type
        self.post_fusion_attention_units = post_fusion_attention_units
        self.post_fusion_pooling = post_fusion_pooling
        self.bert_num_layers = bert_num_layers
        self.bert_d_model = bert_d_model
        self.bert_num_heads = bert_num_heads
        self.bert_ff_dim = bert_ff_dim
        self.bert_dropout = bert_dropout
        self.bert_attention_dropout = bert_attention_dropout
        self.bert_activation = bert_activation
        self.bert_pooling = bert_pooling
        self.bert_use_cls_token = bert_use_cls_token
        self.bert_positional_encoding = bert_positional_encoding
        self.tof_encoder_type = tof_encoder_type
        self.tof_spatial_filters = tof_spatial_filters
        self.tof_spatial_kernel_size = tof_spatial_kernel_size
        self.tof_spatial_depth = tof_spatial_depth
        self.tof_gru_units = tof_gru_units
        self.fusion_type = fusion_type
        self.fusion_units = fusion_units
        self.fusion_dropout = fusion_dropout
        self.use_orientation_aux_head = use_orientation_aux_head
        self.use_phase_aux_head = use_phase_aux_head
        self.orientation_num_classes = orientation_num_classes
        self.custom_aux_heads = custom_aux_heads
        self.orientation_loss_weight = orientation_loss_weight
        self.phase_loss_weight = phase_loss_weight
        self.phase_label_provider = phase_label_provider
        self.invalid_candidate_policy = invalid_candidate_policy
        self.invalid_candidate_score = invalid_candidate_score
        self.candidate_failure_log = candidate_failure_log

    def _default_extractor(self) -> SequenceExtractor:
        return SequenceExtractor(
            output_format="chunks",
            chunk_window_size=160,
            chunk_stride=160,
            padding_value=0.0,
            add_global_context=False,
            resample_modalities=False,
        )

    def _validate_extractor(self) -> None:
        if hasattr(self.extractor_, "output_format"):
            try:
                self.extractor_.set_params(output_format="chunks")
            except Exception:
                pass
        if hasattr(self.extractor_, "_normalize_chunk_stride"):
            self.extractor_._normalize_chunk_stride()
        validate_sequence_extractor_params(self.extractor_.get_params(), for_frame_output=False)

    def _align_y(self, sequence_ids, y):
        seq_col = getattr(self.extractor_, "sequence_col", "sequence_id")
        sequence_ids = pd.Index(sequence_ids)
        fill_value = "non_bfrb" if self.primary_target == "bfrb" else "Unknown"
        if isinstance(y, pd.DataFrame):
            if self.primary_target not in y.columns:
                raise ValueError(f"y DataFrame does not contain target column: {self.primary_target}")
            if seq_col in y.columns:
                y_seq = y[[seq_col, self.primary_target]].drop_duplicates(seq_col).set_index(seq_col)[self.primary_target]
            elif y.index.name == seq_col:
                y_seq = y[self.primary_target].loc[~y.index.duplicated()]
            else:
                raise ValueError("y DataFrame does not contain sequence_id.")
            aligned = y_seq.reindex(sequence_ids).fillna(fill_value)
            return aligned.to_numpy()
        if isinstance(y, pd.Series):
            if y.index.name == seq_col:
                aligned = y.reindex(sequence_ids).fillna(fill_value)
                return aligned.to_numpy()
            if len(y) == len(sequence_ids):
                return y.fillna(fill_value).to_numpy()
            raise ValueError("Could not align y Series to sequence_ids.")
        y_arr = np.asarray(y)
        if len(y_arr) == len(sequence_ids):
            return y_arr
        raise ValueError("Could not align y to sequence_ids.")

    def _split_validation_sequences(self, X, y):
        seq_col = getattr(self.extractor_, "sequence_col", "sequence_id")
        if seq_col not in X.columns:
            raise ValueError(f"X must contain sequence_id column {seq_col!r}.")

        all_sequence_ids = pd.Index(X[seq_col].drop_duplicates(), name=seq_col)
        if self.validation_split is None or self.validation_split <= 0:
            self.train_sequence_ids_ = all_sequence_ids
            self.validation_sequence_ids_ = pd.Index([], name=seq_col)
            return X, None
        if not 0 < self.validation_split < 1:
            raise ValueError("validation_split must be in [0, 1).")
        if len(all_sequence_ids) < 2:
            raise ValueError("At least two unique sequence_ids are required for validation splitting.")

        sequence_labels = self._align_y(all_sequence_ids, y)
        try:
            train_ids, validation_ids = train_test_split(
                all_sequence_ids.to_numpy(),
                test_size=self.validation_split,
                random_state=self.random_state,
                stratify=sequence_labels,
            )
        except ValueError:
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=self.validation_split,
                random_state=self.random_state,
            )
            train_idx, validation_idx = next(
                splitter.split(all_sequence_ids, groups=all_sequence_ids.to_numpy())
            )
            train_ids = all_sequence_ids.to_numpy()[train_idx]
            validation_ids = all_sequence_ids.to_numpy()[validation_idx]

        self.train_sequence_ids_ = pd.Index(train_ids, name=seq_col)
        self.validation_sequence_ids_ = pd.Index(validation_ids, name=seq_col)
        overlap = self.train_sequence_ids_.intersection(self.validation_sequence_ids_)
        if not overlap.empty:
            raise AssertionError(f"Sequence leakage in internal validation split: {overlap.tolist()}")

        train_mask = X[seq_col].isin(self.train_sequence_ids_)
        validation_mask = X[seq_col].isin(self.validation_sequence_ids_)
        return X.loc[train_mask], X.loc[validation_mask]

    def _augment_training_sequences(self, X):
        if self.augmentor is None:
            self.augmentor_ = None
            return X
        self._fit_stage_ = "augmentation"
        self.augmentor_ = clone(self.augmentor)
        self._configure_augmentation_diagnostics(self.augmentor_, "finetuning")
        if self._uses_processed_augmentation(self.augmentor_):
            # Fit records the RNG/column schema; raw orientation and acceleration
            # reach the cleaner intact. Dropout runs after derived features exist.
            self.augmentor_.fit(X)
            self.augmentor_.diagnose_input(X)
            return X
        return self.augmentor_.fit_transform(X)

    def _uses_processed_augmentation(self, augmentor):
        return type(getattr(self, "extractor_", None)) is SequenceExtractor and isinstance(augmentor, SensorAugmentor)

    @contextmanager
    def _training_feature_augmentation(self, augmentor=None):
        """Install augmentation only during extraction of a training partition."""
        augmentor = getattr(self, "augmentor_", None) if augmentor is None else augmentor
        if not self._uses_processed_augmentation(augmentor):
            yield
            return
        missing = object()
        previous = getattr(self.extractor_, "_training_feature_augmentor", missing)
        self.extractor_._training_feature_augmentor = augmentor
        try:
            yield
        finally:
            if previous is missing:
                del self.extractor_._training_feature_augmentor
            else:
                self.extractor_._training_feature_augmentor = previous

    def _candidate_key(self):
        """Stable identifier for a parameter configuration, including across CV clones."""
        params = {
            name: value for name, value in self.get_params(deep=True).items()
            if name not in {"candidate_failure_log", "invalid_candidate_policy"}
            and not hasattr(value, "get_params")
        }
        encoded = json.dumps(params, default=str, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    def _configure_augmentation_diagnostics(self, augmentor, partition):
        # Runtime context is deliberately not an sklearn/search parameter.
        augmentor.diagnostic_context_ = {
            "candidate_id": self._candidate_key(), "partition": partition,
            "model_random_state": self.random_state,
        }
        # File logging alone must not add per-operation scans to normal searches.
        augmentor.diagnostics_enabled_ = bool(getattr(augmentor, "diagnostics", False))

    def _branch_groups(self, feature_names, separate_derived_imu=False):
        stft_prefixes = [e.feature_prefix for e in getattr(self.extractor_, "stft_extractors", [])]
        cwt_prefixes = [e.feature_prefix for e in getattr(self.extractor_, "cwt_extractors", [])]
        return group_feature_names(
            feature_names,
            stft_prefixes,
            cwt_prefixes,
            separate_derived_imu=separate_derived_imu,
        )

    def _split_branches(self, x_arr, mask_arr, groups):
        branch_arrays = {}
        static_mask = None
        static_counts = None
        for name, idxs in groups.items():
            block = x_arr[:, :, idxs]
            if name in STATIC_BRANCHES:
                if static_mask is None:
                    static_mask = np.asarray(mask_arr, dtype=bool)[:, :, None]
                    static_counts = np.maximum(
                        np.sum(mask_arr, axis=1, dtype=np.float32), 1.0,
                    )[:, None]
                # Feature indexing already created a private block. Mask it
                # in place so the original sum order and float32 values stay
                # unchanged without allocating another feature tensor.
                np.multiply(block, static_mask, out=block)
                branch_arrays[name] = block.sum(axis=1) / static_counts
            else:
                branch_arrays[name] = block
        return branch_arrays

    def _model_inputs(self, branch_arrays, mask):
        model_inputs = [branch_arrays[name] for name in self.input_order_]
        if self.model_requires_mask_:
            model_inputs.append(np.asarray(mask, dtype=np.float32))
        return model_inputs

    def _orientation_labels(self, sequence_ids, y):
        seq_col = getattr(self.extractor_, "sequence_col", "sequence_id")
        if not isinstance(y, pd.DataFrame) or "orientation" not in y.columns:
            raise ValueError(
                "use_orientation_aux_head=True requires an 'orientation' column in y."
            )
        if seq_col in y.columns:
            labels = y[[seq_col, "orientation"]].drop_duplicates(seq_col).set_index(seq_col)["orientation"]
        elif y.index.name == seq_col:
            labels = y["orientation"].loc[~y.index.duplicated()]
        else:
            raise ValueError(f"Orientation auxiliary labels require sequence column {seq_col!r}.")
        labels = labels.reindex(sequence_ids)
        if labels.isna().any():
            raise ValueError("Orientation auxiliary labels are missing for one or more sequences.")
        return labels.to_numpy()

    def _classification_targets(self, labels):
        if str(self.loss_name).lower() == "cce":
            return tf.keras.utils.to_categorical(labels, len(self.classes_))
        return labels

    def _prepare_training_targets(self, labels, sequence_ids, y, phase_labels=None):
        targets = {"predictions": self._classification_targets(labels)}
        weights = {}
        weights["predictions"] = (self.sample_weight_ if self.sample_weight_ is not None else np.ones(len(labels), dtype=np.float32))
        if self.use_orientation_aux_head:
            raw_orientation = self._orientation_labels(sequence_ids, y)
            if not hasattr(self, "orientation_le_"):
                self.orientation_le_ = LabelEncoder().fit(raw_orientation)
            try:
                targets["orientation_aux"] = self.orientation_le_.transform(raw_orientation)
            except ValueError as exc:
                raise ValueError("Orientation validation contains an unseen class.") from exc
            weights["orientation_aux"] = np.ones(len(raw_orientation), dtype=np.float32)
        if self.use_phase_aux_head:
            if phase_labels is None:
                raise ValueError(
                    "use_phase_aux_head=True requires fit parameter 'phase_labels' "
                    "aligned to extracted chunks and timesteps."
                )
            phase_labels = np.asarray(phase_labels)
            if (
                phase_labels.ndim != 2
                or phase_labels.shape[0] != len(labels)
                or phase_labels.shape[1] != self._current_chunk_mask.shape[1]
            ):
                raise ValueError("phase_labels must have shape (number_of_chunks, timesteps).")
            valid = np.asarray(self._current_chunk_mask, dtype=bool)
            valid_labels = phase_labels[valid]
            if (not np.isfinite(valid_labels).all() or
                    np.any(valid_labels != np.floor(valid_labels)) or
                    np.any(valid_labels < 0) or np.any(valid_labels >= self.n_phases)):
                raise ValueError(f"phase_labels must be in [0, {self.n_phases}).")
            phase_labels = np.where(valid, phase_labels, 0).astype(np.int32)
            targets["phase_aux"] = phase_labels
            weights["phase_aux"] = np.asarray(self._current_chunk_mask, dtype=np.float32)
        for spec in getattr(self, "custom_aux_heads_", []):
            raw = self._aux_column_labels(sequence_ids, y, spec["target"])
            if spec["task"] == "classification":
                raw = self.custom_aux_encoders_[spec["name"]].transform(raw)
            else:
                raw = np.asarray(raw, dtype=np.float32).reshape(-1, 1)
                if not np.isfinite(raw).all():
                    raise ValueError(f"Non-finite numeric labels for {spec['name']}.")
            targets[spec["name"]] = raw
            weights[spec["name"]] = np.ones(len(labels), dtype=np.float32)
        if len(targets) == 1:
            return targets["predictions"], self.sample_weight_
        return targets, weights

    def _transform_to_branches(self, X):
        check_is_fitted(self, ["extractor_", "branch_groups_"])
        try:
            chunks = self.extractor_.transform_chunks(X)
        except InvalidExtractorParams:
            raise
        except Exception as exc:
            raise InvalidExtractorParams(f"MultiBranchSequenceClassifier transform failed: {exc}") from exc
        branch_arrays = self._split_branches(chunks["X"], chunks["mask"], self.branch_groups_)
        return branch_arrays, chunks["sequence_ids"]

    def _build_callbacks(self, has_validation_data: bool = False):
        cb = []
        if self.early_stopping_patience and self.early_stopping_patience > 0:
            cb.append(callbacks.EarlyStopping(
                monitor="val_loss" if has_validation_data else "loss",
                patience=self.early_stopping_patience,
                restore_best_weights=True,
            ))
        if self.use_lr_scheduler:
            if str(self.lr_scheduler_type).lower() == "plateau" and has_validation_data:
                cb.append(callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=self.lr_factor,
                    patience=self.lr_patience,
                    min_lr=self.min_lr,
                    verbose=0,
                ))
            elif str(self.lr_scheduler_type).lower() == "cosine":
                base_lr = float(self.learning_rate)
                warmup_epochs = max(0, int(self.warmup_epochs))
                t_max = max(1, int(self.cosine_t_max))

                def cosine_schedule(epoch, current_lr):
                    if warmup_epochs and epoch < warmup_epochs:
                        return base_lr * (epoch + 1) / warmup_epochs
                    progress = (epoch - warmup_epochs) % t_max
                    cosine = 0.5 * (1.0 + math.cos(math.pi * progress / t_max))
                    return float(self.min_lr) + (base_lr - float(self.min_lr)) * cosine

                cb.append(callbacks.LearningRateScheduler(cosine_schedule, verbose=0))
        return cb

    def fit(self, X, y=None, **fit_params):
        if self.invalid_candidate_policy not in {"raise", "penalize"}:
            raise ValueError("invalid_candidate_policy must be 'raise' or 'penalize'.")
        if not np.isfinite(self.invalid_candidate_score) or self.invalid_candidate_score >= 0:
            raise ValueError("invalid_candidate_score must be finite and negative.")
        self.fit_failed_ = False
        self.fit_error_ = None
        self.fit_failure_ = None
        self._fit_stage_ = "validation"
        try:
            return self._fit_impl(X, y, **fit_params)
        except Exception as exc:
            self.fit_error_ = f"{type(exc).__name__}: {exc}"
            stage = getattr(exc, "failure_stage", getattr(self, "_fit_stage_", "fit"))
            self._record_candidate_failure(stage, self.fit_error_, exc)
            if self.invalid_candidate_policy == "raise":
                raise
            self.fit_failed_ = True
            self.history_ = {}
            if hasattr(self, "model_"):
                del self.model_
            warnings.warn(f"Candidate skipped: {self.fit_error_}", RuntimeWarning)
            return self

    def _record_candidate_failure(self, stage, error, exception=None):
        """Keep a bounded diagnostic record without retaining data or exception frames."""
        formatted = ("".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
                     if exception is not None else traceback.format_exc())
        if len(formatted) > 16384:
            formatted = formatted[:8192] + "\n... traceback truncated ...\n" + formatted[-8192:]
        diagnostics = getattr(exception, "augmentation_diagnostics", None)
        if diagnostics is None:
            reports = {
                partition: getattr(getattr(self, attribute, None), "diagnostics_", [])
                for partition, attribute in (("finetuning", "augmentor_"), ("pretraining", "pretraining_augmentor_"))
            }
            diagnostics = {partition: report for partition, report in reports.items() if report} or None
        record = {
            "candidate_id": self._candidate_key(), "stage": stage, "error": error,
            "exception_type": type(exception).__name__ if exception is not None else None,
            "original_exception": (f"{type(exception.__cause__).__name__}: {exception.__cause__}"
                                   if exception is not None and exception.__cause__ is not None else error),
            "traceback": formatted, "augmentation_diagnostics": diagnostics,
            "augmentation_seed": getattr(self.augmentor, "seed", None),
            "pretraining_augmentation_seed": getattr(getattr(self, "pretraining_augmentor", None), "seed", None),
            "model_random_state": self.random_state,
        }
        self.fit_failure_ = record
        if self.candidate_failure_log is not None:
            try:
                path = Path(self.candidate_failure_log)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({**record, "params": self.get_params(deep=False)}, default=str) + "\n")
            except Exception as logging_error:
                record["logging_error"] = f"{type(logging_error).__name__}: {logging_error}"
                warnings.warn(f"Candidate failure log could not be written: {logging_error}", RuntimeWarning)
        if self.verbose:
            print(f"Candidate skipped ({stage}): {error}")

    def _aux_column_labels(self, sequence_ids, y, column):
        seq_col = getattr(self.extractor_, "sequence_col", "sequence_id")
        if not isinstance(y, pd.DataFrame) or column not in y:
            raise ValueError(f"Auxiliary target {column!r} requires that column in y.")
        if seq_col in y.columns:
            labels = y[[seq_col, column]]
            grouped = labels.groupby(seq_col, sort=False)[column]
            unique_labels = labels.drop_duplicates(seq_col).set_index(seq_col)[column]
        elif y.index.name == seq_col:
            labels = y[column]
            grouped = labels.groupby(level=seq_col, sort=False)
            unique_labels = labels.loc[~labels.index.duplicated()]
        else:
            raise ValueError(f"Auxiliary labels require sequence column {seq_col!r}.")
        if (grouped.nunique(dropna=False) > 1).any():
            raise ValueError(f"Auxiliary target {column!r} must be constant within each sequence.")
        aligned = unique_labels.reindex(sequence_ids)
        if aligned.isna().any():
            raise ValueError(f"Missing auxiliary labels for {column!r}.")
        return aligned.to_numpy()

    def _extract_training_chunks(self, X_train):
        with self._training_feature_augmentation():
            return self._extract_training_chunks_impl(X_train)

    def _extract_training_chunks_impl(self, X_train):
        self._fit_stage_ = "extraction"
        try:
            # STFT/CWT features summarize each whole sequence; split sequences
            # first, then fit/transform train and validation partitions separately.
            if (
                isinstance(self.extractor_, SequenceExtractor)
                and (
                    type(self.extractor_) is SequenceExtractor
                    or type(self.extractor_).fit_transform is not SequenceExtractor.fit_transform
                )
                and "fit" not in vars(self.extractor_)
                and "transform_chunks" not in vars(self.extractor_)
                and "transform" not in vars(self.extractor_)
            ):
                # The native extractor reuses fitted training features. A
                # subclass must explicitly opt in through fit_transform so
                # customized fit/transform methods retain their usual effect.
                train_chunks = self.extractor_.fit_transform(X_train)
            else:
                self.extractor_.fit(X_train)
                train_chunks = self.extractor_.transform_chunks(X_train)
            return train_chunks
        except InvalidExtractorParams:
            raise
        except Exception as exc:
            error = InvalidExtractorParams(
                f"MultiBranchSequenceClassifier fit failed during extraction: {exc}"
            )
            for name in ("failure_stage", "augmentation_diagnostics"):
                if hasattr(exc, name):
                    setattr(error, name, getattr(exc, name))
            raise error from exc

    def _build_model(self, time_series_shapes, static_shapes, n_classes, sequence_length, **overrides):
        """Build the shared backbone; subclasses may supply stage-specific heads/settings."""
        settings = dict(
            time_series_shapes=time_series_shapes,
            static_shapes=static_shapes,
            n_classes=n_classes,
            branch_filters=self.branch_filters,
            branch_kernel_sizes=self.branch_kernel_sizes,
            branch_pool_sizes=self.branch_pool_sizes,
            branch_num_conv_layers=self.branch_num_conv_layers,
            branch_filter_mode=self.branch_filter_mode,
            branch_dense_units=self.branch_dense_units,
            static_branch_units=self.static_branch_units,
            static_dropout=self.static_dropout,
            dense_units=self.dense_units,
            dropout=self.dropout,
            conv_dropout=self.conv_dropout,
            learning_rate=self.learning_rate,
            use_batchnorm=self.use_batchnorm,
            activation=self.activation,
            pooling_mode=self.pooling_mode,
            optimizer_name=self.optimizer_name,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            gradient_clip_norm=self.gradient_clip_norm,
            gradient_clip_value=self.gradient_clip_value,
            loss_name=self.loss_name,
            label_smoothing=self.label_smoothing,
            random_state=self.random_state,
            temporal_architecture=self.temporal_architecture,
            use_bigru=self.use_bigru,
            gru_units=self.gru_units,
            gru_layers=self.gru_layers,
            gru_dropout=self.gru_dropout,
            gru_recurrent_dropout=self.gru_recurrent_dropout,
            bidirectional_gru=self.bidirectional_gru,
            attention_type=self.attention_type,
            attention_units=self.attention_units,
            cnn_depth=self.cnn_depth,
            cnn_filters=self.cnn_filters,
            cnn_kernel_size=self.cnn_kernel_size,
            cnn_dropout=self.cnn_dropout,
            use_residual=self.use_residual,
            use_se=self.use_se,
            se_ratio=self.se_ratio,
            n_phases=self.n_phases,
            phase_attention_units=self.phase_attention_units,
            phase_attention_temperature=self.phase_attention_temperature,
            phase_fusion_mode=self.phase_fusion_mode,
            branch_encoder_type=self.branch_encoder_type,
            branch_cnn_filters=self.branch_cnn_filters,
            branch_cnn_kernel_size=self.branch_cnn_kernel_size,
            branch_cnn_depth=self.branch_cnn_depth,
            branch_dropout=self.branch_dropout,
            post_fusion_attention_type=self.post_fusion_attention_type,
            post_fusion_attention_units=self.post_fusion_attention_units,
            post_fusion_pooling=self.post_fusion_pooling,
            bert_num_layers=self.bert_num_layers,
            bert_d_model=self.bert_d_model,
            bert_num_heads=self.bert_num_heads,
            bert_ff_dim=self.bert_ff_dim,
            bert_dropout=self.bert_dropout,
            bert_attention_dropout=self.bert_attention_dropout,
            bert_activation=self.bert_activation,
            bert_pooling=self.bert_pooling,
            bert_use_cls_token=self.bert_use_cls_token,
            bert_positional_encoding=self.bert_positional_encoding,
            tof_encoder_type=self.tof_encoder_type,
            tof_spatial_filters=self.tof_spatial_filters,
            tof_spatial_kernel_size=self.tof_spatial_kernel_size,
            tof_spatial_depth=self.tof_spatial_depth,
            tof_gru_units=self.tof_gru_units,
            fusion_type=self.fusion_type,
            fusion_units=self.fusion_units,
            fusion_dropout=self.fusion_dropout,
            use_orientation_aux_head=self.use_orientation_aux_head,
            use_phase_aux_head=self.use_phase_aux_head,
            orientation_num_classes=(len(self.orientation_le_.classes_) if self.use_orientation_aux_head and hasattr(self, "orientation_le_") else self.orientation_num_classes),
            custom_aux_heads=self.custom_aux_heads_,
            orientation_loss_weight=self.orientation_loss_weight,
            phase_loss_weight=self.phase_loss_weight,
            sequence_length=sequence_length,
        )

        settings.update(overrides)
        return build_multibranch_model(**settings)

    def _before_supervised_fit(self, *, time_series_shapes, static_shapes, sequence_length,
                               model_inputs, train_targets, sample_weight, validation_data):
        """Optional pretraining hook; the ordinary classifier has no extra stage."""
        return None

    def _fit_impl(self, X, y=None, **fit_params):
        if y is None:
            raise ValueError("MultiBranchSequenceClassifier requires y.")
        # Validate the effective architecture before expensive feature extraction.
        # Inactive search dimensions remain sklearn parameters but cannot reject
        # an otherwise valid model (e.g. shared CNN depth under a BERT backbone).
        error = candidate_error(
            {}, self.get_params(deep=True), y=y,
            phase_labels_available=fit_params.get("phase_labels") is not None,
        )
        if error is not None:
            raise ValueError(error)
        if (
            self.use_phase_aux_head and self.phase_label_provider is None
            and self.validation_split is not None and self.validation_split > 0
            and fit_params.get("validation_phase_labels") is None
        ):
            raise ValueError("Internal validation with phase supervision requires validation_phase_labels or phase_label_provider.")
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow/Keras is required to fit MultiBranchSequenceClassifier.")
        if tf is not None:
            # Serial population trials must release Keras graph/name state.
            tf.keras.backend.clear_session()

        self.extractor_ = clone(self.extractor) if self.extractor is not None else self._default_extractor()
        self._validate_extractor()
        self.custom_aux_heads_ = _normalise_aux_heads(self.custom_aux_heads)

        self._fit_stage_ = "partitioning"
        X_train, X_validation = self._split_validation_sequences(X, y)
        validation_data = None
        if X_validation is not None and X_validation.empty:
            raise ValueError("Internal validation split produced no validation rows.")

        X_train = self._augment_training_sequences(X_train)

        train_chunks = self._extract_training_chunks(X_train)
        self._fit_stage_ = "extraction"
        validation_chunks = (
            self.extractor_.transform_chunks(X_validation)
            if X_validation is not None else None
        )

        self.feature_names_ = list(train_chunks["feature_names"])
        self.branch_groups_ = self._branch_groups(
            self.feature_names_,
            separate_derived_imu=(
                str(self.temporal_architecture).lower() != "bigru_attention"
                or self.use_bigru
                or self.use_orientation_aux_head
                or self.use_phase_aux_head
                or bool(self.custom_aux_heads_)
            ),
        )
        if not self.branch_groups_:
            raise InvalidExtractorParams("No modality branches could be built from the extracted features.")

        branch_arrays = self._split_branches(train_chunks["X"], train_chunks["mask"], self.branch_groups_)
        if not all(np.isfinite(arr).all() for arr in branch_arrays.values()):
            raise InvalidExtractorParams("Feature extraction produced non-finite branch values.")

        time_series_shapes = {name: arr.shape[1:] for name, arr in branch_arrays.items() if name in TIME_SERIES_BRANCHES}
        static_shapes = {name: arr.shape[1] for name, arr in branch_arrays.items() if name in STATIC_BRANCHES}
        self.input_order_ = list(time_series_shapes.keys()) + list(static_shapes.keys())
        self.model_requires_mask_ = (
            str(self.temporal_architecture).lower() != "bigru_attention"
            or self.use_bigru
            or self.use_orientation_aux_head
            or self.use_phase_aux_head
            or bool(self.custom_aux_heads_)
        )
        model_inputs = self._model_inputs(branch_arrays, train_chunks["mask"])

        y_aligned = self._align_y(train_chunks["sequence_ids"], y)
        self.le_ = LabelEncoder()
        self.le_.fit(y_aligned)
        self.classes_ = self.le_.classes_
        y_enc = self.le_.transform(y_aligned)

        validation_arrays = None
        validation_inputs = None
        validation_labels = None
        if validation_chunks is not None:
            train_chunk_ids = pd.Index(train_chunks["sequence_ids"])
            validation_chunk_ids = pd.Index(validation_chunks["sequence_ids"])
            overlap = train_chunk_ids.unique().intersection(validation_chunk_ids.unique())
            if not overlap.empty:
                raise AssertionError(f"Sequence leakage between train and validation chunks: {overlap.tolist()}")
            validation_arrays = self._split_branches(
                validation_chunks["X"], validation_chunks["mask"], self.branch_groups_
            )
            validation_inputs = self._model_inputs(validation_arrays, validation_chunks["mask"])
            validation_y = self._align_y(validation_chunks["sequence_ids"], y)
            try:
                validation_labels = self.le_.transform(validation_y)
            except ValueError as exc:
                raise ValueError(
                    "Internal validation contains a target class absent from the training split."
                ) from exc
            validation_data = (validation_inputs, self._classification_targets(validation_labels))

        sample_weight = None
        if self.class_weight_mode == "balanced":
            uniq_classes = np.unique(y_enc)
            weights = compute_class_weight("balanced", classes=uniq_classes, y=y_enc)
            weight_map = dict(zip(uniq_classes, weights))
            sample_weight = np.array([weight_map[v] for v in y_enc], dtype=np.float32)
        self.sample_weight_ = sample_weight

        if self.use_orientation_aux_head:
            orientation_training = self._orientation_labels(train_chunks["sequence_ids"], y)
            self.orientation_le_ = LabelEncoder().fit(orientation_training)
        self.custom_aux_encoders_ = {}
        for spec in self.custom_aux_heads_:
            raw = self._aux_column_labels(train_chunks["sequence_ids"], y, spec["target"])
            if spec["task"] == "classification":
                encoder = LabelEncoder().fit(raw)
                self.custom_aux_encoders_[spec["name"]] = encoder
                spec["num_classes"] = len(encoder.classes_)
        phase_training = fit_params.pop("phase_labels", None)
        phase_validation = fit_params.pop("validation_phase_labels", None)
        if self.use_phase_aux_head and self.phase_label_provider is not None:
            phase_training = self.phase_label_provider(X_train, train_chunks)
            if validation_chunks is not None:
                phase_validation = self.phase_label_provider(X_validation, validation_chunks)
        del X_train, X_validation
        self._current_chunk_mask = train_chunks["mask"]
        train_targets, model_sample_weight = self._prepare_training_targets(
            y_enc,
            train_chunks["sequence_ids"],
            y,
            phase_training,
        )
        if validation_chunks is not None and (
            self.use_orientation_aux_head or self.use_phase_aux_head or bool(self.custom_aux_heads_)
        ):
            self._current_chunk_mask = validation_chunks["mask"]
            training_weights = self.sample_weight_
            self.sample_weight_ = None
            validation_targets, validation_weights = self._prepare_training_targets(
                validation_labels,
                validation_chunks["sequence_ids"],
                y,
                phase_validation,
            )
            self.sample_weight_ = training_weights
            validation_data = (validation_inputs, validation_targets, validation_weights)

        # Branch indexing creates independent arrays. Providers have now read
        # the original chunk dictionaries; release their duplicate feature
        # tensors before TensorFlow copies the model inputs to its runtime.
        del train_chunks["X"]
        if validation_chunks is not None:
            del validation_chunks["X"]

        self._fit_stage_ = "training"
        self.model_ = self._build_model(
            time_series_shapes, static_shapes, len(self.classes_), train_chunks["mask"].shape[1],
        )
        self._before_supervised_fit(
            time_series_shapes=time_series_shapes, static_shapes=static_shapes,
            sequence_length=train_chunks["mask"].shape[1], model_inputs=model_inputs,
            train_targets=train_targets, sample_weight=model_sample_weight,
            validation_data=validation_data,
        )

        cb = self._build_callbacks(has_validation_data=validation_data is not None)

        history = self.model_.fit(
            model_inputs, train_targets,
            sample_weight=model_sample_weight,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            validation_split=0.0,
            callbacks=cb,
            verbose=self.verbose,
        )
        self.history_ = history.history
        return self

    def predict_proba(self, X):
        check_is_fitted(self, ["model_", "le_", "input_order_"])
        chunks = self.extractor_.transform_chunks(X)
        branch_arrays = self._split_branches(
            chunks["X"], chunks["mask"], self.branch_groups_
        )
        sequence_ids = chunks["sequence_ids"]
        model_inputs = self._model_inputs(branch_arrays, chunks["mask"])
        # Advanced feature indexing gives each branch its own array; the
        # combined tensor need not overlap TensorFlow's prediction allocations.
        del chunks["X"]
        probs = self.model_.predict(model_inputs, verbose=0)
        if isinstance(probs, dict):
            probs = probs["predictions"]
        elif isinstance(probs, (list, tuple)):
            probs = probs[self.model_.output_names.index("predictions")]
        proba_df = pd.DataFrame(probs, columns=self.le_.classes_)
        proba_df.insert(0, "sequence_id", sequence_ids)
        agg = proba_df.groupby("sequence_id", sort=True).mean()
        agg.index.name = getattr(self.extractor_, "sequence_col", "sequence_id")
        return agg

    def predict(self, X):
        proba = self.predict_proba(X)
        pred_idx = np.argmax(proba.to_numpy(), axis=1)
        preds = self.le_.inverse_transform(pred_idx)
        return pd.Series(preds, index=proba.index, name=self.primary_target).sort_index()

    def score(self, X, y, sample_weight=None):
        if getattr(self, "fit_failed_", False):
            return float(self.invalid_candidate_score)
        preds = self.predict(X)
        seq_col = getattr(self.extractor_, "sequence_col", "sequence_id")
        if isinstance(y, pd.DataFrame):
            columns = [self.primary_target]
            if "is_target" in y.columns and self.primary_target != "is_target":
                columns.append("is_target")
            if seq_col in y.columns:
                columns.insert(0, seq_col)
            yy = y[columns]
            if seq_col not in yy.columns:
                if yy.index.name == seq_col:
                    yy = yy.reset_index()
                else:
                    raise ValueError("y DataFrame must contain sequence_id.")
            y_seq = yy.drop_duplicates(seq_col).sort_values(seq_col)
            y_seq = y_seq[y_seq[seq_col].isin(preds.index)]
            preds_aligned = preds.reindex(y_seq[seq_col]).to_numpy()
            y_true = y_seq[self.primary_target].values
            if "is_target" in y_seq.columns:
                y_true_binary = y_seq["is_target"].astype(int).values
            else:
                y_true_binary = (y_true != "non_bfrb").astype(int)
            if self.primary_target == "bfrb":
                return competition_score(y_true, preds_aligned, y_true_binary=y_true_binary, target_only_macro=True)
            return f1_score(y_true, preds_aligned, average="macro", zero_division=0)
        y_arr = np.asarray(y)
        preds_arr = preds.to_numpy() if hasattr(preds, "to_numpy") else np.asarray(preds)
        return f1_score(y_arr, preds_arr, average="macro", zero_division=0)

    def get_history_dict(self): return getattr(self, "history_", {})
    def summarize_model(self):
        if hasattr(self, "model_"): self.model_.summary()
        else: print("Model is not fitted yet.")
    def branch_summary(self):
        check_is_fitted(self, ["feature_names_", "branch_groups_"])
        return describe_branches(self.feature_names_, self.branch_groups_)


"""Architecture validation and parameter-preserving search helpers.

Validation checks the selected model path without changing candidate settings.
Search caches use complete requested parameter mappings.
"""

ARCHITECTURES = {"bigru_attention", "phase_attention", "cnn_attention_pool", "bert"}
_BERT = {
    "bert_num_layers", "bert_d_model", "bert_num_heads", "bert_ff_dim",
    "bert_dropout", "bert_attention_dropout", "bert_activation", "bert_pooling",
    "bert_use_cls_token", "bert_positional_encoding",
}
_GRU = {"gru_units", "gru_layers", "gru_dropout", "gru_recurrent_dropout", "bidirectional_gru"}
_SHARED_CNN = {"cnn_depth", "cnn_filters", "cnn_kernel_size", "cnn_dropout"}
_PHASE = {"phase_attention_units", "phase_attention_temperature", "phase_fusion_mode"}
_POST_FUSION = {"post_fusion_attention_type", "post_fusion_attention_units", "post_fusion_pooling"}
_BRANCH_CNN = {"branch_encoder_type", "branch_cnn_filters", "branch_cnn_kernel_size", "branch_cnn_depth", "branch_dropout"}
_LEGACY_BRANCH = {
    "branch_filters", "branch_kernel_sizes", "branch_pool_sizes", "branch_num_conv_layers",
    "branch_filter_mode", "branch_dense_units", "conv_dropout", "pooling_mode",
}
_TEMPORAL = _BRANCH_CNN | {"fusion_type", "fusion_units", "fusion_dropout", "use_residual", "use_se", "se_ratio"}
_TOF_SPATIAL = {"tof_spatial_filters", "tof_spatial_kernel_size", "tof_spatial_depth", "tof_gru_units"}
_DEFAULTS = {
    "temporal_architecture": "bigru_attention", "use_bigru": False,
    "use_orientation_aux_head": False, "use_phase_aux_head": False,
    "custom_aux_heads": None, "branch_encoder_type": "conv1d", "tof_encoder_type": "flatten_1d",
    "bert_d_model": 64, "bert_num_heads": 4, "bert_num_layers": 2, "bert_ff_dim": 128,
    "bert_pooling": "attention", "bert_use_cls_token": False, "bert_positional_encoding": "learned",
    "n_phases": 3, "phase_attention_temperature": 1.0, "phase_fusion_mode": "concat",
    "phase_label_provider": None, "fusion_type": "concat", "attention_type": "additive",
    "post_fusion_pooling": "attention", "post_fusion_attention_type": "additive",
    "use_se": False, "pooling_mode": "attention", "branch_filter_mode": "custom",
    "optimizer_name": "adam", "loss_name": "sparse_cce", "use_lr_scheduler": False,
    "lr_scheduler_type": "plateau", "validation_split": 0.15,
}

_EXTRA_CONTROLLERS = (
    "optimizer_name", "loss_name", "use_lr_scheduler", "validation_split", "lr_scheduler_type",
    "augmentor", "augmentor__prob", "augmentor__per_aug_prob", "augmentor__augmentations",
    "extractor__output_format", "extractor__filter_problematic_sequences",
    "extractor__resample_modalities", "extractor__motion_filter_mode",
    "extractor__stft_configs", "extractor__cwt_configs", "extractor__stft_window_seconds",
    "extractor__linear_acc_mode", "extractor__use_highpass_fallback",
    "extractor__acc_modes", "extractor__smooth_alpha", "extractor__rotation_modes",
    "extractor__use_acc_magnitude", "extractor__linear_acc_modes", "extractor__use_linear_acc_magnitude",
    "extractor__use_dead_reckoning", "extractor__max_windows_per_sequence",
    "extractor__sequence_crop_mode", "extractor__maxlen", "extractor__use_chunk_stride_ratio",
)




def _name(key):
    return key[len("estimator__"):] if key.startswith("estimator__") else key


def _merged(params, base_params):
    merged = dict(_DEFAULTS)
    for settings in (base_params or {}, params or {}):
        for key, value in settings.items():
            merged[_name(key)] = value
    return merged


def _extractor_settings(settings):
    extractor = settings.get("extractor")
    if extractor is None:
        result = {"chunk_window_size": 160, "maxlen": 160, "sequence_col": "sequence_id"}
    elif isinstance(extractor, dict):
        result = dict(extractor)
    elif hasattr(extractor, "get_params"):
        result = extractor.get_params(deep=True)
    else:
        result = vars(extractor).copy()
    for key, value in settings.items():
        if key.startswith("extractor__"):
            result[key[len("extractor__"):]] = value
    return result


def _modes(value):
    if value is None:
        return set()
    if isinstance(value, str):
        return {mode.strip().lower() for mode in value.split("|") if mode.strip()}
    return {str(mode).strip().lower() for mode in value}


def _heads(value):
    return json.loads(value) if isinstance(value, str) else value


def _is_temporal(settings):
    try:
        custom = bool(_heads(settings.get("custom_aux_heads")))
    except (TypeError, ValueError):
        custom = True  # Keep malformed specifications active so preflight rejects them.
    return (
        str(settings["temporal_architecture"]).lower() != "bigru_attention"
        or bool(settings["use_bigru"])
        or bool(settings["use_orientation_aux_head"])
        or bool(settings["use_phase_aux_head"])
        or custom
    )




def _positive_integer(value):
    return isinstance(value, Integral) and not isinstance(value, bool) and value > 0


def _finite_number(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _positive_tree(value):
    if isinstance(value, str):
        value = json.loads(value)
    if value is None:
        return True  # Per-branch None invokes model defaults.
    if isinstance(value, dict):
        return bool(value) and all(_positive_tree(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return bool(value) and all(_positive_tree(item) for item in value)
    return _positive_integer(value)


def candidate_error(params, base_params=None, y=None, phase_labels_available=False, label_cache=None):
    """Return a preflight rejection reason, or ``None`` for a valid candidate.

    ``y=None`` omits data-dependent label checks. Phase supervision still needs
    a callable provider or ``phase_labels_available=True``. A search may provide
    a fresh ``label_cache`` dictionary for each fit to reuse column validation
    for its unchanged label table. Individual estimator fits need no cache.
    Only active knobs
    are validated: an unused CNN/GRU value cannot reject a BERT candidate.
    """
    try:
        return _candidate_error(params, base_params, y, phase_labels_available, label_cache)
    except (TypeError, ValueError, AttributeError, OverflowError) as exc:
        return f"Invalid candidate settings: {exc}"


def _candidate_error(params, base_params, y, phase_labels_available, label_cache):
    merged = _merged(params, base_params)
    settings = merged
    architecture = str(settings["temporal_architecture"]).lower()
    if architecture not in ARCHITECTURES:
        return f"Unsupported temporal_architecture: {architecture!r}."
    for name in ("use_bigru", "use_orientation_aux_head", "use_phase_aux_head", "use_residual", "use_se", "bert_use_cls_token"):
        if name in settings and not isinstance(settings[name], (bool, Integral)):
            return f"{name} must be True or False."
    raw_heads = _heads(settings.get("custom_aux_heads"))
    if raw_heads is not None and raw_heads != []:
        heads = _normalise_aux_heads(raw_heads)
    else:
        heads = []
    temporal = _is_temporal(merged)
    extractor = _extractor_settings(merged)
    # These names control validation only. Never pass a filtered mapping to fit.
    validation_names = set(settings)
    if architecture != "bert":
        validation_names.difference_update(_BERT)
    if architecture != "bigru_attention" or not settings["use_bigru"]:
        validation_names.difference_update(_GRU)
    if architecture not in {"phase_attention", "cnn_attention_pool"}:
        validation_names.difference_update(_SHARED_CNN)
    if architecture != "phase_attention":
        validation_names.difference_update(_PHASE)
    if architecture != "cnn_attention_pool":
        validation_names.difference_update(_POST_FUSION)
    if temporal:
        validation_names.difference_update(_LEGACY_BRANCH)
    else:
        validation_names.difference_update(_TEMPORAL | _TOF_SPATIAL | {"tof_encoder_type"})
    if not settings["use_orientation_aux_head"]:
        validation_names.difference_update({"orientation_num_classes", "orientation_loss_weight"})
    if not settings["use_phase_aux_head"]:
        validation_names.difference_update({"phase_loss_weight", "phase_label_provider"})
    if not settings.get("use_se", False):
        validation_names.discard("se_ratio")
    uses_attention = (architecture == "bigru_attention" and temporal
                      or architecture == "bert" and str(settings["bert_pooling"]).lower() == "attention")
    if not uses_attention:
        validation_names.difference_update({"attention_type", "attention_units"})
    elif str(settings.get("attention_type", "additive")).lower() == "linear":
        validation_names.discard("attention_units")
    if architecture == "cnn_attention_pool":
        if str(settings["post_fusion_pooling"]).lower() not in {"attention", "attention_avg_max"}:
            validation_names.difference_update({"post_fusion_attention_type", "post_fusion_attention_units"})
        elif str(settings["post_fusion_attention_type"]).lower() == "linear":
            validation_names.discard("post_fusion_attention_units")
    if not _modes(extractor.get("tof_modes", "sensor_stats")):
        validation_names.difference_update(_TOF_SPATIAL | {"tof_encoder_type"})
    elif str(settings.get("tof_encoder_type", "flatten_1d")).lower() == "flatten_1d":
        validation_names.difference_update(_TOF_SPATIAL)
    elif str(settings.get("tof_encoder_type", "flatten_1d")).lower() != "spatial_cnn_gru":
        validation_names.discard("tof_gru_units")
    if temporal:
        if str(settings.get("branch_encoder_type", "conv1d")).lower() not in {"conv1d", "cnn"}:
            return "branch_encoder_type must be 'conv1d' or 'cnn'."
        if settings.get("fusion_type", "concat") not in {"concat", "sum", "attention"}:
            return "fusion_type must be 'concat', 'sum', or 'attention'."
    elif settings.get("pooling_mode", "attention") not in {"attention", "mean", "max", "last"}:
        return "pooling_mode must be 'attention', 'mean', 'max', or 'last'."
    if not temporal and settings.get("branch_filter_mode", "custom") not in {"custom", "double", "constant"}:
        return "branch_filter_mode must be 'custom', 'double', or 'constant'."
    positive = {
        "dense_units", "static_branch_units", "branch_dense_units", "epochs", "batch_size",
        "branch_cnn_filters", "branch_cnn_kernel_size", "branch_cnn_depth", "fusion_units",
        "cnn_depth", "cnn_filters", "cnn_kernel_size", "gru_units", "gru_layers", "attention_units",
        "phase_attention_units", "post_fusion_attention_units", "bert_num_layers", "bert_d_model",
        "bert_num_heads", "bert_ff_dim", "tof_spatial_filters", "tof_spatial_kernel_size",
        "tof_spatial_depth", "tof_gru_units", "se_ratio",
    }
    for name in positive:
        if name in validation_names and not _positive_integer(settings[name]):
            return f"{name} must be a positive integer."
    for name in ("branch_filters", "branch_kernel_sizes", "branch_pool_sizes", "branch_num_conv_layers"):
        if name in validation_names and not _positive_tree(settings[name]):
            return f"{name} must contain positive integers."
    for name in {"dropout", "static_dropout", "conv_dropout", "branch_dropout", "fusion_dropout",
                 "cnn_dropout", "gru_dropout", "gru_recurrent_dropout", "bert_dropout", "bert_attention_dropout"}:
        if name in validation_names and (not _finite_number(settings[name]) or not 0 <= settings[name] < 1):
            return f"{name} must be in [0, 1)."
    if "learning_rate" in settings and (not _finite_number(settings["learning_rate"]) or settings["learning_rate"] <= 0):
        return "learning_rate must be finite and positive."
    for name in ("orientation_loss_weight", "phase_loss_weight"):
        if name in validation_names and (not _finite_number(settings[name]) or settings[name] < 0):
            return f"{name} must be finite and nonnegative."
    if architecture == "bert":
        if settings["bert_d_model"] % settings["bert_num_heads"]:
            return "bert_d_model must be divisible by bert_num_heads."
        if str(settings["bert_pooling"]).lower() not in {"cls", "attention", "avg", "avg_max"}:
            return "bert_pooling must be 'cls', 'attention', 'avg', or 'avg_max'."
        if str(settings["bert_pooling"]).lower() == "cls" and not settings["bert_use_cls_token"]:
            return "bert_pooling='cls' requires bert_use_cls_token=True."
        if str(settings["bert_positional_encoding"]).lower() not in {"learned", "sinusoidal"}:
            return "bert_positional_encoding must be 'learned' or 'sinusoidal'."
    if architecture == "phase_attention" or settings["use_phase_aux_head"]:
        if not _positive_integer(settings["n_phases"]) or settings["n_phases"] < 2:
            return "n_phases must be an integer >= 2."
    if architecture == "phase_attention":
        value = settings["phase_attention_temperature"]
        if not _finite_number(value) or value <= 0:
            return "phase_attention_temperature must be finite and positive."
        if settings["phase_fusion_mode"] not in {"concat", "sum", "learned"}:
            return "phase_fusion_mode must be 'concat', 'sum', or 'learned'."
    uses_attention = (architecture == "bigru_attention" and temporal
                      or architecture == "bert" and str(settings["bert_pooling"]).lower() == "attention")
    if uses_attention and settings.get("attention_type", "additive") not in {"additive", "linear"}:
        return "attention_type must be 'additive' or 'linear'."
    if architecture == "cnn_attention_pool":
        pooling = settings["post_fusion_pooling"]
        if pooling not in {"attention", "avg", "max", "avg_max", "attention_avg_max"}:
            return "Invalid post_fusion_pooling."
        if pooling in {"attention", "attention_avg_max"} and settings["post_fusion_attention_type"] not in {"additive", "linear"}:
            return "post_fusion_attention_type must be 'additive' or 'linear'."
    if "tof_encoder_type" in validation_names:
        encoder = str(settings["tof_encoder_type"]).lower()
        if encoder not in {"flatten_1d", "spatial_cnn", "spatial_cnn_gru"}:
            return "Unsupported tof_encoder_type."
        if encoder in {"spatial_cnn", "spatial_cnn_gru"} and _modes(extractor.get("tof_modes", "sensor_stats")) != {"raw"}:
            return "Spatial ToF encoders require extractor__tof_modes='raw' (without summary modes)."
    if settings["use_phase_aux_head"]:
        provider = settings.get("phase_label_provider")
        if provider is not None and not callable(provider):
            return "phase_label_provider must be callable."
        if provider is None and not phase_labels_available:
            return "Phase supervision requires phase_label_provider or explicit phase_labels."
    windows = [head for head in heads if head.get("window_size") is not None]
    if windows:
        defaults = {"acc_modes": "raw", "linear_acc_modes": "", "rotation_modes": "quaternion", "tof_modes": "sensor_stats", "thm_modes": "centered_diff"}
        if not any(_modes(extractor.get(name, default)) for name, default in defaults.items()):
            return "Window auxiliary heads require time-series inputs."
        chunk_length = extractor.get("chunk_window_size", 128)
        if not _positive_integer(chunk_length):
            chunk_length = extractor.get("maxlen") if extractor.get("sequence_crop_mode", "tail") != "none" else None
        for head in windows:
            if _positive_integer(chunk_length) and head["window_size"] > chunk_length:
                return f"Auxiliary window {head['window_size']} exceeds chunk length {chunk_length}."
    if y is not None and (settings["use_orientation_aux_head"] or heads):
        columns = getattr(y, "columns", ())
        seq_col = extractor.get("sequence_col", "sequence_id")
        if seq_col not in columns and getattr(getattr(y, "index", None), "name", None) != seq_col:
            return f"Auxiliary labels require y with sequence column {seq_col!r}."
        targets = [("orientation", "classification")] if settings["use_orientation_aux_head"] else []
        targets += [(head["target"], head["task"]) for head in heads]
        for column, task in targets:
            cache_key = (id(y), seq_col, column, task)
            if label_cache is not None and cache_key in label_cache:
                error = label_cache[cache_key]
            else:
                error = _label_error(y, seq_col, column, task)
                if label_cache is not None:
                    label_cache[cache_key] = error
            if error is not None:
                return error
    return None


def _label_error(y, seq_col, column, task):
    if column not in y.columns:
        return f"Auxiliary target {column!r} requires that column in y."
    labels = y[column]
    if labels.isna().any():
        return f"Missing auxiliary labels for {column!r}."
    grouped = (y.groupby(seq_col, sort=False)[column] if seq_col in y.columns
               else y.groupby(level=seq_col, sort=False)[column])
    if (grouped.nunique(dropna=False) > 1).any():
        return f"Auxiliary target {column!r} must be constant within each sequence."
    if task == "regression":
        try:
            values = labels.to_numpy(dtype=float)
        except (TypeError, ValueError):
            return f"Auxiliary regression target {column!r} must be numeric."
        if not np.isfinite(values).all():
            return f"Auxiliary regression target {column!r} must be finite."
    return None


try:
    from skopt import BayesSearchCV
except ImportError:
    BayesSearchCV = None


def conditional_bayesian_spaces(param_space, base_params=None):
    """Split explicit architecture/source choices without pruning parameters.

    Every supplied dimension and value is retained. Splitting these two axes
    only preserves round-robin coverage of requested model/training conditions;
    it does not infer inactive options, replace values, or add missing keys.
    """
    from itertools import product
    from skopt.space import Categorical, Dimension, Integer, check_dimension

    if not isinstance(param_space, dict):
        raise TypeError("conditional_bayesian_spaces requires one parameter dictionary.")

    def categorical(values):
        return Categorical([
            json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
            for value in values
        ])

    dimensions = {}
    for key, value in param_space.items():
        if isinstance(value, Dimension):
            dimensions[key] = value
        elif hasattr(value, "categories"):
            dimensions[key] = categorical(value.categories)
        elif isinstance(value, list):
            dimensions[key] = categorical(value)
        elif isinstance(value, tuple):
            dimensions[key] = check_dimension(value)
        else:
            dimensions[key] = categorical([value])

    split_keys = [key for key in dimensions if _name(key) in {
        "temporal_architecture", "extractor__pretraining_source",
    }]
    choices = []
    for key in split_keys:
        dimension = dimensions[key]
        if isinstance(dimension, Categorical):
            choices.append(tuple(dimension.categories))
        elif isinstance(dimension, Integer) and dimension.high - dimension.low < 4096:
            choices.append(range(int(dimension.low), int(dimension.high) + 1))
        else:
            raise ValueError(f"{_name(key)} requires finite categorical choices.")

    spaces = []
    for values in product(*choices):
        space = dict(dimensions)
        for key, value in zip(split_keys, values):
            space[key] = Categorical([value])
        spaces.append(space)
    return spaces


class _CandidateRules:
    """Shared validation and complete candidate mappings, independent of TensorFlow."""
    _proposal_attempts = 32

    def _initialise_rules(self, y, fit_params):
        self._base_params_ = self.estimator.get_params(deep=True)
        self._rules_y_ = y
        self._phase_labels_available_ = any(
            (key == 'phase_labels' or key.endswith('__phase_labels')) and value is not None
            for key, value in fit_params.items()
        )
        self._label_validation_cache_ = {}
        self.rejected_candidate_count_ = 0
        self.rejected_candidate_reasons_ = Counter()
        self.reused_candidate_count_ = 0
        self.duplicate_proposal_count_ = 0
        self.unique_candidate_count_ = 0
        self.search_exhausted_ = False
        self.proposal_limit_reached_ = False

    def _candidate_params(self, params):
        """Copy the complete requested candidate without changing keys or values."""
        return dict(params)

    def _candidate_error(self, params):
        base_params = getattr(self, '_base_params_', None)
        if base_params is None:
            base_params = self.estimator.get_params(deep=True)
        return candidate_error(params, base_params,
                               y=getattr(self, '_rules_y_', None),
                               phase_labels_available=getattr(self, '_phase_labels_available_', False),
                               label_cache=getattr(self, '_label_validation_cache_', None))

    def _reject(self, reason):
        self.rejected_candidate_count_ += 1
        self.rejected_candidate_reasons_[reason] += 1

    def _penalty(self):
        return float(self._base_params_.get('estimator__invalid_candidate_score',
                     self._base_params_.get('invalid_candidate_score', -1.0)))


class HierarchicalGASearchCV(_CandidateRules, GASearchCV):
    """Evolve every declared gene, with bounded retries for exact duplicates."""
    _controller_order = (
        'temporal_architecture', 'use_bigru', 'use_orientation_aux_head',
        'use_phase_aux_head', 'custom_aux_heads', 'extractor',
    ) + _EXTRA_CONTROLLERS + (
        'extractor__tof_modes', 'tof_encoder_type', 'bert_pooling', 'attention_type',
        'post_fusion_pooling', 'post_fusion_attention_type', 'use_se',
    )
    _finite_enumeration_limit = 4096

    def fit(self, X, y=None, groups=None, **fit_params):
        self._initialise_rules(y, fit_params)
        self._proposal_keys_ = set()
        self._finite_space_id_ = None
        self._finite_candidates_ = None
        self._last_proposal_ = None
        self.duplicate_proposal_count_ = 0
        self.unique_candidate_count_ = 0
        self.search_exhausted_ = False
        self.proposal_limit_reached_ = False
        try:
            return super().fit(X, y, groups=groups, **fit_params)
        finally:
            # Rule validation and finite-space enumeration are fit-local; do
            # not retain the label table or the candidate pool afterwards.
            self._rules_y_ = None
            self._label_validation_cache_.clear()
            self._finite_candidates_ = None

    def _on_candidate_cache_hit(self):
        self.reused_candidate_count_ += 1

    def _ordered_genes(self, space):
        by_name = {_name(key): key for key in space}
        controllers = [by_name[name] for name in self._controller_order if name in by_name]
        return controllers + [key for key in space if key not in controllers]


    def _parameter_proposal(self, space, rng, individual=None):
        proposal = self._candidate_params(individual or {})
        for key in self._ordered_genes(space):
            if (individual is None or key not in proposal
                    or rng.random() <= float(self.mutation_probability)):
                proposal[key] = super()._sample_individual({key: space[key]}, rng)[key]
        return proposal

    @staticmethod
    def _finite_gene_values(descriptor):
        choices = getattr(descriptor, 'choices', getattr(descriptor, 'categories', None))
        if choices is not None:
            return choices
        lower = getattr(descriptor, 'lower', None)
        upper = getattr(descriptor, 'upper', None)
        if isinstance(lower, Integral) and isinstance(upper, Integral):
            return range(int(lower), int(upper) + 1)
        if lower is not None and lower == upper:
            return (lower,)
        if isinstance(descriptor, (tuple, list)):
            return descriptor
        if not hasattr(descriptor, 'sample'):
            return (descriptor,)
        return None

    def _finite_candidate_pool(self, space):
        """Enumerate every declared axis in small finite spaces.

        Enumeration is capped by tree nodes, so a large or continuous space
        cannot cause an unbounded product or claim proven exhaustion.
        """
        if getattr(self, '_finite_space_id_', None) == id(space):
            return self._finite_candidates_
        self._finite_space_id_ = id(space)
        self._finite_candidates_ = None
        ordered = self._ordered_genes(space)
        candidates = {}
        visited = 0

        class IncompleteSpace(Exception):
            pass

        def visit(index, proposal):
            nonlocal visited
            visited += 1
            if visited > self._finite_enumeration_limit:
                raise IncompleteSpace
            if index == len(ordered):
                active = self._candidate_params(proposal)
                if self._candidate_error(active) is None:
                    candidates[self._canonical_key(active)] = active
                return
            key = ordered[index]
            values = self._finite_gene_values(space[key])
            if values is None:
                raise IncompleteSpace
            for value in values:
                proposal[key] = value
                visit(index + 1, proposal)
            proposal.pop(key, None)

        try:
            visit(0, {})
        except IncompleteSpace:
            return None
        self._finite_candidates_ = candidates
        return candidates

    def _remember_proposal(self, proposal):
        self._proposal_keys_.add(self._canonical_key(proposal))
        self._last_proposal_ = dict(proposal)
        return proposal

    def _propose(self, space, rng, individual=None):
        if not hasattr(self, '_proposal_keys_'):
            self._proposal_keys_ = set()
        # Once a finite space has been proved exhausted, subsequent population
        # slots use the evaluation cache without repeating a retry loop.
        if getattr(self, 'search_exhausted_', False):
            self.duplicate_proposal_count_ = getattr(self, 'duplicate_proposal_count_', 0) + 1
            return dict(self._last_proposal_)
        fallback = None
        for attempt in range(self._proposal_attempts):
            # A duplicate child should explore another phenotype even when a
            # zero mutation probability would otherwise recreate it forever.
            proposal = self._parameter_proposal(space, rng, individual if attempt == 0 else None)
            reason = self._candidate_error(proposal)
            if reason is not None:
                self._reject(reason)
                continue
            fallback = proposal
            if self._canonical_key(proposal) not in self._proposal_keys_:
                return self._remember_proposal(proposal)
            self.duplicate_proposal_count_ = getattr(self, 'duplicate_proposal_count_', 0) + 1
        finite = self._finite_candidate_pool(space)
        if finite is not None:
            remaining = [key for key in finite if key not in self._proposal_keys_]
            if remaining:
                key = remaining[int(rng.integers(len(remaining)))]
                return self._remember_proposal(dict(finite[key]))
            self.search_exhausted_ = True
            if fallback is None and finite:
                fallback = dict(next(iter(finite.values())))
        else:
            self.proposal_limit_reached_ = True
        # All-invalid and exhausted spaces terminate with a finite penalty or
        # a cached valid result, respectively; neither schedules extra fits.
        return self._remember_proposal(fallback if fallback is not None else proposal)

    def _sample_individual(self, space, rng):
        return self._propose(space, rng)

    def _mutate(self, individual, space, rng):
        return self._propose(space, rng, individual)

    def _crossover(self, parent1, parent2, rng):
        child1, child2 = dict(parent1), dict(parent2)
        if rng.random() <= float(self.crossover_probability):
            for key in sorted(parent1.keys() | parent2.keys()):
                if key not in parent1:
                    child1[key] = parent2[key]
                elif key not in parent2:
                    child2[key] = parent1[key]
                elif rng.random() < 0.5:
                    child1[key], child2[key] = parent2[key], parent1[key]
        return self._candidate_params(child1), self._candidate_params(child2)


    def _evaluate_individual(self, params, X, y, splits, fit_params):
        # Invalid candidates retain their complete parameters in the finite-penalty row.
        self.unique_candidate_count_ = getattr(self, 'unique_candidate_count_', 0) + 1
        active = self._candidate_params(params)
        reason = self._candidate_error(params)
        if reason is not None:
            self._reject(reason)
            penalty = self._penalty()
            count = len(splits)
            return dict(params=active, fit_times=[0.0]*count, score_times=[0.0]*count,
                        test_scores=[penalty]*count, train_scores=[penalty]*count,
                        mean_test_score=penalty, std_test_score=0.0,
                        mean_train_score=penalty, std_train_score=0.0, candidate_error=reason)
        result = super()._evaluate_individual(active, X, y, splits, fit_params)
        result['candidate_error'] = None
        return result

    def _build_cv_results(self, records, maximize):
        results = super()._build_cv_results(records, maximize)
        # Include every supplied parameter in the CSV, including sparse custom candidates.
        for key in sorted({key for record in records for key in record['params']}):
            results[f'param_{key}'] = [record['params'].get(key) for record in records]
        results['candidate_error'] = [record.get('candidate_error') for record in records]
        return results


if BayesSearchCV is not None:
    class HierarchicalBayesSearchCV(_CandidateRules, BayesSearchCV):
        """Search conditional spaces with a budget of distinct valid CV candidates.

        A dictionary's n_iter is shared across its architecture subspaces. Explicit
        lists of spaces retain skopt's per-entry budgets. Proposals are sequential
        to avoid speculative fits and keep model memory bounded; CV parallelism
        still follows n_jobs. Invalid and cached proposals do not spend the budget.
        Assign ``candidate_checkpoint_path`` after construction to atomically
        save cumulative CV results after each completed candidate, before the
        next optimizer update. It is disabled by default.
        """
        candidate_checkpoint_path = None

        def _checkpoint_candidates(self, results):
            if self.candidate_checkpoint_path is None:
                return
            destination = Path(self.candidate_checkpoint_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + '.tmp')
            pd.DataFrame(results).to_csv(temporary, index=False)
            temporary.replace(destination)
            self.candidate_checkpoint_count_ = len(results['params'])

        def fit(self, X, y=None, groups=None, callback=None, **fit_params):
            self._initialise_rules(y, fit_params)
            self._effective_score_cache_ = {}
            self._invalid_fallback_ = None
            try:
                return super().fit(X, y, groups=groups, callback=callback, **fit_params)
            finally:
                # Preflight needs labels only while searching, never a retained dataset.
                self._rules_y_ = None
                self._label_validation_cache_.clear()

        @staticmethod
        def _finite_grid(optimizer, limit=4096):
            """A lazy fallback for small discrete spaces, never a large Cartesian table."""
            from itertools import product
            from skopt.space import Categorical, Integer
            choices, size = [], 1
            for dimension in optimizer.space.dimensions:
                if isinstance(dimension, Categorical):
                    values = dimension.categories
                elif isinstance(dimension, Integer):
                    values = range(int(dimension.low), int(dimension.high) + 1)
                else:
                    return None, None
                size *= len(values)
                if size > limit:
                    return None, None
                choices.append(values)
            return iter(product(*choices)), size

        def _next_unique_point(self, state):
            optimizer = state['optimizer']
            pending = []

            def inspect_point(raw_point):
                point = [np.asarray(value).item() for value in raw_point]
                coordinate = json.dumps(point, default=str)
                if coordinate in state['seen']:
                    self.duplicate_proposal_count_ += 1
                    return None
                state['seen'].add(coordinate)
                proposal = dict(zip(sorted(state['space']), point))
                reason = self._candidate_error(proposal)
                active = self._candidate_params(proposal)
                if reason is not None:
                    self._reject(reason)
                    self._invalid_fallback_ = active
                    if state['synthetic_count'] < 16:
                        pending.append((point, self._penalty()))
                        state['synthetic_count'] += 1
                    return None
                key = GASearchCV._canonical_key(self, active)
                if key in self._effective_score_cache_:
                    self.reused_candidate_count_ += 1
                    self.duplicate_proposal_count_ += 1
                    if state['synthetic_count'] < 16:
                        pending.append((point, self._effective_score_cache_[key]))
                        state['synthetic_count'] += 1
                    return None
                return point, active, key

            if state['size'] is not None and len(state['seen']) >= state['size']:
                state['exhausted'] = True
                return None, pending
            for attempt in range(self._proposal_attempts):
                # ask(n_points=1) uses constant-liar copies and re-fits the surrogate.
                raw = optimizer.ask() if attempt == 0 else optimizer.space.rvs(
                    n_samples=1, random_state=optimizer.rng,
                )[0]
                candidate = inspect_point(raw)
                if candidate is not None:
                    return candidate, pending
                if state['size'] is not None and len(state['seen']) >= state['size']:
                    state['exhausted'] = True
                    return None, pending
            if state['grid'] is not None:
                for raw in state['grid']:
                    candidate = inspect_point(raw)
                    if candidate is not None:
                        return candidate, pending
                state['exhausted'] = True
            else:
                # Bounded random retries cannot prove a continuous space exhausted.
                self.proposal_limit_reached_ = True
            return None, pending

        @staticmethod
        def _search_score(results):
            names = [name for name in results if name.startswith('mean_test_')]
            if len(names) != 1:
                raise ValueError('Hierarchical Bayesian search requires one scoring metric.')
            return float(results[names[0]][-1])

        def _run_search(self, evaluate_candidates):
            from sklearn.utils import check_random_state
            from skopt.utils import eval_callbacks

            rng = check_random_state(self.random_state)
            self.optimizer_kwargs_['random_state'] = rng
            # Retaining every fitted GP multiplies memory as the search progresses.
            self.optimizer_kwargs_.setdefault('model_queue_size', 1)
            entries = [self.search_spaces] if isinstance(self.search_spaces, dict) else self.search_spaces
            groups, states = [], []
            for entry in entries:
                original, budget = entry if isinstance(entry, tuple) else (entry, self.n_iter)
                if not isinstance(budget, Integral) or budget <= 0:
                    raise ValueError('Each Bayesian search budget must be a positive integer.')
                conditional = conditional_bayesian_spaces(original, self._base_params_)
                group = []
                for space in conditional:
                    optimizer = self._make_optimizer(space)
                    grid, size = self._finite_grid(optimizer)
                    state = dict(space=space, optimizer=optimizer, grid=grid,
                                 size=size, seen=set(), exhausted=False, synthetic_count=0)
                    group.append(state)
                    states.append(state)
                # Fair first visits when the budget is smaller than the architecture count.
                group = [group[index] for index in rng.permutation(len(group))]
                groups.append((group, int(budget)))
            self.optimizers_ = [state['optimizer'] for state in states]
            self.conditional_search_spaces_ = [state['space'] for state in states]
            self.requested_candidate_count_ = sum(budget for _, budget in groups)
            stop = False
            for group, budget in groups:
                available, cursor, spent = list(group), 0, 0
                while spent < budget and available and not stop:
                    cursor %= len(available)
                    state = available[cursor]
                    candidate, pending = self._next_unique_point(state)
                    if candidate is None:
                        # This optimizer will not propose again. Avoid a cubic GP
                        # fit on thousands of synthetic penalties from a finite scan.
                        if pending and not state['optimizer'].yi:
                            point, score = pending[0]
                            state['optimizer'].tell([point], [-score])
                        available.pop(cursor)
                        continue
                    point, active, key = candidate
                    results = evaluate_candidates([active])
                    self._checkpoint_candidates(results)
                    score = self._search_score(results)
                    if not np.isfinite(score):
                        score = self._penalty()
                    self._effective_score_cache_[key] = score
                    self.unique_candidate_count_ += 1
                    spent += 1
                    pending.append((point, score))
                    result = state['optimizer'].tell([p for p, _ in pending], [-s for _, s in pending])
                    if state['size'] is not None and len(state['seen']) >= state['size']:
                        state['exhausted'] = True
                    stop = bool(eval_callbacks(self._callbacks, result))
                    cursor += 1
                if stop:
                    break
            self.search_exhausted_ = bool(states) and all(state['exhausted'] for state in states)
            if self.unique_candidate_count_ == 0:
                if self._invalid_fallback_ is None:
                    raise ValueError('No candidate could be proposed from the conditional search spaces.')
                # Retain one finite-penalty CV row for existing all-failed reporting.
                self._checkpoint_candidates(evaluate_candidates([self._invalid_fallback_]))
            self._optim_results = [state['optimizer'].get_result() for state in states
                                   if state['optimizer'].yi]
            if self.proposal_limit_reached_:
                warnings.warn('Stopped a conditional subspace after bounded retries found no unseen valid candidate.',
                              RuntimeWarning)
else:
    HierarchicalBayesSearchCV = None


def candidate_score(estimator, X, y):
    """Finite penalties keep Bayesian optimisation running even if all fits fail."""
    model = estimator.named_steps['estimator']
    if getattr(model, 'fit_failed_', False):
        return float(model.invalid_candidate_score)
    try:
        return float(estimator.score(X, y))
    except Exception as exc:
        if model.invalid_candidate_policy == 'raise':
            raise
        model.fit_error_ = f'{type(exc).__name__}: {exc}'
        model._record_candidate_failure('scoring', model.fit_error_, exc)
        warnings.warn(f'Candidate scoring skipped: {model.fit_error_}', RuntimeWarning)
        return float(model.invalid_candidate_score)


def refit_valid_candidate(search, X, y, **fit_params):
    """Try successful CV candidates in score order; never refit a penalty winner."""
    results = search.cv_results_
    score_key = 'mean_test_score' if 'mean_test_score' in results else 'mean_test_quality'
    scores = np.asarray(results[score_key], dtype=float)
    split_keys = [k for k in results if k.startswith('split') and
                  (k.endswith('_test_score') or k.endswith('_test_quality'))]
    valid = np.isfinite(scores) & (scores >= 0)
    for key in split_keys:
        valid &= np.isfinite(results[key]) & (np.asarray(results[key]) >= 0)
    search.refit_errors_ = []
    search.best_estimator_ = None
    search.best_index_ = None
    for idx in np.argsort(-scores, kind='stable'):
        if not valid[idx]:
            continue
        estimator = clone(search.estimator).set_params(**results['params'][idx])
        estimator.set_params(estimator__invalid_candidate_policy='raise')
        start = time.perf_counter()
        try:
            estimator.fit(X, y, **fit_params)
        except Exception as exc:
            model = estimator.named_steps.get('estimator')
            search.refit_errors_.append({'candidate': int(idx), 'error': f'{type(exc).__name__}: {exc}',
                                         'failure': getattr(model, 'fit_failure_', None)})
            continue
        search.best_estimator_ = estimator
        search.best_index_ = int(idx)
        search.best_params_ = dict(results['params'][idx])
        search.best_score_ = float(scores[idx])
        search.refit_time_ = time.perf_counter() - start
        return True
    search.best_score_ = np.nan
    search.best_params_ = {}
    return False
