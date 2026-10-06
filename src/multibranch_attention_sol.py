"""
multibranch_attention.py

Multi-branch (modality-specific) Keras classifier with every architecture,
optimizer, loss, and scheduler knob exposed for search.
"""

from __future__ import annotations

import json
import math
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

from base_utils_sol import (
    InvalidExtractorParams,
    SequenceExtractor,
    competition_score,
    prepare_bayesian_space,
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
        warnings.warn(
            "Both gradient_clip_norm and gradient_clip_value were set; keeping only "
            "gradient_clip_norm and ignoring gradient_clip_value.",
            RuntimeWarning,
            stacklevel=2,
        )
        gradient_clip_value = None
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
):
    """Build alternate temporal models while keeping the original builder path intact."""
    bert_positional_encoding = str(bert_positional_encoding).lower()
    bert_pooling = str(bert_pooling).lower()
    tof_encoder_type = str(tof_encoder_type).lower()
    if (
        (architecture == "phase_attention" or use_phase_aux_head)
        and (
            n_phases < 2
            or not np.isfinite(phase_attention_temperature)
            or phase_attention_temperature <= 0
        )
    ):
        raise ValueError("n_phases must be >= 2 and phase_attention_temperature must be positive.")
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

    branch_filters = _json_maybe(branch_filters) or DEFAULT_BRANCH_FILTERS
    branch_kernel_sizes = _json_maybe(branch_kernel_sizes) or DEFAULT_BRANCH_KERNEL_SIZES
    layer_counts = _json_maybe(branch_num_conv_layers)
    if isinstance(layer_counts, dict):
        layer_counts = {str(key): max(1, int(value)) for key, value in layer_counts.items()}
    else:
        layer_counts = max(1, int(layer_counts))
    branch_filter_mode = str(branch_filter_mode).lower()
    if branch_filter_mode not in {"custom", "double", "constant"}:
        raise ValueError("branch_filter_mode must be 'custom', 'double', or 'constant'.")

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
    ):
        inferred_length = int(inferred_length)
        mask_input = layers.Input(shape=(inferred_length,), name="valid_timestep_mask")
    model_mask_input = mask_input

    for name, shape in time_series_shapes.items():
        branch_input = layers.Input(shape=shape, name=f"{name}_input")
        inputs.append(branch_input)
        input_by_name[name] = branch_input
        x = branch_input
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

    for name, shape in static_shapes.items():
        branch_input = layers.Input(shape=(shape,), name=f"{name}_input")
        inputs.append(branch_input)
        input_by_name[name] = branch_input
        x = layers.Dense(static_branch_units, activation="relu", name=f"{name}_static_dense")(branch_input)
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
    if use_orientation_aux_head:
        outputs["orientation_aux"] = layers.Dense(
            int(orientation_num_classes), activation="softmax", name="orientation_aux"
        )(head)
    if use_phase_aux_head:
        if phase_probabilities is None:
            if sequence_mask is None:
                raise ValueError("use_phase_aux_head requires time-series features.")
            phase_probabilities = layers.Softmax(axis=-1, name="phase_aux_probabilities")(
                layers.Dense(int(n_phases), name="phase_aux_logits")(x)
            )
        outputs["phase_aux"] = phase_probabilities
    if not use_orientation_aux_head and not use_phase_aux_head:
        outputs = outputs["predictions"]
    model_inputs = inputs + ([model_mask_input] if model_mask_input is not None else [])
    model = models.Model(
        inputs=model_inputs,
        outputs=outputs,
        name=f"multibranch_{architecture}",
    )
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
    if architecture != "bigru_attention" or use_bigru or use_orientation_aux_head or use_phase_aux_head:
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
        )
        main_loss = _build_loss(loss_name=loss_name, label_smoothing=label_smoothing)
        has_auxiliary_outputs = use_orientation_aux_head or use_phase_aux_head
        model_loss = (
            {
                "predictions": main_loss,
                **({"orientation_aux": "sparse_categorical_crossentropy"} if use_orientation_aux_head else {}),
                **({"phase_aux": "sparse_categorical_crossentropy"} if use_phase_aux_head else {}),
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
            metrics={"predictions": ["accuracy"]} if has_auxiliary_outputs else ["accuracy"],
        )
        return model

    branch_filters = _json_maybe(branch_filters) or DEFAULT_BRANCH_FILTERS
    branch_kernel_sizes = _json_maybe(branch_kernel_sizes) or DEFAULT_BRANCH_KERNEL_SIZES
    branch_pool_sizes = _json_maybe(branch_pool_sizes) or DEFAULT_BRANCH_POOL_SIZES
    branch_num_conv_layers = _json_maybe(branch_num_conv_layers)
    if isinstance(branch_num_conv_layers, dict):
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
        x = layers.Dense(static_branch_units, activation="relu", name=f"{name}_dense1")(branch_in)
        if use_batchnorm:
            x = layers.BatchNormalization(name=f"{name}_bn")(x)
        if static_dropout:
            x = layers.Dropout(static_dropout, name=f"{name}_dropout")(x)
        x = layers.Dense(static_branch_units, activation="relu", name=f"{name}_embed")(x)
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
            yy = y.copy()
            if seq_col not in yy.columns:
                if yy.index.name == seq_col:
                    yy = yy.reset_index()
                else:
                    raise ValueError("y DataFrame does not contain sequence_id.")
            if self.primary_target not in yy.columns:
                raise ValueError(f"y DataFrame does not contain target column: {self.primary_target}")
            y_seq = yy.drop_duplicates(seq_col).set_index(seq_col)[self.primary_target]
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
            return X.copy(), None
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
        return X.loc[train_mask].copy(), X.loc[validation_mask].copy()

    def _augment_training_sequences(self, X):
        if self.augmentor is None:
            self.augmentor_ = None
            return X.copy()
        self.augmentor_ = clone(self.augmentor)
        return self.augmentor_.fit_transform(X)

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
        for name, idxs in groups.items():
            block = x_arr[:, :, idxs]
            if name in STATIC_BRANCHES:
                m = mask_arr[:, :, None].astype(np.float32)
                denom = np.clip(m.sum(axis=1), 1.0, None)
                branch_arrays[name] = (block * m).sum(axis=1) / denom
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
        if isinstance(y, pd.DataFrame) and seq_col not in y.columns and y.index.name == seq_col:
            y = y.reset_index()
        if not isinstance(y, pd.DataFrame) or seq_col not in y.columns or "orientation" not in y.columns:
            raise ValueError(
                "use_orientation_aux_head=True requires an 'orientation' column in y."
            )
        labels = y.drop_duplicates(seq_col).set_index(seq_col)["orientation"].reindex(sequence_ids)
        if labels.isna().any():
            raise ValueError("Orientation auxiliary labels are missing for one or more sequences.")
        return labels.to_numpy()

    def _prepare_training_targets(self, labels, sequence_ids, y, phase_labels=None):
        targets = {"predictions": labels}
        weights = {}
        if self.sample_weight_ is not None:
            weights["predictions"] = self.sample_weight_
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
            phase_labels = np.asarray(phase_labels, dtype=np.int32)
            if (
                phase_labels.ndim != 2
                or phase_labels.shape[0] != len(labels)
                or phase_labels.shape[1] != self._current_chunk_mask.shape[1]
            ):
                raise ValueError("phase_labels must have shape (number_of_chunks, timesteps).")
            if np.any(phase_labels < 0) or np.any(phase_labels >= self.n_phases):
                raise ValueError(f"phase_labels must be in [0, {self.n_phases}).")
            targets["phase_aux"] = phase_labels
            weights["phase_aux"] = np.asarray(self._current_chunk_mask, dtype=np.float32)
        if len(targets) == 1:
            return labels, self.sample_weight_
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
        if y is None:
            raise ValueError("MultiBranchSequenceClassifier requires y.")
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow/Keras is required to fit MultiBranchSequenceClassifier.")

        self.extractor_ = clone(self.extractor) if self.extractor is not None else self._default_extractor()
        self._validate_extractor()

        X_train, X_validation = self._split_validation_sequences(X, y)
        validation_data = None
        if X_validation is not None and X_validation.empty:
            raise ValueError("Internal validation split produced no validation rows.")

        X_train = self._augment_training_sequences(X_train)

        try:
            # STFT/CWT features summarize each whole sequence; split sequences
            # first, then fit/transform train and validation partitions separately.
            self.extractor_.fit(X_train)
            train_chunks = self.extractor_.transform_chunks(X_train)
            validation_chunks = (
                self.extractor_.transform_chunks(X_validation)
                if X_validation is not None
                else None
            )
        except InvalidExtractorParams:
            raise
        except Exception as exc:
            raise InvalidExtractorParams(
                f"MultiBranchSequenceClassifier fit failed during extraction: {exc}"
            ) from exc

        self.feature_names_ = list(train_chunks["feature_names"])
        self.branch_groups_ = self._branch_groups(
            self.feature_names_,
            separate_derived_imu=(
                str(self.temporal_architecture).lower() != "bigru_attention"
                or self.use_bigru
                or self.use_orientation_aux_head
                or self.use_phase_aux_head
            ),
        )
        if not self.branch_groups_:
            raise InvalidExtractorParams("No modality branches could be built from the extracted features.")

        branch_arrays = self._split_branches(train_chunks["X"], train_chunks["mask"], self.branch_groups_)
        flat_check = np.concatenate([arr.reshape(len(arr), -1) for arr in branch_arrays.values()], axis=1)
        if not np.isfinite(flat_check).all():
            raise InvalidExtractorParams("Feature extraction produced non-finite branch values.")

        time_series_shapes = {name: arr.shape[1:] for name, arr in branch_arrays.items() if name in TIME_SERIES_BRANCHES}
        static_shapes = {name: arr.shape[1] for name, arr in branch_arrays.items() if name in STATIC_BRANCHES}
        self.input_order_ = list(time_series_shapes.keys()) + list(static_shapes.keys())
        self.model_requires_mask_ = (
            str(self.temporal_architecture).lower() != "bigru_attention"
            or self.use_bigru
            or self.use_orientation_aux_head
            or self.use_phase_aux_head
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
            validation_data = (validation_inputs, validation_labels)

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
        phase_training = fit_params.pop("phase_labels", None)
        phase_validation = fit_params.pop("validation_phase_labels", None)
        self._current_chunk_mask = train_chunks["mask"]
        train_targets, model_sample_weight = self._prepare_training_targets(
            y_enc,
            train_chunks["sequence_ids"],
            y,
            phase_training,
        )
        if validation_chunks is not None and (
            self.use_orientation_aux_head or self.use_phase_aux_head
        ):
            self._current_chunk_mask = validation_chunks["mask"]
            validation_targets, _ = self._prepare_training_targets(
                validation_labels,
                validation_chunks["sequence_ids"],
                y,
                phase_validation,
            )
            validation_data = (validation_inputs, validation_targets)

        self.model_ = build_multibranch_model(
            time_series_shapes=time_series_shapes,
            static_shapes=static_shapes,
            n_classes=len(self.classes_),
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
            orientation_num_classes=self.orientation_num_classes,
            sequence_length=train_chunks["mask"].shape[1],
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
        preds = self.predict(X)
        seq_col = getattr(self.extractor_, "sequence_col", "sequence_id")
        if isinstance(y, pd.DataFrame):
            yy = y.copy()
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
