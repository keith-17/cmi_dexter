"""Conditional architecture rules shared by search and estimator preflight.

Canonicalisation removes *inactive* dimensions, not valid architecture choices.
In particular a BERT backbone still has modality-specific CNN encoders. The
functions do not construct a model, fit an extractor, or mutate caller settings.
"""
from __future__ import annotations

import json
import math
from numbers import Integral, Real


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
}


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


def canonical_candidate(params, base_params=None):
    """Return supplied settings with irrelevant dimensions removed.

    ``base_params`` can be estimator or pipeline ``get_params(deep=True)``.
    Only supplied keys are returned, retaining their ``estimator__`` prefix.
    Consequently this is suitable both for sklearn ``set_params`` and for a
    phenotype/cache key. A removed value falls back to the base estimator.
    """
    settings = _merged(params, base_params)
    architecture = str(settings["temporal_architecture"]).lower()
    temporal = _is_temporal(settings)
    # The classifier derives its orientation output width from training labels.
    inactive = {"orientation_num_classes"}
    if architecture != "bert":
        inactive |= _BERT
    if architecture != "bigru_attention" or not settings["use_bigru"]:
        inactive |= _GRU
    if architecture not in {"phase_attention", "cnn_attention_pool"}:
        inactive |= _SHARED_CNN
    if architecture != "phase_attention":
        inactive |= _PHASE
        if not settings["use_phase_aux_head"]:
            inactive.add("n_phases")
    if architecture != "cnn_attention_pool":
        inactive |= _POST_FUSION
    elif str(settings["post_fusion_pooling"]).lower() not in {"attention", "attention_avg_max"}:
        inactive |= {"post_fusion_attention_type", "post_fusion_attention_units"}
    elif str(settings["post_fusion_attention_type"]).lower() == "linear":
        inactive.add("post_fusion_attention_units")
    uses_attention = (
        architecture == "bigru_attention" and temporal
        or architecture == "bert" and str(settings["bert_pooling"]).lower() == "attention"
    )
    if not uses_attention:
        inactive |= {"attention_type", "attention_units"}
    elif str(settings["attention_type"]).lower() == "linear":
        inactive.add("attention_units")
    if temporal:
        inactive |= _LEGACY_BRANCH
    else:
        inactive |= _TEMPORAL | _TOF_SPATIAL | {"tof_encoder_type"}
    if not settings["use_se"]:
        inactive.add("se_ratio")
    if not settings["use_orientation_aux_head"]:
        inactive |= {"orientation_num_classes", "orientation_loss_weight"}
    if not settings["use_phase_aux_head"]:
        inactive |= {"phase_loss_weight", "phase_label_provider"}
    extractor = _extractor_settings(settings)
    tof_enabled = bool(_modes(extractor.get("tof_modes", "sensor_stats")))
    tof_encoder = str(settings["tof_encoder_type"]).lower()
    if not temporal or not tof_enabled:
        inactive |= _TOF_SPATIAL | {"tof_encoder_type"}
    elif tof_encoder == "flatten_1d":
        inactive |= _TOF_SPATIAL
    elif tof_encoder != "spatial_cnn_gru":
        inactive.add("tof_gru_units")
    result = {}
    for key, value in (params or {}).items():
        name = _name(key)
        if name in inactive:
            continue
        if name == "use_bigru" and architecture != "bigru_attention":
            value = False
        elif name == "branch_encoder_type" and str(value).lower() == "cnn":
            value = "conv1d"
        elif name in {"temporal_architecture", "branch_encoder_type", "tof_encoder_type"}:
            value = str(value).lower()
        result[key] = value
    return result


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
    settings = canonical_candidate(merged)
    architecture = str(settings["temporal_architecture"]).lower()
    if architecture not in ARCHITECTURES:
        return f"Unsupported temporal_architecture: {architecture!r}."
    for name in ("use_bigru", "use_orientation_aux_head", "use_phase_aux_head", "use_residual", "use_se", "bert_use_cls_token"):
        if name in settings and not isinstance(settings[name], (bool, Integral)):
            return f"{name} must be True or False."
    raw_heads = _heads(settings.get("custom_aux_heads"))
    if raw_heads is not None and raw_heads != []:
        from multibranch_attention_sol import _normalise_aux_heads
        heads = _normalise_aux_heads(raw_heads)
    else:
        heads = []
    temporal = _is_temporal(merged)
    extractor = _extractor_settings(merged)
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
        if name in settings and not _positive_integer(settings[name]):
            return f"{name} must be a positive integer."
    for name in ("branch_filters", "branch_kernel_sizes", "branch_pool_sizes", "branch_num_conv_layers"):
        if name in settings and not _positive_tree(settings[name]):
            return f"{name} must contain positive integers."
    for name in {"dropout", "static_dropout", "conv_dropout", "branch_dropout", "fusion_dropout",
                 "cnn_dropout", "gru_dropout", "gru_recurrent_dropout", "bert_dropout", "bert_attention_dropout"}:
        if name in settings and (not _finite_number(settings[name]) or not 0 <= settings[name] < 1):
            return f"{name} must be in [0, 1)."
    if "learning_rate" in settings and (not _finite_number(settings["learning_rate"]) or settings["learning_rate"] <= 0):
        return "learning_rate must be finite and positive."
    for name in ("orientation_loss_weight", "phase_loss_weight"):
        if name in settings and (not _finite_number(settings[name]) or settings[name] < 0):
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
    if "tof_encoder_type" in settings:
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
    grouped = y.groupby(seq_col)[column] if seq_col in y.columns else y.groupby(level=seq_col)[column]
    if (grouped.nunique(dropna=False) > 1).any():
        return f"Auxiliary target {column!r} must be constant within each sequence."
    if task == "regression":
        try:
            values = labels.to_numpy(dtype=float)
        except (TypeError, ValueError):
            return f"Auxiliary regression target {column!r} must be numeric."
        if not all(math.isfinite(value) for value in values):
            return f"Auxiliary regression target {column!r} must be finite."
    return None
