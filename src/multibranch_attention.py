"""
multibranch_attention.py

Multi-branch (modality-specific) Keras classifier with every architecture,
optimizer, loss, and scheduler knob exposed for search.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

from base_utils_qwen import (
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


BRANCH_ORDER = ["acc", "rotation", "tof", "thermo", "stft", "cwt"]
TIME_SERIES_BRANCHES = {"acc", "rotation", "tof", "thermo"}
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
        return losses.SparseCategoricalCrossentropy(label_smoothing=label_smoothing)
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
):
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("MultiBranchSequenceClassifier requires TensorFlow/Keras.")

    tf.random.set_seed(random_state)

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
            x = _make_activation(activation)
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
    ):
        self.primary_target = primary_target
        self.extractor = extractor
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

    def _branch_groups(self, feature_names):
        stft_prefixes = [e.feature_prefix for e in getattr(self.extractor_, "stft_extractors", [])]
        cwt_prefixes = [e.feature_prefix for e in getattr(self.extractor_, "cwt_extractors", [])]
        return group_feature_names(feature_names, stft_prefixes, cwt_prefixes)

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

    def _build_callbacks(self):
        cb = []
        if self.early_stopping_patience and self.early_stopping_patience > 0:
            cb.append(callbacks.EarlyStopping(
                monitor="val_loss" if self.validation_split else "loss",
                patience=self.early_stopping_patience,
                restore_best_weights=True,
            ))
        if self.use_lr_scheduler and self.validation_split and self.validation_split > 0:
            if str(self.lr_scheduler_type).lower() == "plateau":
                cb.append(callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=self.lr_factor,
                    patience=self.lr_patience,
                    min_lr=self.min_lr,
                    verbose=0,
                ))
            elif str(self.lr_scheduler_type).lower() == "cosine":
                steps = max(1, int(self.epochs) - int(self.warmup_epochs))
                cb.append(callbacks.CosineAnnealingLR(
                    T_max=max(self.cosine_t_max, steps),
                    eta_min=self.min_lr,
                    verbose=0,
                ))
        return cb

    def fit(self, X, y=None, **fit_params):
        if y is None:
            raise ValueError("MultiBranchSequenceClassifier requires y.")
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow/Keras is required to fit MultiBranchSequenceClassifier.")

        self.extractor_ = clone(self.extractor) if self.extractor is not None else self._default_extractor()
        self._validate_extractor()

        try:
            self.extractor_.fit(X)
            chunks = self.extractor_.transform_chunks(X)
        except InvalidExtractorParams:
            raise
        except Exception as exc:
            raise InvalidExtractorParams(
                f"MultiBranchSequenceClassifier fit failed during extraction: {exc}"
            ) from exc

        self.feature_names_ = list(chunks["feature_names"])
        self.branch_groups_ = self._branch_groups(self.feature_names_)
        if not self.branch_groups_:
            raise InvalidExtractorParams("No modality branches could be built from the extracted features.")

        branch_arrays = self._split_branches(chunks["X"], chunks["mask"], self.branch_groups_)
        flat_check = np.concatenate([arr.reshape(len(arr), -1) for arr in branch_arrays.values()], axis=1)
        if not np.isfinite(flat_check).all():
            raise InvalidExtractorParams("Feature extraction produced non-finite branch values.")

        y_aligned = self._align_y(chunks["sequence_ids"], y)
        self.le_ = LabelEncoder()
        self.le_.fit(y_aligned)
        self.classes_ = self.le_.classes_
        y_enc = self.le_.transform(y_aligned)

        sample_weight = None
        if self.class_weight_mode == "balanced":
            uniq_classes = np.unique(y_enc)
            weights = compute_class_weight("balanced", classes=uniq_classes, y=y_enc)
            weight_map = dict(zip(uniq_classes, weights))
            sample_weight = np.array([weight_map[v] for v in y_enc], dtype=np.float32)

        time_series_shapes = {name: arr.shape[1:] for name, arr in branch_arrays.items() if name in TIME_SERIES_BRANCHES}
        static_shapes = {name: arr.shape[1] for name, arr in branch_arrays.items() if name in STATIC_BRANCHES}

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
        )

        self.input_order_ = list(time_series_shapes.keys()) + list(static_shapes.keys())
        model_inputs = [branch_arrays[name] for name in self.input_order_]
        cb = self._build_callbacks()

        history = self.model_.fit(
            model_inputs, y_enc,
            sample_weight=sample_weight,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split or 0.0,
            callbacks=cb,
            verbose=self.verbose,
        )
        self.history_ = history.history
        return self

    def predict_proba(self, X):
        check_is_fitted(self, ["model_", "le_", "input_order_"])
        branch_arrays, sequence_ids = self._transform_to_branches(X)
        model_inputs = [branch_arrays[name] for name in self.input_order_]
        probs = self.model_.predict(model_inputs, verbose=0)
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