"""
dl_utils.py

Multi-branch (modality-specific) deep learning components that build on top
of ``base_utils_qwen.py``.

Every raw sensor modality gets its own branch of the network:

    acc      -> Conv1D stack over accelerometer-derived channels
                (raw / smoothed / velocity / displacement / jerk ...)
    rotation -> Conv1D stack over rotation-derived channels
                (quaternion / euler / angular velocity / rot6d ...)
    tof      -> Conv1D stack over time-of-flight sensor statistics
    thermo   -> Conv1D stack over thermopile features
    stft     -> Dense stack over per-sequence STFT summary statistics
    cwt      -> Dense stack over per-sequence CWT summary statistics

Design notes
------------
``SequenceExtractor`` (in base_utils_qwen.py) already owns every step of
cleaning / motion-filtering / per-modality feature extraction, and its
``transform_chunks`` output concatenates all of that into a single
``(n_chunks, window, n_features)`` tensor with a matching ``feature_names``
list. Rather than re-implement cleaning/filtering/extraction a second time
for a "deep learning" code path, this module reuses ``SequenceExtractor``
verbatim and instead splits its combined tensor back into per-modality
blocks after the fact, using the exact column-naming convention the
individual extractors (``IMUExtractor``, ``RotationExtractor``,
``TOFExtractor``, ``ThermoExtractor``, ``STFTExtractor``, ``CWTExtractor``)
already produce. This keeps the two code paths (Random Forest / frame
output, multi-branch deep model / chunk output) sharing one tested
extraction pipeline, and it means any extractor hyperparameter that is
searchable for the Random Forest notebook (acc_modes, rotation_modes,
tof_modes, thm_modes, stft_configs, cwt_configs, filter_problematic_sequences,
...) is automatically searchable here too.

STFT and CWT features are computed once per *sequence* (not per timestep),
so within a sequence they are constant across every row. Rather than feed a
constant repeated over time into a Conv1D branch, they are collapsed with a
mask-aware mean into a single static vector per sequence/chunk and routed
through a small Dense branch instead.
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
    from tensorflow.keras import callbacks, layers, models, optimizers

    TENSORFLOW_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when TF is missing
    tf = None
    layers = models = optimizers = callbacks = None
    TENSORFLOW_AVAILABLE = False


# ---------------------------------------------------------------------------
# Branch bookkeeping
# ---------------------------------------------------------------------------

BRANCH_ORDER = ["acc", "rotation", "tof", "thermo", "stft", "cwt"]
TIME_SERIES_BRANCHES = {"acc", "rotation", "tof", "thermo"}
STATIC_BRANCHES = {"stft", "cwt"}

DEFAULT_BRANCH_FILTERS = {"acc": 64, "rotation": 64, "tof": 32, "thermo": 32, "stft": 32, "cwt": 32}
DEFAULT_BRANCH_KERNEL_SIZES = {"acc": 5, "rotation": 5, "tof": 3, "thermo": 3, "stft": 3, "cwt": 3}
DEFAULT_BRANCH_POOL_SIZES = {"acc": 2, "rotation": 2, "tof": 2, "thermo": 2, "stft": 2, "cwt": 2}


def _attention_pool_over_time(name: str, x):
    """Attention pooling over the temporal axis for a branch.

    The Kaggle solution pools over each modality's sequence with an attention
    gate before concatenation; this keeps the branch’s temporal information
    while reducing the time dimension to a single embedding vector.
    """
    attention_logits = layers.Dense(1, activation="tanh", name=f"{name}_attention_score")(x)
    attention_logits = layers.Reshape((-1,), name=f"{name}_attention_flat")(attention_logits)
    attention_weights = layers.Softmax(name=f"{name}_attention_weights")(attention_logits)
    attention_weights = layers.Reshape((-1, 1), name=f"{name}_attention_reshape")(attention_weights)
    pooled = layers.Multiply(name=f"{name}_attention_pool")([x, attention_weights])
    return layers.Lambda(
        lambda t: tf.reduce_sum(t, axis=1),
        name=f"{name}_attention_sum",
    )(pooled)


def _json_maybe(value: Any) -> Any:
    """Decode a JSON-encoded dict/list param back into python objects.

    ``prepare_multitask_param_space`` (base_utils_qwen.py) serializes
    dict-valued search-space categories to JSON strings so that skopt can
    hash them as categorical choices. This reverses that on the way back in,
    and is a no-op for values that were never encoded.
    """

    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return value


def prepare_multibranch_bayesian_space(param_space: Dict[str, Any]) -> Dict[str, Any]:
    """Combine the two search-space helpers already in base_utils_qwen.py.

    First JSON-encodes any ``branch_filters`` / ``branch_kernel_sizes`` /
    ``branch_pool_sizes`` dict categories (``prepare_multitask_param_space``),
    then wraps every remaining list into a skopt ``Categorical`` space
    (``prepare_bayesian_space``), so the multi-branch classifier's per-branch
    architecture hyperparameters are searchable with BayesSearchCV exactly
    like every other extractor/estimator hyperparameter.
    """

    return prepare_bayesian_space(prepare_multitask_param_space(param_space, "bayesian"))


def group_feature_names(
    feature_names: Sequence[str],
    stft_prefixes: Sequence[str] = (),
    cwt_prefixes: Sequence[str] = (),
) -> Dict[str, List[int]]:
    """
    Split a flat ``SequenceExtractor`` feature-name list into modality branches.

    STFT/CWT columns are detected first because their ``feature_prefix``
    appears as an infix regardless of which raw sensor column produced them
    (e.g. ``"acc_x_stft0_stft_mean_power"``), so they are never miscounted as
    accelerometer features just because the name starts with ``"acc_"``.
    Everything else is grouped by its native sensor-column prefix. Any feature
    that still does not match the known modal prefixes is treated as a hard
    error so nothing is silently dropped into a ``misc`` bucket.
    """

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


def describe_branches(
    feature_names: Sequence[str],
    groups: Dict[str, List[int]],
) -> pd.DataFrame:
    """Small diagnostic table: how many raw features landed in each branch."""

    rows = []
    for name in BRANCH_ORDER:
        idxs = groups.get(name, [])
        if not idxs:
            continue
        example_cols = [feature_names[i] for i in idxs[:3]]
        rows.append(
            {
                "branch": name,
                "n_features": len(idxs),
                "kind": "static (per-sequence)" if name in STATIC_BRANCHES else "time-series",
                "example_columns": ", ".join(example_cols) + (", ..." if len(idxs) > 3 else ""),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


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
    dense_units: int = 128,
    dropout: float = 0.3,
    conv_dropout: float = 0.1,
    learning_rate: float = 1e-3,
    use_batchnorm: bool = True,
    random_state: int = 42,
):
    """
    Build a modality-specific multi-branch Keras classifier.

    Each key of ``time_series_shapes`` becomes its own ``Input`` followed by
    a small Conv1D stack. The branch depth can be set globally or per-modality
    via ``branch_num_conv_layers``; the common default is a small shallow stack,
    while the custom mode lets you set exact per-branch schedules, such as
    ``{'thermo': 1, 'acc': 3}`` or ``{'acc': [32, 64, 128], 'thermo': [32]}``.
    Each key of ``static_shapes`` becomes its own ``Input`` followed by a small
    Dense stack. Every branch embedding is concatenated and passed through a
    shared classification head.
    """

    if not TENSORFLOW_AVAILABLE:
        raise ImportError(
            "MultiBranchSequenceClassifier requires TensorFlow/Keras. "
            "Install with: pip install tensorflow"
        )

    tf.random.set_seed(random_state)

    branch_filters = _json_maybe(branch_filters) or DEFAULT_BRANCH_FILTERS
    branch_kernel_sizes = _json_maybe(branch_kernel_sizes) or DEFAULT_BRANCH_KERNEL_SIZES
    branch_pool_sizes = _json_maybe(branch_pool_sizes) or DEFAULT_BRANCH_POOL_SIZES
    if isinstance(branch_num_conv_layers, dict):
        branch_num_conv_layers = {
            str(k): max(1, int(v)) for k, v in branch_num_conv_layers.items()
        }
    else:
        branch_num_conv_layers = max(1, int(branch_num_conv_layers))
    branch_filter_mode = str(branch_filter_mode).lower()
    if branch_filter_mode not in {"custom", "double", "constant"}:
        raise ValueError("branch_filter_mode must be 'custom', 'double', or 'constant'.")

    inputs = []
    branch_embeddings = []

    # ---- time-series (Conv1D) branches, one per raw sensor modality ----
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
                layer_filters = int(raw_filter_value) * (2**layer_idx)
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
                filters=layer_filters,
                kernel_size=kernel_size,
                padding="same",
                name=f"{name}_conv{layer_idx + 1}",
            )(x)
            if use_batchnorm:
                x = layers.BatchNormalization(name=f"{name}_bn{layer_idx + 1}")(x)
            x = layers.Activation("relu", name=f"{name}_relu{layer_idx + 1}")(x)
            if conv_dropout:
                x = layers.SpatialDropout1D(conv_dropout, name=f"{name}_drop{layer_idx + 1}")(x)
            x = layers.MaxPooling1D(
                pool_size=pool_size, padding="same", name=f"{name}_pool{layer_idx + 1}"
            )(x)

        x = _attention_pool_over_time(name, x)
        x = layers.Dense(branch_dense_units, activation="relu", name=f"{name}_embed")(x)
        branch_embeddings.append(x)

    # ---- static (Dense) branches, one per per-sequence summary vector ----
    for name, dim in static_shapes.items():
        branch_in = layers.Input(shape=(dim,), name=f"{name}_input")
        inputs.append(branch_in)

        x = layers.Dense(static_branch_units, activation="relu", name=f"{name}_dense1")(branch_in)
        if use_batchnorm:
            x = layers.BatchNormalization(name=f"{name}_bn")(x)
        x = layers.Dense(static_branch_units, activation="relu", name=f"{name}_embed")(x)
        branch_embeddings.append(x)

    if not branch_embeddings:
        raise InvalidExtractorParams(
            "No modality branches were produced; nothing to feed the model."
        )

    merged = (
        layers.Concatenate(name="branch_concat")(branch_embeddings)
        if len(branch_embeddings) > 1
        else branch_embeddings[0]
    )

    head = layers.Dense(dense_units, activation="relu", name="head_dense")(merged)
    head = layers.Dropout(dropout, name="head_dropout")(head)
    outputs = layers.Dense(n_classes, activation="softmax", name="predictions")(head)

    model = models.Model(inputs=inputs, outputs=outputs, name="multibranch_sequence_classifier")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Sklearn-compatible sequence classifier
# ---------------------------------------------------------------------------


class MultiBranchSequenceClassifier(BaseEstimator, ClassifierMixin):
    """
    Sequence-level, modality-specific multi-branch Keras classifier.

    Mirrors ``RandomForestSequenceClassifier``'s contract so the two can drop
    into the same ``Pipeline`` / ``GridSearchCV`` / ``BayesSearchCV`` style:
    it accepts a raw row-level dataframe, uses a ``SequenceExtractor`` to
    build chunked per-modality tensors, aligns labels to ``sequence_id``, and
    returns sequence-level predictions indexed by ``sequence_id``.

    Unlike ``RandomForestSequenceClassifier`` (one flat feature vector per
    sequence), every sensor modality keeps its own branch of the network
    end-to-end -- see ``build_multibranch_model`` and ``group_feature_names``.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        primary_target: str = "bfrb",
        extractor: Optional[SequenceExtractor] = None,
        branch_filters: Optional[Dict[str, int]] = None,
        branch_kernel_sizes: Optional[Dict[str, int]] = None,
        branch_pool_sizes: Optional[Dict[str, int]] = None,
        branch_num_conv_layers: int = 2,
        branch_filter_mode: str = "custom",
        branch_dense_units: int = 64,
        static_branch_units: int = 32,
        dense_units: int = 128,
        dropout: float = 0.3,
        conv_dropout: float = 0.1,
        learning_rate: float = 1e-3,
        use_batchnorm: bool = True,
        epochs: int = 30,
        batch_size: int = 64,
        validation_split: float = 0.15,
        early_stopping_patience: int = 6,
        class_weight_mode: Optional[str] = "balanced",
        verbose: int = 0,
        random_state: int = 42,
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
        self.dense_units = dense_units
        self.dropout = dropout
        self.conv_dropout = conv_dropout
        self.learning_rate = learning_rate
        self.use_batchnorm = use_batchnorm
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.class_weight_mode = class_weight_mode
        self.verbose = verbose
        self.random_state = random_state

    # ----------------------------- defaults -----------------------------

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

    # ------------------------------- labels -------------------------------

    def _align_y(self, sequence_ids: np.ndarray, y: Any) -> np.ndarray:
        """Identical alignment contract to RandomForestSequenceClassifier._align_y,
        but ``sequence_ids`` may contain duplicates here (a sequence longer than
        the chunk window produces more than one chunk, each needing the same
        label)."""

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

    # ------------------------------ branches ------------------------------

    def _branch_groups(self, feature_names: Sequence[str]) -> Dict[str, List[int]]:
        stft_prefixes = [e.feature_prefix for e in getattr(self.extractor_, "stft_extractors", [])]
        cwt_prefixes = [e.feature_prefix for e in getattr(self.extractor_, "cwt_extractors", [])]
        return group_feature_names(feature_names, stft_prefixes, cwt_prefixes)

    def _split_branches(
        self, x_arr: np.ndarray, mask_arr: np.ndarray, groups: Dict[str, List[int]]
    ) -> Dict[str, np.ndarray]:
        branch_arrays: Dict[str, np.ndarray] = {}

        for name, idxs in groups.items():
            block = x_arr[:, :, idxs]

            if name in STATIC_BRANCHES:
                # STFT/CWT features are constant across a sequence's rows, so
                # collapse the time axis with a mask-aware mean rather than
                # feeding a repeated constant into a Conv1D branch.
                m = mask_arr[:, :, None].astype(np.float32)
                denom = np.clip(m.sum(axis=1), 1.0, None)
                branch_arrays[name] = (block * m).sum(axis=1) / denom
            else:
                branch_arrays[name] = block

        return branch_arrays

    def _transform_to_branches(self, X: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        check_is_fitted(self, ["extractor_", "branch_groups_"])

        try:
            chunks = self.extractor_.transform_chunks(X)
        except InvalidExtractorParams:
            raise
        except Exception as exc:
            raise InvalidExtractorParams(
                f"MultiBranchSequenceClassifier transform failed: {exc}"
            ) from exc

        branch_arrays = self._split_branches(chunks["X"], chunks["mask"], self.branch_groups_)
        return branch_arrays, chunks["sequence_ids"]

    # --------------------------------- fit ---------------------------------

    def fit(self, X: pd.DataFrame, y: Any = None, **fit_params):
        if y is None:
            raise ValueError("MultiBranchSequenceClassifier requires y.")

        if not TENSORFLOW_AVAILABLE:
            raise ImportError(
                "TensorFlow/Keras is required to fit MultiBranchSequenceClassifier. "
                "Install with: pip install tensorflow"
            )

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
            raise InvalidExtractorParams(
                "No modality branches could be built from the extracted features."
            )

        branch_arrays = self._split_branches(chunks["X"], chunks["mask"], self.branch_groups_)

        flat_check = np.concatenate(
            [arr.reshape(len(arr), -1) for arr in branch_arrays.values()], axis=1
        )
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

        time_series_shapes = {
            name: arr.shape[1:] for name, arr in branch_arrays.items() if name in TIME_SERIES_BRANCHES
        }
        static_shapes = {
            name: arr.shape[1] for name, arr in branch_arrays.items() if name in STATIC_BRANCHES
        }

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
            dense_units=self.dense_units,
            dropout=self.dropout,
            conv_dropout=self.conv_dropout,
            learning_rate=self.learning_rate,
            use_batchnorm=self.use_batchnorm,
            random_state=self.random_state,
        )

        self.input_order_ = list(time_series_shapes.keys()) + list(static_shapes.keys())
        model_inputs = [branch_arrays[name] for name in self.input_order_]

        cb = []
        if self.early_stopping_patience and self.early_stopping_patience > 0:
            cb.append(
                callbacks.EarlyStopping(
                    monitor="val_loss" if self.validation_split else "loss",
                    patience=self.early_stopping_patience,
                    restore_best_weights=True,
                )
            )

        history = self.model_.fit(
            model_inputs,
            y_enc,
            sample_weight=sample_weight,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split or 0.0,
            callbacks=cb,
            verbose=self.verbose,
        )
        self.history_ = history.history
        return self

    # ------------------------------ inference ------------------------------

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["model_", "le_", "input_order_"])

        branch_arrays, sequence_ids = self._transform_to_branches(X)
        model_inputs = [branch_arrays[name] for name in self.input_order_]
        probs = self.model_.predict(model_inputs, verbose=0)

        proba_df = pd.DataFrame(probs, columns=self.le_.classes_)
        proba_df.insert(0, "sequence_id", sequence_ids)

        # A sequence longer than the chunk window produces more than one
        # chunk; average their probabilities back down to one row/sequence.
        agg = proba_df.groupby("sequence_id", sort=True).mean()
        agg.index.name = getattr(self.extractor_, "sequence_col", "sequence_id")
        return agg

    def predict(self, X: pd.DataFrame) -> pd.Series:
        proba = self.predict_proba(X)
        pred_idx = np.argmax(proba.to_numpy(), axis=1)
        preds = self.le_.inverse_transform(pred_idx)

        return pd.Series(
            preds,
            index=proba.index,
            name=self.primary_target,
        ).sort_index()

    def score(self, X: pd.DataFrame, y: Any, sample_weight=None) -> float:
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
                return competition_score(
                    y_true,
                    preds_aligned,
                    y_true_binary=y_true_binary,
                    target_only_macro=True,
                )

            return f1_score(y_true, preds_aligned, average="macro", zero_division=0)

        y_arr = np.asarray(y)
        preds_arr = preds.to_numpy() if hasattr(preds, "to_numpy") else np.asarray(preds)

        return f1_score(y_arr, preds_arr, average="macro", zero_division=0)

    # ------------------------------ utilities ------------------------------

    def get_history_dict(self) -> Dict[str, List[float]]:
        return getattr(self, "history_", {})

    def summarize_model(self) -> None:
        if hasattr(self, "model_"):
            self.model_.summary()
        else:
            print("Model is not fitted yet.")

    def branch_summary(self) -> pd.DataFrame:
        check_is_fitted(self, ["feature_names_", "branch_groups_"])
        return describe_branches(self.feature_names_, self.branch_groups_)
