"""
level1_tcn.py

Level One architecture:

    Masked Multi-Scale TCN / ResNet
    + per-channel normalization
    + masked temporal convolutions
    + residual dilated blocks
    + masked mean/max pooling
    + class-balanced cross-entropy

This module is designed to be imported by a notebook.
"""

from __future__ import annotations

import json
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.validation import check_is_fitted

warnings.filterwarnings("ignore")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    TF_AVAILABLE = True
except Exception:
    tf = None
    keras = None
    layers = None
    TF_AVAILABLE = False

try:
    from base_utils_qwen import (
        InvalidExtractorParams,
        SequenceExtractor,
        validate_sequence_extractor_params,
    )

    BASE_UTILS_AVAILABLE = True
except Exception:
    SequenceExtractor = None
    InvalidExtractorParams = ValueError
    validate_sequence_extractor_params = None
    BASE_UTILS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Masked pooling helpers
# ---------------------------------------------------------------------------


if TF_AVAILABLE:

    def _masked_mean(inputs):
        x, mask = inputs
        mask = tf.expand_dims(tf.cast(mask, tf.float32), -1)
        summed = tf.reduce_sum(x * mask, axis=1)
        count = tf.reduce_sum(mask, axis=1) + 1e-9
        return summed / count

    def _masked_max(inputs):
        x, mask = inputs
        mask = tf.expand_dims(tf.cast(mask, tf.float32), -1)
        x_masked = x * mask + (1.0 - mask) * (-1e9)
        return tf.reduce_max(x_masked, axis=1)

else:

    def _masked_mean(inputs):
        raise ImportError("TensorFlow is required for MaskedMultiScaleTCNClassifier.")

    def _masked_max(inputs):
        raise ImportError("TensorFlow is required for MaskedMultiScaleTCNClassifier.")


# ---------------------------------------------------------------------------
# Level One TCN classifier
# ---------------------------------------------------------------------------


class MaskedMultiScaleTCNClassifier(BaseEstimator, ClassifierMixin):
    """
    Level One masked multi-scale TCN classifier.

    This estimator accepts raw row-level sensor DataFrames and returns
    sequence-level predictions.

    It is sklearn-compatible enough to be used with GridSearchCV /
    BayesSearchCV, although for deep models manual search is often easier.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        primary_target: str = "bfrb",
        sequence_col: str = "sequence_id",
        extractor: Optional[Any] = None,
        filters: int = 64,
        num_blocks: int = 3,
        kernel_size: int = 3,
        dilations: Tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 20,
        patience: int = 5,
        validation_split: float = 0.15,
        random_state: int = 42,
        verbose: int = 1,
        class_weight: str = "balanced",
    ):
        self.primary_target = primary_target
        self.sequence_col = sequence_col
        self.extractor = extractor
        self.filters = filters
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.validation_split = validation_split
        self.random_state = random_state
        self.verbose = verbose
        self.class_weight = class_weight

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def _parse_int_tuple(self, value: Any, default: Tuple[int, ...]) -> Tuple[int, ...]:
        if value is None:
            return tuple(default)

        if isinstance(value, (list, tuple, np.ndarray)):
            return tuple(int(v) for v in value)

        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return tuple(int(v) for v in parsed)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

            cleaned = value.replace("|", ",").replace(" ", "")
            parts = [p for p in cleaned.split(",") if p]
            return tuple(int(p) for p in parts)

        return (int(value),)

    def _default_extractor(self):
        if not BASE_UTILS_AVAILABLE:
            raise ImportError("SequenceExtractor requires base_utils_qwen.")

        return SequenceExtractor(
            acc_modes="raw|velocity|jerk",
            rotation_modes="quaternion|angular_velocity",
            tof_modes="pooled_stats|sensor_stats",
            thm_modes="centered_diff",
            motion_filter_mode=None,
            use_dead_reckoning=False,
            compute_dt=True,
            interp_mode="linear",
            maxlen=200,
            padding_value=0.0,
            sequence_col=self.sequence_col,
            chunk_window_size=200,
            chunk_stride=None,
            output_format="chunks",
            add_global_context=False,
            resample_modalities=False,
        )

    def _validate_tcn_extractor(self) -> None:
        if self.extractor_ is None:
            return

        if hasattr(self.extractor_, "set_params"):
            try:
                self.extractor_.set_params(output_format="chunks")
            except Exception:
                pass

            params = self.extractor_.get_params()
            chunk_window = params.get("chunk_window_size")
            maxlen = params.get("maxlen")

            if chunk_window is None or int(chunk_window) <= 0:
                try:
                    fixed_win = int(maxlen) if maxlen else 200
                except (TypeError, ValueError):
                    fixed_win = 200

                if fixed_win <= 0:
                    fixed_win = 200

                self.extractor_.set_params(chunk_window_size=fixed_win)

        if validate_sequence_extractor_params is not None:
            validate_sequence_extractor_params(
                self.extractor_.get_params(),
                for_frame_output=False,
            )

    def _chunk_params(self) -> Tuple[int, Optional[int], Optional[int], float]:
        params: Dict[str, Any] = {}

        if hasattr(self, "extractor_") and self.extractor_ is not None:
            params = self.extractor_.get_params()
        elif self.extractor is not None:
            params = self.extractor.get_params()

        return (
            int(params.get("maxlen", 200) or 200),
            params.get("chunk_window_size"),
            params.get("chunk_stride"),
            float(params.get("padding_value", 0.0) or 0.0),
        )

    def _collapse_y(self, y: Any) -> pd.Series:
        seq_col = self.sequence_col

        if isinstance(y, pd.DataFrame):
            yy = y.copy()

            if seq_col not in yy.columns:
                if yy.index.name == seq_col:
                    yy = yy.reset_index()
                else:
                    raise ValueError(f"y DataFrame must contain {seq_col}.")

            if self.primary_target not in yy.columns:
                raise ValueError(f"y DataFrame must contain target column: {self.primary_target}")

            y_seq = (
                yy.drop_duplicates(seq_col)
                .set_index(seq_col)[self.primary_target]
            )

            return y_seq.sort_index()

        if isinstance(y, pd.Series):
            if y.index.name == seq_col:
                return y.sort_index()

            raise ValueError("y Series must have sequence_id as its index.")

        raise ValueError("y must be a pandas DataFrame or Series.")

    # -----------------------------------------------------------------------
    # Fallback sequence extraction
    # -----------------------------------------------------------------------

    def _fallback_extract(self, X: pd.DataFrame, fit: bool = False):
        df = X.copy()

        if self.sequence_col not in df.columns:
            if df.index.name == self.sequence_col:
                df = df.reset_index()
            else:
                df[self.sequence_col] = 0

        sensor_prefixes = ("acc_", "rot_", "tof_", "thm_", "lin_acc")

        if fit:
            sensor_cols = [c for c in df.columns if c.startswith(sensor_prefixes)]

            if not sensor_cols:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                sensor_cols = [
                    c
                    for c in numeric_cols
                    if c
                    not in {
                        self.sequence_col,
                        "sequence_counter",
                        "is_target",
                        "row_id",
                    }
                ]

            if not sensor_cols:
                df["sensor_0"] = 0.0
                sensor_cols = ["sensor_0"]

            self.fallback_sensor_cols_ = list(sensor_cols)

        else:
            sensor_cols = list(getattr(self, "fallback_sensor_cols_", []))

            if not sensor_cols:
                sensor_cols = [c for c in df.columns if c.startswith(sensor_prefixes)]

                if not sensor_cols:
                    df["sensor_0"] = 0.0
                    sensor_cols = ["sensor_0"]

                self.fallback_sensor_cols_ = list(sensor_cols)

            missing = [c for c in sensor_cols if c not in df.columns]
            for c in missing:
                df[c] = 0.0

        for c in sensor_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").replace(-1.0, np.nan)

        df[sensor_cols] = (
            df.groupby(self.sequence_col, sort=False)[sensor_cols]
            .transform(
                lambda g: g.interpolate(method="linear", limit_direction="both")
                .ffill()
                .bfill()
            )
            .fillna(0.0)
        )

        maxlen, chunk_window_size, chunk_stride, _padding_value = self._chunk_params()

        win = chunk_window_size
        if win is None or int(win) <= 0:
            win = maxlen or 64

        stride = chunk_stride
        if stride is None or int(stride) <= 0:
            stride = win

        chunks = []
        masks = []
        seq_ids = []

        for seq_id, g in df.groupby(self.sequence_col, sort=False):
            arr = g[sensor_cols].to_numpy(dtype=np.float32)
            L = len(arr)

            if L == 0:
                continue

            if L <= win:
                pad_len = win - L

                if pad_len > 0:
                    pad = np.zeros((pad_len, arr.shape[1]), dtype=np.float32)
                    chunk = np.vstack([arr, pad])
                    mask = np.concatenate(
                        [
                            np.ones(L, dtype=bool),
                            np.zeros(pad_len, dtype=bool),
                        ]
                    )
                else:
                    chunk = arr[:win]
                    mask = np.ones(win, dtype=bool)

                chunks.append(chunk)
                masks.append(mask)
                seq_ids.append(seq_id)

            else:
                starts = np.arange(0, L - win + 1, stride)

                if len(starts) == 0 or starts[-1] + win < L:
                    starts = np.append(starts, L - win)

                for start in starts:
                    chunks.append(arr[start : start + win])
                    masks.append(np.ones(win, dtype=bool))
                    seq_ids.append(seq_id)

        if not chunks:
            raise ValueError("Fallback sequence extraction produced no chunks.")

        X_chunks = np.stack(chunks, axis=0).astype(np.float32)
        mask_arr = np.stack(masks, axis=0).astype(bool)
        seq_arr = np.array(seq_ids)

        X_chunks = np.nan_to_num(X_chunks, nan=0.0, posinf=0.0, neginf=0.0)

        return X_chunks, mask_arr, seq_arr

    # -----------------------------------------------------------------------
    # Chunk extraction
    # -----------------------------------------------------------------------

    def _extract_chunks(self, X: pd.DataFrame, fit: bool = False):
        if BASE_UTILS_AVAILABLE:
            if fit:
                self.extractor_ = (
                    clone(self.extractor)
                    if self.extractor is not None
                    else self._default_extractor()
                )
                self._validate_tcn_extractor()
                self.extractor_.fit(X)
                out = self.extractor_.transform(X)
            else:
                check_is_fitted(self, ["extractor_"])
                out = self.extractor_.transform(X)

            if isinstance(out, dict):
                X_chunks = out["X"]
                mask = out.get("mask", np.ones(X_chunks.shape[:2], dtype=bool))
                seq_ids = out["sequence_ids"]

                X_chunks = np.asarray(X_chunks, dtype=np.float32)
                mask = np.asarray(mask, dtype=bool)
                seq_ids = np.asarray(seq_ids)

                if len(seq_ids) == 0:
                    raise InvalidExtractorParams(
                        "SequenceExtractor produced zero chunks for TCN input."
                    )

                X_chunks = np.nan_to_num(X_chunks, nan=0.0, posinf=0.0, neginf=0.0)

                if not np.isfinite(X_chunks).all():
                    raise InvalidExtractorParams(
                        "SequenceExtractor produced non-finite chunk values."
                    )

                return X_chunks, mask, seq_ids

        return self._fallback_extract(X, fit=fit)

    # -----------------------------------------------------------------------
    # Normalization
    # -----------------------------------------------------------------------

    def _compute_channel_stats(self, X_chunks: np.ndarray, mask: np.ndarray):
        X_chunks = np.asarray(X_chunks, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)

        valid = mask[..., None].astype(np.float32)

        count = valid.sum(axis=(0, 1))
        count = np.maximum(count, 1.0)

        mean = (X_chunks * valid).sum(axis=(0, 1)) / count

        centered = (X_chunks - mean[None, None, :]) * valid
        std = np.sqrt((centered**2).sum(axis=(0, 1)) / count)
        std = np.where(std < 1e-6, 1.0, std)

        return mean.astype(np.float32), std.astype(np.float32)

    # -----------------------------------------------------------------------
    # Sequence split
    # -----------------------------------------------------------------------

    def _split_sequences(self, seq_ids: np.ndarray):
        if self.validation_split is None or self.validation_split <= 0:
            return None

        unique_seq = np.unique(seq_ids)

        if len(unique_seq) < 6:
            return None

        rng = np.random.default_rng(self.random_state)
        rng.shuffle(unique_seq)

        n_val = max(1, int(len(unique_seq) * float(self.validation_split)))

        val_seqs = set(unique_seq[:n_val])
        train_seqs = set(unique_seq[n_val:])

        if len(train_seqs) == 0:
            return None

        return train_seqs, val_seqs

    # -----------------------------------------------------------------------
    # Model builder
    # -----------------------------------------------------------------------

    def _build_model(self):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for MaskedMultiScaleTCNClassifier.")

        dilations = self._parse_int_tuple(self.dilations, default=(1, 2, 4))

        inp = layers.Input(shape=(self.window_, self.n_features_), name="sequence_input")
        mask_input = layers.Input(shape=(self.window_,), name="mask_input")

        channel_mean = self.channel_mean_
        channel_std = self.channel_std_

        x = layers.Lambda(
            lambda z: (z - channel_mean) / (channel_std + 1e-6),
            name="channel_standardization",
        )(inp)

        mask_float = layers.Lambda(
            lambda m: tf.expand_dims(tf.cast(m, tf.float32), -1),
            name="mask_expand",
        )(mask_input)

        x = layers.Multiply(name="apply_input_mask")([x, mask_float])

        x = layers.Conv1D(self.filters, 1, padding="same", name="input_projection")(x)
        x = layers.LayerNormalization(name="input_norm")(x)
        x = layers.Activation("gelu", name="input_gelu")(x)
        x = layers.SpatialDropout1D(self.dropout, name="input_dropout")(x)

        for block_idx in range(self.num_blocks):
            for dilation_idx, dilation in enumerate(dilations):
                residual = x

                y = layers.Conv1D(
                    self.filters,
                    self.kernel_size,
                    padding="same",
                    dilation_rate=dilation,
                    name=f"block{block_idx}_d{dilation}_conv1",
                )(x)
                y = layers.LayerNormalization(name=f"block{block_idx}_d{dilation}_norm1")(y)
                y = layers.Activation("gelu", name=f"block{block_idx}_d{dilation}_gelu1")(y)
                y = layers.SpatialDropout1D(
                    self.dropout,
                    name=f"block{block_idx}_d{dilation}_drop1",
                )(y)

                y = layers.Conv1D(
                    self.filters,
                    self.kernel_size,
                    padding="same",
                    dilation_rate=dilation,
                    name=f"block{block_idx}_d{dilation}_conv2",
                )(y)
                y = layers.LayerNormalization(name=f"block{block_idx}_d{dilation}_norm2")(y)
                y = layers.SpatialDropout1D(
                    self.dropout,
                    name=f"block{block_idx}_d{dilation}_drop2",
                )(y)

                if residual.shape[-1] != self.filters:
                    residual = layers.Conv1D(
                        self.filters,
                        1,
                        padding="same",
                        name=f"block{block_idx}_d{dilation}_shortcut",
                    )(residual)

                x = layers.Add(name=f"block{block_idx}_d{dilation}_add")([residual, y])
                x = layers.Activation("gelu", name=f"block{block_idx}_d{dilation}_out_gelu")(x)
                x = layers.Multiply(name=f"block{block_idx}_d{dilation}_mask")([x, mask_float])

        mean_pool = layers.Lambda(_masked_mean, name="masked_mean_pool")([x, mask_input])
        max_pool = layers.Lambda(_masked_max, name="masked_max_pool")([x, mask_input])

        pooled = layers.Concatenate(name="pooled_concat")([mean_pool, max_pool])

        z = layers.Dense(self.filters, activation="gelu", name="head_dense")(pooled)
        z = layers.Dropout(self.dropout, name="head_dropout")(z)

        outputs = layers.Dense(self.n_classes_, activation="softmax", name="classifier")(z)

        model = keras.Model(inputs=[inp, mask_input], outputs=outputs, name="level1_masked_tcn")

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    # -----------------------------------------------------------------------
    # Fit
    # -----------------------------------------------------------------------

    def _fit_inner(self, X: pd.DataFrame, y: Any):
        y_seq = self._collapse_y(y)

        self.label_encoder_ = LabelEncoder()
        self.label_encoder_.fit(y_seq.values)

        self.classes_ = self.label_encoder_.classes_
        self.n_classes_ = len(self.classes_)

        X_chunks, mask, seq_ids = self._extract_chunks(X, fit=True)

        known = np.isin(seq_ids, y_seq.index.values)

        X_chunks = X_chunks[known]
        mask = mask[known]
        seq_ids = seq_ids[known]

        if len(seq_ids) == 0:
            raise InvalidExtractorParams("No labeled sequences available after extraction.")

        y_chunk = self.label_encoder_.transform(y_seq.loc[seq_ids].values)

        self.window_ = int(X_chunks.shape[1])
        self.n_features_ = int(X_chunks.shape[2])

        split = self._split_sequences(seq_ids)

        if split is not None:
            train_seqs, val_seqs = split

            train_idx = np.array([s in train_seqs for s in seq_ids])
            val_idx = np.array([s in val_seqs for s in seq_ids])

            if train_idx.sum() == 0 or val_idx.sum() == 0:
                train_idx = np.ones(len(seq_ids), dtype=bool)
                val_idx = None

        else:
            train_idx = np.ones(len(seq_ids), dtype=bool)
            val_idx = None

        self.channel_mean_, self.channel_std_ = self._compute_channel_stats(
            X_chunks[train_idx],
            mask[train_idx],
        )

        self.model_ = self._build_model()

        class_weight_dict = None

        if self.class_weight == "balanced":
            unique_train_classes = np.unique(y_chunk[train_idx])

            if len(unique_train_classes) > 1:
                cw_array = compute_class_weight(
                    class_weight="balanced",
                    classes=unique_train_classes,
                    y=y_chunk[train_idx],
                )

                class_weight_dict = dict(zip(unique_train_classes, cw_array))

        callbacks = []

        if val_idx is not None:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.patience,
                    restore_best_weights=True,
                    verbose=self.verbose,
                )
            )

            history = self.model_.fit(
                [X_chunks[train_idx], mask[train_idx]],
                y_chunk[train_idx],
                validation_data=(
                    [X_chunks[val_idx], mask[val_idx]],
                    y_chunk[val_idx],
                ),
                epochs=self.epochs,
                batch_size=self.batch_size,
                class_weight=class_weight_dict,
                callbacks=callbacks,
                verbose=self.verbose,
                shuffle=True,
            )

        else:
            history = self.model_.fit(
                [X_chunks[train_idx], mask[train_idx]],
                y_chunk[train_idx],
                epochs=self.epochs,
                batch_size=self.batch_size,
                class_weight=class_weight_dict,
                callbacks=callbacks,
                verbose=self.verbose,
                shuffle=True,
            )

        self.history_ = history.history

    def fit(self, X: pd.DataFrame, y: Any = None, **fit_params):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for MaskedMultiScaleTCNClassifier.")

        if y is None:
            raise ValueError("MaskedMultiScaleTCNClassifier requires y.")

        try:
            self._fit_inner(X, y)
        except InvalidExtractorParams:
            raise
        except Exception as exc:
            raise InvalidExtractorParams(
                f"MaskedMultiScaleTCNClassifier fit failed: {exc}"
            ) from exc

        return self

    # -----------------------------------------------------------------------
    # Predict
    # -----------------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["model_", "label_encoder_", "channel_mean_", "channel_std_"])

        X_chunks, mask, seq_ids = self._extract_chunks(X, fit=False)

        if len(seq_ids) == 0:
            return pd.DataFrame()

        probs = self.model_.predict(
            [X_chunks, mask],
            batch_size=self.batch_size,
            verbose=0,
        )

        prob_df = pd.DataFrame(probs)
        prob_df[self.sequence_col] = seq_ids

        prob_seq = (
            prob_df.groupby(self.sequence_col, sort=True)
            .mean()
        )

        return prob_seq

    def predict(self, X: pd.DataFrame) -> pd.Series:
        prob_seq = self.predict_proba(X)

        if len(prob_seq) == 0:
            return pd.Series(dtype=object, name=self.primary_target)

        pred_idx = np.argmax(prob_seq.to_numpy(), axis=1)
        preds = self.label_encoder_.inverse_transform(pred_idx)

        return pd.Series(
            preds,
            index=prob_seq.index,
            name=self.primary_target,
        ).sort_index()

    # -----------------------------------------------------------------------
    # Score
    # -----------------------------------------------------------------------

    def score(self, X: pd.DataFrame, y: Any, sample_weight=None) -> float:
        from sklearn.metrics import f1_score

        preds = self.predict(X)
        y_seq = self._collapse_y(y)

        preds_aligned = preds.reindex(y_seq.index).fillna("non_bfrb").to_numpy()
        y_true = y_seq.values

        return f1_score(
            y_true,
            preds_aligned,
            average="macro",
            zero_division=0,
        )

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    def summarize_model(self):
        if hasattr(self, "model_"):
            self.model_.summary()
        else:
            print("Model is not fitted yet.")

    def get_history_dict(self) -> Dict[str, List[float]]:
        return getattr(self, "history_", {})