"""Supervised transfer and masked-signal pretraining for the v1 backbone.

The notebook supplies raw frames containing both target and non-target examples.
``SequenceExtractor`` owns stage selection and all feature extraction. Every
search fold and internal validation split is made before either training stage.
No evaluation recordings are used to fit feature extraction or pretraining.

Model 1 learns the original pretraining gesture labels, replaces only its output
heads, and fine-tunes the same branches, fusion, attention and dense backbone.
Model 2 masks raw sensor spans *before* recomputing derived/STFT/CWT features and
learns to reconstruct intentionally hidden, originally valid raw measurements.
Its temporary temporal decoder is discarded before target fine-tuning.
"""

from __future__ import annotations

import copy
import inspect
import math
import re
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

if __package__:
    from . import multibranch_utils as mb
else:
    import multibranch_utils as mb


def _sequence_labels(X, y, ids, sequence_col, column):
    """Strict label alignment: never invent an unknown pretraining class."""
    if isinstance(y, pd.DataFrame) and column in y:
        frame = y.reset_index() if sequence_col not in y and y.index.name == sequence_col else y
    elif column in X:
        frame = X
    else:
        raise ValueError(f"Pretraining requires original label column {column!r} in X or y.")
    if sequence_col not in frame:
        raise ValueError(f"Label metadata must contain {sequence_col!r}.")
    grouped = frame.groupby(sequence_col, sort=False)[column]
    if (grouped.nunique(dropna=False) > 1).any():
        raise ValueError(f"{column!r} must be constant within an original sequence.")
    aligned = frame.drop_duplicates(sequence_col).set_index(sequence_col)[column].reindex(ids)
    if aligned.isna().any():
        raise ValueError(f"Missing original labels for pretraining column {column!r}.")
    return aligned.to_numpy()


def _partition(X, labels, fraction, sequence_col, random_state, group_column=None):
    ids = pd.Index(X[sequence_col].drop_duplicates(), name=sequence_col)
    if not fraction:
        return X, None
    if not 0 < float(fraction) < 1:
        raise ValueError("Stage validation fractions must be in [0, 1).")
    if len(ids) < 2:
        raise ValueError("A validation split requires at least two original sequences.")
    if group_column is not None:
        if group_column not in X:
            raise ValueError(f"Validation group column {group_column!r} is absent.")
        meta = X[[sequence_col, group_column]].drop_duplicates()
        if meta[sequence_col].duplicated().any() or meta[group_column].isna().any():
            raise ValueError("Validation groups must be non-missing and constant per sequence.")
        groups = meta.set_index(sequence_col).reindex(ids)[group_column].to_numpy()
        splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=random_state)
        train_idx, val_idx = next(splitter.split(ids, groups=groups))
        train_ids, val_ids = ids[train_idx], ids[val_idx]
    else:
        try:
            train_ids, val_ids = train_test_split(
                ids, test_size=fraction, random_state=random_state, stratify=labels,
            )
        except ValueError:
            train_ids, val_ids = train_test_split(ids, test_size=fraction, random_state=random_state)
    return X.loc[X[sequence_col].isin(train_ids)], X.loc[X[sequence_col].isin(val_ids)]


def _weight_snapshot(model):
    return {layer.name: [value.copy() for value in layer.get_weights()]
            for layer in model.layers if layer.weights}


def _changed_layers(before, model):
    changed = []
    for layer in model.layers:
        weights = layer.get_weights()
        if layer.name in before and any(
            not np.array_equal(old, new) for old, new in zip(before[layer.name], weights)
        ):
            changed.append(layer.name)
    return changed


def transfer_backbone_weights(source, destination, layer_names=None):
    """Transfer every shape-compatible non-output layer, failing on mismatch.

    Explicit layer names are stable. Unnamed activation/reshape layers have no
    weights and therefore do not need name matching. Classification and auxiliary
    prediction heads are intentionally rebuilt for the final label space.
    """
    output_names = set(source.output_names) | set(destination.output_names)
    transferred, skipped = [], []
    for layer in destination.layers:
        if not layer.weights:
            continue
        if (layer.name in output_names
                or layer.name.startswith(tuple(name + "_" for name in output_names))
                or layer.name.startswith(("phase_aux", "ssl_"))):
            skipped.append(layer.name)
            continue
        if layer_names is not None and layer.name not in layer_names:
            skipped.append(layer.name)
            continue
        try:
            previous = source.get_layer(layer.name)
        except ValueError as exc:
            raise ValueError(f"Pretraining backbone is missing layer {layer.name!r}.") from exc
        old, new = previous.get_weights(), layer.get_weights()
        if len(old) != len(new) or any(a.shape != b.shape for a, b in zip(old, new)):
            raise ValueError(
                f"Pretraining/fine-tuning backbone shape mismatch at {layer.name!r}; "
                "keep architecture dimensions identical between stages."
            )
        layer.set_weights(old)
        transferred.append(layer.name)
    if not transferred:
        raise ValueError("No backbone weights were transferred.")
    return transferred, skipped


_DataSequence = mb.tf.keras.utils.PyDataset if mb.TENSORFLOW_AVAILABLE else object


class MaskedSensorSequence(_DataSequence):
    """Bounded-memory SSL batches; recompute every branch from corrupted raw data.

    The index stores only sequence IDs and chunk numbers. A minibatch extracts
    just the recordings needed for that batch. Clean target, deliberate-mask,
    original-validity and padding arrays share the base extractor's cropping,
    window, stride, cap and final-window implementation.
    """

    def __init__(self, owner, X, *, validation=False):
        if mb.TENSORFLOW_AVAILABLE:
            super().__init__(workers=1, use_multiprocessing=False, max_queue_size=1)
        self.owner, self.X, self.validation = owner, X, validation
        self.sequence_col = owner.extractor_.sequence_col
        self.columns = owner.ssl_columns_
        self.positions = X.groupby(self.sequence_col, sort=False).indices
        self.epoch = 0
        self.random_state = (int(owner.ssl_validation_mask_seed) if validation
                             else int(owner.random_state))
        self.index = []
        self.sequence_number = {}
        self._cache = OrderedDict()
        # Chunk geometry uses a one-column view, avoiding a full sensor tensor.
        for number, (seq_id, positions) in enumerate(self.positions.items()):
            frame = X.iloc[positions][[self.sequence_col]].copy()
            frame["__ssl_geometry"] = 0.0
            chunks = self._raw_chunks(frame, ["__ssl_geometry"])
            self.sequence_number[seq_id] = number
            self.index.extend((seq_id, i) for i in range(len(chunks["X"])))
        if not self.index:
            raise ValueError("No pretraining chunks remain after selection and quality filtering.")
        self.order = np.arange(len(self.index))
        self._set_order()

    def _raw_chunks(self, frame, columns):
        extractor = copy.copy(self.owner.extractor_)
        extractor.feature_names_in_ = list(columns)
        extractor.padding_value = 0.0
        return extractor._chunks_from_features(frame.copy())

    def _set_order(self):
        if self.owner.pretraining_shuffle and not self.validation:
            np.random.default_rng(self.random_state + self.epoch).shuffle(self.order)

    def __len__(self):
        return int(math.ceil(len(self.index) / self.owner.pretraining_batch_size))

    def on_epoch_end(self):
        self.epoch += 1
        self._cache.clear()
        self._set_order()

    def _sequence(self, seq_id):
        if seq_id in self._cache:
            return self._cache[seq_id]
        frame = self.X.iloc[self.positions[seq_id]].copy()
        values = frame[self.columns].to_numpy(dtype=np.float32)
        valid = self.owner._ssl_validity(frame)
        epoch = self.epoch if self.owner.ssl_resample_mask_each_epoch and not self.validation else 0
        rng = np.random.default_rng(np.random.SeedSequence([
            self.random_state, epoch, self.sequence_number[seq_id],
        ]))
        masked = np.zeros(values.shape, dtype=bool)
        n_rows, n_channels = values.shape
        groups = [np.arange(n_channels)] if self.owner.ssl_mask_modalities_together else [
            np.asarray(indices) for indices in self.owner._ssl_column_groups().values()
        ]
        for channels in groups:
            available = np.any(valid[:, channels], axis=1)
            needed = max(1, int(np.ceil(np.sum(available) * self.owner.ssl_mask_ratio)))
            chosen = np.zeros(n_rows, dtype=bool)
            attempts = 0
            while np.sum(chosen & available) < needed and attempts < max(1, n_rows * 4):
                eligible = np.flatnonzero(available & ~chosen)
                if not len(eligible):
                    break
                start = int(rng.choice(eligible))
                chosen[start:min(n_rows, start + int(self.owner.ssl_span_length))] = True
                attempts += 1
            masked[:, channels] = chosen[:, None] & valid[:, channels]
        corrupted = frame.copy()
        corrupt_values = values.copy()
        corrupt_values[masked] = float(self.owner.ssl_mask_value)
        corrupted.loc[:, self.columns] = corrupt_values
        # The shared extractor validates nonzero quaternions. A masked rotation
        # is represented by a configurable valid constant quaternion rather than
        # an all-zero vector; every derived rotation feature is still recomputed.
        quaternion = ["rot_w", "rot_x", "rot_y", "rot_z"]
        if all(column in corrupted for column in quaternion):
            replacement = np.asarray(mb._json_maybe(self.owner.ssl_quaternion_mask_value), dtype=float)
            for component, column in enumerate(quaternion):
                if column in self.columns:
                    hidden_rows = masked[:, self.columns.index(column)]
                    corrupted.loc[hidden_rows, column] = replacement[component]
            zero = np.linalg.norm(corrupted[quaternion].to_numpy(dtype=float), axis=1) == 0
            if np.any(zero):
                corrupted.loc[zero, quaternion] = replacement
        # All derived channels, frequency summaries and global context are now
        # recomputed from this corrupted input, never from the clean recording.
        extracted = self.owner.extractor_.transform_chunks(corrupted)
        target_frame = frame[[self.sequence_col] + self.columns].copy()
        target_frame.loc[:, self.columns] = np.where(valid, values, 0.0)
        clean = self._raw_chunks(target_frame, self.columns)
        # Construct the wide ToF/sensor blocks together. Inserting hundreds of
        # columns separately fragments pandas frames and floods training logs.
        mask_frame = pd.concat([
            frame[[self.sequence_col]],
            pd.DataFrame(masked.astype(np.float32), index=frame.index, columns=self.columns),
        ], axis=1)
        hidden = self._raw_chunks(mask_frame, self.columns)["X"]
        valid_frame = pd.concat([
            frame[[self.sequence_col]],
            pd.DataFrame(valid.astype(np.float32), index=frame.index, columns=self.columns),
        ], axis=1)
        originally_valid = self._raw_chunks(valid_frame, self.columns)["X"]
        if extracted["X"].shape[:2] != clean["X"].shape[:2]:
            raise ValueError("Raw reconstruction targets and extracted timesteps do not align.")
        target = ((clean["X"] - self.owner.ssl_target_mean_) / self.owner.ssl_target_scale_)
        weight = hidden * originally_valid * clean["mask"][:, :, None]
        weight *= self.owner.ssl_channel_weights_[None, None, :]
        # Store target values and independent reconstruction weights together;
        # the loss normalises each channel by its actual valid hidden count.
        packed = np.concatenate([np.where(weight > 0, target, 0.0), weight], axis=-1).astype(np.float32)
        branch = self.owner._split_branches(extracted["X"], extracted["mask"], self.owner.branch_groups_)
        inputs = self.owner._model_inputs(branch, extracted["mask"])
        result = (inputs, packed)
        self._cache[seq_id] = result
        # Small bounded cache permits overlapping chunk batches without keeping
        # all recordings or multiple augmented datasets resident in RAM.
        while len(self._cache) > int(self.owner.ssl_cache_sequences):
            self._cache.popitem(last=False)
        return result

    def __getitem__(self, index):
        chosen = self.order[index * self.owner.pretraining_batch_size:(index + 1) * self.owner.pretraining_batch_size]
        batches, targets = [], []
        for position in chosen:
            seq_id, chunk = self.index[position]
            inputs, packed = self._sequence(seq_id)
            batches.append([array[chunk] for array in inputs])
            targets.append(packed[chunk])
        inputs = tuple(np.stack([sample[k] for sample in batches]).astype(np.float32)
                       for k in range(len(batches[0])))
        return inputs, np.stack(targets)


class AdvancedMultiBranchSequenceClassifier(mb.MultiBranchSequenceClassifier):
    """The v1 estimator with one pretraining stage and a replaceable target head.

    ``**model_params`` accepts every parameter of MultiBranchSequenceClassifier;
    sklearn get_params/set_params/clone expose those inherited parameters and
    nested extractor/augmentation parameters normally. Stage architecture and
    regularisation inherit the backbone unless explicitly overridden with
    ``pretraining_model_params``; dimensions must remain transfer-compatible.
    """

    def __init__(
        self, pretraining_mode="supervised", pretraining_epochs=10,
        pretraining_batch_size=64, pretraining_learning_rate=None,
        pretraining_optimizer_name=None, pretraining_weight_decay=0.0,
        pretraining_momentum=0.9, pretraining_gradient_clip_norm=None,
        pretraining_gradient_clip_value=None, pretraining_loss_name="sparse_cce",
        pretraining_label_smoothing=0.0, pretraining_class_weight_mode=None,
        pretraining_validation_split=0.15, pretraining_early_stopping_patience=3,
        pretraining_use_lr_scheduler=False, pretraining_lr_scheduler_type="plateau",
        pretraining_lr_factor=0.5, pretraining_lr_patience=3, pretraining_min_lr=1e-6,
        pretraining_cosine_t_max=100, pretraining_warmup_epochs=0,
        pretraining_model_params=None, pretraining_augmentor=None,
        pretraining_shuffle=True, pretraining_verbose=None,
        pretraining_validation_group_column=None, validation_group_column=None,
        finetuning_head_epochs=0, finetuning_head_learning_rate=None,
        finetuning_learning_rate=None, ssl_mask_ratio=0.15, ssl_span_length=8,
        ssl_mask_value=0.0, ssl_mask_columns=None, ssl_mask_modalities_together=True,
        ssl_reconstruction_loss="huber", ssl_huber_delta=1.0,
        ssl_decoder_units=64, ssl_decoder_dropout=0.0,
        ssl_loss_channel_weights=None, ssl_standardize_targets=True,
        ssl_invalid_values=None, ssl_validation_mask_seed=1729,
        ssl_resample_mask_each_epoch=True, ssl_cache_sequences=4,
        ssl_quaternion_mask_value=(1.0, 0.0, 0.0, 0.0),
        **model_params,
    ):
        super().__init__(**model_params)
        local = locals()
        for name in self._stage_param_names():
            setattr(self, name, local[name])

    @classmethod
    def _stage_param_names(cls):
        return [name for name, param in inspect.signature(cls.__init__).parameters.items()
                if name not in {"self", "model_params"}]

    @classmethod
    def _get_param_names(cls):
        return sorted(set(mb.MultiBranchSequenceClassifier._get_param_names()) | set(cls._stage_param_names()))

    def _validate_pretraining(self):
        if self.pretraining_mode not in {"supervised", "masked_ssl"}:
            raise ValueError("pretraining_mode must be 'supervised' or 'masked_ssl'.")
        for name in ("pretraining_epochs", "pretraining_batch_size", "ssl_span_length"):
            if isinstance(getattr(self, name), bool) or int(getattr(self, name)) != getattr(self, name) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not 0 < float(self.ssl_mask_ratio) < 1:
            raise ValueError("ssl_mask_ratio must be in (0, 1).")
        if self.ssl_reconstruction_loss not in {"mse", "mae", "huber"}:
            raise ValueError("ssl_reconstruction_loss must be mse, mae, or huber.")
        if self.ssl_huber_delta <= 0 or not 0 <= self.ssl_decoder_dropout < 1:
            raise ValueError("ssl_huber_delta must be positive and ssl_decoder_dropout in [0, 1).")
        if int(self.ssl_decoder_units) < 0 or int(self.ssl_cache_sequences) < 0:
            raise ValueError("ssl_decoder_units and ssl_cache_sequences must be nonnegative.")
        for name in ("pretraining_learning_rate", "finetuning_learning_rate", "finetuning_head_learning_rate"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive or None to inherit the base learning rate.")
        quaternion = np.asarray(mb._json_maybe(self.ssl_quaternion_mask_value), dtype=float)
        if quaternion.shape != (4,) or not np.isfinite(quaternion).all() or not np.linalg.norm(quaternion):
            raise ValueError("ssl_quaternion_mask_value must contain four finite values with nonzero norm (w,x,y,z).")
        if self.pretraining_model_params is not None:
            allowed = set(inspect.signature(mb.build_multibranch_model).parameters) - {
                "time_series_shapes", "static_shapes", "n_classes", "sequence_length",
                "use_orientation_aux_head", "use_phase_aux_head", "custom_aux_heads",
            }
            overrides = mb._json_maybe(self.pretraining_model_params)
            if not isinstance(overrides, dict):
                raise ValueError("pretraining_model_params must be a dictionary or JSON dictionary.")
            unknown = set(overrides) - allowed
            if unknown:
                raise ValueError(f"Unsupported pretraining_model_params: {sorted(unknown)}")
        if not hasattr(self.extractor_, "select_pretraining_sequences"):
            raise TypeError("Use base_utils.SequenceExtractor with stage selection support.")

    def _split_validation_sequences(self, X, y):
        self.pretraining_baseline_ = str(self.extractor_.pretraining_source).lower() == "baseline"
        if not self.pretraining_baseline_:
            self._validate_pretraining()
        seq = self.extractor_.sequence_col
        # Reserve the same raw holdout pool for every source candidate. Reserving
        # a fresh target subset only for target/all pretraining would otherwise
        # change the amount of labelled fine-tuning data across experiments.
        fine_fraction = float(self.validation_split or 0.0)
        source_fraction = float(self.pretraining_validation_split or 0.0)
        reserve_fraction = max(fine_fraction, source_fraction)
        ids = X[seq].drop_duplicates()
        group_column = self.validation_group_column or self.pretraining_validation_group_column
        if (self.validation_group_column is not None and self.pretraining_validation_group_column is not None
                and self.validation_group_column != self.pretraining_validation_group_column):
            raise ValueError("Shared validation reservation requires the same group column for both stages.")
        raw_train, reserved = _partition(
            X, self._align_y(ids, y), reserve_fraction, seq, self.random_state,
            group_column,
        )
        self.outer_training_sequence_ids_ = pd.Index(raw_train[seq].drop_duplicates(), name=seq)
        self.validation_sequence_ids_ = pd.Index(
            [] if reserved is None else reserved[seq].drop_duplicates(), name=seq,
        )
        fine = self.extractor_.select_finetuning_sequences(raw_train, y)
        fine_validation = None
        if fine_fraction and reserved is not None:
            fine_validation = self.extractor_.select_finetuning_sequences(reserved, y)
            if fine_fraction < reserve_fraction and not fine_validation.empty:
                _, fine_validation = _partition(
                    fine_validation, self._align_y(fine_validation[seq].drop_duplicates(), y),
                    fine_fraction / reserve_fraction, seq, self.random_state,
                    group_column,
                )
        if fine.empty or (fine_validation is not None and fine_validation.empty):
            raise ValueError("Fine-tuning selection produced an empty training/validation partition.")
        self.train_sequence_ids_ = pd.Index(fine[seq].drop_duplicates(), name=seq)
        self.finetuning_sequence_ids_ = self.train_sequence_ids_.copy()
        if self.pretraining_baseline_:
            self._preprocessing_X = fine
            self.preprocessing_sequence_ids_ = pd.Index(fine[seq].drop_duplicates(), name=seq)
            self.pretraining_sequence_ids_ = pd.Index([], name=seq)
            self.pretraining_validation_sequence_ids_ = pd.Index([], name=seq)
            self.pretraining_history_ = {}
            self.pretraining_completed_ = False
            self.pretraining_changed_layers_ = []
            self.transferred_layer_names_ = []
            self.transferred_changed_layer_names_ = []
            return fine, fine_validation
        source = self.extractor_.select_pretraining_sequences(raw_train, y)
        if source.empty:
            raise ValueError("Pretraining selectors produced no sequences.")
        source_validation = None
        if source_fraction and reserved is not None:
            source_validation = self.extractor_.select_pretraining_sequences(reserved, y)
            if source_fraction < reserve_fraction and not source_validation.empty:
                labels = (_sequence_labels(
                    source_validation, y, source_validation[seq].drop_duplicates(), seq,
                    self.extractor_.pretraining_label_column,
                ) if self.pretraining_mode == "supervised" else None)
                _, source_validation = _partition(
                    source_validation, labels, source_fraction / reserve_fraction, seq,
                    self.extractor_.pretraining_random_state,
                    self.pretraining_validation_group_column or self.validation_group_column,
                )
            if source_validation.empty:
                raise ValueError("No pretraining validation examples satisfy the source selectors.")
        self._pretraining_X = source
        self._pretraining_validation_X = source_validation
        self._pretraining_y = y
        fitting_ids = pd.Index(pd.concat([source[seq], fine[seq]]).drop_duplicates(), name=seq)
        self._preprocessing_X = raw_train.loc[raw_train[seq].isin(fitting_ids)]
        self.preprocessing_sequence_ids_ = fitting_ids
        self.pretraining_validation_sequence_ids_ = pd.Index(
            [] if source_validation is None else source_validation[seq].drop_duplicates(), name=seq,
        )
        return fine, fine_validation

    def _extract_training_chunks(self, X_train):
        if self.pretraining_baseline_:
            chunks = super()._extract_training_chunks(X_train)
            self.finetuning_sequence_ids_ = pd.Index(
                pd.Index(chunks["sequence_ids"]).unique(), name=self.extractor_.sequence_col,
            )
            del self._preprocessing_X
            return chunks
        self.extractor_.fit(self._preprocessing_X)
        del self._preprocessing_X
        chunks = self.extractor_.transform_chunks(X_train)
        seq = self.extractor_.sequence_col
        accepted = pd.Index(chunks["sequence_ids"]).unique()
        self.finetuning_sequence_ids_ = pd.Index(accepted, name=seq)
        if self.extractor_.filter_problematic_sequences:
            self._pretraining_X = self.extractor_.problematic_filter_.transform(self._pretraining_X)
            if self._pretraining_validation_X is not None:
                self._pretraining_validation_X = self.extractor_.problematic_filter_.transform(self._pretraining_validation_X)
        if self._pretraining_X.empty:
            raise ValueError("Problematic-sequence filtering removed every pretraining example.")
        if self._pretraining_validation_X is not None and self._pretraining_validation_X.empty:
            raise ValueError("Problematic-sequence filtering removed every pretraining validation example.")
        self.pretraining_sequence_ids_ = pd.Index(self._pretraining_X[seq].drop_duplicates(), name=seq)
        if self.pretraining_augmentor is not None:
            self.pretraining_augmentor_ = clone(self.pretraining_augmentor)
            self._pretraining_X = self.pretraining_augmentor_.fit_transform(self._pretraining_X)
        else:
            self.pretraining_augmentor_ = None
        return chunks

    def _pretraining_settings(self):
        settings = {
            "learning_rate": self.learning_rate if self.pretraining_learning_rate is None else self.pretraining_learning_rate,
            "optimizer_name": self.optimizer_name if self.pretraining_optimizer_name is None else self.pretraining_optimizer_name,
            "weight_decay": self.pretraining_weight_decay,
            "momentum": self.pretraining_momentum,
            "gradient_clip_norm": self.pretraining_gradient_clip_norm,
            "gradient_clip_value": self.pretraining_gradient_clip_value,
            "loss_name": self.pretraining_loss_name,
            "label_smoothing": self.pretraining_label_smoothing,
            "use_orientation_aux_head": False, "use_phase_aux_head": False, "custom_aux_heads": None,
        }
        settings.update(mb._json_maybe(self.pretraining_model_params) or {})
        return settings

    def _pretraining_callbacks(self, has_validation):
        stage = copy.copy(self)
        for name in ("learning_rate", "early_stopping_patience", "use_lr_scheduler", "lr_scheduler_type",
                     "lr_factor", "lr_patience", "min_lr", "cosine_t_max", "warmup_epochs"):
            value = getattr(self, f"pretraining_{name}")
            if name == "learning_rate" and value is None:
                value = self.learning_rate
            setattr(stage, name, value)
        stage.learning_rate = self._pretraining_settings()["learning_rate"]
        return mb.MultiBranchSequenceClassifier._build_callbacks(stage, has_validation)

    def _supervised_pretrain(self, shapes, static, length, settings):
        chunks = self.extractor_.transform_chunks(self._pretraining_X)
        labels = _sequence_labels(
            self._pretraining_X, self._pretraining_y, chunks["sequence_ids"],
            self.extractor_.sequence_col, self.extractor_.pretraining_label_column,
        )
        self.pretraining_label_encoder_ = LabelEncoder().fit(labels)
        if len(self.pretraining_label_encoder_.classes_) < 2:
            raise ValueError("Supervised pretraining needs at least two original gesture classes.")
        source = self._build_model(shapes, static, len(self.pretraining_label_encoder_.classes_), length, **settings)
        encoded = self.pretraining_label_encoder_.transform(labels)
        if str(settings["loss_name"]).lower() == "cce":
            encoded = mb.tf.keras.utils.to_categorical(encoded, len(self.pretraining_label_encoder_.classes_))
        branch = self._split_branches(chunks["X"], chunks["mask"], self.branch_groups_)
        inputs = self._model_inputs(branch, chunks["mask"])
        del chunks["X"]
        sample_weight = None
        if self.pretraining_class_weight_mode == "balanced":
            sparse = self.pretraining_label_encoder_.transform(labels)
            values = compute_class_weight("balanced", classes=np.unique(sparse), y=sparse)
            sample_weight = np.asarray([dict(zip(np.unique(sparse), values))[v] for v in sparse], dtype=np.float32)
        elif self.pretraining_class_weight_mode is not None:
            raise ValueError("pretraining_class_weight_mode must be None or 'balanced'.")
        validation = None
        if self._pretraining_validation_X is not None:
            val = self.extractor_.transform_chunks(self._pretraining_validation_X)
            raw = _sequence_labels(
                self._pretraining_validation_X, self._pretraining_y, val["sequence_ids"],
                self.extractor_.sequence_col, self.extractor_.pretraining_label_column,
            )
            try:
                val_y = self.pretraining_label_encoder_.transform(raw)
            except ValueError as exc:
                raise ValueError("Pretraining validation contains a class absent from source training.") from exc
            if str(settings["loss_name"]).lower() == "cce":
                val_y = mb.tf.keras.utils.to_categorical(val_y, len(self.pretraining_label_encoder_.classes_))
            arrays = self._split_branches(val["X"], val["mask"], self.branch_groups_)
            validation = (self._model_inputs(arrays, val["mask"]), val_y)
            del val["X"]
        before = _weight_snapshot(source)
        history = source.fit(
            inputs, encoded, sample_weight=sample_weight, validation_data=validation,
            epochs=self.pretraining_epochs, batch_size=self.pretraining_batch_size,
            callbacks=self._pretraining_callbacks(validation is not None),
            shuffle=self.pretraining_shuffle, verbose=self._pretraining_verbose(),
        )
        self.pretraining_history_ = history.history
        self.pretraining_changed_layers_ = _changed_layers(before, source)
        return source

    def _pretraining_verbose(self):
        return self.verbose if self.pretraining_verbose is None else self.pretraining_verbose

    def _ssl_column_groups(self):
        groups = {}
        for index, column in enumerate(self.ssl_columns_):
            groups.setdefault(column.split("_", 1)[0], []).append(index)
        return groups

    def _ssl_validity(self, frame):
        values = frame[self.ssl_columns_].to_numpy(dtype=np.float32)
        valid = np.isfinite(values)
        rules = mb._json_maybe(self.ssl_invalid_values) or {}
        for index, column in enumerate(self.ssl_columns_):
            if column.startswith("tof_"):
                valid[:, index] &= values[:, index] >= 0
            invalid = rules.get(column, rules.get(column.split("_", 1)[0] + "_", []))
            invalid = [invalid] if np.isscalar(invalid) else invalid
            for value in invalid:
                valid[:, index] &= values[:, index] != value
        quaternion = ["rot_w", "rot_x", "rot_y", "rot_z"]
        if all(column in frame for column in quaternion):
            raw_q = frame[quaternion].to_numpy(dtype=float)
            quaternion_valid = np.isfinite(raw_q).all(axis=1) & (np.linalg.norm(raw_q, axis=1) > 0)
            for index, column in enumerate(self.ssl_columns_):
                if column in quaternion:
                    valid[:, index] &= quaternion_valid
        return valid

    def _prepare_ssl_targets(self):
        frame = self._pretraining_X
        if self.ssl_mask_columns is None:
            enabled = set(self.branch_groups_)
            if "derived_imu" in enabled:
                enabled.add("acc")
            wanted = {"acc": "acc", "rot": "rotation", "tof": "tof", "thm": "thermo"}
            columns = [column for column in frame if
                       re.fullmatch(r"acc_[xyz]|rot_[wxyz]|tof_\d+_v\d+|thm_\d+", column)
                       and wanted[column.split("_", 1)[0]] in enabled]
        else:
            columns = list(mb._json_maybe(self.ssl_mask_columns))
        if not columns or len(set(columns)) != len(columns) or any(column not in frame for column in columns):
            raise ValueError("ssl_mask_columns must select unique, available raw sensor columns.")
        self.ssl_columns_ = columns
        # Accumulate training-only moments in bounded blocks, rather than
        # materialising an extra float64 copy of every ToF pixel in the dataset.
        count = np.zeros(len(columns), dtype=np.int64)
        sums = np.zeros(len(columns), dtype=np.float64)
        squares = np.zeros(len(columns), dtype=np.float64)
        for start in range(0, len(frame), 8192):
            block = frame.iloc[start:start + 8192]
            values = block[columns].to_numpy(dtype=np.float64)
            valid = self._ssl_validity(block)
            safe = np.where(valid, values, 0.0)
            count += np.sum(valid, axis=0)
            sums += np.sum(safe, axis=0)
            squares += np.sum(safe ** 2, axis=0)
        if np.any(count == 0):
            raise ValueError("A selected SSL raw channel has no valid training measurements.")
        means = sums / count
        variance = np.maximum(squares / count - means ** 2, 0.0)
        self.ssl_target_mean_ = means.astype(np.float32) if self.ssl_standardize_targets else np.zeros(len(columns), dtype=np.float32)
        self.ssl_target_scale_ = np.maximum(np.sqrt(variance), 1e-6).astype(np.float32) if self.ssl_standardize_targets else np.ones(len(columns), dtype=np.float32)
        weights = mb._json_maybe(self.ssl_loss_channel_weights) or {}
        self.ssl_channel_weights_ = np.asarray([
            weights.get(column, 1.0) for column in columns
        ], dtype=np.float32)
        if not np.isfinite(self.ssl_channel_weights_).all() or np.any(self.ssl_channel_weights_ < 0) or not self.ssl_channel_weights_.sum():
            raise ValueError("SSL channel weights must be finite, nonnegative and include a positive weight.")

    def _ssl_loss(self, y_true, y_pred):
        channels = len(self.ssl_columns_)
        truth, weight = y_true[..., :channels], y_true[..., channels:]
        error = y_pred - truth
        if self.ssl_reconstruction_loss == "mse":
            element = mb.tf.square(error)
        elif self.ssl_reconstruction_loss == "mae":
            element = mb.tf.abs(error)
        else:
            absolute = mb.tf.abs(error)
            delta = float(self.ssl_huber_delta)
            quadratic = mb.tf.minimum(absolute, delta)
            element = 0.5 * mb.tf.square(quadratic) + delta * (absolute - quadratic)
        denominator = mb.tf.reduce_sum(mb.tf.cast(weight > 0, weight.dtype), axis=[0, 1])
        channel_loss = mb.tf.reduce_sum(element * weight, axis=[0, 1]) / mb.tf.maximum(denominator, 1.0)
        present = mb.tf.cast(denominator > 0, channel_loss.dtype)
        return mb.tf.reduce_sum(channel_loss) / mb.tf.maximum(mb.tf.reduce_sum(present), 1.0)

    def _masked_pretrain(self, shapes, static, length, settings):
        source = self._build_model(shapes, static, len(self.classes_), length, **settings)
        if not hasattr(source, "_temporal_representation_tensor"):
            raise ValueError("Masked SSL needs temporal_architecture='bert', 'cnn_attention_pool', 'phase_attention', or use_bigru=True.")
        self._prepare_ssl_targets()
        temporal = source._temporal_representation_tensor
        if self.ssl_decoder_units:
            temporal = mb.layers.Dense(int(self.ssl_decoder_units), activation="gelu", name="ssl_decoder_hidden")(temporal)
        temporal = mb.layers.Dropout(self.ssl_decoder_dropout, name="ssl_decoder_dropout")(temporal)
        reconstruction = mb.layers.Dense(len(self.ssl_columns_), name="ssl_reconstruction")(temporal)
        decoder = mb.models.Model(source.inputs, reconstruction, name="multibranch_masked_ssl")
        decoder.compile(optimizer=mb._build_optimizer(
            settings["optimizer_name"], settings["learning_rate"], settings["weight_decay"],
            settings["momentum"], settings["gradient_clip_norm"], settings["gradient_clip_value"],
        ), loss=self._ssl_loss)
        training = MaskedSensorSequence(self, self._pretraining_X)
        validation = (MaskedSensorSequence(self, self._pretraining_validation_X, validation=True)
                      if self._pretraining_validation_X is not None else None)
        first_inputs, first_target = training[0]
        self.ssl_first_batch_diagnostics_ = {
            "input_shapes": [array.shape for array in first_inputs],
            "target_shape": first_target.shape,
            "hidden_valid_measurements": int(np.sum(first_target[..., len(self.ssl_columns_):] > 0)),
            "raw_columns": list(self.ssl_columns_),
        }
        before = _weight_snapshot(source)
        history = decoder.fit(
            training, validation_data=validation, epochs=self.pretraining_epochs,
            callbacks=self._pretraining_callbacks(validation is not None), shuffle=False,
            verbose=self._pretraining_verbose(),
        )
        self.pretraining_history_ = history.history
        self.pretraining_changed_layers_ = _changed_layers(before, source)
        self.ssl_pretrained_pooling_ = False
        return source

    def _recompile_finetuning(self, rate):
        config = self.model_.get_compile_config()
        multi = len(self.model_.output_names) > 1
        self.model_.compile(
            optimizer=mb._build_optimizer(
                self.optimizer_name, rate, self.weight_decay, self.momentum,
                self.gradient_clip_norm, self.gradient_clip_value,
            ),
            loss=self.model_.loss,
            loss_weights=config.get("loss_weights"),
            metrics={"predictions": ["accuracy"]} if multi else ["accuracy"],
        )

    def _build_callbacks(self, has_validation_data=False):
        stage = copy.copy(self)
        if self.finetuning_learning_rate is not None:
            stage.learning_rate = self.finetuning_learning_rate
        return mb.MultiBranchSequenceClassifier._build_callbacks(stage, has_validation_data)

    def _before_supervised_fit(self, *, time_series_shapes, static_shapes, sequence_length,
                               model_inputs, train_targets, sample_weight, validation_data):
        if not self.pretraining_baseline_:
            settings = self._pretraining_settings()
            source = (self._supervised_pretrain if self.pretraining_mode == "supervised" else self._masked_pretrain)(
                time_series_shapes, static_shapes, sequence_length, settings,
            )
            layer_names = (set(self.pretraining_changed_layers_) if self.pretraining_mode == "masked_ssl" else None)
            self.transferred_layer_names_, self.replaced_head_names_ = transfer_backbone_weights(
                source, self.model_, layer_names=layer_names,
            )
            self.transferred_changed_layer_names_ = sorted(set(self.pretraining_changed_layers_) & set(self.transferred_layer_names_))
            if not self.transferred_changed_layer_names_:
                raise ValueError("Pretraining did not change any transferred backbone weights.")
            self.pretraining_completed_ = True
            # Release source arrays/models before the target stage. Only diagnostic
            # IDs and histories survive; the complete source dataset is not pickled.
            del self._pretraining_X, self._pretraining_validation_X, self._pretraining_y
        if self.finetuning_head_epochs:
            original = {layer.name: layer.trainable for layer in self.model_.layers}
            for layer in self.model_.layers:
                layer.trainable = layer.name in self.model_.output_names or layer.name.startswith("head_")
            rate = self.learning_rate if self.finetuning_head_learning_rate is None else self.finetuning_head_learning_rate
            self._recompile_finetuning(float(rate))
            history = self.model_.fit(
                model_inputs, train_targets, sample_weight=sample_weight,
                validation_data=validation_data, epochs=int(self.finetuning_head_epochs),
                batch_size=self.batch_size, verbose=self.verbose,
            )
            self.finetuning_head_history_ = history.history
            for layer in self.model_.layers:
                layer.trainable = original[layer.name]
            rate = self.learning_rate if self.finetuning_learning_rate is None else self.finetuning_learning_rate
            self._recompile_finetuning(float(rate))
        else:
            self.finetuning_head_history_ = {}
        rate = self.learning_rate if self.finetuning_learning_rate is None else self.finetuning_learning_rate
        self.model_.optimizer.learning_rate.assign(float(rate))

    def score(self, X, y, sample_weight=None):
        if getattr(self, "fit_failed_", False):
            return float(self.invalid_candidate_score)
        fine = self.extractor_.select_finetuning_sequences(X, y)
        if fine.empty:
            raise ValueError("No evaluation examples satisfy the fine-tuning selectors.")
        return super().score(fine, y, sample_weight)


DEFAULT_PRETRAINING_PARAMS = {
    name: param.default for name, param in inspect.signature(AdvancedMultiBranchSequenceClassifier.__init__).parameters.items()
    if name not in {"self", "model_params"}
}


def build_pretraining_model(*args, **kwargs):
    """Public alias retaining the complete multibranch builder's architecture API."""
    return mb.build_multibranch_model(*args, **kwargs)


prepare_pretraining_bayesian_space = mb.prepare_multibranch_bayesian_space
prepare_pretraining_evolutionary_space = mb.prepare_multibranch_evolutionary_space


def pretraining_score(estimator, X, y):
    """Search scorer that evaluates only the configured fine-tuning domain."""
    return estimator.score(X, y)
