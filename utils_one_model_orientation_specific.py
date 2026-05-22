from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

import keras
from tensorflow.keras import layers
import tensorflow as tf

from utils_hierarchy_position_aug import *  # reuse corrector, extractors, CNN, multibranch classes


class KerasTCNSequenceClassifier(KerasAugmentedCNN1DSequenceClassifier):
    """Sklearn-compatible temporal convolutional network classifier.

    Reuses the same padding, y collapsing, augmentation and scoring behaviour as
    KerasAugmentedCNN1DSequenceClassifier, but swaps the backbone to dilated
    residual Conv1D blocks.
    """

    def __init__(
        self,
        target: str = "gesture_action",
        maxlen: int = 128,
        padding_value: float = -999.0,
        tcn_filters: int = 128,
        kernel_size: int = 5,
        dilations: str = "1-2-4-8",
        tcn_blocks: int = 1,
        use_batch_norm: bool = True,
        spatial_dropout: float = 0.1,
        dense_units: str = "64",
        dropout: float = 0.2,
        learning_rate: float = 5e-5,
        batch_size: int = 32,
        epochs: int = 80,
        patience: int = 10,
        validation_fraction: float = 0.15,
        verbose: int = 0,
        random_state: int = 42,
        use_mixup: bool = False,
        mixup_alpha: float = 0.4,
        mixup_size: float = 1.0,
        mixup_prob: float = 1.0,
        use_time_shift: bool = False,
        max_shift_pct: float = 0.25,
        use_time_stretch: bool = False,
        time_stretch_min_rate: float = 0.5,
        time_stretch_max_rate: float = 1.5,
        use_gaussian_noise: bool = False,
        noise_std: float = 0.01,
        use_magnitude_scaling: bool = False,
        magnitude_scale_min: float = 0.9,
        magnitude_scale_max: float = 1.1,
        use_time_mask: bool = False,
        time_mask_ratio: float = 0.1,
        use_channel_dropout: bool = False,
        channel_dropout_prob: float = 0.0,
        use_tof_dropout: bool = False,
        tof_dropout_prob: float = 0.0,
        use_modality_dropout: bool = False,
        drop_acc_prob: float = 0.0,
        drop_rot_prob: float = 0.0,
        drop_tof_prob: float = 0.0,
        drop_thm_prob: float = 0.0,
        use_quaternion_sign_flip: bool = False,
        quaternion_flip_prob: float = 0.5,
        use_cutmix: bool = False,
        cutmix_prob: float = 0.0,
        cutmix_size: float = 0.5,
        use_frequency_filter: bool = False,
        freq_keep_low: float = 0.1,
        freq_keep_high: float = 0.9,
    ) -> None:
        super().__init__(
            target=target,
            maxlen=maxlen,
            padding_value=padding_value,
            conv_filters=str(tcn_filters),
            kernel_sizes=str(kernel_size),
            pool_sizes="none",
            use_batch_norm=use_batch_norm,
            spatial_dropout=spatial_dropout,
            dense_units=dense_units,
            dropout=dropout,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            validation_fraction=validation_fraction,
            verbose=verbose,
            random_state=random_state,
            use_mixup=use_mixup,
            mixup_alpha=mixup_alpha,
            mixup_size=mixup_size,
            mixup_prob=mixup_prob,
            use_time_shift=use_time_shift,
            max_shift_pct=max_shift_pct,
            use_time_stretch=use_time_stretch,
            time_stretch_min_rate=time_stretch_min_rate,
            time_stretch_max_rate=time_stretch_max_rate,
            use_gaussian_noise=use_gaussian_noise,
            noise_std=noise_std,
            use_magnitude_scaling=use_magnitude_scaling,
            magnitude_scale_min=magnitude_scale_min,
            magnitude_scale_max=magnitude_scale_max,
            use_time_mask=use_time_mask,
            time_mask_ratio=time_mask_ratio,
            use_channel_dropout=use_channel_dropout,
            channel_dropout_prob=channel_dropout_prob,
            use_tof_dropout=use_tof_dropout,
            tof_dropout_prob=tof_dropout_prob,
            use_modality_dropout=use_modality_dropout,
            drop_acc_prob=drop_acc_prob,
            drop_rot_prob=drop_rot_prob,
            drop_tof_prob=drop_tof_prob,
            drop_thm_prob=drop_thm_prob,
            use_quaternion_sign_flip=use_quaternion_sign_flip,
            quaternion_flip_prob=quaternion_flip_prob,
            use_cutmix=use_cutmix,
            cutmix_prob=cutmix_prob,
            cutmix_size=cutmix_size,
            use_frequency_filter=use_frequency_filter,
            freq_keep_low=freq_keep_low,
            freq_keep_high=freq_keep_high,
        )
        self.tcn_filters = tcn_filters
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.tcn_blocks = tcn_blocks

    def _parse_dilations(self) -> tuple[int, ...]:
        if isinstance(self.dilations, str):
            return tuple(int(x) for x in self.dilations.split("-") if str(x).strip())
        return tuple(int(x) for x in self.dilations)

    def _build(self, shape: tuple[int, int], n_classes: int) -> keras.Model:
        tf.keras.backend.clear_session()
        keras.utils.set_random_seed(int(self.random_state))

        inp = keras.Input(shape=shape)
        x = inp
        filters = int(self.tcn_filters)
        kernel_size = int(self.kernel_size)
        dilations = self._parse_dilations()

        if shape[-1] != filters:
            residual = layers.Conv1D(filters, 1, padding="same")(x)
        else:
            residual = x

        x = residual
        for _ in range(int(self.tcn_blocks)):
            for dilation in dilations:
                shortcut = x
                y = layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=int(dilation), activation="relu")(x)
                if self.use_batch_norm:
                    y = layers.BatchNormalization()(y)
                if float(self.spatial_dropout) > 0:
                    y = layers.SpatialDropout1D(float(self.spatial_dropout))(y)
                y = layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=int(dilation), activation="relu")(y)
                if self.use_batch_norm:
                    y = layers.BatchNormalization()(y)
                if shortcut.shape[-1] != y.shape[-1]:
                    shortcut = layers.Conv1D(filters, 1, padding="same")(shortcut)
                x = layers.Add()([shortcut, y])
                x = layers.Activation("relu")(x)

        x = layers.GlobalAveragePooling1D()(x)
        for units in self._to_tuple(self.dense_units):
            x = layers.Dense(int(units), activation="relu")(x)
            if float(self.dropout) > 0:
                x = layers.Dropout(float(self.dropout))(x)
        out = layers.Dense(n_classes, activation="softmax")(x)
        model = keras.Model(inp, out)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=float(self.learning_rate)),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model


class KerasInceptionTimeSequenceClassifier(KerasAugmentedCNN1DSequenceClassifier):
    """Small InceptionTime-style temporal classifier."""

    def __init__(
        self,
        target: str = "gesture_action",
        maxlen: int = 128,
        padding_value: float = -999.0,
        n_filters: int = 64,
        kernel_sizes: str = "9-19-39",
        inception_blocks: int = 2,
        bottleneck_size: int = 32,
        use_batch_norm: bool = True,
        spatial_dropout: float = 0.1,
        dense_units: str = "64",
        dropout: float = 0.2,
        learning_rate: float = 5e-5,
        batch_size: int = 32,
        epochs: int = 80,
        patience: int = 10,
        validation_fraction: float = 0.15,
        verbose: int = 0,
        random_state: int = 42,
        use_mixup: bool = False,
        mixup_alpha: float = 0.4,
        mixup_size: float = 1.0,
        mixup_prob: float = 1.0,
        use_time_shift: bool = False,
        max_shift_pct: float = 0.25,
        use_time_stretch: bool = False,
        time_stretch_min_rate: float = 0.5,
        time_stretch_max_rate: float = 1.5,
        use_gaussian_noise: bool = False,
        noise_std: float = 0.01,
        use_magnitude_scaling: bool = False,
        magnitude_scale_min: float = 0.9,
        magnitude_scale_max: float = 1.1,
        use_time_mask: bool = False,
        time_mask_ratio: float = 0.1,
        use_channel_dropout: bool = False,
        channel_dropout_prob: float = 0.0,
        use_tof_dropout: bool = False,
        tof_dropout_prob: float = 0.0,
        use_modality_dropout: bool = False,
        drop_acc_prob: float = 0.0,
        drop_rot_prob: float = 0.0,
        drop_tof_prob: float = 0.0,
        drop_thm_prob: float = 0.0,
        use_quaternion_sign_flip: bool = False,
        quaternion_flip_prob: float = 0.5,
        use_cutmix: bool = False,
        cutmix_prob: float = 0.0,
        cutmix_size: float = 0.5,
        use_frequency_filter: bool = False,
        freq_keep_low: float = 0.1,
        freq_keep_high: float = 0.9,
    ) -> None:
        super().__init__(
            target=target,
            maxlen=maxlen,
            padding_value=padding_value,
            conv_filters=str(n_filters),
            kernel_sizes="5",
            pool_sizes="none",
            use_batch_norm=use_batch_norm,
            spatial_dropout=spatial_dropout,
            dense_units=dense_units,
            dropout=dropout,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            validation_fraction=validation_fraction,
            verbose=verbose,
            random_state=random_state,
            use_mixup=use_mixup,
            mixup_alpha=mixup_alpha,
            mixup_size=mixup_size,
            mixup_prob=mixup_prob,
            use_time_shift=use_time_shift,
            max_shift_pct=max_shift_pct,
            use_time_stretch=use_time_stretch,
            time_stretch_min_rate=time_stretch_min_rate,
            time_stretch_max_rate=time_stretch_max_rate,
            use_gaussian_noise=use_gaussian_noise,
            noise_std=noise_std,
            use_magnitude_scaling=use_magnitude_scaling,
            magnitude_scale_min=magnitude_scale_min,
            magnitude_scale_max=magnitude_scale_max,
            use_time_mask=use_time_mask,
            time_mask_ratio=time_mask_ratio,
            use_channel_dropout=use_channel_dropout,
            channel_dropout_prob=channel_dropout_prob,
            use_tof_dropout=use_tof_dropout,
            tof_dropout_prob=tof_dropout_prob,
            use_modality_dropout=use_modality_dropout,
            drop_acc_prob=drop_acc_prob,
            drop_rot_prob=drop_rot_prob,
            drop_tof_prob=drop_tof_prob,
            drop_thm_prob=drop_thm_prob,
            use_quaternion_sign_flip=use_quaternion_sign_flip,
            quaternion_flip_prob=quaternion_flip_prob,
            use_cutmix=use_cutmix,
            cutmix_prob=cutmix_prob,
            cutmix_size=cutmix_size,
            use_frequency_filter=use_frequency_filter,
            freq_keep_low=freq_keep_low,
            freq_keep_high=freq_keep_high,
        )
        self.n_filters = n_filters
        self.kernel_sizes = kernel_sizes
        self.inception_blocks = inception_blocks
        self.bottleneck_size = bottleneck_size

    def _parse_kernels(self) -> tuple[int, ...]:
        if isinstance(self.kernel_sizes, str):
            return tuple(int(x) for x in self.kernel_sizes.split("-") if str(x).strip())
        return tuple(int(x) for x in self.kernel_sizes)

    def _inception_block(self, x):
        shortcut = x
        bottleneck = layers.Conv1D(int(self.bottleneck_size), 1, padding="same", activation="relu")(x)
        branches = []
        for k in self._parse_kernels():
            branches.append(layers.Conv1D(int(self.n_filters), int(k), padding="same", activation="relu")(bottleneck))
        pool = layers.MaxPooling1D(pool_size=3, strides=1, padding="same")(x)
        pool = layers.Conv1D(int(self.n_filters), 1, padding="same", activation="relu")(pool)
        branches.append(pool)
        y = layers.Concatenate()(branches)
        if self.use_batch_norm:
            y = layers.BatchNormalization()(y)
        if float(self.spatial_dropout) > 0:
            y = layers.SpatialDropout1D(float(self.spatial_dropout))(y)
        if shortcut.shape[-1] != y.shape[-1]:
            shortcut = layers.Conv1D(int(y.shape[-1]), 1, padding="same")(shortcut)
        return layers.Activation("relu")(layers.Add()([shortcut, y]))

    def _build(self, shape: tuple[int, int], n_classes: int) -> keras.Model:
        tf.keras.backend.clear_session()
        keras.utils.set_random_seed(int(self.random_state))
        inp = keras.Input(shape=shape)
        x = inp
        for _ in range(int(self.inception_blocks)):
            x = self._inception_block(x)
        x = layers.GlobalAveragePooling1D()(x)
        for units in self._to_tuple(self.dense_units):
            x = layers.Dense(int(units), activation="relu")(x)
            if float(self.dropout) > 0:
                x = layers.Dropout(float(self.dropout))(x)
        out = layers.Dense(n_classes, activation="softmax")(x)
        model = keras.Model(inp, out)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=float(self.learning_rate)),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model


class MiniRocketSequenceClassifier(ClassifierMixin, BaseEstimator):
    """MiniRocket classifier for multivariate sequence data.

    Requires sktime. It consumes the same sequence-indexed dataframe produced by
    AdvancedMultiDomainSequenceExtractor.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        target: str = "gesture_action",
        maxlen: int = 160,
        padding_value: float = 0.0,
        num_kernels: int = 10000,
        alphas: tuple[float, ...] = (0.1, 1.0, 10.0),
        random_state: int = 42,
    ) -> None:
        self.target = target
        self.maxlen = maxlen
        self.padding_value = padding_value
        self.num_kernels = num_kernels
        self.alphas = alphas
        self.random_state = random_state

    def _pad(self, X: pd.DataFrame) -> tuple[np.ndarray, pd.Series]:
        grouped = list(X.groupby(level=0, sort=False))
        n_seq = len(grouped)
        n_feat = X.shape[1]
        out = np.full((n_seq, n_feat, int(self.maxlen)), float(self.padding_value), dtype=np.float32)
        seq_ids = []
        for i, (sid, g) in enumerate(grouped):
            arr = g.to_numpy(dtype=np.float32)
            length = min(len(arr), int(self.maxlen))
            out[i, :, :length] = arr[:length].T
            seq_ids.append(sid)
        return out, pd.Series(seq_ids, name="sequence_id")

    def _collapse_y(self, seq_ids: pd.Series, y: Any) -> pd.Series:
        if isinstance(y, pd.DataFrame):
            if "sequence_id" not in y.columns:
                raise ValueError("y dataframe must contain sequence_id.")
            if self.target not in y.columns:
                raise ValueError(f"y dataframe must contain target column: {self.target}")
            target_map = y.drop_duplicates("sequence_id").set_index("sequence_id")[self.target]
            y_seq = seq_ids.map(target_map)
        else:
            y_seq = pd.Series(y).reset_index(drop=True)
            if len(y_seq) != len(seq_ids):
                raise ValueError("If y is not a dataframe, it must already be one label per sequence.")
        if y_seq.isna().any():
            missing = seq_ids[y_seq.isna()].head(10).tolist()
            raise ValueError(f"Missing labels for sequence_ids: {missing}")
        return y_seq.reset_index(drop=True)

    def fit(self, X: pd.DataFrame, y: Any) -> "MiniRocketSequenceClassifier":
        try:
            from sktime.transformations.panel.rocket import MiniRocketMultivariate
        except Exception as exc:
            raise ImportError("MiniRocketSequenceClassifier requires sktime. On Kaggle, install/add sktime before using model_type='minirocket'.") from exc

        X_arr, seq_ids = self._pad(X)
        y_seq = self._collapse_y(seq_ids, y)
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y_seq)
        self.classes_ = self.le_.classes_
        self.transformer_ = MiniRocketMultivariate(num_kernels=int(self.num_kernels), random_state=int(self.random_state))
        X_feat = self.transformer_.fit_transform(X_arr)
        self.classifier_ = RidgeClassifierCV(alphas=self.alphas)
        self.classifier_.fit(X_feat, y_enc)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, ["transformer_", "classifier_", "le_"])
        X_arr, _ = self._pad(X)
        X_feat = self.transformer_.transform(X_arr)
        pred_enc = self.classifier_.predict(X_feat)
        return self.le_.inverse_transform(pred_enc)

    def score(self, X: pd.DataFrame, y: Any) -> float:
        X_arr, seq_ids = self._pad(X)
        y_seq = self._collapse_y(seq_ids, y)
        X_feat = self.transformer_.transform(X_arr)
        pred_enc = self.classifier_.predict(X_feat)
        preds = self.le_.inverse_transform(pred_enc)
        return f1_score(y_seq.to_numpy(), preds, average="macro")
