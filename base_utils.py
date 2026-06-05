
"""
base_utils.py
Sequence extraction utilities for prototypical_v2 notebook.
No nested functions. sklearn-compatible transformer.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SequenceExtractor(BaseEstimator, TransformerMixin):
    """
    Input
    -----
    X : pd.DataFrame
        Row-level sensor dataframe containing sequence_id and sensor columns.

    Output
    ------
    np.ndarray of shape (n_sequences, maxlen, n_features)
    """

    def __init__(
        self,
        sequence_column="sequence_id",
        counter_column="sequence_counter",
        acc_mode="raw",
        rotation_mode="quaternion",
        tof_mode="sensor_stats",
        thm_mode="centered_diff",
        sampling_rate=20,
        compute_dt=True,
        window_size=7,
        maxlen=160,
        **kwargs,
    ):
        self.sequence_column = sequence_column
        self.counter_column = counter_column
        self.acc_mode = acc_mode
        self.rotation_mode = rotation_mode
        self.tof_mode = tof_mode
        self.thm_mode = thm_mode
        self.sampling_rate = sampling_rate
        self.compute_dt = compute_dt
        self.window_size = window_size
        self.maxlen = maxlen

    def fit(self, X, y=None):
        return self

    def _feature_columns(self, df):
        exclude = {
            self.sequence_column,
            self.counter_column,
            "gesture",
            "subject",
            "sequence_type",
        }
        cols = []
        for c in df.columns:
            if c in exclude:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)
        return cols

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        feature_cols = self._feature_columns(X)
        sequences = []

        for _, g in X.groupby(self.sequence_column):
            arr = g[feature_cols].fillna(0).to_numpy(dtype=np.float32)

            if len(arr) >= self.maxlen:
                arr = arr[: self.maxlen]
            else:
                pad = np.zeros((self.maxlen - len(arr), arr.shape[1]), dtype=np.float32)
                arr = np.vstack([arr, pad])

            sequences.append(arr)

        return np.stack(sequences, axis=0)
