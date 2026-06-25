import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectPercentile, f_classif

try:
    from sktime.transformations.panel.rocket import MiniRocket
except ImportError:
    from sktime.classification.feature_based import MiniRocket

class SingleMiniRocketClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        target_col="bfrb",
        num_kernels=1000,
        alpha=1.0,
        feature_selection_percentile=None,
        class_weight="balanced",
        random_state=42
    ):
        self.target_col = target_col
        self.num_kernels = num_kernels
        self.alpha = alpha
        self.feature_selection_percentile = feature_selection_percentile
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self, X, y):
        # X is a dict from SequenceExtractor
        if not isinstance(X, dict) or 'X' not in X or 'sequence_ids' not in X:
            raise ValueError("Expected dict with keys 'X' and 'sequence_ids' from SequenceExtractor")

        seq_ids = X['sequence_ids']
        X_arr = X['X']

        # Align y using sequence_ids
        if isinstance(y, pd.DataFrame):
            y_map = y.drop_duplicates('sequence_id').set_index('sequence_id')[self.target_col]
            y_target = np.array([y_map.get(sid, None) for sid in seq_ids])
        else:
            y_target = np.asarray(y)

        # Remove any samples with missing labels
        valid_mask = pd.notna(y_target)
        if not valid_mask.all():
            X_arr = X_arr[valid_mask]
            y_target = y_target[valid_mask]

        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y_target)

        # MiniRocket expects (n_samples, n_channels, n_timepoints)
        X_sk = X_arr.transpose(0, 2, 1)

        # Replace padding value (-999.0) with 0.0
        X_sk = np.where(X_sk == -999.0, 0.0, X_sk)
        X_sk = np.nan_to_num(X_sk, nan=0.0, posinf=0.0, neginf=0.0)

        self.rocket_ = MiniRocket(num_kernels=self.num_kernels, random_state=self.random_state)
        X_feat = self.rocket_.fit_transform(X_sk)

        if self.feature_selection_percentile is not None and self.feature_selection_percentile < 100:
            self.selector_ = SelectPercentile(score_func=f_classif, percentile=self.feature_selection_percentile)
            X_feat = self.selector_.fit_transform(X_feat, y_enc)
        else:
            self.selector_ = None

        self.clf_ = RidgeClassifier(alpha=self.alpha, class_weight=self.class_weight)
        self.clf_.fit(X_feat, y_enc)
        return self

    def predict(self, X):
        if not isinstance(X, dict) or 'X' not in X:
            raise ValueError("Expected dict with key 'X' from SequenceExtractor")

        X_arr = X['X']
        X_sk = X_arr.transpose(0, 2, 1)

        X_sk = np.where(X_sk == -999.0, 0.0, X_sk)
        X_sk = np.nan_to_num(X_sk, nan=0.0, posinf=0.0, neginf=0.0)

        X_feat = self.rocket_.transform(X_sk)
        if self.selector_ is not None:
            X_feat = self.selector_.transform(X_feat)

        preds_enc = self.clf_.predict(X_feat)
        return self.le_.inverse_transform(preds_enc)