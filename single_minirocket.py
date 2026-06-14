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
    """
    Single end-to-end MiniRocket classifier.
    Allows flexible target switching (bfrb, orientation, gesture, etc.)
    """
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
        # Align y with sequence-level X output by SequenceExtractor
        if isinstance(y, pd.DataFrame) and "sequence_id" in y.columns:
            y = y.drop_duplicates("sequence_id").sort_values("sequence_id").reset_index(drop=True)
        
        y_target = y[self.target_col].values
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y_target)
        
        # MiniRocket expects (n_instances, n_channels, n_timepoints)
        if X.ndim == 3:
            X_sk = X.transpose(0, 2, 1)
        else:
            X_sk = X
            
        # Handle padding (SequenceExtractor defaults to -999.0 or 0.0)
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
        if X.ndim == 3:
            X_sk = X.transpose(0, 2, 1)
        else:
            X_sk = X
            
        X_sk = np.where(X_sk == -999.0, 0.0, X_sk)
        X_sk = np.nan_to_num(X_sk, nan=0.0, posinf=0.0, neginf=0.0)
        
        X_feat = self.rocket_.transform(X_sk)
        if self.selector_ is not None:
            X_feat = self.selector_.transform(X_feat)
            
        preds_enc = self.clf_.predict(X_feat)
        return self.le_.inverse_transform(preds_enc)