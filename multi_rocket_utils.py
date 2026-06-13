"""
multi_rocket_utils.py
Hierarchical and Single Multi-Rocket Classifiers for Sensor Data.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV, LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.decomposition import PCA
from typing import Optional, Dict, Any, List, Type
import warnings
warnings.filterwarnings("ignore")

try:
    from sktime.transformations.panel.rocket import MultiRocket
except ImportError:
    try:
        from sktime.classification.feature_based import MultiRocket
    except ImportError:
        raise ImportError("Please install sktime: pip install sktime")


class RocketClassifierLayer(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        num_kernels: int = 1680,
        feature_selection_percentile: Optional[int] = 50,
        base_classifier_class: Type = RidgeClassifier,
        base_classifier_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        padding_value: float = 0.0,
        max_channels: int = 20
    ):
        self.num_kernels = num_kernels
        self.feature_selection_percentile = feature_selection_percentile
        self.base_classifier_class = base_classifier_class
        self.base_classifier_params = base_classifier_params
        self.random_state = random_state
        self.padding_value = padding_value
        self.max_channels = max_channels

    def fit(self, X: np.ndarray, y: np.ndarray):
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values
            
        if X.ndim == 3:
            X_sk = X.transpose(0, 2, 1)
        else:
            X_sk = X
            
        X_sk = np.where(X_sk == self.padding_value, 0.0, X_sk)
        X_sk = np.nan_to_num(X_sk, nan=0.0, posinf=0.0, neginf=0.0)
        
        n_instances, n_channels, n_timepoints = X_sk.shape
        
        # PCA Safeguard to prevent OOM crashes on high-dim sensor data
        if n_channels > self.max_channels:
            X_reshaped = X_sk.transpose(0, 2, 1).reshape(-1, n_channels)
            self.pca_ = PCA(n_components=self.max_channels, random_state=self.random_state)
            X_pca = self.pca_.fit_transform(X_reshaped)
            X_sk = X_pca.reshape(n_instances, n_timepoints, self.max_channels).transpose(0, 2, 1)
        else:
            self.pca_ = None

        self.rocket_ = MultiRocket(num_kernels=self.num_kernels, random_state=self.random_state)
        X_feat = self.rocket_.fit_transform(X_sk)
        X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.feature_selection_percentile is not None and self.feature_selection_percentile < 100:
            self.selector_ = SelectPercentile(score_func=f_classif, percentile=self.feature_selection_percentile)
            X_feat = self.selector_.fit_transform(X_feat, y)
        else:
            self.selector_ = None
            
        clf_params = self.base_classifier_params.copy() if self.base_classifier_params else {}
        
        if self.base_classifier_class in [RidgeClassifier, RidgeClassifierCV]:
            if 'alpha' not in clf_params: clf_params['alpha'] = 100.0
            if 'class_weight' not in clf_params: clf_params['class_weight'] = 'balanced'
        elif self.base_classifier_class == LogisticRegression:
            if 'C' not in clf_params: clf_params['C'] = 1.0
            if 'class_weight' not in clf_params: clf_params['class_weight'] = 'balanced'
            if 'max_iter' not in clf_params: clf_params['max_iter'] = 1000
                
        self.clf_ = self.base_classifier_class(**clf_params)
        self.clf_.fit(X_feat, y)
        return self

    def predict(self, X: np.ndarray):
        if X.ndim == 3:
            X_sk = X.transpose(0, 2, 1)
        else:
            X_sk = X
            
        X_sk = np.where(X_sk == self.padding_value, 0.0, X_sk)
        X_sk = np.nan_to_num(X_sk, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.pca_ is not None:
            n_instances, n_channels, n_timepoints = X_sk.shape
            X_reshaped = X_sk.transpose(0, 2, 1).reshape(-1, n_channels)
            X_pca = self.pca_.transform(X_reshaped)
            X_sk = X_pca.reshape(n_instances, n_timepoints, self.max_channels).transpose(0, 2, 1)
            
        X_feat = self.rocket_.transform(X_sk)
        X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.selector_ is not None:
            X_feat = self.selector_.transform(X_feat)
            
        return self.clf_.predict(X_feat)


class HierarchicalMultiRocketEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        orientation_col: str = "orientation",
        target_col: str = "bfrb",
        base_classifier_class: Type = RidgeClassifier,
        base_classifier_params: Optional[Dict[str, Any]] = None,
        
        # Layer 1
        l1_num_kernels: int = 1680, l1_alpha: float = 100.0, l1_feature_selection_percentile: Optional[int] = 50, l1_max_channels: int = 20,
        # Layer 2
        l2_num_kernels: int = 1680, l2_alpha: float = 100.0, l2_feature_selection_percentile: Optional[int] = 50, l2_max_channels: int = 20,
        # Layer 3
        l3_1_num_kernels: int = 1680, l3_1_alpha: float = 100.0, l3_1_feature_selection_percentile: Optional[int] = 50, l3_1_max_channels: int = 20,
        l3_2_num_kernels: int = 1680, l3_2_alpha: float = 100.0, l3_2_feature_selection_percentile: Optional[int] = 50, l3_2_max_channels: int = 20,
        l3_3_num_kernels: int = 1680, l3_3_alpha: float = 100.0, l3_3_feature_selection_percentile: Optional[int] = 50, l3_3_max_channels: int = 20,
        l3_4_num_kernels: int = 1680, l3_4_alpha: float = 100.0, l3_4_feature_selection_percentile: Optional[int] = 50, l3_4_max_channels: int = 20,
        
        random_state: int = 42,
        padding_value: float = 0.0,
        ensemble_models: Optional[Dict[str, Any]] = None
    ):
        self.orientation_col = orientation_col
        self.target_col = target_col
        self.base_classifier_class = base_classifier_class
        self.base_classifier_params = base_classifier_params
        
        self.l1_num_kernels, self.l1_alpha, self.l1_feature_selection_percentile, self.l1_max_channels = l1_num_kernels, l1_alpha, l1_feature_selection_percentile, l1_max_channels
        self.l2_num_kernels, self.l2_alpha, self.l2_feature_selection_percentile, self.l2_max_channels = l2_num_kernels, l2_alpha, l2_feature_selection_percentile, l2_max_channels
        self.l3_1_num_kernels, self.l1_alpha, self.l3_1_feature_selection_percentile, self.l3_1_max_channels = l3_1_num_kernels, l3_1_alpha, l3_1_feature_selection_percentile, l3_1_max_channels
        self.l3_2_num_kernels, self.l3_2_alpha, self.l3_2_feature_selection_percentile, self.l3_2_max_channels = l3_2_num_kernels, l3_2_alpha, l3_2_feature_selection_percentile, l3_2_max_channels
        self.l3_3_num_kernels, self.l3_3_alpha, self.l3_3_feature_selection_percentile, self.l3_3_max_channels = l3_3_num_kernels, l3_3_alpha, l3_3_feature_selection_percentile, l3_3_max_channels
        self.l3_4_num_kernels, self.l3_4_alpha, self.l3_4_feature_selection_percentile, self.l3_4_max_channels = l3_4_num_kernels, l3_4_alpha, l3_4_feature_selection_percentile, l3_4_max_channels
        
        self.random_state = random_state
        self.padding_value = padding_value
        self.ensemble_models = ensemble_models  # Strict assignment for sklearn clone()

    def _build_layer(self, num_kernels, alpha, fs_percentile, max_channels):
        clf_params = {}
        if self.base_classifier_class in [RidgeClassifier, RidgeClassifierCV]:
            clf_params['alpha'] = alpha
            clf_params['class_weight'] = 'balanced'
        elif self.base_classifier_class == LogisticRegression:
            clf_params['C'] = 1.0 / alpha if alpha > 0 else 1.0
            clf_params['class_weight'] = 'balanced'
            clf_params['max_iter'] = 1000
            
        if self.base_classifier_params:
            clf_params.update(self.base_classifier_params)
            
        return RocketClassifierLayer(
            num_kernels=num_kernels,
            feature_selection_percentile=fs_percentile,
            base_classifier_class=self.base_classifier_class,
            base_classifier_params=clf_params,
            random_state=self.random_state,
            padding_value=self.padding_value,
            max_channels=max_channels
        )

    def fit(self, X: np.ndarray, y: pd.DataFrame):
        # Align y with sequence-level X output by SequenceExtractor
        if isinstance(y, pd.DataFrame) and "sequence_id" in y.columns:
            y = y.drop_duplicates("sequence_id").sort_values("sequence_id").reset_index(drop=True)

        _models = self.ensemble_models if self.ensemble_models is not None else {}

        self.le_l1_ = LabelEncoder()
        y_l1 = self.le_l1_.fit_transform(y['is_target'].values)
        
        if 'l1' in _models: self.model_l1_ = _models['l1']
        else:
            self.model_l1_ = self._build_layer(self.l1_num_kernels, self.l1_alpha, self.l1_feature_selection_percentile, self.l1_max_channels)
            self.model_l1_.fit(X, y_l1)
            
        pred_l1 = self.model_l1_.predict(X)
        is_target_mask = self.le_l1_.inverse_transform(pred_l1) == True
        X_targ = X[is_target_mask]
        y_targ = y[is_target_mask]
        
        if len(X_targ) == 0: return self

        self.le_l2_ = LabelEncoder()
        y_l2 = self.le_l2_.fit_transform(y_targ[self.orientation_col].values)
        
        if 'l2' in _models: self.model_l2_ = _models['l2']
        else:
            self.model_l2_ = self._build_layer(self.l2_num_kernels, self.l2_alpha, self.l2_feature_selection_percentile, self.l2_max_channels)
            self.model_l2_.fit(X_targ, y_l2)
            
        pred_l2 = self.model_l2_.predict(X_targ)
        pred_orient = self.le_l2_.inverse_transform(pred_l2)
        
        self.le_l3_encoders_ = {}
        self.models_l3_ = {}
        
        orientations = self.le_l2_.classes_
        for i, orient in enumerate(orientations):
            mask = (pred_orient == orient)
            if mask.sum() > 0:
                le_l3 = LabelEncoder()
                y_l3 = le_l3.fit_transform(y_targ.loc[mask, self.target_col].values)
                self.le_l3_encoders_[orient] = le_l3
                
                layer_name = f"l3_{i+1}"
                if layer_name in _models:
                    model = _models[layer_name]
                else:
                    num_k = getattr(self, f"{layer_name}_num_kernels")
                    alp = getattr(self, f"{layer_name}_alpha")
                    fs = getattr(self, f"{layer_name}_feature_selection_percentile")
                    mc = getattr(self, f"{layer_name}_max_channels")
                    model = self._build_layer(num_k, alp, fs, mc)
                    model.fit(X_targ[mask], y_l3)
                self.models_l3_[orient] = model
                
        return self

    def predict(self, X: np.ndarray):
        preds_final = np.empty(len(X), dtype=object)
        preds_final[:] = 'non_bfrb'
        
        pred_l1 = self.model_l1_.predict(X)
        is_target_mask = self.le_l1_.inverse_transform(pred_l1) == True
        
        if is_target_mask.sum() > 0:
            X_targ = X[is_target_mask]
            idx_targ = np.where(is_target_mask)[0]
            
            pred_l2 = self.model_l2_.predict(X_targ)
            pred_orient = self.le_l2_.inverse_transform(pred_l2)
            
            for orient, model in self.models_l3_.items():
                mask = (pred_orient == orient)
                if mask.sum() > 0:
                    p_l3 = model.predict(X_targ[mask])
                    preds_final[idx_targ[mask]] = self.le_l3_encoders_[orient].inverse_transform(p_l3)
                    
        return preds_final


class FlexibleMultiRocketClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        target_col: str = "bfrb",
        orientation_col: str = "orientation",
        slice_by_orientation: Optional[List[str]] = None,
        slice_by_bfrb: Optional[bool] = None,
        base_classifier_class: Type = RidgeClassifier,
        base_classifier_params: Optional[Dict[str, Any]] = None,
        num_kernels: int = 1680,
        alpha: float = 100.0,
        feature_selection_percentile: Optional[int] = 50,
        random_state: int = 42,
        padding_value: float = 0.0,
        max_channels: int = 20
    ):
        self.target_col = target_col
        self.orientation_col = orientation_col
        self.slice_by_orientation = slice_by_orientation
        self.slice_by_bfrb = slice_by_bfrb
        self.base_classifier_class = base_classifier_class
        self.base_classifier_params = base_classifier_params
        self.num_kernels = num_kernels
        self.alpha = alpha
        self.feature_selection_percentile = feature_selection_percentile
        self.random_state = random_state
        self.padding_value = padding_value
        self.max_channels = max_channels
        
    def _slice_data(self, X: np.ndarray, y: pd.DataFrame):
        if isinstance(y, pd.DataFrame) and "sequence_id" in y.columns:
            y = y.drop_duplicates("sequence_id").sort_values("sequence_id").reset_index(drop=True)

        mask = np.ones(len(y), dtype=bool)
        if self.slice_by_orientation is not None and self.orientation_col in y.columns:
            mask &= y[self.orientation_col].isin(self.slice_by_orientation).values
        if self.slice_by_bfrb is not None and 'is_target' in y.columns:
            if self.slice_by_bfrb: mask &= y['is_target'].values
            else: mask &= ~y['is_target'].values
        return X[mask], y[mask]

    def fit(self, X: np.ndarray, y: pd.DataFrame):
        X_sliced, y_sliced = self._slice_data(X, y)
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y_sliced[self.target_col].values)
        
        clf_params = {}
        if self.base_classifier_class in [RidgeClassifier, RidgeClassifierCV]:
            clf_params['alpha'] = self.alpha
            clf_params['class_weight'] = 'balanced'
        elif self.base_classifier_class == LogisticRegression:
            clf_params['C'] = 1.0 / self.alpha if self.alpha > 0 else 1.0
            clf_params['class_weight'] = 'balanced'
            clf_params['max_iter'] = 1000
            
        if self.base_classifier_params: clf_params.update(self.base_classifier_params)
        
        self.model_ = RocketClassifierLayer(
            num_kernels=self.num_kernels,
            feature_selection_percentile=self.feature_selection_percentile,
            base_classifier_class=self.base_classifier_class,
            base_classifier_params=clf_params,
            random_state=self.random_state,
            padding_value=self.padding_value,
            max_channels=self.max_channels
        )
        self.model_.fit(X_sliced, y_enc)
        return self

    def predict(self, X: np.ndarray):
        preds_enc = self.model_.predict(X)
        return self.le_.inverse_transform(preds_enc)