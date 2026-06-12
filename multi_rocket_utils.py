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
    """
    Wraps MultiRocket as a transformer, followed by optional feature selection 
    and a malleable base classifier (e.g., RidgeClassifier, LogisticRegression).
    """
    def __init__(
        self,
        num_kernels: int = 1000,
        feature_selection_percentile: Optional[int] = None,
        base_classifier_class: Type = RidgeClassifier,
        base_classifier_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        padding_value: float = 0.0
    ):
        self.num_kernels = num_kernels
        self.feature_selection_percentile = feature_selection_percentile
        self.base_classifier_class = base_classifier_class
        self.base_classifier_params = base_classifier_params
        self.random_state = random_state
        self.padding_value = padding_value

    def fit(self, X: np.ndarray, y: np.ndarray):
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values
            
        if X.ndim == 3:
            X_sk = X.transpose(0, 2, 1)
        else:
            X_sk = X
            
        X_sk = np.where(X_sk == self.padding_value, 0.0, X_sk)
        
        self.rocket_ = MultiRocket(num_kernels=self.num_kernels, random_state=self.random_state)
        X_feat = self.rocket_.fit_transform(X_sk)
        
        if self.feature_selection_percentile is not None and self.feature_selection_percentile < 100:
            self.selector_ = SelectPercentile(score_func=f_classif, percentile=self.feature_selection_percentile)
            X_feat = self.selector_.fit_transform(X_feat, y)
        else:
            self.selector_ = None
            
        clf_params = self.base_classifier_params.copy() if self.base_classifier_params else {}
        
        # Fallback defaults for Ridge/Logistic if not explicitly provided
        if self.base_classifier_class in [RidgeClassifier, RidgeClassifierCV]:
            if 'alpha' not in clf_params: clf_params['alpha'] = 1000.0
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
        
        X_feat = self.rocket_.transform(X_sk)
        if self.selector_ is not None:
            X_feat = self.selector_.transform(X_feat)
            
        return self.clf_.predict(X_feat)


class HierarchicalMultiRocketEnsemble(BaseEstimator, ClassifierMixin):
    """
    Hierarchical Ensemble:
    Layer 1: Binary (Target vs Non-Target)
    Layer 2: Orientation (4 classes)
    Layer 3_1 to 3_4: BFRB for Orientation 1 to 4
    """
    def __init__(
        self,
        orientation_col: str = "orientation",
        target_col: str = "bfrb",
        base_classifier_class: Type = RidgeClassifier,
        base_classifier_params: Optional[Dict[str, Any]] = None,
        
        # Layer 1 (Binary)
        l1_num_kernels: int = 2000,
        l1_alpha: float = 1000.0,
        l1_feature_selection_percentile: Optional[int] = None,
        l1_class_weight: Optional[str] = 'balanced',
        
        # Layer 2 (Orientation)
        l2_num_kernels: int = 2000,
        l2_alpha: float = 1000.0,
        l2_feature_selection_percentile: Optional[int] = None,
        l2_class_weight: Optional[str] = 'balanced',
        
        # Layer 3 (BFRB per Orientation 1-4)
        l3_1_num_kernels: int = 2000, l3_1_alpha: float = 1000.0, l3_1_feature_selection_percentile: Optional[int] = None,
        l3_2_num_kernels: int = 2000, l3_2_alpha: float = 1000.0, l3_2_feature_selection_percentile: Optional[int] = None,
        l3_3_num_kernels: int = 2000, l3_3_alpha: float = 1000.0, l3_3_feature_selection_percentile: Optional[int] = None,
        l3_4_num_kernels: int = 2000, l3_4_alpha: float = 1000.0, l3_4_feature_selection_percentile: Optional[int] = None,
        l3_class_weight: Optional[str] = 'balanced',
        
        random_state: int = 42,
        padding_value: float = 0.0,
        ensemble_models: Optional[Dict[str, Any]] = None
    ):
        self.orientation_col = orientation_col
        self.target_col = target_col
        self.base_classifier_class = base_classifier_class
        self.base_classifier_params = base_classifier_params
        
        # Store flattened params
        self.l1_num_kernels, self.l1_alpha, self.l1_feature_selection_percentile, self.l1_class_weight = l1_num_kernels, l1_alpha, l1_feature_selection_percentile, l1_class_weight
        self.l2_num_kernels, self.l2_alpha, self.l2_feature_selection_percentile, self.l2_class_weight = l2_num_kernels, l2_alpha, l2_feature_selection_percentile, l2_class_weight
        
        self.l3_1_num_kernels, self.l3_1_alpha, self.l3_1_feature_selection_percentile = l3_1_num_kernels, l3_1_alpha, l3_1_feature_selection_percentile
        self.l3_2_num_kernels, self.l3_2_alpha, self.l3_2_feature_selection_percentile = l3_2_num_kernels, l3_2_alpha, l3_2_feature_selection_percentile
        self.l3_3_num_kernels, self.l3_3_alpha, self.l3_3_feature_selection_percentile = l3_3_num_kernels, l3_3_alpha, l3_3_feature_selection_percentile
        self.l3_4_num_kernels, self.l3_4_alpha, self.l3_4_feature_selection_percentile = l3_4_num_kernels, l3_4_alpha, l3_4_feature_selection_percentile
        self.l3_class_weight = l3_class_weight
        
        self.random_state = random_state
        self.padding_value = padding_value
        self.ensemble_models = ensemble_models or {}

    def _build_layer(self, num_kernels, alpha, fs_percentile, class_weight='balanced'):
        clf_params = {}
        if self.base_classifier_class in [RidgeClassifier, RidgeClassifierCV]:
            clf_params['alpha'] = alpha
            clf_params['class_weight'] = class_weight
        elif self.base_classifier_class == LogisticRegression:
            clf_params['C'] = 1.0 / alpha if alpha > 0 else 1.0
            clf_params['class_weight'] = class_weight
            clf_params['max_iter'] = 1000
            
        if self.base_classifier_params:
            clf_params.update(self.base_classifier_params)
            
        return RocketClassifierLayer(
            num_kernels=num_kernels,
            feature_selection_percentile=fs_percentile,
            base_classifier_class=self.base_classifier_class,
            base_classifier_params=clf_params,
            random_state=self.random_state,
            padding_value=self.padding_value
        )

    def fit(self, X: np.ndarray, y: pd.DataFrame):
        # Layer 1: Binary
        self.le_l1_ = LabelEncoder()
        y_l1 = self.le_l1_.fit_transform(y['is_target'].values)
        
        if 'l1' in self.ensemble_models:
            self.model_l1_ = self.ensemble_models['l1']
        else:
            self.model_l1_ = self._build_layer(self.l1_num_kernels, self.l1_alpha, self.l1_feature_selection_percentile, self.l1_class_weight)
            self.model_l1_.fit(X, y_l1)
            
        pred_l1 = self.model_l1_.predict(X)
        is_target_mask = self.le_l1_.inverse_transform(pred_l1) == True
        X_targ = X[is_target_mask]
        y_targ = y[is_target_mask]
        
        if len(X_targ) == 0: return self

        # Layer 2: Orientation
        self.le_l2_ = LabelEncoder()
        y_l2 = self.le_l2_.fit_transform(y_targ[self.orientation_col].values)
        
        if 'l2' in self.ensemble_models:
            self.model_l2_ = self.ensemble_models['l2']
        else:
            self.model_l2_ = self._build_layer(self.l2_num_kernels, self.l2_alpha, self.l2_feature_selection_percentile, self.l2_class_weight)
            self.model_l2_.fit(X_targ, y_l2)
            
        pred_l2 = self.model_l2_.predict(X_targ)
        pred_orient = self.le_l2_.inverse_transform(pred_l2)
        
        # Layer 3_1 to 3_4: BFRB per Orientation
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
                if layer_name in self.ensemble_models:
                    model = self.ensemble_models[layer_name]
                else:
                    num_k = getattr(self, f"{layer_name}_num_kernels")
                    alp = getattr(self, f"{layer_name}_alpha")
                    fs = getattr(self, f"{layer_name}_feature_selection_percentile")
                    model = self._build_layer(num_k, alp, fs, self.l3_class_weight)
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
    """
    A single Multi-Rocket model that allows flexible target switching.
    """
    def __init__(
        self,
        target_col: str = "bfrb",
        orientation_col: str = "orientation",
        slice_by_orientation: Optional[List[str]] = None,
        slice_by_bfrb: Optional[bool] = None,
        base_classifier_class: Type = RidgeClassifier,
        base_classifier_params: Optional[Dict[str, Any]] = None,
        num_kernels: int = 1000,
        alpha: float = 1000.0,
        feature_selection_percentile: Optional[int] = None,
        class_weight: Optional[str] = 'balanced',
        random_state: int = 42,
        padding_value: float = 0.0
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
        self.class_weight = class_weight
        self.random_state = random_state
        self.padding_value = padding_value
        
    def _slice_data(self, X: np.ndarray, y: pd.DataFrame):
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
            clf_params['class_weight'] = self.class_weight
        elif self.base_classifier_class == LogisticRegression:
            clf_params['C'] = 1.0 / self.alpha if self.alpha > 0 else 1.0
            clf_params['class_weight'] = self.class_weight
            clf_params['max_iter'] = 1000
            
        if self.base_classifier_params: clf_params.update(self.base_classifier_params)
        
        self.model_ = RocketClassifierLayer(
            num_kernels=self.num_kernels,
            feature_selection_percentile=self.feature_selection_percentile,
            base_classifier_class=self.base_classifier_class,
            base_classifier_params=clf_params,
            random_state=self.random_state,
            padding_value=self.padding_value
        )
        self.model_.fit(X_sliced, y_enc)
        return self

    def predict(self, X: np.ndarray):
        preds_enc = self.model_.predict(X)
        return self.le_.inverse_transform(preds_enc)