import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from baselines_utils import RidgeRocketClassifier

class HierarchicalBFRBEnsemble(BaseEstimator, ClassifierMixin):
    """
    3-Layer Hierarchical MiniRocket Ensemble for BFRB classification.
    Layer 1: Binary (is_bfrb vs non_bfrb)
    Layer 2: Orientation prediction
    Layer 3: 4 specialized BFRB gesture models routed by orientation
    """
    def __init__(
        self,
        # Layer 1: Binary
        l1_num_kernels=1000, l1_alpha=1.0, l1_class_weight="balanced",
        # Layer 2: Orientation
        l2_num_kernels=1000, l2_alpha=1.0, l2_class_weight="balanced",
        # Layer 3: BFRB Gesture per orientation (Models 1 to 4)
        l3_1_num_kernels=1000, l3_1_alpha=1.0,
        l3_2_num_kernels=1000, l3_2_alpha=1.0,
        l3_3_num_kernels=1000, l3_3_alpha=1.0,
        l3_4_num_kernels=1000, l3_4_alpha=1.0,
        l3_class_weight="balanced",
        # Metadata
        orientation_col="orientation",
        target_col="bfrb",
        l3_params_dict=None,  # Optional: Dict to override L3 params manually
        random_state=42
    ):
        # Layer 1
        self.l1_num_kernels = l1_num_kernels
        self.l1_alpha = l1_alpha
        self.l1_class_weight = l1_class_weight
        
        # Layer 2
        self.l2_num_kernels = l2_num_kernels
        self.l2_alpha = l2_alpha
        self.l2_class_weight = l2_class_weight
        
        # Layer 3
        self.l3_1_num_kernels = l3_1_num_kernels
        self.l3_1_alpha = l3_1_alpha
        self.l3_2_num_kernels = l3_2_num_kernels
        self.l3_2_alpha = l3_2_alpha
        self.l3_3_num_kernels = l3_3_num_kernels
        self.l3_3_alpha = l3_3_alpha
        self.l3_4_num_kernels = l3_4_num_kernels
        self.l3_4_alpha = l3_4_alpha
        self.l3_class_weight = l3_class_weight
        
        self.orientation_col = orientation_col
        self.target_col = target_col
        
        # 🚨 FIX: Must be exact assignment for sklearn cloning to work!
        self.l3_params_dict = l3_params_dict  
        
        self.random_state = random_state

    def fit(self, X, y):
        seq_ids = X["sequence_ids"]
        y_seq = y.drop_duplicates("sequence_id").set_index("sequence_id").reindex(seq_ids).reset_index(drop=True)
        
        # --- Layer 1: Binary (is_bfrb) ---
        y_l1 = (y_seq[self.target_col] != 'non_bfrb').astype(int)
        self.l1_model_ = RidgeRocketClassifier(
            num_kernels=self.l1_num_kernels, alpha=self.l1_alpha, 
            class_weight=self.l1_class_weight, random_state=self.random_state
        )
        self.l1_model_.fit(X, y_l1)
        l1_preds = self.l1_model_.predict(X)
        
        # --- Layer 2: Orientation ---
        y_l2 = y_seq[self.orientation_col]
        self.l2_model_ = RidgeRocketClassifier(
            num_kernels=self.l2_num_kernels, alpha=self.l2_alpha,
            class_weight=self.l2_class_weight, random_state=self.random_state
        )
        self.l2_model_.fit(X, y_l2)
        
        # --- Layer 3: BFRB Gesture per Orientation ---
        self.orientations_ = sorted(y_seq[self.orientation_col].unique())
        self.l3_models_ = {}
        
        # Safe access to the dictionary inside fit (not in __init__)
        l3_dict = self.l3_params_dict or {}
        
        for idx, orient in enumerate(self.orientations_):
            # Slice data: True orientation AND predicted as bfrb by Layer 1
            mask_l3 = (y_seq[self.orientation_col] == orient) & (l1_preds == 1)
            X_l3 = {k: v[mask_l3] for k, v in X.items()}
            y_l3 = y_seq.loc[mask_l3, self.target_col]
            
            if len(y_l3) > 0 and len(np.unique(y_l3)) >= 2:
                param_suffix = f"l3_{idx + 1}" if idx < 4 else "l3_4"
                
                if orient in l3_dict:
                    nk = l3_dict[orient].get('num_kernels', getattr(self, f'{param_suffix}_num_kernels'))
                    al = l3_dict[orient].get('alpha', getattr(self, f'{param_suffix}_alpha'))
                else:
                    nk = getattr(self, f'{param_suffix}_num_kernels')
                    al = getattr(self, f'{param_suffix}_alpha')
                
                model = RidgeRocketClassifier(
                    num_kernels=nk, alpha=al, 
                    class_weight=self.l3_class_weight, random_state=self.random_state
                )
                model.fit(X_l3, y_l3)
                self.l3_models_[orient] = model
            else:
                unique_class = np.unique(y_l3)[0] if len(y_l3) > 0 else 'non_bfrb'
                self.l3_models_[orient] = {'dummy': True, 'class': unique_class}
                
        self.classes_ = np.unique(y_seq[self.target_col])
        return self

    def predict(self, X):
        l1_preds = self.l1_model_.predict(X)
        l2_preds = self.l2_model_.predict(X)
        
        final_preds = np.full(len(X["sequence_ids"]), 'non_bfrb', dtype=object)
        
        for i in range(len(X["sequence_ids"])):
            if l1_preds[i] == 1:
                orient = l2_preds[i]
                if orient in self.l3_models_:
                    model = self.l3_models_[orient]
                    if isinstance(model, dict) and model.get('dummy'):
                        final_preds[i] = model['class']
                    else:
                        X_single = {k: v[i:i+1] for k, v in X.items()}
                        final_preds[i] = model.predict(X_single)[0]
                        
        return final_preds