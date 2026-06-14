import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.decomposition import PCA

try:
    from sktime.transformations.panel.rocket import MiniRocket
    HAS_MULTI = False  # Standard MiniRocket
except ImportError:
    try:
        from sktime.transformations.panel.multirocket import MultiRocket
        HAS_MULTI = True
    except ImportError:
        # Fallback to MiniRocket if MultiRocket not available
        from sktime.transformations.panel.rocket import MiniRocket
        HAS_MULTI = False


class FlexibleMultiRocketClassifier(BaseEstimator, ClassifierMixin):
    """
    Flexible MultiRocket/Rocket classifier that handles dict input from SequenceExtractor.
    """
    def __init__(
        self,
        target_col="bfrb",
        orientation_col="orientation",
        num_kernels=1000,
        alpha=1.0,
        feature_selection_percentile=None,
        max_channels=None,
        base_classifier_class=RidgeClassifier,
        class_weight="balanced",
        slice_by_orientation=None,
        slice_by_bfrb=None,
        padding_value=0.0,
        random_state=42
    ):
        self.target_col = target_col
        self.orientation_col = orientation_col
        self.num_kernels = num_kernels
        self.alpha = alpha
        self.feature_selection_percentile = feature_selection_percentile
        self.max_channels = max_channels
        self.base_classifier_class = base_classifier_class
        self.class_weight = class_weight
        self.slice_by_orientation = slice_by_orientation
        self.slice_by_bfrb = slice_by_bfrb
        self.padding_value = padding_value
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit the classifier.
        
        Parameters
        ----------
        X : dict or numpy array
            If dict, must have keys 'X' (array of shape n_seq, maxlen, n_features) 
            and 'sequence_ids' (array of sequence IDs)
        y : pandas DataFrame
            Must have columns 'sequence_id' and self.target_col
        """
        # Handle dict input from SequenceExtractor
        if isinstance(X, dict):
            if 'X' not in X or 'sequence_ids' not in X:
                raise ValueError("Dict must have 'X' and 'sequence_ids' keys")
            seq_ids = X['sequence_ids']
            X_arr = X['X']
        else:
            # Assume numpy array (legacy mode)
            X_arr = X
            seq_ids = None
        
        # Align labels using sequence_ids if provided
        if seq_ids is not None:
            if not isinstance(y, pd.DataFrame):
                raise ValueError("When X is a dict, y must be a DataFrame with 'sequence_id' column")
            
            # Build mapping from sequence_id to label
            y_map = y.drop_duplicates('sequence_id').set_index('sequence_id')[self.target_col]
            y_target = np.array([y_map.get(sid, None) for sid in seq_ids])
            
            # Remove any samples with missing labels
            valid_mask = pd.notna(y_target)
            if not valid_mask.all():
                X_arr = X_arr[valid_mask]
                y_target = y_target[valid_mask]
                seq_ids = seq_ids[valid_mask]
        else:
            # Legacy: assume y is already in correct order
            if isinstance(y, pd.DataFrame):
                y_target = y[self.target_col].values
            else:
                y_target = np.asarray(y)
        
        # Apply slicing if specified
        if self.slice_by_orientation is not None or self.slice_by_bfrb is not None:
            if seq_ids is None:
                raise ValueError("Slicing requires sequence_ids from dict input")
            
            # Reconstruct mapping to filter
            y_full = y.drop_duplicates('sequence_id').set_index('sequence_id')
            mask = np.ones(len(seq_ids), dtype=bool)
            
            if self.slice_by_orientation is not None:
                orient_mask = y_full.loc[seq_ids, self.orientation_col].isin(self.slice_by_orientation).values
                mask &= orient_mask
            
            if self.slice_by_bfrb is not None:
                if self.slice_by_bfrb:  # True = BFRB only
                    bfrb_mask = (y_full.loc[seq_ids, self.target_col] != 'non_bfrb').values
                else:  # False = non-BFRB only
                    bfrb_mask = (y_full.loc[seq_ids, self.target_col] == 'non_bfrb').values
                mask &= bfrb_mask
            
            X_arr = X_arr[mask]
            y_target = y_target[mask]
            seq_ids = seq_ids[mask]
        
        # Encode labels
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y_target)
        self.classes_ = self.le_.classes_
        
        # Handle MultiRocket input shape: (n_samples, n_channels, n_timepoints)
        if X_arr.ndim == 3:
            X_sk = X_arr.transpose(0, 2, 1)  # (n_seq, n_features, maxlen)
        else:
            X_sk = X_arr
        
        # Replace any remaining padding with 0.0
        X_sk = np.nan_to_num(X_sk, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply channel compression if specified (to prevent OOM)
        if self.max_channels is not None and X_sk.shape[1] > self.max_channels:
            # Reshape to (n_samples * n_timepoints, n_channels) for PCA fitting
            n_samples, n_channels, n_timepoints = X_sk.shape
            X_flat = X_sk.reshape(-1, n_channels)
            self.pca_ = PCA(n_components=self.max_channels, random_state=self.random_state)
            X_flat_pca = self.pca_.fit_transform(X_flat)
            X_sk = X_flat_pca.reshape(n_samples, self.max_channels, n_timepoints)
        else:
            self.pca_ = None
        
        # Initialize and fit Rocket/MultiRocket
        if HAS_MULTI:
            self.rocket_ = MultiRocket(num_kernels=self.num_kernels, random_state=self.random_state)
        else:
            self.rocket_ = MiniRocket(num_kernels=self.num_kernels, random_state=self.random_state)
        
        X_feat = self.rocket_.fit_transform(X_sk)
        
        # Optional feature selection
        if self.feature_selection_percentile is not None and self.feature_selection_percentile < 100:
            self.selector_ = SelectPercentile(
                score_func=f_classif, 
                percentile=self.feature_selection_percentile
            )
            X_feat = self.selector_.fit_transform(X_feat, y_enc)
        else:
            self.selector_ = None
        
        # Initialize and fit classifier
        if self.base_classifier_class == RidgeClassifier:
            self.clf_ = RidgeClassifier(
                alpha=self.alpha, 
                class_weight=self.class_weight,
                random_state=self.random_state
            )
        elif self.base_classifier_class == LogisticRegression:
            self.clf_ = LogisticRegression(
                C=1.0/self.alpha if self.alpha > 0 else 1.0,
                class_weight=self.class_weight,
                random_state=self.random_state,
                max_iter=1000
            )
        else:
            self.clf_ = self.base_classifier_class(
                alpha=self.alpha, 
                class_weight=self.class_weight,
                random_state=self.random_state
            )
        
        self.clf_.fit(X_feat, y_enc)
        return self

    def predict(self, X):
        """Predict classes for X."""
        # Handle dict input
        if isinstance(X, dict):
            if 'X' not in X:
                raise ValueError("Dict must have 'X' key")
            X_arr = X['X']
        else:
            X_arr = X
        
        # Handle shape
        if X_arr.ndim == 3:
            X_sk = X_arr.transpose(0, 2, 1)
        else:
            X_sk = X_arr
        
        # Clean up
        X_sk = np.nan_to_num(X_sk, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply PCA if used during training
        if hasattr(self, 'pca_') and self.pca_ is not None:
            n_samples, n_channels, n_timepoints = X_sk.shape
            X_flat = X_sk.reshape(-1, n_channels)
            X_flat_pca = self.pca_.transform(X_flat)
            X_sk = X_flat_pca.reshape(n_samples, self.max_channels, n_timepoints)
        
        # Transform
        X_feat = self.rocket_.transform(X_sk)
        
        # Apply feature selection if used
        if hasattr(self, 'selector_') and self.selector_ is not None:
            X_feat = self.selector_.transform(X_feat)
        
        # Predict
        preds_enc = self.clf_.predict(X_feat)
        return self.le_.inverse_transform(preds_enc)

    def predict_proba(self, X):
        """Predict class probabilities."""
        # Handle dict input
        if isinstance(X, dict):
            if 'X' not in X:
                raise ValueError("Dict must have 'X' key")
            X_arr = X['X']
        else:
            X_arr = X
        
        # Handle shape
        if X_arr.ndim == 3:
            X_sk = X_arr.transpose(0, 2, 1)
        else:
            X_sk = X_arr
        
        # Clean up
        X_sk = np.nan_to_num(X_sk, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply PCA if used during training
        if hasattr(self, 'pca_') and self.pca_ is not None:
            n_samples, n_channels, n_timepoints = X_sk.shape
            X_flat = X_sk.reshape(-1, n_channels)
            X_flat_pca = self.pca_.transform(X_flat)
            X_sk = X_flat_pca.reshape(n_samples, self.max_channels, n_timepoints)
        
        # Transform
        X_feat = self.rocket_.transform(X_sk)
        
        # Apply feature selection if used
        if hasattr(self, 'selector_') and self.selector_ is not None:
            X_feat = self.selector_.transform(X_feat)
        
        # Get probabilities
        if hasattr(self.clf_, 'predict_proba'):
            proba = self.clf_.predict_proba(X_feat)
        else:
            # For RidgeClassifier, convert decision function to probabilities
            dec = self.clf_.decision_function(X_feat)
            if dec.ndim == 1:
                dec = np.column_stack([-dec, dec])
            exp = np.exp(dec - dec.max(axis=1, keepdims=True))
            proba = exp / exp.sum(axis=1, keepdims=True)
        
        return proba


class HierarchicalMultiRocketEnsemble(BaseEstimator, ClassifierMixin):
    """
    Hierarchical MultiRocket Ensemble with 3 layers:
    Layer 1: Binary BFRB detection
    Layer 2: Orientation prediction (on BFRB samples only)
    Layer 3: Specialized BFRB gesture models per orientation
    """
    def __init__(
        self,
        # Layer 1 parameters
        l1_num_kernels=1000, l1_alpha=1.0, l1_feature_selection_percentile=None, l1_max_channels=None, l1_class_weight="balanced",
        # Layer 2 parameters
        l2_num_kernels=1000, l2_alpha=1.0, l2_feature_selection_percentile=None, l2_max_channels=None, l2_class_weight="balanced",
        # Layer 3 parameters (for up to 4 orientation models)
        l3_1_num_kernels=1000, l3_1_alpha=1.0, l3_1_feature_selection_percentile=None, l3_1_max_channels=None,
        l3_2_num_kernels=1000, l3_2_alpha=1.0, l3_2_feature_selection_percentile=None, l3_2_max_channels=None,
        l3_3_num_kernels=1000, l3_3_alpha=1.0, l3_3_feature_selection_percentile=None, l3_3_max_channels=None,
        l3_4_num_kernels=1000, l3_4_alpha=1.0, l3_4_feature_selection_percentile=None, l3_4_max_channels=None,
        l3_class_weight="balanced",
        # Common parameters
        orientation_col="orientation",
        target_col="bfrb",
        base_classifier_class=RidgeClassifier,
        padding_value=0.0,
        random_state=42
    ):
        # Layer 1
        self.l1_num_kernels = l1_num_kernels
        self.l1_alpha = l1_alpha
        self.l1_feature_selection_percentile = l1_feature_selection_percentile
        self.l1_max_channels = l1_max_channels
        self.l1_class_weight = l1_class_weight
        
        # Layer 2
        self.l2_num_kernels = l2_num_kernels
        self.l2_alpha = l2_alpha
        self.l2_feature_selection_percentile = l2_feature_selection_percentile
        self.l2_max_channels = l2_max_channels
        self.l2_class_weight = l2_class_weight
        
        # Layer 3
        self.l3_1_num_kernels = l3_1_num_kernels
        self.l3_1_alpha = l3_1_alpha
        self.l3_1_feature_selection_percentile = l3_1_feature_selection_percentile
        self.l3_1_max_channels = l3_1_max_channels
        
        self.l3_2_num_kernels = l3_2_num_kernels
        self.l3_2_alpha = l3_2_alpha
        self.l3_2_feature_selection_percentile = l3_2_feature_selection_percentile
        self.l3_2_max_channels = l3_2_max_channels
        
        self.l3_3_num_kernels = l3_3_num_kernels
        self.l3_3_alpha = l3_3_alpha
        self.l3_3_feature_selection_percentile = l3_3_feature_selection_percentile
        self.l3_3_max_channels = l3_3_max_channels
        
        self.l3_4_num_kernels = l3_4_num_kernels
        self.l3_4_alpha = l3_4_alpha
        self.l3_4_feature_selection_percentile = l3_4_feature_selection_percentile
        self.l3_4_max_channels = l3_4_max_channels
        
        self.l3_class_weight = l3_class_weight
        
        # Common
        self.orientation_col = orientation_col
        self.target_col = target_col
        self.base_classifier_class = base_classifier_class
        self.padding_value = padding_value
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit the hierarchical ensemble.
        
        Parameters
        ----------
        X : dict or numpy array
            If dict, must have keys 'X' (array of shape n_seq, maxlen, n_features) 
            and 'sequence_ids' (array of sequence IDs)
        y : pandas DataFrame
            Must have columns 'sequence_id', self.orientation_col, and self.target_col
        """
        # Handle dict input from SequenceExtractor
        if isinstance(X, dict):
            if 'X' not in X or 'sequence_ids' not in X:
                raise ValueError("Dict must have 'X' and 'sequence_ids' keys")
            seq_ids = X['sequence_ids']
            X_arr = X['X']
        else:
            X_arr = X
            seq_ids = None
        
        # Align labels using sequence_ids
        if seq_ids is not None:
            if not isinstance(y, pd.DataFrame):
                raise ValueError("When X is a dict, y must be a DataFrame with 'sequence_id' column")
            
            # Build mapping from sequence_id to labels
            y_map = y.drop_duplicates('sequence_id').set_index('sequence_id')
            y_target = y_map.loc[seq_ids, self.target_col].values
            y_orient = y_map.loc[seq_ids, self.orientation_col].values
            y_is_target = (y_target != 'non_bfrb').astype(int)
            
            # Remove any samples with missing labels
            valid_mask = pd.notna(y_target) & pd.notna(y_orient)
            if not valid_mask.all():
                X_arr = X_arr[valid_mask]
                seq_ids = seq_ids[valid_mask]
                y_target = y_target[valid_mask]
                y_orient = y_orient[valid_mask]
                y_is_target = y_is_target[valid_mask]
        else:
            # Legacy: assume y is aligned
            if isinstance(y, pd.DataFrame):
                y_target = y[self.target_col].values
                y_orient = y[self.orientation_col].values
            else:
                y_target = np.asarray(y)
                y_orient = np.array(['Unknown'] * len(y_target))
            y_is_target = (y_target != 'non_bfrb').astype(int)
        
        # Prepare X dict for child models
        X_dict = {'X': X_arr, 'sequence_ids': seq_ids} if seq_ids is not None else X_arr
        
        # --- Layer 1: Binary BFRB detection ---
        self.l1_model_ = FlexibleMultiRocketClassifier(
            target_col=self.target_col,
            orientation_col=self.orientation_col,
            num_kernels=self.l1_num_kernels,
            alpha=self.l1_alpha,
            feature_selection_percentile=self.l1_feature_selection_percentile,
            max_channels=self.l1_max_channels,
            base_classifier_class=self.base_classifier_class,
            class_weight=self.l1_class_weight,
            padding_value=self.padding_value,
            random_state=self.random_state
        )
        self.l1_model_.fit(X_dict, y_is_target)
        
        # Get predictions for routing
        l1_preds = self.l1_model_.predict(X_dict)
        
        # --- Layer 2: Orientation prediction (only on BFRB samples) ---
        bfrb_mask = (y_is_target == 1)
        if bfrb_mask.sum() > 0:
            X_l2 = X_arr[bfrb_mask]
            seq_ids_l2 = seq_ids[bfrb_mask] if seq_ids is not None else None
            y_orient_bfrb = y_orient[bfrb_mask]
            
            X_l2_dict = {'X': X_l2, 'sequence_ids': seq_ids_l2} if seq_ids_l2 is not None else X_l2
            
            self.l2_model_ = FlexibleMultiRocketClassifier(
                target_col=self.orientation_col,
                orientation_col=self.orientation_col,
                num_kernels=self.l2_num_kernels,
                alpha=self.l2_alpha,
                feature_selection_percentile=self.l2_feature_selection_percentile,
                max_channels=self.l2_max_channels,
                base_classifier_class=self.base_classifier_class,
                class_weight=self.l2_class_weight,
                padding_value=self.padding_value,
                random_state=self.random_state
            )
            self.l2_model_.fit(X_l2_dict, y_orient_bfrb)
            l2_preds = self.l2_model_.predict(X_l2_dict)
        else:
            self.l2_model_ = None
            l2_preds = np.array([])
        
        # --- Layer 3: Specialized models per orientation ---
        self.orientations_ = sorted(np.unique(y_orient))
        self.l3_models_ = {}
        
        # Map orientation to index
        orient_to_idx = {orient: idx for idx, orient in enumerate(self.orientations_)}
        
        for idx, orient in enumerate(self.orientations_):
            # Get samples: predicted as BFRB by L1 AND have this orientation
            orient_mask = (y_orient == orient) & (y_is_target == 1)
            
            if orient_mask.sum() == 0:
                self.l3_models_[orient] = {'dummy': True, 'class': 'non_bfrb'}
                continue
            
            X_l3 = X_arr[orient_mask]
            seq_ids_l3 = seq_ids[orient_mask] if seq_ids is not None else None
            y_l3 = y_target[orient_mask]
            
            # Get parameters for this orientation
            if idx == 0:
                num_kernels = self.l3_1_num_kernels
                alpha = self.l3_1_alpha
                fs_percentile = self.l3_1_feature_selection_percentile
                max_channels = self.l3_1_max_channels
            elif idx == 1:
                num_kernels = self.l3_2_num_kernels
                alpha = self.l3_2_alpha
                fs_percentile = self.l3_2_feature_selection_percentile
                max_channels = self.l3_2_max_channels
            elif idx == 2:
                num_kernels = self.l3_3_num_kernels
                alpha = self.l3_3_alpha
                fs_percentile = self.l3_3_feature_selection_percentile
                max_channels = self.l3_3_max_channels
            else:
                num_kernels = self.l3_4_num_kernels
                alpha = self.l3_4_alpha
                fs_percentile = self.l3_4_feature_selection_percentile
                max_channels = self.l3_4_max_channels
            
            X_l3_dict = {'X': X_l3, 'sequence_ids': seq_ids_l3} if seq_ids_l3 is not None else X_l3
            
            model = FlexibleMultiRocketClassifier(
                target_col=self.target_col,
                orientation_col=self.orientation_col,
                num_kernels=num_kernels,
                alpha=alpha,
                feature_selection_percentile=fs_percentile,
                max_channels=max_channels,
                base_classifier_class=self.base_classifier_class,
                class_weight=self.l3_class_weight,
                padding_value=self.padding_value,
                random_state=self.random_state
            )
            
            if len(np.unique(y_l3)) >= 2:
                model.fit(X_l3_dict, y_l3)
                self.l3_models_[orient] = model
            else:
                self.l3_models_[orient] = {'dummy': True, 'class': np.unique(y_l3)[0]}
        
        self.classes_ = np.unique(y_target)
        return self

    def predict(self, X):
        """Predict using the hierarchical ensemble."""
        # Handle dict input
        if isinstance(X, dict):
            if 'X' not in X:
                raise ValueError("Dict must have 'X' key")
            X_arr = X['X']
            seq_ids = X.get('sequence_ids', None)
            X_dict = X
        else:
            X_arr = X
            seq_ids = None
            X_dict = X_arr
        
        # Step 1: Binary BFRB prediction
        l1_preds = self.l1_model_.predict(X_dict)
        
        # Step 2: Orientation prediction on BFRB samples
        bfrb_mask = (l1_preds == 'non_bfrb')
        
        # Initialize with non_bfrb
        final_preds = np.full(len(X_arr), 'non_bfrb', dtype=object)
        
        # For BFRB samples, predict orientation and then specialized model
        bfrb_indices = np.where(l1_preds != 'non_bfrb')[0]
        
        if len(bfrb_indices) > 0 and self.l2_model_ is not None:
            # Create X dict for BFRB samples only
            X_bfrb_arr = X_arr[bfrb_indices]
            if seq_ids is not None:
                X_bfrb_dict = {'X': X_bfrb_arr, 'sequence_ids': seq_ids[bfrb_indices]}
            else:
                X_bfrb_dict = X_bfrb_arr
            
            # Predict orientation
            orientations = self.l2_model_.predict(X_bfrb_dict)
            
            # For each orientation, use specialized model
            for i, idx in enumerate(bfrb_indices):
                orient = orientations[i]
                if orient in self.l3_models_:
                    model = self.l3_models_[orient]
                    if isinstance(model, dict) and model.get('dummy'):
                        final_preds[idx] = model['class']
                    else:
                        # Create single-sample dict for prediction
                        X_single_arr = X_bfrb_arr[i:i+1]
                        if seq_ids is not None:
                            X_single_dict = {'X': X_single_arr, 'sequence_ids': seq_ids[bfrb_indices][i:i+1]}
                        else:
                            X_single_dict = X_single_arr
                        final_preds[idx] = model.predict(X_single_dict)[0]
        
        return final_preds

    def predict_proba(self, X):
        """Predict probabilities (simplified - returns probabilities from L1 and L3)."""
        # Handle dict input
        if isinstance(X, dict):
            if 'X' not in X:
                raise ValueError("Dict must have 'X' key")
            X_arr = X['X']
            seq_ids = X.get('sequence_ids', None)
            X_dict = X
        else:
            X_arr = X
            seq_ids = None
            X_dict = X_arr
        
        n_samples = len(X_arr)
        n_classes = len(self.classes_)
        proba = np.zeros((n_samples, n_classes))
        
        # Get L1 probabilities first
        l1_proba = self.l1_model_.predict_proba(X_dict)
        
        # For non-BFRB samples, probability is 1 for non_bfrb class
        non_bfrb_idx = np.where(self.classes_ == 'non_bfrb')[0][0]
        
        for i in range(n_samples):
            if l1_preds[i] == 0:  # Non-BFRB
                proba[i, non_bfrb_idx] = 1.0
            else:
                # For BFRB samples, we'd need L3 probabilities
                # Simplified: use uniform distribution over BFRB classes
                bfrb_class_indices = [j for j, c in enumerate(self.classes_) if c != 'non_bfrb']
                proba[i, bfrb_class_indices] = 1.0 / len(bfrb_class_indices)
        
        return proba