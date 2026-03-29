import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from sklearn.base import BaseEstimator, ClassifierMixin, clone, TransformerMixin
from joblib import parallel
from contextlib import contextmanager
from tqdm.auto import tqdm
from scipy.ndimage import label, center_of_mass
from sklearn.preprocessing import LabelEncoder


def calculate_fft(array_values: np.ndarray) -> np.ndarray:
    return abs(fft(array_values))[0:len(array_values) // 2]


def convert_frame_to_fft(df: pd.DataFrame, sampling_rate: int, sample_points_N: int = None) -> pd.Series:
    sample_interval_T = 1 / sampling_rate
    if sample_points_N is None:
        sample_points_N = len(df)
    
    array_values = df.values
    single_axis_fft_values = abs(fft(array_values))[0:sample_points_N // 2]
    single_axis_fft_columns = fftfreq(sample_points_N, sample_interval_T)[0:sample_points_N // 2]
    return pd.Series(single_axis_fft_values, index=single_axis_fft_columns, name='freq')


def extract_features(
    signal,
    sampling_rate: float = 1000.0,
    band_edges=None,
) -> pd.Series:
    # INCORRECT
    x = np.asarray(signal, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        raise ValueError("signal is empty or invalid")

    # ---------- Time domain ----------
    features = {
        "mean": np.mean(x),
        "std": np.std(x),
        "min": np.min(x),
        "max": np.max(x),
        "median": np.median(x),
        "range": np.max(x) - np.min(x),
        "energy": np.sum(x ** 2),
        "rms": np.sqrt(np.mean(x ** 2)),
    }

    # ---------- FFT ----------
    n = len(x)
    fft_vals = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1 / sampling_rate)
    mags = np.abs(fft_vals)

    peak_idx = np.argmax(mags)

    features.update({
        "fft_peak_frequency": freqs[peak_idx],
        "fft_peak_magnitude": mags[peak_idx],
        "fft_total_energy": np.sum(mags ** 2),
        "fft_mean": np.mean(mags),
        "fft_std": np.std(mags),
    })

    # ---------- Bands ----------
    if band_edges is None:
        band_edges = np.linspace(0, sampling_rate / 2, 6)  # 5 bands

    band_edges = np.asarray(band_edges)

    for i, (low, high) in enumerate(zip(band_edges[:-1], band_edges[1:]), start=1):
        mask = (freqs >= low) & (freqs < high) if i < len(band_edges) - 1 else (freqs >= low) & (freqs <= high)
        features[f"fft_band_{i}_energy"] = np.sum(mags[mask] ** 2) if np.any(mask) else 0.0

    return pd.Series(features)


class ImuExtractor(BaseEstimator, TransformerMixin):
    def __init__(self,
                 imu_sensor_list: list = None,
                 rotation_sensor_list: list = None,
                 thermopile_sensor_list: list = None,
                 sampling_rate: int = 100,
                 imu_domain: str = 'acceleration',
                 rotation_domain: str = 'acceleration',
                 dc_offset: float = 2.0,
                 band_edges: list = None,
                 subject_df: pd.DataFrame = None,
                 disable_tqdm: bool = True,
                 category_data: bool = True,
                 segmentation: str = None,
                 window: float = 2.0,
                 step_sec: float = 1.0,
                 combine_imu_axes: bool = False,
                 combine_rot_axes: bool = False,
                 tof_sensor_list: list = None,
                 tof_mode: str = None
                 ):
        self.imu_sensor_list = imu_sensor_list
        self.rotation_sensor_list = rotation_sensor_list
        self.sampling_rate = sampling_rate
        self.imu_domain = imu_domain
        self.dc_offset = dc_offset
        self.band_edges = band_edges
        self.subject_df = subject_df
        self.disable_tqdm = disable_tqdm
        self.category_data = category_data
        self.segmentation = segmentation
        self.window = window
        self.step_sec = step_sec
        self.combine_imu_axes = combine_imu_axes
        self.rotation_domain = rotation_domain
        self.combine_rot_axes = combine_rot_axes
        self.thermopile_sensor_list = thermopile_sensor_list
        self.tof_sensor_list = tof_sensor_list
        self.tof_mode = tof_mode

    def fit(self, X, y=None):
        # print(X.shape, y.shape)
        if self.imu_domain == 'time' and self.band_edges is not None:
            self.band_edges = None
        return self

    def transform(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        sequence_list = []
        sequence_ids = raw_data['sequence_id'].unique()

        if self.disable_tqdm:
            iterable = sequence_ids
        else:
            iterable = tqdm(sequence_ids, desc="ImuExtractor", leave=False)

        for a_sequence in iterable:
            single_sequence_df = raw_data[raw_data['sequence_id'] == a_sequence]
            if self.segmentation == 'window':
                sequence_groups_list = self.window_segments(single_sequence_df, self.window, self.step_sec,
                                                            self.sampling_rate)
            else:
                sequence_groups_list = self.split_segments(single_sequence_df, self.segmentation)

            for segmented_sequence_df in sequence_groups_list:
                singular_record_list = []
                if self.imu_sensor_list:
                    imu_df = self.process_for_imu_values(segmented_sequence_df)
                    if self.imu_domain == 'time':
                        imu_features_df = self.extract_time_features(imu_df)
                    else:
                        imu_features_df = self.extract_features_from_imu(imu_df, self.band_edges, self.combine_imu_axes)

                    imu_features_df = imu_features_df.reset_index(drop=True)
                    imu_features_df['sequence_id'] = a_sequence
                    singular_record_list.append(imu_features_df)

                if self.rotation_sensor_list:
                    rot_df = segmented_sequence_df[self.rotation_sensor_list]
                    if self.rotation_domain != 'time':
                        rot_df = rot_df.apply(self.convert_frame_to_fft, axis=0, args=(self.sampling_rate,))
                        rot_df = self.filter_signal(rot_df)
                        rot_features_df = self.extract_rotation_features(rot_df, self.combine_rot_axes)
                        singular_record_list.append(rot_features_df)

                # After rotation features, add thermopile
                if self.thermopile_sensor_list:
                    thm_df = segmented_sequence_df[self.thermopile_sensor_list]
                    thm_df = self.extract_thermopile_features(thm_df)
                    singular_record_list.append(thm_df)

                if self.tof_sensor_list:
                    tof_df = self.extract_tof_features(segmented_sequence_df)
                    singular_record_list.append(tof_df)

                category_df = self.return_single_category_desc_record(segmented_sequence_df, self.category_data)
                singular_record_list.append(category_df)

                singular_record_df = pd.concat(singular_record_list, axis=1)
                singular_record_df = singular_record_df.loc[:, ~singular_record_df.columns.duplicated()]
                sequence_list.append(singular_record_df)

        final_df = pd.concat(sequence_list, ignore_index=True)

        if self.subject_df is not None and 'subject' in final_df.columns:
            final_df = final_df.merge(self.subject_df, how='left', on='subject')

        final_df = final_df.set_index('sequence_id').fillna(0)
        return final_df

    @staticmethod
    def extract_thermopile_features(thm_df: pd.DataFrame) -> pd.DataFrame:
        """Extract simple stats from thermopile sensors"""
        if thm_df.empty:
            return pd.DataFrame()

        features = {}
        for col in thm_df.columns:
            features[f'{col}_mean'] = thm_df[col].mean()
            features[f'{col}_std'] = thm_df[col].std()
            features[f'{col}_min'] = thm_df[col].min()
            features[f'{col}_max'] = thm_df[col].max()

        return pd.DataFrame([features])

    def window_segments(self, df: pd.DataFrame, window_sec: float, step_sec: float, sampling_rate: int) -> list:
        """Split dataframe into windows based on sequence_counter"""
        window_size = int(window_sec * sampling_rate)
        step_size = int(step_sec * sampling_rate)

        segments = []
        max_counter = df['sequence_counter'].max()
        start = 0

        while start + window_size <= max_counter:
            segment = df[(df['sequence_counter'] >= start) & (df['sequence_counter'] < start + window_size)].copy()
            if not segment.empty:
                segment['segment_id'] = start // step_size
                segments.append(segment)
            start += step_size

        return segments if segments else [df.assign(segment_id=0)]

    def return_single_category_desc_record(self, df: pd.DataFrame, category_data_bool: bool
                                           ) -> pd.DataFrame:
        group_list = ['segment_id']
        if category_data_bool:
            group_list.append('orientation')
            group_list.append('subject')
        return df.groupby('sequence_id')[group_list].agg(lambda x: x.unique()[0]).reset_index()

    def process_for_imu_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.get_accelerometer_values(df)
        if self.imu_domain != 'time':
            df = df.apply(self.convert_frame_to_fft, axis=0, args=(self.sampling_rate,))
            df = self.apply_domain_scaling(df, df.index, self.imu_domain)
            df = self.filter_signal(df)
        return df

    def get_accelerometer_values(self, sensor_values: pd.DataFrame) -> pd.DataFrame:
        return sensor_values[self.imu_sensor_list]

    def filter_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        df.loc[0:self.dc_offset] = 0
        return df

    def apply_domain_scaling(self, fft_df: pd.DataFrame, freqs: np.array, domain: str) -> pd.DataFrame:
        safe_freqs = np.where(freqs == 0, 1e-6, freqs)
        if domain == 'velocity':
            scale_factor = 1 / (2 * np.pi * safe_freqs)
        elif domain == 'displacement':
            scale_factor = 1 / ((2 * np.pi * safe_freqs) ** 2)
        else:
            return fft_df
        return fft_df.mul(scale_factor, axis=0)

    @staticmethod
    def convert_frame_to_fft(df: pd.DataFrame, sampling_rate: int, sample_points_N: int = None, **kwargs) -> pd.Series:
        sample_interval_T = 1 / sampling_rate
        if sample_points_N is None:
            sample_points_N = len(df)

        array_values = df.values
        single_axis_fft_values = abs(fft(array_values))[0:sample_points_N // 2]
        single_axis_fft_columns = fftfreq(sample_points_N, sample_interval_T)[0:sample_points_N // 2]
        return pd.Series(single_axis_fft_values, index=single_axis_fft_columns, name='Frequency')

    @staticmethod
    def extract_features_from_imu(fft_df: pd.DataFrame, band_edges: list = None,
                                  combine_axes: bool = True) -> pd.DataFrame:
        if band_edges is None:
            band_edges = np.arange(0, 101, 10)

        features = {}
        freqs = fft_df.index.values

        # Single axis features (always extract)
        for axis in fft_df.columns:
            mags = fft_df[axis].values
            peak_idx = np.argmax(mags)
            features[f'{axis}_peak_freq'] = freqs[peak_idx]
            features[f'{axis}_peak_mag'] = mags[peak_idx]
            features[f'{axis}_total_energy'] = np.sum(mags ** 2)

            for low, high in zip(band_edges[:-1], band_edges[1:]):
                mask = (freqs >= low) & (freqs < high)
                if np.any(mask):
                    band_mags = mags[mask]
                    features[f'{axis}_band_{low}_{high}_energy'] = np.sum(band_mags ** 2)

        # Combined axes (if combine_axes=True and have at least 2 axes)
        if combine_axes and len(fft_df.columns) >= 2:
            # All axes together
            if len(fft_df.columns) >= 2:
                combined = np.sqrt(np.sum([fft_df[col].values ** 2 for col in fft_df.columns], axis=0))

                peak_idx = np.argmax(combined)
                features[f'all_peak_freq'] = freqs[peak_idx]
                features[f'all_peak_mag'] = combined[peak_idx]
                features[f'all_total_energy'] = np.sum(combined ** 2)

                for low, high in zip(band_edges[:-1], band_edges[1:]):
                    mask = (freqs >= low) & (freqs < high)
                    if np.any(mask):
                        band_mags = combined[mask]
                        features[f'all_band_{low}_{high}_energy'] = np.sum(band_mags ** 2)

            # Pairwise combinations
            from itertools import combinations
            for ax1, ax2 in combinations(fft_df.columns, 2):
                pair_mags = np.sqrt(fft_df[ax1].values ** 2 + fft_df[ax2].values ** 2)
                name = f'{ax1}_{ax2}'

                peak_idx = np.argmax(pair_mags)
                features[f'{name}_peak_freq'] = freqs[peak_idx]
                features[f'{name}_peak_mag'] = pair_mags[peak_idx]
                features[f'{name}_total_energy'] = np.sum(pair_mags ** 2)

        return pd.DataFrame([features])

    @staticmethod
    def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Simple, explicit time-domain feature extraction.
        Input: DataFrame with sensor columns (acc_x, acc_y, etc.)
        Output: Single-row DataFrame with statistical features.
        """
        features = {}
        for axis in df.columns:
            signal = df[axis].values

            # Basic Statistics
            features[f'{axis}_mean'] = np.mean(signal)
            features[f'{axis}_std'] = np.std(signal)
            features[f'{axis}_min'] = np.min(signal)
            features[f'{axis}_max'] = np.max(signal)
            features[f'{axis}_rms'] = np.sqrt(np.mean(signal ** 2))

            # Signal Characteristics
            features[f'{axis}_peak_to_peak'] = np.ptp(signal)
            features[f'{axis}_zero_crossings'] = np.where(np.diff(np.sign(signal)))[0].size

            # Energy
            features[f'{axis}_energy'] = np.sum(signal ** 2)

        return pd.DataFrame([features])

    def split_segments(self, df: pd.DataFrame, split_by: str = None):
        """Return list of segment dataframes"""
        if split_by == 'phase':
            groups = df.groupby('phase', sort=False)
        elif split_by == 'behavior':
            groups = df.groupby('behavior', sort=False)
        elif split_by == 'both':
            groups = df.groupby(['phase', 'behavior'], sort=False)
        else:
            return [df.assign(segment_id=0)]

        segments = []
        for i, (_, group) in enumerate(groups):
            segments.append(group.assign(segment_id=i))
        return segments

    def process_rotation_values(self, df: pd.DataFrame, domain: str) -> pd.DataFrame:
        if domain != 'time':
            rot_df = df.apply(self.convert_frame_to_fft, axis=0, args=(self.sampling_rate,))
            rot_df = self.apply_domain_scaling(rot_df, rot_df.index, domain)
            rot_df = self.filter_signal(rot_df)
        else:
            return df
        return rot_df

    @staticmethod
    def extract_rotation_features(rot_df: pd.DataFrame, combine_axes: bool = False) -> pd.DataFrame:
        """Extract simple stats from rotation quaternions"""
        if rot_df.empty:
            return pd.DataFrame()

        features = {}

        # Single axis features
        for col in rot_df.columns:
            features[f'{col}_mean'] = rot_df[col].mean()
            features[f'{col}_std'] = rot_df[col].std()
            features[f'{col}_min'] = rot_df[col].min()
            features[f'{col}_max'] = rot_df[col].max()

        # Combined magnitude (sqrt(w² + x² + y² + z²) = 1 for unit quaternions, but may vary)
        if combine_axes and len(rot_df.columns) >= 2:
            # Combined magnitude
            combined = np.sqrt(np.sum([rot_df[col].values ** 2 for col in rot_df.columns], axis=0))
            features['rot_magnitude_mean'] = np.mean(combined)
            features['rot_magnitude_std'] = np.std(combined)
            features['rot_magnitude_min'] = np.min(combined)
            features['rot_magnitude_max'] = np.max(combined)

            # Pairwise combinations
            from itertools import combinations
            for ax1, ax2 in combinations(rot_df.columns, 2):
                pair = np.sqrt(rot_df[ax1].values ** 2 + rot_df[ax2].values ** 2)
                features[f'{ax1}_{ax2}_mean'] = np.mean(pair)
                features[f'{ax1}_{ax2}_std'] = np.std(pair)

        return pd.DataFrame([features])

    @staticmethod
    def process_thermopile_values(df: pd.DataFrame) -> pd.DataFrame:
        """Extract stats from thermopile sensors (no FFT)"""
        thm_cols = ['thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5']
        thm_cols = [col for col in thm_cols if col in df.columns]

        if not thm_cols:
            return pd.DataFrame()

        features = {}
        for col in thm_cols:
            features[f'{col}_mean'] = df[col].mean()
            features[f'{col}_std'] = df[col].std()
            features[f'{col}_min'] = df[col].min()
            features[f'{col}_max'] = df[col].max()

        return pd.DataFrame([features])

    def extract_tof_features(self, segmented_sequence_df: pd.DataFrame) -> pd.DataFrame:
        if segmented_sequence_df.empty:
            return pd.DataFrame()

        all_cols = segmented_sequence_df.columns
        sensor_prefixes = sorted(list(set(['_'.join(c.split('_')[:-1])
                                           for c in all_cols if '_v' in c])))

        all_sensor_features = {}

        for sensor_id in sensor_prefixes:
            s_cols = [c for c in all_cols if c.startswith(f"{sensor_id}_v")]
            s_cols = sorted(s_cols, key=lambda x: int(x.split('_v')[-1]))

            if len(s_cols) != 64:
                continue

            raw_values = segmented_sequence_df[s_cols].values
            raw_values = np.where((raw_values <= 0) | (raw_values >= 4000), np.nan, raw_values)
            frames = raw_values.reshape(-1, 8, 8)

            if self.tof_mode == 'research':
                all_sensor_features.update(self.tof_research_logic(frames, sensor_id))
            else:
                all_sensor_features.update(self.tof_simple_logic(frames, sensor_id))
        return pd.DataFrame([all_sensor_features])

    @staticmethod
    def tof_simple_logic(frames: np.ndarray, sensor_id: str) -> dict:
        """Simple Version: Zone-based averages"""
        mean_frame = np.nanmean(frames, axis=0)
        if np.all(np.isnan(mean_frame)):
            return {
                f"{sensor_id}_simple_min": 4000.0,
                f"{sensor_id}_simple_avg": 4000.0,
                f"{sensor_id}_center_avg": 4000.0,
                f"{sensor_id}_top_avg": 4000.0,
                f"{sensor_id}_bottom_avg": 4000.0,
            }
        mean_frame = np.nan_to_num(mean_frame, nan=4000.0)
        return {
            f"{sensor_id}_simple_min": np.min(mean_frame),
            f"{sensor_id}_simple_avg": np.mean(mean_frame),
            f"{sensor_id}_center_avg": np.mean(mean_frame[2:6, 2:6]),
            f"{sensor_id}_top_avg": np.mean(mean_frame[:4, :]),
            f"{sensor_id}_bottom_avg": np.mean(mean_frame[4:, :]),
        }

       
    @staticmethod
    def tof_research_logic(frames: np.ndarray, sensor_id: str) -> dict:
        mean_frame = np.nanmean(frames, axis=0)

        feat = {
            f"{sensor_id}_blob_area": 0,
            f"{sensor_id}_mass_r": 4,
            f"{sensor_id}_mass_c": 4,
            f"{sensor_id}_motion_intensity": 0,
        }

        if np.all(np.isnan(mean_frame)):
            return feat

        mean_frame = np.nan_to_num(mean_frame, nan=4000.0)

        mask = mean_frame < 1500
        labeled, num_blobs = label(mask)

        if num_blobs > 0:
            counts = np.bincount(labeled.ravel())
            counts[0] = 0
            largest = counts.argmax()

            weights = np.maximum(0, 1500 - mean_frame)
            weights[labeled != largest] = 0
            if weights.sum() > 0:
                r, c = center_of_mass(weights)
                feat[f"{sensor_id}_blob_area"] = np.sum(labeled == largest)
                feat[f"{sensor_id}_mass_r"] = r
                feat[f"{sensor_id}_mass_c"] = c

        if frames.shape[0] > 1:
            diffs = np.abs(np.diff(frames, axis=0))
            motion = np.nanmean(diffs)
            feat[f"{sensor_id}_motion_intensity"] = 0 if np.isnan(motion) else motion

        return feat


class ManyToOneWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, extractor, mode=None):
        self.estimator = estimator
        self.extractor = extractor
        self.mode = mode

    def _collapse_y_to_sequence(self, X, y):
        # build one label per sequence_id
        gesture_map = (
            y.drop_duplicates(subset='sequence_id')
             .set_index('sequence_id')['gesture']
        )

        y_seq = pd.Series(X.index.map(gesture_map), index=X.index, name='gesture')

        if y_seq.isna().any():
            missing_ids = y_seq[y_seq.isna()].index.unique().tolist()
            raise ValueError(
                f"Missing gesture labels after aligning to X.index. "
                f"Example missing sequence_id values: {missing_ids[:10]}"
            )

        if len(y_seq) == 0:
            raise ValueError("y_seq is empty after collapsing to sequence level.")

        return y_seq

    def fit(self, X, y):
        gesture_map = (
            y.drop_duplicates(subset='sequence_id')
                .set_index('sequence_id')['gesture']
        )

        y_seq = X.index.to_series().map(gesture_map)

        # 🚨 DROP BAD ALIGNMENTS
        valid_mask = y_seq.notna()
        X = X.loc[valid_mask]
        y_seq = y_seq.loc[valid_mask]

        if len(y_seq) == 0:
            raise ValueError("No valid labels after alignment — check sequence_id mismatch.")

        if self.mode == 'xgboost':
            self.label_encoder_ = LabelEncoder()
            y_enc = self.label_encoder_.fit_transform(y_seq)

            if len(np.unique(y_enc)) == 0:
                raise ValueError("No classes in this fold")

            self.estimator_ = clone(self.estimator)
            self.estimator_.fit(X, y_enc)

        else:
            self.estimator_ = clone(self.estimator)
            self.estimator_.fit(X, y_seq)

        return self

    def predict(self, X):
        preds = self.estimator_.predict(X)
        if self.mode == 'xgboost':
            preds = self.label_encoder_.inverse_transform(preds.astype(int))
        return preds

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def score(self, X, y, sample_weight=None):
        y_true = self._collapse_y_to_sequence(X, y).to_numpy()
        y_pred = self.predict(X)

        if sample_weight is not None:
            correct = (y_true == y_pred).astype(int)
            return np.average(correct, weights=sample_weight)

        return np.mean(y_true == y_pred)

@contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = parallel.BatchCompletionCallBack
    parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()


class existing_cols:
    def __init__(self, cols):
        self.cols = cols

    def __call__(self, X):
        return [c for c in self.cols if c in X.columns]


def attach_metadata(grid_search):
    model = grid_search.best_estimator_

    wrapped_clf = model.named_steps["classifier"]
    rf = wrapped_clf.estimator_

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()

    model.metadata_ = {
        "best_score": float(grid_search.best_score_),
        "best_params": grid_search.best_params_,
    }

    model.feature_importances_ = {
        "feature_names": feature_names.tolist(),
        "importances": rf.feature_importances_.tolist()
    }

    return model


def sample_balanced_split(df, train_pct=0.20, test_pct=0.05, random_state=42):
    total_sequences = df['sequence_id'].nunique()
    n_gestures = df['gesture'].nunique()
    n_subjects = df['subject'].nunique()

    train_target = int(total_sequences * train_pct)
    test_target  = int(total_sequences * test_pct)

    train_seqs_per_cell = train_target // (n_subjects * n_gestures)
    test_seqs_per_cell  = test_target  // (n_subjects * n_gestures)

    min_pct = n_subjects * n_gestures / total_sequences

    if train_seqs_per_cell == 0 or test_seqs_per_cell == 0:
        raise ValueError(
            f"Percentage too small. "
            f"Min viable pct for this data: {min_pct:.1%}"
        )

    train_ids, test_ids = [], []

    for _, group in df.groupby(['subject', 'gesture']):
        unique_seqs = group['sequence_id'].drop_duplicates().sample(frac=1, random_state=random_state)

        n_train = min(train_seqs_per_cell, len(unique_seqs))
        n_test  = min(test_seqs_per_cell,  len(unique_seqs) - n_train)

        train_ids.extend(unique_seqs.iloc[:n_train].tolist())
        test_ids.extend( unique_seqs.iloc[n_train:n_train + n_test].tolist())

    train_df = df[df['sequence_id'].isin(train_ids)]
    test_df = df[df['sequence_id'].isin(test_ids)]

    assert len(set(train_ids) & set(test_ids)) == 0, "Overlap detected!"

    print(f"Train: {train_df['sequence_id'].nunique()} seqs | {100*train_df['sequence_id'].nunique()/total_sequences:.1f}%")
    print(f"Test:  {test_df['sequence_id'].nunique()} seqs  | {100*test_df['sequence_id'].nunique()/total_sequences:.1f}%")

    return train_df, test_df

