import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from sklearn.base import BaseEstimator, ClassifierMixin, clone, TransformerMixin
from joblib import parallel
from contextlib import contextmanager
from tqdm.auto import tqdm


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
                 imu_sensor_list: list = None, # Default values
                 sampling_rate: int =100,
                 domain: str = 'acceleration',
                 dc_offset: float = 2.0,
                 band_edges: list = None,
                 subject_df: pd.DataFrame = None,
                 disable_tqdm: bool = True,
                 category_data: bool = True,
                 segmentation: str = None
                 ):
        self.imu_sensor_list = imu_sensor_list
        self.sampling_rate = sampling_rate
        self.domain = domain
        self.dc_offset = dc_offset
        self.band_edges = band_edges
        self.subject_df = subject_df
        self.disable_tqdm = disable_tqdm
        self.category_data = category_data
        self.segmentation = segmentation

    def fit(self, X, y=None):
        if self.domain == 'time' and self.band_edges is not None:
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

            sequence_groups_list = self.split_segments(single_sequence_df, self.segmentation)
            for segmented_sequence_df in sequence_groups_list:
                singular_record_list = []
                if self.imu_sensor_list:
                    imu_df = self.process_for_imu_values(segmented_sequence_df)
                    if self.domain == 'time':
                        imu_features_df = self.extract_time_features(imu_df)
                    else:
                        imu_features_df = self.extract_features_from_imu(imu_df, self.band_edges)

                    imu_features_df = imu_features_df.reset_index(drop=True)
                    imu_features_df['sequence_id'] = a_sequence
                    singular_record_list.append(imu_features_df)

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

    def return_single_category_desc_record(self, df: pd.DataFrame, category_data_bool: bool
                                           ) -> pd.DataFrame:
        group_list = ['segment_id']
        if category_data_bool:
            group_list.append('orientation')
            group_list.append('subject')
        return df.groupby('sequence_id')[group_list].agg(lambda x: x.unique()[0]).reset_index()

    def process_for_imu_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.get_accelerometer_values(df)
        if self.domain != 'time':
            df = df.apply(self.convert_frame_to_fft, axis=0, args=(self.sampling_rate,))
            df = self.apply_domain_scaling(df, df.index, self.domain)
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
    def extract_features_from_imu(fft_df: pd.DataFrame, band_edges: list = None) -> pd.DataFrame:
        if band_edges is None:
            band_edges = np.arange(0, 101, 10)

        features = {}
        freqs = fft_df.index.values

        for axis in fft_df.columns:
            mags = fft_df[axis].values

            # Peak features
            peak_idx = np.argmax(mags)
            features[f'{axis}_peak_freq'] = freqs[peak_idx]
            features[f'{axis}_peak_mag'] = mags[peak_idx]

            # Energy features
            features[f'{axis}_total_energy'] = np.sum(mags ** 2)
            features[f'{axis}_mean'] = np.mean(mags)
            features[f'{axis}_std'] = np.std(mags)

            # Spectral centroid
            if np.sum(mags) > 0:
                features[f'{axis}_centroid'] = np.average(freqs, weights=mags)
            else:
                features[f'{axis}_centroid'] = 0

            # Band features
            for low, high in zip(band_edges[:-1], band_edges[1:]):
                mask = (freqs >= low) & (freqs < high)
                if np.any(mask):
                    band_mags = mags[mask]
                    band_freqs = freqs[mask]
                    features[f'{axis}_band_{low}_{high}_energy'] = np.sum(band_mags ** 2)
                    peak_idx_band = np.argmax(band_mags)
                    features[f'{axis}_band_{low}_{high}_peak_freq'] = band_freqs[peak_idx_band]

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


class ManyToOneWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def _get_tags(self):
        return {'multioutput': True}

    def fit(self, X, y):
        # Slices y to get one label per sequence
        y_seq = y.groupby('sequence_id', sort=False)['gesture'].first()
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y_seq)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def score(self, X, y, sample_weight=None):
        y_true_seq = y.groupby('sequence_id', sort=False)['gesture'].first()
        y_pred = self.predict(X)

        # Manual accuracy calculation
        if sample_weight is not None:
            # Handle weighted accuracy if needed
            correct = (y_true_seq == y_pred).astype(int)
            return np.average(correct, weights=sample_weight)
        else:
            return np.mean(y_true_seq == y_pred)

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


def existing_cols(cols):
    def selector(X):
        return [c for c in cols if c in X.columns]
    return selector