import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone, TransformerMixin
from joblib import parallel
from contextlib import contextmanager
from tqdm.auto import tqdm
from scipy.ndimage import label, center_of_mass
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from scipy.fft import fft, fftfreq, ifft
from pathlib import Path
from scipy import ndimage
import pywt
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.class_weight import compute_class_weight


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
                 rotation_domain: str = 'motion',
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
                 tof_mode: str = 'baseline',
                 thermopile_mode: str = 'baseline'
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
        self.thermopile_mode = thermopile_mode

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
                    rot_df = self.process_rotation_values(rot_df, self.rotation_domain)
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

    def process_rotation_values(self, df: pd.DataFrame, domain: str) -> pd.DataFrame:
        euler_df = self.quaternion_to_euler(df)
        euler_df = self.unwrap_angles(euler_df)

        if domain == 'orientation':
            return euler_df

        elif domain == 'motion':
            motion_df = euler_df.diff().fillna(0)
            return motion_df

        elif domain == 'frequency':
            motion_df = euler_df.diff().fillna(0)
            fft_df = motion_df.apply(self.convert_frame_to_fft, axis=0, args=(self.sampling_rate,))
            fft_df = self.filter_signal(fft_df)
            return fft_df

        else:
            raise ValueError(f"Unknown rotation_domain: {domain}")

    @staticmethod
    def quaternion_to_euler(df: pd.DataFrame) -> pd.DataFrame:
        q = df.astype(float).copy()

        w = q.iloc[:, 0].values
        x = q.iloc[:, 1].values
        y = q.iloc[:, 2].values
        z = q.iloc[:, 3].values

        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)

        t2 = 2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch = np.arcsin(t2)

        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)

        return pd.DataFrame({
            'rot_roll': roll,
            'rot_pitch': pitch,
            'rot_yaw': yaw,
        }, index=df.index)

    def extract_rotation_features(self, rot_df: pd.DataFrame, combine_axes: bool = False) -> pd.DataFrame:
        if rot_df.empty:
            return pd.DataFrame()

        if self.rotation_domain in ['orientation', 'motion']:
            return self.extract_rotation_time_features(rot_df, combine_axes)
        elif self.rotation_domain == 'frequency':
            return self.extract_rotation_frequency_features(rot_df, combine_axes)
        else:
            raise ValueError(f"Unknown rotation_domain: {self.rotation_domain}")

    @staticmethod
    def extract_rotation_time_features(rot_df: pd.DataFrame, combine_axes: bool = False) -> pd.DataFrame:
        features = {}

        for col in rot_df.columns:
            values = rot_df[col].values

            features[f'{col}_mean'] = np.mean(values)
            features[f'{col}_std'] = np.std(values)
            features[f'{col}_min'] = np.min(values)
            features[f'{col}_max'] = np.max(values)
            features[f'{col}_rms'] = np.sqrt(np.mean(values ** 2))
            features[f'{col}_energy'] = np.sum(values ** 2)
            features[f'{col}_abs_mean'] = np.mean(np.abs(values))
            features[f'{col}_peak_to_peak'] = np.ptp(values)
            features[f'{col}_zero_crossings'] = np.where(np.diff(np.sign(values)))[0].size

        if combine_axes and len(rot_df.columns) >= 2:
            combined = np.sqrt(np.sum([rot_df[col].values ** 2 for col in rot_df.columns], axis=0))

            features['rot_magnitude_mean'] = np.mean(combined)
            features['rot_magnitude_std'] = np.std(combined)
            features['rot_magnitude_min'] = np.min(combined)
            features['rot_magnitude_max'] = np.max(combined)
            features['rot_magnitude_rms'] = np.sqrt(np.mean(combined ** 2))
            features['rot_magnitude_energy'] = np.sum(combined ** 2)
            features['rot_magnitude_abs_mean'] = np.mean(np.abs(combined))
            features['rot_magnitude_peak_to_peak'] = np.ptp(combined)

        return pd.DataFrame([features])

    @staticmethod
    def unwrap_angles(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {col: np.unwrap(df[col].values) for col in df.columns},
            index=df.index
        )

    def extract_rotation_frequency_features(self, rot_df: pd.DataFrame, combine_axes: bool = False) -> pd.DataFrame:
        features = {}
        freqs = rot_df.index.values

        if self.band_edges is None:
            max_freq = float(np.nanmax(freqs)) if len(freqs) > 0 else 0.0
            band_edges = np.arange(0, max_freq + 10, 10)
            if len(band_edges) < 2:
                band_edges = np.array([0, max_freq + 1])
        else:
            band_edges = np.asarray(self.band_edges)

        for col in rot_df.columns:
            mags = rot_df[col].values
            peak_idx = np.argmax(mags)

            features[f'{col}_peak_freq'] = freqs[peak_idx]
            features[f'{col}_peak_mag'] = mags[peak_idx]
            features[f'{col}_total_energy'] = np.sum(mags ** 2)
            features[f'{col}_mean'] = np.mean(mags)
            features[f'{col}_std'] = np.std(mags)

            for low, high in zip(band_edges[:-1], band_edges[1:]):
                mask = (freqs >= low) & (freqs < high)
                if np.any(mask):
                    features[f'{col}_band_{low}_{high}_energy'] = np.sum(mags[mask] ** 2)

        if combine_axes and len(rot_df.columns) >= 2:
            combined = np.sqrt(np.sum([rot_df[col].values ** 2 for col in rot_df.columns], axis=0))
            peak_idx = np.argmax(combined)

            features['rot_all_peak_freq'] = freqs[peak_idx]
            features['rot_all_peak_mag'] = combined[peak_idx]
            features['rot_all_total_energy'] = np.sum(combined ** 2)
            features['rot_all_mean'] = np.mean(combined)
            features['rot_all_std'] = np.std(combined)

            for low, high in zip(band_edges[:-1], band_edges[1:]):
                mask = (freqs >= low) & (freqs < high)
                if np.any(mask):
                    features[f'rot_all_band_{low}_{high}_energy'] = np.sum(combined[mask] ** 2)

        return pd.DataFrame([features])

    def filter_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.loc[0:self.dc_offset] = 0
        return df

    def extract_thermopile_features(self, thm_df: pd.DataFrame) -> pd.DataFrame:
        if thm_df.empty:
            return pd.DataFrame()

        if self.thermopile_mode == 'baseline':
            return self.extract_thermopile_baseline(thm_df)
        elif self.thermopile_mode == 'spatial':
            return self.extract_thermopile_spatial(thm_df)
        else:
            raise ValueError(f"Unknown thermopile_mode: {self.thermopile_mode}")

    @staticmethod
    def extract_thermopile_baseline(thm_df: pd.DataFrame) -> pd.DataFrame:
        values = thm_df.astype(float).values.flatten()
        values = values[np.isfinite(values)]

        if len(values) == 0:
            return pd.DataFrame([{
                'thm_all_mean': 0.0,
                'thm_all_std': 0.0,
                'thm_all_min': 0.0,
                'thm_all_max': 0.0,
            }])

        return pd.DataFrame([{
            'thm_all_mean': np.mean(values),
            'thm_all_std': np.std(values),
            'thm_all_min': np.min(values),
            'thm_all_max': np.max(values),
        }])

    @staticmethod
    def extract_thermopile_spatial(thm_df: pd.DataFrame) -> pd.DataFrame:
        thm_df = thm_df.astype(float)

        sensor_means = thm_df.mean(axis=0)
        sensor_stds = thm_df.std(axis=0)

        all_values = thm_df.values.flatten()
        all_values = all_values[np.isfinite(all_values)]

        if len(all_values) == 0:
            return pd.DataFrame([{
                'thm_all_mean': 0.0,
                'thm_all_std': 0.0,
                'thm_all_min': 0.0,
                'thm_all_max': 0.0,
                'thm_spatial_range': 0.0,
                'thm_hottest_sensor_idx': -1,
                'thm_coolest_sensor_idx': -1,
                'thm_left_right_diff': 0.0,
                'thm_center_edge_diff': 0.0,
                'thm_adjacent_diff_mean': 0.0,
                'thm_adjacent_diff_max': 0.0,
                'thm_sensor_mean_std': 0.0,
            }])

        sensor_mean_vals = sensor_means.values
        hottest_idx = int(np.argmax(sensor_mean_vals))
        coolest_idx = int(np.argmin(sensor_mean_vals))

        adjacent_diffs = np.abs(np.diff(sensor_mean_vals))

        # assumes thm_3 is roughly central and others are more peripheral
        center_val = sensor_means.iloc[len(sensor_means) // 2]
        edge_vals = sensor_means.drop(sensor_means.index[len(sensor_means) // 2]).values

        # simple left/right split based on sensor order
        left_mean = np.mean(sensor_mean_vals[:len(sensor_mean_vals) // 2])
        right_mean = np.mean(sensor_mean_vals[len(sensor_mean_vals) // 2:])

        features = {
            'thm_all_mean': np.mean(all_values),
            'thm_all_std': np.std(all_values),
            'thm_all_min': np.min(all_values),
            'thm_all_max': np.max(all_values),

            'thm_spatial_range': np.max(sensor_mean_vals) - np.min(sensor_mean_vals),
            'thm_hottest_sensor_idx': hottest_idx,
            'thm_coolest_sensor_idx': coolest_idx,

            'thm_left_right_diff': left_mean - right_mean,
            'thm_center_edge_diff': center_val - np.mean(edge_vals),

            'thm_adjacent_diff_mean': np.mean(adjacent_diffs) if len(adjacent_diffs) > 0 else 0.0,
            'thm_adjacent_diff_max': np.max(adjacent_diffs) if len(adjacent_diffs) > 0 else 0.0,

            'thm_sensor_mean_std': np.std(sensor_mean_vals),
        }

        # optional: per-sensor temporal stability
        for col in thm_df.columns:
            values = thm_df[col].values
            diffs = np.diff(values) if len(values) > 1 else np.array([0.0])

            features[f'{col}_mean'] = np.mean(values)
            features[f'{col}_std'] = np.std(values)
            features[f'{col}_diff_mean'] = np.mean(diffs)
            features[f'{col}_diff_std'] = np.std(diffs)

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
        if self.imu_domain == 'time':
            return df
        elif self.imu_domain == 'acceleration':
            df = df.apply(self.preprocess_time_signal, axis=0)
            return df.apply(self.convert_frame_to_fft, axis=0, args=(self.sampling_rate,))
        elif self.imu_domain in ['velocity', 'displacement']:
            complex_fft = df.apply(self._fft_complex, axis=0)
            scaled_fft = self._scale_frequency_domain(complex_fft, self.imu_domain)
            return self._ifft_to_time(scaled_fft, len(df))
        else:
            raise ValueError(f"Unknown imu_domain: {self.imu_domain}")

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

    def preprocess_time_signal(self, signal: pd.Series) -> pd.Series:
        signal = signal.astype(float).copy()
        signal = signal - signal.mean()
        signal = signal.rolling(window=3, center=True, min_periods=1).mean()
        window = np.hanning(len(signal))
        signal = signal * window
        return pd.Series(signal, index=signal.index, name=signal.name)

    @staticmethod
    def convert_frame_to_fft(df: pd.DataFrame, sampling_rate: int, sample_points_N: int = None, **kwargs) -> pd.Series:
        sample_interval_T = 1 / sampling_rate
        if sample_points_N is None:
            sample_points_N = len(df)

        array_values = df.values
        single_axis_fft_values = abs(fft(array_values))[0:sample_points_N // 2]
        single_axis_fft_columns = fftfreq(sample_points_N, sample_interval_T)[0:sample_points_N // 2]
        return pd.Series(single_axis_fft_values, index=single_axis_fft_columns, name='Frequency')

    def _fft_complex(self, series: pd.Series) -> np.ndarray:
        """Return complex FFT (preserves phase information)."""
        return fft(series.values)

    def _scale_frequency_domain(self, fft_complex: pd.DataFrame, domain: str) -> pd.DataFrame:
        """Scale complex FFT for integration/differentiation."""
        n = len(fft_complex.index)
        freqs = fftfreq(n, 1 / self.sampling_rate)
        freqs = np.where(freqs == 0, 1e-6, freqs)  # Avoid division by zero

        if domain == 'velocity':
            # V = A / (j * 2πf)
            scale = 1 / (2j * np.pi * freqs)
        elif domain == 'displacement':
            # D = A / (-(2πf)²)
            scale = 1 / (-(2 * np.pi * freqs) ** 2)
        else:
            scale = 1

        # Apply scaling to each column
        scaled = fft_complex.copy()
        for col in fft_complex.columns:
            scaled[col] = fft_complex[col].values * scale
        return scaled

    def _ifft_to_time(self, scaled_fft: pd.DataFrame, original_length: int) -> pd.DataFrame:
        """Convert scaled FFT back to time domain."""
        result = {}
        for col in scaled_fft.columns:
            time_signal = np.real(ifft(scaled_fft[col].values))[:original_length]
            result[col] = time_signal
        return pd.DataFrame(result, index=range(original_length))

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
        sensor_prefixes = sorted(list(set([
            '_'.join(c.split('_')[:-1])
            for c in all_cols if '_v' in c
        ])))

        all_sensor_features = {}

        for sensor_id in sensor_prefixes:
            s_cols = [c for c in all_cols if c.startswith(f"{sensor_id}_v")]
            s_cols = sorted(s_cols, key=lambda x: int(x.split('_v')[-1]))

            if len(s_cols) != 64:
                continue

            raw_values = segmented_sequence_df[s_cols].values
            raw_values = np.where((raw_values <= 0) | (raw_values >= 4000), np.nan, raw_values)
            frames = raw_values.reshape(-1, 8, 8)

            if self.tof_mode == 'blob':
                all_sensor_features.update(self.tof_research_logic(frames, sensor_id))

            elif self.tof_mode == 'baseline':
                all_sensor_features.update(self.tof_simple_logic(frames, sensor_id))

            elif self.tof_mode == 'spatial':
                all_sensor_features.update(self.tof_spatial_logic(frames, sensor_id))

            elif self.tof_mode == 'edge':
                all_sensor_features.update(self.tof_edge_logic(frames, sensor_id))

            elif self.tof_mode == 'fft2':
                all_sensor_features.update(self.tof_fft2_logic(frames, sensor_id))

            elif self.tof_mode == 'svd':
                all_sensor_features.update(self.tof_svd_logic(frames, sensor_id))

            elif self.tof_mode == 'wavelet':
                all_sensor_features.update(self.tof_wavelet_logic(frames, sensor_id))

            else:
                raise ValueError(f"Unknown tof_mode: {self.tof_mode}")

        return pd.DataFrame([all_sensor_features])

    @staticmethod
    def tof_spatial_logic(frames: np.ndarray, sensor_id: str) -> dict:
        frame_features = []

        for frame in frames:
            valid_mask = np.isfinite(frame)
            valid_ratio = np.mean(valid_mask)

            if valid_ratio == 0:
                frame_features.append({
                    'min': 4000.0,
                    'mean': 4000.0,
                    'std': 0.0,
                    'center_mean': 4000.0,
                    'top_mean': 4000.0,
                    'bottom_mean': 4000.0,
                    'left_mean': 4000.0,
                    'right_mean': 4000.0,
                    'q1_mean': 4000.0,
                    'q2_mean': 4000.0,
                    'q3_mean': 4000.0,
                    'q4_mean': 4000.0,
                    'row_com': 4.0,
                    'col_com': 4.0,
                    'valid_ratio': 0.0,
                    'lr_asym': 0.0,
                    'tb_asym': 0.0,
                })
                continue

            filled = np.nan_to_num(frame, nan=4000.0)

            weights = np.maximum(0, 4000.0 - filled)
            if weights.sum() > 0:
                row_com, col_com = center_of_mass(weights)
            else:
                row_com, col_com = 4.0, 4.0

            top_mean = np.mean(filled[:4, :])
            bottom_mean = np.mean(filled[4:, :])
            left_mean = np.mean(filled[:, :4])
            right_mean = np.mean(filled[:, 4:])

            frame_features.append({
                'min': np.min(filled),
                'mean': np.mean(filled),
                'std': np.std(filled),
                'center_mean': np.mean(filled[2:6, 2:6]),
                'top_mean': top_mean,
                'bottom_mean': bottom_mean,
                'left_mean': left_mean,
                'right_mean': right_mean,
                'q1_mean': np.mean(filled[:4, :4]),
                'q2_mean': np.mean(filled[:4, 4:]),
                'q3_mean': np.mean(filled[4:, :4]),
                'q4_mean': np.mean(filled[4:, 4:]),
                'row_com': row_com,
                'col_com': col_com,
                'valid_ratio': valid_ratio,
                'lr_asym': left_mean - right_mean,
                'tb_asym': top_mean - bottom_mean,
            })

        feat_df = pd.DataFrame(frame_features)

        out = {}
        for col in feat_df.columns:
            out[f"{sensor_id}_{col}_mean"] = feat_df[col].mean()
            out[f"{sensor_id}_{col}_std"] = feat_df[col].std()
            out[f"{sensor_id}_{col}_min"] = feat_df[col].min()
            out[f"{sensor_id}_{col}_max"] = feat_df[col].max()

        return out

    @staticmethod
    def tof_edge_logic(frames: np.ndarray, sensor_id: str) -> dict:
        frame_features = []

        for frame in frames:
            if np.all(~np.isfinite(frame)):
                frame_features.append({
                    'grad_mean': 0.0,
                    'grad_std': 0.0,
                    'grad_max': 0.0,
                    'sobel_x_energy': 0.0,
                    'sobel_y_energy': 0.0,
                    'edge_density': 0.0,
                })
                continue

            filled = np.nan_to_num(frame, nan=4000.0)

            sobel_x = ndimage.sobel(filled, axis=1, mode='nearest')
            sobel_y = ndimage.sobel(filled, axis=0, mode='nearest')
            grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

            thresh = np.mean(grad_mag) + np.std(grad_mag)
            edge_density = np.mean(grad_mag > thresh)

            frame_features.append({
                'grad_mean': np.mean(grad_mag),
                'grad_std': np.std(grad_mag),
                'grad_max': np.max(grad_mag),
                'sobel_x_energy': np.sum(sobel_x ** 2),
                'sobel_y_energy': np.sum(sobel_y ** 2),
                'edge_density': edge_density,
            })

        feat_df = pd.DataFrame(frame_features)

        out = {}
        for col in feat_df.columns:
            out[f"{sensor_id}_{col}_mean"] = feat_df[col].mean()
            out[f"{sensor_id}_{col}_std"] = feat_df[col].std()
            out[f"{sensor_id}_{col}_min"] = feat_df[col].min()
            out[f"{sensor_id}_{col}_max"] = feat_df[col].max()

        return out

    @staticmethod
    def tof_fft2_logic(frames: np.ndarray, sensor_id: str) -> dict:
        frame_features = []

        for frame in frames:
            if np.all(~np.isfinite(frame)):
                frame_features.append({
                    'fft_total_energy': 0.0,
                    'fft_low_energy': 0.0,
                    'fft_high_energy': 0.0,
                    'fft_low_high_ratio': 0.0,
                    'fft_dc': 0.0,
                })
                continue

            filled = np.nan_to_num(frame, nan=4000.0)
            centered = filled - np.mean(filled)

            fft2 = np.fft.fft2(centered)
            fft2_shift = np.fft.fftshift(fft2)
            mag = np.abs(fft2_shift)

            total_energy = np.sum(mag ** 2)

            center_block = mag[3:5, 3:5]
            low_energy = np.sum(center_block ** 2)
            high_energy = total_energy - low_energy

            ratio = low_energy / high_energy if high_energy > 0 else 0.0

            frame_features.append({
                'fft_total_energy': total_energy,
                'fft_low_energy': low_energy,
                'fft_high_energy': high_energy,
                'fft_low_high_ratio': ratio,
                'fft_dc': mag[4, 4],
            })

        feat_df = pd.DataFrame(frame_features)

        out = {}
        for col in feat_df.columns:
            out[f"{sensor_id}_{col}_mean"] = feat_df[col].mean()
            out[f"{sensor_id}_{col}_std"] = feat_df[col].std()
            out[f"{sensor_id}_{col}_min"] = feat_df[col].min()
            out[f"{sensor_id}_{col}_max"] = feat_df[col].max()

        return out

    @staticmethod
    def tof_svd_logic(frames: np.ndarray, sensor_id: str) -> dict:
        frame_features = []

        for frame in frames:
            if np.all(~np.isfinite(frame)):
                frame_features.append({
                    'sv1': 0.0,
                    'sv2': 0.0,
                    'sv3': 0.0,
                    'sv1_ratio': 0.0,
                    'sv12_ratio': 0.0,
                    'rank_energy_2': 0.0,
                })
                continue

            filled = np.nan_to_num(frame, nan=4000.0)
            centered = filled - np.mean(filled)

            _, s, _ = np.linalg.svd(centered, full_matrices=False)

            total_sv = np.sum(s)
            total_sv_sq = np.sum(s ** 2)

            sv1 = s[0] if len(s) > 0 else 0.0
            sv2 = s[1] if len(s) > 1 else 0.0
            sv3 = s[2] if len(s) > 2 else 0.0

            sv1_ratio = sv1 / total_sv if total_sv > 0 else 0.0
            sv12_ratio = (sv1 + sv2) / total_sv if total_sv > 0 else 0.0
            rank_energy_2 = np.sum(s[:2] ** 2) / total_sv_sq if total_sv_sq > 0 else 0.0

            frame_features.append({
                'sv1': sv1,
                'sv2': sv2,
                'sv3': sv3,
                'sv1_ratio': sv1_ratio,
                'sv12_ratio': sv12_ratio,
                'rank_energy_2': rank_energy_2,
            })

        feat_df = pd.DataFrame(frame_features)

        out = {}
        for col in feat_df.columns:
            out[f"{sensor_id}_{col}_mean"] = feat_df[col].mean()
            out[f"{sensor_id}_{col}_std"] = feat_df[col].std()
            out[f"{sensor_id}_{col}_min"] = feat_df[col].min()
            out[f"{sensor_id}_{col}_max"] = feat_df[col].max()

        return out

    @staticmethod
    def tof_wavelet_logic(frames: np.ndarray, sensor_id: str, wavelet: str = 'haar') -> dict:
        frame_features = []

        for frame in frames:
            if np.all(~np.isfinite(frame)):
                frame_features.append({
                    'approx_energy': 0.0,
                    'horiz_energy': 0.0,
                    'vert_energy': 0.0,
                    'diag_energy': 0.0,
                    'detail_total_energy': 0.0,
                    'approx_detail_ratio': 0.0,
                })
                continue

            filled = np.nan_to_num(frame, nan=4000.0)
            centered = filled - np.mean(filled)

            cA, (cH, cV, cD) = pywt.dwt2(centered, wavelet)

            approx_energy = np.sum(cA ** 2)
            horiz_energy = np.sum(cH ** 2)
            vert_energy = np.sum(cV ** 2)
            diag_energy = np.sum(cD ** 2)

            detail_total = horiz_energy + vert_energy + diag_energy
            ratio = approx_energy / detail_total if detail_total > 0 else 0.0

            frame_features.append({
                'approx_energy': approx_energy,
                'horiz_energy': horiz_energy,
                'vert_energy': vert_energy,
                'diag_energy': diag_energy,
                'detail_total_energy': detail_total,
                'approx_detail_ratio': ratio,
            })

        feat_df = pd.DataFrame(frame_features)

        out = {}
        for col in feat_df.columns:
            out[f"{sensor_id}_{col}_mean"] = feat_df[col].mean()
            out[f"{sensor_id}_{col}_std"] = feat_df[col].std()
            out[f"{sensor_id}_{col}_min"] = feat_df[col].min()
            out[f"{sensor_id}_{col}_max"] = feat_df[col].max()

        return out

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
    def __init__(self, estimator, extractor, mode=None, target: str = 'gesture', **kwargs):
        self.estimator = estimator
        self.extractor = extractor
        self.mode = mode
        self.target = target

    def _collapse_y_to_sequence(self, X, y):
        # build one label per sequence_id
        gesture_map = (
            y.drop_duplicates(subset='sequence_id')
             .set_index('sequence_id')[self.target]
        )

        y_seq = pd.Series(X.index.map(gesture_map), index=X.index, name=self.target)

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
                .set_index('sequence_id')[self.target]
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


class IndexPreservingPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

    def fit(self, X, y=None):
        self.pca.fit(X)
        return self

    def transform(self, X):
        # Transform to numpy array
        X_transformed = self.pca.transform(X)
        # Return as DataFrame with same index
        return pd.DataFrame(X_transformed, index=X.index)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


def find_data_root(local_dir='data', kaggle_root='/kaggle/input'):
    local_path = Path(local_dir)
    required = [
        'train.csv',
        'test.csv',
        'train_demographics.csv',
        'test_demographics.csv'
    ]

    if local_path.exists() and all((local_path / f).exists() for f in required):
        print(f'Using local data folder: {local_path.resolve()}')
        return local_path

    kaggle_root = Path(kaggle_root)
    if kaggle_root.exists():
        for csv_path in kaggle_root.rglob('train.csv'):
            candidate = csv_path.parent
            if all((candidate / f).exists() for f in required):
                print(f'Using Kaggle data folder: {candidate}')
                return candidate

    raise FileNotFoundError(
        'Could not find the dataset locally or in /kaggle/input. '
        'Place the CSV files in ./data/ or attach the Kaggle dataset.'
    )


class RawSequenceExtractor(BaseEstimator, TransformerMixin):
    """
    Replacement for your old RawSequenceExtractor.
    Outputs raw time-series data (one row per timestep) ready for RNN.
    """
    def __init__(
        self,
        acc_cols=None,
        rot_cols=None,
        rotation_mode="delta_euler",   # "euler", "delta_euler", "quaternion"
        thm_cols=None,
        tof_cols=None,                 # list of all 320 tof_?_v? columns (auto-detected if None)
        tof_mode="raw",                # "raw" (recommended for RNN) or "baseline"
        mask_invalid=-999.0,           # value for invalid ToF readings
    ):
        self.acc_cols = acc_cols
        self.rot_cols = rot_cols
        self.rotation_mode = rotation_mode
        self.thm_cols = thm_cols
        self.tof_cols = tof_cols
        self.tof_mode = tof_mode
        self.mask_invalid = mask_invalid

    def fit(self, X, y=None):
        # Auto-detect ToF columns if not provided
        if self.tof_cols is None:
            self.tof_cols_ = [c for c in X.columns if c.startswith("tof_") and "_v" in c]
        else:
            self.tof_cols_ = self.tof_cols
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        parts = []

        # === 1. Accelerometer (raw) ===
        if self.acc_cols:
            parts.append(X[self.acc_cols].copy())

        # === 2. Rotation (quaternion → euler / delta_euler) ===
        if self.rot_cols:
            rot_df = self._process_rotation(X[self.rot_cols], X["sequence_id"])
            parts.append(rot_df)

        # === 3. Thermopile (raw) ===
        if self.thm_cols:
            parts.append(X[self.thm_cols].copy())

        # === 4. Time-of-Flight (raw or light preprocessing) ===
        if self.tof_cols_:
            tof_df = X[self.tof_cols_].copy()
            tof_df = tof_df.replace(-1.0, self.mask_invalid)   # mark invalids
            if self.tof_mode == "baseline":
                # very light per-sensor mean/std (optional)
                for sensor in range(1, 6):
                    s_cols = [c for c in tof_df.columns if f"tof_{sensor}_" in c]
                    if s_cols:
                        tof_df[f"tof{sensor}_mean"] = tof_df[s_cols].mean(axis=1)
                        tof_df[f"tof{sensor}_std"] = tof_df[s_cols].std(axis=1)
            parts.append(tof_df)

        # Concatenate everything
        result = pd.concat(parts, axis=1)

        # Set index to sequence_id so SequencePadder can group correctly
        result.index = X["sequence_id"].values

        # Ensure column names are clean
        result.columns = [str(c) for c in result.columns]
        return result

    def _process_rotation(self, df: pd.DataFrame, seq_ids: pd.Series) -> pd.DataFrame:
        q = df.astype(float).copy()
        w, x, y, z = q.iloc[:, 0], q.iloc[:, 1], q.iloc[:, 2], q.iloc[:, 3]

        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

        euler = pd.DataFrame({
            "rot_roll": roll.values,
            "rot_pitch": pitch.values,
            "rot_yaw": yaw.values,
        }, index=df.index)

        if self.rotation_mode == "euler":
            return euler
        elif self.rotation_mode == "delta_euler":
            # Delta per sequence (critical!)
            return euler.groupby(seq_ids.values).transform(lambda g: g.diff().fillna(0))
        elif self.rotation_mode == "quaternion":
            return df.rename(columns=dict(zip(df.columns, ["rot_w", "rot_x", "rot_y", "rot_z"])))
        raise ValueError(f"Unknown rotation_mode: {self.rotation_mode}")


# ====================== IMPROVED PADDER ======================
class SequencePadder(BaseEstimator, TransformerMixin):
    """
    Turns the per-timestep DataFrame into 3D tensors for RNN.
    """
    def __init__(self, maxlen=60, padding_value=-999.0, dtype=np.float32):
        self.maxlen = maxlen
        self.padding_value = padding_value
        self.dtype = dtype

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SequencePadder expects a pandas DataFrame.")
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X):
        check_is_fitted(self, ["n_features_in_"])

        grouped = list(X.groupby(level=0, sort=False))  # group by sequence_id
        n_seq = len(grouped)

        X_pad = np.full(
            (n_seq, self.maxlen, self.n_features_in_),
            fill_value=self.padding_value,
            dtype=self.dtype,
        )

        sequence_ids = []
        lengths = []

        for i, (seq_id, grp) in enumerate(grouped):
            arr = grp.to_numpy(dtype=self.dtype, copy=True)
            seq_len = min(len(arr), self.maxlen)
            X_pad[i, :seq_len, :] = arr[:seq_len]
            sequence_ids.append(seq_id)
            lengths.append(len(arr))

        return {
            "X": X_pad,                          # (n_seq, maxlen, n_features)
            "sequence_ids": np.array(sequence_ids),
            "lengths": np.array(lengths),
        }


# ====================== ENHANCED RNN CLASSIFIER ======================
class KerasRNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        rnn_type="lstm",           # lstm, gru, rnn
        rnn_units=(128, 64),
        dense_units=(64,),
        dropout=0.2,
        bidirectional=True,        # ← big boost for gestures
        learning_rate=5e-4,
        batch_size=16,
        epochs=150,
        patience=20,
        class_weight_mode="balanced",   # ← fixes your majority-class problem
        verbose=0,
        random_state=42,
        mask_value=-999.0,
    ):
        self.rnn_type = rnn_type
        self.rnn_units = rnn_units
        self.dense_units = dense_units
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.class_weight_mode = class_weight_mode
        self.verbose = verbose
        self.random_state = random_state
        self.mask_value = mask_value

    def _extract_array(self, X):
        return X["X"] if isinstance(X, dict) else X

    def _get_rnn_layer(self):
        return {"lstm": layers.LSTM, "gru": layers.GRU, "rnn": layers.SimpleRNN}[self.rnn_type.lower()]

    def _build_model(self, input_shape, n_classes):
        RNNLayer = self._get_rnn_layer()
        reg = keras.regularizers.l2(1e-4)

        inputs = keras.Input(shape=input_shape)
        x = layers.Masking(mask_value=self.mask_value)(inputs)

        units_list = list(self.rnn_units)
        for i, units in enumerate(units_list):
            return_seq = i < len(units_list) - 1
            layer = RNNLayer(units, return_sequences=return_seq, dropout=self.dropout,
                             kernel_regularizer=reg)
            if self.bidirectional:
                x = layers.Bidirectional(layer)(x)
            else:
                x = layer(x)

        for units in list(self.dense_units):
            x = layers.Dense(units, activation="relu", kernel_regularizer=reg)(x)
            x = layers.Dropout(self.dropout)(x)

        outputs = layers.Dense(n_classes, activation="softmax")(x)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, X, y):
        X_arr = self._extract_array(X)
        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(pd.Series(y))
        self.classes_ = self.label_encoder_.classes_

        self.model_ = self._build_model((X_arr.shape[1], X_arr.shape[2]), len(self.classes_))

        callbacks = [keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=self.patience, restore_best_weights=True
        )]

        class_weight = None
        if self.class_weight_mode == "balanced":
            weights = compute_class_weight("balanced", classes=np.unique(y_enc), y=y_enc)
            class_weight = dict(enumerate(weights))

        self.history_ = self.model_.fit(
            X_arr, y_enc,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.15,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=self.verbose,
            shuffle=True,
        )
        return self

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.label_encoder_.inverse_transform(np.argmax(proba, axis=1))

    def predict_proba(self, X):
        X_arr = self._extract_array(X)
        return self.model_.predict(X_arr, verbose=0)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


class ManyToOneWrapperRNN(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, target="gesture_action"):
        self.estimator = estimator
        self.target = target

    def _collapse_y(self, X, y):
        """Convert sequence_ids to Series and handle label mapping correctly."""
        # Get sequence IDs from X (could be dict from SequencePadder or DataFrame)
        if isinstance(X, dict):
            seq_ids = X["sequence_ids"]
        else:
            seq_ids = X.index

        # Convert to pandas Series if it's a numpy array
        if not isinstance(seq_ids, pd.Series):
            seq_ids = pd.Series(seq_ids, name='sequence_id')

        # IMPORTANT: y is a DataFrame with 'sequence_id' and target columns
        # Create mapping from sequence_id to label
        target_map = y.drop_duplicates('sequence_id').set_index('sequence_id')[self.target]

        # Map and ensure we have a valid Series
        y_seq = seq_ids.map(target_map)

        # Check for missing values
        if y_seq.isna().any():
            missing_seq_ids = seq_ids[y_seq.isna()].tolist()
            print(f"Warning: Missing labels for {len(missing_seq_ids)} sequences")
            # Drop missing values
            valid_mask = ~y_seq.isna()
            return y_seq[valid_mask]

        return y_seq

    def fit(self, X, y):
        y_seq = self._collapse_y(X, y)

        # CRITICAL: Remove any NaN labels
        if hasattr(y_seq, 'isna') and y_seq.isna().any():
            print(f"Dropping {y_seq.isna().sum()} NaN labels")
            valid_mask = ~y_seq.isna()
            # Also need to filter X - but this is tricky with dict input
            # For now, let's just check

        # Clone and fit
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y_seq)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)