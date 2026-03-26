import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

def calculate_fft(array_values: np.ndarray) -> np.ndarray:
    """
    Calculate the Fast Fourier Transform (FFT) of an array of values.

    Parameters
    ----------
    array_values : numpy.ndarray
        The array of values to transform.

    Returns
    -------
    numpy.ndarray
        The absolute values of the FFT of the input array, truncated to half the length of the input array.
    """
    return abs(fft(array_values))[0:len(array_values) // 2]

def convert_frame_to_fft(df: pd.DataFrame, sampling_rate: int, sample_points_N: int = None) -> pd.Series:
    """
    Convert a pandas DataFrame into a pandas Series containing the Fast Fourier Transform (FFT) of the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to transform.
    sampling_rate : int
        The sampling rate of the DataFrame.
    sample_points_N : int, optional
        The number of sample points to include in the FFT. If None, the number of sample points is equal to the length of the DataFrame.

    Returns
    -------
    pandas.Series
        A pandas Series containing the absolute values of the FFT of the input DataFrame, truncated to half the length of the input DataFrame.
    """
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
                 imu_sensor_list=None, # Default values
                 sampling_rate=100,
                 domain='acceleration',
                 dc_offset=2.0,
                 band_edges=None
                 ):
        if imu_sensor_list is None:
            imu_sensor_list = ['acc_x', 'acc_y', 'acc_z']
        self.imu_sensor_list = imu_sensor_list
        self.sampling_rate = sampling_rate
        self.domain = domain
        self.dc_offset = dc_offset
        self.band_edges = band_edges

    def fit(self, X, y=None):
        self.transform(X)
        return self

    def transform(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        sequence_list = []
        for a_sequence in raw_data['sequence_id'].unique():
            single_sequence_df = raw_data[raw_data['sequence_id'] == a_sequence]
            imu_df = self.process_for_imu_values(single_sequence_df)
            if self.domain == 'time':
                imu_features_df = self.extract_time_features(imu_df)
            else:
                imu_features_df = self.extract_features_from_imu(imu_df, self.band_edges)
            category_df = self.return_single_category_desc_record(raw_data)
            temp_feat_df = imu_features_df.join(category_df)
            sequence_list.append(temp_feat_df)

        final_df = pd.concat(sequence_list).reset_index(drop=True)
        sequence_list.clear()
        return final_df

    def return_single_category_desc_record(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby('sequence_id')[['orientation', 'subject', 'gesture']].agg(lambda x: x.unique()[0]).reset_index()

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
    def convert_frame_to_fft(df: pd.DataFrame, sampling_rate: int, sample_points_N: int = None) -> pd.Series:
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
