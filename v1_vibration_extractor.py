import numpy as np
import pandas as pd
from scipy import signal
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Dict, Tuple, Union, Optional

def compute_fft_spectrum(signal_data: np.ndarray, sampling_rate: float) -> pd.Series:
    """
    Computes the single-sided amplitude spectrum of a 1D signal.
    
    Args:
        signal_data (np.ndarray): 1D array of signal values.
        sampling_rate (float): Sampling rate in Hz.
        
    Returns:
        pd.Series: Spectrum with frequency as index and amplitude as values.
    """
    n = len(signal_data)
    fft_vals = np.fft.rfft(signal_data)
    fft_freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
    amplitude = np.abs(fft_vals) / n
    return pd.Series(amplitude, index=fft_freqs, name='amplitude')

def get_moments_auc(u: np.ndarray, v: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Calculates the statistical moments and Area Under Curve (AUC) of a distribution.
    
    Args:
        u (np.ndarray): Independent variable (e.g., frequency).
        v (np.ndarray): Dependent variable (e.g., amplitude/power), will be min-max scaled.
        
    Returns:
        Tuple[float, float, float, float, float]: mean, std, skew, kurtosis, auc.
    """
    v_scaled = (v - np.min(v)) / (np.max(v) - np.min(v) + 1e-10)
    w = v_scaled
    z = u
    
    sum_w = np.sum(w)
    if sum_w == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
        
    mean = np.sum(w * z) / sum_w
    variance = np.sum(w * (z - mean)**2) / sum_w
    std = np.sqrt(variance)
    skew = np.sum(w * (z - mean)**3) / (sum_w * (std**3 + 1e-10))
    kurtosis = np.sum(w * (z - mean)**4) / (sum_w * (std**4 + 1e-10))
    auc = np.sum(w)
    
    return float(mean), float(std), float(skew), float(kurtosis), float(auc)

def get_freq_bands_cut_points(spectrum: pd.Series, max_no_peaks: int = 10, peaks_prominence: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identifies peak frequencies in a spectrum and computes cut points for frequency bands.
    
    Args:
        spectrum (pd.Series): Spectrum data with frequency as index.
        max_no_peaks (int): Maximum number of peaks to consider.
        peaks_prominence (float): Minimum prominence for a peak to be considered.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Array of cut points and array of peak frequencies.
    """
    idx_peaks, _ = signal.find_peaks(spectrum.values, prominence=peaks_prominence)
    freq_peaks = spectrum.iloc[idx_peaks].index.values
    
    if len(freq_peaks) > max_no_peaks:
        peak_amps = spectrum.iloc[idx_peaks].values
        top_indices = np.argsort(peak_amps)[-max_no_peaks:]
        freq_peaks = np.sort(freq_peaks[top_indices])
        
    if len(freq_peaks) < 2:
        return np.array([spectrum.index.min(), spectrum.index.max()]), freq_peaks
        
    between_peaks = np.abs(np.diff(freq_peaks)) / 2.0
    midpoints = freq_peaks[:-1] + between_peaks
    
    cut_points = np.concatenate([
        [spectrum.index.min()],
        midpoints,
        [spectrum.index.max()]
    ])
    
    return cut_points, freq_peaks

def extract_band_features(spectrum: pd.Series, cut_points: np.ndarray, band_prefix: str) -> pd.Series:
    """
    Extracts peak and distributional features for each frequency band.
    
    Args:
        spectrum (pd.Series): Spectrum data.
        cut_points (np.ndarray): Array of frequency band boundaries.
        band_prefix (str): Prefix for feature names (e.g., 'acc_x').
        
    Returns:
        pd.Series: Flattened series of extracted features.
    """
    features = {}
    spectrum_reset = spectrum.reset_index()
    spectrum_reset.columns = ['freq', 'power']
    
    spectrum_reset['band'] = pd.cut(spectrum_reset['freq'], bins=cut_points, include_lowest=True)
    
    for i, (band_interval, group) in enumerate(spectrum_reset.groupby('band', observed=True)):
        band_name = f"{band_prefix}_b{i+1}"
        
        idx_max = group['power'].idxmax()
        features[f"{band_name}_peak_freq"] = float(group.loc[idx_max, 'freq'])
        features[f"{band_name}_peak_power"] = float(group.loc[idx_max, 'power'])
        
        mean, std, skew, kurt, auc = get_moments_auc(group['freq'].values, group['power'].values)
        features[f"{band_name}_mean"] = mean
        features[f"{band_name}_std"] = std
        features[f"{band_name}_skew"] = skew
        features[f"{band_name}_kurt"] = kurt
        features[f"{band_name}_auc"] = auc
        
    return pd.Series(features)

class SensorFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible feature extractor for IMU, ToF, and Thermo sensor data.
    Inspired by spectral feature extraction pipelines (e.g., PFSpectraFeatExtr).
    """
    def __init__(
        self,
        sampling_rate: float = 100.0,
        max_no_peaks: int = 10,
        peaks_prominence: float = 1e-4,
        imu_channels: List[str] = ['acc_x', 'acc_y', 'acc_z', 'rot_x', 'rot_y', 'rot_z'],
        tof_channels: List[str] = ['tof'],
        thermo_channels: List[str] = ['thermo'],
        apply_spectral_to_low_freq: bool = False
    ):
        self.sampling_rate = sampling_rate
        self.max_no_peaks = max_no_peaks
        self.peaks_prominence = peaks_prominence
        self.imu_channels = imu_channels
        self.tof_channels = tof_channels
        self.thermo_channels = thermo_channels
        self.apply_spectral_to_low_freq = apply_spectral_to_low_freq
        
        self.cut_points_ = {}
        self.peaks_ = {}
        self.feature_names_out_ = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'SensorFeatureExtractor':
        """
        Fits the extractor by determining global frequency band cut points for each channel.
        """
        all_channels = self.imu_channels + self.tof_channels + self.thermo_channels
        
        for ch in all_channels:
            if ch not in X.columns:
                continue
            if 'sequence_id' in X.columns:
                agg_signal = X.groupby('sequence_id')[ch].apply(lambda x: np.mean(np.abs(np.fft.rfft(x))) / len(x)).values
            else:
                agg_signal = X[ch].values
                
            spectrum = compute_fft_spectrum(agg_signal, self.sampling_rate)
            cut_points, freq_peaks = get_freq_bands_cut_points(
                spectrum, max_no_peaks=self.max_no_peaks, peaks_prominence=self.peaks_prominence
            )
            self.cut_points_[ch] = cut_points
            self.peaks_[ch] = freq_peaks
            
        self._generate_feature_names()
        return self

    def _generate_feature_names(self):
        """Generates the list of expected output feature names."""
        names = []
        low_freq_channels = (self.tof_channels if self.apply_spectral_to_low_freq else []) + \
                            (self.thermo_channels if self.apply_spectral_to_low_freq else [])
        all_spectral_channels = self.imu_channels + low_freq_channels
        
        for ch in all_spectral_channels:
            if ch in self.cut_points_:
                n_bands = len(self.cut_points_[ch]) - 1
                for i in range(1, n_bands + 1):
                    names.extend([
                        f"{ch}_b{i}_peak_freq", f"{ch}_b{i}_peak_power",
                        f"{ch}_b{i}_mean", f"{ch}_b{i}_std", 
                        f"{ch}_b{i}_skew", f"{ch}_b{i}_kurt", f"{ch}_b{i}_auc"
                    ])
                    
        if not self.apply_spectral_to_low_freq:
            for ch in self.tof_channels + self.thermo_channels:
                names.extend([f"{ch}_mean", f"{ch}_std", f"{ch}_min", f"{ch}_max", f"{ch}_trend"])
                
        self.feature_names_out_ = names

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms input data into a flat feature matrix.
        """
        if 'sequence_id' not in X.columns:
            raise ValueError("Input DataFrame must contain a 'sequence_id' column for segmentation.")
            
        transformed_data = []
        all_channels = self.imu_channels + self.tof_channels + self.thermo_channels
        
        for seq_id, group in X.groupby('sequence_id'):
            seq_features = {}
            
            for ch in all_channels:
                if ch not in group.columns or ch not in self.cut_points_:
                    continue
                    
                signal_data = group[ch].dropna().values
                if len(signal_data) < 10:
                    continue
                    
                if ch in self.imu_channels or self.apply_spectral_to_low_freq:
                    spectrum = compute_fft_spectrum(signal_data, self.sampling_rate)
                    band_feats = extract_band_features(spectrum, self.cut_points_[ch], band_prefix=ch)
                    seq_features.update(band_feats.to_dict())
                else:
                    seq_features[f"{ch}_mean"] = float(np.mean(signal_data))
                    seq_features[f"{ch}_std"] = float(np.std(signal_data))
                    seq_features[f"{ch}_min"] = float(np.min(signal_data))
                    seq_features[f"{ch}_max"] = float(np.max(signal_data))
                    if len(signal_data) > 1:
                        trend = np.polyfit(np.arange(len(signal_data)), signal_data, 1)[0]
                        seq_features[f"{ch}_trend"] = float(trend)
                    else:
                        seq_features[f"{ch}_trend"] = 0.0
                        
            seq_features['sequence_id'] = seq_id
            transformed_data.append(seq_features)
            
        out_df = pd.DataFrame(transformed_data)
        for col in self.feature_names_out_:
            if col not in out_df.columns:
                out_df[col] = 0.0
                
        return out_df[['sequence_id'] + self.feature_names_out_]