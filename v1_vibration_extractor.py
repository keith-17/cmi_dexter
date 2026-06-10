import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Any
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.ndimage import center_of_mass
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


def get_moments_auc(u: np.ndarray, v: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Calculates the distributional moments and Area Under Curve (AUC) of a spectrum.
    Inspired by pumpflow package spectral analysis.
    
    Parameters
    ----------
    u : np.ndarray
        Frequency values (x-axis).
    v : np.ndarray
        Magnitude/Power values (y-axis).
        
    Returns
    -------
    Tuple[float, float, float, float, float]
        (mean, std, skew, kurtosis, auc)
    """
    v_scaled = (v - np.min(v)) / (np.max(v) - np.min(v) + 1e-10)
    w = v_scaled.reshape(-1)
    z = u.reshape(-1)
    
    total_w = np.sum(w) + 1e-10
    mean = np.sum(w * z) / total_w
    variance = np.sum(w * (z - mean) ** 2) / total_w
    std = np.sqrt(variance) + 1e-10
    skew = np.sum(w * (z - mean) ** 3) / (total_w * (std ** 3))
    kurtosis = np.sum(w * (z - mean) ** 4) / (total_w * (std ** 4))
    auc = np.sum(w)
    
    return float(mean), float(std), float(skew), float(kurtosis), float(auc)


def get_freq_bands_cut_points(spectrum: pd.Series, max_no_peaks: int = 10, prominence: float = 1e-4) -> np.ndarray:
    """
    Obtains frequency band cut points from peaks in the aggregated spectrum.
    
    Parameters
    ----------
    spectrum : pd.Series
        Spectrum with frequency as index and magnitude as values.
    max_no_peaks : int
        Maximum number of peaks to consider.
    prominence : float
        Prominence threshold for peak detection.
        
    Returns
    -------
    np.ndarray
        Array of frequency cut points for binning.
    """
    idx_peaks, _ = find_peaks(spectrum.values, prominence=prominence)
    freq_peaks = spectrum.iloc[idx_peaks].index.values
    
    if len(freq_peaks) > max_no_peaks:
        # Sort by magnitude and take top N
        peak_mags = spectrum.iloc[idx_peaks].values
        top_idx = np.argsort(peak_mags)[-max_no_peaks:]
        freq_peaks = np.sort(freq_peaks[top_idx])
        
    if len(freq_peaks) < 2:
        return np.array([0.0, spectrum.index.max()])
        
    between_peaks = np.abs(np.diff(freq_peaks)) / 2.0
    midpoints = freq_peaks[:-1] + between_peaks
    
    cut_points = np.concatenate([[0.0], midpoints, [spectrum.index.max()]])
    return np.unique(cut_points)


def extract_spectral_band_features(signal: np.ndarray, sampling_rate: float, axis_name: str) -> Dict[str, float]:
    """
    Extracts peak coordinates and distributional/AUC features per frequency band for a 1D signal.
    
    Parameters
    ----------
    signal : np.ndarray
        1D time-series signal.
    sampling_rate : float
        Sampling rate in Hz.
    axis_name : str
        Name of the sensor axis (e.g., 'acc_x').
        
    Returns
    -------
    Dict[str, float]
        Dictionary of extracted spectral features.
    """
    features = {}
    n = len(signal)
    if n < 4:
        return features
        
    fft_vals = np.abs(fft(signal))[:n // 2]
    freqs = fftfreq(n, d=1.0 / sampling_rate)[:n // 2]
    
    spectrum = pd.Series(fft_vals, index=freqs)
    cut_points = get_freq_bands_cut_points(spectrum, max_no_peaks=8, prominence=1e-4)
    
    # Global peak
    peak_idx = np.argmax(fft_vals)
    features[f'{axis_name}_peak_freq'] = float(freqs[peak_idx])
    features[f'{axis_name}_peak_mag'] = float(fft_vals[peak_idx])
    features[f'{axis_name}_total_energy'] = float(np.sum(fft_vals ** 2))
    
    # Band features
    for i in range(len(cut_points) - 1):
        low, high = cut_points[i], cut_points[i + 1]
        mask = (freqs >= low) & (freqs < high)
        if np.any(mask):
            band_freqs = freqs[mask]
            band_mags = fft_vals[mask]
            mean, std, skew, kurt, auc = get_moments_auc(band_freqs, band_mags)
            
            features[f'{axis_name}_b{i+1}_mean'] = mean
            features[f'{axis_name}_b{i+1}_std'] = std
            features[f'{axis_name}_b{i+1}_skew'] = skew
            features[f'{axis_name}_b{i+1}_kurt'] = kurt
            features[f'{axis_name}_b{i+1}_auc'] = auc
            
    return features


def extract_thermopile_spatial_features(thm_df: pd.DataFrame) -> Dict[str, float]:
    """
    Extracts spatial and temporal statistics from thermopile sensors.
    
    Parameters
    ----------
    thm_df : pd.DataFrame
        DataFrame containing thermopile sensor columns.
        
    Returns
    -------
    Dict[str, float]
        Dictionary of thermopile features.
    """
    features = {}
    all_values = thm_df.values.flatten()
    all_values = all_values[np.isfinite(all_values)]
    
    if len(all_values) == 0:
        return {f'thm_all_{stat}': 0.0 for stat in ['mean', 'std', 'min', 'max']}
        
    features['thm_all_mean'] = float(np.mean(all_values))
    features['thm_all_std'] = float(np.std(all_values))
    features['thm_all_min'] = float(np.min(all_values))
    features['thm_all_max'] = float(np.max(all_values))
    
    # Per-sensor stats
    for col in thm_df.columns:
        vals = thm_df[col].dropna().values
        if len(vals) > 0:
            features[f'{col}_mean'] = float(np.mean(vals))
            features[f'{col}_std'] = float(np.std(vals))
            features[f'{col}_diff_mean'] = float(np.mean(np.diff(vals))) if len(vals) > 1 else 0.0
            
    # Spatial asymmetry (assuming ordered sensors)
    sensor_means = thm_df.mean(axis=0).values
    if len(sensor_means) >= 2:
        mid = len(sensor_means) // 2
        features['thm_left_right_diff'] = float(np.mean(sensor_means[:mid]) - np.mean(sensor_means[mid:]))
        features['thm_sensor_mean_std'] = float(np.std(sensor_means))
        
    return features


def extract_tof_spatial_features(tof_df: pd.DataFrame) -> Dict[str, float]:
    """
    Extracts spatial features from Time-of-Flight (ToF) data, treating 64 values as an 8x8 grid.
    
    Parameters
    ----------
    tof_df : pd.DataFrame
        DataFrame containing ToF sensor columns (e.g., tof_1_v0 to tof_1_v63).
        
    Returns
    -------
    Dict[str, float]
        Dictionary of ToF spatial features.
    """
    features = {}
    sensor_prefixes = sorted(list(set([c.split('_v')[0] for c in tof_df.columns if '_v' in c])))
    
    for sensor in sensor_prefixes:
        s_cols = [c for c in tof_df.columns if c.startswith(f"{sensor}_v")]
        s_cols = sorted(s_cols, key=lambda x: int(x.split('_v')[-1]))
        
        if len(s_cols) != 64:
            continue
            
        # Aggregate over time for stability, or take mean frame
        mean_frame = tof_df[s_cols].mean(axis=0).values.astype(float)
        mean_frame = np.where((mean_frame <= 0) | (mean_frame >= 4000), np.nan, mean_frame)
        
        if np.all(np.isnan(mean_frame)):
            continue
            
        frame = mean_frame.reshape(8, 8)
        valid_mask = np.isfinite(frame)
        valid_ratio = np.sum(valid_mask) / 64.0
        
        if valid_ratio > 0:
            median_val = np.nanmedian(frame[valid_mask])
            frame = np.nan_to_num(frame, nan=median_val)
        else:
            frame = np.zeros((8, 8))
            
        features[f'{sensor}_valid_ratio'] = float(valid_ratio)
        features[f'{sensor}_mean'] = float(np.mean(frame))
        features[f'{sensor}_std'] = float(np.std(frame))
        
        # Quadrants
        features[f'{sensor}_q1_mean'] = float(np.mean(frame[:4, :4]))
        features[f'{sensor}_q2_mean'] = float(np.mean(frame[:4, 4:]))
        features[f'{sensor}_q3_mean'] = float(np.mean(frame[4:, :4]))
        features[f'{sensor}_q4_mean'] = float(np.mean(frame[4:, 4:]))
        
        # Center vs Edge
        center = frame[2:6, 2:6]
        edge = np.concatenate([frame[:2, :], frame[6:, :], frame[:, :2], frame[:, 6:]])
        features[f'{sensor}_center_mean'] = float(np.mean(center))
        features[f'{sensor}_edge_mean'] = float(np.mean(edge))
        
        # Center of Mass
        weights = np.maximum(0, 4000.0 - frame)
        if np.sum(weights) > 0:
            com_r, com_c = center_of_mass(weights)
            features[f'{sensor}_com_r'] = float(com_r)
            features[f'{sensor}_com_c'] = float(com_c)
        else:
            features[f'{sensor}_com_r'] = 4.0
            features[f'{sensor}_com_c'] = 4.0
            
    return features


class V1CargoExtractor(BaseEstimator, TransformerMixin):
    """
    Tabular feature extractor for cargo behavior data (IMU, Rotation, Thermopile, ToF).
    Heavily inspired by spectral moment extraction and spatial grid analysis.
    """
    def __init__(
        self,
        imu_cols: Optional[List[str]] = None,
        rot_cols: Optional[List[str]] = None,
        thm_cols: Optional[List[str]] = None,
        tof_cols: Optional[List[str]] = None,
        sampling_rate: float = 100.0,
        disable_tqdm: bool = True
    ):
        self.imu_cols = imu_cols or ['acc_x', 'acc_y', 'acc_z']
        self.rot_cols = rot_cols or ['rot_w', 'rot_x', 'rot_y', 'rot_z']
        self.thm_cols = thm_cols or ['thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5']
        self.tof_cols = tof_cols
        self.sampling_rate = sampling_rate
        self.disable_tqdm = disable_tqdm

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> 'V1CargoExtractor':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        sequence_list = []
        sequence_ids = X['sequence_id'].unique()
        
        for seq_id in sequence_ids:
            seq_df = X[X['sequence_id'] == seq_id]
            record = {'sequence_id': seq_id}
            
            # 1. IMU Spectral Features
            if self.imu_cols:
                imu_df = seq_df[self.imu_cols].dropna()
                if not imu_df.empty:
                    for col in self.imu_cols:
                        record.update(extract_spectral_band_features(
                            imu_df[col].values, self.sampling_rate, col
                        ))
                        
            # 2. Rotation Spectral Features (on delta/derivative)
            if self.rot_cols:
                rot_df = seq_df[self.rot_cols].dropna()
                if not rot_df.empty:
                    # Convert quaternion to euler or just use delta of existing
                    rot_delta = rot_df.diff().fillna(0.0)
                    for col in rot_delta.columns:
                        record.update(extract_spectral_band_features(
                            rot_delta[col].values, self.sampling_rate, f"rot_{col}"
                        ))
                        
            # 3. Thermopile Spatial Features
            thm_available = [c for c in self.thm_cols if c in seq_df.columns]
            if thm_available:
                thm_df = seq_df[thm_available].dropna()
                if not thm_df.empty:
                    record.update(extract_thermopile_spatial_features(thm_df))
                    
            # 4. ToF Spatial Features
            if self.tof_cols:
                tof_available = [c for c in self.tof_cols if c in seq_df.columns]
                if tof_available:
                    tof_df = seq_df[tof_available].dropna()
                    if not tof_df.empty:
                        record.update(extract_tof_spatial_features(tof_df))
                        
            # 5. Metadata / Category
            for cat_col in ['subject', 'gesture', 'sequence_type']:
                if cat_col in seq_df.columns:
                    record[cat_col] = seq_df[cat_col].iloc[0]
                    
            sequence_list.append(record)
            
        final_df = pd.DataFrame(sequence_list)
        final_df = final_df.set_index('sequence_id').fillna(0.0)
        return final_df