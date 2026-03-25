import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq

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