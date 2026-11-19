# References, Open-Source Libraries & License Audit (`REFERENCES.md`)

All open-source Python packages, academic papers, and public repositories referenced in `cmi_dexter` v4, with verified URLs and license status.

---

## 1. Installed Open-Source Libraries Used Directly (`requirements-features.txt`)

| Library | Pinned Version (`ds_kaggle`) | License | Official Docs / Repository | Role in `cmi_dexter` |
|---|---|---|---|---|
| **NumPy** | `2.2.6` | BSD-3-Clause | https://numpy.org/doc/stable/ | Core array math, `np.trapezoid`, FFT/rFFT |
| **Pandas** | `2.3.3` | BSD-3-Clause | https://pandas.pydata.org/docs/ | Tabular sequence DataFrames & metadata |
| **SciPy** | `1.15.2` | BSD-3-Clause | https://docs.scipy.org/doc/scipy/ | `spatial.transform.Rotation`, `signal` (`butter`, `sosfiltfilt`, `resample_poly`, `periodogram`, `welch`, `coherence`, `stft`, `correlate`, `find_peaks`, `medfilt`, `savgol_filter`), `ndimage` (`gaussian_filter1d`, `label`, `sobel`, `median_filter`), `stats` (`linregress`, `spearmanr`, `entropy`, `kurtosis`, `skew`, `trim_mean`) |
| **scikit-learn** | `1.7.2` | BSD-3-Clause | https://scikit-learn.org/stable/ | `BaseEstimator`, `Pipeline`, `GroupKFold`, `GroupShuffleSplit`, `f_classif`, `f1_score` |
| **PyTorch** | `2.14.1` | BSD-3-Clause | https://pytorch.org/docs/stable/ | Multimodal neural network & 2D ToF CNN training |
| **tsai** | `1.0.1` | Apache-2.0 | https://github.com/timeseriesAI/tsai | Sequence backbones (`InceptionTimePlus`, `ResNetPlus`, `FCNPlus`, `TCN`, `TSTPlus`, `GRUPlus`, `LSTMPlus`, `XCMPlus`) |
| **TSFEL** | `0.2.0` | BSD-3-Clause | https://github.com/fraunhoferportugal/tsfel · https://tsfel.readthedocs.io | Authentic `statistical`, `temporal`, `spectral`, and `fractal` time-series feature extraction |
| **tsfresh** | `0.21.2` | MIT | https://github.com/blue-yonder/tsfresh · https://tsfresh.readthedocs.io | Authentic batched statistical, spectral, autocorrelation, peak, and `approximate_entropy` extraction |
| **pycatch22** | `1.0.2` | GPL-3.0 (runtime optional plugin) | https://github.com/DynamicsAndNeuralSystems/pycatch22 | Fast C-backed 24 canonical time-series characteristics (`catch24=True`) & DFA/AMI complexity |
| **tsflex** | `0.4.1` | MIT | https://github.com/predict-idlab/tsflex · https://predict-idlab.github.io/tsflex | Multi-rate strided `FeatureCollection`, multi-scale window summaries & actigraphy rolling bank |
| **seglearn** | `1.2.5` | BSD-3-Clause | https://github.com/dmbee/seglearn | Lightweight segment-level `FeatureRep` statistical baseline backend |
| **sktime** | `0.40.1` | BSD-3-Clause | https://github.com/sktime/sktime · https://www.sktime.net | Modular `SummaryTransformer`, `Catch22`, and `MiniRocket` transformer backend |
| **PyWavelets** | `1.8.0` | MIT | https://pywavelets.readthedocs.io | CWT (`pywt.cwt`), DWT (`wavedec`/`waverec`), WaveletPacket energy entropy, VisuShrink denoising |
| **scikit-optimize** | `0.10.2` | BSD-3-Clause | https://scikit-optimize.github.io | Bayesian hyperparameter search (`Optimizer`, `Categorical`, `Integer`, `Real`) |
| **tsaug** | `0.2.1` | Apache-2.0 (**Archived upstream**) | https://github.com/arundo/tsaug | Isolated inside `TsaugAdapter` (`src/base_utils.py`) for time-series augmentation with NumPy fallback |

---

## 2. Reference Repositories License Audit

| Repository | URL | Verified License | Usage Policy in `cmi_dexter` |
|---|---|---|---|
| **MaD Lab FAU `sleep_analysis`** | https://github.com/mad-lab-fau/sleep_analysis | **MIT** (Copyright 2021 MaD Lab, FAU Erlangen-Nuremberg) | Concept + multi-scale rolling actigraphy bank adapted via `tsflex` with attribution in `TSFlexBackend` (`src/base_utils.py`); upstream off-by-one (`np.arange(1, 20)` omitting scale 20) fixed via `include_max_window=True`. |
| **OpenStrap `edge` & `analytics`** | https://github.com/OpenStrap/edge · https://github.com/OpenStrap/analytics | **AGPL-3.0** | **Zero code copied.** Clean-room conceptual inspiration only (device-side motion envelope / streaming framing concepts). |
| **Joffcom `my-whoop`** | https://github.com/Joffcom/my-whoop | **No LICENSE file** (All Rights Reserved by default) | **Zero code copied.** Conceptual workflow / metric schema inspection only. |
| **patrickloeber `whoop-analyzer`** | https://github.com/patrickloeber/whoop-analyzer | **No LICENSE file** (All Rights Reserved by default) | **Zero code copied.** Conceptual exploratory analysis inspection only. |
| **A1fred00-datascience `WHOOP`** | https://github.com/A1fred00-datascience/WHOOP | **No LICENSE file** (All Rights Reserved by default) | **Zero code copied.** Inspected for wearable ML pitfalls; explicitly avoided its pre-split scaler leakage pattern. |

---

## 3. Academic & Algorithmic Papers Cited

1. **Barandas, M. et al. (2020)**. *TSFEL: Time Series Feature Extraction Library*. SoftwareX, 11, 100456. https://doi.org/10.1016/j.softx.2020.100456
2. **Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018)**. *Time Series FeatuRe Extraction on basis of Scalable Hypothesis tests (tsfresh – A Python package)*. Neurocomputing, 307, 72–77. https://doi.org/10.1016/j.neucom.2018.03.067
3. **Lubba, C. H. et al. (2019)**. *catch22: CAnonical Time-series CHaracteristics*. Data Mining and Knowledge Discovery, 33(6), 1821–1852. https://doi.org/10.1007/s10618-019-00647-x
4. **Van Der Donckt, J. et al. (2022)**. *tsflex: Flexible time series processing & feature extraction*. SoftwareX, 17, 100971. https://doi.org/10.1016/j.softx.2021.100971
5. **Burns, D. M., & Whyne, C. M. (2018)**. *Seglearn: A Python Package for Learning Sequences and Time Series*. JMLR, 19(83), 1–7. https://jmlr.org/papers/v19/18-160.html
6. **Löning, M. et al. (2019)**. *sktime: A Unified Interface for Machine Learning with Time Series*. NeurIPS Workshop on Systems for ML.
7. **Dempster, A., Schmidt, D. F., & Webb, G. I. (2021)**. *MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series Classification*. KDD 2021.
8. **Zhou, Y. et al. (2019)**. *On the Continuity of Rotation Representations in Neural Networks* (6D rotation representation). CVPR 2019.
9. **Walch, O. et al. (2019)**. *Sleep stage prediction with raw acceleration and photoplethysmography heart rate data derived from a consumer wearable device* (multi-scale Difference-of-Gaussians). Sleep, 42(12), zsz180. https://doi.org/10.1093/sleep/zsz180
10. **Baevsky, R. M., & Berseneva, A. P. (2008)**. *Methodical recommendations: Use of Kardivar system for determination of the stress level and estimation of the body adaptability* (Baevsky Stress Index AMo%, MxDMn, SI).
11. **Straczkiewicz, M. et al. (2023)**. *A systematic review of smartphone-based human activity recognition methods for health research* (calibration-invariant local-baseline dynamic amplitude concepts). npj Digital Medicine. https://doi.org/10.1038/s41746-022-00745-z
12. **Hjorth, B. (1970)**. *EEG analysis based on time domain properties* (Activity, Mobility, Complexity). Electroencephalography and Clinical Neurophysiology, 29(3), 306–310.
13. **Shannon, C. E. (1948)**. *A Mathematical Theory of Communication*. Bell System Technical Journal, 27(3), 379–423.
14. **Donoho, D. L., & Johnstone, I. M. (1994)**. *Ideal spatial adaptation by wavelet shrinkage* (VisuShrink universal threshold). Biometrika, 81(3), 425–455.
15. **Krauss, S. et al. (2025)**. *MaD Lab FAU `sleep_analysis` actigraphy feature engineering benchmark*. https://github.com/mad-lab-fau/sleep_analysis
