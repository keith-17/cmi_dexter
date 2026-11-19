"""utils.py -- child multi-branch PyTorch & unified search layer built on ``base_utils``.

Inherits all Honeycomb parent classes, shared physics, pluggable domain feature
cells, real open-source feature backends, modular scientific signal descriptors,
augmentation adapter, and evolutionary descriptors directly from ``base_utils``:

  - ``MultiModalSequenceExtractor(BaseSequenceExtractor)``  (child of ``HoneycombBase``)
  - ``TSAugmentor(BaseAugmentor)``                          (child of ``HoneycombBase``)
  - ``MultiBranchTSAIClassifier(BaseSequenceClassifier)``   (PyTorch multi-branch neural child)
  - ``MultiBranchGASearchCV(BaseGASearchCV)``               (aliased as ``GASearchCV``)
  - ``HierarchicalBayesSearchCV(BaseSearchCV)``             (Bayesian search child)
"""
from __future__ import annotations

import ast
import gc
import hashlib
import itertools
import json
import math
import os
import time
import warnings
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import signal as sps
from scipy.spatial.transform import Rotation
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, ParameterGrid, ParameterSampler
from sklearn.preprocessing import LabelEncoder

import torch
from torch import nn

# ---------------------------------------------------------------------------
# 1. Unified Honeycomb Parent Foundation (from base_utils)
# ---------------------------------------------------------------------------

try:
    from .base_utils import (
        ACC_COLUMNS,
        ALGO_VERSION,
        BASE_TO_CANONICAL,
        BUILTIN_WINDOW_CELLS,
        CANONICAL_RATES,
        CANONICAL_TO_LEGACY,
        COUNTER_COL,
        DEFAULT_TOF_INVALID_SENTINELS,
        DESCRIPTOR_CATALOG,
        DESCRIPTOR_SIGNALS,
        DYNAMIC_KINEMATIC_ATOMS,
        EXTENDED_MOTION_ATOMS,
        FEATURE_REGISTRY,
        FEATURE_VERSION,
        GRAVITY,
        HANN_SPECTRAL_ATOMS,
        LEGACY_MOTION_ATOMS,
        LEGACY_TO_CANONICAL,
        MODALITIES,
        MOTION_DESCRIPTOR_MODES,
        PERIODICITY_MODES,
        POWER_BRANCHES,
        QUALITY_MODES,
        ROT_COLUMNS,
        SENSOR_PREFIXES,
        SEQ_COL,
        SPECTRAL_DESCRIPTOR_MODES,
        THM_COLUMNS,
        TOF_COLUMNS,
        TOF_FILL_VALUES,
        TOF_SHAPE,
        TSAUG_AVAILABLE,
        TSFEL_ALL_DOMAINS,
        TSFEL_DEFAULT_FEATURES,
        TSFRESH_PROFILES,
        BaseAugmentor,
        BaseSearchCV,
        BaseSequenceClassifier,
        BaseSequenceExtractor,
        Catch22Backend,
        CrossModalCell,
        DenoisingFilterCell,
        DescriptorCatalogEntry,
        ECat,
        EFloat,
        EInt,
        EventsComplexityCell,
        ExampleMeanCell,
        FeatureBackend,
        FeatureFamilySpec,
        FeatureManifest,
        FeatureManifestEntry,
        FeatureRegistry,
        GASearchCV as BaseGASearchCV,
        HandednessCanonicalizer,
        HoneycombBase,
        IMUWindow,
        InvalidExtractorParams,
        InvalidParams,
        MetricEnvelope,
        MultiScaleCell,
        NoopOptions,
        OrientationPostureCell,
        PhysicsKernel,
        SeglearnBackend,
        SequenceAggregationCell,
        SignalDescriptorCell,
        SignalDescriptorOptions,
        SktimeBackend,
        TSFELBackend,
        TSFlexBackend,
        TSFreshBackend,
        ThermalDynamicsCell,
        ToFSpatialCell,
        ToFTemporalCell,
        TsaugAdapter,
        VibrationMultiaxialCell,
        WindowFeatureCell,
        acf_cadence_estimate,
        acf_dominant_frequency,
        acf_parabolic_fundamental,
        aggregate_window_features_by_sequence,
        angular_distance_rad,
        angular_velocity_body,
        apply_filter_1d_or_2d,
        baevsky_stress_metrics,
        bounded_fill,
        build_quick_results_df_shared,
        build_tsfel_config,
        build_tsfresh_fc_parameters,
        canonicalize_handedness_frame,
        clean_tof_array,
        compute_multiscale_window_summaries,
        compute_tof_block_channels,
        contiguous_runs,
        descriptor_subset_modes,
        detect_peak_anchor_starts,
        detrend_values,
        fill_missing_time,
        find_data_root,
        fix_quaternion_continuity,
        format_descriptor_name,
        highpass_zero_phase,
        inactive_descriptor_parameters,
        infer_hold_rate,
        library_available,
        library_status,
        nearest_indices,
        normalize_quaternions,
        nyquist,
        one_sided_psd,
        parse_pipe_modes,
        parse_structured_value,
        positive_rate,
        quaternions_to_euler,
        quaternions_to_rot6d,
        rate_ratio,
        register_feature_family,
        reject_tof_spikes,
        remove_gravity_quaternion,
        require_library,
        resample_array,
        resolve_power_bands,
        sample_balanced_subset,
        seconds_to_samples,
        shift_time,
        summarize_branch_window,
        summarize_imu_window,
        walch_dog_variability,
        window_psd,
        window_starts,
    )
except ImportError:
    from base_utils import (
        ACC_COLUMNS,
        ALGO_VERSION,
        BASE_TO_CANONICAL,
        BUILTIN_WINDOW_CELLS,
        CANONICAL_RATES,
        CANONICAL_TO_LEGACY,
        COUNTER_COL,
        DEFAULT_TOF_INVALID_SENTINELS,
        DESCRIPTOR_CATALOG,
        DESCRIPTOR_SIGNALS,
        DYNAMIC_KINEMATIC_ATOMS,
        EXTENDED_MOTION_ATOMS,
        FEATURE_REGISTRY,
        FEATURE_VERSION,
        GRAVITY,
        HANN_SPECTRAL_ATOMS,
        LEGACY_MOTION_ATOMS,
        LEGACY_TO_CANONICAL,
        MODALITIES,
        MOTION_DESCRIPTOR_MODES,
        PERIODICITY_MODES,
        POWER_BRANCHES,
        QUALITY_MODES,
        ROT_COLUMNS,
        SENSOR_PREFIXES,
        SEQ_COL,
        SPECTRAL_DESCRIPTOR_MODES,
        THM_COLUMNS,
        TOF_COLUMNS,
        TOF_FILL_VALUES,
        TOF_SHAPE,
        TSAUG_AVAILABLE,
        TSFEL_ALL_DOMAINS,
        TSFEL_DEFAULT_FEATURES,
        TSFRESH_PROFILES,
        BaseAugmentor,
        BaseSearchCV,
        BaseSequenceClassifier,
        BaseSequenceExtractor,
        Catch22Backend,
        CrossModalCell,
        DenoisingFilterCell,
        DescriptorCatalogEntry,
        ECat,
        EFloat,
        EInt,
        EventsComplexityCell,
        ExampleMeanCell,
        FeatureBackend,
        FeatureFamilySpec,
        FeatureManifest,
        FeatureManifestEntry,
        FeatureRegistry,
        GASearchCV as BaseGASearchCV,
        HandednessCanonicalizer,
        HoneycombBase,
        IMUWindow,
        InvalidExtractorParams,
        InvalidParams,
        MetricEnvelope,
        MultiScaleCell,
        NoopOptions,
        OrientationPostureCell,
        PhysicsKernel,
        SeglearnBackend,
        SequenceAggregationCell,
        SignalDescriptorCell,
        SignalDescriptorOptions,
        SktimeBackend,
        TSFELBackend,
        TSFlexBackend,
        TSFreshBackend,
        ThermalDynamicsCell,
        ToFSpatialCell,
        ToFTemporalCell,
        TsaugAdapter,
        VibrationMultiaxialCell,
        WindowFeatureCell,
        acf_cadence_estimate,
        acf_dominant_frequency,
        acf_parabolic_fundamental,
        aggregate_window_features_by_sequence,
        angular_distance_rad,
        angular_velocity_body,
        apply_filter_1d_or_2d,
        baevsky_stress_metrics,
        bounded_fill,
        build_quick_results_df_shared,
        build_tsfel_config,
        build_tsfresh_fc_parameters,
        canonicalize_handedness_frame,
        clean_tof_array,
        compute_multiscale_window_summaries,
        compute_tof_block_channels,
        contiguous_runs,
        descriptor_subset_modes,
        detect_peak_anchor_starts,
        detrend_values,
        fill_missing_time,
        find_data_root,
        fix_quaternion_continuity,
        format_descriptor_name,
        highpass_zero_phase,
        inactive_descriptor_parameters,
        infer_hold_rate,
        library_available,
        library_status,
        nearest_indices,
        normalize_quaternions,
        nyquist,
        one_sided_psd,
        parse_pipe_modes,
        parse_structured_value,
        positive_rate,
        quaternions_to_euler,
        quaternions_to_rot6d,
        rate_ratio,
        register_feature_family,
        reject_tof_spikes,
        remove_gravity_quaternion,
        require_library,
        resample_array,
        resolve_power_bands,
        sample_balanced_subset,
        seconds_to_samples,
        shift_time,
        summarize_branch_window,
        summarize_imu_window,
        walch_dog_variability,
        window_psd,
        window_starts,
    )

try:
    from .pretraining_ssl_utils import (
        PRETRAINING_MODES,
        PRETRAINING_SOURCES,
        SSL_LOSSES,
        MaskedSSLReconstructionHead,
        PretrainingAndSSLMixin,
        PretrainedMultiBranchTSAIClassifier,
        select_pretraining_sequences,
        transfer_torch_backbone_weights,
    )
except ImportError:
    from pretraining_ssl_utils import (
        PRETRAINING_MODES,
        PRETRAINING_SOURCES,
        SSL_LOSSES,
        MaskedSSLReconstructionHead,
        PretrainingAndSSLMixin,
        PretrainedMultiBranchTSAIClassifier,
        select_pretraining_sequences,
        transfer_torch_backbone_weights,
    )

# Feature-availability flags (derived from canonical features.physics.library_status)
_LIB_STATUS = library_status()
TSFEL_AVAILABLE = _LIB_STATUS["tsfel"]
TSFRESH_AVAILABLE = _LIB_STATUS["tsfresh"]
TSAI_AVAILABLE = _LIB_STATUS["tsai"]
SKOPT_AVAILABLE = _LIB_STATUS["scikit-optimize"]
CATCH22_AVAILABLE = _LIB_STATUS["pycatch22"]
TSFLEX_AVAILABLE = _LIB_STATUS["tsflex"]
SEGLEARN_AVAILABLE = _LIB_STATUS["seglearn"]
SKTIME_AVAILABLE = _LIB_STATUS["sktime"]

try:
    from skopt import BayesSearchCV
    from skopt.space import Categorical as SkCategorical, Integer as SkInteger, Real as SkReal
except Exception:  # pragma: no cover
    BayesSearchCV = SkCategorical = SkInteger = SkReal = None


def _require(flag: bool, name: str, pip: Optional[str] = None) -> None:
    require_library(flag, name, pip)


def _structured(value: Any) -> Any:
    """Accept JSON / Python-literal strings for list-like search values."""
    return parse_structured_value(value, exc_type=InvalidParams)


def _pipe_modes(
    value: Any,
    allowed: Iterable[str],
    *,
    name: str,
    aliases: Optional[Mapping[str, Iterable[str]]] = None,
) -> set:
    """Parse ``None`` / ``'all'`` / ``'a|b'`` categorical mode strings."""
    return parse_pipe_modes(
        value,
        allowed,
        name=name,
        aliases=aliases,
        support_all_alias=True,
        exc_type=InvalidParams,
    )


# ---------------------------------------------------------------------------
# 2-3. Signal descriptor backward-compatible aliases (delegating to signal_descriptors)
# ---------------------------------------------------------------------------

_LEGACY_NOOP_MOTION_ATOMS = LEGACY_MOTION_ATOMS
_KIMITCH_MOTION_ATOMS = EXTENDED_MOTION_ATOMS
NOOP_MOTION_ATOMS = DYNAMIC_KINEMATIC_ATOMS
NOOP_SPECTRAL_ATOMS = HANN_SPECTRAL_ATOMS
NOOP_PERIODICITY_MODES = PERIODICITY_MODES
NOOP_QUALITY_MODES = QUALITY_MODES
NOOP_SIGNALS = DESCRIPTOR_SIGNALS
NOOP_POWER_BRANCHES = POWER_BRANCHES
NOOP_MOTION_MODES = MOTION_DESCRIPTOR_MODES
NOOP_SPECTRAL_MODES = SPECTRAL_DESCRIPTOR_MODES

noop_subset_modes = descriptor_subset_modes
inactive_noop_parameters = inactive_descriptor_parameters
cadence_estimate = acf_cadence_estimate
kimmitch_acf_estimate = acf_parabolic_fundamental


def noop_imu_window(
    src: Mapping[str, np.ndarray],
    fs: float,
    options: SignalDescriptorOptions,
    *,
    naming_mode: str = "legacy",
):
    """Backward-compatible wrapper around ``signal_descriptors.summarize_imu_window``."""
    return summarize_imu_window(src, fs, options, naming_mode=naming_mode)


def noop_power_window(
    values: np.ndarray,
    observed: np.ndarray,
    fs: float,
    options: SignalDescriptorOptions,
    prefix: str,
    *,
    naming_mode: str = "legacy",
):
    """Backward-compatible wrapper around ``signal_descriptors.summarize_branch_window``."""
    return summarize_branch_window(values, observed, fs, options, prefix, naming_mode=naming_mode)


# ---------------------------------------------------------------------------
# 4. Sequence extractor (per-modality chunks, sample-rate adaptive)
# ---------------------------------------------------------------------------

STATIC_SOURCES = ("acc_mag", "lin_mag", "ang_speed", "thm_mean", "tof_mean")
STATIC_BACKENDS = ("none", "tsfel", "tsfresh", "catch22", "tsflex", "seglearn", "sktime")
TSFEL_DOMAINS = TSFEL_ALL_DOMAINS
IMU_FEATURE_PRESETS = (None, "public_core", "public_core_jerk")
FILTER_MODES = ("none", "lowpass", "highpass", "bandpass", "median", "savgol", "median_lowpass", "wavelet")
BASELINE_MODES = ("none", "first_n_seconds", "sequence_mean")
LINEAR_ACC_FRAMES = ("body", "world")
DESCRIPTOR_NAMING_MODES = ("legacy", "canonical")
ROTATION_MODE_SET = {"quaternion", "angular_velocity", "euler", "rot6d", "angular_distance", "angular_accel", "angular_jerk"}
TOF_MODE_SET = {"raw", "pooled_stats", "sensor_stats", "block_2x2", "block_4x4"}


class MultiModalSequenceExtractor(BaseSequenceExtractor):
    """Turn raw CMI sensor rows into per-modality window tensors + pluggable static features.

    Every sensor keeps its **own sample rate**.  ``window_seconds`` is a duration, so a
    2.5 s window is 250 samples of IMU at 100 Hz but 50 samples of ToF/thermopile at
    20 Hz; band edges, external backend ``fs`` (TSFEL, TSFresh, catch22, tsflex,
    seglearn, sktime), domain feature families, and modular signal descriptors all
    derive from the rate of the array they describe.

    ``align_mode='time'``: rows share the ``row_sampling_rate`` clock and each sensor is
    anti-alias resampled to its target rate.  ``'rows'``: one row is one native sample of
    every sensor (legacy behaviour); only target-rate resampling is applied.

    Core orchestrator dispatches all domain feature families and external OSS backends
    dynamically via ``FEATURE_REGISTRY`` and emits ``self.feature_manifest_`` on every
    ``fit`` / ``featurize`` call.
    """

    def __init__(
        self, noop=None,
        # --- rates (Hz) ---
        row_sampling_rate=100.0, align_mode="time", resample_method="poly",
        imu_native_sampling_rate=100.0, rot_native_sampling_rate=100.0,
        thm_native_sampling_rate=20.0, tof_native_sampling_rate=20.0,
        imu_target_sampling_rate=None, thm_target_sampling_rate=None, tof_target_sampling_rate=None,
        # --- windows (seconds) ---
        window_seconds=2.5, stride_ratio=0.75, final_window_mode="overlap",
        max_windows_per_sequence=200, window_cap_policy="uniform",
        # --- IMU ---
        acc_modes="raw|linear", use_acc_magnitude=True, use_linear_acc_magnitude=True,
        rotation_modes="quaternion|angular_velocity", fix_quaternion_sign=True,
        linear_acc_mode="baseline", use_highpass_fallback=True, highpass_hz=0.3,
        # --- thermopile / ToF ---
        thm_modes="centered_diff", tof_modes="pooled_stats|sensor_stats", tof_fill_mode="far_500",
        # --- static (open-source backends) ---
        static_backend="tsfel", static_channels="acc_mag|ang_speed|thm_mean|tof_mean",
        tsfel_domains="statistical|temporal|spectral", tsfel_features=None,
        tsfresh_profile="minimal", tsfresh_fft_bands=((0.5, 2.0), (2.0, 4.0), (4.0, 8.0)),
        static_clip=10.0, sequence_col=SEQ_COL, counter_col=COUNTER_COL,
        # --- v4 modular preprocessing, modality control & pluggable feature families ---
        modalities="all",
        canonicalize_handedness=False,
        handedness_col="handedness",
        imu_feature_preset=None,
        linear_acc_frame="body",
        lag_diff_scales=None,
        filter_mode="none",
        tof_spike_threshold=None,
        tof_invalid_sentinels=DEFAULT_TOF_INVALID_SENTINELS,
        baseline_mode="none",
        min_observed_fraction=0.0,
        scale_observed_only=False,
        descriptor_naming="legacy",
        window_seconds_list=None,
        sequence_aggregation_features=False,
        orientation_features=False,
        vibration_features=False,
        tof_spatial_features=False,
        tof_temporal_features=False,
        thermal_features=False,
        cross_modal_features=False,
        events_complexity_features=False,
        extra_feature_families=None,
    ):
        self.noop = noop
        self.row_sampling_rate = row_sampling_rate
        self.align_mode = align_mode
        self.resample_method = resample_method
        self.imu_native_sampling_rate = imu_native_sampling_rate
        self.rot_native_sampling_rate = rot_native_sampling_rate
        self.thm_native_sampling_rate = thm_native_sampling_rate
        self.tof_native_sampling_rate = tof_native_sampling_rate
        self.imu_target_sampling_rate = imu_target_sampling_rate
        self.thm_target_sampling_rate = thm_target_sampling_rate
        self.tof_target_sampling_rate = tof_target_sampling_rate
        self.window_seconds = window_seconds
        self.stride_ratio = stride_ratio
        self.final_window_mode = final_window_mode
        self.max_windows_per_sequence = max_windows_per_sequence
        self.window_cap_policy = window_cap_policy
        self.acc_modes = acc_modes
        self.use_acc_magnitude = use_acc_magnitude
        self.use_linear_acc_magnitude = use_linear_acc_magnitude
        self.rotation_modes = rotation_modes
        self.fix_quaternion_sign = fix_quaternion_sign
        self.linear_acc_mode = linear_acc_mode
        self.use_highpass_fallback = use_highpass_fallback
        self.highpass_hz = highpass_hz
        self.thm_modes = thm_modes
        self.tof_modes = tof_modes
        self.tof_fill_mode = tof_fill_mode
        self.static_backend = static_backend
        self.static_channels = static_channels
        self.tsfel_domains = tsfel_domains
        self.tsfel_features = tsfel_features
        self.tsfresh_profile = tsfresh_profile
        self.tsfresh_fft_bands = tsfresh_fft_bands
        self.static_clip = static_clip
        self.sequence_col = sequence_col
        self.counter_col = counter_col
        self.modalities = modalities
        self.canonicalize_handedness = canonicalize_handedness
        self.handedness_col = handedness_col
        self.imu_feature_preset = imu_feature_preset
        self.linear_acc_frame = linear_acc_frame
        self.lag_diff_scales = lag_diff_scales
        self.filter_mode = filter_mode
        self.tof_spike_threshold = tof_spike_threshold
        self.tof_invalid_sentinels = tof_invalid_sentinels
        self.baseline_mode = baseline_mode
        self.min_observed_fraction = min_observed_fraction
        self.scale_observed_only = scale_observed_only
        self.descriptor_naming = descriptor_naming
        self.window_seconds_list = window_seconds_list
        self.sequence_aggregation_features = sequence_aggregation_features
        self.orientation_features = orientation_features
        self.vibration_features = vibration_features
        self.tof_spatial_features = tof_spatial_features
        self.tof_temporal_features = tof_temporal_features
        self.thermal_features = thermal_features
        self.cross_modal_features = cross_modal_features
        self.events_complexity_features = events_complexity_features
        self.extra_feature_families = extra_feature_families

    # -- sklearn plumbing: ``noop__*`` parameters configure SignalDescriptorOptions --
    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        if deep and self.noop is None:
            params.update({f"noop__{k}": v for k, v in NoopOptions().get_params(deep=True).items()})
        return params

    def set_params(self, **params):
        if any(name.startswith("noop__") for name in params) and params.get("noop", self.noop) is None:
            params = dict(params)
            params["noop"] = NoopOptions()
        return super().set_params(**params)

    # -- configuration helpers --------------------------------------------------
    def noop_options(self) -> Optional[NoopOptions]:
        if self.noop is None:
            return None
        if not (isinstance(self.noop, SignalDescriptorOptions) or type(self.noop).__name__ in {"NoopOptions", "SignalDescriptorOptions"}):
            raise InvalidParams("noop must be None or SignalDescriptorOptions / NoopOptions")
        return self.noop if self.noop.enabled() else None

    def active_modalities(self) -> set:
        mods = _pipe_modes(self.modalities, MODALITIES, name="modalities")
        if not mods:
            mods = set(MODALITIES)
        mods.add("imu")
        return mods

    def branch_rates(self) -> Dict[str, float]:
        """Sample rate (Hz) of each enabled branch array."""
        mods = self.active_modalities()
        rates = {"imu": positive_rate(self.imu_target_sampling_rate or self.imu_native_sampling_rate, "imu rate")}
        if "thm" in mods and self.thm_modes is not None:
            rates["thm"] = positive_rate(self.thm_target_sampling_rate or self.thm_native_sampling_rate, "thm rate")
        if "tof" in mods and self.tof_modes is not None:
            rates["tof"] = positive_rate(self.tof_target_sampling_rate or self.tof_native_sampling_rate, "tof rate")
        return rates

    def branch_names(self) -> List[str]:
        return list(self.branch_rates())

    def window_lengths(self) -> Dict[str, int]:
        return {name: seconds_to_samples(self.window_seconds, rate) for name, rate in self.branch_rates().items()}

    def _parsed_lag_diff_scales(self) -> Tuple[int, ...]:
        if self.lag_diff_scales is None:
            return ()
        val = _structured(self.lag_diff_scales)
        if isinstance(val, (int, np.integer)):
            scales = (int(val),)
        elif isinstance(val, str):
            parts = [p.strip() for p in val.split("|") if p.strip()]
            scales = tuple(int(p) for p in parts)
        else:
            scales = tuple(int(v) for v in val)
        for k in scales:
            if k < 1:
                raise InvalidParams(f"lag_diff_scales values must be >= 1, got {k}")
        return tuple(sorted(set(scales)))

    def _parsed_window_seconds_list(self) -> Tuple[float, ...]:
        if self.window_seconds_list is None:
            return ()
        val = _structured(self.window_seconds_list)
        if isinstance(val, (int, float, np.number)):
            scales = (float(val),)
        elif isinstance(val, str):
            scales = tuple(float(p.strip()) for p in val.split("|") if p.strip())
        else:
            scales = tuple(float(v) for v in val)
        for s in scales:
            if not np.isfinite(s) or s <= 0:
                raise InvalidParams(f"window_seconds_list values must be positive, got {s}")
        return scales

    def _parsed_tof_sentinels(self) -> Tuple[float, ...]:
        if self.tof_invalid_sentinels is None:
            return ()
        val = _structured(self.tof_invalid_sentinels)
        if isinstance(val, (int, float, np.number)):
            return (float(val),)
        return tuple(float(v) for v in val)

    def _modes(self):
        acc = _pipe_modes(self.acc_modes, {"raw", "linear", "jerk"}, name="acc_modes")
        rot = _pipe_modes(self.rotation_modes, ROTATION_MODE_SET, name="rotation_modes")
        thm = _pipe_modes(self.thm_modes, {"raw", "centered", "diff", "centered_diff", "pooled_stats"}, name="thm_modes")
        tof = _pipe_modes(self.tof_modes, TOF_MODE_SET, name="tof_modes")
        static = _pipe_modes(self.static_channels, STATIC_SOURCES, name="static_channels")
        return acc, rot, thm, tof, static

    def validate(self):
        self.active_modalities()
        self._modes()
        self._parsed_lag_diff_scales()
        self._parsed_window_seconds_list()
        self._parsed_tof_sentinels()
        if self.align_mode not in {"time", "rows"}:
            raise InvalidParams("align_mode must be 'time' or 'rows'")
        if self.static_backend not in set(STATIC_BACKENDS) | {None}:
            raise InvalidParams(f"static_backend must be one of {STATIC_BACKENDS}, got {self.static_backend!r}")
        if self.tof_fill_mode not in {"far_500", "far_255", "zero", "nan_interpolate"}:
            raise InvalidParams("tof_fill_mode must be 'far_500', 'far_255', 'zero' or 'nan_interpolate'")
        if self.tsfresh_profile not in TSFRESH_PROFILES:
            raise InvalidParams(f"tsfresh_profile must be one of {TSFRESH_PROFILES}")
        if self.imu_feature_preset not in IMU_FEATURE_PRESETS:
            raise InvalidParams(f"imu_feature_preset must be one of {IMU_FEATURE_PRESETS}")
        if self.linear_acc_frame not in LINEAR_ACC_FRAMES:
            raise InvalidParams(f"linear_acc_frame must be one of {LINEAR_ACC_FRAMES}")
        if self.filter_mode not in FILTER_MODES:
            raise InvalidParams(f"filter_mode must be one of {FILTER_MODES}")
        if self.baseline_mode not in BASELINE_MODES:
            raise InvalidParams(f"baseline_mode must be one of {BASELINE_MODES}")
        if self.descriptor_naming not in DESCRIPTOR_NAMING_MODES:
            raise InvalidParams(f"descriptor_naming must be one of {DESCRIPTOR_NAMING_MODES}")
        if not 0.0 <= float(self.min_observed_fraction) <= 1.0:
            raise InvalidParams("min_observed_fraction must be in [0, 1]")
        if self.tof_spike_threshold is not None and float(self.tof_spike_threshold) <= 0:
            raise InvalidParams("tof_spike_threshold must be None or positive")
        if not 0 < float(self.stride_ratio) <= 1:
            raise InvalidParams("stride_ratio must be in (0, 1]")
        if float(self.window_seconds) <= 0:
            raise InvalidParams("window_seconds must be positive")
        rates = self.branch_rates()
        for name in ("imu", "rot", "thm", "tof"):
            positive_rate(getattr(self, f"{name}_native_sampling_rate"), f"{name} native rate")
        row = positive_rate(self.row_sampling_rate, "row_sampling_rate")
        if self.align_mode == "time":
            for name in ("imu", "rot", "thm", "tof"):
                if getattr(self, f"{name}_native_sampling_rate") > row * (1 + 1e-9):
                    raise InvalidParams(f"{name} native rate exceeds the row clock; raise row_sampling_rate")
        for name, rate in rates.items():
            if self.window_lengths()[name] < 4:
                raise InvalidParams(f"window_seconds is too short for the {name} branch at {rate:g} Hz")
        opts = self.noop_options()
        if opts is not None:
            opts.validate(rates["imu"])
        return self

    def _imu_plan(self):
        acc, rot, _, _, static = self._modes()
        opts = self.noop_options()
        motion, spectral = opts.mode_names() if opts else (set(), set())
        signal_used = bool(opts) and (opts.periodicity is not None or bool(spectral) or opts.quality is not None)
        preset = self.imu_feature_preset
        if preset in {"public_core", "public_core_jerk"}:
            acc = set(acc) | {"raw", "linear"}
            if preset == "public_core_jerk":
                acc.add("jerk")
            rot = set(rot) | {"quaternion", "angular_velocity", "angular_distance"}
        need_linear = "linear" in acc or (signal_used and opts.signal == "linear") or "lin_mag" in static
        need_ang = (
            "angular_velocity" in rot
            or "angular_accel" in rot
            or "angular_jerk" in rot
            or "ang_speed" in static
            or bool(motion & {"rotation_mean", "rotation_rms", "gyro_energy", "acc_gyro_coupling"})
            or (signal_used and opts.signal == "angular_speed")
            or bool(self.orientation_features)
        )
        names = ["acc_x", "acc_y", "acc_z"]
        if self.use_acc_magnitude or preset is not None:
            names.append("acc_mag")
        if need_linear:
            names += ["lin_x", "lin_y", "lin_z"]
            if self.use_linear_acc_magnitude or preset is not None:
                names.append("lin_mag")
        if "jerk" in acc:
            names += ["jerk_x", "jerk_y", "jerk_z"]
        if "quaternion" in rot or bool(self.orientation_features):
            names += ["rot_w", "rot_x", "rot_y", "rot_z"]
        if need_ang:
            names += ["ang_x", "ang_y", "ang_z"]
            if preset is not None:
                names.append("ang_speed")
        if "angular_distance" in rot:
            names.append("angular_distance")
        if "angular_accel" in rot:
            names += ["ang_acc_x", "ang_acc_y", "ang_acc_z"]
        if "angular_jerk" in rot:
            names += ["ang_jerk_x", "ang_jerk_y", "ang_jerk_z"]
        if "rot6d" in rot:
            names += [f"rot6d_{k}" for k in range(6)]
        if "euler" in rot:
            names += ["euler_x", "euler_y", "euler_z"]
        for k in self._parsed_lag_diff_scales():
            names += [f"acc_{a}_diff{k}" for a in "xyz"]
            if need_linear:
                names += [f"lin_{a}_diff{k}" for a in "xyz"]
        return names, need_linear, need_ang

    def channel_names(self) -> Dict[str, List[str]]:
        _, _, thm, tof, _ = self._modes()
        rates = self.branch_rates()
        names = {"imu": self._imu_plan()[0]}
        if "thm" in rates:
            cols = []
            if "raw" in thm:
                cols += [f"thm_raw_{i}" for i in range(1, 6)]
            if "centered" in thm or "centered_diff" in thm:
                cols += [f"thm_centered_{i}" for i in range(1, 6)]
            if "diff" in thm or "centered_diff" in thm:
                cols += [f"thm_diff_{i}" for i in range(1, 6)]
            if "pooled_stats" in thm:
                cols += ["thm_std", "thm_min", "thm_max"]
            names["thm"] = cols + ["thm_mean"]
        if "tof" in rates:
            cols = [f"tof_raw_{i}" for i in range(320)] if "raw" in tof else []
            cols.append("tof_mean")
            if "pooled_stats" in tof:
                cols += ["tof_std", "tof_min", "tof_max", "tof_valid_fraction"]
            if "sensor_stats" in tof:
                cols += [f"tof_sensor_mean_{i}" for i in range(1, 6)] + [f"tof_sensor_std_{i}" for i in range(1, 6)]
            if "block_2x2" in tof:
                _, b2_names = compute_tof_block_channels(np.zeros((0, 320)), block_size=2, include_temporal_diff=True)
                cols += b2_names
            if "block_4x4" in tof:
                _, b4_names = compute_tof_block_channels(np.zeros((0, 320)), block_size=4, include_temporal_diff=True)
                cols += b4_names
            names["tof"] = cols
        return names

    # -- fitting ---------------------------------------------------------------
    def fit(self, X, y=None):
        self.validate()
        chunks = self.featurize(self.raw_chunks(X))
        self.fit_scalers(chunks)
        return self

    def transform(self, X):
        return self.transform_chunks(X)

    def transform_chunks(self, X):
        return self.scale(self.featurize(self.raw_chunks(X)))

    # -- raw windows -----------------------------------------------------------
    @staticmethod
    def _extract_column_block(X: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
        """Extract ``columns`` from ``X`` in one vectorized pass (missing columns filled with NaN)."""
        present_cols = [c for c in columns if c in X.columns]
        if len(present_cols) == len(columns):
            return X[list(columns)].to_numpy(dtype=np.float64, copy=False)
        out = np.full((len(X), len(columns)), np.nan, dtype=np.float64)
        if present_cols:
            col_map = {c: i for i, c in enumerate(columns)}
            idx = [col_map[c] for c in present_cols]
            out[:, idx] = X[present_cols].to_numpy(dtype=np.float64, copy=False)
        return out

    def raw_chunks(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Per-modality window tensors from raw rows (no static features, unscaled)."""
        self.validate()
        if self.sequence_col not in X:
            raise ValueError(f"X must contain {self.sequence_col!r}")
        work_df = (
            canonicalize_handedness_frame(X, handedness_col=self.handedness_col)
            if self.canonicalize_handedness else X
        )
        names = self.channel_names()
        rates = self.branch_rates()
        wins = self.window_lengths()
        groups = work_df.groupby(self.sequence_col, sort=False).indices

        counter_all = (
            pd.to_numeric(work_df[self.counter_col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
            if self.counter_col in work_df.columns else None
        )
        acc_all = self._extract_column_block(work_df, ACC_COLUMNS)
        rot_all = self._extract_column_block(work_df, ROT_COLUMNS)
        thm_all = self._extract_column_block(work_df, THM_COLUMNS) if "thm" in rates else None
        tof_all = self._extract_column_block(work_df, TOF_COLUMNS) if "tof" in rates else None

        acc = {name: [] for name in rates}
        mask = {name: [] for name in rates}
        extra = {"obs_axes": [], "q_obs": [], "row_present": [], "obs_thm": [], "obs_tof": []}
        present, seq_ids, win_idx, win_starts_s, seq_durs_s, thm_seq_bases = [], [], [], [], [], []
        fs_imu = rates["imu"]
        stride_imu = max(1, int(round(wins["imu"] * float(self.stride_ratio))))
        for seq_id, positions in groups.items():
            prepared = self._prepare_sequence_arrays(
                counter_seq=counter_all[positions] if counter_all is not None else None,
                acc_seq=acc_all[positions],
                rot_seq=rot_all[positions],
                thm_seq=thm_all[positions] if thm_all is not None else None,
                tof_seq=tof_all[positions] if tof_all is not None else None,
                n_seq_rows=len(positions),
            )
            n_imu = len(prepared["imu"])
            seq_dur = float(n_imu) / fs_imu
            starts = window_starts(
                n_imu, wins["imu"], stride_imu,
                self.final_window_mode, self.max_windows_per_sequence, self.window_cap_policy)
            for k, s_imu in enumerate(starts):
                t0 = s_imu / fs_imu
                for name in rates:
                    arr = prepared[name]
                    start = int(round(t0 * rates[name]))
                    seg = arr[start:start + wins[name]]
                    out = np.zeros((wins[name], arr.shape[1]), dtype=np.float32)
                    out[:len(seg)] = seg
                    m = np.zeros(wins[name], dtype=bool)
                    m[:len(seg)] = True
                    acc[name].append(out)
                    mask[name].append(m)
                for key, src_key, name in (("obs_axes", "obs_axes", "imu"), ("q_obs", "q_obs", "imu"),
                                           ("row_present", "row_present", "imu")):
                    arr = prepared[src_key]
                    start = int(round(t0 * rates[name]))
                    seg = arr[start:start + wins[name]]
                    out = np.zeros((wins[name],) + arr.shape[1:], dtype=bool)
                    out[:len(seg)] = seg
                    extra[key].append(out)
                for key, name in (("obs_thm", "thm"), ("obs_tof", "tof")):
                    if name in rates:
                        arr = prepared[key]
                        start = int(round(t0 * rates[name]))
                        seg = arr[start:start + wins[name]]
                        out = np.zeros(wins[name], dtype=bool)
                        out[:len(seg)] = seg
                        extra[key].append(out)
                flags = [bool(prepared["present"]["imu"]),
                         bool(prepared["present"]["thm"]) and "thm" in rates,
                         bool(prepared["present"]["tof"]) and "tof" in rates]
                present.append(flags)
                seq_ids.append(seq_id)
                win_idx.append(k)
                win_starts_s.append(float(t0))
                seq_durs_s.append(seq_dur)
                thm_seq_bases.append(float(prepared.get("thm_seq_baseline", np.nan)))
        chunks = {
            "branches": {n: np.stack(v) for n, v in acc.items()},
            "masks": {n: np.stack(v) for n, v in mask.items()},
            "present": np.asarray(present, dtype=bool).reshape(-1, 3),
            "sequence_ids": np.asarray(seq_ids), "window_index": np.asarray(win_idx),
            "window_starts_s": np.asarray(win_starts_s, dtype=np.float64),
            "sequence_durations_s": np.asarray(seq_durs_s, dtype=np.float64),
            "thm_seq_baseline": np.asarray(thm_seq_bases, dtype=np.float64),
            "channel_names": names, "rates": dict(rates),
            "obs_axes": np.stack(extra["obs_axes"]), "q_obs": np.stack(extra["q_obs"]),
            "row_present": np.stack(extra["row_present"]),
        }
        for key in ("obs_thm", "obs_tof"):
            if extra[key]:
                chunks[key] = np.stack(extra[key])
        # a modality that is absent for a window carries zeros, never stale values
        for j, name in enumerate(MODALITIES):
            if name in chunks["branches"]:
                absent = ~chunks["present"][:, j]
                chunks["branches"][name][absent] = 0.0
        return chunks

    def _prepare_sequence(self, frame: pd.DataFrame) -> Dict[str, Any]:
        """One sequence DataFrame -> arrays at each branch rate plus observation masks."""
        rates = self.branch_rates()
        counter_seq = (
            pd.to_numeric(frame[self.counter_col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
            if self.counter_col in frame.columns else None
        )
        return self._prepare_sequence_arrays(
            counter_seq=counter_seq,
            acc_seq=self._extract_column_block(frame, ACC_COLUMNS),
            rot_seq=self._extract_column_block(frame, ROT_COLUMNS),
            thm_seq=self._extract_column_block(frame, THM_COLUMNS) if "thm" in rates else None,
            tof_seq=self._extract_column_block(frame, TOF_COLUMNS) if "tof" in rates else None,
            n_seq_rows=len(frame),
        )

    def _prepare_sequence_arrays(
        self,
        *,
        counter_seq: Optional[np.ndarray],
        acc_seq: np.ndarray,
        rot_seq: np.ndarray,
        thm_seq: Optional[np.ndarray],
        tof_seq: Optional[np.ndarray],
        n_seq_rows: int,
    ) -> Dict[str, Any]:
        """One sequence's raw arrays -> resampled arrays at each branch rate plus observation masks."""
        rates = self.branch_rates()
        row = positive_rate(self.row_sampling_rate)
        counter = counter_seq if (counter_seq is not None and np.isfinite(counter_seq).all()) else np.arange(n_seq_rows, dtype=float)
        diffs = np.diff(counter)
        is_dense_monotonic = len(counter) == 0 or (np.all(diffs == 1) and counter[0] == counter.min())
        if is_dense_monotonic:
            n_rows = len(counter)
            row_present = np.ones(n_rows, dtype=bool)
            order = None
            pos = None
        else:
            order = np.argsort(counter, kind="stable")
            counter_sorted = counter[order].astype(np.int64)
            keep = np.r_[np.diff(counter_sorted) != 0, True]
            order, counter_sorted = order[keep], counter_sorted[keep]
            pos = counter_sorted - counter_sorted.min()
            if pos.max() > 20 * len(pos) + 1000:
                pos = np.arange(len(pos))
            n_rows = int(pos.max()) + 1
            row_present = np.zeros(n_rows, dtype=bool)
            row_present[pos] = True

        def place_dense(raw_arr: np.ndarray) -> np.ndarray:
            if is_dense_monotonic:
                return raw_arr.copy()
            out = np.full((n_rows, raw_arr.shape[1]), np.nan, dtype=np.float64)
            out[pos] = raw_arr[order]
            return out

        def to_rate(values, native, target, observed_rows):
            """values (n_rows, C) NaN-filled -> (resampled, observed at the output rate)."""
            src_rate = row if self.align_mode == "time" else native
            filled, _ = fill_missing_time(values)
            out = resample_array(filled, src_rate, target, self.resample_method)
            idx = nearest_indices(len(filled), src_rate, target)[:len(out)]
            if len(idx) < len(out):
                idx = np.r_[idx, np.full(len(out) - len(idx), len(filled) - 1)]
            return out, observed_rows[idx]

        fs_i = rates["imu"]
        names, need_linear, need_ang = self._imu_plan()
        acc_rows = place_dense(acc_seq)
        acc_obs_rows = np.isfinite(acc_rows)
        acc_rate_native = float(self.imu_native_sampling_rate)
        acc, _ = to_rate(acc_rows, acc_rate_native, fs_i, np.ones(n_rows, dtype=bool))
        if self.filter_mode != "none" and len(acc):
            acc = apply_filter_1d_or_2d(acc, fs_i, method=self.filter_mode)
        if self.baseline_mode == "first_n_seconds" and len(acc):
            n_base = max(1, min(len(acc), seconds_to_samples(0.5, fs_i)))
            acc = acc - np.mean(acc[:n_base], axis=0, keepdims=True)
        elif self.baseline_mode == "sequence_mean" and len(acc):
            acc = acc - np.mean(acc, axis=0, keepdims=True)
        n_imu = len(acc)
        idx_acc = nearest_indices(n_rows, row if self.align_mode == "time" else acc_rate_native, fs_i)[:n_imu]
        obs_axes = acc_obs_rows[np.clip(idx_acc, 0, n_rows - 1)] if n_imu else np.zeros((0, 3), dtype=bool)
        present_row = row_present[np.clip(idx_acc, 0, n_rows - 1)]
        quats_rows = place_dense(rot_seq)
        q_valid_rows = np.isfinite(quats_rows).all(axis=1) & (np.linalg.norm(np.nan_to_num(quats_rows), axis=1) > 0)
        if self.fix_quaternion_sign:
            quats_rows = fix_quaternion_continuity(quats_rows)
        rot_native = float(self.rot_native_sampling_rate)
        q_fill, _ = fill_missing_time(np.where(q_valid_rows[:, None], quats_rows, np.nan), fill_value=0.0)
        q_res, _ = to_rate(q_fill, rot_native, fs_i, np.ones(n_rows, dtype=bool))
        q_res_u, _ = normalize_quaternions(q_res)
        q_res = np.nan_to_num(q_res_u, nan=0.0)
        idx_rot = nearest_indices(n_rows, row if self.align_mode == "time" else rot_native, fs_i)[:n_imu]
        q_obs = q_valid_rows[np.clip(idx_rot, 0, n_rows - 1)] if n_imu else np.zeros(0, dtype=bool)
        columns: Dict[str, np.ndarray] = {}
        for i, a in enumerate("xyz"):
            columns[f"acc_{a}"] = acc[:, i]
        columns["acc_mag"] = np.linalg.norm(acc, axis=1)
        lin = np.zeros_like(acc)
        if need_linear and n_imu:
            lin = self._linear_acceleration(acc, q_res, q_obs, fs_i)
            for i, a in enumerate("xyz"):
                columns[f"lin_{a}"] = lin[:, i]
            columns["lin_mag"] = np.linalg.norm(lin, axis=1)
        if n_imu:
            jerk = np.vstack([np.zeros((1, 3)), np.diff(acc, axis=0) * fs_i])
            for i, a in enumerate("xyz"):
                columns[f"jerk_{a}"] = jerk[:, i]
            for i, a in enumerate("wxyz"):
                columns[f"rot_{a}"] = q_res[:, i]
            if need_ang:
                ang = np.nan_to_num(angular_velocity_body(q_res, fs_i, q_obs), nan=0.0)
                for i, a in enumerate("xyz"):
                    columns[f"ang_{a}"] = ang[:, i]
                columns["ang_speed"] = np.linalg.norm(ang, axis=1)
                if "ang_acc_x" in names:
                    ang_acc = np.vstack([np.zeros((1, 3)), np.diff(ang, axis=0) * fs_i])
                    for i, a in enumerate("xyz"):
                        columns[f"ang_acc_{a}"] = ang_acc[:, i]
                if "ang_jerk_x" in names:
                    ang_acc_tmp = np.vstack([np.zeros((1, 3)), np.diff(ang, axis=0) * fs_i])
                    ang_jerk = np.vstack([np.zeros((1, 3)), np.diff(ang_acc_tmp, axis=0) * fs_i])
                    for i, a in enumerate("xyz"):
                        columns[f"ang_jerk_{a}"] = ang_jerk[:, i]
            if "angular_distance" in names:
                columns["angular_distance"] = angular_distance_rad(q_res, q_obs)
            if "rot6d_0" in names:
                r6 = np.nan_to_num(quaternions_to_rot6d(q_res), nan=0.0)
                for k in range(6):
                    columns[f"rot6d_{k}"] = r6[:, k]
            if "euler_x" in names:
                eul = np.nan_to_num(quaternions_to_euler(q_res, seq="xyz", unwrap=False), nan=0.0)
                for i, a in enumerate("xyz"):
                    columns[f"euler_{a}"] = eul[:, i]
            for lag_k in self._parsed_lag_diff_scales():
                d_acc = np.zeros_like(acc)
                if n_imu > lag_k:
                    d_acc[lag_k:] = acc[lag_k:] - acc[:-lag_k]
                for i, a in enumerate("xyz"):
                    columns[f"acc_{a}_diff{lag_k}"] = d_acc[:, i]
                if need_linear:
                    d_lin = np.zeros_like(lin)
                    if n_imu > lag_k:
                        d_lin[lag_k:] = lin[lag_k:] - lin[:-lag_k]
                    for i, a in enumerate("xyz"):
                        columns[f"lin_{a}_diff{lag_k}"] = d_lin[:, i]
        imu = (np.stack([columns[n] for n in names], axis=1) if n_imu else np.zeros((0, len(names))))
        prepared: Dict[str, Any] = {
            "imu": imu, "obs_axes": obs_axes, "q_obs": q_obs, "row_present": present_row,
            "present": {"imu": bool(acc_obs_rows.any()), "thm": False, "tof": False},
            "thm_seq_baseline": float("nan"),
        }
        if "thm" in rates and thm_seq is not None:
            thm_rows = place_dense(thm_seq)
            thm_obs_rows = np.isfinite(thm_rows).all(axis=1)
            prepared["present"]["thm"] = bool(np.isfinite(thm_rows).any())
            if prepared["present"]["thm"]:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    prepared["thm_seq_baseline"] = float(np.nanmedian(thm_rows))
            prepared["thm"], prepared["obs_thm"] = self._thermopile(thm_rows, thm_obs_rows, rates["thm"], to_rate)
        if "tof" in rates and tof_seq is not None:
            tof_rows = place_dense(tof_seq)
            tof_rows, _, _ = clean_tof_array(
                tof_rows, invalid_sentinels=self._parsed_tof_sentinels(), fill_mode=self.tof_fill_mode
            )
            if self.tof_spike_threshold is not None and float(self.tof_spike_threshold) > 0:
                tof_rows, _ = reject_tof_spikes(tof_rows, spike_threshold=float(self.tof_spike_threshold))
            prepared["present"]["tof"] = bool(np.isfinite(tof_rows).any())
            prepared["tof"], prepared["obs_tof"] = self._tof(tof_rows, rates["tof"], to_rate)
        return prepared

    def _linear_acceleration(self, acc, quats, q_obs, fs):
        """Per-row quaternion gravity removal via SciPy Rotation, with high-pass / mean fallback for invalid rows."""
        return remove_gravity_quaternion(
            acc,
            quats,
            q_valid=q_obs,
            mode=self.linear_acc_mode,
            frame=self.linear_acc_frame,
            fs=fs,
            highpass_hz=float(self.highpass_hz),
            use_highpass_fallback=bool(self.use_highpass_fallback),
        )

    def _thermopile(self, rows, obs_rows, target, to_rate):
        _, _, thm, _, _ = self._modes()
        native = float(self.thm_native_sampling_rate)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(np.where(np.isfinite(rows), rows, np.nan), axis=1) if np.isfinite(rows).any() else np.full(len(rows), np.nan)
        base, obs = to_rate(rows, native, target, obs_rows)
        mean_r, _ = to_rate(mean[:, None], native, target, obs_rows)
        if not np.isfinite(rows).any():
            base = np.zeros_like(base)
            mean_r = np.zeros_like(mean_r)
        else:
            if self.filter_mode != "none" and len(base):
                base = apply_filter_1d_or_2d(base, target, method=self.filter_mode)
                mean_r = np.mean(base, axis=1, keepdims=True)
            if self.baseline_mode == "first_n_seconds" and len(base):
                n_b = max(1, min(len(base), seconds_to_samples(0.5, target)))
                base = base - np.mean(base[:n_b], axis=0, keepdims=True)
                mean_r = np.mean(base, axis=1, keepdims=True)
            elif self.baseline_mode == "sequence_mean" and len(base):
                base = base - np.mean(base, axis=0, keepdims=True)
                mean_r = np.mean(base, axis=1, keepdims=True)
        centered = base - mean_r
        diff = np.vstack([np.zeros((1, base.shape[1])), np.diff(base, axis=0)]) if len(base) else base
        cdiff = np.vstack([np.zeros((1, base.shape[1])), np.diff(centered, axis=0)]) if len(base) else base
        parts = []
        if "raw" in thm:
            parts.append(base)
        if "centered" in thm or "centered_diff" in thm:
            parts.append(centered)
        if "diff" in thm or "centered_diff" in thm:
            parts.append(cdiff if "centered_diff" in thm and "diff" not in thm else diff)
        if "pooled_stats" in thm:
            p_stats = np.stack([
                np.std(base, axis=1),
                np.min(base, axis=1),
                np.max(base, axis=1),
            ], axis=1) if len(base) else np.zeros((0, 3))
            parts.append(p_stats)
        parts.append(mean_r)
        return np.concatenate(parts, axis=1), obs

    def _tof(self, rows, target, to_rate):
        _, _, _, tof, _ = self._modes()
        native = float(self.tof_native_sampling_rate)
        valid = np.isfinite(rows)
        any_valid = valid.any(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sensors = rows.reshape(len(rows), TOF_SHAPE[0], -1)
            pooled = np.stack([np.nanmean(rows, axis=1), np.nanstd(rows, axis=1), np.nanmin(rows, axis=1),
                               np.nanmax(rows, axis=1), valid.mean(axis=1)], axis=1)
            s_mean = np.nanmean(sensors, axis=2)
            s_std = np.nanstd(sensors, axis=2)
        pooled[~any_valid, :4] = np.nan
        pooled[:, 4] = np.where(any_valid | True, pooled[:, 4], 0.0)
        s_mean[~valid.reshape(len(rows), TOF_SHAPE[0], -1).any(axis=2)] = np.nan
        s_std[~valid.reshape(len(rows), TOF_SHAPE[0], -1).any(axis=2)] = np.nan
        parts = []
        if "raw" in tof:
            if self.tof_fill_mode == "far_500":
                raw = np.where(valid, rows, 500.0)
            elif self.tof_fill_mode == "far_255":
                raw = np.where(valid, rows, 255.0)
            elif self.tof_fill_mode == "zero":
                raw = np.where(valid, rows, 0.0)
            else:
                raw = rows
            parts.append(raw)
        stats = [pooled[:, :1]]
        if "pooled_stats" in tof:
            stats.append(pooled[:, 1:])
        if "sensor_stats" in tof:
            stats += [s_mean, s_std]
        parts.append(np.concatenate(stats, axis=1))
        if "block_2x2" in tof:
            b2, _ = compute_tof_block_channels(rows, block_size=2, include_temporal_diff=True)
            parts.append(b2)
        if "block_4x4" in tof:
            b4, _ = compute_tof_block_channels(rows, block_size=4, include_temporal_diff=True)
            parts.append(b4)
        values = np.concatenate(parts, axis=1)
        out, obs = to_rate(values, native, target, any_valid)
        if not valid.any():
            out = np.zeros_like(out)
        return out, obs

    # -- static features (real OSS backends + modular descriptors + registry families) --
    def _sources(self, chunks) -> Dict[str, Dict[str, np.ndarray]]:
        """Named 1-D series per chunk that static backends are computed from."""
        _, _, _, _, wanted = self._modes()
        names = chunks["channel_names"]
        rates = chunks["rates"]
        imu = chunks["branches"]["imu"]
        imu_mask = chunks["masks"]["imu"]
        obs_imu = chunks["obs_axes"].all(axis=-1) & imu_mask
        sources: Dict[str, Dict[str, np.ndarray]] = {}

        def channel(branch, label):
            return chunks["branches"][branch][:, :, names[branch].index(label)]

        def vector(prefix):
            idx = [names["imu"].index(f"{prefix}_{a}") for a in "xyz"]
            return imu[:, :, idx]

        if "acc_mag" in wanted:
            sources["acc_mag"] = {"x": np.linalg.norm(vector("acc"), axis=-1), "fs": rates["imu"], "obs": obs_imu,
                                  "mask": imu_mask, "branch": 0}
        if "lin_mag" in wanted and "lin_x" in names["imu"]:
            sources["lin_mag"] = {"x": np.linalg.norm(vector("lin"), axis=-1), "fs": rates["imu"], "obs": obs_imu,
                                  "mask": imu_mask, "branch": 0}
        if "ang_speed" in wanted and "ang_x" in names["imu"]:
            sources["ang_speed"] = {"x": np.linalg.norm(vector("ang"), axis=-1), "fs": rates["imu"],
                                    "obs": chunks["q_obs"] & imu_mask, "mask": imu_mask, "branch": 0}
        if "thm_mean" in wanted and "thm" in rates:
            sources["thm_mean"] = {"x": channel("thm", "thm_mean"), "fs": rates["thm"],
                                   "obs": chunks["obs_thm"] & chunks["masks"]["thm"],
                                   "mask": chunks["masks"]["thm"], "branch": 1}
        if "tof_mean" in wanted and "tof" in rates:
            sources["tof_mean"] = {"x": channel("tof", "tof_mean"), "fs": rates["tof"],
                                   "obs": chunks["obs_tof"] & chunks["masks"]["tof"],
                                   "mask": chunks["masks"]["tof"], "branch": 2}
        return sources

    def tsfel_config(self):
        """Real TSFEL feature dictionary for the requested domains / feature names."""
        try:
            return build_tsfel_config(domains=self.tsfel_domains, features=self.tsfel_features)
        except ValueError as exc:
            raise InvalidParams(str(exc)) from exc

    def tsfresh_settings(self, fs: float, n: int):
        """Real TSFresh settings whose FFT / autocorrelation indices follow the sensor's rate."""
        return build_tsfresh_fc_parameters(
            fs, n, profile=self.tsfresh_profile, fft_bands=self.tsfresh_fft_bands
        )

    def _external_static_features(self, sources: Mapping[str, Mapping[str, Any]]) -> Dict[str, np.ndarray]:
        """Dispatch static feature extraction to the real registered OSS backend."""
        backend_name = self.static_backend
        if backend_name in {None, "none"} or not sources:
            return {}
        backend = FEATURE_REGISTRY.get_backend(str(backend_name))
        cfg = {
            "tsfel_domains": self.tsfel_domains,
            "tsfel_features": self.tsfel_features,
            "tsfresh_profile": self.tsfresh_profile,
            "tsfresh_fft_bands": self.tsfresh_fft_bands,
            "min_observed_fraction": float(self.min_observed_fraction),
        }
        return backend.compute_features(sources, cfg)

    def _noop_features(self, chunks):
        opts = self.noop_options()
        if opts is None:
            return {}, None
        rates = chunks["rates"]
        names = chunks["channel_names"]["imu"]
        imu, imu_mask = chunks["branches"]["imu"], chunks["masks"]["imu"]
        acc_idx = [names.index(f"acc_{a}") for a in "xyz"]
        lin_idx = [names.index(f"lin_{a}") for a in "xyz"] if "lin_x" in names else None
        ang_idx = [names.index(f"ang_{a}") for a in "xyz"] if "ang_x" in names else None
        fs = rates["imu"]
        naming_mode = str(self.descriptor_naming)
        reports, rows, schema = [], [], None
        power = opts.branch_names() - {"imu"}
        motion, spectral = opts.mode_names()
        for i in range(len(imu)):
            L = int(imu_mask[i].sum())
            window = imu[i, :L].astype(float)
            present = chunks["present"][i, 0]
            nan3 = np.full((L, 3), np.nan)
            src = {
                "raw": window[:, acc_idx], "observed": chunks["obs_axes"][i, :L] & present,
                "linear": window[:, lin_idx] if lin_idx else nan3,
                "rotation": window[:, ang_idx] if ang_idx else nan3,
                "rotation_observed": chunks["q_obs"][i, :L] & present,
                "counter": np.arange(L, dtype=float), "dt": np.full(L, 1.0 / fs),
                "row_present": chunks["row_present"][i, :L],
            }
            feats, report = summarize_imu_window(src, fs, opts, naming_mode=naming_mode)
            if motion or spectral:
                for j, branch in ((1, "thm"), (2, "tof")):
                    if branch in power and branch in rates:
                        label = f"{branch}_mean"
                        length = int(chunks["masks"][branch][i].sum())
                        series = chunks["branches"][branch][i, :length, names_index(chunks, branch, label)][:, None]
                        observed = chunks[f"obs_{branch}"][i, :length] & chunks["present"][i, j]
                        feats.update(
                            summarize_branch_window(
                                series.astype(float), observed, rates[branch], opts, branch, naming_mode=naming_mode
                            )
                        )
            keys = list(feats)
            if schema is None:
                schema = keys
            elif schema != keys:
                raise AssertionError("Signal descriptor summary schema changed between windows")
            report["sequence_id"] = str(chunks["sequence_ids"][i])
            rows.append([feats[k] for k in keys])
            reports.append(report)
        if schema is None:
            schema = self._noop_schema_probe(chunks)
        arr = np.asarray(rows, dtype=np.float32).reshape(len(rows), len(schema))
        return {name: arr[:, j] for j, name in enumerate(schema)}, (reports if opts.quality is not None else None)

    def _noop_schema_probe(self, chunks):
        opts = self.noop_options()
        fs = chunks["rates"]["imu"]
        L = 32
        naming_mode = str(self.descriptor_naming)
        src = {"raw": np.zeros((L, 3)), "observed": np.zeros((L, 3), bool), "linear": np.zeros((L, 3)),
               "rotation": np.zeros((L, 3)), "rotation_observed": np.zeros(L, bool),
               "counter": np.arange(L, dtype=float), "dt": np.full(L, 1 / fs)}
        feats, _ = summarize_imu_window(src, fs, opts, naming_mode=naming_mode)
        motion, spectral = opts.mode_names()
        if motion or spectral:
            for branch in opts.branch_names() - {"imu"}:
                if branch in chunks["rates"]:
                    feats.update(
                        summarize_branch_window(
                            np.zeros((L, 1)), np.zeros(L, bool),
                            chunks["rates"][branch], opts, branch, naming_mode=naming_mode,
                        )
                    )
        return list(feats)

    def _family_config(self) -> Dict[str, Any]:
        return {
            "orientation_features": self.orientation_features,
            "vibration_features": self.vibration_features,
            "tof_spatial_features": self.tof_spatial_features,
            "tof_temporal_features": self.tof_temporal_features,
            "thermal_features": self.thermal_features,
            "cross_modal_features": self.cross_modal_features,
            "events_complexity_features": self.events_complexity_features,
            "extra_feature_families": self.extra_feature_families,
        }

    def featurize(self, chunks: Dict[str, Any]) -> Dict[str, Any]:
        """Attach static features (OSS backends + signal descriptors + registered families + presence flags)."""
        n = len(chunks["sequence_ids"])
        rates = chunks["rates"]
        manifest = FeatureManifest(
            algo_version=ALGO_VERSION,
            feature_version=FEATURE_VERSION,
            extractor_class=type(self).__name__,
            rates=dict(rates),
        )
        # Record temporal branch channels in manifest
        for b_name, ch_list in chunks["channel_names"].items():
            b_fs = float(rates.get(b_name, rates["imu"]))
            for ch in ch_list:
                manifest.add(
                    FeatureManifestEntry(
                        feature_name=f"{b_name}:{ch}",
                        canonical_name=f"{b_name}:{ch}",
                        family=f"branch_{b_name}",
                        library="scipy/numpy",
                        fs=b_fs,
                        units="sensor_native",
                        scope="sample",
                        version=FEATURE_VERSION,
                    )
                )

        columns: Dict[str, np.ndarray] = {}
        # 1. Real external open-source static backend
        backend = self.static_backend
        if backend not in {None, "none"} and n:
            sources = self._sources(chunks)
            if sources:
                ext_cols = self._external_static_features(sources)
                columns.update(ext_cols)
                for feat_name in ext_cols:
                    manifest.add(
                        FeatureManifestEntry(
                            feature_name=feat_name,
                            canonical_name=feat_name,
                            family=f"external_{backend}",
                            library=str(backend),
                            fs=float(rates["imu"]),
                            units="mixed",
                            scope="window",
                            version=FEATURE_VERSION,
                        )
                    )
        generic = list(columns)

        # 2. Modular scientific signal descriptors (legacy noop_* or canonical naming)
        noop_cols, reports = self._noop_features(chunks)
        columns.update(noop_cols)
        for feat_name in noop_cols:
            canon = LEGACY_TO_CANONICAL.get(feat_name, feat_name)
            legacy = CANONICAL_TO_LEGACY.get(feat_name, feat_name if feat_name.startswith("noop_") else None)
            manifest.add(
                FeatureManifestEntry(
                    feature_name=feat_name,
                    canonical_name=canon,
                    legacy_alias=legacy,
                    family="signal_descriptors",
                    library="scipy/numpy",
                    fs=float(rates["imu"]),
                    units="physical",
                    scope="window",
                    version=FEATURE_VERSION,
                )
            )

        # 3. Dynamically execute all enabled registered domain & extension feature families
        family_cols = FEATURE_REGISTRY.compute_enabled(chunks, self._family_config(), manifest=manifest)
        columns.update(family_cols)

        # 4. Optional multi-scale sub-window summaries (tsflex-backed)
        ms_scales = self._parsed_window_seconds_list()
        if ms_scales and n:
            ms_cols = compute_multiscale_window_summaries(
                chunks["branches"]["imu"],
                chunks["masks"]["imu"],
                chunks["channel_names"]["imu"],
                float(rates["imu"]),
                window_seconds_list=ms_scales,
            )
            columns.update(ms_cols)
            for feat_name in ms_cols:
                manifest.add(
                    FeatureManifestEntry(
                        feature_name=feat_name,
                        canonical_name=feat_name,
                        family="multiscale",
                        library="tsflex" if TSFLEX_AVAILABLE else "numpy",
                        fs=float(rates["imu"]),
                        units="m/s^2",
                        scope="window",
                        version=FEATURE_VERSION,
                    )
                )

        # 5. Optional per-sequence aggregation over window features (strictly intra-sequence)
        if self.sequence_aggregation_features and n:
            src_names = list(columns.keys())
            src_mat = (
                np.column_stack([np.asarray(columns[k], dtype=np.float64) for k in src_names])
                if src_names else np.zeros((n, 0), dtype=np.float64)
            )
            agg_mat, agg_names = aggregate_window_features_by_sequence(
                src_mat,
                src_names,
                chunks["sequence_ids"],
                window_starts_s=chunks.get("window_starts_s"),
                seq_durations_s=chunks.get("sequence_durations_s"),
            )
            for j, fn in enumerate(agg_names):
                columns[fn] = agg_mat[:, j]
                manifest.add(
                    FeatureManifestEntry(
                        feature_name=fn,
                        canonical_name=fn,
                        family="sequence_aggregation",
                        library="scipy.stats/numpy",
                        fs=None,
                        units="mixed",
                        scope="sequence",
                        version=FEATURE_VERSION,
                    )
                )

        # 6. Modality presence flags (when modalities!='all', emit only active modalities so IMU-only has zero THM/ToF cols)
        emit_mods = MODALITIES if self.modalities == "all" else tuple(m for m in MODALITIES if m in self.active_modalities())
        for j, name in enumerate(MODALITIES):
            if name in emit_mods:
                flag_name = f"present_{name}"
                columns[flag_name] = chunks["present"][:, j].astype(np.float32)
                manifest.add(
                    FeatureManifestEntry(
                        feature_name=flag_name,
                        canonical_name=flag_name,
                        family="modality_presence",
                        library="numpy",
                        fs=None,
                        units="bool",
                        scope="window",
                        version=FEATURE_VERSION,
                    )
                )

        expected = getattr(self, "static_names_", None)
        names = list(expected) if expected is not None else list(columns)
        matrix = np.zeros((n, len(names)), dtype=np.float32)
        for j, name in enumerate(names):
            if name in columns:
                matrix[:, j] = np.nan_to_num(np.asarray(columns[name], dtype=np.float64), nan=0.0,
                                             posinf=0.0, neginf=0.0).astype(np.float32)
        self.feature_manifest_ = manifest
        out = dict(chunks)
        out.update(
            static=matrix,
            static_names=names,
            noop_feature_names=list(noop_cols),
            generic_feature_names=generic,
            domain_feature_names=list(family_cols),
            feature_manifest=manifest,
        )
        if reports is not None:
            out["noop_quality"] = reports
        return out

    # -- scaling -----------------------------------------------------------------
    def fit_scalers(self, chunks: Dict[str, Any]):
        """Per-channel mean/std over present, real (unpadded) samples; static column z-scores."""
        self.channel_names_ = {k: list(v) for k, v in chunks["channel_names"].items()}
        self.rates_ = dict(chunks["rates"])
        self.scalers_ = {}
        obs_map = {
            "imu": chunks["obs_axes"].all(axis=-1) if "obs_axes" in chunks else None,
            "thm": chunks.get("obs_thm"),
            "tof": chunks.get("obs_tof"),
        }
        for j, name in enumerate(MODALITIES):
            if name not in chunks["branches"]:
                continue
            arr = chunks["branches"][name]
            mask = chunks["masks"][name] & chunks["present"][:, j][:, None]
            if self.scale_observed_only and obs_map.get(name) is not None:
                obs_mask = mask & obs_map[name]
                if obs_mask.any():
                    mask = obs_mask
            flat = arr[mask]
            if len(flat):
                mean, std = flat.mean(axis=0), flat.std(axis=0)
            else:
                mean, std = np.zeros(arr.shape[2]), np.ones(arr.shape[2])
            self.scalers_[name] = (mean.astype(np.float32), np.where(std > 1e-6, std, 1.0).astype(np.float32))
        static = chunks["static"]
        mean = static.mean(axis=0) if len(static) else np.zeros(static.shape[1])
        std = static.std(axis=0) if len(static) else np.ones(static.shape[1])
        self.static_scaler_ = (mean.astype(np.float32), np.where(std > 1e-6, std, 1.0).astype(np.float32))
        self.static_names_ = list(chunks["static_names"])
        self.noop_feature_names_ = list(chunks["noop_feature_names"])
        if "feature_manifest" in chunks:
            self.feature_manifest_ = chunks["feature_manifest"]
        return self

    def scale(self, chunks: Dict[str, Any]) -> Dict[str, Any]:
        """Standardise branch tensors and static features; absent modalities stay exactly zero."""
        out = dict(chunks)
        branches = {}
        for j, name in enumerate(MODALITIES):
            if name not in chunks["branches"]:
                continue
            mean, std = self.scalers_[name]
            arr = (chunks["branches"][name] - mean) / std
            arr = np.clip(arr, -self.static_clip * 5, self.static_clip * 5)
            arr = arr * chunks["masks"][name][:, :, None] * chunks["present"][:, j][:, None, None]
            branches[name] = arr.astype(np.float32)
        out["branches"] = branches
        mean, std = self.static_scaler_
        static = np.clip((chunks["static"] - mean) / std, -float(self.static_clip), float(self.static_clip))
        out["static"] = static.astype(np.float32)
        return out

    def select_static_top_k(self, scaled_chunks: Dict[str, Any], y_windows: np.ndarray, k: Optional[int]):
        """Keep the ``k`` static features with the highest ANOVA F-score against the labels.

        Must be called right after ``fit_scalers`` on TRAINING data only. It narrows
        ``static_names_`` and ``static_scaler_`` so every later ``featurize`` / ``scale`` /
        ``transform_chunks`` call (augmentation, validation, predict, importance) emits the
        same reduced column set. ``present_*`` flags are always kept and do not count toward ``k``.
        Returns ``(keep_idx, scores)`` where ``keep_idx`` indexes the pre-selection columns.
        """
        names = list(self.static_names_)
        protected = [j for j, n in enumerate(names) if n.startswith("present_")]
        candidates = [j for j in range(len(names)) if j not in set(protected)]
        y_windows = np.asarray(y_windows)
        if k is None or int(k) >= len(candidates) or len(np.unique(y_windows)) < 2:
            return np.arange(len(names)), None
        from sklearn.feature_selection import f_classif
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f_vals, _ = f_classif(np.asarray(scaled_chunks["static"])[:, candidates], y_windows)
        f_vals = np.nan_to_num(np.asarray(f_vals, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        order = np.argsort(-f_vals, kind="stable")[: int(k)]
        keep = np.array(sorted(set(protected) | {candidates[i] for i in order}), dtype=int)
        mean, std = self.static_scaler_
        self.static_scaler_ = (mean[keep], std[keep])
        self.static_names_ = [names[j] for j in keep]
        scores = {names[candidates[i]]: float(f_vals[i]) for i in range(len(candidates))}
        return keep, scores

    def select_finetuning_sequences(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Return the valid sequence dataframe for holdout/internal-validation evaluation."""
        return X.copy()


def names_index(chunks, branch, label):
    return chunks["channel_names"][branch].index(label)


# ---------------------------------------------------------------------------
# 5. Augmentation (swappable TsaugAdapter isolating archived tsaug)
# ---------------------------------------------------------------------------

AUGMENTATIONS = ("jitter", "noise", "scaling", "magnitude_warp", "time_warp", "time_shift", "crop",
                 "temporal_mask", "channel_drop")
_TIMING_OPS = {"time_warp", "time_shift", "crop"}


class TSAugmentor(BaseAugmentor):
    """Training-time augmentation of per-modality window tensors via ``TsaugAdapter``.

    Every knob is an sklearn parameter, so grid / Bayesian / evolutionary searches tune
    them directly.  All ``tsaug`` calls are isolated behind ``augmentation.TsaugAdapter``
    so the upstream archived dependency can be swapped cleanly.

    When ``per_modality_augmentation=True``, IMU receives Gaussian jitter/scaling with
    automatic unit-norm quaternion protection on ``rot_w/x/y/z``, while thermopile and
    ToF receive uniform noise and gentle drift/dropout so thermal baselines are not
    distorted by acceleration-scale Gaussian jitter.
    """

    def __init__(
        self, prob=0.0, per_aug_prob=0.5, augmentations=None, n_copies=1,
        jitter_sigma=0.0, noise_std=0.0, scaling_sigma=0.0,
        warp_sigma=0.0, warp_num_knots=4,
        time_warp_max_speed_ratio=1.0, time_warp_n_speed_change=3,
        time_shift_frac=0.0, crop_frac_range=(1.0, 1.0),
        temporal_num_masks=0, temporal_mask_frac=0.0, temporal_mask_fill="ffill",
        channel_drop_prob=0.0, sensor_drop_prob=0.0, sensor_drop_joint_prob=0.0,
        per_modality_augmentation=False,
        random_state=None,
    ):
        self.prob = prob
        self.per_aug_prob = per_aug_prob
        self.augmentations = augmentations
        self.n_copies = n_copies
        self.jitter_sigma = jitter_sigma
        self.noise_std = noise_std
        self.scaling_sigma = scaling_sigma
        self.warp_sigma = warp_sigma
        self.warp_num_knots = warp_num_knots
        self.time_warp_max_speed_ratio = time_warp_max_speed_ratio
        self.time_warp_n_speed_change = time_warp_n_speed_change
        self.time_shift_frac = time_shift_frac
        self.crop_frac_range = crop_frac_range
        self.temporal_num_masks = temporal_num_masks
        self.temporal_mask_frac = temporal_mask_frac
        self.temporal_mask_fill = temporal_mask_fill
        self.channel_drop_prob = channel_drop_prob
        self.sensor_drop_prob = sensor_drop_prob
        self.sensor_drop_joint_prob = sensor_drop_joint_prob
        self.per_modality_augmentation = per_modality_augmentation
        self.random_state = random_state

    # -- configuration -------------------------------------------------------
    def crop_range(self) -> Tuple[float, float]:
        value = _structured(self.crop_frac_range)
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise InvalidParams("crop_frac_range must be (low, high)")
        lo, hi = float(value[0]), float(value[1])
        if not 0 < lo <= hi <= 1:
            raise InvalidParams("crop_frac_range must satisfy 0 < low <= high <= 1")
        return lo, hi

    def allowed_ops(self) -> set:
        value = _structured(self.augmentations)
        if value is None:
            return set(AUGMENTATIONS)
        if isinstance(value, str):
            value = value.split("|")
        unknown = set(value) - set(AUGMENTATIONS)
        if unknown:
            raise InvalidParams(f"unknown augmentations {sorted(unknown)}; choose from {AUGMENTATIONS}")
        return set(value)

    def validate(self):
        for name in ("prob", "per_aug_prob", "channel_drop_prob", "sensor_drop_prob", "sensor_drop_joint_prob"):
            if not 0 <= float(getattr(self, name)) <= 1:
                raise InvalidParams(f"{name} must be in [0, 1]")
        if self.sensor_drop_prob >= 1 or self.channel_drop_prob >= 1:
            raise InvalidParams("removal probabilities must stay below 1")
        for name in ("jitter_sigma", "noise_std", "scaling_sigma", "warp_sigma", "time_shift_frac", "temporal_mask_frac"):
            if float(getattr(self, name)) < 0 or not np.isfinite(float(getattr(self, name))):
                raise InvalidParams(f"{name} must be a finite non-negative number")
        if float(self.time_warp_max_speed_ratio) < 1:
            raise InvalidParams("time_warp_max_speed_ratio must be >= 1 (1 disables time warping)")
        if int(self.warp_num_knots) < 1 or int(self.time_warp_n_speed_change) < 1:
            raise InvalidParams("warp knot / speed-change counts must be >= 1")
        if int(self.n_copies) < 0:
            raise InvalidParams("n_copies must be >= 0")
        self.crop_range()
        self.allowed_ops()
        return self

    def transform(self, X):
        return self.augment(X) if isinstance(X, Mapping) else X

    def active_ops(self) -> List[str]:
        allowed = self.allowed_ops()
        lo, hi = self.crop_range()
        strengths = {
            "jitter": self.jitter_sigma > 0, "noise": self.noise_std > 0, "scaling": self.scaling_sigma > 0,
            "magnitude_warp": self.warp_sigma > 0,
            "time_warp": self.time_warp_max_speed_ratio > 1, "time_shift": self.time_shift_frac > 0,
            "crop": hi < 1, "temporal_mask": self.temporal_num_masks > 0 and self.temporal_mask_frac > 0,
            "channel_drop": self.channel_drop_prob > 0,
        }
        return [op for op in AUGMENTATIONS if op in allowed and strengths[op]]

    def enabled(self) -> bool:
        return (self.n_copies > 0 and self.prob > 0 and self.per_aug_prob > 0 and bool(self.active_ops()))

    # -- operations ------------------------------------------------------------
    def augment_array(
        self,
        x,
        rng,
        gate,
        protect=None,
        reference_std=None,
        *,
        modality: str = "imu",
        channel_names: Optional[Sequence[str]] = None,
    ):
        """Augment ``x`` ``(N, T, C)`` -- all samples share the same length ``T``."""
        x = np.asarray(x, dtype=np.float64)
        n, length, channels = x.shape
        out = x.copy()
        protect = np.zeros(n, dtype=bool) if protect is None else np.asarray(protect, dtype=bool)
        ref = np.ones(channels) if reference_std is None else np.where(np.asarray(reference_std) > 1e-9, reference_std, 1.0)
        noise_distr = "uniform" if (self.per_modality_augmentation and modality in {"thm", "tof"}) else "gaussian"

        for op in self.active_ops():
            if self.per_modality_augmentation and modality in {"thm", "tof"} and op == "scaling":
                continue  # Avoid multiplicative gain distortion on calibrated degC / mm sensors
            fire = gate & (rng.random(n) < float(self.per_aug_prob))
            if op in _TIMING_OPS:
                fire &= ~protect
            idx = np.flatnonzero(fire)
            if not len(idx) or length < 4:
                continue
            sub = out[idx]
            seed = int(rng.integers(0, 2 ** 31 - 1))
            if op == "time_shift":
                sub = shift_time(sub, float(self.time_shift_frac), rng)
            elif op == "scaling":
                sub = sub * rng.normal(1.0, float(self.scaling_sigma), size=(len(idx), 1, channels))
            elif op == "jitter":
                sub = TsaugAdapter.add_noise(sub, float(self.jitter_sigma), distr=noise_distr, seed=seed, rng=rng)
            elif op == "noise":
                sub = TsaugAdapter.add_noise(sub / ref, float(self.noise_std), distr=noise_distr, seed=seed, rng=rng) * ref
            elif op == "magnitude_warp":
                sub = TsaugAdapter.multiplicative_drift(
                    sub, float(self.warp_sigma), int(self.warp_num_knots), seed=seed, rng=rng
                )
            elif op == "time_warp":
                sub = TsaugAdapter.time_warp(
                    sub, int(self.time_warp_n_speed_change), float(self.time_warp_max_speed_ratio), seed=seed, rng=rng
                )
            elif op == "crop":
                sub = TsaugAdapter.crop_resize(sub, self.crop_range(), seed=seed, rng=rng)
            elif op == "temporal_mask":
                sub = TsaugAdapter.temporal_dropout(
                    sub, int(self.temporal_num_masks), float(self.temporal_mask_frac), self.temporal_mask_fill,
                    seed=seed, rng=rng,
                )
            elif op == "channel_drop":
                sub = TsaugAdapter.channel_dropout(sub, float(self.channel_drop_prob), seed=seed, rng=rng)
            else:
                raise InvalidParams(f"operation {op!r} has no mapping")
            out[idx] = sub

        if self.per_modality_augmentation and modality == "imu" and channel_names is not None:
            out = TsaugAdapter.protect_quaternion_norm(out, channel_names)
        return out.astype(x.dtype if x.dtype == np.float32 else np.float64)

    def augment_chunks(self, chunks: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
        """Return a copy of raw (unscaled) chunks with augmented branch tensors."""
        self.validate()
        out = dict(chunks)
        n = len(chunks["sequence_ids"])
        gate = rng.random(n) < float(self.prob)
        branches = {}
        obs = {"imu": chunks["obs_axes"].all(axis=-1), "thm": chunks.get("obs_thm"), "tof": chunks.get("obs_tof")}
        ch_names_map = chunks.get("channel_names", {})
        for j, name in enumerate(MODALITIES):
            if name not in chunks["branches"]:
                continue
            arr, mask = chunks["branches"][name], chunks["masks"][name]
            lengths = mask.sum(axis=1)
            real = arr[mask & chunks["present"][:, j][:, None]]
            ref_std = real.std(axis=0) if len(real) else np.ones(arr.shape[2])
            observed = obs[name]
            partial = ((~observed) & mask).any(axis=1) if observed is not None else np.zeros(n, dtype=bool)
            new = arr.copy()
            usable = gate & chunks["present"][:, j]
            for length in np.unique(lengths):
                members = np.flatnonzero((lengths == length) & usable)
                if not len(members) or length < 4:
                    continue
                aug = self.augment_array(
                    arr[members, :length],
                    rng,
                    np.ones(len(members), dtype=bool),
                    protect=partial[members],
                    reference_std=ref_std,
                    modality=name,
                    channel_names=ch_names_map.get(name),
                )
                new[members, :length] = aug
            branches[name] = new.astype(np.float32)
        out["branches"] = branches
        return out

    def sample_modality_keep(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """``(n, 3)`` bool keep-matrix for [imu, thm, tof]; thm/tof drop independently or jointly."""
        keep = np.ones((n, 3), dtype=bool)
        if self.sensor_drop_joint_prob > 0:
            joint = rng.random(n) < float(self.sensor_drop_joint_prob)
            keep[joint, 1:] = False
        if self.sensor_drop_prob > 0:
            keep[:, 1] &= rng.random(n) >= float(self.sensor_drop_prob)
            keep[:, 2] &= rng.random(n) >= float(self.sensor_drop_prob)
        return keep


# ---------------------------------------------------------------------------
# 6. Network (tsai backbones per modality + spatial ToF + fusion + estimator)
# ---------------------------------------------------------------------------

TSAI_ARCHS = (
    "InceptionTimePlus", "ResNetPlus", "FCNPlus", "TCN",
    "TSTPlus", "GRUPlus", "LSTMPlus", "XCMPlus", "Conv1d",
)
FUSION_TYPES = ("concat", "attention", "gated")
POOLING_MODES = ("default", "concat_pool", "attention")
TOF_SPATIAL_MODES = ("auto", "conv2d", "flat")
ACTIVATIONS = ("relu", "gelu", "silu", "elu")
OPTIMIZERS = ("adamw", "adam", "sgd")
LOSS_NAMES = ("cross_entropy", "focal")
LR_SCHEDULERS = ("cosine", "plateau", "onecycle", "none")
SEQUENCE_AGGREGATIONS = ("mean_proba", "last_window", "majority_vote")

_TSAI_MODEL_CACHE: Optional[Dict[str, Any]] = None


def _get_tsai_classes() -> Dict[str, Any]:
    """Lazily import tsai model classes so module import stays fast."""
    global _TSAI_MODEL_CACHE
    if _TSAI_MODEL_CACHE is not None:
        return _TSAI_MODEL_CACHE
    _require(TSAI_AVAILABLE, "tsai")
    from tsai.models.all import (
        FCNPlus, GRUPlus, InceptionTimePlus, LSTMPlus,
        ResNetPlus, TCN, TSTPlus, XCMPlus,
    )
    _TSAI_MODEL_CACHE = {
        "InceptionTimePlus": InceptionTimePlus,
        "ResNetPlus": ResNetPlus,
        "FCNPlus": FCNPlus,
        "TCN": TCN,
        "TSTPlus": TSTPlus,
        "GRUPlus": GRUPlus,
        "LSTMPlus": LSTMPlus,
        "XCMPlus": XCMPlus,
    }
    return _TSAI_MODEL_CACHE


_ACT_CLASSES: Dict[str, type] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "elu": nn.ELU,
}


def _make_activation(name: str) -> nn.Module:
    key = str(name).lower()
    if key in _ACT_CLASSES:
        return _ACT_CLASSES[key]()
    raise InvalidParams(f"activation must be one of {ACTIVATIONS}, got {name!r}")


class _Conv1dBackbone(nn.Module):
    """Lightweight residual 1-D CNN backbone expecting (B, C_in, T) and returning (B, C_out)."""

    def __init__(self, c_in: int, c_out: int, nf: int = 64, depth: int = 3, ks: int = 7,
                 conv_dropout: float = 0.0, fc_dropout: float = 0.0, use_bn: bool = True,
                 concat_pool: bool = False, act: str = "gelu"):
        super().__init__()
        nf = max(4, int(nf))
        depth = max(1, int(depth))
        ks = max(3, int(ks)) | 1
        blocks: List[nn.Module] = []
        in_ch = int(c_in)
        for i in range(depth):
            out_ch = nf if i == 0 else nf
            block = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=ks, padding=ks // 2, bias=not use_bn),
                nn.BatchNorm1d(out_ch) if use_bn else nn.Identity(),
                _make_activation(act),
                nn.Dropout(float(conv_dropout)) if conv_dropout > 0 else nn.Identity(),
            )
            blocks.append(block)
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)
        self.concat_pool = bool(concat_pool)
        pool_dim = nf * 2 if self.concat_pool else nf
        self.head = nn.Sequential(
            nn.Dropout(float(fc_dropout)) if fc_dropout > 0 else nn.Identity(),
            nn.Linear(pool_dim, int(c_out)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for block in self.blocks:
            out = block(h)
            h = out + h if out.shape == h.shape else out
        if self.concat_pool:
            pooled = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=-1)
        else:
            pooled = h.mean(dim=-1)
        return self.head(pooled)


def build_tsai_backbone(
    arch: str, c_in: int, c_out: int, seq_len: int, *,
    filters: int = 64, depth: int = 3, kernel_size: int = 7,
    dropout: float = 0.1, conv_dropout: float = 0.0,
    num_heads: int = 4, use_batchnorm: bool = True,
    pooling_mode: str = "default", activation: str = "gelu",
    inception_bottleneck: bool = True, inception_residual: bool = True,
    conv_coord: bool = False, branch_sa: bool = False, branch_se: Optional[int] = None,
    fcn_separable: bool = False, fcn_residual: bool = True,
    xcm_window_perc: float = 1.0,
    tst_d_ff_mult: float = 2.0, tst_attn_dropout: float = 0.0,
    tst_res_attention: bool = True, tst_pre_norm: bool = False,
    tst_pe: Optional[str] = "zeros",
    rnn_bidirectional: bool = True,
) -> nn.Module:
    """Instantiate a ``tsai`` temporal backbone that maps ``(B, C_in, T) -> (B, C_out)``."""
    if arch not in TSAI_ARCHS:
        raise InvalidParams(f"arch must be one of {TSAI_ARCHS}, got {arch!r}")
    c_in = max(1, int(c_in))
    c_out = max(2, int(c_out))
    seq_len = max(4, int(seq_len))
    nf = max(4, int(filters))
    depth = max(1, int(depth))
    ks = max(3, int(kernel_size))
    fc_drop = float(dropout)
    c_drop = float(conv_dropout)
    concat_pool = (pooling_mode == "concat_pool")
    act_key = str(activation).lower()
    if act_key not in _ACT_CLASSES:
        raise InvalidParams(f"activation must be one of {ACTIVATIONS}, got {activation!r}")
    act_cls = _ACT_CLASSES[act_key]

    if arch == "Conv1d" or not TSAI_AVAILABLE:
        if arch != "Conv1d":
            _require(TSAI_AVAILABLE, "tsai")
        return _Conv1dBackbone(
            c_in, c_out, nf=nf, depth=depth, ks=ks,
            conv_dropout=c_drop, fc_dropout=fc_drop,
            use_bn=bool(use_batchnorm), concat_pool=concat_pool, act=activation,
        )

    classes = _get_tsai_classes()
    cls = classes[arch]
    se_val = int(branch_se) if branch_se is not None else None
    if arch == "InceptionTimePlus":
        return cls(
            c_in, c_out, seq_len=seq_len, nf=nf, depth=depth, ks=ks,
            bottleneck=bool(inception_bottleneck), residual=bool(inception_residual),
            coord=bool(conv_coord), sa=bool(branch_sa), se=se_val, act=act_cls,
            conv_dropout=c_drop, fc_dropout=fc_drop, bn=bool(use_batchnorm),
            concat_pool=concat_pool,
        )
    if arch == "ResNetPlus":
        return cls(
            c_in, c_out, seq_len=seq_len, nf=nf,
            coord=bool(conv_coord), sa=bool(branch_sa), se=se_val, act=act_cls,
            fc_dropout=fc_drop, concat_pool=concat_pool,
        )
    if arch == "FCNPlus":
        layers = [nf, nf * 2, nf]
        kss = [max(3, (ks - 2 * i) | 1) for i in range(3)]
        return cls(
            c_in, c_out, layers=layers, kss=kss,
            coord=bool(conv_coord), separable=bool(fcn_separable),
            use_bn=bool(use_batchnorm), fc_dropout=fc_drop,
            act=act_cls, residual=bool(fcn_residual),
        )
    if arch == "TCN":
        layers = [nf] * depth
        return cls(c_in, c_out, layers=layers, ks=max(2, ks), conv_dropout=c_drop, fc_dropout=fc_drop)
    if arch == "TSTPlus":
        heads = max(1, int(num_heads))
        d_model = max(heads * 2, (nf // heads) * heads)
        d_ff = max(8, int(round(d_model * float(tst_d_ff_mult))))
        return cls(
            c_in, c_out, seq_len=seq_len, max_seq_len=max(seq_len, 512),
            n_layers=depth, d_model=d_model, n_heads=heads, d_ff=d_ff,
            attn_dropout=float(tst_attn_dropout), dropout=c_drop, act=act_key,
            res_attention=bool(tst_res_attention), pre_norm=bool(tst_pre_norm),
            pe=tst_pe, fc_dropout=fc_drop, bn=bool(use_batchnorm),
            concat_pool=concat_pool,
        )
    if arch in {"GRUPlus", "LSTMPlus"}:
        return cls(
            c_in, c_out, seq_len=seq_len, hidden_size=nf, n_layers=depth,
            rnn_dropout=c_drop if depth > 1 else 0.0,
            bidirectional=bool(rnn_bidirectional),
            fc_dropout=fc_drop, bn=bool(use_batchnorm),
        )
    if arch == "XCMPlus":
        return cls(
            c_in, c_out, seq_len=seq_len, nf=nf,
            window_perc=float(xcm_window_perc),
            fc_dropout=fc_drop, bn=bool(use_batchnorm), concat_pool=concat_pool,
        )
    raise InvalidParams(f"Unsupported architecture {arch!r}")


class ToFSpatialEncoder(nn.Module):
    """2-D spatial encoder over the 5 Time-of-Flight 8x8 pixel grids per timestep.

    When raw ToF pixels are enabled (320 channels = 5 sensors x 8 x 8), reshapes the
    first 320 channels of each timestep into ``(B * T, 5, 8, 8)``, applies 2-D spatial
    convolutions across the 8x8 sensor grids, and concatenates the pooled spatial
    features with any remaining ToF summary channels.
    """

    def __init__(self, in_channels: int, spatial_filters: int = 32, spatial_kernel: int = 3,
                 use_batchnorm: bool = True, activation: str = "gelu"):
        super().__init__()
        self.raw_pixels = int(np.prod(TOF_SHAPE))  # 320 = 5 * 8 * 8
        if in_channels < self.raw_pixels:
            raise ValueError(f"ToFSpatialEncoder requires at least {self.raw_pixels} channels, got {in_channels}")
        self.extra_channels = int(in_channels) - self.raw_pixels
        sf = max(4, int(spatial_filters))
        k = max(1, int(spatial_kernel))
        pad = k // 2
        self.conv2d = nn.Sequential(
            nn.Conv2d(TOF_SHAPE[0], sf, kernel_size=k, padding=pad, bias=not use_batchnorm),
            nn.BatchNorm2d(sf) if use_batchnorm else nn.Identity(),
            _make_activation(activation),
            nn.Conv2d(sf, sf, kernel_size=k, padding=pad, bias=not use_batchnorm),
            nn.BatchNorm2d(sf) if use_batchnorm else nn.Identity(),
            _make_activation(activation),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.out_channels = sf + self.extra_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        b, t, _ = x.shape
        raw = x[:, :, :self.raw_pixels].contiguous().view(b * t, TOF_SHAPE[0], TOF_SHAPE[1], TOF_SHAPE[2])
        spatial = self.conv2d(raw).view(b, t, -1)
        if self.extra_channels > 0:
            return torch.cat([spatial, x[:, :, self.raw_pixels:]], dim=-1)
        return spatial


class TemporalAttentionPool(nn.Module):
    """Masked additive attention over timesteps ``(B, T, C) -> (B, out_dim)``."""

    def __init__(self, in_channels: int, attn_units: int, out_dim: int, activation: str = "gelu"):
        super().__init__()
        u = max(8, int(attn_units))
        self.score = nn.Sequential(
            nn.Linear(int(in_channels), u),
            nn.Tanh(),
            nn.Linear(u, 1, bias=False),
        )
        self.proj = nn.Sequential(
            nn.Linear(int(in_channels), int(out_dim)),
            _make_activation(activation),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C), mask: (B, T) bool
        logits = self.score(x).squeeze(-1)
        valid = mask.bool()
        any_valid = valid.any(dim=1, keepdim=True)
        safe_mask = torch.where(any_valid, valid, torch.ones_like(valid))
        logits = logits.masked_fill(~safe_mask, -1e9)
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)
        pooled = (x * weights).sum(dim=1)
        pooled = pooled * any_valid.to(x.dtype)
        return self.proj(pooled)


class ModalityBranch(nn.Module):
    """Per-modality temporal branch combining optional spatial encoding, tsai backbone, and attention."""

    def __init__(
        self, name: str, in_channels: int, seq_len: int, out_dim: int, *,
        arch: str = "InceptionTimePlus", filters: int = 64, depth: int = 3,
        kernel_size: int = 7, dropout: float = 0.1, conv_dropout: float = 0.0,
        num_heads: int = 4, use_batchnorm: bool = True, pooling_mode: str = "default",
        attention_units: int = 64, tof_spatial_mode: str = "auto",
        tof_spatial_filters: int = 32, tof_spatial_kernel: int = 3,
        activation: str = "gelu",
        inception_bottleneck: bool = True, inception_residual: bool = True,
        conv_coord: bool = False, branch_sa: bool = False, branch_se: Optional[int] = None,
        fcn_separable: bool = False, fcn_residual: bool = True,
        xcm_window_perc: float = 1.0,
        tst_d_ff_mult: float = 2.0, tst_attn_dropout: float = 0.0,
        tst_res_attention: bool = True, tst_pre_norm: bool = False,
        tst_pe: Optional[str] = "zeros",
        rnn_bidirectional: bool = True,
    ):
        super().__init__()
        self.name = name
        self.spatial: Optional[ToFSpatialEncoder] = None
        eff_channels = int(in_channels)
        use_spatial = (
            name == "tof"
            and tof_spatial_mode in {"auto", "conv2d"}
            and in_channels >= int(np.prod(TOF_SHAPE))
            and (tof_spatial_mode == "conv2d" or in_channels >= int(np.prod(TOF_SHAPE)))
        )
        if name == "tof" and tof_spatial_mode == "conv2d" and in_channels < int(np.prod(TOF_SHAPE)):
            # Graceful fallback when raw ToF pixels were not extracted
            use_spatial = False
        if use_spatial:
            self.spatial = ToFSpatialEncoder(
                in_channels, spatial_filters=tof_spatial_filters,
                spatial_kernel=tof_spatial_kernel, use_batchnorm=use_batchnorm,
                activation=activation,
            )
            eff_channels = self.spatial.out_channels

        self.backbone = build_tsai_backbone(
            arch, eff_channels, out_dim, seq_len,
            filters=filters, depth=depth, kernel_size=kernel_size,
            dropout=dropout, conv_dropout=conv_dropout, num_heads=num_heads,
            use_batchnorm=use_batchnorm, pooling_mode=pooling_mode, activation=activation,
            inception_bottleneck=inception_bottleneck, inception_residual=inception_residual,
            conv_coord=conv_coord, branch_sa=branch_sa, branch_se=branch_se,
            fcn_separable=fcn_separable, fcn_residual=fcn_residual,
            xcm_window_perc=xcm_window_perc,
            tst_d_ff_mult=tst_d_ff_mult, tst_attn_dropout=tst_attn_dropout,
            tst_res_attention=tst_res_attention, tst_pre_norm=tst_pre_norm,
            tst_pe=tst_pe, rnn_bidirectional=rnn_bidirectional,
        )
        self.attn_pool: Optional[TemporalAttentionPool] = None
        if pooling_mode == "attention":
            self.attn_pool = TemporalAttentionPool(eff_channels, attention_units, out_dim, activation=activation)
        self.norm = nn.LayerNorm(int(out_dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C), mask: (B, T), present: (B,)
        m = mask.unsqueeze(-1).to(x.dtype)
        x = x * m
        if self.spatial is not None:
            x = self.spatial(x) * m
        # tsai expects (B, C, T)
        out = self.backbone(x.transpose(1, 2))
        if self.attn_pool is not None:
            out = out + self.attn_pool(x, mask)
        out = self.norm(out)
        return out * present.unsqueeze(-1).to(out.dtype)


class MultiBranchTSAINet(nn.Module):
    """Multimodal tsai network fusing IMU (100 Hz), Thermopile (20 Hz), ToF (20 Hz), and Static features."""

    def __init__(
        self, branch_specs: Mapping[str, Tuple[int, int]], static_dim: int, n_classes: int, *,
        arch: str = "InceptionTimePlus",
        imu_arch: Optional[str] = None, thm_arch: Optional[str] = None, tof_arch: Optional[str] = None,
        branch_filters: int = 64, branch_depth: int = 3, branch_kernel_size: int = 7,
        branch_dropout: float = 0.1, conv_dropout: float = 0.0, branch_units: int = 128,
        pooling_mode: str = "default", attention_units: int = 64,
        tof_spatial_mode: str = "auto", tof_spatial_filters: int = 32, tof_spatial_kernel: int = 3,
        static_units: int = 64, static_layers: int = 2, static_dropout: float = 0.1,
        fusion_type: str = "attention", fusion_units: int = 128, fusion_dropout: float = 0.1,
        attention_heads: int = 4, dense_units: int = 128, dense_layers: int = 2,
        dropout: float = 0.2, activation: str = "gelu", use_batchnorm: bool = True,
        inception_bottleneck: bool = True, inception_residual: bool = True,
        conv_coord: bool = False, branch_sa: bool = False, branch_se: Optional[int] = None,
        fcn_separable: bool = False, fcn_residual: bool = True,
        xcm_window_perc: float = 1.0,
        tst_d_ff_mult: float = 2.0, tst_attn_dropout: float = 0.0,
        tst_res_attention: bool = True, tst_pre_norm: bool = False,
        tst_pe: Optional[str] = "zeros",
        rnn_bidirectional: bool = True,
    ):
        super().__init__()
        self.branch_order = [name for name in MODALITIES if name in branch_specs]
        arch_map = {
            "imu": imu_arch or arch,
            "thm": thm_arch or arch,
            "tof": tof_arch or arch,
        }
        b_units = max(8, int(branch_units))
        self.branches = nn.ModuleDict()
        for name in self.branch_order:
            seq_len, n_ch = branch_specs[name]
            self.branches[name] = ModalityBranch(
                name, n_ch, seq_len, b_units,
                arch=arch_map[name], filters=branch_filters, depth=branch_depth,
                kernel_size=branch_kernel_size, dropout=branch_dropout,
                conv_dropout=conv_dropout, num_heads=attention_heads,
                use_batchnorm=use_batchnorm, pooling_mode=pooling_mode,
                attention_units=attention_units, tof_spatial_mode=tof_spatial_mode,
                tof_spatial_filters=tof_spatial_filters, tof_spatial_kernel=tof_spatial_kernel,
                activation=activation,
                inception_bottleneck=inception_bottleneck, inception_residual=inception_residual,
                conv_coord=conv_coord, branch_sa=branch_sa, branch_se=branch_se,
                fcn_separable=fcn_separable, fcn_residual=fcn_residual,
                xcm_window_perc=xcm_window_perc,
                tst_d_ff_mult=tst_d_ff_mult, tst_attn_dropout=tst_attn_dropout,
                tst_res_attention=tst_res_attention, tst_pre_norm=tst_pre_norm,
                tst_pe=tst_pe, rnn_bidirectional=rnn_bidirectional,
            )

        self.static_dim = int(static_dim)
        s_units = max(8, int(static_units)) if self.static_dim > 0 else 0
        if self.static_dim > 0 and int(static_layers) > 0:
            s_blocks: List[nn.Module] = []
            in_d = self.static_dim
            for _ in range(int(static_layers)):
                s_blocks.extend([
                    nn.Linear(in_d, s_units),
                    nn.LayerNorm(s_units),
                    _make_activation(activation),
                    nn.Dropout(float(static_dropout)) if static_dropout > 0 else nn.Identity(),
                ])
                in_d = s_units
            self.static_net: Optional[nn.Module] = nn.Sequential(*s_blocks)
            self.static_out_dim = s_units
        else:
            self.static_net = None
            self.static_out_dim = 0

        self.fusion_type = str(fusion_type)
        f_units = max(8, int(fusion_units))
        n_temporal = len(self.branch_order)
        has_static = self.static_out_dim > 0
        n_tokens = n_temporal + (1 if has_static else 0)
        concat_dim = n_temporal * b_units + self.static_out_dim + 3

        if self.fusion_type == "attention":
            heads = max(1, int(attention_heads))
            while f_units % heads != 0 and heads > 1:
                heads -= 1
            self.token_projs = nn.ModuleDict({
                name: nn.Linear(b_units, f_units) for name in self.branch_order
            })
            if has_static:
                self.token_projs["static"] = nn.Linear(self.static_out_dim, f_units)
            self.modality_emb = nn.Parameter(torch.randn(1, n_tokens, f_units) * 0.02)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=f_units, num_heads=heads,
                dropout=float(fusion_dropout), batch_first=True,
            )
            self.fusion_norm = nn.LayerNorm(f_units)
            self.fusion_proj = nn.Sequential(
                nn.Linear(f_units + 3, f_units),
                _make_activation(activation),
                nn.Dropout(float(fusion_dropout)) if fusion_dropout > 0 else nn.Identity(),
            )
        elif self.fusion_type == "gated":
            self.gate_net = nn.Sequential(
                nn.Linear(concat_dim, f_units),
                _make_activation(activation),
                nn.Linear(f_units, n_tokens),
                nn.Sigmoid(),
            )
            self.token_projs = nn.ModuleDict({
                name: nn.Linear(b_units, f_units) for name in self.branch_order
            })
            if has_static:
                self.token_projs["static"] = nn.Linear(self.static_out_dim, f_units)
            self.fusion_proj = nn.Sequential(
                nn.LayerNorm(f_units),
                _make_activation(activation),
                nn.Dropout(float(fusion_dropout)) if fusion_dropout > 0 else nn.Identity(),
            )
        elif self.fusion_type == "concat":
            self.fusion_proj = nn.Sequential(
                nn.Linear(concat_dim, f_units),
                nn.LayerNorm(f_units),
                _make_activation(activation),
                nn.Dropout(float(fusion_dropout)) if fusion_dropout > 0 else nn.Identity(),
            )
        else:
            raise InvalidParams(f"fusion_type must be one of {FUSION_TYPES}, got {fusion_type!r}")

        head_blocks: List[nn.Module] = []
        in_h = f_units
        d_units = max(8, int(dense_units))
        for layer_idx in range(max(1, int(dense_layers))):
            out_h = d_units if layer_idx == 0 else max(8, d_units // (2 if layer_idx > 1 else 1))
            head_blocks.extend([
                nn.Linear(in_h, out_h, bias=not use_batchnorm),
                nn.BatchNorm1d(out_h) if use_batchnorm else nn.LayerNorm(out_h),
                _make_activation(activation),
                nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            ])
            in_h = out_h
        head_blocks.append(nn.Linear(in_h, int(n_classes)))
        self.classifier = nn.Sequential(*head_blocks)

    def encode_with_branches(
        self, branches: Mapping[str, torch.Tensor], masks: Mapping[str, torch.Tensor],
        present: torch.Tensor, static: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Compute per-modality embeddings and fused latent vector before the classification head."""
        mod_index = {"imu": 0, "thm": 1, "tof": 2}
        embs: Dict[str, torch.Tensor] = {}
        pres_flags: List[torch.Tensor] = []
        for name in self.branch_order:
            j = mod_index[name]
            p_j = present[:, j].bool()
            embs[name] = self.branches[name](branches[name], masks[name], p_j)
            pres_flags.append(p_j)

        static_emb: Optional[torch.Tensor] = None
        if self.static_net is not None and static is not None and static.shape[1] > 0:
            static_emb = self.static_net(static)

        pres_float = present.to(next(iter(embs.values())).dtype)
        if self.fusion_type == "concat":
            parts = [embs[n] for n in self.branch_order]
            if static_emb is not None:
                parts.append(static_emb)
            parts.append(pres_float)
            fused = self.fusion_proj(torch.cat(parts, dim=-1))
        elif self.fusion_type == "attention":
            tokens = [self.token_projs[n](embs[n]) for n in self.branch_order]
            token_present = list(pres_flags)
            if static_emb is not None:
                tokens.append(self.token_projs["static"](static_emb))
                token_present.append(torch.ones_like(pres_flags[0], dtype=torch.bool))
            seq = torch.stack(tokens, dim=1) + self.modality_emb
            pad_mask = ~torch.stack(token_present, dim=1)
            # Ensure at least the first token is unmasked per row to avoid NaN softmax
            all_masked = pad_mask.all(dim=1)
            if all_masked.any():
                pad_mask = pad_mask.clone()
                pad_mask[all_masked, 0] = False
            attn_out, _ = self.cross_attn(seq, seq, seq, key_padding_mask=pad_mask, need_weights=False)
            seq = self.fusion_norm(seq + attn_out)
            valid_tok = (~pad_mask).unsqueeze(-1).to(seq.dtype)
            pooled = (seq * valid_tok).sum(dim=1) / valid_tok.sum(dim=1).clamp_min(1.0)
            fused = self.fusion_proj(torch.cat([pooled, pres_float], dim=-1))
        else:  # gated
            raw_parts = [embs[n] for n in self.branch_order]
            tokens = [self.token_projs[n](embs[n]) for n in self.branch_order]
            token_present = list(pres_flags)
            if static_emb is not None:
                raw_parts.append(static_emb)
                tokens.append(self.token_projs["static"](static_emb))
                token_present.append(torch.ones_like(pres_flags[0], dtype=torch.bool))
            raw_parts.append(pres_float)
            gates = self.gate_net(torch.cat(raw_parts, dim=-1))
            valid_tok = torch.stack(token_present, dim=1).to(gates.dtype)
            gates = gates * valid_tok
            stacked = torch.stack(tokens, dim=1)
            gated_sum = (stacked * gates.unsqueeze(-1)).sum(dim=1) / gates.sum(dim=1, keepdim=True).clamp_min(1e-6)
            fused = self.fusion_proj(gated_sum)

        return embs, fused

    def forward(
        self, branches: Mapping[str, torch.Tensor], masks: Mapping[str, torch.Tensor],
        present: torch.Tensor, static: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, fused = self.encode_with_branches(branches, masks, present, static)
        return self.classifier(fused)


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional class weights and label smoothing."""

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("weight", weight)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(
            logits, targets, reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce).clamp(1e-7, 1.0 - 1e-7)
        focal = ((1.0 - pt) ** self.gamma) * ce
        if self.weight is not None:
            focal = focal * self.weight[targets]
        return focal.mean()


def _add_select_finetuning_sequences():
    """Attach ``select_finetuning_sequences`` to ``MultiModalSequenceExtractor`` for notebook compatibility."""
    if not hasattr(MultiModalSequenceExtractor, "select_finetuning_sequences"):
        def select_finetuning_sequences(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> pd.DataFrame:
            return X
        MultiModalSequenceExtractor.select_finetuning_sequences = select_finetuning_sequences  # type: ignore[attr-defined]


_add_select_finetuning_sequences()


class MultiBranchTSAIClassifier(PretrainingAndSSLMixin, BaseSequenceClassifier):
    """Sklearn-compatible multimodal time-series classifier built on ``tsai``, ``tsaug``, and ``tsfel``/``tsfresh`` + ``NOOP``.

    Forwards ``extractor__*`` (including ``extractor__noop__*``) and ``augmentor__*`` via
    ``get_params`` / ``set_params`` so a single ``Pipeline([('estimator', clf)])`` exposes
    every hyperparameter in one unified namespace.
    """

    def __init__(
        self,
        extractor: Optional[MultiModalSequenceExtractor] = None,
        augmentor: Optional[TSAugmentor] = None,
        # --- tsai backbone & branch architecture ---
        arch: str = "InceptionTimePlus",
        imu_arch: Optional[str] = None,
        thm_arch: Optional[str] = None,
        tof_arch: Optional[str] = None,
        branch_filters: int = 64,
        branch_depth: int = 3,
        branch_kernel_size: int = 7,
        branch_dropout: float = 0.1,
        conv_dropout: float = 0.0,
        branch_units: int = 128,
        pooling_mode: str = "default",
        attention_units: int = 64,
        tof_spatial_mode: str = "auto",
        tof_spatial_filters: int = 32,
        tof_spatial_kernel: int = 3,
        # --- tsai backbone-specific knobs ---
        inception_bottleneck: bool = True,
        inception_residual: bool = True,
        conv_coord: bool = False,
        branch_sa: bool = False,
        branch_se: Optional[int] = None,
        fcn_separable: bool = False,
        fcn_residual: bool = True,
        xcm_window_perc: float = 1.0,
        tst_d_ff_mult: float = 2.0,
        tst_attn_dropout: float = 0.0,
        tst_res_attention: bool = True,
        tst_pre_norm: bool = False,
        tst_pe: Optional[str] = "zeros",
        rnn_bidirectional: bool = True,
        # --- static branch & multimodal fusion ---
        static_units: int = 64,
        static_layers: int = 2,
        static_dropout: float = 0.1,
        static_top_k: Optional[int] = None,
        fusion_type: str = "attention",
        fusion_units: int = 128,
        fusion_dropout: float = 0.1,
        attention_heads: int = 4,
        # --- classification head ---
        dense_units: int = 128,
        dense_layers: int = 2,
        dropout: float = 0.2,
        activation: str = "gelu",
        use_batchnorm: bool = True,
        # --- optimization & regularization ---
        epochs: int = 40,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer_name: str = "adamw",
        momentum: float = 0.9,
        loss_name: str = "cross_entropy",
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
        class_weight_mode: Optional[str] = "balanced",
        use_lr_scheduler: bool = True,
        lr_scheduler_type: str = "cosine",
        lr_factor: float = 0.5,
        lr_patience: int = 5,
        min_lr: float = 1e-6,
        cosine_t_max: int = 40,
        warmup_epochs: int = 0,
        gradient_clip_norm: Optional[float] = 1.0,
        early_stopping_patience: int = 10,
        restore_best_weights: bool = True,
        # --- validation & evaluation ---
        validation_split: float = 0.15,
        validation_group_column: Optional[str] = None,
        sequence_aggregation: str = "mean_proba",
        primary_target: str = "bfrb",
        # --- pretraining & masked-signal SSL (0 / 'none' = disabled) ---
        pretraining_mode: str = "none",
        pretraining_source: str = "all",
        pretraining_label_column: str = "gesture",
        pretraining_epochs: int = 0,
        pretraining_learning_rate: float = 0.0,
        pretraining_batch_size: int = 0,
        pretraining_validation_split: float = 0.0,
        pretraining_early_stopping_patience: int = 0,
        finetuning_head_epochs: int = 0,
        finetuning_head_learning_rate: float = 0.0,
        finetuning_learning_rate: float = 0.0,
        ssl_mask_ratio: float = 0.0,
        ssl_span_length: int = 0,
        ssl_decoder_units: int = 0,
        ssl_decoder_dropout: float = 0.0,
        ssl_loss_name: str = "mse",
        ssl_resample_mask_each_epoch: bool = False,
        ssl_mask_modalities_together: bool = False,
        invalid_candidate_policy: str = "penalize",
        invalid_candidate_score: float = -1.0,
        candidate_failure_log: Optional[str] = None,
        verbose: int = 0,
        random_state: int = 42,
        device: str = "auto",
    ):
        self.extractor = extractor
        self.augmentor = augmentor
        self.arch = arch
        self.imu_arch = imu_arch
        self.thm_arch = thm_arch
        self.tof_arch = tof_arch
        self.branch_filters = branch_filters
        self.branch_depth = branch_depth
        self.branch_kernel_size = branch_kernel_size
        self.branch_dropout = branch_dropout
        self.conv_dropout = conv_dropout
        self.branch_units = branch_units
        self.pooling_mode = pooling_mode
        self.attention_units = attention_units
        self.tof_spatial_mode = tof_spatial_mode
        self.tof_spatial_filters = tof_spatial_filters
        self.tof_spatial_kernel = tof_spatial_kernel
        self.inception_bottleneck = inception_bottleneck
        self.inception_residual = inception_residual
        self.conv_coord = conv_coord
        self.branch_sa = branch_sa
        self.branch_se = branch_se
        self.fcn_separable = fcn_separable
        self.fcn_residual = fcn_residual
        self.xcm_window_perc = xcm_window_perc
        self.tst_d_ff_mult = tst_d_ff_mult
        self.tst_attn_dropout = tst_attn_dropout
        self.tst_res_attention = tst_res_attention
        self.tst_pre_norm = tst_pre_norm
        self.tst_pe = tst_pe
        self.rnn_bidirectional = rnn_bidirectional
        self.static_units = static_units
        self.static_layers = static_layers
        self.static_dropout = static_dropout
        self.static_top_k = static_top_k
        self.fusion_type = fusion_type
        self.fusion_units = fusion_units
        self.fusion_dropout = fusion_dropout
        self.attention_heads = attention_heads
        self.dense_units = dense_units
        self.dense_layers = dense_layers
        self.dropout = dropout
        self.activation = activation
        self.use_batchnorm = use_batchnorm
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.momentum = momentum
        self.loss_name = loss_name
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.class_weight_mode = class_weight_mode
        self.use_lr_scheduler = use_lr_scheduler
        self.lr_scheduler_type = lr_scheduler_type
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.min_lr = min_lr
        self.cosine_t_max = cosine_t_max
        self.warmup_epochs = warmup_epochs
        self.gradient_clip_norm = gradient_clip_norm
        self.early_stopping_patience = early_stopping_patience
        self.restore_best_weights = restore_best_weights
        self.validation_split = validation_split
        self.validation_group_column = validation_group_column
        self.sequence_aggregation = sequence_aggregation
        self.primary_target = primary_target
        self.pretraining_mode = pretraining_mode
        self.pretraining_source = pretraining_source
        self.pretraining_label_column = pretraining_label_column
        self.pretraining_epochs = pretraining_epochs
        self.pretraining_learning_rate = pretraining_learning_rate
        self.pretraining_batch_size = pretraining_batch_size
        self.pretraining_validation_split = pretraining_validation_split
        self.pretraining_early_stopping_patience = pretraining_early_stopping_patience
        self.finetuning_head_epochs = finetuning_head_epochs
        self.finetuning_head_learning_rate = finetuning_head_learning_rate
        self.finetuning_learning_rate = finetuning_learning_rate
        self.ssl_mask_ratio = ssl_mask_ratio
        self.ssl_span_length = ssl_span_length
        self.ssl_decoder_units = ssl_decoder_units
        self.ssl_decoder_dropout = ssl_decoder_dropout
        self.ssl_loss_name = ssl_loss_name
        self.ssl_resample_mask_each_epoch = ssl_resample_mask_each_epoch
        self.ssl_mask_modalities_together = ssl_mask_modalities_together
        self.invalid_candidate_policy = invalid_candidate_policy
        self.invalid_candidate_score = invalid_candidate_score
        self.candidate_failure_log = candidate_failure_log
        self.verbose = verbose
        self.random_state = random_state
        self.device = device

    # -- nested parameter plumbing --------------------------------------------
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        params = super().get_params(deep=deep)
        if deep:
            if self.extractor is None:
                ext = MultiModalSequenceExtractor(noop=NoopOptions())
                params.update({f"extractor__{k}": v for k, v in ext.get_params(deep=True).items()})
            if self.augmentor is None:
                aug = TSAugmentor()
                params.update({f"augmentor__{k}": v for k, v in aug.get_params(deep=True).items()})
        return params

    def set_params(self, **params):
        params = dict(params)
        if any(k.startswith("extractor__") for k in params) and params.get("extractor", self.extractor) is None:
            params["extractor"] = MultiModalSequenceExtractor(noop=NoopOptions())
        if any(k.startswith("augmentor__") for k in params) and params.get("augmentor", self.augmentor) is None:
            params["augmentor"] = TSAugmentor()
        for key, val in list(params.items()):
            if isinstance(val, str) and (val.startswith("[") or val.startswith("{") or val.startswith("(")):
                params[key] = _structured(val)
        return super().set_params(**params)

    # -- validation -----------------------------------------------------------
    def _build_subcomponents(self) -> Tuple[MultiModalSequenceExtractor, TSAugmentor]:
        ext = clone(self.extractor) if self.extractor is not None else MultiModalSequenceExtractor(noop=NoopOptions())
        if ext.noop is None:
            ext.noop = NoopOptions()
        aug = clone(self.augmentor) if self.augmentor is not None else TSAugmentor(random_state=self.random_state)
        ext.validate()
        aug.validate()
        return ext, aug

    def validate(self):
        self._build_subcomponents()
        for name, val in (("arch", self.arch), ("imu_arch", self.imu_arch),
                          ("thm_arch", self.thm_arch), ("tof_arch", self.tof_arch)):
            if val is not None and val not in TSAI_ARCHS:
                raise InvalidParams(f"{name} must be one of {TSAI_ARCHS}, got {val!r}")
        if self.fusion_type not in FUSION_TYPES:
            raise InvalidParams(f"fusion_type must be one of {FUSION_TYPES}")
        if self.pooling_mode not in POOLING_MODES:
            raise InvalidParams(f"pooling_mode must be one of {POOLING_MODES}")
        if self.tof_spatial_mode not in TOF_SPATIAL_MODES:
            raise InvalidParams(f"tof_spatial_mode must be one of {TOF_SPATIAL_MODES}")
        if str(self.activation).lower() not in ACTIVATIONS:
            raise InvalidParams(f"activation must be one of {ACTIVATIONS}")
        if str(self.optimizer_name).lower() not in OPTIMIZERS:
            raise InvalidParams(f"optimizer_name must be one of {OPTIMIZERS}")
        if str(self.loss_name).lower() not in LOSS_NAMES:
            raise InvalidParams(f"loss_name must be one of {LOSS_NAMES}")
        if str(self.lr_scheduler_type).lower() not in LR_SCHEDULERS:
            raise InvalidParams(f"lr_scheduler_type must be one of {LR_SCHEDULERS}")
        if self.sequence_aggregation not in SEQUENCE_AGGREGATIONS:
            raise InvalidParams(f"sequence_aggregation must be one of {SEQUENCE_AGGREGATIONS}")
        if self.class_weight_mode not in {None, "none", "balanced"}:
            raise InvalidParams("class_weight_mode must be None, 'none', or 'balanced'")
        if int(self.epochs) < 1 or int(self.batch_size) < 1:
            raise InvalidParams("epochs and batch_size must be >= 1")
        if self.static_top_k is not None and int(self.static_top_k) < 1:
            raise InvalidParams("static_top_k must be None or >= 1")
        if float(self.learning_rate) <= 0 or not np.isfinite(float(self.learning_rate)):
            raise InvalidParams("learning_rate must be positive and finite")
        if float(self.weight_decay) < 0:
            raise InvalidParams("weight_decay must be >= 0")
        for drop_name in ("branch_dropout", "conv_dropout", "static_dropout", "fusion_dropout", "dropout", "tst_attn_dropout"):
            val = float(getattr(self, drop_name))
            if not 0.0 <= val < 1.0:
                raise InvalidParams(f"{drop_name} must be in [0, 1)")
        if self.branch_se is not None and int(self.branch_se) < 1:
            raise InvalidParams("branch_se must be None or >= 1")
        if not 0.0 < float(self.xcm_window_perc) <= 1.0:
            raise InvalidParams("xcm_window_perc must be in (0, 1]")
        if float(self.tst_d_ff_mult) <= 0.0:
            raise InvalidParams("tst_d_ff_mult must be > 0")
        if self.tst_pe not in {None, "zeros", "normal", "sincos"}:
            raise InvalidParams("tst_pe must be None, 'zeros', 'normal', or 'sincos'")
        if not 0.0 <= float(self.label_smoothing) < 1.0:
            raise InvalidParams("label_smoothing must be in [0, 1)")
        if not 0.0 <= float(self.validation_split) < 0.9:
            raise InvalidParams("validation_split must be in [0, 0.9)")
        self.validate_pretraining_params()
        return self

    def _build_net(
        self, branch_specs: Mapping[str, Tuple[int, int]], static_dim: int, n_classes: int,
    ) -> MultiBranchTSAINet:
        return MultiBranchTSAINet(
            branch_specs, static_dim, n_classes,
            arch=self.arch, imu_arch=self.imu_arch, thm_arch=self.thm_arch, tof_arch=self.tof_arch,
            branch_filters=self.branch_filters, branch_depth=self.branch_depth,
            branch_kernel_size=self.branch_kernel_size, branch_dropout=self.branch_dropout,
            conv_dropout=self.conv_dropout, branch_units=self.branch_units,
            pooling_mode=self.pooling_mode, attention_units=self.attention_units,
            tof_spatial_mode=self.tof_spatial_mode, tof_spatial_filters=self.tof_spatial_filters,
            tof_spatial_kernel=self.tof_spatial_kernel, static_units=self.static_units,
            static_layers=self.static_layers, static_dropout=self.static_dropout,
            fusion_type=self.fusion_type, fusion_units=self.fusion_units,
            fusion_dropout=self.fusion_dropout, attention_heads=self.attention_heads,
            dense_units=self.dense_units, dense_layers=self.dense_layers,
            dropout=self.dropout, activation=self.activation, use_batchnorm=self.use_batchnorm,
            inception_bottleneck=self.inception_bottleneck, inception_residual=self.inception_residual,
            conv_coord=self.conv_coord, branch_sa=self.branch_sa, branch_se=self.branch_se,
            fcn_separable=self.fcn_separable, fcn_residual=self.fcn_residual,
            xcm_window_perc=self.xcm_window_perc,
            tst_d_ff_mult=self.tst_d_ff_mult, tst_attn_dropout=self.tst_attn_dropout,
            tst_res_attention=self.tst_res_attention, tst_pre_norm=self.tst_pre_norm,
            tst_pe=self.tst_pe, rnn_bidirectional=self.rnn_bidirectional,
        ).to(self.device_)

    def _resolve_device(self) -> torch.device:
        if self.device in {None, "auto"}:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    # -- sequence label extraction & internal validation split ----------------
    def _extract_sequence_labels(self, X: pd.DataFrame, y: Any) -> Tuple[pd.Series, Optional[pd.Series]]:
        seq_col = self.extractor.sequence_col if self.extractor is not None else SEQ_COL
        seq_order = pd.Index(X[seq_col].astype(str).unique(), name=seq_col)
        group_series: Optional[pd.Series] = None
        g_col = self.validation_group_column

        if isinstance(y, pd.DataFrame):
            target_col = self.primary_target if self.primary_target in y.columns else [
                c for c in y.columns if c not in {seq_col, "subject", "is_target"}
            ][0]
            if seq_col in y.columns:
                grouped = y.groupby(y[seq_col].astype(str), sort=False)[target_col].first()
                labels = grouped.reindex(seq_order)
                if g_col and g_col in y.columns:
                    group_series = y.groupby(y[seq_col].astype(str), sort=False)[g_col].first().reindex(seq_order)
            elif len(y) == len(X):
                tmp = pd.DataFrame({seq_col: X[seq_col].astype(str).to_numpy(), target_col: y[target_col].to_numpy()})
                labels = tmp.groupby(seq_col, sort=False)[target_col].first().reindex(seq_order)
            else:
                labels = pd.Series(y[target_col].to_numpy(), index=seq_order[:len(y)])
        elif isinstance(y, pd.Series):
            if len(y) == len(X):
                tmp = pd.DataFrame({seq_col: X[seq_col].astype(str).to_numpy(), "y": y.to_numpy()})
                labels = tmp.groupby(seq_col, sort=False)["y"].first().reindex(seq_order)
            elif y.index.isin(seq_order).all():
                labels = y.reindex(seq_order)
            else:
                labels = pd.Series(y.to_numpy(), index=seq_order[:len(y)])
        else:
            arr = np.asarray(y)
            if len(arr) == len(X):
                tmp = pd.DataFrame({seq_col: X[seq_col].astype(str).to_numpy(), "y": arr})
                labels = tmp.groupby(seq_col, sort=False)["y"].first().reindex(seq_order)
            else:
                labels = pd.Series(arr, index=seq_order[:len(arr)])

        if group_series is None and g_col and g_col in X.columns:
            group_series = X.groupby(X[seq_col].astype(str), sort=False)[g_col].first().reindex(seq_order)
        if labels.isna().any():
            raise ValueError("Missing sequence labels for one or more sequences in X")
        return labels.astype(str), group_series

    def _split_validation_sequences(
        self, X: pd.DataFrame, y: Any
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        seq_col = self.extractor_.sequence_col
        labels, groups = self._extract_sequence_labels(X, y)
        seq_ids = labels.index
        val_frac = float(self.validation_split)
        if val_frac <= 0 or len(seq_ids) < 2:
            self.train_sequence_ids_ = seq_ids
            self.preprocessing_sequence_ids_ = seq_ids
            self.validation_sequence_ids_ = pd.Index([], name=seq_col)
            return X, X.iloc[0:0].copy(), labels, labels.iloc[0:0]

        if groups is not None and groups.nunique() >= 2:
            splitter = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=int(self.random_state))
            fit_idx, val_idx = next(splitter.split(seq_ids, labels, groups=groups))
        else:
            rng = np.random.default_rng(int(self.random_state))
            perm = rng.permutation(len(seq_ids))
            n_val = max(1, min(len(seq_ids) - 1, int(round(len(seq_ids) * val_frac))))
            val_idx = np.sort(perm[:n_val])
            fit_idx = np.sort(perm[n_val:])

        fit_seqs = seq_ids[fit_idx]
        val_seqs = seq_ids[val_idx]
        self.train_sequence_ids_ = fit_seqs
        self.preprocessing_sequence_ids_ = fit_seqs
        self.validation_sequence_ids_ = val_seqs

        seq_str = X[seq_col].astype(str)
        X_fit = X.loc[seq_str.isin(fit_seqs)].copy()
        X_val = X.loc[seq_str.isin(val_seqs)].copy()
        return X_fit, X_val, labels.loc[fit_seqs], labels.loc[val_seqs]

    # -- fit / predict / score ------------------------------------------------
    def _record_failure(self, exc: Exception):
        self.fit_failed_ = True
        self.fit_error_ = f"{type(exc).__name__}: {exc}"
        if self.candidate_failure_log:
            try:
                path = Path(self.candidate_failure_log)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"error": self.fit_error_, "time": time.time()}) + "\n")
            except Exception:
                pass
        warnings.warn(f"Candidate skipped: {self.fit_error_}", RuntimeWarning)

    def fit(self, X: pd.DataFrame, y: Any):
        self.fit_failed_ = False
        self.fit_error_ = None
        try:
            return self._fit_impl(X, y)
        except Exception as exc:
            if self.invalid_candidate_policy == "raise":
                raise
            self._record_failure(exc)
            return self

    def _fit_impl(self, X: pd.DataFrame, y: Any):
        self.validate()
        seed = int(self.random_state or 42)
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.extractor_, self.augmentor_ = self._build_subcomponents()
        X_fit, X_val, y_fit_seq, y_val_seq = self._split_validation_sequences(X, y)

        self.le_ = LabelEncoder()
        self.le_.fit(pd.concat([y_fit_seq, y_val_seq], axis=0))
        self.classes_ = self.le_.classes_
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise InvalidParams("Training partition must contain at least 2 distinct target classes")

        raw_fit = self.extractor_.raw_chunks(X_fit)
        feat_fit = self.extractor_.featurize(raw_fit)
        self.extractor_.fit_scalers(feat_fit)
        train_chunks = self.extractor_.scale(feat_fit)

        # Optional supervised top-k selection of static features (training windows only)
        self.static_feature_scores_ = None
        if self.static_top_k is not None:
            win_labels = self.le_.transform(
                pd.Series(train_chunks["sequence_ids"].astype(str)).map(y_fit_seq).to_numpy()
            )
            keep, self.static_feature_scores_ = self.extractor_.select_static_top_k(
                train_chunks, win_labels, int(self.static_top_k),
            )
            if len(keep) != train_chunks["static"].shape[1]:
                feat_fit = dict(feat_fit)
                feat_fit["static"] = feat_fit["static"][:, keep]
                feat_fit["static_names"] = list(self.extractor_.static_names_)
                train_chunks = self.extractor_.scale(feat_fit)
            if self.verbose and int(self.verbose) > 0:
                print(f"  [feature selection] kept {len(keep)} static features (top_k={self.static_top_k})", flush=True)
        self.noop_feature_names_ = list(self.extractor_.noop_feature_names_)
        self.static_feature_names_ = list(self.extractor_.static_names_)
        self.feature_manifest_ = getattr(self.extractor_, "feature_manifest_", None)

        # Augment training chunks if enabled and recompute static features on augmented signals
        if self.augmentor_.enabled():
            replicas = [train_chunks]
            for _ in range(int(self.augmentor_.n_copies)):
                aug_raw = self.augmentor_.augment_chunks(raw_fit, rng)
                aug_feat = self.extractor_.featurize(aug_raw)
                replicas.append(self.extractor_.scale(aug_feat))
            train_chunks = _concat_chunk_dicts(replicas)

        val_chunks = self.extractor_.transform_chunks(X_val) if len(X_val) else None

        y_train_win = self.le_.transform(
            pd.Series(train_chunks["sequence_ids"].astype(str)).map(y_fit_seq).to_numpy()
        )
        y_val_win = (
            self.le_.transform(pd.Series(val_chunks["sequence_ids"].astype(str)).map(y_val_seq).to_numpy())
            if val_chunks is not None and len(val_chunks["sequence_ids"]) else None
        )

        branch_specs = {
            name: (arr.shape[1], arr.shape[2])
            for name, arr in train_chunks["branches"].items()
        }
        static_dim = train_chunks["static"].shape[1]
        self.branch_specs_ = branch_specs
        self.static_dim_ = static_dim
        self.device_ = self._resolve_device()

        self.model_ = self._build_net(branch_specs, static_dim, n_classes)

        class_weights_t: Optional[torch.Tensor] = None
        if self.class_weight_mode == "balanced":
            counts = np.bincount(y_train_win, minlength=n_classes).astype(np.float64)
            weights = np.where(counts > 0, len(y_train_win) / (n_classes * np.maximum(counts, 1.0)), 1.0)
            class_weights_t = torch.tensor(weights, dtype=torch.float32, device=self.device_)

        if str(self.loss_name).lower() == "focal":
            criterion: nn.Module = FocalLoss(
                gamma=self.focal_gamma, weight=class_weights_t,
                label_smoothing=self.label_smoothing,
            )
        else:
            criterion = nn.CrossEntropyLoss(
                weight=class_weights_t, label_smoothing=float(self.label_smoothing),
            )

        self._run_pretraining_and_warmup_stages(
            X_fit, y, train_chunks, y_train_win,
            branch_specs, static_dim, rng, criterion,
            val_chunks=val_chunks, y_val_win=y_val_win,
        )

        eff_lr = float(self.finetuning_learning_rate) if float(self.finetuning_learning_rate) > 0.0 else float(self.learning_rate)
        opt_name = str(self.optimizer_name).lower()
        if opt_name == "adamw":
            optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                self.model_.parameters(), lr=eff_lr, weight_decay=float(self.weight_decay),
            )
        elif opt_name == "adam":
            optimizer = torch.optim.Adam(
                self.model_.parameters(), lr=eff_lr, weight_decay=float(self.weight_decay),
            )
        else:
            optimizer = torch.optim.SGD(
                self.model_.parameters(), lr=eff_lr,
                momentum=float(self.momentum), weight_decay=float(self.weight_decay),
            )

        n_samples = len(y_train_win)
        batch_size = max(1, int(self.batch_size))
        steps_per_epoch = max(1, int(math.ceil(n_samples / batch_size)))
        total_epochs = max(1, int(self.epochs))
        sched_type = str(self.lr_scheduler_type).lower() if self.use_lr_scheduler else "none"
        scheduler: Any = None
        if sched_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, int(self.cosine_t_max or total_epochs)), eta_min=float(self.min_lr),
            )
        elif sched_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=float(self.lr_factor),
                patience=max(1, int(self.lr_patience)), min_lr=float(self.min_lr),
            )
        elif sched_type == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=eff_lr,
                epochs=total_epochs, steps_per_epoch=steps_per_epoch,
            )

        self.history_ = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
        best_val_loss = float("inf")
        best_state: Optional[Dict[str, torch.Tensor]] = None
        self.best_epoch_ = 1
        patience_left = int(self.early_stopping_patience)
        present_col_idx = {
            name: train_chunks["static_names"].index(f"present_{name}")
            for name in MODALITIES
            if f"present_{name}" in train_chunks["static_names"]
        }

        for epoch in range(1, total_epochs + 1):
            self.model_.train()
            if int(self.warmup_epochs) > 0 and epoch <= int(self.warmup_epochs) and sched_type != "onecycle":
                warm_lr = eff_lr * (epoch / float(self.warmup_epochs))
                for pg in optimizer.param_groups:
                    pg["lr"] = warm_lr

            order = rng.permutation(n_samples)
            epoch_loss = 0.0
            correct = 0
            total = 0

            for start in range(0, n_samples, batch_size):
                idx = order[start:start + batch_size]
                # Batch-Norm requires >= 2 samples in training mode
                if len(idx) == 1 and n_samples > 1:
                    idx = np.r_[idx, order[0]]
                b_branches, b_masks, b_present, b_static = self._slice_batch(
                    train_chunks, idx, rng=rng, apply_sensor_drop=True, present_col_idx=present_col_idx,
                )
                b_y = torch.as_tensor(y_train_win[idx], dtype=torch.long, device=self.device_)

                optimizer.zero_grad(set_to_none=True)
                logits = self.model_(b_branches, b_masks, b_present, b_static)
                loss = criterion(logits, b_y)
                loss.backward()
                if self.gradient_clip_norm is not None and float(self.gradient_clip_norm) > 0:
                    nn.utils.clip_grad_norm_(self.model_.parameters(), float(self.gradient_clip_norm))
                optimizer.step()
                if sched_type == "onecycle" and scheduler is not None:
                    scheduler.step()

                bs = len(idx)
                epoch_loss += float(loss.detach().cpu().item()) * bs
                preds = logits.detach().argmax(dim=-1)
                correct += int((preds == b_y).sum().cpu().item())
                total += bs

            train_loss = epoch_loss / max(1, total)
            train_acc = correct / max(1, total)

            if val_chunks is not None and y_val_win is not None and len(y_val_win):
                val_loss, val_acc = self._evaluate_chunks(val_chunks, y_val_win, criterion)
            else:
                val_loss, val_acc = train_loss, train_acc

            self.history_["loss"].append(float(train_loss))
            self.history_["val_loss"].append(float(val_loss))
            self.history_["accuracy"].append(float(train_acc))
            self.history_["val_accuracy"].append(float(val_acc))

            if self.verbose and int(self.verbose) > 0:
                print(
                    f"  [epoch {epoch}/{total_epochs}] loss={train_loss:.4f} acc={train_acc:.4f} | "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}",
                    flush=True,
                )

            if sched_type == "cosine" and scheduler is not None and epoch > int(self.warmup_epochs):
                scheduler.step()
            elif sched_type == "plateau" and scheduler is not None:
                scheduler.step(val_loss)

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                self.best_epoch_ = epoch
                if self.restore_best_weights:
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
                patience_left = int(self.early_stopping_patience)
            elif int(self.early_stopping_patience) > 0:
                patience_left -= 1
                if patience_left <= 0:
                    break

        self.epochs_run_ = len(self.history_["loss"])
        if self.restore_best_weights and best_state is not None:
            self.model_.load_state_dict({k: v.to(self.device_) for k, v in best_state.items()})
        self.model_.eval()
        return self

    def _slice_batch(
        self, chunks: Mapping[str, Any], idx: np.ndarray, *,
        rng: Optional[np.random.Generator] = None, apply_sensor_drop: bool = False,
        present_col_idx: Optional[Mapping[str, int]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        present = chunks["present"][idx].copy()
        static = chunks["static"][idx].copy()
        branches_np = {k: v[idx] for k, v in chunks["branches"].items()}

        if apply_sensor_drop and rng is not None and (
            self.augmentor_.sensor_drop_prob > 0 or self.augmentor_.sensor_drop_joint_prob > 0
        ):
            keep = self.augmentor_.sample_modality_keep(len(idx), rng)
            present &= keep
            for j, name in ((1, "thm"), (2, "tof")):
                dropped = ~keep[:, j]
                if dropped.any():
                    if name in branches_np:
                        branches_np[name] = branches_np[name].copy()
                        branches_np[name][dropped] = 0.0
                    if present_col_idx and name in present_col_idx:
                        static[dropped, present_col_idx[name]] = 0.0

        b_branches = {
            k: torch.as_tensor(v, dtype=torch.float32, device=self.device_)
            for k, v in branches_np.items()
        }
        b_masks = {
            k: torch.as_tensor(v[idx], dtype=torch.bool, device=self.device_)
            for k, v in chunks["masks"].items()
        }
        b_present = torch.as_tensor(present, dtype=torch.bool, device=self.device_)
        b_static = torch.as_tensor(static, dtype=torch.float32, device=self.device_)
        return b_branches, b_masks, b_present, b_static

    def _evaluate_chunks(self, chunks: Mapping[str, Any], y_win: np.ndarray, criterion: nn.Module) -> Tuple[float, float]:
        self.model_.eval()
        n = len(y_win)
        bs = max(1, int(self.batch_size))
        total_loss = 0.0
        correct = 0
        with torch.no_grad():
            for start in range(0, n, bs):
                idx = np.arange(start, min(n, start + bs))
                b_branches, b_masks, b_present, b_static = self._slice_batch(chunks, idx)
                b_y = torch.as_tensor(y_win[idx], dtype=torch.long, device=self.device_)
                logits = self.model_(b_branches, b_masks, b_present, b_static)
                loss = criterion(logits, b_y)
                total_loss += float(loss.cpu().item()) * len(idx)
                correct += int((logits.argmax(dim=-1) == b_y).sum().cpu().item())
        return total_loss / max(1, n), correct / max(1, n)

    def get_history_dict(self) -> Dict[str, List[float]]:
        return dict(getattr(self, "history_", {}))

    def predict_window_proba(self, chunks: Mapping[str, Any]) -> np.ndarray:
        """Predict class probabilities ``(N_windows, n_classes)`` for pre-extracted chunks."""
        if getattr(self, "fit_failed_", False) or not hasattr(self, "model_"):
            raise RuntimeError(f"Model was not fitted successfully: {getattr(self, 'fit_error_', 'unfitted')}")
        self.model_.eval()
        n = len(chunks["sequence_ids"])
        if n == 0:
            return np.zeros((0, len(self.classes_)), dtype=np.float32)
        bs = max(1, int(self.batch_size))
        out = []
        with torch.no_grad():
            for start in range(0, n, bs):
                idx = np.arange(start, min(n, start + bs))
                b_branches, b_masks, b_present, b_static = self._slice_batch(chunks, idx)
                logits = self.model_(b_branches, b_masks, b_present, b_static)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                out.append(probs)
        return np.vstack(out).astype(np.float32)

    def aggregate_window_proba(
        self, window_proba: np.ndarray, sequence_ids: np.ndarray,
        seq_order: Optional[Sequence[str]] = None,
    ) -> Tuple[pd.Index, np.ndarray]:
        """Aggregate window probabilities into one probability vector per sequence."""
        seq_str = pd.Series(sequence_ids.astype(str))
        if seq_order is None:
            unique_seqs = pd.Index(seq_str.unique(), name=self.extractor_.sequence_col)
        else:
            unique_seqs = pd.Index([str(s) for s in seq_order], name=self.extractor_.sequence_col)
        n_classes = window_proba.shape[1]
        agg = np.zeros((len(unique_seqs), n_classes), dtype=np.float32)
        groups = seq_str.groupby(seq_str, sort=False).indices
        for i, seq_id in enumerate(unique_seqs):
            pos = groups.get(str(seq_id))
            if pos is None or len(pos) == 0:
                agg[i] = 1.0 / max(1, n_classes)
                continue
            sub = window_proba[pos]
            if self.sequence_aggregation == "last_window":
                agg[i] = sub[-1]
            elif self.sequence_aggregation == "majority_vote":
                votes = np.bincount(sub.argmax(axis=1), minlength=n_classes).astype(np.float32)
                agg[i] = (votes + 1e-3 * sub.mean(axis=0)) / (votes.sum() + 1e-3)
            else:
                agg[i] = sub.mean(axis=0)
        return unique_seqs, agg

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        seq_col = self.extractor_.sequence_col
        seq_order = X[seq_col].astype(str).unique()
        chunks = self.extractor_.transform_chunks(X)
        win_proba = self.predict_window_proba(chunks)
        _, seq_proba = self.aggregate_window_proba(win_proba, chunks["sequence_ids"], seq_order=seq_order)
        return seq_proba

    def predict(self, X: pd.DataFrame) -> pd.Series:
        seq_col = self.extractor_.sequence_col
        seq_order = pd.Index(X[seq_col].astype(str).unique(), name=seq_col)
        proba = self.predict_proba(X)
        labels = self.le_.inverse_transform(proba.argmax(axis=1))
        return pd.Series(labels, index=seq_order, name=self.primary_target)

    def score(self, X: pd.DataFrame, y: Any, sample_weight=None) -> float:
        if getattr(self, "fit_failed_", False):
            return float(self.invalid_candidate_score)
        try:
            preds = self.predict(X)
            labels, _ = self._extract_sequence_labels(X, y)
            labels = labels.reindex(preds.index)
            if self.primary_target == "bfrb":
                return float(competition_score(labels, preds)["competition_score"])
            return float(f1_score(labels, preds, average="macro", zero_division=0))
        except Exception as exc:
            if self.invalid_candidate_policy == "raise":
                raise
            warnings.warn(f"Candidate scoring failed: {exc}", RuntimeWarning)
            return float(self.invalid_candidate_score)

    # -- ensemble bundle save / load (weights.pt + config.json + oof/holdout CSV + manifest.json) --
    @staticmethod
    def _serialize_proba_frame(
        proba: np.ndarray,
        classes: Sequence[str],
        sequence_ids: Sequence[str],
        y_true: Optional[Sequence[Any]] = None,
    ) -> pd.DataFrame:
        data: Dict[str, Any] = {SEQ_COL: [str(s) for s in sequence_ids]}
        if y_true is not None:
            data["y_true"] = [str(v) for v in y_true]
        for j, cls_name in enumerate(classes):
            data[f"proba::{cls_name}"] = np.asarray(proba[:, j], dtype=np.float64)
        return pd.DataFrame(data)

    @staticmethod
    def _extract_proba_matrix(df: pd.DataFrame) -> Tuple[pd.Index, List[str], np.ndarray, Optional[pd.Series]]:
        proba_cols = [c for c in df.columns if c.startswith("proba::")]
        classes = [c[len("proba::"):] for c in proba_cols]
        seq_ids = pd.Index(df[SEQ_COL].astype(str), name=SEQ_COL)
        mat = df[proba_cols].to_numpy(dtype=np.float32) if proba_cols else np.zeros((len(df), 0), dtype=np.float32)
        y_true = pd.Series(df["y_true"].astype(str).to_numpy(), index=seq_ids) if "y_true" in df.columns else None
        return seq_ids, classes, mat, y_true

    def save_bundle(
        self,
        bundle_dir: Any,
        *,
        oof_proba: Optional[np.ndarray] = None,
        oof_sequence_ids: Optional[Sequence[str]] = None,
        oof_y: Optional[Sequence[Any]] = None,
        holdout_proba: Optional[np.ndarray] = None,
        holdout_sequence_ids: Optional[Sequence[str]] = None,
        holdout_y: Optional[Sequence[Any]] = None,
        cv_score: Optional[float] = None,
        holdout_score: Optional[float] = None,
        extra_meta: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        """Persist weights, config, scaler state, OOF/holdout probabilities, and manifest for ensembling."""
        if getattr(self, "fit_failed_", False) or not hasattr(self, "model_"):
            raise RuntimeError("Cannot save bundle from an unfitted or failed estimator.")
        b_dir = Path(bundle_dir)
        b_dir.mkdir(parents=True, exist_ok=True)

        classes = [str(c) for c in self.classes_]
        branch_specs = {str(k): [int(v[0]), int(v[1])] for k, v in self.branch_specs_.items()}

        weights_payload = {
            "state_dict": {k: v.detach().cpu() for k, v in self.model_.state_dict().items()},
            "branch_specs": {str(k): (int(v[0]), int(v[1])) for k, v in self.branch_specs_.items()},
            "static_dim": int(self.static_dim_),
            "classes": classes,
        }
        weights_path = b_dir / "weights.pt"
        torch.save(weights_payload, weights_path)

        raw_params = self.get_params(deep=True)
        leaf_params = {
            k: v for k, v in raw_params.items()
            if k not in {"extractor", "augmentor", "extractor__noop"}
        }
        ext = self.extractor_
        extractor_state = {
            "scalers": {k: [np.asarray(m).tolist(), np.asarray(s).tolist()] for k, (m, s) in ext.scalers_.items()},
            "static_scaler": [np.asarray(ext.static_scaler_[0]).tolist(), np.asarray(ext.static_scaler_[1]).tolist()],
            "static_names": list(ext.static_names_),
            "noop_feature_names": list(ext.noop_feature_names_),
            "channel_names": {k: list(v) for k, v in getattr(ext, "channel_names_", {}).items()},
            "rates": {k: float(v) for k, v in getattr(ext, "rates_", {}).items()},
        }
        config_payload = {
            "bundle_format": "multibranch_bundle_v1",
            "estimator_params": leaf_params,
            "branch_specs": branch_specs,
            "static_dim": int(self.static_dim_),
            "classes": classes,
            "primary_target": str(self.primary_target),
            "modalities": str(ext.modalities),
            "extractor_state": extractor_state,
        }
        config_path = b_dir / "config.json"
        config_path.write_text(json.dumps(config_payload, indent=2, default=str), encoding="utf-8")

        files_written: Dict[str, str] = {
            "weights": weights_path.name,
            "config": config_path.name,
        }
        if oof_proba is not None and oof_sequence_ids is not None:
            oof_df = self._serialize_proba_frame(oof_proba, classes, oof_sequence_ids, oof_y)
            oof_path = b_dir / "oof_proba.csv"
            oof_df.to_csv(oof_path, index=False)
            files_written["oof_proba"] = oof_path.name
        if holdout_proba is not None and holdout_sequence_ids is not None:
            ho_df = self._serialize_proba_frame(holdout_proba, classes, holdout_sequence_ids, holdout_y)
            ho_path = b_dir / "holdout_proba.csv"
            ho_df.to_csv(ho_path, index=False)
            files_written["holdout_proba"] = ho_path.name

        manifest_payload: Dict[str, Any] = {
            "bundle_format": "multibranch_bundle_v1",
            "bundle_dir": str(b_dir),
            "arch": str(self.arch),
            "imu_arch": self.imu_arch,
            "thm_arch": self.thm_arch,
            "tof_arch": self.tof_arch,
            "fusion_type": str(self.fusion_type),
            "modalities": str(ext.modalities),
            "classes": classes,
            "primary_target": str(self.primary_target),
            "cv_score": float(cv_score) if cv_score is not None and np.isfinite(cv_score) else None,
            "holdout_score": float(holdout_score) if holdout_score is not None and np.isfinite(holdout_score) else None,
            "pretraining_mode": str(self.pretraining_mode),
            "pretraining_epochs": int(self.pretraining_epochs),
            "files": files_written,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if extra_meta:
            manifest_payload["extra"] = dict(extra_meta)
        manifest_path = b_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest_payload, indent=2, default=str), encoding="utf-8")
        return b_dir

    @classmethod
    def load_bundle(cls, bundle_dir: Any, device: str = "auto") -> "MultiBranchTSAIClassifier":
        """Restore a trained ``MultiBranchTSAIClassifier`` from ``weights.pt`` + ``config.json``."""
        b_dir = Path(bundle_dir)
        cfg = json.loads((b_dir / "config.json").read_text(encoding="utf-8"))
        try:
            payload = torch.load(b_dir / "weights.pt", map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(b_dir / "weights.pt", map_location="cpu")

        est = cls(
            extractor=MultiModalSequenceExtractor(noop=NoopOptions()),
            augmentor=TSAugmentor(),
        )
        valid_keys = set(est.get_params(deep=True))
        restored_params: Dict[str, Any] = {}
        tuple_keys = {"lag_diff_scales", "window_seconds_list", "tof_invalid_sentinels",
                      "tsfresh_fft_bands", "crop_frac_range"}
        for k, v in cfg.get("estimator_params", {}).items():
            if k not in valid_keys:
                continue
            leaf = k.split("__")[-1]
            if leaf in tuple_keys and isinstance(v, list):
                v = tuple(tuple(x) if isinstance(x, list) else x for x in v)
            restored_params[k] = v
        restored_params["device"] = device
        est.set_params(**restored_params)
        est.device_ = est._resolve_device()
        est.extractor_, est.augmentor_ = est._build_subcomponents()

        ext_state = cfg["extractor_state"]
        est.extractor_.scalers_ = {
            k: (np.asarray(pair[0], dtype=np.float32), np.asarray(pair[1], dtype=np.float32))
            for k, pair in ext_state["scalers"].items()
        }
        ss0, ss1 = ext_state["static_scaler"]
        est.extractor_.static_scaler_ = (np.asarray(ss0, dtype=np.float32), np.asarray(ss1, dtype=np.float32))
        est.extractor_.static_names_ = list(ext_state["static_names"])
        est.extractor_.noop_feature_names_ = list(ext_state["noop_feature_names"])
        est.extractor_.channel_names_ = {k: list(v) for k, v in ext_state.get("channel_names", {}).items()}
        est.extractor_.rates_ = {k: float(v) for k, v in ext_state.get("rates", {}).items()}

        classes = [str(c) for c in cfg["classes"]]
        est.le_ = LabelEncoder()
        est.le_.classes_ = np.asarray(classes, dtype=object)
        est.classes_ = est.le_.classes_
        est.noop_feature_names_ = list(est.extractor_.noop_feature_names_)
        est.static_feature_names_ = list(est.extractor_.static_names_)

        branch_specs = {str(k): (int(v[0]), int(v[1])) for k, v in cfg["branch_specs"].items()}
        static_dim = int(cfg["static_dim"])
        est.branch_specs_ = branch_specs
        est.static_dim_ = static_dim
        est.model_ = est._build_net(branch_specs, static_dim, len(classes))
        est.model_.load_state_dict({k: v.to(est.device_) for k, v in payload["state_dict"].items()}, strict=True)
        est.model_.eval()
        est.fit_failed_ = False
        est.fit_error_ = None
        return est


def _concat_chunk_dicts(dicts: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if len(dicts) == 1:
        return dict(dicts[0])
    first = dicts[0]
    out = dict(first)
    out["branches"] = {
        k: np.concatenate([d["branches"][k] for d in dicts], axis=0)
        for k in first["branches"]
    }
    out["masks"] = {
        k: np.concatenate([d["masks"][k] for d in dicts], axis=0)
        for k in first["masks"]
    }
    for key in ("present", "static", "sequence_ids", "window_index", "obs_axes", "q_obs", "row_present",
                "window_starts_s", "sequence_durations_s", "thm_seq_baseline"):
        if key in first:
            out[key] = np.concatenate([d[key] for d in dicts], axis=0)
    for key in ("obs_thm", "obs_tof"):
        if key in first:
            out[key] = np.concatenate([d[key] for d in dicts], axis=0)
    return out


# Alias for backward compatibility
AdvancedMultiBranchSequenceClassifier = MultiBranchTSAIClassifier


# ---------------------------------------------------------------------------
# 6b. Multi-Model & Hierarchical Ensemble (2..5+ models, saved bundles or E2E)
# ---------------------------------------------------------------------------

ENSEMBLE_MODES = ("soft_vote", "stacking", "hierarchical")
ENSEMBLE_WEIGHT_MODES = ("equal", "oof_opt", "fixed")

DEFAULT_ENSEMBLE_MODEL_SPECS: Tuple[Dict[str, Any], ...] = (
    {"name": "multimodal_inception", "arch": "InceptionTimePlus", "extractor__modalities": "all", "fusion_type": "attention"},
    {"name": "imu_specialist_conv", "arch": "Conv1d", "extractor__modalities": "imu", "fusion_type": "concat"},
    {"name": "multimodal_resnet", "arch": "ResNetPlus", "extractor__modalities": "all", "fusion_type": "gated"},
    {"name": "imu_specialist_resnet", "arch": "ResNetPlus", "extractor__modalities": "imu", "fusion_type": "concat"},
    {"name": "multimodal_fcn", "arch": "FCNPlus", "extractor__modalities": "all", "fusion_type": "attention"},
)


class MultiBranchEnsemble(BaseSequenceClassifier):
    """Scalable 2-to-5+ model ensemble supporting end-to-end training or saved bundles.

    Supports three blending regimes selectable via ``ensemble_mode``:

    * ``'soft_vote'``: Normalized weighted average over member probability distributions.
    * ``'stacking'``: Out-of-fold ``LogisticRegression`` meta-learner over member class
      probabilities and sequence modality availability (THM/ToF presence).
    * ``'hierarchical'``: Competition-aware two-stage routing — Stage 1 blends ``P(bfrb)``
      across members with optional IMU-specialist boost when THM/ToF are absent and tunes
      ``binary_threshold`` on OOF predictions; Stage 2 predicts the highest-scoring BFRB
      gesture when Stage 1 fires above threshold, otherwise ``'non_bfrb'``.
    """

    def __init__(
        self,
        bundle_dirs: Optional[Any] = None,
        estimators: Optional[Any] = None,
        model_specs: Optional[Any] = None,
        base_estimator_params: Optional[Mapping[str, Any]] = None,
        n_models: int = 3,
        use_saved_bundles: bool = False,
        ensemble_mode: str = "hierarchical",
        model_weights_mode: str = "oof_opt",
        model_weights: Optional[Any] = None,
        binary_threshold: Any = "auto",
        imu_specialist_boost: float = 0.25,
        meta_C: float = 1.0,
        cv_splits: int = 3,
        base_epochs: Optional[int] = None,
        base_batch_size: Optional[int] = None,
        sequence_col: str = SEQ_COL,
        primary_target: str = "bfrb",
        validation_group_column: Optional[str] = "subject",
        invalid_candidate_policy: str = "penalize",
        invalid_candidate_score: float = -1.0,
        verbose: int = 0,
        random_state: int = 42,
        device: str = "auto",
    ):
        self.bundle_dirs = bundle_dirs
        self.estimators = estimators
        self.model_specs = model_specs
        self.base_estimator_params = base_estimator_params
        self.n_models = n_models
        self.use_saved_bundles = use_saved_bundles
        self.ensemble_mode = ensemble_mode
        self.model_weights_mode = model_weights_mode
        self.model_weights = model_weights
        self.binary_threshold = binary_threshold
        self.imu_specialist_boost = imu_specialist_boost
        self.meta_C = meta_C
        self.cv_splits = cv_splits
        self.base_epochs = base_epochs
        self.base_batch_size = base_batch_size
        self.sequence_col = sequence_col
        self.primary_target = primary_target
        self.validation_group_column = validation_group_column
        self.invalid_candidate_policy = invalid_candidate_policy
        self.invalid_candidate_score = invalid_candidate_score
        self.verbose = verbose
        self.random_state = random_state
        self.device = device

    def set_params(self, **params):
        cleaned = {}
        for k, v in params.items():
            if k in {"bundle_dirs", "model_specs", "model_weights", "base_estimator_params"} and isinstance(v, str):
                if v.startswith(("[", "{", "(")):
                    v = _structured(v)
                elif "|" in v:
                    parts = [p.strip() for p in v.split("|") if p.strip()]
                    v = [float(p) for p in parts] if k == "model_weights" else parts
            cleaned[k] = v
        return super().set_params(**cleaned)

    def validate(self):
        if str(self.ensemble_mode) not in ENSEMBLE_MODES:
            raise InvalidParams(f"ensemble_mode must be one of {ENSEMBLE_MODES}, got {self.ensemble_mode!r}")
        if str(self.model_weights_mode) not in ENSEMBLE_WEIGHT_MODES:
            raise InvalidParams(f"model_weights_mode must be one of {ENSEMBLE_WEIGHT_MODES}")
        if int(self.n_models) < 2:
            raise InvalidParams("n_models must be >= 2 for ensembling")
        if int(self.cv_splits) < 2 and not bool(self.use_saved_bundles):
            raise InvalidParams("cv_splits must be >= 2 for end-to-end OOF blending")
        if float(self.imu_specialist_boost) < 0.0:
            raise InvalidParams("imu_specialist_boost must be >= 0.0")
        if float(self.meta_C) <= 0.0:
            raise InvalidParams("meta_C must be > 0.0")
        if isinstance(self.binary_threshold, str):
            if self.binary_threshold != "auto":
                try:
                    t = float(self.binary_threshold)
                except ValueError as exc:
                    raise InvalidParams("binary_threshold must be 'auto' or in (0, 1)") from exc
                if not 0.0 < t < 1.0:
                    raise InvalidParams("binary_threshold must be in (0, 1)")
        elif self.binary_threshold is not None:
            t = float(self.binary_threshold)
            if not 0.0 < t < 1.0:
                raise InvalidParams("binary_threshold must be in (0, 1)")
        return self

    def _sequence_modality_flags(self, X: pd.DataFrame, seq_order: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        thm_cols = [c for c in THM_COLUMNS if c in X.columns]
        tof_cols = [c for c in TOF_COLUMNS if c in X.columns]
        groups = X.groupby(X[self.sequence_col].astype(str), sort=False).indices
        has_thm = np.zeros(len(seq_order), dtype=bool)
        has_tof = np.zeros(len(seq_order), dtype=bool)
        thm_row_ok = X[thm_cols].notna().any(axis=1).to_numpy() if thm_cols else np.zeros(len(X), dtype=bool)
        tof_row_ok = X[tof_cols].notna().any(axis=1).to_numpy() if tof_cols else np.zeros(len(X), dtype=bool)
        for i, sid in enumerate(seq_order):
            pos = groups.get(str(sid))
            if pos is not None and len(pos):
                has_thm[i] = bool(thm_row_ok[pos].any())
                has_tof[i] = bool(tof_row_ok[pos].any())
        return has_thm, has_tof

    @staticmethod
    def _align_proba(proba: np.ndarray, src_classes: Sequence[str], target_classes: Sequence[str]) -> np.ndarray:
        out = np.zeros((proba.shape[0], len(target_classes)), dtype=np.float64)
        idx_map = {str(c): j for j, c in enumerate(target_classes)}
        for j_src, cls_name in enumerate(src_classes):
            j_dst = idx_map.get(str(cls_name))
            if j_dst is not None:
                out[:, j_dst] = proba[:, j_src]
        row_sum = out.sum(axis=1, keepdims=True)
        out = np.where(row_sum > 0, out / np.maximum(row_sum, 1e-12), 1.0 / max(1, len(target_classes)))
        return out

    @staticmethod
    def _build_default_member(
        spec: Mapping[str, Any],
        base_params: Optional[Mapping[str, Any]],
        *,
        base_epochs: Optional[int],
        base_batch_size: Optional[int],
        primary_target: str,
        random_state: int,
        device: str,
        verbose: int,
    ) -> MultiBranchTSAIClassifier:
        est = MultiBranchTSAIClassifier(
            extractor=MultiModalSequenceExtractor(noop=NoopOptions()),
            augmentor=TSAugmentor(),
        )
        merged: Dict[str, Any] = {}
        if base_params:
            merged.update({k: v for k, v in base_params.items() if k not in {"name"}})
        for k, v in spec.items():
            if k == "name":
                continue
            merged[k] = v
        merged["primary_target"] = primary_target
        merged["random_state"] = int(random_state)
        merged["device"] = device
        merged.setdefault("validation_split", 0.0)
        if verbose <= 0:
            merged.setdefault("verbose", 0)
        if base_epochs is not None:
            merged["epochs"] = int(base_epochs)
        if base_batch_size is not None:
            merged["batch_size"] = int(base_batch_size)
        valid_keys = set(est.get_params(deep=True))
        filtered = {k: v for k, v in merged.items() if k in valid_keys}
        est.set_params(**filtered)
        return est

    def _resolve_members(self) -> Tuple[List[Any], List[str], List[bool]]:
        n = int(self.n_models)
        if bool(self.use_saved_bundles):
            if not self.bundle_dirs:
                raise InvalidParams("bundle_dirs must be provided when use_saved_bundles=True")
            dirs = list(_structured(self.bundle_dirs))[:n]
            if len(dirs) < 2:
                raise InvalidParams("At least 2 bundle_dirs are required for ensembling")
            members, names, is_imu = [], [], []
            for i, d in enumerate(dirs):
                m = MultiBranchTSAIClassifier.load_bundle(d, device=self.device)
                members.append(m)
                names.append(f"bundle_{i+1}_{Path(d).name}")
                is_imu.append(getattr(m.extractor_, "modalities", "all") == "imu")
            return members, names, is_imu

        if self.estimators is not None:
            raw_ests = list(self.estimators)
            if len(raw_ests) < n:
                raise InvalidParams(f"Expected at least n_models={n} estimators, got {len(raw_ests)}")
            members, names, is_imu = [], [], []
            for i in range(n):
                est = clone(raw_ests[i])
                step = est.named_steps["estimator"] if hasattr(est, "named_steps") else est
                names.append(f"est_{i+1}_{type(step).__name__}")
                ext = getattr(step, "extractor", None)
                is_imu.append(getattr(ext, "modalities", "all") == "imu")
                members.append(est)
            return members, names, is_imu

        raw_specs = list(_structured(self.model_specs)) if self.model_specs is not None else list(DEFAULT_ENSEMBLE_MODEL_SPECS)
        members, names, is_imu = [], [], []
        for i in range(n):
            spec = dict(raw_specs[i % len(raw_specs)])
            name = spec.get("name") or f"model_{i+1}_{spec.get('arch', 'InceptionTimePlus')}"
            if i >= len(raw_specs):
                name = f"{name}_seed{i+1}"
            m = self._build_default_member(
                spec,
                self.base_estimator_params,
                base_epochs=self.base_epochs,
                base_batch_size=self.base_batch_size,
                primary_target=self.primary_target,
                random_state=int(self.random_state or 42) + i,
                device=self.device,
                verbose=int(self.verbose),
            )
            members.append(m)
            names.append(str(name))
            is_imu.append(getattr(m.extractor, "modalities", "all") == "imu")
        return members, names, is_imu

    def _effective_row_weights(self, has_thm: np.ndarray, has_tof: np.ndarray) -> np.ndarray:
        k = len(self.members_)
        base_w = np.asarray(self.model_weights_, dtype=np.float64)[None, :]
        w = np.repeat(base_w, len(has_thm), axis=0)
        imu_only_seq = ~(has_thm | has_tof)
        boost = float(self.imu_specialist_boost)
        if boost > 0 and imu_only_seq.any():
            for idx_m, flag in enumerate(self.is_imu_specialist_):
                if flag:
                    w[imu_only_seq, idx_m] *= (1.0 + boost)
        w_sum = np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
        return w / w_sum

    def _optimize_weights(
        self, oof_list: List[np.ndarray], y_idx: np.ndarray,
        has_thm: np.ndarray, has_tof: np.ndarray,
    ) -> np.ndarray:
        k = len(oof_list)
        if str(self.model_weights_mode) == "fixed" and self.model_weights is not None:
            raw = np.asarray(list(_structured(self.model_weights))[:k], dtype=np.float64)
            if len(raw) == k and raw.sum() > 0:
                return raw / raw.sum()
        if str(self.model_weights_mode) == "equal":
            return np.full(k, 1.0 / k, dtype=np.float64)

        from scipy.optimize import minimize

        def _nll(w_vec: np.ndarray) -> float:
            w_norm = w_vec / max(w_vec.sum(), 1e-12)
            blend = np.zeros_like(oof_list[0], dtype=np.float64)
            for j, mat in enumerate(oof_list):
                blend += w_norm[j] * mat
            p_true = np.clip(blend[np.arange(len(y_idx)), y_idx], 1e-7, 1.0)
            return float(-np.mean(np.log(p_true)))

        init = np.full(k, 1.0 / k, dtype=np.float64)
        res = minimize(
            _nll,
            init,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * k,
            constraints=({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}),
        )
        w_opt = np.clip(res.x if res.success else init, 0.0, 1.0)
        if w_opt.sum() <= 0:
            return init
        return w_opt / w_opt.sum()

    def _tune_binary_threshold(
        self, proba: np.ndarray, y_true_str: pd.Series, classes: Sequence[str],
    ) -> float:
        if self.binary_threshold not in {None, "auto"}:
            return float(self.binary_threshold)
        if "non_bfrb" not in set(classes):
            return 0.5
        non_idx = list(classes).index("non_bfrb")
        p_bfrb = 1.0 - proba[:, non_idx]
        target_mask = np.ones(len(classes), dtype=bool)
        target_mask[non_idx] = False
        best_thr = 0.5
        best_score = -1.0
        for thr in np.linspace(0.15, 0.85, 29):
            preds = []
            for i in range(len(p_bfrb)):
                if p_bfrb[i] < thr:
                    preds.append("non_bfrb")
                else:
                    sub = np.where(target_mask, proba[i], -1.0)
                    preds.append(str(classes[int(np.argmax(sub))]))
            sc = competition_score(y_true_str, pd.Series(preds, index=y_true_str.index))["competition_score"]
            if sc > best_score:
                best_score = sc
                best_thr = float(thr)
        return best_thr

    def _combine_member_probas(
        self,
        member_probas: List[np.ndarray],
        has_thm: np.ndarray,
        has_tof: np.ndarray,
    ) -> np.ndarray:
        row_w = self._effective_row_weights(has_thm, has_tof)  # (N, K)
        blended = np.zeros_like(member_probas[0], dtype=np.float64)
        for j, mat in enumerate(member_probas):
            blended += row_w[:, j:j + 1] * mat

        mode = str(self.ensemble_mode)
        if mode == "stacking" and getattr(self, "meta_clf_", None) is not None:
            feats = np.hstack(member_probas + [has_thm.astype(np.float64)[:, None], has_tof.astype(np.float64)[:, None]])
            meta_p = self.meta_clf_.predict_proba(feats)
            # Align LogisticRegression classes_ back to self.classes_
            return self._align_proba(meta_p, [str(c) for c in self.meta_clf_.classes_], self.classes_).astype(np.float32)

        if mode == "hierarchical" and "non_bfrb" in set(self.classes_):
            non_idx = list(self.classes_).index("non_bfrb")
            thr = float(self.binary_threshold_)
            p_bfrb = np.clip(1.0 - blended[:, non_idx], 0.0, 1.0)
            target_idx = [i for i in range(len(self.classes_)) if i != non_idx]
            out = blended.copy()
            # Recalibrate probability mass across non_bfrb vs target around tuned binary threshold
            for i in range(len(out)):
                if p_bfrb[i] < thr:
                    # Push probability below 0.5 for targets while keeping conditional ranking intact
                    non_mass = max(0.51, out[i, non_idx])
                    rem = max(1e-6, 1.0 - non_mass)
                    t_vals = out[i, target_idx]
                    t_sum = max(1e-12, float(t_vals.sum()))
                    out[i, non_idx] = non_mass
                    out[i, target_idx] = rem * (t_vals / t_sum)
                else:
                    # Ensure argmax falls on best target gesture when Stage 1 fires
                    t_vals = out[i, target_idx]
                    t_sum = max(1e-12, float(t_vals.sum()))
                    rem = min(out[i, non_idx], 0.49 * (1.0 - p_bfrb[i] + 1e-6))
                    out[i, non_idx] = rem
                    out[i, target_idx] = (1.0 - rem) * (t_vals / t_sum)
            return out.astype(np.float32)

        row_sum = np.maximum(blended.sum(axis=1, keepdims=True), 1e-12)
        return (blended / row_sum).astype(np.float32)

    def fit(self, X: pd.DataFrame, y: Any):
        self.fit_failed_ = False
        self.fit_error_ = None
        try:
            return self._fit_impl(X, y)
        except Exception as exc:
            if self.invalid_candidate_policy == "raise":
                raise
            self.fit_failed_ = True
            self.fit_error_ = f"{type(exc).__name__}: {exc}"
            warnings.warn(f"Ensemble candidate skipped: {self.fit_error_}", RuntimeWarning)
            return self

    def _fit_impl(self, X: pd.DataFrame, y: Any):
        self.validate()
        seq_order = pd.Index(X[self.sequence_col].astype(str).unique(), name=self.sequence_col)
        ref_helper = MultiBranchTSAIClassifier(
            extractor=MultiModalSequenceExtractor(noop=NoopOptions()),
            primary_target=self.primary_target,
            validation_group_column=self.validation_group_column,
        )
        seq_labels, seq_groups = ref_helper._extract_sequence_labels(X, y)
        seq_labels = seq_labels.reindex(seq_order)
        self.le_ = LabelEncoder().fit(seq_labels.to_numpy())
        self.classes_ = np.asarray([str(c) for c in self.le_.classes_], dtype=object)

        members, member_names, is_imu = self._resolve_members()
        self.members_ = members
        self.member_names_ = member_names
        self.is_imu_specialist_ = is_imu
        has_thm, has_tof = self._sequence_modality_flags(X, seq_order)
        oof_probas: List[np.ndarray] = []

        if bool(self.use_saved_bundles):
            dirs = list(_structured(self.bundle_dirs))[:len(members)]
            for b_dir, est in zip(dirs, members):
                oof_file = Path(b_dir) / "oof_proba.csv"
                loaded_oof = None
                if oof_file.exists():
                    s_ids, cls_s, mat, _ = MultiBranchTSAIClassifier._extract_proba_matrix(pd.read_csv(oof_file))
                    if set(seq_order).issubset(set(s_ids)):
                        df_m = pd.DataFrame(mat, index=s_ids)
                        aligned_m = df_m.reindex(seq_order).to_numpy(dtype=np.float64)
                        loaded_oof = self._align_proba(aligned_m, cls_s, self.classes_)
                if loaded_oof is None:
                    raw_p = est.predict_proba(X)
                    loaded_oof = self._align_proba(raw_p, [str(c) for c in est.classes_], self.classes_)
                oof_probas.append(loaded_oof)
        else:
            # Generate true out-of-fold probabilities across members
            n_seq = len(seq_order)
            n_splits = min(int(self.cv_splits), max(2, n_seq))
            if seq_groups is not None and seq_groups.nunique() >= n_splits:
                splitter = GroupKFold(n_splits=n_splits)
                split_iter = list(splitter.split(seq_order, seq_labels, groups=seq_groups.reindex(seq_order)))
            else:
                from sklearn.model_selection import StratifiedKFold
                try:
                    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(self.random_state or 42))
                    split_iter = list(splitter.split(seq_order, seq_labels))
                except ValueError:
                    from sklearn.model_selection import KFold
                    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=int(self.random_state or 42))
                    split_iter = list(splitter.split(seq_order))

            seq_col_series = X[self.sequence_col].astype(str)
            oof_probas = [np.zeros((n_seq, len(self.classes_)), dtype=np.float64) for _ in members]
            for tr_idx, va_idx in split_iter:
                tr_seqs = set(seq_order[tr_idx].astype(str))
                va_seqs = seq_order[va_idx]
                X_tr_f = X.loc[seq_col_series.isin(tr_seqs)]
                X_va_f = X.loc[seq_col_series.isin(set(va_seqs.astype(str)))]
                y_tr_f = y.loc[X_tr_f.index] if isinstance(y, (pd.DataFrame, pd.Series)) and len(y) == len(X) else y
                for k_i, est in enumerate(members):
                    fold_est = clone(est)
                    fold_est.fit(X_tr_f, y_tr_f)
                    if getattr(fold_est, "fit_failed_", False):
                        raise RuntimeError(f"Ensemble member {member_names[k_i]} failed on OOF fold: {fold_est.fit_error_}")
                    p_va = fold_est.predict_proba(X_va_f)
                    # Align X_va_f unique sequence order back to va_idx
                    va_actual = pd.Index(X_va_f[self.sequence_col].astype(str).unique())
                    p_df = pd.DataFrame(
                        self._align_proba(p_va, [str(c) for c in fold_est.classes_], self.classes_),
                        index=va_actual,
                    )
                    oof_probas[k_i][va_idx] = p_df.reindex(va_seqs).to_numpy(dtype=np.float64)

            # Fit final members on full training data
            for k_i, est in enumerate(members):
                est.fit(X, y)
                if getattr(est, "fit_failed_", False):
                    raise RuntimeError(f"Final fit failed for member {member_names[k_i]}: {est.fit_error_}")

        self.oof_sequence_ids_ = seq_order
        self.oof_member_probas_ = oof_probas
        y_idx = self.le_.transform(seq_labels.to_numpy())
        self.model_weights_ = self._optimize_weights(oof_probas, y_idx, has_thm, has_tof)

        self.meta_clf_ = None
        if str(self.ensemble_mode) == "stacking":
            from sklearn.linear_model import LogisticRegression
            feats = np.hstack(oof_probas + [has_thm.astype(np.float64)[:, None], has_tof.astype(np.float64)[:, None]])
            self.meta_clf_ = LogisticRegression(C=float(self.meta_C), max_iter=1000, random_state=int(self.random_state or 42))
            self.meta_clf_.fit(feats, seq_labels.to_numpy())

        self.binary_threshold_ = 0.5
        pre_blend = self._combine_member_probas(oof_probas, has_thm, has_tof)
        if str(self.ensemble_mode) == "hierarchical":
            # Tune threshold on raw weighted blend (prior to step adjustment)
            row_w = self._effective_row_weights(has_thm, has_tof)
            raw_blend = sum(row_w[:, j:j + 1] * mat for j, mat in enumerate(oof_probas))
            self.binary_threshold_ = self._tune_binary_threshold(raw_blend, seq_labels, self.classes_)
            self.oof_proba_ = self._combine_member_probas(oof_probas, has_thm, has_tof)
        else:
            self.oof_proba_ = pre_blend

        self.oof_y_ = seq_labels.to_numpy()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if getattr(self, "fit_failed_", False) or not hasattr(self, "members_"):
            raise RuntimeError(f"Ensemble was not fitted: {getattr(self, 'fit_error_', 'unfitted')}")
        seq_order = pd.Index(X[self.sequence_col].astype(str).unique(), name=self.sequence_col)
        has_thm, has_tof = self._sequence_modality_flags(X, seq_order)
        member_probas = [
            self._align_proba(est.predict_proba(X), [str(c) for c in est.classes_], self.classes_)
            for est in self.members_
        ]
        return self._combine_member_probas(member_probas, has_thm, has_tof)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        seq_order = pd.Index(X[self.sequence_col].astype(str).unique(), name=self.sequence_col)
        proba = self.predict_proba(X)
        labels = np.asarray(self.classes_)[proba.argmax(axis=1)]
        return pd.Series(labels, index=seq_order, name=self.primary_target)

    def score(self, X: pd.DataFrame, y: Any, sample_weight=None) -> float:
        if getattr(self, "fit_failed_", False):
            return float(self.invalid_candidate_score)
        try:
            preds = self.predict(X)
            ref_helper = MultiBranchTSAIClassifier(
                extractor=MultiModalSequenceExtractor(noop=NoopOptions()),
                primary_target=self.primary_target,
                validation_group_column=self.validation_group_column,
            )
            labels, _ = ref_helper._extract_sequence_labels(X, y)
            labels = labels.reindex(preds.index)
            if self.primary_target == "bfrb":
                return float(competition_score(labels, preds)["competition_score"])
            return float(f1_score(labels, preds, average="macro", zero_division=0))
        except Exception as exc:
            if self.invalid_candidate_policy == "raise":
                raise
            warnings.warn(f"Ensemble scoring failed: {exc}", RuntimeWarning)
            return float(self.invalid_candidate_score)

    def summarize_members(self) -> pd.DataFrame:
        """Return a summary table of member weights, modalities, and OOF competition/macro-F1."""
        rows = []
        y_true = pd.Series(self.oof_y_, index=self.oof_sequence_ids_)
        for k_i, (name, w, is_imu, mat) in enumerate(zip(
            self.member_names_, self.model_weights_, self.is_imu_specialist_, self.oof_member_probas_
        )):
            preds = pd.Series(np.asarray(self.classes_)[mat.argmax(axis=1)], index=self.oof_sequence_ids_)
            sc = competition_score(y_true, preds) if self.primary_target == "bfrb" else {
                "competition_score": float(f1_score(y_true, preds, average="macro", zero_division=0)),
                "binary_f1": np.nan, "gesture_f1": np.nan,
            }
            rows.append({
                "slot": k_i + 1,
                "member": name,
                "imu_specialist": bool(is_imu),
                "weight": float(w),
                "oof_score": float(sc["competition_score"]),
                "oof_binary_f1": float(sc["binary_f1"]),
                "oof_gesture_f1": float(sc["gesture_f1"]),
            })
        ens_preds = pd.Series(np.asarray(self.classes_)[self.oof_proba_.argmax(axis=1)], index=self.oof_sequence_ids_)
        ens_sc = competition_score(y_true, ens_preds) if self.primary_target == "bfrb" else {
            "competition_score": float(f1_score(y_true, ens_preds, average="macro", zero_division=0)),
            "binary_f1": np.nan, "gesture_f1": np.nan,
        }
        rows.append({
            "slot": 0,
            "member": f"ENSEMBLE ({self.ensemble_mode})",
            "imu_specialist": False,
            "weight": 1.0,
            "oof_score": float(ens_sc["competition_score"]),
            "oof_binary_f1": float(ens_sc["binary_f1"]),
            "oof_gesture_f1": float(ens_sc["gesture_f1"]),
        })
        return pd.DataFrame(rows)

    def save_bundle(
        self,
        bundle_dir: Any,
        *,
        holdout_proba: Optional[np.ndarray] = None,
        holdout_sequence_ids: Optional[Sequence[str]] = None,
        holdout_y: Optional[Sequence[Any]] = None,
        cv_score: Optional[float] = None,
        holdout_score: Optional[float] = None,
        **kwargs,
    ) -> Path:
        """Persist ensemble config, member bundles, OOF/holdout probabilities, and manifest."""
        b_dir = Path(bundle_dir)
        b_dir.mkdir(parents=True, exist_ok=True)
        classes = [str(c) for c in self.classes_]
        member_dirs: List[str] = []
        if not bool(self.use_saved_bundles):
            for k_i, est in enumerate(self.members_):
                step = est.named_steps["estimator"] if hasattr(est, "named_steps") else est
                if hasattr(step, "save_bundle"):
                    m_dir = b_dir / f"member_{k_i+1}"
                    step.save_bundle(
                        m_dir,
                        oof_proba=self.oof_member_probas_[k_i],
                        oof_sequence_ids=self.oof_sequence_ids_,
                        oof_y=self.oof_y_,
                    )
                    member_dirs.append(str(m_dir))
        else:
            member_dirs = [str(d) for d in list(_structured(self.bundle_dirs))[:len(self.members_)]]

        MultiBranchTSAIClassifier._serialize_proba_frame(
            self.oof_proba_, classes, self.oof_sequence_ids_, self.oof_y_
        ).to_csv(b_dir / "oof_proba.csv", index=False)
        if holdout_proba is not None and holdout_sequence_ids is not None:
            MultiBranchTSAIClassifier._serialize_proba_frame(
                holdout_proba, classes, holdout_sequence_ids, holdout_y
            ).to_csv(b_dir / "holdout_proba.csv", index=False)

        self.summarize_members().to_csv(b_dir / "member_summary.csv", index=False)
        cfg = {
            "bundle_format": "multibranch_ensemble_v1",
            "ensemble_mode": str(self.ensemble_mode),
            "model_weights_mode": str(self.model_weights_mode),
            "model_weights": [float(w) for w in self.model_weights_],
            "binary_threshold": float(self.binary_threshold_),
            "imu_specialist_boost": float(self.imu_specialist_boost),
            "n_models": int(len(self.members_)),
            "member_names": list(self.member_names_),
            "member_dirs": member_dirs,
            "classes": classes,
            "cv_score": float(cv_score) if cv_score is not None and np.isfinite(cv_score) else None,
            "holdout_score": float(holdout_score) if holdout_score is not None and np.isfinite(holdout_score) else None,
        }
        (b_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")
        (b_dir / "manifest.json").write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")
        return b_dir


Ensemble = MultiBranchEnsemble


# ---------------------------------------------------------------------------
# 7. Competition Scorer (Original + 50% Hidden-Test Nullification Simulation)
# ---------------------------------------------------------------------------

def competition_score(
    y_true_gesture: Any, y_pred: Any,
    y_true_binary: Optional[Any] = None,
    target_only_macro: bool = True,
) -> Dict[str, float]:
    """Compute the exact CMI competition metric: ``(binary_f1 + gesture_f1) / 2.0``.

    * ``binary_f1`` is the binary F1 for BFRB (``!= 'non_bfrb'``) vs everyday (``'non_bfrb'``).
    * ``gesture_f1`` is the macro F1 across the target BFRB gesture classes.
    """
    if isinstance(y_true_gesture, pd.Series) and isinstance(y_pred, pd.Series) and y_true_gesture.index.equals(y_pred.index):
        y_true_s = y_true_gesture.astype(str)
        y_pred_s = y_pred.astype(str)
    elif isinstance(y_true_gesture, pd.Series) and isinstance(y_pred, pd.Series) and set(y_pred.index).issubset(set(y_true_gesture.index)):
        y_pred_s = y_pred.astype(str)
        y_true_s = y_true_gesture.reindex(y_pred.index).astype(str)
    else:
        y_true_s = pd.Series(np.asarray(y_true_gesture)).astype(str)
        y_pred_s = pd.Series(np.asarray(y_pred)).astype(str)

    pred_binary = (y_pred_s != "non_bfrb").astype(int)
    if y_true_binary is not None:
        true_binary = pd.Series(np.asarray(y_true_binary), index=y_true_s.index).astype(int)
    else:
        true_binary = (y_true_s != "non_bfrb").astype(int)

    binary_f1 = float(f1_score(true_binary, pred_binary, average="binary", pos_label=1, zero_division=0))

    if target_only_macro and (true_binary == 1).any():
        mask = true_binary == 1
        target_classes = np.unique(y_true_s[mask])
        gesture_f1 = float(f1_score(
            y_true_s[mask], y_pred_s[mask],
            labels=target_classes, average="macro", zero_division=0,
        ))
    else:
        gesture_f1 = float(f1_score(y_true_s, y_pred_s, average="macro", zero_division=0))

    comp = (binary_f1 + gesture_f1) / 2.0
    return {
        "competition_score": float(comp),
        "binary_f1": float(binary_f1),
        "gesture_f1": float(gesture_f1),
    }


def simulate_hidden_test_nullification(
    X_df: pd.DataFrame,
    null_fraction: float = 0.5,
    random_state: int = 42,
    sequence_col: str = SEQ_COL,
) -> pd.DataFrame:
    """Simulate the Kaggle hidden test set by nullifying Thermopile and ToF for 50% of sequences.

    Deterministically selects ``round(null_fraction * n_sequences)`` sequences in ``X_df``
    and replaces all ``thm_*`` and ``tof_*`` columns for those sequences with ``np.nan``.
    """
    if not 0.0 <= float(null_fraction) <= 1.0:
        raise ValueError("null_fraction must be in [0, 1]")
    out = X_df.copy()
    seq_ids = pd.Index(out[sequence_col].astype(str).unique())
    n_seq = len(seq_ids)
    n_null = int(round(n_seq * float(null_fraction)))
    if 0 < float(null_fraction) < 1.0 and n_seq >= 2:
        n_null = max(1, min(n_seq - 1, n_null))
    rng = np.random.default_rng(int(random_state))
    chosen_idx = np.sort(rng.choice(n_seq, size=n_null, replace=False)) if n_null > 0 else np.empty(0, dtype=int)
    null_seqs = set(seq_ids[chosen_idx])
    sensor_cols = [c for c in out.columns if c.startswith("thm_") or c.startswith("tof_")]
    if null_seqs and sensor_cols:
        mask = out[sequence_col].astype(str).isin(null_seqs)
        out.loc[mask, sensor_cols] = np.nan
    out.attrs["nullified_sequence_ids"] = sorted(null_seqs)
    out.attrs["null_fraction"] = float(null_fraction)
    return out


def _sequence_level_truth(y_df: Any, target_col: str, seq_index: Optional[pd.Index] = None) -> pd.DataFrame:
    if isinstance(y_df, pd.DataFrame):
        if SEQ_COL in y_df.columns:
            seq_y = y_df.groupby(y_df[SEQ_COL].astype(str), sort=False).first()
        else:
            seq_y = y_df.copy()
            seq_y.index = seq_y.index.astype(str)
    elif isinstance(y_df, pd.Series):
        seq_y = pd.DataFrame({target_col: y_df.astype(str)})
        seq_y.index = seq_y.index.astype(str)
    else:
        arr = np.asarray(y_df)
        idx = seq_index if seq_index is not None else pd.Index([str(i) for i in range(len(arr))])
        seq_y = pd.DataFrame({target_col: arr}, index=idx.astype(str))
    if seq_index is not None:
        seq_y = seq_y.reindex(seq_index.astype(str))
    return seq_y


def evaluate_holdout(
    y_test_df: Any,
    y_pred: Any,
    target_col: str = "bfrb",
    verbose: bool = True,
    *,
    model: Optional[Any] = None,
    X_test: Optional[pd.DataFrame] = None,
    simulate_hidden_test: bool = True,
    null_fraction: float = 0.5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Evaluate holdout predictions with both the normal competition score and the 50%-nullified hidden-test simulation."""
    pred_index = y_pred.index if isinstance(y_pred, pd.Series) else None
    seq_y = _sequence_level_truth(y_test_df, target_col, seq_index=pred_index)
    y_true = seq_y[target_col].astype(str)
    y_pred_s = pd.Series(np.asarray(y_pred), index=y_true.index, name=target_col).astype(str)
    y_true_bin = seq_y["is_target"].astype(int) if "is_target" in seq_y.columns else None

    macro_f1 = float(f1_score(y_true, y_pred_s, average="macro", zero_division=0))
    results: Dict[str, Any] = {"macro_f1": macro_f1}

    if target_col == "bfrb":
        comp = competition_score(y_true, y_pred_s, y_true_binary=y_true_bin)
        results.update(comp)
    else:
        results.update({
            "competition_score": macro_f1,
            "binary_f1": macro_f1,
            "gesture_f1": macro_f1,
        })

    if simulate_hidden_test and model is not None and X_test is not None and len(X_test):
        X_null = simulate_hidden_test_nullification(
            X_test, null_fraction=null_fraction, random_state=random_state,
        )
        y_pred_null = model.predict(X_null)
        seq_y_null = _sequence_level_truth(
            y_test_df, target_col,
            seq_index=y_pred_null.index if isinstance(y_pred_null, pd.Series) else pred_index,
        )
        y_true_null = seq_y_null[target_col].astype(str)
        y_pred_null_s = pd.Series(np.asarray(y_pred_null), index=y_true_null.index, name=target_col).astype(str)
        y_bin_null = seq_y_null["is_target"].astype(int) if "is_target" in seq_y_null.columns else None
        hidden_macro = float(f1_score(y_true_null, y_pred_null_s, average="macro", zero_division=0))
        if target_col == "bfrb":
            hidden_comp = competition_score(y_true_null, y_pred_null_s, y_true_binary=y_bin_null)
        else:
            hidden_comp = {
                "competition_score": hidden_macro,
                "binary_f1": hidden_macro,
                "gesture_f1": hidden_macro,
            }
        results.update({
            "hidden_test_competition_score": float(hidden_comp["competition_score"]),
            "hidden_test_binary_f1": float(hidden_comp["binary_f1"]),
            "hidden_test_gesture_f1": float(hidden_comp["gesture_f1"]),
            "hidden_test_macro_f1": float(hidden_macro),
            "nullified_sequence_ids": list(X_null.attrs.get("nullified_sequence_ids", [])),
            "nullified_sequence_count": len(X_null.attrs.get("nullified_sequence_ids", [])),
            "total_holdout_sequences": int(X_test[SEQ_COL].nunique()) if SEQ_COL in X_test else len(y_true_null),
            "hidden_test_null_fraction": float(null_fraction),
        })

    if verbose:
        print(f"Holdout Normal Competition Score: {results['competition_score']:.4f} "
              f"(Binary F1: {results['binary_f1']:.4f}, Gesture Macro F1: {results['gesture_f1']:.4f}, "
              f"Overall Macro F1: {results['macro_f1']:.4f})")
        if "hidden_test_competition_score" in results:
            print(f"Holdout Hidden-Test Competition Score (50% ToF/Thermopile Nullified, "
                  f"{results['nullified_sequence_count']}/{results['total_holdout_sequences']} seqs): "
                  f"{results['hidden_test_competition_score']:.4f} "
                  f"(Binary F1: {results['hidden_test_binary_f1']:.4f}, "
                  f"Gesture Macro F1: {results['hidden_test_gesture_f1']:.4f})")
        print(classification_report(y_true, y_pred_s, zero_division=0))

    return results


def make_competition_scorer(
    target_col: str = "bfrb",
    simulate_hidden_test: bool = False,
    null_fraction: float = 0.5,
    random_state: int = 42,
):
    """Build an sklearn scorer callable ``(estimator, X, y) -> float``."""
    def _scorer(estimator, X, y, sample_weight=None) -> float:
        est = estimator.named_steps["estimator"] if hasattr(estimator, "named_steps") and "estimator" in estimator.named_steps else estimator
        if getattr(est, "fit_failed_", False):
            return float(getattr(est, "invalid_candidate_score", -1.0))
        try:
            X_eval = (
                simulate_hidden_test_nullification(X, null_fraction=null_fraction, random_state=random_state)
                if simulate_hidden_test else X
            )
            preds = estimator.predict(X_eval)
            seq_y = _sequence_level_truth(y, target_col, seq_index=preds.index if isinstance(preds, pd.Series) else None)
            y_true = seq_y[target_col].astype(str)
            y_bin = seq_y["is_target"].astype(int) if "is_target" in seq_y.columns else None
            if target_col == "bfrb":
                return float(competition_score(y_true, preds, y_true_binary=y_bin)["competition_score"])
            return float(f1_score(y_true, preds, average="macro", zero_division=0))
        except Exception as exc:
            if getattr(est, "invalid_candidate_policy", "penalize") == "raise":
                raise
            warnings.warn(f"Scorer failed: {exc}", RuntimeWarning)
            return float(getattr(est, "invalid_candidate_score", -1.0))
    return _scorer


def candidate_score(estimator, X, y, sample_weight=None) -> float:
    """Default CV scorer delegating to the fitted estimator's competition score."""
    est = estimator.named_steps["estimator"] if hasattr(estimator, "named_steps") and "estimator" in estimator.named_steps else estimator
    return float(est.score(X, y, sample_weight=sample_weight))


# ---------------------------------------------------------------------------
# 8. Search Space (ONE Unified Declarative Block -> Grid / Bayes / Evo / Best)
# ---------------------------------------------------------------------------
# ECat, EInt, and EFloat are imported from ``search_drivers`` (shared with base_utils).


def _encode_skopt_item(val: Any) -> Any:
    """Encode non-scalar objects (tuples, lists, dicts) as JSON strings for skopt Categorical."""
    if isinstance(val, (tuple, list, dict)):
        return json.dumps(val)
    return val


def _decode_skopt_params(params: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, np.generic):
            v = v.item()
        out[k] = _structured(v)
    return out


class ParamSpec:
    """Single-source hyperparameter declaration for Best, Grid, Bayesian, and Evolutionary modes.

    Parameters
    ----------
    default : Any
        Value placed in ``BEST_PARAMS``.
    grid : Sequence[Any]
        Candidate list placed in ``GRID_PARAM_SPACE``.
    kind : {'cat', 'int', 'float', 'log_float'}
        Dimension kind for Bayesian (``skopt``) and Evolutionary (``ECat``/``EInt``/``EFloat``) spaces.
    bounds : Optional[Tuple[float, float]]
        Explicit ``(low, high)`` range for ``int``, ``float``, or ``log_float`` dimensions.
        Defaults to ``(min(grid), max(grid))`` when omitted.
    choices : Optional[Sequence[Any]]
        Explicit categorical choices for Bayesian/Evolutionary spaces when different from ``grid``.
    """

    def __init__(
        self,
        default: Any,
        grid: Sequence[Any],
        kind: str = "cat",
        bounds: Optional[Tuple[Any, Any]] = None,
        choices: Optional[Sequence[Any]] = None,
    ):
        if kind not in {"cat", "int", "float", "log_float"}:
            raise ValueError(f"ParamSpec kind must be 'cat', 'int', 'float', or 'log_float'; got {kind!r}")
        self.default = default
        self.grid = list(grid)
        if not self.grid:
            raise ValueError("ParamSpec grid must contain at least one value")
        self.kind = kind
        self.bounds = bounds
        self.choices = list(choices) if choices is not None else list(self.grid)

    def _numeric_bounds(self) -> Tuple[float, float]:
        if self.bounds is not None:
            return float(self.bounds[0]), float(self.bounds[1])
        nums = [float(x) for x in self.grid if x is not None]
        return min(nums), max(nums)

    def to_grid(self) -> List[Any]:
        return list(self.grid)

    def to_evo(self):
        if self.kind == "cat":
            return ECat(self.choices)
        lo, hi = self._numeric_bounds()
        if lo == hi:
            return ECat([int(lo) if self.kind == "int" else float(lo)])
        if self.kind == "int":
            return EInt(int(round(lo)), int(round(hi)))
        dist = "log-uniform" if self.kind == "log_float" else "uniform"
        return EFloat(float(lo), float(hi), distribution=dist)

    def to_skopt(self, skopt_available: bool = SKOPT_AVAILABLE):
        if not skopt_available or SkCategorical is None:
            return None
        if self.kind == "cat":
            encoded = [_encode_skopt_item(c) for c in self.choices]
            return SkCategorical(encoded)
        lo, hi = self._numeric_bounds()
        if lo == hi:
            return SkCategorical([int(round(lo)) if self.kind == "int" else float(lo)])
        if self.kind == "int":
            return SkInteger(int(round(lo)), int(round(hi)))
        prior = "log-uniform" if self.kind == "log_float" else "uniform"
        return SkReal(float(lo), float(hi), prior=prior)


PS = Spec = ParamSpec


def compile_unified_search_space(
    unified_space: Mapping[str, Any],
    *,
    skopt_available: bool = SKOPT_AVAILABLE,
) -> Tuple[Dict[str, Any], Dict[str, List[Any]], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Compile ONE consolidated ``UNIFIED_PARAM_SPACE`` dictionary into all search views.

    Returns ``(BEST_PARAMS, GRID_PARAM_SPACE, BAYESIAN_PARAM_SPACE, EVOLUTIONARY_PARAM_SPACE, EVOLUTIONARY_SEARCH_SPACE)``.
    """
    best_params: Dict[str, Any] = {}
    grid_space: Dict[str, List[Any]] = {}
    bayes_space: Dict[str, Any] = {}
    evo_space: Dict[str, Any] = {}

    for key, raw_spec in unified_space.items():
        if isinstance(raw_spec, ParamSpec):
            spec = raw_spec
        elif isinstance(raw_spec, Mapping):
            spec = ParamSpec(
                default=raw_spec["default"],
                grid=raw_spec.get("grid", [raw_spec["default"]]),
                kind=raw_spec.get("kind", "cat"),
                bounds=raw_spec.get("bounds"),
                choices=raw_spec.get("choices"),
            )
        elif isinstance(raw_spec, (list, tuple)):
            spec = ParamSpec(default=raw_spec[0], grid=list(raw_spec), kind="cat")
        else:
            spec = ParamSpec(default=raw_spec, grid=[raw_spec], kind="cat")

        best_params[key] = spec.default
        grid_space[key] = spec.to_grid()
        evo_space[key] = spec.to_evo()
        if skopt_available and SkCategorical is not None:
            b_dim = spec.to_skopt(skopt_available=True)
            if b_dim is not None:
                bayes_space[key] = b_dim

    return best_params, grid_space, bayes_space, evo_space, dict(evo_space)


def prepare_bayesian_space(space: Mapping[str, Any], skopt_available: bool = SKOPT_AVAILABLE) -> Dict[str, Any]:
    if not skopt_available or SkCategorical is None:
        return {}
    out: Dict[str, Any] = {}
    for k, v in space.items():
        if isinstance(v, ParamSpec):
            out[k] = v.to_skopt(skopt_available=True)
        elif isinstance(v, (list, tuple)):
            out[k] = SkCategorical([_encode_skopt_item(x) for x in v])
        else:
            out[k] = v
    return out


def prepare_evolutionary_space(space: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in space.items():
        if isinstance(v, ParamSpec):
            out[k] = v.to_evo()
        elif isinstance(v, (ECat, EInt, EFloat)):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = ECat(v)
        else:
            out[k] = ECat([v])
    return out


def inactive_parameters_for_candidate(params: Mapping[str, Any]) -> set:
    """Return the set of full parameter keys that have no effect under ``params``."""
    p = {
        (k[len("estimator__"):] if k.startswith("estimator__") else k): _structured(v)
        for k, v in params.items()
    }
    prefix = "estimator__" if any(k.startswith("estimator__") for k in params) else ""
    inactive: set = set()

    # 1. NOOP inactive tuning parameters
    noop_kwargs = {
        k[len("extractor__noop__"):]: v
        for k, v in p.items()
        if k.startswith("extractor__noop__")
    }
    try:
        noop_opts = NoopOptions(**noop_kwargs)
        for name in inactive_noop_parameters(noop_opts):
            inactive.add(f"{prefix}extractor__{name}")
    except Exception:
        pass

    # 2. Static backend (tsfel / tsfresh / catch22 / tsflex / seglearn / sktime / none)
    backend = p.get("extractor__static_backend", "tsfel")
    if backend in {"none", None}:
        for name in ("static_channels", "tsfel_domains", "tsfel_features", "tsfresh_profile", "tsfresh_fft_bands"):
            inactive.add(f"{prefix}extractor__{name}")
    elif backend == "tsfel":
        for name in ("tsfresh_profile", "tsfresh_fft_bands"):
            inactive.add(f"{prefix}extractor__{name}")
    elif backend == "tsfresh":
        for name in ("tsfel_domains", "tsfel_features"):
            inactive.add(f"{prefix}extractor__{name}")
    else:
        for name in ("tsfel_domains", "tsfel_features", "tsfresh_profile", "tsfresh_fft_bands"):
            inactive.add(f"{prefix}extractor__{name}")

    # 3. Modality disables & preprocessing toggles
    if not p.get("extractor__canonicalize_handedness", False):
        inactive.add(f"{prefix}extractor__handedness_col")
    mods_str = p.get("extractor__modalities", "all")
    try:
        active_mods = _pipe_modes(mods_str, MODALITIES, name="modalities") or set(MODALITIES)
    except Exception:
        active_mods = set(MODALITIES)
    if p.get("extractor__thm_modes", "centered_diff") is None or "thm" not in active_mods:
        inactive.add(f"{prefix}extractor__thm_target_sampling_rate")
        inactive.add(f"{prefix}extractor__thermal_features")
        inactive.add(f"{prefix}thm_arch")
    tof_modes = p.get("extractor__tof_modes", "pooled_stats|sensor_stats")
    if tof_modes is None or "tof" not in active_mods:
        for name in ("extractor__tof_target_sampling_rate", "extractor__tof_fill_mode",
                     "extractor__tof_spike_threshold", "extractor__tof_invalid_sentinels",
                     "extractor__tof_spatial_features", "extractor__tof_temporal_features",
                     "tof_arch", "tof_spatial_mode", "tof_spatial_filters", "tof_spatial_kernel"):
            inactive.add(f"{prefix}{name}")
    else:
        try:
            parsed_tof = _pipe_modes(tof_modes, TOF_MODE_SET, name="tof_modes")
            if "raw" not in parsed_tof or p.get("tof_spatial_mode", "auto") == "flat":
                inactive.add(f"{prefix}tof_spatial_filters")
                inactive.add(f"{prefix}tof_spatial_kernel")
            if "raw" not in parsed_tof:
                inactive.add(f"{prefix}extractor__tof_spatial_features")
        except Exception:
            pass

    # 4. Augmentation inactive knobs
    aug_prob = float(p.get("augmentor__prob", 0.0) or 0.0)
    per_prob = float(p.get("augmentor__per_aug_prob", 0.5) or 0.0)
    n_copies = int(p.get("augmentor__n_copies", 1) or 0)
    if aug_prob <= 0 or per_prob <= 0 or n_copies <= 0:
        for name in (
            "per_aug_prob", "augmentations", "n_copies",
            "jitter_sigma", "noise_std", "scaling_sigma",
            "warp_sigma", "warp_num_knots",
            "time_warp_max_speed_ratio", "time_warp_n_speed_change",
            "time_shift_frac", "crop_frac_range",
            "temporal_num_masks", "temporal_mask_frac", "temporal_mask_fill",
            "channel_drop_prob",
        ):
            if name != "prob":
                inactive.add(f"{prefix}augmentor__{name}")
    else:
        try:
            aug = TSAugmentor(**{
                k[len("augmentor__"):]: v for k, v in p.items() if k.startswith("augmentor__")
            })
            allowed = aug.allowed_ops()
            op_params = {
                "jitter": ("jitter_sigma",),
                "noise": ("noise_std",),
                "scaling": ("scaling_sigma",),
                "magnitude_warp": ("warp_sigma", "warp_num_knots"),
                "time_warp": ("time_warp_max_speed_ratio", "time_warp_n_speed_change"),
                "time_shift": ("time_shift_frac",),
                "crop": ("crop_frac_range",),
                "temporal_mask": ("temporal_num_masks", "temporal_mask_frac", "temporal_mask_fill"),
                "channel_drop": ("channel_drop_prob",),
            }
            for op, fields in op_params.items():
                if op not in allowed:
                    for f in fields:
                        inactive.add(f"{prefix}augmentor__{f}")
            if float(p.get("augmentor__warp_sigma", 0.0) or 0.0) <= 0:
                inactive.add(f"{prefix}augmentor__warp_num_knots")
            if float(p.get("augmentor__time_warp_max_speed_ratio", 1.0) or 1.0) <= 1.0:
                inactive.add(f"{prefix}augmentor__time_warp_n_speed_change")
            if int(p.get("augmentor__temporal_num_masks", 0) or 0) <= 0 or float(p.get("augmentor__temporal_mask_frac", 0.0) or 0.0) <= 0:
                inactive.add(f"{prefix}augmentor__temporal_mask_fill")
        except Exception:
            pass

    # 5. Network / optimizer / scheduler inactive knobs
    if p.get("pooling_mode", "default") != "attention":
        inactive.add(f"{prefix}attention_units")
    if str(p.get("optimizer_name", "adamw")).lower() != "sgd":
        inactive.add(f"{prefix}momentum")
    if str(p.get("loss_name", "cross_entropy")).lower() != "focal":
        inactive.add(f"{prefix}focal_gamma")
    use_sched = bool(p.get("use_lr_scheduler", True))
    sched_type = str(p.get("lr_scheduler_type", "cosine")).lower() if use_sched else "none"
    if not use_sched or sched_type == "none":
        for name in ("lr_scheduler_type", "lr_factor", "lr_patience", "min_lr", "cosine_t_max", "warmup_epochs"):
            if not use_sched or name != "lr_scheduler_type":
                inactive.add(f"{prefix}{name}")
    elif sched_type == "cosine":
        inactive.add(f"{prefix}lr_factor")
        inactive.add(f"{prefix}lr_patience")
    elif sched_type == "plateau":
        inactive.add(f"{prefix}cosine_t_max")
    elif sched_type == "onecycle":
        for name in ("lr_factor", "lr_patience", "min_lr", "cosine_t_max", "warmup_epochs"):
            inactive.add(f"{prefix}{name}")
    if int(p.get("early_stopping_patience", 10) or 0) <= 0:
        inactive.add(f"{prefix}restore_best_weights")

    # 6. Architecture-specific tsai backbone knobs
    base_arch = p.get("arch", "InceptionTimePlus")
    active_archs = {p.get("imu_arch") or base_arch}
    if p.get("extractor__thm_modes", "centered_diff") is not None:
        active_archs.add(p.get("thm_arch") or base_arch)
    if p.get("extractor__tof_modes", "pooled_stats|sensor_stats") is not None:
        active_archs.add(p.get("tof_arch") or base_arch)

    if not (active_archs & {"Conv1d", "InceptionTimePlus", "TCN", "TSTPlus", "GRUPlus", "LSTMPlus"}):
        inactive.add(f"{prefix}branch_depth")
        inactive.add(f"{prefix}conv_dropout")
    if not (active_archs & {"Conv1d", "InceptionTimePlus", "FCNPlus", "TCN"}):
        inactive.add(f"{prefix}branch_kernel_size")
    if "InceptionTimePlus" not in active_archs:
        inactive.add(f"{prefix}inception_bottleneck")
        inactive.add(f"{prefix}inception_residual")
    if not (active_archs & {"InceptionTimePlus", "ResNetPlus", "FCNPlus"}):
        inactive.add(f"{prefix}conv_coord")
    if not (active_archs & {"InceptionTimePlus", "ResNetPlus"}):
        inactive.add(f"{prefix}branch_sa")
        inactive.add(f"{prefix}branch_se")
    if "FCNPlus" not in active_archs:
        inactive.add(f"{prefix}fcn_separable")
        inactive.add(f"{prefix}fcn_residual")
    if "XCMPlus" not in active_archs:
        inactive.add(f"{prefix}xcm_window_perc")
    if "TSTPlus" not in active_archs:
        for name in ("tst_d_ff_mult", "tst_attn_dropout", "tst_res_attention", "tst_pre_norm", "tst_pe"):
            inactive.add(f"{prefix}{name}")
    if not (active_archs & {"GRUPlus", "LSTMPlus"}):
        inactive.add(f"{prefix}rnn_bidirectional")

    # 7. Pretraining & masked-signal SSL knobs (all disabled when pretraining_epochs <= 0)
    pre_epochs = int(p.get("pretraining_epochs", 0) or 0)
    pre_mode = str(p.get("pretraining_mode", "none") or "none").lower()
    if pre_epochs <= 0 or pre_mode == "none":
        for name in (
            "pretraining_source", "pretraining_label_column",
            "pretraining_learning_rate", "pretraining_batch_size",
            "pretraining_validation_split", "pretraining_early_stopping_patience",
            "ssl_mask_ratio", "ssl_span_length", "ssl_decoder_units",
            "ssl_decoder_dropout", "ssl_loss_name",
            "ssl_resample_mask_each_epoch", "ssl_mask_modalities_together",
        ):
            inactive.add(f"{prefix}{name}")
    elif pre_mode == "supervised":
        for name in (
            "ssl_mask_ratio", "ssl_span_length", "ssl_decoder_units",
            "ssl_decoder_dropout", "ssl_loss_name",
            "ssl_resample_mask_each_epoch", "ssl_mask_modalities_together",
        ):
            inactive.add(f"{prefix}{name}")
    elif pre_mode == "masked_ssl":
        inactive.add(f"{prefix}pretraining_label_column")
    if int(p.get("finetuning_head_epochs", 0) or 0) <= 0:
        inactive.add(f"{prefix}finetuning_head_learning_rate")

    return inactive


def candidate_error(params: Mapping[str, Any], base_params: Optional[Mapping[str, Any]] = None) -> Optional[str]:
    """Return an error string if ``params`` is invalid, or ``None`` if valid."""
    merged = dict(base_params or {})
    merged.update(_decode_skopt_params(params))
    try:
        is_ensemble = ("ensemble_mode" in merged or "estimator__ensemble_mode" in merged)
        clf = (
            MultiBranchEnsemble()
            if is_ensemble else
            MultiBranchTSAIClassifier(
                extractor=MultiModalSequenceExtractor(noop=NoopOptions()),
                augmentor=TSAugmentor(),
            )
        )
        valid_keys = set(clf.get_params(deep=True))
        filtered = {
            (k[len("estimator__"):] if k.startswith("estimator__") else k): v
            for k, v in merged.items()
            if (k[len("estimator__"):] if k.startswith("estimator__") else k) in valid_keys
        }
        clf.set_params(**filtered)
        clf.validate()
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def _canonical_candidate_key(params: Mapping[str, Any], base_params: Optional[Mapping[str, Any]] = None) -> str:
    merged = dict(base_params or {})
    merged.update(_decode_skopt_params(params))
    inactive = inactive_parameters_for_candidate(merged)
    active_items = {}
    for k in sorted(params.keys()):
        if k in inactive:
            active_items[k] = "__INACTIVE__"
        else:
            val = _structured(params[k])
            active_items[k] = val
    return json.dumps(active_items, sort_keys=True, default=str)


def deduplicate_param_grid(
    param_grid: Any,
    base_params: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, List[Any]]]:
    """Expand, validate, and deduplicate a parameter grid by pruning inactive parameter duplicates."""
    raw_candidates = list(ParameterGrid(param_grid))
    seen: set = set()
    unique: List[Dict[str, List[Any]]] = []
    for cand in raw_candidates:
        if candidate_error(cand, base_params=base_params) is not None:
            continue
        sig = _canonical_candidate_key(cand, base_params=base_params)
        if sig in seen:
            continue
        seen.add(sig)
        unique.append({k: [v] for k, v in cand.items()})
    if not unique and raw_candidates:
        # Keep first candidate if all were filtered so GridSearchCV still runs and logs failure
        first = raw_candidates[0]
        unique.append({k: [v] for k, v in first.items()})
    return unique


# ---------------------------------------------------------------------------
# 9. Search Drivers, Validation Runner, Permutation Importance, Synthetic Data
# ---------------------------------------------------------------------------

def _finalize_search_results(
    search: Any,
    all_params: List[Dict[str, Any]],
    all_fold_scores: List[List[float]],
    all_times: List[float],
    X: pd.DataFrame,
    y: Any,
):
    """Populate ``cv_results_``, select the best candidate index/score/params, and refit if requested."""
    mean_scores = np.array([float(np.mean(s)) for s in all_fold_scores], dtype=float)
    std_scores = np.array([float(np.std(s)) for s in all_fold_scores], dtype=float)
    ranks = (len(mean_scores) - np.argsort(np.argsort(mean_scores))).astype(int)
    n_splits = len(all_fold_scores[0]) if all_fold_scores else 1
    search.cv_results_ = {
        "params": all_params,
        "mean_test_score": mean_scores,
        "std_test_score": std_scores,
        "rank_test_score": ranks,
        "mean_fit_time": np.array(all_times, dtype=float),
    }
    for split_i in range(n_splits):
        search.cv_results_[f"split{split_i}_test_score"] = np.array(
            [s[split_i] for s in all_fold_scores], dtype=float,
        )

    if callable(search.refit):
        best_idx = int(search.refit(search.cv_results_))
    else:
        finite = np.flatnonzero(np.isfinite(mean_scores))
        best_idx = int(finite[int(np.argmax(mean_scores[finite]))]) if len(finite) else 0
    search.best_index_ = int(best_idx)
    search.best_params_ = dict(all_params[search.best_index_])
    search.best_score_ = float(mean_scores[search.best_index_])

    if search.refit:
        search.best_estimator_ = clone(search.estimator).set_params(**search.best_params_)
        search.best_estimator_.fit(X, y)
    return search


def _evaluate_candidate_cv(
    estimator: Any,
    params: Mapping[str, Any],
    X: pd.DataFrame,
    y: Any,
    groups: Optional[Any],
    cv: Any,
    scoring: Any,
    error_score: float = -1.0,
) -> Tuple[List[float], float]:
    decoded = _decode_skopt_params(params)
    err = candidate_error(decoded, base_params=estimator.get_params(deep=True))
    if err is not None:
        warnings.warn(f"Candidate skipped before CV ({err})", RuntimeWarning)
        splits = list(cv.split(X, y, groups))
        return [float(error_score)] * len(splits), 0.0

    fold_scores: List[float] = []
    t0 = time.time()
    scorer = scoring if callable(scoring) else candidate_score
    for train_idx, val_idx in cv.split(X, y, groups):
        X_tr = X.iloc[train_idx]
        X_va = X.iloc[val_idx]
        y_tr = y.iloc[train_idx] if isinstance(y, (pd.DataFrame, pd.Series)) and len(y) == len(X) else y
        y_va = y.iloc[val_idx] if isinstance(y, (pd.DataFrame, pd.Series)) and len(y) == len(X) else y
        model = clone(estimator).set_params(**decoded)
        try:
            model.fit(X_tr, y_tr)
            sc = float(scorer(model, X_va, y_va))
        except Exception as exc:
            warnings.warn(f"Fold evaluation failed: {exc}", RuntimeWarning)
            sc = float(error_score)
        fold_scores.append(sc)
    return fold_scores, time.time() - t0


class MultiBranchGASearchCV(BaseGASearchCV):
    """Genetic / Evolutionary hyperparameter search with hierarchical inactive-parameter caching."""

    def __init__(
        self,
        estimator: Any,
        param_space: Optional[Mapping[str, Any]] = None,
        cv: Any = None,
        scoring: Any = None,
        population_size: int = 10,
        generations: int = 2,
        tournament_size: int = 3,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.2,
        elitism: bool = True,
        refit: Any = True,
        random_state: int = 42,
        verbose: int = 1,
        error_score: float = -1.0,
        n_jobs: int = 1,
        *,
        param_grid: Optional[Mapping[str, Any]] = None,
        crossover_probability: Optional[float] = None,
        mutation_probability: Optional[float] = None,
        return_train_score: bool = True,
    ):
        self.estimator = estimator
        self.param_space = param_space if param_space is not None else param_grid
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_probability if crossover_probability is not None else crossover_prob
        self.mutation_prob = mutation_probability if mutation_probability is not None else mutation_prob
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.elitism = elitism
        self.refit = refit
        self.random_state = random_state
        self.verbose = verbose
        self.error_score = error_score
        self.n_jobs = n_jobs
        self.return_train_score = return_train_score

    def _sample_individual(self, space: Mapping[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
        base = self.estimator.get_params(deep=True)
        for _ in range(25):
            cand = {k: v.sample(rng) for k, v in space.items()}
            if candidate_error(cand, base_params=base) is None:
                return cand
        return {k: v.sample(rng) for k, v in space.items()}

    def _crossover(self, p1: Mapping[str, Any], p2: Mapping[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
        return {k: (p1[k] if rng.random() < 0.5 else p2[k]) for k in p1}

    def _mutate(self, ind: Mapping[str, Any], space: Mapping[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
        out = dict(ind)
        m_prob = float(self.mutation_probability if self.mutation_probability is not None else self.mutation_prob)
        for k, desc in space.items():
            if rng.random() < m_prob:
                out[k] = desc.mutate(out[k], rng)
        return out

    def fit(self, X: pd.DataFrame, y: Any, groups: Optional[Any] = None):
        rng = np.random.default_rng(int(self.random_state or 42))
        raw_space = self.param_space if self.param_space is not None else self.param_grid
        space = prepare_evolutionary_space(raw_space)
        cv = self.cv or GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=int(self.random_state or 42))
        base_params = self.estimator.get_params(deep=True)

        pop_size = max(2, int(self.population_size))
        n_gen = max(0, int(self.generations))
        population = [self._sample_individual(space, rng) for _ in range(pop_size)]

        cache: Dict[str, Tuple[List[float], float]] = {}
        all_params: List[Dict[str, Any]] = []
        all_fold_scores: List[List[float]] = []
        all_times: List[float] = []

        def evaluate_pop(pop: List[Dict[str, Any]], gen_idx: int) -> List[float]:
            means: List[float] = []
            for cand_idx, cand in enumerate(pop, start=1):
                sig = _canonical_candidate_key(cand, base_params=base_params)
                cached_hit = sig in cache
                if cached_hit:
                    fold_scores, elapsed = cache[sig]
                else:
                    fold_scores, elapsed = _evaluate_candidate_cv(
                        self.estimator, cand, X, y, groups, cv, self.scoring,
                        error_score=float(self.error_score),
                    )
                    cache[sig] = (fold_scores, elapsed)
                decoded = _decode_skopt_params(cand)
                all_params.append(decoded)
                all_fold_scores.append(fold_scores)
                all_times.append(elapsed)
                mean_sc = float(np.mean(fold_scores))
                means.append(mean_sc)
                if self.verbose and int(self.verbose) > 0:
                    tag = " (cached)" if cached_hit else ""
                    print(
                        f"[GA gen {gen_idx}/{n_gen} | cand {cand_idx}/{len(pop)}] "
                        f"cv_score={mean_sc:.4f} ({elapsed:.1f}s){tag}",
                        flush=True,
                    )
            return means

        fitness = evaluate_pop(population, 0)
        c_prob = float(self.crossover_probability if self.crossover_probability is not None else self.crossover_prob)
        for gen_i in range(1, n_gen + 1):
            next_pop: List[Dict[str, Any]] = []
            if self.elitism and len(population):
                best_i = int(np.argmax(fitness))
                next_pop.append(dict(population[best_i]))
            while len(next_pop) < pop_size:
                t_size = min(len(population), max(2, int(self.tournament_size)))
                idx1 = rng.choice(len(population), size=t_size, replace=False)
                p1 = population[int(idx1[int(np.argmax([fitness[i] for i in idx1]))])]
                idx2 = rng.choice(len(population), size=t_size, replace=False)
                p2 = population[int(idx2[int(np.argmax([fitness[i] for i in idx2]))])]
                child = self._crossover(p1, p2, rng) if rng.random() < c_prob else dict(p1)
                child = self._mutate(child, space, rng)
                next_pop.append(child)
            population = next_pop
            fitness = evaluate_pop(population, gen_i)

        return _finalize_search_results(self, all_params, all_fold_scores, all_times, X, y)

# Inherits predict / predict_proba / score from BaseSearchCV via BaseGASearchCV
GASearchCV = MultiBranchGASearchCV
HierarchicalGASearchCV = MultiBranchGASearchCV


class HierarchicalBayesSearchCV(BaseSearchCV):
    """Bayesian optimization search wrapper supporting JSON-encoded tuple/list categories and hierarchical caching.

    Iteration 0 deterministically evaluates the clean baseline starting point:
    augmentations switched off (0.0), minimal regularization (weight_decay=0.0,
    label_smoothing=0.0), all dropouts set to 0.0, and the first option of each
    Categorical dimension (or explicit ``x0`` if provided).
    """

    def __init__(
        self,
        estimator: Any,
        search_spaces: Mapping[str, Any],
        n_iter: int = 10,
        cv: Any = None,
        scoring: Any = None,
        refit: Any = True,
        random_state: int = 42,
        verbose: int = 0,
        error_score: Any = -1.0,
        n_jobs: int = 1,
        *,
        optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        pre_dispatch: Any = "n_jobs",
        return_train_score: bool = True,
        x0: Optional[Mapping[str, Any]] = None,
    ):
        self.estimator = estimator
        self.search_spaces = search_spaces
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.refit = refit
        self.random_state = random_state
        self.verbose = verbose
        self.error_score = error_score
        self.n_jobs = n_jobs
        self.optimizer_kwargs = optimizer_kwargs
        self.pre_dispatch = pre_dispatch
        self.return_train_score = return_train_score
        self.x0 = x0

    @staticmethod
    def _default_starting_point(keys: Sequence[str], dimensions: Sequence[Any], base_params: Mapping[str, Any], user_x0: Optional[Mapping[str, Any]] = None) -> List[Any]:
        """Construct the iteration-0 point with augmentations off, minimal regularization, and dropout=0."""
        point: List[Any] = []
        for k, dim in zip(keys, dimensions):
            is_zero_baseline = (
                "augmentor__" in k
                or "dropout" in k
                or k.endswith(("weight_decay", "label_smoothing"))
            )
            if user_x0 is not None and k in user_x0:
                val = user_x0[k]
                if hasattr(dim, "categories"):
                    cats = list(dim.categories)
                    if val in cats:
                        point.append(val)
                        continue
                    enc = json.dumps(val, sort_keys=True)
                    if enc in cats:
                        point.append(enc)
                        continue
                point.append(val)
                continue
            if hasattr(dim, "categories"):
                cats = list(dim.categories)
                if is_zero_baseline:
                    if 0.0 in cats:
                        point.append(0.0)
                        continue
                    if 0 in cats:
                        point.append(0)
                        continue
                point.append(cats[0])
            else:
                low, high = getattr(dim, "low", 0.0), getattr(dim, "high", 1.0)
                is_int = isinstance(low, (int, np.integer)) and isinstance(high, (int, np.integer))
                if is_zero_baseline and low <= 0 <= high:
                    point.append(int(0) if is_int else 0.0)
                elif k in base_params and isinstance(base_params[k], (int, float, np.number)) and low <= float(base_params[k]) <= high:
                    point.append(int(round(float(base_params[k]))) if is_int else float(base_params[k]))
                else:
                    point.append(int(low) if is_int else float(low))
        return point

    def fit(self, X: pd.DataFrame, y: Any, groups: Optional[Any] = None):
        _require(SKOPT_AVAILABLE, "scikit-optimize")
        from skopt import Optimizer

        cv = self.cv or GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=int(self.random_state or 42))
        space_dict = prepare_bayesian_space(self.search_spaces, skopt_available=True)
        keys = list(space_dict.keys())
        dimensions = [space_dict[k] for k in keys]
        opt_kwargs = dict(self.optimizer_kwargs or {})
        opt_kwargs.setdefault("random_state", int(self.random_state or 42))
        opt = Optimizer(dimensions, **opt_kwargs)

        base_params = self.estimator.get_params(deep=True)
        cache: Dict[str, Tuple[List[float], float]] = {}
        all_params: List[Dict[str, Any]] = []
        all_fold_scores: List[List[float]] = []
        all_times: List[float] = []

        initial_point = self._default_starting_point(keys, dimensions, base_params, self.x0)
        total_iters = max(1, int(self.n_iter))
        best_so_far = -float("inf")
        for it in range(total_iters):
            x_point = initial_point if it == 0 else opt.ask()
            raw_cand = dict(zip(keys, x_point))
            decoded = _decode_skopt_params(raw_cand)
            sig = _canonical_candidate_key(decoded, base_params=base_params)
            cached_hit = sig in cache
            if cached_hit:
                fold_scores, elapsed = cache[sig]
            else:
                if self.error_score == "raise":
                    err = candidate_error(decoded, base_params=base_params)
                    if err is not None:
                        raise InvalidParams(err)
                fold_scores, elapsed = _evaluate_candidate_cv(
                    self.estimator, decoded, X, y, groups, cv, self.scoring,
                    error_score=-1.0 if self.error_score == "raise" else float(self.error_score),
                )
                cache[sig] = (fold_scores, elapsed)

            mean_sc = float(np.mean(fold_scores))
            if mean_sc > best_so_far:
                best_so_far = mean_sc
            opt.tell(x_point, -mean_sc)
            all_params.append(decoded)
            all_fold_scores.append(fold_scores)
            all_times.append(elapsed)
            if self.verbose and int(self.verbose) > 0:
                tag = " (cached)" if cached_hit else ""
                print(
                    f"[Bayes {it + 1}/{total_iters}] cv_score={mean_sc:.4f} | "
                    f"best={best_so_far:.4f} ({elapsed:.1f}s){tag}",
                    flush=True,
                )

        return _finalize_search_results(self, all_params, all_fold_scores, all_times, X, y)
    # Inherits predict / predict_proba / score from BaseSearchCV


# -- Data loading & validation protocol helpers ------------------------------

# ``find_data_root`` is imported directly from ``base_utils`` (no duplication).

def sample_balanced_split(
    df: pd.DataFrame,
    target_col: str = "bfrb",
    fraction: float = 1.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample a sequence-balanced subset of ``df`` preserving class distribution (delegates to ``search_drivers``)."""
    return sample_balanced_subset(df, target_col=target_col, fraction=fraction, random_state=random_state, seq_col=SEQ_COL)


def validation_protocols(space: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Optional[str]]]:
    """Expand ``VALIDATION_PARAM_SPACE`` into protocol dictionaries."""
    if not space:
        return [{"validation_mode": "sequence", "group_column": None}]
    modes = space.get("validation_mode", ["sequence"])
    if isinstance(modes, str):
        modes = [modes]
    protocols: List[Dict[str, Optional[str]]] = []
    for mode in modes:
        key = str(mode).strip().lower()
        if key == "sequence":
            protocols.append({"validation_mode": "sequence", "group_column": None})
        elif key == "subject":
            protocols.append({"validation_mode": "subject", "group_column": "subject"})
        else:
            raise ValueError(f"Unsupported validation_mode {mode!r}; choose 'sequence' or 'subject'")
    return protocols


def split_validation_protocol(
    df: pd.DataFrame,
    protocol: Mapping[str, Optional[str]],
    *,
    train_size: float = 0.8,
    n_splits: int = 3,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, Any]:
    """Split ``df`` into ``(X_train, X_test, groups, cv_object)`` according to ``protocol``."""
    mode = protocol.get("validation_mode", "sequence")
    if mode not in {"sequence", "subject"}:
        raise ValueError(f"Unsupported validation_mode {mode!r}")
    if mode == "subject" and "subject" not in df.columns:
        raise ValueError("subject validation_mode requires a 'subject' column in the dataframe")

    group_col = "subject" if mode == "subject" else SEQ_COL
    test_size = max(0.05, min(0.95, 1.0 - float(train_size)))

    seq_df = df.groupby(SEQ_COL, sort=False).first().reset_index()
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=int(random_state))
    tr_idx, te_idx = next(splitter.split(seq_df, groups=seq_df[group_col]))

    # If the dataset has both target and non-target domains, ensure both appear in train and test when possible
    if "is_target" in seq_df.columns and seq_df["is_target"].nunique() >= 2:
        for seed_offset in range(1, 20):
            if (seq_df.iloc[tr_idx]["is_target"].nunique() >= 2
                    and seq_df.iloc[te_idx]["is_target"].nunique() >= 2):
                break
            alt = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=int(random_state) + seed_offset)
            tr_idx, te_idx = next(alt.split(seq_df, groups=seq_df[group_col]))

    train_seqs = set(seq_df.iloc[tr_idx][SEQ_COL].astype(str))
    test_seqs = set(seq_df.iloc[te_idx][SEQ_COL].astype(str))

    X_train = df[df[SEQ_COL].astype(str).isin(train_seqs)].copy()
    X_test = df[df[SEQ_COL].astype(str).isin(test_seqs)].copy()
    groups = X_train[group_col]

    if int(n_splits) <= 1:
        cv_object: Any = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=int(random_state))
    else:
        n_unique_groups = X_train[group_col].nunique()
        eff_splits = max(2, min(int(n_splits), int(n_unique_groups)))
        cv_object = GroupKFold(n_splits=eff_splits)

    return X_train, X_test, groups, cv_object


def build_quick_results_df(search_or_results: Any) -> pd.DataFrame:
    """Convert ``search.cv_results_`` (or a search object) into a sorted summary DataFrame."""
    return build_quick_results_df_shared(search_or_results)


def run_validation_searches(
    search: Any,
    experiment_df: pd.DataFrame,
    *,
    target_col: str = "bfrb",
    validation_space: Optional[Mapping[str, Any]] = None,
    validation_param_space: Optional[Mapping[str, Any]] = None,
    sensor_columns: Optional[Sequence[str]] = None,
    train_size: float = 0.8,
    n_splits: int = 3,
    random_state: int = 42,
    results_dir: Any = "results",
    experiment_name: str = "multibranch",
    search_mode: str = "bayes",
    timestamp: Optional[str] = None,
    simulate_hidden_test: bool = False,
    hidden_test_null_fraction: float = 0.5,
    quick_save: bool = True,
    save_ensemble_bundle: bool = True,
    save_oof: Any = "auto",
    bundle_dir: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run hyperparameter search across configured validation protocols, evaluate holdout, and persist ensemble bundles."""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = timestamp or time.strftime("%Y%m%d_%H%M%S")
    eff_vspace = validation_param_space if validation_param_space is not None else validation_space
    protocols = validation_protocols(eff_vspace)

    label_cols = [c for c in (SEQ_COL, "subject", "is_target", "gesture", "bfrb", "gesture_action", "gesture_position", "orientation")
                  if c in experiment_df.columns]
    if target_col not in label_cols and target_col in experiment_df.columns:
        label_cols.append(target_col)

    runs: List[Dict[str, Any]] = []
    protocol_results: List[Dict[str, Any]] = []
    all_experiment_rows: List[pd.DataFrame] = []
    last_run: Dict[str, Any] = {}

    for protocol in protocols:
        v_mode = protocol["validation_mode"]
        g_col = protocol["group_column"]
        X_train, X_test, groups, cv_object = split_validation_protocol(
            experiment_df, protocol,
            train_size=train_size, n_splits=n_splits, random_state=random_state,
        )
        n_tr_seq = X_train[SEQ_COL].nunique()
        n_te_seq = X_test[SEQ_COL].nunique()
        print(
            f"\nValidation protocol: mode={v_mode} | group_column={g_col} | "
            f"Train: {n_tr_seq} seqs ({len(X_train)} rows) | Test: {n_te_seq} seqs ({len(X_test)} rows)",
            flush=True,
        )
        y_train = X_train[label_cols].copy()
        y_test = X_test[label_cols].copy()

        run_search = clone(search)
        if hasattr(run_search, "cv"):
            run_search.cv = cv_object
        est = run_search.estimator
        if hasattr(est, "named_steps") and "estimator" in est.named_steps:
            est.set_params(estimator__validation_group_column=g_col, estimator__primary_target=target_col)
        elif hasattr(est, "validation_group_column"):
            est.set_params(validation_group_column=g_col, primary_target=target_col)

        run_search.fit(X_train, y_train, groups=groups)
        best_model = getattr(run_search, "best_estimator_", None)
        best_score = getattr(run_search, "best_score_", np.nan)
        if not np.isfinite(best_score) and getattr(run_search, "best_index_", None) is not None:
            mean_scores = getattr(run_search, "cv_results_", {}).get("mean_test_score")
            if mean_scores is not None:
                best_score = float(mean_scores[int(run_search.best_index_)])
        best_score = float(best_score)
        best_params = dict(getattr(run_search, "best_params_", {}))

        best_est = (
            best_model.named_steps["estimator"]
            if best_model is not None and hasattr(best_model, "named_steps")
            else best_model
        )
        best_fit_ok = best_model is not None and not getattr(best_est, "fit_failed_", False)
        succeeded = best_fit_ok and np.isfinite(best_score) and best_score > -1.0

        eval_res: Dict[str, Any] = {}
        ho_proba: Optional[np.ndarray] = None
        ho_seq_ids: Optional[pd.Index] = None
        ho_y_true: Optional[pd.Series] = None
        saved_bundle_path: Optional[str] = None

        if succeeded and len(X_test):
            y_pred = best_model.predict(X_test)
            eval_res = evaluate_holdout(
                y_test, y_pred, target_col=target_col, verbose=True,
                model=best_model, X_test=X_test,
                simulate_hidden_test=simulate_hidden_test,
                null_fraction=hidden_test_null_fraction,
                random_state=random_state,
            )
            if save_ensemble_bundle and hasattr(best_model, "predict_proba"):
                ho_seq_ids = pd.Index(X_test[SEQ_COL].astype(str).unique(), name=SEQ_COL)
                ho_proba = best_model.predict_proba(X_test)
                ho_y_meta = y_test.drop_duplicates(SEQ_COL).set_index(y_test.drop_duplicates(SEQ_COL)[SEQ_COL].astype(str))[target_col]
                ho_y_true = ho_y_meta.reindex(ho_seq_ids)

        if succeeded and save_ensemble_bundle and hasattr(best_est, "save_bundle"):
            b_target = (
                Path(bundle_dir)
                if bundle_dir is not None else
                out_dir / f"{experiment_name}_{ts}_{v_mode}_bundle"
            )
            oof_proba: Optional[np.ndarray] = None
            oof_seq_ids: Optional[pd.Index] = None
            oof_y_true: Optional[pd.Series] = None
            if isinstance(best_est, MultiBranchTSAIClassifier):
                want_oof = bool(save_oof) if isinstance(save_oof, bool) else (int(n_splits) >= 2)
                if want_oof:
                    seq_all = pd.Index(X_train[SEQ_COL].astype(str).unique(), name=SEQ_COL)
                    tr_y_meta = y_train.drop_duplicates(SEQ_COL).set_index(
                        y_train.drop_duplicates(SEQ_COL)[SEQ_COL].astype(str)
                    )[target_col].reindex(seq_all)
                    oof_mat = np.zeros((len(seq_all), len(best_est.classes_)), dtype=np.float32)
                    filled = np.zeros(len(seq_all), dtype=bool)
                    seq_s = X_train[SEQ_COL].astype(str)
                    splits = list(cv_object.split(X_train, y_train, groups=groups))
                    if len(splits) >= 2:
                        for tr_i, va_i in splits:
                            X_tr_f, X_va_f = X_train.iloc[tr_i], X_train.iloc[va_i]
                            y_tr_f = y_train.iloc[tr_i]
                            f_est = clone(best_est)
                            f_est.fit(X_tr_f, y_tr_f)
                            if not getattr(f_est, "fit_failed_", False):
                                va_ids = pd.Index(X_va_f[SEQ_COL].astype(str).unique())
                                aligned = MultiBranchEnsemble._align_proba(
                                    f_est.predict_proba(X_va_f), f_est.classes_, best_est.classes_,
                                )
                                pos_map = {s: idx_s for idx_s, s in enumerate(seq_all)}
                                for r_i, s_id in enumerate(va_ids):
                                    if s_id in pos_map:
                                        oof_mat[pos_map[s_id]] = aligned[r_i]
                                        filled[pos_map[s_id]] = True
                        if not filled.all():
                            rem_seqs = set(seq_all[~filled])
                            X_rem = X_train.loc[seq_s.isin(rem_seqs)]
                            if len(X_rem):
                                rem_ids = pd.Index(X_rem[SEQ_COL].astype(str).unique())
                                rem_p = best_model.predict_proba(X_rem)
                                pos_map = {s: idx_s for idx_s, s in enumerate(seq_all)}
                                for r_i, s_id in enumerate(rem_ids):
                                    if s_id in pos_map:
                                        oof_mat[pos_map[s_id]] = rem_p[r_i]
                        oof_proba, oof_seq_ids, oof_y_true = oof_mat, seq_all, tr_y_meta
                elif save_oof == "auto" and len(getattr(best_est, "validation_sequence_ids_", [])):
                    val_ids = pd.Index(best_est.validation_sequence_ids_.astype(str))
                    X_iv = X_train.loc[X_train[SEQ_COL].astype(str).isin(set(val_ids))]
                    if len(X_iv):
                        oof_seq_ids = pd.Index(X_iv[SEQ_COL].astype(str).unique(), name=SEQ_COL)
                        oof_proba = best_model.predict_proba(X_iv)
                        tr_y_meta = y_train.drop_duplicates(SEQ_COL).set_index(
                            y_train.drop_duplicates(SEQ_COL)[SEQ_COL].astype(str)
                        )[target_col]
                        oof_y_true = tr_y_meta.reindex(oof_seq_ids)

            saved_path = best_est.save_bundle(
                b_target,
                oof_proba=oof_proba,
                oof_sequence_ids=oof_seq_ids,
                oof_y=oof_y_true,
                holdout_proba=ho_proba,
                holdout_sequence_ids=ho_seq_ids,
                holdout_y=ho_y_true,
                cv_score=best_score,
                holdout_score=eval_res.get("competition_score"),
            )
            saved_bundle_path = str(saved_path)

        cv_df = pd.DataFrame(run_search.cv_results_)
        cv_df["validation_mode"] = v_mode
        cv_df["search_mode"] = search_mode
        cv_df["best_cv_score"] = best_score
        cv_df["holdout_competition_score"] = eval_res.get("competition_score", np.nan)
        cv_df["holdout_hidden_test_score"] = eval_res.get("hidden_test_competition_score", np.nan)
        all_experiment_rows.append(cv_df)

        summary_row = {
            "validation_mode": v_mode,
            "group_column": g_col,
            "search_mode": search_mode,
            "best_cv_score": best_score,
            "holdout_score": eval_res.get("competition_score", np.nan),
            "holdout_competition_score": eval_res.get("competition_score", np.nan),
            "holdout_binary_f1": eval_res.get("binary_f1", np.nan),
            "holdout_gesture_f1": eval_res.get("gesture_f1", np.nan),
            "holdout_macro_f1": eval_res.get("macro_f1", np.nan),
            "holdout_hidden_test_competition_score": eval_res.get("hidden_test_competition_score", np.nan),
            "holdout_hidden_test_binary_f1": eval_res.get("hidden_test_binary_f1", np.nan),
            "holdout_hidden_test_gesture_f1": eval_res.get("hidden_test_gesture_f1", np.nan),
        }
        protocol_results.append(summary_row)
        run_entry = {
            "validation_mode": v_mode,
            "protocol": protocol,
            "search": run_search,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "groups": groups,
            "cv_object": cv_object,
            "best_score": best_score,
            "best_params": best_params,
            "eval_results": eval_res,
            "holdout_results": eval_res,
            "experiment_succeeded": succeeded,
            "bundle_dir": saved_bundle_path,
        }
        runs.append(run_entry)
        last_run = dict(run_entry)

    experiments_df = pd.concat(all_experiment_rows, ignore_index=True) if all_experiment_rows else pd.DataFrame()
    exp_path = out_dir / f"{experiment_name}_{ts}_experiments.csv"
    sum_path = out_dir / f"{experiment_name}_{ts}_summary.json"
    experiments_df.to_csv(exp_path, index=False)
    sum_path.write_text(
        json.dumps({
            "experiment_name": experiment_name,
            "target_col": target_col,
            "search_mode": search_mode,
            "timestamp": ts,
            "protocol_results": protocol_results,
        }, indent=2, default=str),
        encoding="utf-8",
    )

    return {
        "last_run": last_run,
        "runs": runs,
        "protocol_results": protocol_results,
        "results_dir": out_dir,
        "experiments_df": experiments_df,
        "experiments_path": str(exp_path),
        "summary_path": str(sum_path),
        "bundle_dir": last_run.get("bundle_dir"),
    }


# -- Trained-Model Sequence-Aligned Permutation Importance -------------------

def _sequence_layout(sequence_ids: np.ndarray, mask: np.ndarray):
    """Group window indices by sequence and stratify by (window_count, mask_signature)."""
    seq_str = np.asarray(sequence_ids, dtype=str)
    unique_seqs: List[str] = []
    rows_by_seq: List[np.ndarray] = []
    strata: Dict[Tuple[int, bytes], List[int]] = {}

    seen: Dict[str, List[int]] = {}
    for idx, s in enumerate(seq_str):
        if s not in seen:
            unique_seqs.append(s)
            seen[s] = []
        seen[s].append(idx)

    for s_idx, s in enumerate(unique_seqs):
        r = np.asarray(seen[s], dtype=int)
        rows_by_seq.append(r)
        key = (len(r), np.asarray(mask[r], dtype=bool).tobytes())
        strata.setdefault(key, []).append(s_idx)

    return unique_seqs, rows_by_seq, strata


def _donor_rows(
    rows_by_seq: List[np.ndarray],
    strata: Mapping[Tuple[int, bytes], List[int]],
    rng: np.random.Generator,
    n_total: int,
) -> Tuple[np.ndarray, int]:
    """Permute sequences within matching window-layout strata while preserving window order."""
    donor = np.arange(n_total, dtype=int)
    moved_sequences = 0
    for seq_indices in strata.values():
        if len(seq_indices) <= 1:
            continue
        k = len(seq_indices)
        shift = int(rng.integers(1, k))
        order = list(rng.permutation(k))
        # Derangement: ensure no sequence maps to itself when k >= 2
        if any(order[i] == i for i in range(k)):
            order = [(i + shift) % k for i in range(k)]
        for dst_pos, src_pos in enumerate(order):
            dst_rows = rows_by_seq[seq_indices[dst_pos]]
            src_rows = rows_by_seq[seq_indices[src_pos]]
            donor[dst_rows] = src_rows
            if dst_pos != src_pos:
                moved_sequences += 1
    return donor, moved_sequences


def save_permutation_importance(
    importance_df: pd.DataFrame,
    output_dir: Any,
    prefix: str = "permutation_importance",
) -> Dict[str, str]:
    """Save permutation importance table (.csv), metadata (.json), and bar plot (.png)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{prefix}.csv"
    json_path = out / f"{prefix}.json"
    png_path = out / f"{prefix}.png"

    importance_df.to_csv(csv_path, index=False)
    payload = {
        "metadata": {
            "scope": importance_df.attrs.get("scope", "saved internal validation sequences"),
            "baseline_macro_f1": float(importance_df.attrs.get("baseline_macro_f1", 0.0)),
            "baseline_competition_score": float(importance_df.attrs.get("baseline_competition_score", 0.0)),
            "permutable_fraction": float(importance_df.attrs.get("permutable_fraction", 0.0)),
            "repeats": int(importance_df.attrs.get("repeats", 1)),
        },
        "importance": importance_df.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, max(3.5, 0.35 * len(importance_df))))
        ordered = importance_df.sort_values("importance_mean", ascending=True)
        labels = [f"{r['kind']}: {r['name']}" for _, r in ordered.iterrows()]
        ax.barh(labels, ordered["importance_mean"], xerr=ordered["importance_std"], color="#2b6cb0", alpha=0.85)
        ax.axvline(0.0, color="#4a5568", linestyle="--", linewidth=1)
        ax.set_xlabel("Drop in Macro F1 (higher = more important)")
        ax.set_title("Validation Permutation Importance (v3 Multimodal Stack)")
        fig.tight_layout()
        fig.savefig(png_path, dpi=140)
        plt.close(fig)
    except Exception:
        # Write a minimal valid 1x1 PNG header if matplotlib is unavailable in a headless test env
        png_path.write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
            b"\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
        )

    paths = {"csv": str(csv_path), "json": str(json_path), "png": str(png_path), "plot": str(png_path)}
    importance_df.attrs["paths"] = paths
    return paths


def compute_permutation_importance(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.DataFrame,
    *,
    repeats: int = 5,
    random_state: int = 42,
    include_noop_individual: bool = True,
    require_internal_validation: bool = True,
    output_dir: Optional[Any] = None,
    prefix: str = "permutation_importance",
) -> pd.DataFrame:
    """Compute sequence-aligned permutation importance on the internal validation partition.

    Extracts chunks once, verifies partition hygiene (rejects training or holdout sequences
    when ``require_internal_validation=True``), permutes modality groups and individual
    NOOP/static features across sequences with matching window layouts, and always restores
    chunk arrays in a ``finally`` block.
    """
    if not isinstance(repeats, (int, np.integer)) or int(repeats) < 1:
        raise ValueError("repeats must be a positive integer")

    est = estimator.named_steps["estimator"] if hasattr(estimator, "named_steps") and "estimator" in estimator.named_steps else estimator
    extractor = getattr(est, "extractor_", None) or getattr(est, "extractor", None)
    seq_col = getattr(extractor, "sequence_col", SEQ_COL)
    target_col = getattr(est, "primary_target", "bfrb")

    seq_ids_in_x = pd.Index(X[seq_col].astype(str).unique())
    if require_internal_validation:
        val_ids = pd.Index(getattr(est, "validation_sequence_ids_", [])).astype(str)
        train_ids = pd.Index(getattr(est, "train_sequence_ids_", [])).astype(str)
        if not len(val_ids) or not set(seq_ids_in_x).issubset(set(val_ids)):
            raise ValueError("Provided X contains sequences outside that partition (validation_sequence_ids_)")
        if set(seq_ids_in_x) & set(train_ids):
            raise ValueError("Provided X contains training sequences; importance must use internal validation only")

    # Validate sequence labels
    if seq_col not in y.columns or target_col not in y.columns:
        raise ValueError(f"y must contain {seq_col!r} and {target_col!r}")
    y_grouped = y.groupby(y[seq_col].astype(str), sort=False)[target_col].nunique()
    if (y_grouped > 1).any():
        raise ValueError(f"Target {target_col!r} must be constant within each sequence")
    label_map = y.groupby(y[seq_col].astype(str), sort=False)[target_col].first()
    if not set(seq_ids_in_x).issubset(set(label_map.index)):
        raise ValueError("Missing labels for one or more sequences in X")
    y_true = label_map.reindex(seq_ids_in_x).astype(str)

    chunks = extractor.transform_chunks(X)
    c_seq_ids = np.asarray(chunks["sequence_ids"], dtype=str)
    primary_mask = chunks["masks"]["imu"] if "masks" in chunks and "imu" in chunks["masks"] else chunks["mask"]
    unique_seqs, rows_by_seq, strata = _sequence_layout(c_seq_ids, primary_mask)
    permutable_seqs = sum(len(v) for v in strata.values() if len(v) > 1)
    permutable_fraction = permutable_seqs / max(1, len(unique_seqs))

    def score_current_chunks() -> Tuple[float, float]:
        if hasattr(est, "predict_window_proba"):
            win_proba = est.predict_window_proba(chunks)
            _, seq_proba = est.aggregate_window_proba(win_proba, c_seq_ids, seq_order=seq_ids_in_x)
            classes = np.asarray(est.le_.classes_)
        else:
            branches = est._branches_from_chunks(chunks)
            inputs = est._model_inputs(branches, chunks["mask"])
            win_proba = est.model_.predict(inputs, verbose=0)
            classes = np.asarray(est.le_.classes_)
            seq_proba = np.vstack([
                win_proba[c_seq_ids == s].mean(axis=0) for s in seq_ids_in_x
            ])
        preds = pd.Series(classes[seq_proba.argmax(axis=1)], index=seq_ids_in_x)
        macro = float(f1_score(y_true, preds, average="macro", zero_division=0))
        comp = float(competition_score(y_true, preds)["competition_score"]) if target_col == "bfrb" else macro
        return macro, comp

    baseline_macro, baseline_comp = score_current_chunks()

    # Build permutation specs
    specs: List[Tuple[str, str, Any]] = []
    if "branches" in chunks:
        # v3 MultiModalSequenceExtractor chunk structure
        ch_names = chunks["channel_names"]
        st_names = list(chunks["static_names"])
        imu_names = ch_names.get("imu", [])
        acc_ch = [i for i, n in enumerate(imu_names) if n.startswith(("acc_", "lin_", "jerk_"))]
        rot_ch = [i for i, n in enumerate(imu_names) if n.startswith(("rot_", "rot6d_", "ang_", "angular_", "euler_"))]
        oss_prefixes = ("tsfel__", "tsfresh__", "catch22__", "tsflex__", "seglearn__", "sktime__")
        domain_prefixes = ("orientation__", "vib__", "tof_spatial__", "tof_temp__", "thm_dyn__", "xmodal__", "evcomp__", "mscale__", "seq__")
        noop_name_set = set(chunks.get("noop_feature_names", []))
        imu_st = [
            i for i, n in enumerate(st_names)
            if (n in noop_name_set or any(tok in n for tok in ("acc_mag", "lin_mag", "ang_speed", "noop_", "orientation__", "vib__", "evcomp__", "mscale__")))
        ]
        thm_st = [i for i, n in enumerate(st_names) if "thm" in n]
        tof_st = [i for i, n in enumerate(st_names) if "tof" in n]
        gen_st = [i for i, n in enumerate(st_names) if n.startswith(oss_prefixes)]
        dom_st = [i for i, n in enumerate(st_names) if n.startswith(domain_prefixes)]
        noop_st = [i for i, n in enumerate(st_names) if n in noop_name_set or n.startswith("noop_")]

        specs.append(("group", "motion_coherent", {"branches": {"imu": list(range(len(imu_names)))}, "static": imu_st}))
        if acc_ch:
            specs.append(("group", "imu_acceleration", {"branches": {"imu": acc_ch}, "static": []}))
        if rot_ch:
            specs.append(("group", "imu_rotation", {"branches": {"imu": rot_ch}, "static": []}))
        if "thm" in chunks["branches"]:
            specs.append(("group", "thm", {"branches": {"thm": list(range(chunks["branches"]["thm"].shape[2]))}, "static": thm_st}))
        if "tof" in chunks["branches"]:
            specs.append(("group", "tof", {"branches": {"tof": list(range(chunks["branches"]["tof"].shape[2]))}, "static": tof_st}))
        if gen_st:
            specs.append(("group", "static_open_source", {"branches": {}, "static": gen_st}))
        if dom_st:
            specs.append(("group", "domain_families", {"branches": {}, "static": dom_st}))
        if noop_st:
            specs.append(("group", "noop_all", {"branches": {}, "static": noop_st}))

        if include_noop_individual:
            for feat_name in chunks.get("noop_feature_names", []):
                if feat_name.endswith("_supported"):
                    continue
                cols = [st_names.index(feat_name)] if feat_name in st_names else []
                sup_name = f"{feat_name}_supported"
                if sup_name in st_names:
                    cols.append(st_names.index(sup_name))
                if cols:
                    specs.append(("noop_feature", feat_name, {"branches": {}, "static": cols}))
    else:
        # Legacy fixture compatibility
        f_names = list(chunks.get("feature_names", []))
        noop_names = list(chunks.get("noop_feature_names", []))
        acc_idx = [i for i, n in enumerate(f_names) if not n.startswith("tof")]
        tof_idx = [i for i, n in enumerate(f_names) if n.startswith("tof")]
        specs.append(("group", "motion_coherent", {"X": acc_idx, "noop": list(range(len(noop_names)))}))
        specs.append(("group", "tof", {"X": tof_idx, "noop": []}))
        if include_noop_individual:
            for j, name in enumerate(noop_names):
                specs.append(("noop_feature", name, {"X": [], "noop": [j]}))

    # Snapshot originals so they are always restored
    if "branches" in chunks:
        orig_branches = {k: v.copy() for k, v in chunks["branches"].items()}
        orig_static = chunks["static"].copy()
    else:
        orig_X = chunks["X"].copy()
        orig_noop = chunks["noop_features"].copy()

    records: List[Dict[str, Any]] = []
    try:
        for kind, name, target_spec in specs:
            rng = np.random.default_rng(int(random_state))
            drops: List[float] = []
            perm_scores: List[float] = []
            for _ in range(int(repeats)):
                donor, _ = _donor_rows(rows_by_seq, strata, rng, len(c_seq_ids))
                if "branches" in chunks:
                    for b_name, ch_idx in target_spec["branches"].items():
                        chunks["branches"][b_name][:, :, ch_idx] = orig_branches[b_name][donor][:, :, ch_idx]
                    if target_spec["static"]:
                        s_idx = target_spec["static"]
                        chunks["static"][:, s_idx] = orig_static[donor][:, s_idx]
                else:
                    if target_spec["X"]:
                        x_idx = target_spec["X"]
                        chunks["X"][:, :, x_idx] = orig_X[donor][:, :, x_idx]
                    if target_spec["noop"]:
                        n_idx = target_spec["noop"]
                        chunks["noop_features"][:, n_idx] = orig_noop[donor][:, n_idx]

                perm_macro, _ = score_current_chunks()
                perm_scores.append(perm_macro)
                drops.append(baseline_macro - perm_macro)

                # Restore after each repeat
                if "branches" in chunks:
                    for b_name in target_spec["branches"]:
                        chunks["branches"][b_name][:] = orig_branches[b_name]
                    if target_spec["static"]:
                        chunks["static"][:] = orig_static
                else:
                    chunks["X"][:] = orig_X
                    chunks["noop_features"][:] = orig_noop

            records.append({
                "kind": kind,
                "name": name,
                "importance_mean": float(np.mean(drops)),
                "importance_std": float(np.std(drops)),
                "permuted_macro_f1_mean": float(np.mean(perm_scores)),
                "baseline_macro_f1": float(baseline_macro),
                "n_sequences": int(len(unique_seqs)),
                "n_permutable_sequences": int(permutable_seqs),
            })
    finally:
        if "branches" in chunks:
            for b_name, arr in orig_branches.items():
                chunks["branches"][b_name][:] = arr
            chunks["static"][:] = orig_static
        else:
            chunks["X"][:] = orig_X
            chunks["noop_features"][:] = orig_noop

    result = pd.DataFrame(records).sort_values("importance_mean", ascending=False, kind="stable").reset_index(drop=True)
    result.attrs["baseline_macro_f1"] = float(baseline_macro)
    result.attrs["baseline_competition_score"] = float(baseline_comp)
    result.attrs["permutable_fraction"] = float(permutable_fraction)
    result.attrs["repeats"] = int(repeats)
    result.attrs["scope"] = "saved internal validation sequences"

    if output_dir is not None:
        save_permutation_importance(result, output_dir, prefix=prefix)
    return result


# -- Synthetic dataset builder for fast unit & integration testing -----------

def make_synthetic_cmi_df(
    n_subjects: int = 6,
    sequences_per_subject: int = 4,
    n_timesteps: int = 60,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create a realistic synthetic CMI multimodal dataset with IMU, Thermopile, and 5x8x8 ToF."""
    rng = np.random.default_rng(int(random_state))
    gestures = [
        ("Forehead - pull hairline", 1, "Forehead", "pull hairline"),
        ("Cheek - pinch skin", 1, "Cheek", "pinch skin"),
        ("Drink from bottle/cup", 0, "other", "other"),
        ("Wave hello", 0, "other", "other"),
    ]
    frames: List[pd.DataFrame] = []
    for subj_i in range(int(n_subjects)):
        subj = f"SUBJ_{subj_i:03d}"
        for seq_j in range(int(sequences_per_subject)):
            g_name, is_target, g_pos, g_act = gestures[seq_j % len(gestures)]
            bfrb = g_name if is_target == 1 else "non_bfrb"
            seq_id = f"{subj}_SEQ_{seq_j:02d}"
            t = np.arange(int(n_timesteps), dtype=float) / 100.0
            freq = 2.0 + 0.8 * (seq_j % len(gestures))
            amp = 1.5 if is_target == 1 else 0.5
            phase = rng.uniform(0, 2 * np.pi)

            acc_x = amp * np.sin(2 * np.pi * freq * t + phase) + rng.normal(0, 0.05, len(t))
            acc_y = amp * np.cos(2 * np.pi * freq * t + phase) + rng.normal(0, 0.05, len(t))
            acc_z = GRAVITY + 0.3 * amp * np.sin(4 * np.pi * freq * t) + rng.normal(0, 0.05, len(t))

            angle = 0.25 * amp * np.sin(2 * np.pi * freq * t)
            rot_w = np.cos(angle / 2.0)
            axis_v = np.sin(angle / 2.0) / np.sqrt(3.0)

            data: Dict[str, Any] = {
                SEQ_COL: seq_id,
                "subject": subj,
                COUNTER_COL: np.arange(int(n_timesteps), dtype=int),
                "is_target": int(is_target),
                "gesture": g_name,
                "bfrb": bfrb,
                "gesture_position": g_pos,
                "gesture_action": g_act,
                "orientation": "Seated Straight",
                "acc_x": acc_x,
                "acc_y": acc_y,
                "acc_z": acc_z,
                "rot_w": rot_w,
                "rot_x": axis_v,
                "rot_y": axis_v,
                "rot_z": axis_v,
            }
            # Thermopile and ToF are held across 5 rows (20 Hz on a 100 Hz row clock)
            t_hold = np.floor(np.arange(int(n_timesteps)) / 5.0) * (5.0 / 100.0)
            for s in range(1, 6):
                data[f"thm_{s}"] = 31.0 + 0.8 * is_target + 0.1 * s + 0.2 * np.sin(2 * np.pi * freq * t_hold)
            for s in range(1, 6):
                base_dist = 120.0 if is_target == 1 else 260.0
                for p in range(64):
                    col_vals = base_dist + s * 4.0 + (p % 8) * 1.5 + 10.0 * np.cos(2 * np.pi * freq * t_hold)
                    if p == 63:
                        col_vals = np.where(np.arange(int(n_timesteps)) % 15 == 0, -1.0, col_vals)
                    data[f"tof_{s}_v{p}"] = col_vals
            frames.append(pd.DataFrame(data))
    return pd.concat(frames, ignore_index=True)
