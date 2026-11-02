"""Optional chunk-local motion descriptors alongside the unchanged CMI extractor.

The general DSP ideas are inspired by NOOP's interpretable motion/regularity
descriptors. This is an independent implementation, not a port of NOOP source.
All additions are disabled by default. Missing observations are never compacted
into a faster artificial clock, and unavailable estimates have explicit support
flags rather than non-finite values.
"""
from __future__ import annotations

import ast
import json
from contextlib import contextmanager

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

try:
    from .base_utils import (
        SequenceExtractor, RotationExtractor, InvalidExtractorParams,
        _one_sided_psd, _positive_rate,
    )
except ImportError:
    from base_utils import (
        SequenceExtractor, RotationExtractor, InvalidExtractorParams,
        _one_sided_psd, _positive_rate,
    )


_MOTION = {"acc_ac_rms", "axis_rms", "jerk_rms", "rotation_mean", "rotation_rms"}
_SPECTRAL = {"peak_fraction", "peak_neighbourhood_fraction", "band_fractions", "entropy"}
_MODE_PARAMETERS = ("motion_stats", "periodicity", "spectral_stats", "quality")
_TUNING_PARAMETERS = (
    "signal", "band", "bands", "min_cycles", "min_samples", "max_gap_samples",
    "acf_harmonic_tolerance", "spectral_neighbourhood_hz", "detrend",
)


def _structured(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError) as exc:
                raise InvalidExtractorParams("NOOP frequency ranges must be lists/tuples or their JSON representation") from exc
    return value


def _modes(value, allowed, *, name, aliases=None):
    if value is None:
        return set()
    if not isinstance(value, str):
        raise InvalidExtractorParams(f"noop__{name} must be None or a categorical mode string")
    if aliases and value in aliases:
        return set(aliases[value])
    result = {part.strip() for part in value.split("|") if part.strip()}
    if not result or not result <= allowed:
        raise InvalidExtractorParams(f"Invalid noop__{name}: {value!r}; expected {sorted(allowed)} or None")
    return result


class NoopOptions(BaseEstimator):
    """Nested sklearn parameters; actual ``None`` is the canonical off value.

    ``signal`` selects the signed signal used for periodicity/spectral features.
    Motion statistics always describe raw acceleration (in its declared input
    units) or quaternion-derived angular speed (rad/s). ``quality`` never removes
    windows. ``report`` returns metadata only; ``features`` also feeds quality
    descriptors to the optional static model branch.
    """

    def __init__(
        self, motion_stats=None, periodicity=None, spectral_stats=None, quality=None,
        signal="raw", band=(0.5, 8.0), bands=((0.5, 2.0), (2.0, 4.0), (4.0, 8.0)),
        min_cycles=2.0, min_samples=16, max_gap_samples=0,
        acf_harmonic_tolerance=0.1, spectral_neighbourhood_hz=0.5,
        detrend="constant",
    ):
        self.motion_stats = motion_stats
        self.periodicity = periodicity
        self.spectral_stats = spectral_stats
        self.quality = quality
        self.signal = signal
        self.band = band
        self.bands = bands
        self.min_cycles = min_cycles
        self.min_samples = min_samples
        self.max_gap_samples = max_gap_samples
        self.acf_harmonic_tolerance = acf_harmonic_tolerance
        self.spectral_neighbourhood_hz = spectral_neighbourhood_hz
        self.detrend = detrend

    def mode_names(self):
        motion = _modes(self.motion_stats, _MOTION, name="motion_stats", aliases={
            "basic": {"acc_ac_rms", "rotation_mean"}, "full": _MOTION,
        })
        spectral = _modes(self.spectral_stats, _SPECTRAL, name="spectral_stats", aliases={"all": _SPECTRAL})
        _modes(self.periodicity, {"acf", "acf_harmonic_checked"}, name="periodicity")
        _modes(self.quality, {"report", "features"}, name="quality")
        if self.periodicity is not None and "|" in self.periodicity:
            raise InvalidExtractorParams("noop__periodicity selects one method")
        if self.quality is not None and "|" in self.quality:
            raise InvalidExtractorParams("noop__quality selects one method")
        return motion, spectral

    def enabled(self):
        return any(getattr(self, name) is not None for name in _MODE_PARAMETERS)

    def validate(self, sampling_rate=None):
        motion, spectral = self.mode_names()
        if not self.enabled():
            return self
        signal_used = self.periodicity is not None or bool(spectral) or self.quality is not None
        if signal_used and self.signal not in {"raw", "linear", "angular_speed"}:
            raise InvalidExtractorParams("noop__signal must be raw, linear or angular_speed")
        for name, minimum in (("min_samples", 4), ("max_gap_samples", 0)):
            if f"noop__{name}" in inactive_noop_parameters(self):
                continue
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise InvalidExtractorParams(f"noop__{name} must be an integer >= {minimum}")
        if (self.periodicity is not None or spectral) and self.detrend not in {"constant", "linear"}:
            raise InvalidExtractorParams("noop__detrend must be constant or linear")
        if self.periodicity is not None or spectral:
            if not np.isfinite(self.min_cycles) or self.min_cycles <= 0:
                raise InvalidExtractorParams("noop__min_cycles must be finite and positive")
            self.frequency_ranges(sampling_rate)
        if self.periodicity == "acf_harmonic_checked":
            if not np.isfinite(self.acf_harmonic_tolerance) or not 0 <= self.acf_harmonic_tolerance <= 0.5:
                raise InvalidExtractorParams("noop__acf_harmonic_tolerance must be in [0, 0.5]")
        if "peak_neighbourhood_fraction" in spectral:
            if not np.isfinite(self.spectral_neighbourhood_hz) or self.spectral_neighbourhood_hz <= 0:
                raise InvalidExtractorParams("noop__spectral_neighbourhood_hz must be finite and positive")
        return self

    def frequency_ranges(self, sampling_rate=None):
        fs = None if sampling_rate is None else _positive_rate(sampling_rate, "NOOP sampling_rate")
        def bounds(value):
            value = _structured(value)
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                raise InvalidExtractorParams("NOOP bands must contain (low, high) bounds")
            lo, hi = map(float, value)
            if not np.isfinite([lo, hi]).all() or not 0 <= lo < hi or (fs is not None and hi > fs / 2):
                raise InvalidExtractorParams("NOOP bands must satisfy 0 <= low < high <= Nyquist")
            return lo, hi
        band = bounds(self.band)
        if band[0] <= 0 and self.periodicity is not None:
            raise InvalidExtractorParams("Autocorrelation requires a strictly positive lower frequency")
        _, spectral = self.mode_names()
        bands = tuple(bounds(b) for b in _structured(self.bands)) if "band_fractions" in spectral else ()
        return band, bands


def _options_from_settings(settings):
    if isinstance(settings, NoopOptions):
        return settings
    settings = settings or {}
    options = settings.get("noop")
    nested = {key[len("noop__"):]: value for key, value in settings.items() if key.startswith("noop__")}
    if options is None:
        options = NoopOptions()
    elif not isinstance(options, NoopOptions):
        raise InvalidExtractorParams("noop must be None or NoopOptions")
    if nested:
        options = NoopOptions(**options.get_params(deep=False)).set_params(**nested)
    return options


def validate_noop_settings(settings, sampling_rate=None):
    options = _options_from_settings(settings)
    if sampling_rate is None and isinstance(settings, dict):
        key = "rot_native_sampling_rate" if options.signal == "angular_speed" else "imu_native_sampling_rate"
        sampling_rate = settings.get(key, 20)
    return options.validate(sampling_rate)


def inactive_noop_parameters(settings):
    options = _options_from_settings(settings)
    motion, spectral = options.mode_names()
    active = set()
    if motion or options.periodicity is not None or spectral:
        active.add("min_samples")
    if options.periodicity is not None or spectral or options.quality is not None:
        active |= {"max_gap_samples", "signal"}
    if options.periodicity is not None or spectral:
        active |= {"signal", "band", "min_cycles", "detrend"}
    if options.periodicity == "acf_harmonic_checked":
        active.add("acf_harmonic_tolerance")
    if "peak_neighbourhood_fraction" in spectral:
        active.add("spectral_neighbourhood_hz")
    if "band_fractions" in spectral:
        active.add("bands")
    return {f"noop__{name}" for name in _TUNING_PARAMETERS if name not in active}


def _runs(valid, counter):
    """Contiguous positions; neither missing rows nor counter gaps are joined."""
    positions = np.flatnonzero(valid)
    if not len(positions):
        return []
    cuts = np.flatnonzero((np.diff(positions) != 1) | (np.diff(counter[positions]) != 1)) + 1
    return np.split(positions, cuts)


def _bounded_fill(values, observed, counter, max_gap):
    result = np.array(values, dtype=float, copy=True)
    valid = np.asarray(observed, dtype=bool).copy()
    filled = np.zeros(len(valid), dtype=bool)
    if max_gap:
        for gap in _runs(~valid, np.arange(len(valid))):
            first, last = int(gap[0]), int(gap[-1])
            if (len(gap) > max_gap or first == 0 or last == len(valid) - 1
                    or not valid[first - 1] or not valid[last + 1]
                    or np.any(np.diff(counter[first - 1:last + 2]) != 1)):
                continue
            fraction = np.arange(1, len(gap) + 1) / (len(gap) + 1)
            result[gap] = (result[first - 1][None, :] * (1 - fraction[:, None])
                           + result[last + 1][None, :] * fraction[:, None])
            valid[gap] = True
            filled[gap] = True
    return result, valid, filled


def _detrended(values, mode):
    values = np.asarray(values, dtype=float)
    centered = values - np.mean(values, axis=0)
    if mode == "linear" and len(values) > 1:
        t = np.arange(len(values), dtype=float)
        t -= t.mean()
        centered -= t[:, None] * (t @ centered / (t @ t))[None, :]
    return centered


def _acf_estimate(values, fs, options):
    x = _detrended(values, options.detrend)
    axis = int(np.argmax(np.sum(x * x, axis=0)))
    x = x[:, axis]
    energy = float(x @ x)
    if energy <= np.finfo(float).eps * max(1, len(x)):
        return 0.0, 0.0, False
    (low, high), _ = options.frequency_ranges(fs)
    first = max(1, int(np.ceil(fs / high)))
    last = min(len(x) - 2, int(np.floor(fs / low)), int(np.floor(len(x) / options.min_cycles)))
    if last < first:
        return 0.0, 0.0, False
    corr = np.correlate(x, x, mode="full")[len(x) - 1:] / energy
    peaks = [lag for lag in range(first, last + 1)
             if corr[lag] > corr[lag - 1] and corr[lag] >= corr[lag + 1] and corr[lag] > 0]
    if not peaks:
        return 0.0, 0.0, False
    lag = max(peaks, key=lambda pos: corr[pos])
    return fs / lag, float(np.clip(corr[lag], 0, 1)), True


def _window_psd(values, fs, detrend):
    """One-sided density with the same power convention as CMI's STFT helper."""
    x = _detrended(values, detrend)
    window = np.hanning(len(x) + 1)[:-1]  # periodic Hann, as SciPy's fftbins=True
    coefficients = np.fft.rfft(x * window[:, None], axis=0) / np.sqrt(fs * np.sum(window ** 2))
    density = _one_sided_psd(coefficients, len(x)).sum(axis=1)
    return np.fft.rfftfreq(len(x), d=1 / fs), density


class NoopSequenceExtractor(SequenceExtractor):
    """Legacy extractor plus explicitly enabled, separately returned summaries.

    All parent parameters remain sklearn parameters. The parent chunk builder
    also carries temporary exact row tokens, so crop/tail/cap choices cannot
    drift between the temporal tensor and these summaries.
    """

    def __init__(self, noop=None, **kwargs):
        self.noop = noop
        super().__init__(**kwargs)

    @classmethod
    def _get_param_names(cls):
        return sorted(set(SequenceExtractor._get_param_names()) | {"noop"})

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        if deep and self.noop is None:
            params.update({f"noop__{name}": value for name, value in NoopOptions().get_params(deep=True).items()})
        return params

    def set_params(self, **params):
        if any(name.startswith("noop__") for name in params) and params.get("noop", self.noop) is None:
            params = dict(params)
            params["noop"] = NoopOptions()
        return super().set_params(**params)

    def _noop_options(self):
        if self.noop is None:
            return None
        if not isinstance(self.noop, NoopOptions):
            raise InvalidExtractorParams("noop must be None or NoopOptions")
        return self.noop if self.noop.enabled() else None

    @contextmanager
    def _noop_input(self, X):
        options = self._noop_options()
        if options is None:
            yield
            return
        options.validate(self.rot_native_sampling_rate if options.signal == "angular_speed" else self.imu_native_sampling_rate)
        if str(self.output_format).lower() != "chunks":
            raise InvalidExtractorParams("Enabled NOOP summaries require output_format='chunks'")
        self._noop_original_input = X
        try:
            yield
        finally:
            self.__dict__.pop("_noop_original_input", None)
            self.__dict__.pop("_noop_source", None)

    def fit(self, X, y=None):
        with self._noop_input(X):
            return super().fit(X, y)

    def fit_transform(self, X, y=None, **fit_params):
        with self._noop_input(X):
            return super().fit_transform(X, y, **fit_params)

    def transform_chunks(self, X):
        with self._noop_input(X):
            return super().transform_chunks(X)

    def _combine_feature_outputs(self, processed, add_context=False):
        combined = super()._combine_feature_outputs(processed, add_context=add_context)
        if self._noop_options() is not None:
            self._noop_source = self._source_view(processed, combined)
        return combined

    def _source_view(self, processed, combined):
        original = self._noop_original_input
        n = len(processed)
        source = {
            "raw": np.full((n, 3), np.nan), "linear": np.full((n, 3), np.nan),
            "rotation": np.full((n, 3), np.nan),
            "observed": np.zeros((n, 3), dtype=bool),
            "rotation_observed": np.zeros(n, dtype=bool),
            "counter": np.zeros(n, dtype=float),
            "dt": processed["dt"].to_numpy(dtype=float),
            "rotation_augmented": False,
        }
        for axis, name in enumerate(("acc_x", "acc_y", "acc_z")):
            if name in processed:
                source["raw"][:, axis] = processed[name].to_numpy(dtype=float)
            if "lin_" + name in processed:
                source["linear"][:, axis] = processed["lin_" + name].to_numpy(dtype=float)
        raw_groups = original.groupby(self.sequence_col, sort=False).indices
        for seq_id, positions in processed.groupby(self.sequence_col, sort=False).indices.items():
            raw_positions = raw_groups[seq_id]
            if len(positions) != len(raw_positions):
                raise InvalidExtractorParams("NOOP source alignment requires the parent's fixed-length source grid")
            raw = original.iloc[raw_positions]
            source["counter"][positions] = (
                pd.to_numeric(raw[self.counter_col], errors="coerce").to_numpy(dtype=float)
                if self.counter_col in raw else np.arange(len(raw), dtype=float)
            )
            for axis, name in enumerate(("acc_x", "acc_y", "acc_z")):
                if name in raw:
                    source["observed"][positions, axis] = np.isfinite(pd.to_numeric(raw[name], errors="coerce"))
            qcols = ["rot_w", "rot_x", "rot_y", "rot_z"]
            if all(name in raw for name in qcols):
                q = raw[qcols].to_numpy(dtype=float)
                valid_q = np.isfinite(q).all(axis=1) & (np.linalg.norm(np.nan_to_num(q), axis=1) > 0)
                source["rotation_observed"][positions] = valid_q
        rcols = [f"rot_angular_velocity_body_{axis}_rad_s" for axis in "xyz"]
        if all(name in combined for name in rcols):
            source["rotation"] = combined[rcols].to_numpy(dtype=float, copy=True)
        elif all(name in processed for name in ("rot_w", "rot_x", "rot_y", "rot_z")):
            safe = processed[[self.sequence_col, "rot_w", "rot_x", "rot_y", "rot_z", "rot_dt"]].copy()
            q = safe[["rot_w", "rot_x", "rot_y", "rot_z"]].to_numpy(dtype=float)
            valid = np.isfinite(q).all(axis=1) & (np.linalg.norm(np.nan_to_num(q), axis=1) > 0)
            q[~valid] = [1, 0, 0, 0]
            safe[["rot_w", "rot_x", "rot_y", "rot_z"]] = q
            source["rotation"] = RotationExtractor(rotation_modes="angular_velocity", sequence_col=self.sequence_col).fit_transform(safe)[rcols].to_numpy(dtype=float)
        return source

    def _apply_feature_augmentation(self, combined):
        result = super()._apply_feature_augmentation(combined)
        source = getattr(self, "_noop_source", None)
        augmentor = getattr(self, "_training_feature_augmentor", None)
        if source is None or augmentor is None or augmentor.prob <= 0 or augmentor.per_aug_prob <= 0:
            return result
        # Use exactly the augmented model channels. If the parent did not expose
        # a requested source, do not silently supply an unaugmented counterpart.
        for kind, names in (
            ("raw", [f"acc_{axis}_raw" for axis in "xyz"]),
            ("linear", [f"lin_acc_{axis}_linear" for axis in "xyz"]),
            ("rotation", [f"rot_angular_velocity_body_{axis}_rad_s" for axis in "xyz"]),
        ):
            for axis, name in enumerate(names):
                source[kind][:, axis] = result[name].to_numpy(dtype=float) if name in result else np.nan
        source["rotation_augmented"] = True
        return result

    def _chunks_from_features(self, out):
        options = self._noop_options()
        if options is None:
            return super()._chunks_from_features(out)
        source = self._noop_source
        # Two float32-exact tokens encode row positions without a 2**24 limit.
        names = list(self.feature_names_in_)
        token_names = ["__noop_row_hi", "__noop_row_lo"]
        if any(name in out for name in token_names):
            raise InvalidExtractorParams("Reserved NOOP row-token columns are already present")
        indexed = out.copy(deep=False)
        positions = np.arange(len(out), dtype=np.int64)
        indexed[token_names[0]] = positions // 65536
        indexed[token_names[1]] = positions % 65536
        self.feature_names_in_ = names + token_names
        try:
            chunks = super()._chunks_from_features(indexed)
        finally:
            self.feature_names_in_ = names
        row_tokens = chunks["X"][:, :, -2:]
        row_ids = row_tokens[:, :, 0].astype(np.int64) * 65536 + row_tokens[:, :, 1].astype(np.int64)
        # Keep a view: copying the complete temporal tensor would double its
        # peak resident allocation merely to remove two private token channels.
        chunks["X"] = chunks["X"][:, :, :-2]
        chunks["feature_names"] = names
        rows, reports, feature_names = [], [], None
        for idx, mask in enumerate(chunks["mask"]):
            selected = row_ids[idx, mask]
            features, report = self._summarize_window(source, selected, options)
            if feature_names is None:
                feature_names = list(features)
            elif feature_names != list(features):
                raise AssertionError("NOOP summary schema changed between windows")
            rows.append(list(features.values()))
            report["sequence_id"] = str(chunks["sequence_ids"][idx])
            reports.append(report)
        self.noop_feature_names_ = feature_names or []
        chunks["noop_features"] = np.asarray(rows, dtype=np.float32).reshape(len(rows), len(self.noop_feature_names_))
        chunks["noop_feature_names"] = self.noop_feature_names_.copy()
        if options.quality is not None:
            chunks["noop_quality"] = reports
        return chunks

    def _summarize_window(self, source, indices, options):
        motion, spectral = options.mode_names()
        min_samples = options.min_samples if motion or options.periodicity is not None or spectral else 16
        max_gap = options.max_gap_samples if options.periodicity is not None or spectral or options.quality is not None else 0
        fs = _positive_rate(self.rot_native_sampling_rate if options.signal == "angular_speed" else self.imu_native_sampling_rate,
                            "NOOP sampling_rate")
        counter = source["counter"][indices]
        raw = source["raw"][indices]
        observed_axes = source["observed"][indices]
        raw_observed = observed_axes.all(axis=1) & np.isfinite(raw).all(axis=1)
        q_observed = source["rotation_observed"][indices]
        contiguous = np.r_[False, np.diff(counter) == 1]
        rotation = source["rotation"][indices]
        rotation_valid = (q_observed & np.r_[False, q_observed[:-1]] & contiguous
                          & np.isfinite(rotation).all(axis=1))
        result = {}
        def add(name, value, supported):
            result[f"noop_{name}"] = float(value) if supported and np.isfinite(value) else 0.0
            result[f"noop_{name}_supported"] = float(bool(supported and np.isfinite(value)))
        enough = lambda valid: int(np.sum(valid)) >= min_samples
        if "acc_ac_rms" in motion:
            magnitude = np.linalg.norm(raw[raw_observed], axis=1)
            add("acc_ac_rms", np.std(magnitude) if len(magnitude) else 0, enough(raw_observed))
        if "axis_rms" in motion:
            for axis, name in enumerate("xyz"):
                valid = observed_axes[:, axis] & np.isfinite(raw[:, axis])
                add(f"acc_{name}_rms", np.sqrt(np.mean(raw[valid, axis] ** 2)) if valid.any() else 0, enough(valid))
        if "jerk_rms" in motion:
            valid = raw_observed[1:] & raw_observed[:-1] & contiguous[1:]
            dt = source["dt"][indices][1:]
            valid &= np.isfinite(dt) & (dt > 0)
            jerk = np.diff(raw, axis=0)[valid] / dt[valid, None]
            add("jerk_rms", np.sqrt(np.mean(np.sum(jerk ** 2, axis=1))) if len(jerk) else 0, int(valid.sum()) >= min_samples - 1)
        speed = np.linalg.norm(rotation[rotation_valid], axis=1)
        if "rotation_mean" in motion:
            add("rotation_mean_rad_s", np.mean(speed) if len(speed) else 0, enough(rotation_valid))
        if "rotation_rms" in motion:
            add("rotation_rms_rad_s", np.sqrt(np.mean(speed ** 2)) if len(speed) else 0, enough(rotation_valid))
        if options.signal == "raw":
            signal, observed = raw, raw_observed
        elif options.signal == "linear":
            signal = source["linear"][indices]
            observed = raw_observed & q_observed & np.isfinite(signal).all(axis=1)
        else:
            signal = np.linalg.norm(rotation, axis=1)[:, None]
            observed = rotation_valid
        signal, valid, imputed = _bounded_fill(signal, observed, counter, max_gap)
        runs = _runs(valid, counter)
        longest = max(runs, key=len) if runs else np.empty(0, dtype=int)
        selected = signal[longest]
        usable = len(selected) >= min_samples
        if options.periodicity is not None:
            frequency, strength, supported = _acf_estimate(selected, fs, options) if usable else (0, 0, False)
            add("acf_frequency_hz", frequency, supported)
            add("acf_strength", strength, supported)
            if options.periodicity == "acf_harmonic_checked":
                # A transverse sinusoid can become a double-frequency magnitude.
                # Flag agreement with that ambiguity; never guess a half-period.
                checked = supported and selected.shape[1] == 3
                magnitude_frequency, _, magnitude_supported = (
                    _acf_estimate(np.linalg.norm(selected, axis=1)[:, None], fs, options)
                    if checked else (0, 0, False)
                )
                ambiguous = (magnitude_supported and abs(magnitude_frequency / frequency - 2) <= 2 * options.acf_harmonic_tolerance)
                add("acf_magnitude_harmonic", float(ambiguous), checked and magnitude_supported)
        if spectral:
            band, bands = options.frequency_ranges(fs)
            frequencies, density = _window_psd(selected, fs, options.detrend) if usable else (np.empty(0), np.empty(0))
            resolved_low = max(band[0], options.min_cycles * fs / max(1, len(selected)))
            in_band = (frequencies >= resolved_low) & (frequencies <= band[1])
            power = density[in_band]
            total = float(power.sum())
            supported = usable and len(power) >= 2 and total > np.finfo(float).tiny
            peak = int(np.argmax(power)) if supported else 0
            peak_frequency = float(frequencies[in_band][peak]) if supported else 0
            if "peak_fraction" in spectral:
                add("spectral_peak_fraction", power[peak] / total if supported else 0, supported)
            if "peak_neighbourhood_fraction" in spectral:
                near = in_band & (np.abs(frequencies - peak_frequency) <= options.spectral_neighbourhood_hz)
                add("spectral_peak_neighbourhood_fraction", density[near].sum() / total if supported else 0, supported)
            if "band_fractions" in spectral:
                for idx, (low, high) in enumerate(bands):
                    inside = in_band & (frequencies >= low) & (frequencies < high)
                    add(f"spectral_band_{idx}_fraction", density[inside].sum() / total if supported else 0, supported and inside.any())
            if "entropy" in spectral:
                p = power[power > 0] / total if supported else np.empty(0)
                entropy = -np.sum(p * np.log2(p)) / np.log2(len(power)) if supported else 0
                add("spectral_entropy_normalized", entropy, supported)
        gaps = np.diff(counter) != 1
        report = {
            "real_samples": int(len(indices)), "observed_samples": int(observed.sum()),
            "observed_fraction": float(observed.mean()) if len(indices) else 0.0,
            "imputed_samples": int(imputed.sum()), "longest_contiguous_samples": int(len(longest)),
            "counter_gap_count": int(gaps.sum()),
            "start_counter": float(counter[0]) if len(counter) else 0.0,
            "end_counter": float(counter[-1]) if len(counter) else 0.0,
            "source": options.signal,
        }
        if options.quality == "features":
            add("observed_fraction", report["observed_fraction"], len(indices) > 0)
            add("imputed_fraction", imputed.sum() / max(1, len(indices)), len(indices) > 0)
            add("longest_contiguous_fraction", len(longest) / max(1, len(indices)), len(indices) > 0)
            add("counter_gap_fraction", gaps.sum() / max(1, len(indices) - 1), len(indices) > 1)
        return result, report
