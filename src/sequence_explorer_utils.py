"""
sequence_explorer_utils.py

Interactive explorer for the CMI sensor data, built directly on the
SignalCleaner / MotionFilter / IMUExtractor / RotationExtractor /
SequenceExtractor pipeline in base_utils_qwen.py. All widget callbacks are
bound methods so notebooks stay function-free.

Tabs:
  1. Catalogue  - filter by subject/gesture/orientation/... and build up a
                  working set of sequence_ids one selection at a time
                  (single add, or add-all-filtered). Small-multiples view
                  of whatever is currently catalogued.
  2. Features   - before/after on one sequence, run through the real
                  SequenceExtractor with every constructor parameter exposed.
  3. Spectrum   - FFT / PSD / spectrogram of the raw -> velocity ->
                  displacement -> jerk chain, computed via
                  SignalCleaner -> MotionFilter -> IMUExtractor exactly as
                  SequenceExtractor.fit does it internally.
  4. Clusters   - frame-level features -> scale -> reduce -> cluster, over
                  either the catalogue built in tab 1 or the current filter
                  selection (subject / orientation / ...).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

from base_utils_qwen import (
    SequenceExtractor,
    SignalCleaner,
    MotionFilter,
    IMUExtractor,
)

try:
    from umap import UMAP
    _UMAP = True
except Exception:
    _UMAP = False

try:
    from hdbscan import HDBSCAN
    _HDBSCAN = True
except Exception:
    _HDBSCAN = False

ACC_MODE_OPTS = ["raw", "smoothed", "velocity", "displacement", "jerk"]
ROT_MODE_OPTS = ["quaternion", "euler", "delta_euler", "angular_velocity", "rot6d"]
TOF_MODE_OPTS = ["raw", "sensor_stats", "pooled_stats", "pooled"]
THM_MODE_OPTS = ["raw", "centered", "diff", "centered_diff"]
FRAME_STAT_OPTS = ["mean", "std", "min", "max", "first", "last", "median", "rms", "abs_mean"]
IMU_DOMAIN_CHAIN = ["raw", "velocity", "displacement", "jerk"]


def _center_zero(ax, *arrays, pad=1.1):
    """Symmetric y-limits about 0 and a dashed reference line, so drift and
    offset are visible at a glance."""
    vals = [np.nanmax(np.abs(a)) for a in arrays if np.asarray(a).size]
    m = max(vals) if vals else 1.0
    if not np.isfinite(m) or m <= 0:
        m = 1.0
    ax.set_ylim(-m * pad, m * pad)
    ax.axhline(0.0, color="black", lw=0.7, alpha=0.5, ls="--")


class SequenceExplorer:
    """Row-level sensor data in, interactive panels out."""

    def __init__(self, df, label_col="gesture", fs=20.0,
                 sequence_col="sequence_id", counter_col="sequence_counter",
                 random_state=42):
        self.sequence_col = sequence_col
        self.counter_col = counter_col
        self.label_col = label_col
        self.fs = float(fs)
        self.random_state = random_state

        self.df = df.copy()
        if self.counter_col in self.df.columns:
            self.df = self.df.sort_values([sequence_col, counter_col], kind="stable")

        self.filter_cols_ = [c for c in ["subject", "sequence_type", "gesture",
                                         "orientation", "behavior", "phase"]
                             if c in self.df.columns]
        self.meta_ = (self.df[[sequence_col] + self.filter_cols_]
                      .groupby(sequence_col, sort=True).first())
        self.meta_["length"] = self.df.groupby(sequence_col, sort=True).size()
        self.raw_channels_ = [c for c in self.df.columns
                              if c.startswith(("acc_", "rot_", "thm_"))
                              and not c.startswith("tof_")]
        self.acc_axes_ = [c for c in ["acc_x", "acc_y", "acc_z"] if c in self.df.columns]

        self.catalogue_: list = []

    # ------------------------------------------------------------------
    # data access
    # ------------------------------------------------------------------
    def candidates(self, filters):
        m = pd.Series(True, index=self.meta_.index)
        for col, vals in filters.items():
            if col in self.meta_.columns and vals and "(all)" not in vals:
                m &= self.meta_[col].astype(str).isin([str(v) for v in vals])
        return list(self.meta_.index[m])

    def sequence(self, seq_id):
        return self.df[self.df[self.sequence_col] == seq_id]

    def _resolve(self, seq_id, seq_text):
        sid = (seq_text or "").strip() or seq_id
        if sid not in set(self.meta_.index):
            raise KeyError(f"{sid} not in data")
        return sid

    def _current_filters(self):
        return {c: list(f.value) for c, f in self.filters_.items()}

    # ------------------------------------------------------------------
    # SequenceExtractor - full parameter surface
    # ------------------------------------------------------------------
    def _extractor_kwargs(self, vals):
        """Translate raw widget values into exactly the kwargs
        SequenceExtractor.__init__ accepts, no more and no less."""
        def none_if_zero(x):
            return None if x is None or x <= 0 else float(x)

        def none_if_zero_int(x):
            return None if x is None or int(x) <= 0 else int(x)

        acc_modes = "|".join(vals["acc_modes"]) or "raw"
        rotation_modes = "|".join(vals["rotation_modes"]) or "quaternion"
        tof_modes = "|".join(vals["tof_modes"]) or "pooled_stats"
        thm_modes = "|".join(vals["thm_modes"]) or "centered_diff"
        frame_stats = ",".join(vals["frame_stats"]) or "mean"

        chunk_window = none_if_zero_int(vals["chunk_window_size"])
        chunk_stride = none_if_zero_int(vals["chunk_stride"])
        if chunk_window is not None and chunk_stride is not None:
            chunk_stride = min(chunk_stride, chunk_window)

        return dict(
            acc_modes=acc_modes,
            rotation_modes=rotation_modes,
            tof_modes=tof_modes,
            thm_modes=thm_modes,
            motion_filter_mode=None if vals["motion_filter_mode"] == "none" else vals["motion_filter_mode"],
            use_dead_reckoning=bool(vals["use_dead_reckoning"]),
            dead_reckoning_detrend=bool(vals["dead_reckoning_detrend"]),
            kalman_process_noise=10.0 ** vals["kalman_process_noise"],
            kalman_measurement_noise=10.0 ** vals["kalman_measurement_noise"],
            compute_dt=bool(vals["compute_dt"]),
            window_size=int(vals["window_size"]),
            smooth_alpha=none_if_zero(vals["smooth_alpha"]),
            clip_value=none_if_zero(vals["clip_value"]),
            interp_mode=vals["interp_mode"],
            maxlen=int(vals["maxlen"]),
            padding_value=float(vals["padding_value"]),
            imu_native_sampling_rate=int(vals["imu_native_sampling_rate"]),
            imu_target_sampling_rate=int(vals["imu_target_sampling_rate"]),
            rot_native_sampling_rate=int(vals["rot_native_sampling_rate"]),
            rot_target_sampling_rate=int(vals["rot_target_sampling_rate"]),
            tof_native_sampling_rate=int(vals["tof_native_sampling_rate"]),
            tof_target_sampling_rate=int(vals["tof_target_sampling_rate"]),
            thm_native_sampling_rate=int(vals["thm_native_sampling_rate"]),
            thm_target_sampling_rate=int(vals["thm_target_sampling_rate"]),
            chunk_window_size=chunk_window,
            chunk_stride=chunk_stride,
            frame_stats=frame_stats,
            add_global_context=bool(vals["add_global_context"]),
            resample_modalities=bool(vals["resample_modalities"]),
        )

    def extract(self, seq_df, extractor_kwargs, output_format="chunks"):
        """Row-level engineered features for one or many sequences, using the
        exact preprocessing path SequenceExtractor.fit/transform use."""
        ex = SequenceExtractor(output_format=output_format, **extractor_kwargs)
        ex.fit(seq_df)
        return ex, ex._preprocess_features(seq_df)

    def _imu_domain_chain(self, seq_df, axis, extractor_kwargs):
        """raw -> velocity -> displacement -> jerk for one accelerometer axis,
        via the literal SignalCleaner -> MotionFilter -> IMUExtractor chain
        SequenceExtractor.fit runs internally (cleaning/filter params only;
        acc_modes is overridden per domain)."""
        cleaner = SignalCleaner(
            native_sampling_rate=extractor_kwargs["imu_native_sampling_rate"],
            compute_dt=extractor_kwargs["compute_dt"],
            clip_value=extractor_kwargs["clip_value"],
            interp_mode=extractor_kwargs["interp_mode"],
            window_size=extractor_kwargs["window_size"],
        )
        cleaned = cleaner.fit_transform(seq_df)

        mf = MotionFilter(
            motion_filter_mode=extractor_kwargs["motion_filter_mode"],
            kalman_process_noise=extractor_kwargs["kalman_process_noise"],
            kalman_measurement_noise=extractor_kwargs["kalman_measurement_noise"],
            use_dead_reckoning=extractor_kwargs["use_dead_reckoning"],
            dead_reckoning_detrend=extractor_kwargs["dead_reckoning_detrend"],
        )
        filtered = mf.transform(cleaned)

        domains = {}
        for mode in IMU_DOMAIN_CHAIN:
            imu = IMUExtractor(acc_modes=mode,
                               window_size=extractor_kwargs["window_size"],
                               smooth_alpha=extractor_kwargs["smooth_alpha"])
            imu.fit(filtered)
            out = imu.transform(filtered)
            col = next((c for c in out.columns if c.startswith(axis + "_")), None)
            domains[mode] = out[col].to_numpy(dtype=float) if col is not None \
                else np.zeros(len(filtered))
        return domains, filtered["dt"].to_numpy(dtype=float)

    def _spectrum(self, x, mode, detrend):
        x = np.nan_to_num(np.asarray(x, dtype=float))
        if detrend and len(x) > 1:
            x = signal.detrend(x, type="linear")
        nper = min(256, max(8, len(x)))
        if mode == "psd":
            return signal.welch(x, fs=self.fs, nperseg=nper, window="hann")
        win = np.hanning(len(x))
        X = np.fft.rfft(x * win)
        f = np.fft.rfftfreq(len(x), d=1.0 / self.fs)
        return f, (2.0 / max(win.sum(), 1e-12)) * np.abs(X)

    # ------------------------------------------------------------------
    # controls
    # ------------------------------------------------------------------
    def make_controls(self):
        import ipywidgets as w

        self.filters_ = {
            c: w.SelectMultiple(options=["(all)"] + sorted(self.meta_[c].astype(str).unique()),
                                value=("(all)",), description=c[:11] + ":", rows=6,
                                layout=w.Layout(width="230px"))
            for c in self.filter_cols_
        }
        for f in self.filters_.values():
            f.observe(self._on_filter_change, names="value")

        seqs = self.candidates({})
        self.seq_pick_ = w.Dropdown(options=seqs, value=seqs[0], description="Sequence:",
                                    layout=w.Layout(width="330px"))
        self.seq_text_ = w.Text(value="", description="or ID:", placeholder="SEQ_000007",
                                layout=w.Layout(width="330px"))

        # ---- catalogue controls ----
        self.add_one_btn_ = w.Button(description="+ Add current", button_style="success")
        self.add_filtered_btn_ = w.Button(description="+ Add all filtered", button_style="info")
        self.remove_one_btn_ = w.Button(description="- Remove current", button_style="warning")
        self.clear_btn_ = w.Button(description="Clear catalogue", button_style="danger")
        self.cat_label_ = w.HTML(value="<b>Catalogue: 0 sequences</b>")
        self.cat_version_ = w.IntText(value=0)
        self.cat_version_.layout.display = "none"
        self.add_one_btn_.on_click(self._add_current)
        self.add_filtered_btn_.on_click(self._add_filtered)
        self.remove_one_btn_.on_click(self._remove_current)
        self.clear_btn_.on_click(self._clear_catalogue)

        # ---- full SequenceExtractor parameter surface ----
        p = {
            "acc_modes": w.SelectMultiple(options=ACC_MODE_OPTS, value=("raw", "velocity", "jerk"),
                                          description="acc_modes:", rows=5),
            "rotation_modes": w.SelectMultiple(options=ROT_MODE_OPTS,
                                               value=("quaternion", "angular_velocity"),
                                               description="rotation_modes:", rows=5),
            "tof_modes": w.SelectMultiple(options=TOF_MODE_OPTS, value=("pooled_stats",),
                                          description="tof_modes:", rows=4),
            "thm_modes": w.SelectMultiple(options=THM_MODE_OPTS, value=("centered_diff",),
                                          description="thm_modes:", rows=4),

            "motion_filter_mode": w.Dropdown(options=["none", "kalman", "extended_kalman"],
                                             value="none", description="motion_filter_mode:"),
            "use_dead_reckoning": w.Checkbox(value=False, description="use_dead_reckoning"),
            "dead_reckoning_detrend": w.Checkbox(value=False, description="dead_reckoning_detrend"),
            "kalman_process_noise": w.FloatSlider(min=-6, max=0, step=0.5, value=-3,
                                                   description="log10 kalman_process_noise:",
                                                   style={"description_width": "220px"}),
            "kalman_measurement_noise": w.FloatSlider(min=-6, max=0, step=0.5, value=-2,
                                                       description="log10 kalman_measurement_noise:",
                                                       style={"description_width": "220px"}),

            "compute_dt": w.Checkbox(value=True, description="compute_dt"),
            "window_size": w.IntSlider(min=3, max=31, step=2, value=7, description="window_size:"),
            "smooth_alpha": w.FloatSlider(min=0.0, max=1.0, step=0.05, value=0.0,
                                          description="smooth_alpha (0=None):",
                                          style={"description_width": "160px"}),
            "clip_value": w.FloatSlider(min=0.0, max=500.0, step=10.0, value=0.0,
                                        description="clip_value (0=None):",
                                        style={"description_width": "160px"}),
            "interp_mode": w.Dropdown(options=["linear", "ffill"], value="linear",
                                      description="interp_mode:"),
            "maxlen": w.IntSlider(min=50, max=500, step=10, value=160, description="maxlen:"),
            "padding_value": w.FloatText(value=0.0, description="padding_value:",
                                         style={"description_width": "120px"}),

            "imu_native_sampling_rate": w.IntSlider(min=1, max=100, value=int(self.fs),
                                                     description="imu_native_hz:"),
            "imu_target_sampling_rate": w.IntSlider(min=1, max=100, value=int(self.fs),
                                                     description="imu_target_hz:"),
            "rot_native_sampling_rate": w.IntSlider(min=1, max=100, value=int(self.fs),
                                                     description="rot_native_hz:"),
            "rot_target_sampling_rate": w.IntSlider(min=1, max=100, value=int(self.fs),
                                                     description="rot_target_hz:"),
            "tof_native_sampling_rate": w.IntSlider(min=1, max=100, value=5,
                                                     description="tof_native_hz:"),
            "tof_target_sampling_rate": w.IntSlider(min=1, max=100, value=5,
                                                     description="tof_target_hz:"),
            "thm_native_sampling_rate": w.IntSlider(min=1, max=100, value=5,
                                                     description="thm_native_hz:"),
            "thm_target_sampling_rate": w.IntSlider(min=1, max=100, value=5,
                                                     description="thm_target_hz:"),
            "resample_modalities": w.Checkbox(value=False, description="resample_modalities"),

            "chunk_window_size": w.IntSlider(min=0, max=400, step=8, value=0,
                                             description="chunk_window_size (0=None):",
                                             style={"description_width": "200px"}),
            "chunk_stride": w.IntSlider(min=0, max=400, step=8, value=0,
                                        description="chunk_stride (0=None):",
                                        style={"description_width": "200px"}),
            "frame_stats": w.SelectMultiple(options=FRAME_STAT_OPTS,
                                            value=("mean", "std", "min", "max", "last"),
                                            description="frame_stats:", rows=9),
            "add_global_context": w.Checkbox(value=False, description="add_global_context"),
        }
        self.p = p

        # extra, per-tab widgets
        self.feature_options_ = self._all_feature_names()
        self.feature_ = w.Dropdown(options=self.feature_options_, value=self.feature_options_[0],
                                   description="Feature:")
        self.cat_channel_ = w.Dropdown(options=self.raw_channels_, value=self.raw_channels_[0],
                                       description="Channel:")
        self.axis_ = w.Dropdown(options=self.acc_axes_, value=self.acc_axes_[0],
                                description="IMU axis:")
        self.spec_mode_ = w.Dropdown(options=["fft", "psd", "spectrogram"], value="fft",
                                     description="Spectral:")
        self.detrend_ = w.Checkbox(value=True, description="detrend")
        self.logy_ = w.Checkbox(value=True, description="log power")

        self.reduction_ = w.Dropdown(options=["pca", "tsne"] + (["umap"] if _UMAP else []),
                                     value="pca", description="Reduce:")
        self.method_ = w.Dropdown(options=["kmeans", "gmm", "agglomerative", "dbscan"]
                                  + (["hdbscan"] if _HDBSCAN else []),
                                  value="kmeans", description="Cluster:")
        self.n_clusters_ = w.IntSlider(min=2, max=12, step=1, value=5, description="k:")
        self.eps_ = w.FloatSlider(min=0.1, max=5.0, step=0.1, value=0.8, description="eps:")
        self.cluster_space_ = w.Dropdown(options=["features", "embedding"], value="features",
                                         description="Fit on:")
        self.population_ = w.Dropdown(options=["catalogue", "filtered"], value="filtered",
                                      description="Population:")
        self.max_seqs_ = w.IntSlider(min=20, max=2000, step=20, value=400, description="Max seqs:")
        self.color_by_ = w.Dropdown(options=["cluster"] + self.filter_cols_, value="cluster",
                                    description="Colour:")

    def _all_feature_names(self):
        """Union of engineered columns over the widest mode combination."""
        probe = self.sequence(self.meta_.index[0])
        widest = self._extractor_kwargs(dict(
            acc_modes=ACC_MODE_OPTS, rotation_modes=ROT_MODE_OPTS, tof_modes=("pooled_stats",),
            thm_modes=("centered_diff",), motion_filter_mode="none", use_dead_reckoning=False,
            dead_reckoning_detrend=False, kalman_process_noise=-3, kalman_measurement_noise=-2,
            compute_dt=True, window_size=7, smooth_alpha=0.0, clip_value=0.0,
            interp_mode="linear", maxlen=160, padding_value=0.0,
            imu_native_sampling_rate=int(self.fs), imu_target_sampling_rate=int(self.fs),
            rot_native_sampling_rate=int(self.fs), rot_target_sampling_rate=int(self.fs),
            tof_native_sampling_rate=5, tof_target_sampling_rate=5,
            thm_native_sampling_rate=5, thm_target_sampling_rate=5,
            chunk_window_size=0, chunk_stride=0, frame_stats=("mean",),
            add_global_context=False, resample_modalities=False,
        ))
        _, out = self.extract(probe, widest)
        cols = [c for c in out.columns if c != self.sequence_col]
        return cols + [c for c in self.raw_channels_ if c not in cols]

    # ------------------------------------------------------------------
    # catalogue handlers
    # ------------------------------------------------------------------
    def _on_filter_change(self, _):
        seqs = self.candidates(self._current_filters())
        if not seqs:
            return
        self.seq_pick_.options = seqs
        self.seq_pick_.value = seqs[0]

    def _bump(self):
        n = len(self.catalogue_)
        preview = ", ".join(self.catalogue_[-8:])
        tail = f" — last added: {preview}" if n else " (empty — Clusters tab falls back to Population=filtered)"
        self.cat_label_.value = f"<b>Catalogue: {n} sequence(s)</b>{tail}"
        self.cat_version_.value += 1

    def _add_current(self, _btn):
        sid = self._resolve(self.seq_pick_.value, self.seq_text_.value)
        if sid not in self.catalogue_:
            self.catalogue_.append(sid)
        self._bump()

    def _add_filtered(self, _btn):
        for sid in self.candidates(self._current_filters()):
            if sid not in self.catalogue_:
                self.catalogue_.append(sid)
        self._bump()

    def _remove_current(self, _btn):
        sid = self._resolve(self.seq_pick_.value, self.seq_text_.value)
        if sid in self.catalogue_:
            self.catalogue_.remove(sid)
        self._bump()

    def _clear_catalogue(self, _btn):
        self.catalogue_.clear()
        self._bump()

    # ------------------------------------------------------------------
    # panel 1: catalogue
    # ------------------------------------------------------------------
    def update_catalogue(self, cat_version, channel):
        ids = list(self.catalogue_)
        if not ids:
            print("catalogue is empty — use '+ Add current' or '+ Add all filtered' above")
            return
        show = ids[:24]
        ncols = 4
        nrows = int(np.ceil(len(show) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 1.9 * nrows), squeeze=False)
        for ax, sid in zip(axes.ravel(), show):
            g = self.sequence(sid)
            y = g[channel].to_numpy(dtype=float) if channel in g.columns else np.zeros(len(g))
            ax.plot(np.arange(len(g)) / self.fs, y, lw=0.8, color="steelblue")
            _center_zero(ax, y)
            lab = self.meta_.loc[sid, self.label_col] if self.label_col in self.meta_.columns else ""
            ax.set_title(f"{sid}\n{lab}", fontsize=7)
            ax.tick_params(labelsize=6)
        for ax in axes.ravel()[len(show):]:
            ax.axis("off")
        fig.suptitle(f"Catalogue ({len(ids)} sequences, showing first {len(show)}) — {channel}",
                     fontsize=10)
        fig.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # panel 2: feature extraction before/after (full SequenceExtractor)
    # ------------------------------------------------------------------
    def update_features(self, seq_id, seq_text, feature, **pvals):
        sid = self._resolve(seq_id, seq_text)
        g = self.sequence(sid)
        kwargs = self._extractor_kwargs(pvals)
        ex, out = self.extract(g, kwargs, output_format="chunks")
        cols = [c for c in out.columns if c != self.sequence_col]
        if feature not in cols:
            print(f"'{feature}' not produced by this parameter set — showing {cols[0]}")
            feature = cols[0]

        base = "_".join(feature.split("_")[:2])
        raw_ch = base if base in g.columns else None
        t = np.arange(len(g)) / self.fs

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        if raw_ch is not None:
            y0 = g[raw_ch].to_numpy(dtype=float)
            axes[0].plot(t, y0, lw=0.9, color="slategray")
            _center_zero(axes[0], y0)
            axes[0].set_title(f"before: {raw_ch}", fontsize=10)
        else:
            axes[0].text(0.5, 0.5, "no matching raw channel", ha="center", transform=axes[0].transAxes)
        y1 = out[feature].to_numpy(dtype=float)
        axes[1].plot(t[:len(out)], y1, lw=0.9, color="seagreen")
        _center_zero(axes[1], y1)
        axes[1].set_title(f"after: {feature}", fontsize=10)
        for ax in axes:
            ax.grid(alpha=0.3)
        axes[1].set_xlabel("time (s)")
        fig.suptitle(f"{sid} | acc={kwargs['acc_modes']} | rot={kwargs['rotation_modes']} | "
                     f"filter={kwargs['motion_filter_mode']} | dr={kwargs['use_dead_reckoning']}",
                     fontsize=9)
        fig.tight_layout()
        plt.show()

        stats = pd.DataFrame({feature: [y1.mean(), y1.std(), y1.min(), y1.max()]},
                             index=["mean", "std", "min", "max"]).T
        if raw_ch is not None:
            r = g[raw_ch].to_numpy(dtype=float)
            stats.loc[raw_ch] = [np.nanmean(r), np.nanstd(r), np.nanmin(r), np.nanmax(r)]
        print(f"{len(cols)} engineered columns | base_feature_names_: {len(ex.base_feature_names_)}")
        print(stats.round(4).to_string())

    # ------------------------------------------------------------------
    # panel 3: spectrum over the raw -> velocity -> displacement -> jerk chain
    # ------------------------------------------------------------------
    def update_spectrum(self, seq_id, seq_text, axis, spec_mode, detrend, logy, **pvals):
        sid = self._resolve(seq_id, seq_text)
        g = self.sequence(sid)
        kwargs = self._extractor_kwargs(pvals)
        domains, dt = self._imu_domain_chain(g, axis, kwargs)
        t = np.arange(len(g)) / self.fs

        if spec_mode == "spectrogram":
            fig, axes = plt.subplots(1, len(IMU_DOMAIN_CHAIN), figsize=(4 * len(IMU_DOMAIN_CHAIN), 4),
                                     sharey=True)
            for ax, mode in zip(axes, IMU_DOMAIN_CHAIN):
                x = np.nan_to_num(domains[mode])
                nper = min(64, max(8, len(x) // 4))
                f, tt, S = signal.spectrogram(x, fs=self.fs, nperseg=nper,
                                              noverlap=nper // 2, window="hann")
                ax.pcolormesh(tt, f, 10 * np.log10(S + 1e-12), shading="gouraud", cmap="magma")
                ax.set_title(mode, fontsize=10)
                ax.set_xlabel("time (s)")
            axes[0].set_ylabel("Hz")
            fig.suptitle(f"{sid} | {axis} | spectrogram over raw->velocity->displacement->jerk",
                         fontsize=10)
            fig.tight_layout()
            plt.show()
            return

        fig, axes = plt.subplots(2, len(IMU_DOMAIN_CHAIN), figsize=(4 * len(IMU_DOMAIN_CHAIN), 6.5))
        summary = []
        for i, mode in enumerate(IMU_DOMAIN_CHAIN):
            x = domains[mode]
            axes[0, i].plot(t[:len(x)], x, lw=0.8, color="darkorange")
            _center_zero(axes[0, i], x)
            axes[0, i].set_title(mode, fontsize=10)
            axes[0, i].grid(alpha=0.3)

            f, P = self._spectrum(x, spec_mode, detrend)
            axes[1, i].plot(f, P, lw=0.9, color="steelblue")
            if logy:
                axes[1, i].set_yscale("log")
            axes[1, i].set_xlabel("Hz")
            axes[1, i].grid(alpha=0.3)

            P = np.asarray(P, dtype=float)
            if P[1:].sum() > 0:
                centroid = float((f[1:] * P[1:]).sum() / P[1:].sum())
                peak = float(f[1:][np.argmax(P[1:])])
            else:
                centroid = peak = 0.0
            summary.append((mode, peak, centroid, float(P[1:].sum())))

        axes[0, 0].set_ylabel("amplitude")
        axes[1, 0].set_ylabel("PSD" if spec_mode == "psd" else "|X(f)|")
        fig.suptitle(f"{sid} | {axis} | native_hz={kwargs['imu_native_sampling_rate']} | "
                     f"filter={kwargs['motion_filter_mode']} | Nyquist={self.fs / 2:g} Hz", fontsize=10)
        fig.tight_layout()
        plt.show()

        print(pd.DataFrame(summary, columns=["mode", "peak_hz", "centroid_hz", "power"])
              .set_index("mode").round(4).to_string())

    # ------------------------------------------------------------------
    # panel 4: unsupervised
    # ------------------------------------------------------------------
    def update_cluster(self, reduction, method, n_clusters, eps, cluster_space,
                       population, max_seqs, color_by, cat_version, **pvals):
        if population == "catalogue":
            seqs = list(self.catalogue_)
            if not seqs:
                print("catalogue is empty — add sequences in the Catalogue tab, "
                      "or set Population='filtered'")
                return
        else:
            seqs = self.candidates(self._current_filters())
            if len(seqs) < 5:
                print("need at least 5 sequences after filtering")
                return

        rng = np.random.RandomState(self.random_state)
        if len(seqs) > max_seqs:
            seqs = list(rng.choice(seqs, size=int(max_seqs), replace=False))
        sub = self.df[self.df[self.sequence_col].isin(seqs)]

        kwargs = self._extractor_kwargs(pvals)
        ex = SequenceExtractor(output_format="frame", **kwargs)
        ex.fit(sub)
        F = ex.transform_frame(sub)
        M = StandardScaler().fit_transform(np.nan_to_num(F.to_numpy(dtype=float)))

        if reduction == "tsne":
            Z = TSNE(n_components=2, init="pca", learning_rate="auto",
                     perplexity=min(30, max(5, len(M) // 4)),
                     random_state=self.random_state).fit_transform(M)
        elif reduction == "umap" and _UMAP:
            Z = UMAP(n_components=2, random_state=self.random_state).fit_transform(M)
        else:
            Z = PCA(n_components=2, random_state=self.random_state).fit_transform(M)

        S = M if cluster_space == "features" else Z
        if method == "kmeans":
            labels = KMeans(n_clusters=n_clusters, n_init=10,
                            random_state=self.random_state).fit_predict(S)
        elif method == "gmm":
            labels = GaussianMixture(n_components=n_clusters, n_init=3,
                                     random_state=self.random_state).fit_predict(S)
        elif method == "agglomerative":
            labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(S)
        elif method == "dbscan":
            labels = DBSCAN(eps=float(eps), min_samples=5).fit_predict(S)
        else:
            labels = HDBSCAN(min_cluster_size=max(5, len(S) // 50)).fit_predict(S)

        n_lab = len(set(labels)) - (1 if -1 in labels else 0)
        sil = silhouette_score(S, labels) if 2 <= n_lab < len(S) else np.nan
        truth = self.meta_.loc[F.index, self.label_col].astype(str) \
            if self.label_col in self.meta_.columns else None
        ari = adjusted_rand_score(truth, labels) if truth is not None and n_lab >= 2 else np.nan

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        axes[0].scatter(Z[:, 0], Z[:, 1], c=labels, cmap="tab10", s=14, alpha=0.8)
        axes[0].set_title(f"{method} on {cluster_space} | pop={population} ({len(seqs)}) | "
                          f"{n_lab} clusters | silhouette={sil:.3f} | ARI={ari:.3f}", fontsize=9)
        key = color_by if color_by != "cluster" else self.label_col
        if key in self.meta_.columns:
            codes = self.meta_.loc[F.index, key].astype("category")
            axes[1].scatter(Z[:, 0], Z[:, 1], c=codes.cat.codes, cmap="tab20", s=14, alpha=0.8)
            axes[1].set_title(f"true {key} ({codes.nunique()} levels)", fontsize=9)
        for ax in axes:
            ax.set_xlabel(f"{reduction} 1")
            ax.set_ylabel(f"{reduction} 2")
            ax.grid(alpha=0.3)
        fig.tight_layout()
        plt.show()

        print(f"{len(F)} sequences | {F.shape[1]} frame features")
        if truth is not None:
            ct = pd.crosstab(pd.Series(labels, name="cluster"), truth.values)
            summary = pd.DataFrame({
                "n": ct.sum(axis=1),
                "dominant": ct.idxmax(axis=1),
                "purity": (ct.max(axis=1) / ct.sum(axis=1)).round(3),
            })
            print(summary.to_string())

    # ------------------------------------------------------------------
    # layout
    # ------------------------------------------------------------------
    def show(self):
        import ipywidgets as w
        from IPython.display import display

        self.make_controls()
        p = self.p

        filter_row = w.HBox(list(self.filters_.values()))
        picker_row = w.HBox([self.seq_pick_, self.seq_text_])
        catalogue_btns = w.HBox([self.add_one_btn_, self.add_filtered_btn_,
                                 self.remove_one_btn_, self.clear_btn_])
        top = w.VBox([filter_row, picker_row, catalogue_btns, self.cat_label_, self.cat_version_])

        param_accordion = w.Accordion(children=[
            w.HBox([p["acc_modes"], p["rotation_modes"], p["tof_modes"], p["thm_modes"]]),
            w.VBox([w.HBox([p["motion_filter_mode"], p["use_dead_reckoning"],
                            p["dead_reckoning_detrend"]]),
                    w.HBox([p["kalman_process_noise"], p["kalman_measurement_noise"]])]),
            w.VBox([w.HBox([p["compute_dt"], p["window_size"], p["interp_mode"]]),
                    w.HBox([p["smooth_alpha"], p["clip_value"]]),
                    w.HBox([p["maxlen"], p["padding_value"]])]),
            w.VBox([w.HBox([p["imu_native_sampling_rate"], p["imu_target_sampling_rate"]]),
                    w.HBox([p["rot_native_sampling_rate"], p["rot_target_sampling_rate"]]),
                    w.HBox([p["tof_native_sampling_rate"], p["tof_target_sampling_rate"]]),
                    w.HBox([p["thm_native_sampling_rate"], p["thm_target_sampling_rate"]]),
                    p["resample_modalities"]]),
            w.VBox([w.HBox([p["chunk_window_size"], p["chunk_stride"]]),
                    p["frame_stats"], p["add_global_context"]]),
        ])
        for i, t in enumerate(["Modes", "Motion filter / dead reckoning", "Cleaning",
                               "Sampling rates & resampling", "Chunking / frame stats"]):
            param_accordion.set_title(i, t)

        pvals_kw = {k: v for k, v in p.items()}

        o_cat = w.interactive_output(self.update_catalogue,
                                     {"cat_version": self.cat_version_, "channel": self.cat_channel_})
        o_feat = w.interactive_output(self.update_features,
                                      {"seq_id": self.seq_pick_, "seq_text": self.seq_text_,
                                       "feature": self.feature_, **pvals_kw})
        o_spec = w.interactive_output(self.update_spectrum,
                                      {"seq_id": self.seq_pick_, "seq_text": self.seq_text_,
                                       "axis": self.axis_, "spec_mode": self.spec_mode_,
                                       "detrend": self.detrend_, "logy": self.logy_, **pvals_kw})
        o_clus = w.interactive_output(self.update_cluster,
                                      {"reduction": self.reduction_, "method": self.method_,
                                       "n_clusters": self.n_clusters_, "eps": self.eps_,
                                       "cluster_space": self.cluster_space_,
                                       "population": self.population_, "max_seqs": self.max_seqs_,
                                       "color_by": self.color_by_, "cat_version": self.cat_version_,
                                       **pvals_kw})

        tabs = w.Tab(children=[
            w.VBox([self.cat_channel_, o_cat]),
            w.VBox([param_accordion, self.feature_, o_feat]),
            w.VBox([param_accordion, w.HBox([self.axis_, self.spec_mode_, self.detrend_, self.logy_]),
                    o_spec]),
            w.VBox([param_accordion,
                    w.HBox([self.population_, self.reduction_, self.method_, self.cluster_space_]),
                    w.HBox([self.n_clusters_, self.eps_, self.max_seqs_, self.color_by_]),
                    o_clus]),
        ])
        for i, t in enumerate(["Catalogue", "Features", "Spectrum", "Clusters"]):
            tabs.set_title(i, t)
        display(w.VBox([top, tabs]))
