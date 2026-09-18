"""
sequence_explorer_utils.py

Interactive explorer for the CMI sensor data. Four panels, all callbacks are
bound methods so notebooks stay function-free:

  1. Sequence   - filter by subject/gesture/orientation/type, or paste a
                  sequence_id, plot raw channels, build a sequence catalog.
  2. Features   - before/after for any SequenceExtractor parameter set.
  3. Spectrum   - Hann rFFT / Welch PSD / spectrogram of raw vs processed.
  4. Clusters   - frame-level features -> scale -> reduce -> cluster.

Depends on SequenceExtractor from base_utils_qwen.
"""
from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional, Sequence

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

from base_utils_qwen import SequenceExtractor

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

ACC_MODES = [
    "raw",
    "raw|velocity",
    "raw|velocity|jerk",
    "raw|velocity|displacement|jerk",
    "smoothed|velocity|jerk",
]
ROT_MODES = [
    "quaternion",
    "quaternion|euler",
    "quaternion|angular_velocity",
    "quaternion|euler|angular_velocity",
    "quaternion|delta_euler|angular_velocity",
]
TOF_MODES = ["pooled_stats", "sensor_stats", "pooled_stats|sensor_stats"]
THM_MODES = ["centered", "diff", "centered_diff"]
INTERP_MODES = ["linear", "ffill"]
FRAME_STATS = ["mean", "mean,std", "mean,std,min,max", "mean,std,min,max,last"]

_FEATURE_SUFFIXES = (
    "_centered_diff", "_centered", "_smooth", "_angvel", "_delta",
    "_quat", "_raw", "_vel", "_disp", "_jerk", "_dr_vel", "_dr_pos",
)


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
        self.catalog_: List[str] = []

        self.df = df.copy()
        if self.counter_col in self.df.columns:
            self.df = self.df.sort_values([sequence_col, counter_col], kind="stable")

        self.filter_cols_ = [
            c for c in ["subject", "sequence_type", "gesture",
                        "orientation", "behavior", "phase"]
            if c in self.df.columns
        ]
        self.meta_ = (
            self.df[[sequence_col] + self.filter_cols_]
            .groupby(sequence_col, sort=True)
            .first()
        )
        self.meta_["length"] = self.df.groupby(sequence_col, sort=True).size()
        self.raw_channels_ = [
            c for c in self.df.columns
            if c.startswith(("acc_", "rot_", "thm_"))
            and not c.startswith("tof_")
        ]
        self.extractor_param_names_ = list(
            inspect.signature(SequenceExtractor.__init__).parameters.keys()
        )
        self.extractor_param_names_.remove("self")

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

    def extract(self, seq_df, **params):
        """Row-level engineered features for one or many sequences."""
        params.setdefault("output_format", "chunks")
        ex = SequenceExtractor(**params)
        ex.fit(seq_df)
        return ex._preprocess_features(seq_df)

    def extractor_defaults(self) -> Dict[str, Any]:
        """Full default parameter dict from SequenceExtractor."""
        return SequenceExtractor().get_params()

    def _params(self, acc_modes, rotation_modes, tof_modes, thm_modes,
                motion_filter_mode, use_dead_reckoning, dead_reckoning_detrend,
                kalman_process_noise, kalman_measurement_noise,
                smooth_alpha, window_size, clip_value, interp_mode, compute_dt,
                imu_native_sampling_rate, imu_target_sampling_rate,
                rot_native_sampling_rate, rot_target_sampling_rate,
                tof_native_sampling_rate, tof_target_sampling_rate,
                thm_native_sampling_rate, thm_target_sampling_rate,
                resample_modalities, maxlen, add_global_context, frame_stats,
                **extra):
        p = dict(
            acc_modes=acc_modes,
            rotation_modes=rotation_modes,
            tof_modes=tof_modes,
            thm_modes=thm_modes,
            motion_filter_mode=None if motion_filter_mode == "none" else motion_filter_mode,
            use_dead_reckoning=bool(use_dead_reckoning),
            dead_reckoning_detrend=bool(dead_reckoning_detrend),
            kalman_process_noise=float(kalman_process_noise),
            kalman_measurement_noise=float(kalman_measurement_noise),
            smooth_alpha=None if smooth_alpha <= 0 else float(smooth_alpha),
            window_size=int(window_size),
            clip_value=None if clip_value <= 0 else float(clip_value),
            interp_mode=str(interp_mode),
            compute_dt=bool(compute_dt),
            imu_native_sampling_rate=int(imu_native_sampling_rate),
            imu_target_sampling_rate=int(imu_target_sampling_rate),
            rot_native_sampling_rate=int(rot_native_sampling_rate),
            rot_target_sampling_rate=int(rot_target_sampling_rate),
            tof_native_sampling_rate=int(tof_native_sampling_rate),
            tof_target_sampling_rate=int(tof_target_sampling_rate),
            thm_native_sampling_rate=int(thm_native_sampling_rate),
            thm_target_sampling_rate=int(thm_target_sampling_rate),
            resample_modalities=bool(resample_modalities),
            maxlen=int(maxlen),
            add_global_context=bool(add_global_context),
            frame_stats=str(frame_stats),
        )
        p.update(extra)
        return p

    def _params_from_controls(self, c, **extra):
        keys = (
            "acc_modes", "rotation_modes", "tof_modes", "thm_modes",
            "motion_filter_mode", "use_dead_reckoning", "dead_reckoning_detrend",
            "kalman_process_noise", "kalman_measurement_noise",
            "smooth_alpha", "window_size", "clip_value", "interp_mode", "compute_dt",
            "imu_native_sampling_rate", "imu_target_sampling_rate",
            "rot_native_sampling_rate", "rot_target_sampling_rate",
            "tof_native_sampling_rate", "tof_target_sampling_rate",
            "thm_native_sampling_rate", "thm_target_sampling_rate",
            "resample_modalities", "maxlen", "add_global_context", "frame_stats",
        )
        return self._params(**{k: c[k].value for k in keys}, **extra)

    @staticmethod
    def _match_raw_channel(feature: str, columns: Sequence[str]) -> Optional[str]:
        """Match engineered feature to raw channel by prefix (acc_x_vel -> acc_x)."""
        cols = set(columns)
        name = str(feature)
        for suf in _FEATURE_SUFFIXES:
            if name.endswith(suf):
                name = name[: -len(suf)]
                break
        if name in cols:
            return name
        parts = feature.split("_")
        for i in range(len(parts), 0, -1):
            candidate = "_".join(parts[:i])
            if candidate in cols:
                return candidate
        return None

    @staticmethod
    def _center_axis(ax, *arrays):
        vals = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays if len(a)])
        if len(vals) == 0:
            return
        ymax = float(np.nanmax(np.abs(vals)))
        if ymax <= 0:
            ymax = 1.0
        ax.set_ylim(-ymax * 1.05, ymax * 1.05)
        ax.axhline(0.0, color="gray", lw=0.5, alpha=0.5)

    # ------------------------------------------------------------------
    # controls
    # ------------------------------------------------------------------
    def _all_feature_names(self):
        probe = self.sequence(self.meta_.index[0])
        p = self._params(
            ACC_MODES[-1], ROT_MODES[-1], TOF_MODES[-1], THM_MODES[-1],
            "none", False, False, 1e-3, 1e-2, 0.0, 7, 0.0, "linear", True,
            int(self.fs), int(self.fs), int(self.fs), int(self.fs),
            5, 5, 5, 5, False, 160, False, "mean,std,min,max,last",
        )
        cols = [c for c in self.extract(probe, **p).columns if c != self.sequence_col]
        return cols + [c for c in self.raw_channels_ if c not in cols]

    def _refresh_feature_options(self):
        try:
            p = self._params_from_controls(self.c)
            probe = self.sequence(self.meta_.index[0])
            cols = [
                c for c in self.extract(probe, **p).columns
                if c != self.sequence_col
            ]
            options = cols + [c for c in self.raw_channels_ if c not in cols]
            if options:
                self.c["feature"].options = options
                if self.c["feature"].value not in options:
                    self.c["feature"].value = options[0]
        except Exception:
            pass

    def _refresh_catalog_ui(self):
        self.catalog_view_.options = list(self.catalog_)
        self.catalog_count_.value = (
            f"<b>Catalog: {len(self.catalog_)} sequence(s)</b>"
        )

    def _add_to_catalog(self, _btn=None):
        sid = self._resolve(self.seq_pick_.value, self.seq_text_.value)
        if sid not in self.catalog_:
            self.catalog_.append(sid)
        self._refresh_catalog_ui()

    def _remove_from_catalog(self, _btn=None):
        selected = list(self.catalog_view_.value)
        self.catalog_ = [s for s in self.catalog_ if s not in selected]
        self._refresh_catalog_ui()

    def _clear_catalog(self, _btn=None):
        self.catalog_ = []
        self._refresh_catalog_ui()

    def make_controls(self):
        import ipywidgets as w

        self.feature_options_ = self._all_feature_names()
        self.filters_ = {
            c: w.SelectMultiple(
                options=["(all)"] + sorted(self.meta_[c].astype(str).unique()),
                value=("(all)",),
                description=c[:11] + ":",
                rows=6,
                layout=w.Layout(width="260px"),
            )
            for c in self.filter_cols_
        }
        seqs = self.candidates({})
        self.seq_pick_ = w.Dropdown(
            options=seqs, value=seqs[0], description="Sequence:",
            layout=w.Layout(width="330px"),
        )
        self.seq_text_ = w.Text(
            value="", description="or ID:", placeholder="SEQ_000007",
            layout=w.Layout(width="330px"),
        )
        for f in self.filters_.values():
            f.observe(self._on_filter_change, names="value")

        defaults = self.extractor_defaults()
        fs_int = int(self.fs)

        self.catalog_view_ = w.SelectMultiple(
            options=[], rows=4, description="Catalog:",
            layout=w.Layout(width="420px"),
        )
        self.catalog_count_ = w.HTML(value="<b>Catalog: 0 sequence(s)</b>")
        self.add_catalog_btn_ = w.Button(description="Add current", button_style="info")
        self.remove_catalog_btn_ = w.Button(description="Remove selected")
        self.clear_catalog_btn_ = w.Button(description="Clear catalog")
        self.add_catalog_btn_.on_click(self._add_to_catalog)
        self.remove_catalog_btn_.on_click(self._remove_from_catalog)
        self.clear_catalog_btn_.on_click(self._clear_catalog)

        self.c = {
            "channels": w.SelectMultiple(
                options=self.raw_channels_,
                value=tuple(self.raw_channels_[:3]),
                description="Channels:", rows=8,
            ),
            "view_mode": w.Dropdown(
                options=["current", "catalog overlay", "catalog grid"],
                value="current", description="View:",
            ),
            "acc_modes": w.Dropdown(options=ACC_MODES, value=ACC_MODES[3], description="acc:"),
            "rotation_modes": w.Dropdown(
                options=ROT_MODES, value="quaternion|angular_velocity", description="rot:",
            ),
            "tof_modes": w.Dropdown(options=TOF_MODES, value=TOF_MODES[0], description="tof:"),
            "thm_modes": w.Dropdown(options=THM_MODES, value=THM_MODES[-1], description="thm:"),
            "motion_filter_mode": w.Dropdown(
                options=["none", "kalman", "extended_kalman"], value="none", description="filter:",
            ),
            "use_dead_reckoning": w.Checkbox(value=False, description="dead reckoning"),
            "dead_reckoning_detrend": w.Checkbox(value=False, description="dr detrend"),
            "kalman_process_noise": w.FloatLogSlider(
                value=-3, base=10, min=-6, max=0, step=0.25, description="kalman Q:",
            ),
            "kalman_measurement_noise": w.FloatLogSlider(
                value=-2, base=10, min=-6, max=0, step=0.25, description="kalman R:",
            ),
            "smooth_alpha": w.FloatSlider(
                min=0.0, max=1.0, step=0.05, value=0.0, description="ewm a:",
            ),
            "window_size": w.IntSlider(min=3, max=31, step=2, value=7, description="window:"),
            "clip_value": w.FloatSlider(
                min=0.0, max=200.0, step=10.0, value=0.0, description="clip:",
            ),
            "interp_mode": w.Dropdown(options=INTERP_MODES, value="linear", description="interp:"),
            "compute_dt": w.Checkbox(value=True, description="compute dt"),
            "resample_modalities": w.Checkbox(value=False, description="resample"),
            "imu_native_sampling_rate": w.IntSlider(
                min=1, max=100, step=1, value=fs_int, description="imu native:",
            ),
            "imu_target_sampling_rate": w.IntSlider(
                min=1, max=100, step=1, value=fs_int, description="imu target:",
            ),
            "rot_native_sampling_rate": w.IntSlider(
                min=1, max=100, step=1, value=fs_int, description="rot native:",
            ),
            "rot_target_sampling_rate": w.IntSlider(
                min=1, max=100, step=1, value=fs_int, description="rot target:",
            ),
            "tof_native_sampling_rate": w.IntSlider(
                min=1, max=20, step=1, value=5, description="tof native:",
            ),
            "tof_target_sampling_rate": w.IntSlider(
                min=1, max=20, step=1, value=5, description="tof target:",
            ),
            "thm_native_sampling_rate": w.IntSlider(
                min=1, max=20, step=1, value=5, description="thm native:",
            ),
            "thm_target_sampling_rate": w.IntSlider(
                min=1, max=20, step=1, value=5, description="thm target:",
            ),
            "maxlen": w.IntSlider(
                min=32, max=512, step=16, value=int(defaults.get("maxlen", 160)),
                description="maxlen:",
            ),
            "add_global_context": w.Checkbox(
                value=bool(defaults.get("add_global_context", False)),
                description="global ctx",
            ),
            "frame_stats": w.Dropdown(
                options=FRAME_STATS, value="mean,std,min,max,last", description="frame stats:",
            ),
            "feature": w.Dropdown(
                options=self.feature_options_,
                value=self.feature_options_[0], description="Feature:",
            ),
            "spec_mode": w.Dropdown(
                options=["fft", "psd", "spectrogram"], value="fft", description="Spectral:",
            ),
            "detrend": w.Checkbox(value=True, description="detrend"),
            "logy": w.Checkbox(value=True, description="log power"),
            "reduction": w.Dropdown(
                options=["pca", "tsne"] + (["umap"] if _UMAP else []),
                value="pca", description="Reduce:",
            ),
            "method": w.Dropdown(
                options=["kmeans", "gmm", "agglomerative", "dbscan"]
                + (["hdbscan"] if _HDBSCAN else []),
                value="kmeans", description="Cluster:",
            ),
            "n_clusters": w.IntSlider(min=2, max=12, step=1, value=5, description="k:"),
            "eps": w.FloatSlider(min=0.1, max=5.0, step=0.1, value=0.8, description="eps:"),
            "cluster_space": w.Dropdown(
                options=["features", "embedding"], value="features", description="Fit on:",
            ),
            "cluster_source": w.Dropdown(
                options=["filtered", "catalog", "filtered+catalog"],
                value="filtered", description="Source:",
            ),
            "max_seqs": w.IntSlider(min=50, max=2000, step=50, value=400, description="Max seqs:"),
            "color_by": w.Dropdown(
                options=["cluster"] + self.filter_cols_, value="cluster", description="Colour:",
            ),
        }

        for key in ("acc_modes", "rotation_modes", "tof_modes", "thm_modes",
                    "motion_filter_mode", "use_dead_reckoning", "smooth_alpha",
                    "window_size", "clip_value"):
            self.c[key].observe(lambda _: self._refresh_feature_options(), names="value")

        return self.c

    def _on_filter_change(self, _):
        seqs = self.candidates({c: list(f.value) for c, f in self.filters_.items()})
        if not seqs:
            return
        self.seq_pick_.options = seqs
        self.seq_pick_.value = seqs[0]

    def _resolve(self, seq_id, seq_text):
        sid = seq_text.strip() or seq_id
        if sid not in set(self.meta_.index):
            raise KeyError(f"{sid} not in data")
        return sid

    def _sequence_ids_for_view(self, seq_id, seq_text, view_mode):
        if view_mode == "current":
            return [self._resolve(seq_id, seq_text)]
        if not self.catalog_:
            print("Catalog is empty — add sequences with 'Add current'.")
            return [self._resolve(seq_id, seq_text)]
        return list(self.catalog_)

    def _cluster_sequence_ids(self, cluster_source, max_seqs):
        filtered = self.candidates({c: list(f.value) for c, f in self.filters_.items()})
        catalog = list(self.catalog_)

        if cluster_source == "catalog":
            seqs = catalog
        elif cluster_source == "filtered+catalog":
            seqs = sorted(set(filtered) | set(catalog))
        else:
            seqs = filtered

        if len(seqs) < 5:
            print(f"need at least 5 sequences (have {len(seqs)}) for source={cluster_source}")
            return []

        rng = np.random.RandomState(self.random_state)
        if len(seqs) > max_seqs:
            seqs = list(rng.choice(seqs, size=int(max_seqs), replace=False))
        return seqs

    # ------------------------------------------------------------------
    # panel 1: raw sequence
    # ------------------------------------------------------------------
    def update_raw(self, seq_id, seq_text, channels, view_mode):
        sids = self._sequence_ids_for_view(seq_id, seq_text, view_mode)
        chans = list(channels) or self.raw_channels_[:3]

        if view_mode == "catalog grid" and len(sids) > 1:
            n = len(sids)
            ncols = min(3, n)
            nrows = int(np.ceil(n / ncols))
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(4.5 * ncols, 2.2 * nrows), squeeze=False,
            )
            for ax, sid in zip(axes.ravel(), sids):
                g = self.sequence(sid)
                t = np.arange(len(g)) / self.fs
                for ch in chans[:1]:
                    ax.plot(t, g[ch].to_numpy(), lw=0.8)
                meta = self.meta_.loc[sid]
                ax.set_title(
                    f"{sid}\n{meta.get('gesture', '')} | {meta.get('orientation', '')}",
                    fontsize=8,
                )
                self._center_axis(ax, g[chans[:1]].to_numpy())
                ax.grid(alpha=0.3)
            for ax in axes.ravel()[len(sids):]:
                ax.axis("off")
            fig.suptitle(f"Catalog grid ({len(sids)} sequences)", fontsize=10)
            fig.tight_layout()
            plt.show()
            print(f"Showing {len(sids)} catalog sequence(s) | channels: {chans[:1]}")
            return

        if view_mode == "catalog overlay" and len(sids) > 1:
            fig, axes = plt.subplots(len(chans), 1, figsize=(12, 1.8 * len(chans)),
                                     sharex=True, squeeze=False)
            for ax, ch in zip(axes[:, 0], chans):
                for sid in sids:
                    g = self.sequence(sid)
                    t = np.arange(len(g)) / self.fs
                    ax.plot(t, g[ch].to_numpy(), lw=0.7, alpha=0.75, label=sid)
                self._center_axis(ax, *(self.sequence(s)[ch].to_numpy() for s in sids))
                ax.set_ylabel(ch, fontsize=8)
                ax.grid(alpha=0.3)
                if len(sids) <= 8:
                    ax.legend(fontsize=6, ncol=2, loc="upper right")
            axes[-1, 0].set_xlabel("time (s)")
            fig.suptitle(f"Catalog overlay ({len(sids)} sequences)", fontsize=10)
            fig.tight_layout()
            plt.show()
            print(f"Catalog: {len(sids)} sequence(s)")
            return

        sid = sids[0]
        g = self.sequence(sid)
        meta = self.meta_.loc[sid]

        fig, axes = plt.subplots(len(chans), 1, figsize=(12, 1.6 * len(chans)),
                                 sharex=True, squeeze=False)
        for ax, ch in zip(axes[:, 0], chans):
            y = g[ch].to_numpy()
            ax.plot(np.arange(len(g)) / self.fs, y, lw=0.9)
            self._center_axis(ax, y)
            ax.set_ylabel(ch, fontsize=8)
            ax.grid(alpha=0.3)
        axes[-1, 0].set_xlabel("time (s)")
        fig.suptitle(
            f"{sid} | " + " | ".join(f"{c}={meta[c]}" for c in self.filter_cols_[:4]),
            fontsize=10,
        )
        fig.tight_layout()
        plt.show()
        print(
            f"{len(g)} samples ({len(g) / self.fs:.1f}s @ {self.fs:g} Hz) | "
            f"NaN cells: {int(g[chans].isna().sum().sum())} | "
            f"Catalog total: {len(self.catalog_)}"
        )

    # ------------------------------------------------------------------
    # panel 2: feature extraction before/after
    # ------------------------------------------------------------------
    def update_features(self, seq_id, seq_text, feature, **control_values):
        sid = self._resolve(seq_id, seq_text)
        g = self.sequence(sid)
        params = self._params(**control_values)
        out = self.extract(g, **params)
        cols = [c for c in out.columns if c != self.sequence_col]
        if feature not in cols:
            print(f"'{feature}' not produced by these parameters — showing {cols[0]}")
            feature = cols[0]

        raw_ch = self._match_raw_channel(feature, g.columns)
        t = np.arange(len(g)) / self.fs
        after = out[feature].to_numpy(dtype=float)

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        if raw_ch is not None:
            before = g[raw_ch].to_numpy(dtype=float)
            axes[0].plot(t, before, lw=0.9, color="slategray")
            axes[0].set_title(f"before: {raw_ch}", fontsize=10)
            self._center_axis(axes[0], before)
        else:
            axes[0].text(0.5, 0.5, "no matching raw channel", ha="center", va="center")
        axes[1].plot(t[: len(out)], after, lw=0.9, color="seagreen")
        axes[1].set_title(f"after: {feature}", fontsize=10)
        self._center_axis(axes[1], after)
        for ax in axes:
            ax.grid(alpha=0.3)
        axes[1].set_xlabel("time (s)")
        fig.suptitle(
            f"{sid} | acc={params['acc_modes']} | rot={params['rotation_modes']} | "
            f"filter={params['motion_filter_mode']} | dr={params['use_dead_reckoning']}",
            fontsize=9,
        )
        fig.tight_layout()
        plt.show()

        rows = []
        v = after
        rows.append([feature, v.mean(), v.std(), v.min(), v.max(), float(np.nanmedian(v))])
        if raw_ch is not None:
            r = g[raw_ch].to_numpy(dtype=float)
            rows.append([
                raw_ch, np.nanmean(r), np.nanstd(r), np.nanmin(r), np.nanmax(r),
                float(np.nanmedian(r)),
            ])
        stats = pd.DataFrame(rows, columns=["channel", "mean", "std", "min", "max", "median"])
        if raw_ch is not None:
            stats["delta_mean"] = stats["mean"] - stats["mean"].iloc[1]
            stats.loc[stats.index[1], "delta_mean"] = np.nan
            stats["delta_std"] = stats["std"] - stats["std"].iloc[1]
            stats.loc[stats.index[1], "delta_std"] = np.nan

        print(f"{len(cols)} engineered columns from SequenceExtractor._preprocess_features")
        print("Active extractor params:")
        for k in self.extractor_param_names_:
            if k in params:
                print(f"  {k}: {params[k]}")
        print(stats.round(4).to_string(index=False))

    # ------------------------------------------------------------------
    # panel 3: spectral
    # ------------------------------------------------------------------
    def _spectrum(self, x, mode, detrend):
        x = np.nan_to_num(np.asarray(x, dtype=float))
        if detrend:
            x = signal.detrend(x, type="linear") if len(x) > 1 else x
        nper = min(256, max(8, len(x)))
        if mode == "psd":
            return signal.welch(x, fs=self.fs, nperseg=nper, window="hann")
        win = np.hanning(len(x))
        X = np.fft.rfft(x * win)
        f = np.fft.rfftfreq(len(x), d=1.0 / self.fs)
        amp = (2.0 / max(win.sum(), 1e-12)) * np.abs(X)
        return f, amp

    def _print_spectral_stats(self, name, f, P):
        P = np.asarray(P, dtype=float)
        if len(P) <= 1 or P[1:].sum() <= 0:
            print(f"{name:6s} peak=n/a | centroid=n/a | power={P.sum():.4g}")
            return
        body = P[1:]
        freqs = f[1:]
        print(
            f"{name:6s} peak={freqs[np.argmax(body)]:6.2f} Hz | "
            f"centroid={(freqs * body).sum() / body.sum():6.2f} Hz | "
            f"power={body.sum():.4g}"
        )

    def update_spectrum(self, seq_id, seq_text, feature, spec_mode, detrend, logy,
                        **control_values):
        sid = self._resolve(seq_id, seq_text)
        g = self.sequence(sid)
        params = self._params(**control_values)
        out = self.extract(g, **params)
        cols = [c for c in out.columns if c != self.sequence_col]
        if feature not in cols:
            feature = cols[0]

        raw_ch = self._match_raw_channel(feature, g.columns)
        if raw_ch is None:
            raw_ch = self.raw_channels_[0]

        before = g[raw_ch].to_numpy(dtype=float)
        after = out[feature].to_numpy(dtype=float)

        if spec_mode == "spectrogram":
            fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
            for ax, (x, ttl) in zip(
                axes,
                [(before, f"before: {raw_ch}"), (after, f"after: {feature}")],
            ):
                nper = min(64, max(8, len(x) // 4))
                f, t_spec, S = signal.spectrogram(
                    np.nan_to_num(x), fs=self.fs, nperseg=nper,
                    noverlap=nper // 2, window="hann",
                )
                ax.pcolormesh(
                    t_spec, f, 10 * np.log10(S + 1e-12),
                    shading="gouraud", cmap="magma",
                )
                ax.set_title(ttl, fontsize=10)
                ax.set_xlabel("time (s)")
                ax.axhline(self.fs / 2, color="white", ls="--", lw=0.6, alpha=0.7)
            axes[0].set_ylabel("Hz")
            fig.suptitle(
                f"{sid} | pipeline: raw→{params['acc_modes']} | Nyquist={self.fs / 2:g} Hz",
                fontsize=10,
            )
            fig.tight_layout()
            plt.show()
            return

        fb, Pb = self._spectrum(before, spec_mode, detrend)
        fa, Pa = self._spectrum(after, spec_mode, detrend)
        t = np.arange(len(g)) / self.fs

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        axes[0].plot(t, before, lw=0.8, color="slategray", label=f"before: {raw_ch}")
        axes[0].plot(t[: len(after)], after, lw=0.8, color="seagreen", label=f"after: {feature}")
        self._center_axis(axes[0], before, after)
        axes[0].set_xlabel("time (s)")
        axes[0].legend(fontsize=8)
        axes[0].set_title("time domain (centered on 0)", fontsize=10)

        axes[1].plot(fb, Pb, lw=1.0, color="slategray", label=f"before: {raw_ch}")
        axes[1].plot(fa, Pa, lw=1.0, color="seagreen", label=f"after: {feature}")
        axes[1].axvline(self.fs / 2, color="gray", ls="--", lw=0.8, alpha=0.7)
        axes[1].set_xlabel("Hz")
        axes[1].set_ylabel("PSD" if spec_mode == "psd" else "|X(f)|")
        axes[1].set_title(
            f"{spec_mode.upper()} (Nyquist = {self.fs / 2:g} Hz)", fontsize=10,
        )
        axes[1].legend(fontsize=8)
        if logy:
            axes[1].set_yscale("log")
        for ax in axes:
            ax.grid(alpha=0.3)
        fig.suptitle(sid, fontsize=10)
        fig.tight_layout()
        plt.show()

        self._print_spectral_stats("before", fb, Pb)
        self._print_spectral_stats("after", fa, Pa)

    # ------------------------------------------------------------------
    # panel 4: unsupervised
    # ------------------------------------------------------------------
    def update_cluster(self, reduction, method, n_clusters, eps, cluster_space,
                       cluster_source, max_seqs, color_by, **control_values):
        seqs = self._cluster_sequence_ids(cluster_source, max_seqs)
        if not seqs:
            return
        sub = self.df[self.df[self.sequence_col].isin(seqs)]

        params = self._params(
            **control_values,
            output_format="frame",
        )
        F = SequenceExtractor(**params).fit_transform(sub)
        M = StandardScaler().fit_transform(np.nan_to_num(F.to_numpy(dtype=float)))

        if reduction == "tsne":
            Z = TSNE(
                n_components=2, init="pca", learning_rate="auto",
                perplexity=min(30, max(5, len(M) // 4)),
                random_state=self.random_state,
            ).fit_transform(M)
        elif reduction == "umap" and _UMAP:
            Z = UMAP(n_components=2, random_state=self.random_state).fit_transform(M)
        else:
            Z = PCA(n_components=2, random_state=self.random_state).fit_transform(M)

        S = M if cluster_space == "features" else Z
        if method == "kmeans":
            labels = KMeans(
                n_clusters=n_clusters, n_init=10, random_state=self.random_state,
            ).fit_predict(S)
        elif method == "gmm":
            labels = GaussianMixture(
                n_components=n_clusters, n_init=3, random_state=self.random_state,
            ).fit_predict(S)
        elif method == "agglomerative":
            labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(S)
        elif method == "dbscan":
            labels = DBSCAN(eps=float(eps), min_samples=5).fit_predict(S)
        else:
            labels = HDBSCAN(min_cluster_size=max(5, len(S) // 50)).fit_predict(S)

        n_lab = len(set(labels)) - (1 if -1 in labels else 0)
        sil = silhouette_score(S, labels) if 2 <= n_lab < len(S) else np.nan
        truth = (
            self.meta_.loc[F.index, self.label_col].astype(str)
            if self.label_col in self.meta_.columns else None
        )
        ari = (
            adjusted_rand_score(truth, labels)
            if truth is not None and n_lab >= 2 else np.nan
        )

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        axes[0].scatter(Z[:, 0], Z[:, 1], c=labels, cmap="tab10", s=14, alpha=0.8)
        axes[0].set_title(
            f"{method} on {cluster_space} | source={cluster_source} | "
            f"{n_lab} clusters | silhouette={sil:.3f} | ARI={ari:.3f}",
            fontsize=9,
        )
        key = color_by if color_by != "cluster" else self.label_col
        if key in self.meta_.columns:
            codes = self.meta_.loc[F.index, key].astype("category")
            axes[1].scatter(
                Z[:, 0], Z[:, 1], c=codes.cat.codes, cmap="tab20", s=14, alpha=0.8,
            )
            axes[1].set_title(f"true {key} ({codes.nunique()} levels)", fontsize=9)
        for ax in axes:
            ax.set_xlabel(f"{reduction} 1")
            ax.set_ylabel(f"{reduction} 2")
            ax.grid(alpha=0.3)
        fig.tight_layout()
        plt.show()

        print(
            f"{len(F)} sequences | {F.shape[1]} frame features | "
            f"source={cluster_source} | catalog={len(self.catalog_)}"
        )
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

        c = self.make_controls()
        param_keys = (
            "acc_modes", "rotation_modes", "tof_modes", "thm_modes",
            "motion_filter_mode", "use_dead_reckoning", "dead_reckoning_detrend",
            "kalman_process_noise", "kalman_measurement_noise",
            "smooth_alpha", "window_size", "clip_value", "interp_mode", "compute_dt",
            "imu_native_sampling_rate", "imu_target_sampling_rate",
            "rot_native_sampling_rate", "rot_target_sampling_rate",
            "tof_native_sampling_rate", "tof_target_sampling_rate",
            "thm_native_sampling_rate", "thm_target_sampling_rate",
            "resample_modalities", "maxlen", "add_global_context", "frame_stats",
        )
        fx_kw = {k: c[k] for k in param_keys}

        picker = w.VBox([
            w.HBox(list(self.filters_.values())),
            w.HBox([self.seq_pick_, self.seq_text_]),
            w.HBox([
                self.add_catalog_btn_, self.remove_catalog_btn_,
                self.clear_catalog_btn_, self.catalog_count_,
            ]),
            w.HBox([self.catalog_view_]),
        ])

        core_extractor = w.VBox([
            w.HBox([c["acc_modes"], c["rotation_modes"], c["tof_modes"], c["thm_modes"]]),
            w.HBox([
                c["motion_filter_mode"], c["use_dead_reckoning"],
                c["dead_reckoning_detrend"], c["interp_mode"], c["compute_dt"],
            ]),
            w.HBox([c["smooth_alpha"], c["window_size"], c["clip_value"]]),
            w.HBox([c["kalman_process_noise"], c["kalman_measurement_noise"]]),
        ])

        advanced_extractor = w.VBox([
            w.HBox([
                c["imu_native_sampling_rate"], c["imu_target_sampling_rate"],
                c["rot_native_sampling_rate"], c["rot_target_sampling_rate"],
            ]),
            w.HBox([
                c["tof_native_sampling_rate"], c["tof_target_sampling_rate"],
                c["thm_native_sampling_rate"], c["thm_target_sampling_rate"],
            ]),
            w.HBox([c["resample_modalities"], c["maxlen"], c["add_global_context"], c["frame_stats"]]),
        ])
        extractor_accordion = w.Accordion(children=[advanced_extractor])
        extractor_accordion.set_title(0, "Advanced SequenceExtractor params")

        o1 = w.interactive_output(
            self.update_raw,
            {
                "seq_id": self.seq_pick_, "seq_text": self.seq_text_,
                "channels": c["channels"], "view_mode": c["view_mode"],
            },
        )
        o2 = w.interactive_output(
            self.update_features,
            {"seq_id": self.seq_pick_, "seq_text": self.seq_text_,
             "feature": c["feature"], **fx_kw},
        )
        o3 = w.interactive_output(
            self.update_spectrum,
            {"seq_id": self.seq_pick_, "seq_text": self.seq_text_,
             "feature": c["feature"], "spec_mode": c["spec_mode"],
             "detrend": c["detrend"], "logy": c["logy"], **fx_kw},
        )
        o4 = w.interactive_output(
            self.update_cluster,
            {"reduction": c["reduction"], "method": c["method"],
             "n_clusters": c["n_clusters"], "eps": c["eps"],
             "cluster_space": c["cluster_space"], "cluster_source": c["cluster_source"],
             "max_seqs": c["max_seqs"], "color_by": c["color_by"], **fx_kw},
        )

        tabs = w.Tab(children=[
            w.VBox([c["view_mode"], c["channels"], o1]),
            w.VBox([core_extractor, extractor_accordion, c["feature"], o2]),
            w.VBox([
                core_extractor, extractor_accordion,
                w.HBox([c["feature"], c["spec_mode"], c["detrend"], c["logy"]]), o3,
            ]),
            w.VBox([
                core_extractor, extractor_accordion,
                w.HBox([c["reduction"], c["method"], c["cluster_space"], c["cluster_source"]]),
                w.HBox([c["n_clusters"], c["eps"], c["max_seqs"], c["color_by"]]), o4,
            ]),
        ])
        for i, t in enumerate(["Sequence", "Features", "Spectrum", "Clusters"]):
            tabs.set_title(i, t)
        display(w.VBox([picker, tabs]))
