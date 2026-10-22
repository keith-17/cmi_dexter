"""
sequence_explorer_utils.py
Interactive explorer for the CMI sensor data, built directly on the
SignalCleaner / MotionFilter / IMUExtractor / RotationExtractor /
SequenceExtractor pipeline in base_utils_qwen.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal as sp_signal
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

try:
    from umap import UMAP
    _UMAP = True
except Exception:
    _UMAP = False

try:
    import pywt
    _PYWT = True
except Exception:
    _PYWT = False

try:
    from sklearn.cluster import HDBSCAN as SKHDBSCAN
    _HDBSCAN_AVAILABLE = True
except Exception:
    _HDBSCAN_AVAILABLE = False

from base_utils_qwen import (
    SequenceExtractor,
    SignalCleaner,
    MotionFilter,
    IMUExtractor,
    RotationExtractor,
    STFTExtractor,
    CWTExtractor,
)


ACC_MODE_OPTS     = ('raw', 'velocity', 'displacement', 'jerk', 'smoothed')
ROT_MODE_OPTS     = ('quaternion', 'euler', 'delta_euler', 'angular_velocity', 'rot6d')
TOF_MODE_OPTS     = ('pooled_stats', 'sensor_stats', 'raw', 'pooled')
THM_MODE_OPTS     = ('centered_diff', 'raw', 'centered', 'diff')
FRAME_STAT_OPTS   = ('mean', 'std', 'min', 'max', 'last', 'first', 'median', 'rms', 'abs_mean')
SPEC_MODE_OPTS    = ('fft', 'psd', 'spectrogram', 'stft', 'cwt')
CLUSTER_REDUCTION = ('pca', 'tsne', 'umap')
CLUSTER_METHODS   = ('kmeans', 'gmm', 'agg', 'dbscan', 'hdbscan') if _HDBSCAN_AVAILABLE else \
                    ('kmeans', 'gmm', 'agg', 'dbscan')


def _center_zero(ax, y):
    if len(y) == 0:
        return
    ymin, ymax = np.nanmin(y), np.nanmax(y)
    margin = max(abs(ymin), abs(ymax)) * 1.1
    if margin == 0:
        margin = 1
    ax.set_ylim(-margin, margin)


def _norm_modes(v, default):
    """Normalize a SelectMultiple tuple / string to a '|' joined string,
    stripping duplicates and 'none' sentinels."""
    if v is None:
        return default
    if isinstance(v, (tuple, list)):
        seen = []
        for item in v:
            for part in str(item).split('|'):
                part = part.strip()
                if part and part.lower() != 'none' and part not in seen:
                    seen.append(part)
        return '|'.join(seen) if seen else default
    return str(v)


def _to_sequence_ids(seqs):
    """Utility: accept a list of sequence ids, return unique list."""
    return list(dict.fromkeys([s for s in seqs if s is not None]))


class SequenceExplorer:
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

        # Derive gesture position / action
        if 'gesture_position' not in self.df.columns:
            self.df['gesture_position'] = (
                self.df['gesture'].astype(str).str.split(' - ').str[0].fillna('unknown')
            )
        if 'gesture_action' not in self.df.columns:
            self.df['gesture_action'] = (
                self.df['gesture'].astype(str).str.split(' - ').str[-1].fillna('unknown')
            )

        # Precompute useful auxiliary channels so the UI has smaller dropdowns
        # (raw TOF has 320 columns; users mostly want the pooled mean per sensor)
        for i in range(1, 6):
            cols = [c for c in self.df.columns if c.startswith(f'tof_{i}_v')]
            if cols:
                self.df[f'tof_{i}_mean'] = (
                    self.df[cols].replace(-1.0, np.nan).mean(axis=1).fillna(0.0)
                )
        self.acc_axes_ = [c for c in self.df.columns if c.startswith('acc_')]
        self.rot_axes_ = [c for c in self.df.columns if c.startswith('rot_')]
        self.thm_axes_ = [c for c in self.df.columns if c.startswith('thm_')]
        if all(c in self.df.columns for c in ('acc_x', 'acc_y', 'acc_z')):
            self.df['acc_mag'] = np.sqrt(
                (self.df[['acc_x', 'acc_y', 'acc_z']].astype(float) ** 2).sum(axis=1)
            )

        # Filter columns exposed as scrollable multi-selects
        self.filter_cols_ = [c for c in [
            "subject", "sequence_type", "gesture",
            "gesture_position", "gesture_action",
            "orientation", "behavior", "phase", "problematic_sequence",
        ] if c in self.df.columns]

        self.meta_ = self.df.groupby(self.sequence_col).first()

        # Raw channels available for plotting/spectrum
        excluded = set(self.filter_cols_ + [
            self.sequence_col, self.counter_col, self.label_col,
            'row_id', 'handedness', 'is_target', 'bfrb', 'problematic_sequence',
        ])
        self.raw_channels_ = [c for c in self.df.columns if c not in excluded]

        # Smaller, friendlier channel options for spectrum
        self.spectrum_channels_ = (
            [c for c in self.raw_channels_ if c.startswith(('acc_', 'rot_'))]
            + ['acc_mag']
            + [f'tof_{i}_mean' for i in range(1, 6) if f'tof_{i}_mean' in self.df.columns]
            + [c for c in self.raw_channels_ if c.startswith('thm_')]
        )
        # de-dupe preserving order
        self.spectrum_channels_ = list(dict.fromkeys(self.spectrum_channels_))

        self.catalogue_ = []
        self._seq_cache = {}

    # ---------------------------------------------------------------- data --
    def sequence(self, seq_id):
        if seq_id not in self._seq_cache:
            self._seq_cache[seq_id] = self.df[self.df[self.sequence_col] == seq_id].copy()
        return self._seq_cache[seq_id]

    def candidates(self, filters):
        mask = pd.Series(True, index=self.meta_.index)
        for col, vals in filters.items():
            if not vals or "(all)" in vals:
                continue
            mask &= self.meta_[col].astype(str).isin(vals)
        return self.meta_.index[mask].tolist()

    def _current_filters(self):
        return {col: list(w.value) for col, w in self.filters_.items()}

    def _filter_text(self):
        parts = []
        for col, vals in self._current_filters().items():
            vals = [v for v in vals if v != "(all)"]
            if vals:
                parts.append(f"{col}={{{','.join(map(str, vals))}}}")
        return " & ".join(parts) if parts else "(no filters)"

    def _resolve(self, seq_id, seq_text):
        if seq_text and str(seq_text).strip():
            return str(seq_text).strip()
        return seq_id

    # -------------------------------------------------------- extractor cfg --
    def _extractor_kwargs(self, vals):
        return dict(
            acc_modes=_norm_modes(vals.get("acc_modes"), "raw"),
            rotation_modes=_norm_modes(vals.get("rotation_modes"), "quaternion"),
            tof_modes=_norm_modes(vals.get("tof_modes"), "pooled_stats"),
            thm_modes=_norm_modes(vals.get("thm_modes"), "centered_diff"),
            motion_filter_mode=vals.get("motion_filter_mode", "none") or None,
            use_dead_reckoning=bool(vals.get("use_dead_reckoning", False)),
            dead_reckoning_detrend=bool(vals.get("dead_reckoning_detrend", False)),
            kalman_process_noise=10 ** float(vals.get("kalman_process_noise", -3)),
            kalman_measurement_noise=10 ** float(vals.get("kalman_measurement_noise", -2)),
            compute_dt=bool(vals.get("compute_dt", True)),
            window_size=int(vals.get("window_size", 7)),
            smooth_alpha=float(vals.get("smooth_alpha", 0.0)) or None,
            clip_value=float(vals.get("clip_value", 0.0)) or None,
            interp_mode=vals.get("interp_mode", "linear"),
            maxlen=int(vals.get("maxlen", 160)),
            padding_value=float(vals.get("padding_value", 0.0)),
            imu_native_sampling_rate=int(vals.get("imu_native_sampling_rate", self.fs)),
            imu_target_sampling_rate=int(vals.get("imu_target_sampling_rate", self.fs)),
            rot_native_sampling_rate=int(vals.get("rot_native_sampling_rate", self.fs)),
            rot_target_sampling_rate=int(vals.get("rot_target_sampling_rate", self.fs)),
            tof_native_sampling_rate=int(vals.get("tof_native_sampling_rate", 5)),
            tof_target_sampling_rate=int(vals.get("tof_target_sampling_rate", 5)),
            thm_native_sampling_rate=int(vals.get("thm_native_sampling_rate", 5)),
            thm_target_sampling_rate=int(vals.get("thm_target_sampling_rate", 5)),
            resample_modalities=bool(vals.get("resample_modalities", False)),
            chunk_window_size=int(vals.get("chunk_window_size", 0)) or None,
            chunk_stride=int(vals.get("chunk_stride", 0)) or None,
            frame_stats=",".join(list(vals.get("frame_stats", FRAME_STAT_OPTS)))
                          if isinstance(vals.get("frame_stats"), (tuple, list))
                          else vals.get("frame_stats", "mean,std,min,max,last"),
            add_global_context=bool(vals.get("add_global_context", False)),
            stft_nperseg=int(vals.get("stft_nperseg", 64)),
            stft_noverlap=int(vals.get("stft_noverlap", 32)),
            stft_use_log_scale=bool(vals.get("stft_use_log_scale", True)),
            cwt_wavelet=vals.get("cwt_wavelet", "morl"),
            cwt_max_scale=int(vals.get("cwt_max_scale", 100)),
            cwt_n_scales=int(vals.get("cwt_n_scales", 50)),
            cwt_use_log_scale=bool(vals.get("cwt_use_log_scale", True)),
        )

    def extract(self, seq_df, extractor_kwargs, output_format="chunks"):
        ex = SequenceExtractor(output_format=output_format, **extractor_kwargs)
        out = ex.fit_transform(seq_df)
        return ex, out

    # ------------------------------------------------------------- widgets --
    def make_controls(self):
        import ipywidgets as w

        # ---- Filter widgets (scrollable) ----
        self.filters_ = {}
        for c in self.filter_cols_:
            opts = ["(all)"] + sorted(self.meta_[c].astype(str).unique().tolist())
            self.filters_[c] = w.SelectMultiple(
                options=opts,
                value=("(all)",),
                description=c[:14] + ":",
                rows=6,
                layout=w.Layout(width="220px"),
                style={"description_width": "100px"},
            )
        for f in self.filters_.values():
            f.observe(self._on_filter_change, names="value")

        seqs = self.candidates({})
        self.seq_pick_ = w.Dropdown(
            options=seqs, value=seqs[0] if seqs else None,
            description="Sequence:", layout=w.Layout(width="320px"),
        )
        self.seq_text_ = w.Text(
            value="", description="or ID:", placeholder="SEQ_000007",
            layout=w.Layout(width="320px"),
        )

        # group add vs current add
        self.add_mode_ = w.RadioButtons(
            options=['Current', 'Filtered (group)'],
            value='Filtered (group)',
            description='Add mode:',
            layout=w.Layout(width="260px"),
        )
        self.add_btn_ = w.Button(description="+ Add Sequences", button_style="success")
        self.remove_one_btn_ = w.Button(description="- Remove Current", button_style="warning")
        self.clear_btn_ = w.Button(description="Clear Catalogue", button_style="danger")
        self.cat_label_ = w.HTML(value="<b>Catalogue: 0 sequences</b>")
        self.cat_version_ = w.IntText(value=0)
        self.cat_version_.layout.display = "none"

        self.add_btn_.on_click(self._add_sequences)
        self.remove_one_btn_.on_click(self._remove_current)
        self.clear_btn_.on_click(self._clear_catalogue)

        p = {}
        # Modes
        p["acc_modes"] = w.SelectMultiple(options=ACC_MODE_OPTS, value=("raw",),
                                          description="acc_modes:", rows=5,
                                          layout=w.Layout(width="260px"))
        p["rotation_modes"] = w.SelectMultiple(options=ROT_MODE_OPTS, value=("quaternion",),
                                               description="rot_modes:", rows=5,
                                               layout=w.Layout(width="260px"))
        p["tof_modes"] = w.SelectMultiple(options=TOF_MODE_OPTS, value=("pooled_stats",),
                                          description="tof_modes:", rows=4,
                                          layout=w.Layout(width="260px"))
        p["thm_modes"] = w.SelectMultiple(options=THM_MODE_OPTS, value=("centered_diff",),
                                          description="thm_modes:", rows=4,
                                          layout=w.Layout(width="260px"))
        # Motion filter
        p["motion_filter_mode"] = w.Dropdown(options=["none", "kalman", "extended_kalman"],
                                             value="none", description="motion_filter:")
        p["use_dead_reckoning"] = w.Checkbox(value=False, description="use_dead_reckoning")
        p["dead_reckoning_detrend"] = w.Checkbox(value=False, description="dead_reckoning_detrend")
        p["kalman_process_noise"] = w.FloatSlider(min=-6, max=0, step=0.5, value=-3,
                                                  description="log10 Q:",
                                                  style={"description_width": "100px"})
        p["kalman_measurement_noise"] = w.FloatSlider(min=-6, max=0, step=0.5, value=-2,
                                                      description="log10 R:",
                                                      style={"description_width": "100px"})
        # Cleaning / padding
        p["interp_mode"] = w.Dropdown(options=["linear", "ffill"], value="linear",
                                      description="interp_mode:")
        p["window_size"] = w.IntSlider(min=1, max=51, step=2, value=7, description="window_size:")
        p["smooth_alpha"] = w.FloatSlider(min=0, max=1, step=0.05, value=0.0,
                                          description="smooth_alpha:")
        p["clip_value"] = w.FloatSlider(min=0, max=100, step=1, value=0.0,
                                        description="clip_value:")
        p["maxlen"] = w.IntSlider(min=50, max=500, step=10, value=160, description="maxlen:")
        p["padding_value"] = w.FloatText(value=0.0, description="padding_value:",
                                         style={"description_width": "120px"})
        p["compute_dt"] = w.Checkbox(value=True, description="compute_dt")
        # Sampling rates
        p["imu_native_sampling_rate"] = w.IntSlider(min=1, max=100, value=int(self.fs),
                                                    description="imu_native_hz:")
        p["imu_target_sampling_rate"] = w.IntSlider(min=1, max=100, value=int(self.fs),
                                                    description="imu_target_hz:")
        p["rot_native_sampling_rate"] = w.IntSlider(min=1, max=100, value=int(self.fs),
                                                    description="rot_native_hz:")
        p["rot_target_sampling_rate"] = w.IntSlider(min=1, max=100, value=int(self.fs),
                                                    description="rot_target_hz:")
        p["tof_native_sampling_rate"] = w.IntSlider(min=1, max=100, value=5,
                                                    description="tof_native_hz:")
        p["tof_target_sampling_rate"] = w.IntSlider(min=1, max=100, value=5,
                                                    description="tof_target_hz:")
        p["thm_native_sampling_rate"] = w.IntSlider(min=1, max=100, value=5,
                                                    description="thm_native_hz:")
        p["thm_target_sampling_rate"] = w.IntSlider(min=1, max=100, value=5,
                                                    description="thm_target_hz:")
        p["resample_modalities"] = w.Checkbox(value=False, description="resample_modalities")
        # Chunking / frame stats
        p["chunk_window_size"] = w.IntSlider(min=0, max=400, step=8, value=0,
                                             description="chunk_window (0=None):",
                                             style={"description_width": "200px"})
        p["chunk_stride"] = w.IntSlider(min=0, max=400, step=8, value=0,
                                        description="chunk_stride (0=None):",
                                        style={"description_width": "200px"})
        p["frame_stats"] = w.SelectMultiple(options=FRAME_STAT_OPTS,
                                            value=("mean", "std", "min", "max", "last"),
                                            description="frame_stats:", rows=9,
                                            layout=w.Layout(width="260px"))
        p["add_global_context"] = w.Checkbox(value=False, description="add_global_context")
        # STFT / CWT
        p["stft_nperseg"] = w.IntSlider(min=16, max=256, step=16, value=64,
                                        description="stft_nperseg:")
        p["stft_noverlap"] = w.IntSlider(min=0, max=256, step=8, value=32,
                                         description="stft_noverlap (overlap):")
        p["stft_use_log_scale"] = w.Checkbox(value=True, description="stft_log_scale")
        p["cwt_wavelet"] = w.Dropdown(options=['morl', 'mexh', 'gaus1'], value='morl',
                                      description="cwt_wavelet:")
        p["cwt_max_scale"] = w.IntSlider(min=10, max=200, step=10, value=100,
                                         description="cwt_max_scale:")
        p["cwt_n_scales"] = w.IntSlider(min=10, max=100, step=10, value=50,
                                        description="cwt_n_scales:")
        p["cwt_use_log_scale"] = w.Checkbox(value=True, description="cwt_log_scale")

        self.p = p

        # ---- Feature tab: multi-select ----
        self.feature_options_ = self._all_feature_names()
        self.feature_ = w.SelectMultiple(
            options=self.feature_options_,
            value=(self.feature_options_[0],) if self.feature_options_ else (),
            description="Features:", rows=10,
            layout=w.Layout(width="330px"),
        )

        # ---- Spectrum tab: channel and mode ----
        self.spec_channel_ = w.Dropdown(
            options=self.spectrum_channels_,
            value=self.spectrum_channels_[0] if self.spectrum_channels_ else None,
            description="Channel:", layout=w.Layout(width="320px"),
        )
        self.spec_mode_ = w.Dropdown(
            options=list(SPEC_MODE_OPTS), value="fft", description="Spec Mode:",
        )
        self.spec_detrend_ = w.Checkbox(value=False, description="detrend")
        self.spec_nperseg_ = w.IntSlider(min=16, max=512, step=16, value=128,
                                         description="spec nperseg:")
        self.spec_noverlap_ = w.IntSlider(min=0, max=512, step=8, value=64,
                                          description="spec overlap:")
        self.spec_scale_ = w.Dropdown(options=["linear", "log", "db"], value="log",
                                      description="spec scale:")

        # ---- CWT/STFT feature panel ----
        self.cwt_stft_channel_ = w.Dropdown(
            options=self.spectrum_channels_,
            value=self.spectrum_channels_[0] if self.spectrum_channels_ else None,
            description="Channel:", layout=w.Layout(width="320px"),
        )
        self.cwt_stft_method_ = w.Dropdown(options=["STFT", "CWT"], value="CWT",
                                           description="Method:")

        # ---- Chunking compare tab ----
        self.chunk_compare_seq_ = w.Dropdown(
            options=seqs, value=seqs[0] if seqs else None,
            description="Sequence:", layout=w.Layout(width="320px"),
        )
        self.chunk_compare_channel_ = w.Dropdown(
            options=self.spectrum_channels_,
            value=self.spectrum_channels_[0] if self.spectrum_channels_ else None,
            description="Channel:", layout=w.Layout(width="320px"),
        )
        # Config A
        self.chunk_a_maxlen_ = w.IntSlider(min=50, max=500, step=10, value=160,
                                           description="A maxlen:")
        self.chunk_a_win_ = w.IntSlider(min=0, max=200, step=4, value=64,
                                        description="A chunk win:")
        self.chunk_a_stride_ = w.IntSlider(min=0, max=200, step=4, value=32,
                                           description="A chunk stride:")
        self.chunk_a_smooth_ = w.IntSlider(min=1, max=51, step=2, value=7,
                                           description="A window_size:")
        # Config B
        self.chunk_b_maxlen_ = w.IntSlider(min=50, max=500, step=10, value=240,
                                           description="B maxlen:")
        self.chunk_b_win_ = w.IntSlider(min=0, max=200, step=4, value=96,
                                        description="B chunk win:")
        self.chunk_b_stride_ = w.IntSlider(min=0, max=200, step=4, value=48,
                                           description="B chunk stride:")
        self.chunk_b_smooth_ = w.IntSlider(min=1, max=51, step=2, value=15,
                                           description="B window_size:")

        # ---- Cluster tab ----
        self.population_ = w.Dropdown(
            options=["filtered", "catalogue", "both"],
            value="both",
            description="Population:",
        )
        self.max_seqs_ = w.IntSlider(min=20, max=2000, step=20, value=400,
                                     description="Max seqs:")
        self.color_by_ = w.Dropdown(
            options=["cluster"] + self.filter_cols_,
            value="cluster",
            description="Colour:",
        )

    # ----------------------------------------------------- feature catalogue --
    def _all_feature_names(self):
        """Build feature names from a lightweight representative probe."""
        if not len(self.meta_):
            return []
        probe = self.sequence(self.meta_.index[0])
        try:
            representative = self._extractor_kwargs(dict(
                acc_modes=('raw', 'velocity'),
                rotation_modes=('quaternion',),
                tof_modes=("pooled_stats",),
                thm_modes=('centered_diff',),
                motion_filter_mode="none",
                use_dead_reckoning=False,
                dead_reckoning_detrend=False,
                kalman_process_noise=-3,
                kalman_measurement_noise=-2,
                compute_dt=True,
                window_size=7,
                smooth_alpha=0.0,
                clip_value=0.0,
                interp_mode="linear",
                maxlen=160,
                padding_value=0.0,
                imu_native_sampling_rate=int(self.fs),
                imu_target_sampling_rate=int(self.fs),
                rot_native_sampling_rate=int(self.fs),
                rot_target_sampling_rate=int(self.fs),
                tof_native_sampling_rate=5,
                tof_target_sampling_rate=5,
                thm_native_sampling_rate=5,
                thm_target_sampling_rate=5,
                resample_modalities=False,
                chunk_window_size=0,
                chunk_stride=0,
                frame_stats=('mean', 'std'),
                add_global_context=False,
                stft_nperseg=64, stft_noverlap=32,
                cwt_wavelet='morl', cwt_max_scale=20, cwt_n_scales=10,
            ))
            ex, out = self.extract(probe, representative, output_format="frame")
            cols = [c for c in out.columns if c != self.sequence_col]
        except Exception as e:
            print(f"[SequenceExplorer] feature probe failed: {e}")
            cols = []
        # raw channels are also selectable
        extras = [c for c in self.raw_channels_ if c not in cols]
        return cols + extras

    # ------------------------------------------------------- filter handler --
    def _on_filter_change(self, _):
        seqs = self.candidates(self._current_filters())
        if not seqs:
            return
        self.seq_pick_.options = seqs
        self.seq_pick_.value = seqs[0]
        if hasattr(self, "chunk_compare_seq_"):
            self.chunk_compare_seq_.options = seqs
            if self.chunk_compare_seq_.value not in seqs:
                self.chunk_compare_seq_.value = seqs[0]

    def _bump(self):
        n = len(self.catalogue_)
        preview = ", ".join(self.catalogue_[-8:]) if n else ""
        tail = f" — last added: {preview}" if n else " (empty)"
        self.cat_label_.value = f"<b>Catalogue: {n} sequence(s)</b>{tail}"
        self.cat_version_.value += 1

    def _add_sequences(self, _btn):
        if self.add_mode_.value.startswith('Filtered'):
            added = 0
            for sid in self.candidates(self._current_filters()):
                if sid not in self.catalogue_:
                    self.catalogue_.append(sid)
                    added += 1
            if added > 0:
                self._bump()
        else:
            sid = self._resolve(self.seq_pick_.value, self.seq_text_.value)
            if sid and sid not in self.catalogue_:
                self.catalogue_.append(sid)
                self._bump()

    def _remove_current(self, _btn):
        sid = self._resolve(self.seq_pick_.value, self.seq_text_.value)
        if sid in self.catalogue_:
            self.catalogue_.remove(sid)
            self._bump()

    def _clear_catalogue(self, _btn):
        self.catalogue_ = []
        self._bump()

    # ------------------------------------------------------ tab: catalogue --
    def update_catalogue(self, cat_version, channel):
        ids = list(self.catalogue_)
        if not ids:
            print("catalogue is empty — use '+ Add Sequences' above "
                  "(switch Add mode between Current / Filtered (group))")
            return

        show = ids[:24]
        # ---- x-axis normalised to the *longest* sequence in the catalogue ----
        lengths = [len(self.sequence(sid)) for sid in ids]
        max_len = max(lengths) if lengths else 1

        ncols = 4
        nrows = int(np.ceil(len(show) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.0 * nrows),
                                 squeeze=False)

        for ax, sid in zip(axes.ravel(), show):
            g = self.sequence(sid)
            y = g[channel].to_numpy(dtype=float) if channel in g.columns else np.zeros(len(g))
            x = np.arange(len(g)) / max_len          # normalised to [0, 1]
            ax.plot(x, y, lw=0.8, color="steelblue")
            _center_zero(ax, y)
            lab = self.meta_.loc[sid, self.label_col] if self.label_col in self.meta_.columns else ""
            ax.set_title(f"{sid}\n{lab}", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.set_xlim(0, 1)
        for ax in axes.ravel()[len(show):]:
            ax.axis("off")
        fig.suptitle(
            f"Catalogue ({len(ids)} seqs, showing first {len(show)}) — "
            f"{channel} (x-axis normalised to longest = {max_len} samples)",
            fontsize=10,
        )
        fig.tight_layout()
        plt.show()

    # ------------------------------------------------------- tab: features --
    def update_features(self, seq_id, seq_text, features, **pvals):
        sid = self._resolve(seq_id, seq_text)
        g = self.sequence(sid)
        if not len(g):
            print(f"Sequence {sid} not found.")
            return
        kwargs = self._extractor_kwargs(pvals)

        features = list(features) if features else []
        # split into raw channels and extracted feature names
        raw_feats = [f for f in features if f in g.columns and not f.startswith(('vel_',))]
        # anything that is a raw channel will be plotted from g
        extracted_feats = [f for f in features if f not in raw_feats]

        # if extraction was requested for non-raw names, run extractor
        out = None
        if extracted_feats:
            try:
                ex, out = self.extract(g, kwargs, output_format="chunks")
            except Exception as e:
                print(f"[update_features] extractor failed: {e}")
                out = None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Left: all raw channels selected (or first feature)
        if raw_feats:
            for f in raw_feats:
                ax1.plot(np.arange(len(g)) / self.fs, g[f].to_numpy(dtype=float),
                         lw=1.0, label=f)
            ax1.legend(fontsize=8, loc='upper right')
            y_stack = g[raw_feats].to_numpy(dtype=float).ravel()
            _center_zero(ax1, y_stack)
        elif features and features[0] in g.columns:
            y = g[features[0]].to_numpy(dtype=float)
            ax1.plot(np.arange(len(y)) / self.fs, y, color="steelblue", lw=1.0)
            _center_zero(ax1, y)
        ax1.set_title(f"Raw signal(s) — {sid}")
        ax1.set_xlabel("Time (s)")

        # Right: extracted features
        plotted = 0
        if out is not None:
            for feat in extracted_feats:
                if feat in out.columns:
                    y_feat = out[feat].to_numpy()
                    ax2.plot(np.arange(len(y_feat)), y_feat, lw=1.4, label=feat)
                    plotted += 1
                elif feat in g.columns:
                    y_feat = g[feat].to_numpy(dtype=float)
                    ax2.plot(np.arange(len(y_feat)), y_feat, lw=1.4, label=f"{feat} (raw)")
                    plotted += 1
        if plotted == 0 and out is not None:
            # fallback: plot first few extracted columns
            cols = [c for c in out.columns if c != self.sequence_col][:5]
            for c in cols:
                ax2.plot(out[c].to_numpy(), lw=1.0, label=c)
            plotted = len(cols)
        if plotted:
            ax2.legend(fontsize=8, loc='upper right')
        ax2.set_title("Extracted features")
        ax2.set_xlabel("Frame / Chunk index")

        fig.tight_layout()
        plt.show()

    # ---------------------------------------------------- internal chains ----
    def _compute_chain(self, g, channel):
        """raw -> velocity (cumsum x dt) -> displacement -> jerk using the
        sequence's own dt / counter column."""
        y = g[channel].to_numpy(dtype=float)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        n = len(y)
        if self.counter_col in g.columns and n > 1:
            c = g[self.counter_col].to_numpy(dtype=float)
            dc = np.diff(c, prepend=c[0])
            dc[dc <= 0] = 1.0
            dt_arr = dc / self.fs
        else:
            dt_arr = np.full(n, 1.0 / self.fs)

        vel = np.cumsum(y * dt_arr)
        disp = np.cumsum(vel * dt_arr)
        if n > 1:
            jerk = np.gradient(y, dt_arr)
        else:
            jerk = np.zeros_like(y)
        return {"raw": y, "velocity": vel, "displacement": disp, "jerk": jerk}

    # ------------------------------------------------------ tab: spectrum ---
    def update_spectrum(self, seq_id, seq_text, channel, spec_mode, detrend,
                        nperseg, noverlap, spec_scale, **pvals):
        sid = self._resolve(seq_id, seq_text)
        g = self.sequence(sid)
        if not len(g):
            print(f"Sequence {sid} not found.")
            return
        if channel not in g.columns:
            print(f"Channel {channel} not found in sequence.")
            return

        y = g[channel].to_numpy(dtype=float)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        if detrend:
            y = sp_signal.detrend(y)

        # sanity clamps
        nperseg = max(8, min(int(nperseg), max(8, len(y))))
        noverlap = max(0, min(int(noverlap), nperseg - 1))

        kwargs = self._extractor_kwargs(pvals)
        cwt_wavelet = kwargs['cwt_wavelet']
        cwt_max_scale = kwargs['cwt_max_scale']
        cwt_n_scales = kwargs['cwt_n_scales']

        # STFT / CWT use the widget's overlap / nperseg directly too
        stft_nperseg = kwargs['stft_nperseg']
        stft_noverlap = kwargs['stft_noverlap']

        fig, ax = plt.subplots(figsize=(11, 5))
        t = np.arange(len(y)) / self.fs

        if spec_mode == "fft":
            f, Pxx = sp_signal.welch(y, fs=self.fs, nperseg=min(nperseg, len(y)))
            _plot_psd(ax, f, Pxx, spec_scale)
            ax.set_title(f"Welch PSD — {channel}")
        elif spec_mode == "psd":
            f, Pxx = sp_signal.periodogram(y, fs=self.fs)
            _plot_psd(ax, f, Pxx, spec_scale)
            ax.set_title(f"Periodogram — {channel}")
        elif spec_mode == "spectrogram":
            f, t_spec, Sxx = sp_signal.spectrogram(y, fs=self.fs,
                                                   nperseg=nperseg,
                                                   noverlap=noverlap)
            Sxx_db = 10 * np.log10(Sxx + 1e-12)
            ax.pcolormesh(t_spec, f, Sxx_db, shading='gouraud')
            ax.set_ylabel("Frequency (Hz)")
            ax.set_xlabel("Time (s)")
            ax.set_title(f"Spectrogram — {channel} "
                         f"(nperseg={nperseg}, overlap={noverlap})")
            plt.colorbar(ax.collections[0], ax=ax, label="dB")
        elif spec_mode == "stft":
            f, t_stft, Zxx = sp_signal.stft(y, fs=self.fs,
                                            nperseg=stft_nperseg,
                                            noverlap=stft_noverlap)
            mag = np.abs(Zxx)
            if kwargs.get('stft_use_log_scale', True):
                mag = np.log1p(mag)
            ax.pcolormesh(t_stft, f, mag, shading='gouraud')
            ax.set_ylabel("Frequency (Hz)")
            ax.set_xlabel("Time (s)")
            ax.set_title(f"STFT — {channel} "
                         f"(nperseg={stft_nperseg}, overlap={stft_noverlap})")
            plt.colorbar(ax.collections[0], ax=ax, label="log|X|" if kwargs.get('stft_use_log_scale', True) else "|X|")
        elif spec_mode == "cwt":
            if not _PYWT:
                ax.text(0.5, 0.5, "PyWavelets not installed", ha='center', va='center')
            else:
                scales = np.logspace(np.log10(1), np.log10(cwt_max_scale), cwt_n_scales)
                coeffs, freqs = pywt.cwt(y, scales, cwt_wavelet, sampling_period=1.0 / self.fs)
                mag = np.abs(coeffs)
                if kwargs.get('cwt_use_log_scale', True):
                    mag = np.log1p(mag)
                ax.pcolormesh(t, freqs, mag, shading='gouraud')
                ax.set_yscale('log')
                ax.set_ylabel("Frequency (Hz)")
                ax.set_xlabel("Time (s)")
                ax.set_title(f"CWT — {channel} "
                             f"(wavelet={cwt_wavelet}, scales={cwt_n_scales}, "
                             f"max_scale={cwt_max_scale})")
                plt.colorbar(ax.collections[0], ax=ax, label="log|W|" if kwargs.get('cwt_use_log_scale', True) else "|W|")
        plt.tight_layout()
        plt.show()

    # -------------------------------------------- tab: STFT/CWT features ----
    def update_cwt_stft(self, seq_id, seq_text, channel, method, **pvals):
        sid = self._resolve(seq_id, seq_text)
        g = self.sequence(sid)
        if not len(g) or channel not in g.columns:
            print(f"Sequence {sid} / channel {channel} not available.")
            return
        y = g[channel].to_numpy(dtype=float)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        kwargs = self._extractor_kwargs(pvals)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        t = np.arange(len(y)) / self.fs

        # ---- left: time-frequency representation ----
        if method.upper() == "STFT":
            nperseg = max(8, min(int(kwargs['stft_nperseg']), max(8, len(y))))
            noverlap = max(0, min(int(kwargs['stft_noverlap']), nperseg - 1))
            f, t_stft, Zxx = sp_signal.stft(y, fs=self.fs, nperseg=nperseg, noverlap=noverlap)
            mag = np.abs(Zxx)
            if kwargs.get('stft_use_log_scale', True):
                mag = np.log1p(mag)
            axes[0].pcolormesh(t_stft, f, mag, shading='gouraud')
            axes[0].set_title(f"STFT (nperseg={nperseg}, overlap={noverlap})")
            axes[0].set_ylabel("Frequency (Hz)")
            axes[0].set_xlabel("Time (s)")
            plt.colorbar(axes[0].collections[0], ax=axes[0])
        else:
            if not _PYWT:
                axes[0].text(0.5, 0.5, "PyWavelets not installed",
                             ha='center', va='center')
            else:
                scales = np.logspace(np.log10(1), np.log10(kwargs['cwt_max_scale']),
                                     kwargs['cwt_n_scales'])
                coeffs, freqs = pywt.cwt(y, scales, kwargs['cwt_wavelet'],
                                         sampling_period=1.0 / self.fs)
                mag = np.abs(coeffs)
                if kwargs.get('cwt_use_log_scale', True):
                    mag = np.log1p(mag)
                axes[0].pcolormesh(t, freqs, mag, shading='gouraud')
                axes[0].set_yscale('log')
                axes[0].set_title(f"CWT (wavelet={kwargs['cwt_wavelet']}, "
                                  f"n_scales={kwargs['cwt_n_scales']}, "
                                  f"max_scale={kwargs['cwt_max_scale']})")
                axes[0].set_ylabel("Frequency (Hz)")
                axes[0].set_xlabel("Time (s)")
                plt.colorbar(axes[0].collections[0], ax=axes[0])

        # ---- right: extracted feature values ----
        try:
            ex, out = self.extract(g, kwargs, output_format="frame")
            cols = [c for c in out.columns
                    if c != self.sequence_col and
                    (c.startswith('acc_') and ('stft_' in c or 'cwt_' in c))]
            cols = sorted(set(cols))[:20]
            if cols:
                vals = out.iloc[0][cols].astype(float).to_numpy()
                axes[1].barh(np.arange(len(cols)), vals)
                axes[1].set_yticks(np.arange(len(cols)))
                axes[1].set_yticklabels(cols, fontsize=7)
                axes[1].set_title(f"Extracted {method} features (top {len(cols)})")
            else:
                axes[1].text(0.5, 0.5, "No STFT/CWT features found "
                                       "(enable acc_modes)", ha='center', va='center')
        except Exception as e:
            axes[1].text(0.5, 0.5, f"Extract failed:\n{e}", ha='center', va='center')

        plt.tight_layout()
        plt.show()

    # ----------------------------------------------------- tab: chunking ---
    def update_chunk_compare(self, seq_id, channel,
                             a_maxlen, a_win, a_stride, a_smooth,
                             b_maxlen, b_win, b_stride, b_smooth):
        g = self.sequence(seq_id)
        if not len(g) or channel not in g.columns:
            print(f"Sequence {seq_id} / channel {channel} not available.")
            return
        y = g[channel].to_numpy(dtype=float)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        n = len(y)

        fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
        configs = [
            ("A", a_maxlen, a_win, a_stride, a_smooth, axes[0], "tab:orange"),
            ("B", b_maxlen, b_win, b_stride, b_smooth, axes[1], "tab:green"),
        ]
        for name, maxlen, win, stride, smooth, ax, colour in configs:
            ax.plot(np.arange(n), y, color="steelblue", lw=1.0, label="raw")
            # smooth overlay using pandas rolling
            if smooth and smooth > 1:
                smooth_y = pd.Series(y).rolling(window=smooth, center=True, min_periods=1).mean().to_numpy()
                ax.plot(np.arange(n), smooth_y, color="black", lw=1.3,
                        label=f"smooth w={smooth}")
            # maxlen cutoff
            if maxlen and maxlen < n:
                ax.axvline(x=maxlen, color="red", ls="--", lw=1.5,
                           label=f"maxlen cutoff ({maxlen})")
                ax.axvspan(maxlen, n, alpha=0.10, color="red")
            # chunk windows
            if win and win > 0:
                s = stride if stride and stride > 0 else win
                limit = min(n, maxlen) if maxlen else n
                first = True
                for start in range(0, max(1, limit - win + 1), s):
                    end = min(start + win, limit)
                    ax.axvspan(start, end, alpha=0.15, color=colour,
                               label="chunks" if first else None)
                    ax.axvline(x=start, color=colour, ls=":", alpha=0.55)
                    first = False
                if limit - win < 0:
                    ax.axvspan(0, limit, alpha=0.15, color=colour, label="pad to win")
            n_chunks = 0
            if win and win > 0:
                limit = min(n, maxlen) if maxlen else n
                if limit <= win:
                    n_chunks = 1
                else:
                    s = stride if stride and stride > 0 else win
                    starts = list(range(0, limit - win + 1, s))
                    if not starts or starts[-1] + win < limit:
                        starts.append(limit - win)
                    n_chunks = len(starts)
            ax.set_title(
                f"Config {name}: maxlen={maxlen}, chunk_win={win}, "
                f"stride={stride}, window_size={smooth}  →  {n_chunks} chunk(s)"
            )
            ax.legend(fontsize=8, loc="upper right")
            ax.set_ylabel(channel)

        axes[-1].set_xlabel("Sample index")
        fig.suptitle(f"Chunking comparison — {seq_id} — {channel} (len={n})",
                     fontsize=11)
        fig.tight_layout()
        plt.show()

    # ----------------------------------------------------- tab: clusters ---
    def update_cluster(self, reduction, method, n_clusters, eps,
                       cluster_space, population, max_seqs, color_by,
                       cat_version, **pvals):
        # --- population: filtered vs catalogue vs both ---
        filtered_ids = self.candidates(self._current_filters())
        catalogue_ids = list(self.catalogue_)

        # cap
        filtered_ids = filtered_ids[:max_seqs] if max_seqs else filtered_ids
        catalogue_ids = catalogue_ids[:max_seqs] if max_seqs else catalogue_ids

        kwargs = self._extractor_kwargs(pvals)

        def _features_for(ids):
            frames = []
            for sid in ids:
                g = self.sequence(sid)
                if not len(g):
                    continue
                try:
                    _, out = self.extract(g, kwargs, output_format="frame")
                except Exception:
                    continue
                out[self.sequence_col] = sid
                frames.append(out)
            if not frames:
                return None, None
            F = pd.concat(frames, ignore_index=True)
            feat_cols = [c for c in F.columns if c != self.sequence_col]
            X = F[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()
            return F, X

        def _reduce(X):
            if reduction == "pca":
                return PCA(n_components=2, random_state=self.random_state).fit_transform(X)
            if reduction == "tsne":
                per = max(2, min(30, len(X) - 1))
                return TSNE(n_components=2, random_state=self.random_state,
                            perplexity=per).fit_transform(X)
            if reduction == "umap" and _UMAP:
                return UMAP(n_components=2, random_state=self.random_state).fit_transform(X)
            if X.shape[1] >= 2:
                return X[:, :2]
            return np.column_stack([X[:, 0], np.zeros(len(X))])

        def _cluster(Xr):
            if method == "kmeans":
                return KMeans(n_clusters=n_clusters, random_state=self.random_state,
                              n_init=10).fit_predict(Xr)
            if method == "gmm":
                return GaussianMixture(n_components=n_clusters,
                                       random_state=self.random_state).fit_predict(Xr)
            if method == "agg":
                return AgglomerativeClustering(n_clusters=n_clusters).fit_predict(Xr)
            if method == "dbscan":
                return DBSCAN(eps=eps, min_samples=5).fit_predict(Xr)
            if method == "hdbscan" and _HDBSCAN_AVAILABLE:
                return SKHDBSCAN(min_cluster_size=max(3, n_clusters)).fit_predict(Xr)
            return np.zeros(len(Xr), dtype=int)

        def _color_array(F, labels):
            if color_by == "cluster":
                return labels
            seqs = F[self.sequence_col].unique()
            meta_sub = self.meta_.loc[seqs, color_by].astype('category')
            codes = meta_sub.cat.codes
            mapping = dict(zip(meta_sub.index, codes.values))
            return F[self.sequence_col].map(mapping).fillna(-1).astype(int).to_numpy()

        # Build panels
        panels = []  # list of (title, ids)
        if population == "both":
            panels = [
                ("Without filtering (all data)", filtered_ids if not catalogue_ids else
                 self.meta_.index.tolist()[:max_seqs]),
                (f"With filtering — {self._filter_text()}", filtered_ids),
            ]
            if catalogue_ids:
                panels[0] = ("Without filtering (all data)",
                             self.meta_.index.tolist()[:max_seqs])
                panels[1] = (f"Filtered — {self._filter_text()}", filtered_ids)
                panels.append(("Catalogue (group adds)", catalogue_ids))
        elif population == "catalogue":
            panels = [("Catalogue (group adds)", catalogue_ids)]
        else:
            panels = [(f"Filtered — {self._filter_text()}", filtered_ids)]

        n_panels = len(panels)
        if n_panels == 0 or all(len(p[1]) == 0 for p in panels):
            print("No sequences in the selected populations.")
            return

        fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 6), squeeze=False)
        axes = axes.ravel()

        for ax, (title, ids) in zip(axes, panels):
            if not ids:
                ax.axis("off")
                ax.set_title(title + " — empty")
                continue
            F, X = _features_for(ids)
            if F is None or len(X) < 3:
                ax.axis("off")
                ax.set_title(title + " — not enough data")
                continue
            Xr = _reduce(X)
            labels = _cluster(Xr)
            colors = _color_array(F, labels)
            scatter = ax.scatter(Xr[:, 0], Xr[:, 1], c=colors, cmap="tab20",
                                 s=12, alpha=0.75)
            ax.set_title(f"{title}\n{len(ids)} seqs, {F.shape[1]} feats, "
                         f"colour={color_by}")
            ax.set_xlabel(f"{reduction.upper()}1")
            ax.set_ylabel(f"{reduction.upper()}2")
            # silhouette (only if 2+ clusters and no noise-only case)
            try:
                if len(set(labels)) > 1 and len(set(labels)) < len(labels):
                    sil = silhouette_score(Xr, labels)
                    ax.text(0.02, 0.98, f"silhouette≈{sil:.3f}",
                            transform=ax.transAxes, va="top", fontsize=8)
            except Exception:
                pass

        fig.suptitle(f"Clustering — method={method}, reduction={reduction}, "
                     f"cluster_space={cluster_space}", fontsize=11)
        fig.tight_layout()
        plt.show()

        # extra diagnostic text
        print(f"Filters: {self._filter_text()}")
        print(f"Filtered size: {len(filtered_ids)} | "
              f"Catalogue size: {len(catalogue_ids)}")

    # --------------------------------------------------------- layout -----
    def show(self):
        import ipywidgets as w
        from IPython.display import display

        self.make_controls()
        p = self.p

        # filter row (scrollable)
        filter_row = w.HBox(list(self.filters_.values()),
                            layout=w.Layout(flex_flow="row wrap"))

        picker_row = w.HBox([self.seq_pick_, self.seq_text_, self.add_mode_])
        catalogue_btns = w.HBox([self.add_btn_, self.remove_one_btn_, self.clear_btn_])
        top = w.VBox([filter_row, picker_row, catalogue_btns,
                      self.cat_label_, self.cat_version_])

        param_accordion = w.Accordion(children=[
            w.VBox([w.HBox([p["acc_modes"], p["rotation_modes"]]),
                    w.HBox([p["tof_modes"], p["thm_modes"]])]),
            w.VBox([w.HBox([p["motion_filter_mode"], p["use_dead_reckoning"],
                            p["dead_reckoning_detrend"]]),
                    w.HBox([p["kalman_process_noise"], p["kalman_measurement_noise"]])]),
            w.VBox([w.HBox([p["interp_mode"], p["window_size"], p["compute_dt"]]),
                    w.HBox([p["smooth_alpha"], p["clip_value"]]),
                    w.HBox([p["maxlen"], p["padding_value"]])]),
            w.VBox([w.HBox([p["imu_native_sampling_rate"], p["imu_target_sampling_rate"]]),
                    w.HBox([p["rot_native_sampling_rate"], p["rot_target_sampling_rate"]]),
                    w.HBox([p["tof_native_sampling_rate"], p["tof_target_sampling_rate"]]),
                    w.HBox([p["thm_native_sampling_rate"], p["thm_target_sampling_rate"]]),
                    p["resample_modalities"]]),
            w.VBox([w.HBox([p["chunk_window_size"], p["chunk_stride"]]),
                    p["frame_stats"], p["add_global_context"]]),
            w.VBox([w.HBox([p["stft_nperseg"], p["stft_noverlap"], p["stft_use_log_scale"]]),
                    w.HBox([p["cwt_wavelet"], p["cwt_max_scale"], p["cwt_n_scales"]]),
                    p["cwt_use_log_scale"]]),
        ])
        for i, t in enumerate(["Modes", "Motion filter", "Cleaning & Padding",
                               "Sampling rates", "Chunking / frame stats",
                               "STFT & CWT"]):
            param_accordion.set_title(i, t)

        pvals_kw = {k: v for k, v in p.items()}

        def lazy_update(callback):
            ready = [False]

            def wrapper(**kwargs):
                if not ready[0]:
                    ready[0] = True
                    return
                return callback(**kwargs)

            return wrapper

        # ---- interactive outputs ----
        o_cat = w.interactive_output(
            lazy_update(self.update_catalogue),
            {"cat_version": self.cat_version_, "channel": self.spec_channel_},
        )
        o_feat = w.interactive_output(
            lazy_update(self.update_features),
            {"seq_id": self.seq_pick_, "seq_text": self.seq_text_,
             "features": self.feature_, **pvals_kw},
        )
        o_spec = w.interactive_output(
            lazy_update(self.update_spectrum),
            {"seq_id": self.seq_pick_, "seq_text": self.seq_text_,
             "channel": self.spec_channel_, "spec_mode": self.spec_mode_,
             "detrend": self.spec_detrend_,
             "nperseg": self.spec_nperseg_, "noverlap": self.spec_noverlap_,
             "spec_scale": self.spec_scale_,
             **pvals_kw},
        )
        o_cwt_stft = w.interactive_output(
            lazy_update(self.update_cwt_stft),
            {"seq_id": self.seq_pick_, "seq_text": self.seq_text_,
             "channel": self.cwt_stft_channel_, "method": self.cwt_stft_method_,
             **pvals_kw},
        )
        o_chunk = w.interactive_output(
            lazy_update(self.update_chunk_compare),
            {"seq_id": self.chunk_compare_seq_,
             "channel": self.chunk_compare_channel_,
             "a_maxlen": self.chunk_a_maxlen_, "a_win": self.chunk_a_win_,
             "a_stride": self.chunk_a_stride_, "a_smooth": self.chunk_a_smooth_,
             "b_maxlen": self.chunk_b_maxlen_, "b_win": self.chunk_b_win_,
             "b_stride": self.chunk_b_stride_, "b_smooth": self.chunk_b_smooth_},
        )
        def update_cluster_when_ready(_ready=[False], **kwargs):
            if _ready[0]:
                self.update_cluster(**kwargs)
            else:
                _ready[0] = True

        o_clust = w.interactive_output(
            update_cluster_when_ready,
            {"reduction": w.Dropdown(options=list(CLUSTER_REDUCTION), value="pca"),
             "method": w.Dropdown(options=list(CLUSTER_METHODS), value="kmeans"),
             "n_clusters": w.IntSlider(min=2, max=20, value=5),
             "eps": w.FloatSlider(min=0.1, max=5.0, step=0.1, value=0.5),
             "cluster_space": w.Dropdown(options=list(CLUSTER_REDUCTION), value="pca"),
             "population": self.population_,
             "max_seqs": self.max_seqs_,
             "color_by": self.color_by_,
             "cat_version": self.cat_version_,
             **pvals_kw},
        )

        tabs = w.Tab(children=[
            w.VBox([top, param_accordion, o_cat]),
            w.VBox([top, param_accordion, self.feature_, o_feat]),
            w.VBox([top, param_accordion,
                    w.HBox([self.spec_channel_, self.spec_mode_,
                            self.spec_detrend_, self.spec_scale_]),
                    w.HBox([self.spec_nperseg_, self.spec_noverlap_]),
                    o_spec]),
            w.VBox([top, param_accordion,
                    w.HBox([self.cwt_stft_channel_, self.cwt_stft_method_]),
                    o_cwt_stft]),
            w.VBox([top,
                    w.HBox([self.chunk_compare_seq_, self.chunk_compare_channel_]),
                    w.HBox([self.chunk_a_maxlen_, self.chunk_a_win_,
                            self.chunk_a_stride_, self.chunk_a_smooth_]),
                    w.HBox([self.chunk_b_maxlen_, self.chunk_b_win_,
                            self.chunk_b_stride_, self.chunk_b_smooth_]),
                    o_chunk]),
            w.VBox([top, param_accordion,
                    w.HBox([self.population_, self.max_seqs_, self.color_by_]),
                    o_clust]),
        ])
        for i, t in enumerate(["Catalogue", "Features", "Spectrum",
                               "CWT/STFT", "Chunking", "Clusters"]):
            tabs.set_title(i, t)

        display(tabs)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _plot_psd(ax, f, Pxx, scale):
    if scale == "db":
        ax.plot(f, 10 * np.log10(Pxx + 1e-12))
        ax.set_ylabel("Power (dB)")
    elif scale == "log":
        ax.semilogy(f, Pxx)
        ax.set_ylabel("PSD")
    else:
        ax.plot(f, Pxx)
        ax.set_ylabel("PSD")
    ax.set_xlabel("Frequency (Hz)")