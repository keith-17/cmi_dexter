"""
visual_utils_interactive.py
═══════════════════════════
IMU Data Explorer  ·  filter → slice → extract → analyse

Architecture
────────────
  DataFilterManager          pure data / filtering logic
  ─────────────────────────────────────────────────────
  AnalysisBase               abstract base for every analysis block
    ├─ KMeansAnalysis         elbow · silhouette · coloured scatter (2D/3D toggle)
    ├─ PCAAnalysis            scree · feature-contrib heatmap · coloured scatter (2D/3D toggle)
    ├─ DBSCANAnalysis         k-dist elbow · cluster scatter · reachability
    ├─ GMMAnalysis            BIC/AIC · confidence ellipsoids · density map
    └─ HierarchicalAnalysis   dendrogram · cluster heatmap · silhouette bars
  ─────────────────────────────────────────────────────
  UIBuilder                  widget factory (no logic)
  DataExplorerApp            controller – wires everything together

Adding a new analysis
─────────────────────
1.  Subclass AnalysisBase, set class attribute `name`, implement `run()`.
2.  Pass an instance in DataExplorerApp(analyses=[...]) or add to DEFAULT_ANALYSES.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

try:
    from utils import ImuExtractor
    _HAS_EXTRACTOR = True
except ImportError:
    _HAS_EXTRACTOR = False

_PALETTE = px.colors.qualitative.Plotly


# ══════════════════════════════════════════════════════════════════════════════
# 0.  SHARED DATA PREP
# ══════════════════════════════════════════════════════════════════════════════
def prepare_numeric(df: pd.DataFrame,
                    feature_cols: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """Scale and return (array, feature_names) from a dataframe."""
    num = df.select_dtypes(include=[np.number])
    if feature_cols:
        num = num[[c for c in feature_cols if c in num.columns]]
    num = num.dropna(axis=1, thresh=max(1, int(len(num) * 0.8)))
    num = num.fillna(num.mean())
    arr = StandardScaler().fit_transform(num.values)
    return arr, num.columns.tolist()


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA FILTER MANAGER
# ══════════════════════════════════════════════════════════════════════════════
class DataFilterManager:
    """Pure data logic: filtering, sequence resolution, summary statistics."""

    FILTER_COLS = ["sequence_type", "orientation",
                   "gesture_action", "gesture_position"]

    def __init__(self, df: pd.DataFrame, subject_df: pd.DataFrame):
        self.raw_df = df
        self.subject_df = subject_df
        self.existing_filters = [c for c in self.FILTER_COLS if c in df.columns]

    def get_candidate_sequences(self, active_filters: dict) -> np.ndarray:
        mask = pd.Series(True, index=self.raw_df.index)
        for col, vals in active_filters.items():
            if vals and col in self.raw_df.columns:
                mask &= self.raw_df[col].isin(vals)
        sub = self.raw_df.loc[mask]
        return sub["sequence_id"].unique() if "sequence_id" in sub.columns else np.array([])

    def get_rows_for_sequences(self, sequence_ids) -> pd.DataFrame:
        if len(sequence_ids) == 0:
            return self.raw_df.iloc[:0]
        return self.raw_df[self.raw_df["sequence_id"].isin(sequence_ids)]

    def summary(self, selected_seq_ids) -> dict:
        total_rows = len(self.raw_df)
        total_seqs = (self.raw_df["sequence_id"].nunique()
                      if "sequence_id" in self.raw_df.columns else 0)
        n_sel = len(selected_seq_ids)
        sel_rows = int(self.raw_df["sequence_id"].isin(selected_seq_ids).sum()) if n_sel else 0
        return dict(
            total_rows=total_rows, sel_rows=sel_rows,
            row_pct=sel_rows / total_rows * 100 if total_rows else 0,
            total_seqs=total_seqs, sel_seqs=n_sel,
            seq_pct=n_sel / total_seqs * 100 if total_seqs else 0,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 2.  ANALYSIS BASE CLASS
# ══════════════════════════════════════════════════════════════════════════════
class AnalysisBase(ABC):
    """
    Base class for all analysis blocks.

    Subclasses must define:
      name            : str   — label for the toggle checkbox
      option_widgets  : dict  — extra ipywidgets controls (can be empty)
      run(data, feature_names, options) -> None
    """

    name: str = "Analysis"

    def __init__(self):
        self.toggle = widgets.Checkbox(value=False, description=self.name, indent=False)
        self.option_widgets: dict[str, widgets.Widget] = {}

    def build_options_box(self) -> widgets.VBox | None:
        if not self.option_widgets:
            return None
        return widgets.VBox(list(self.option_widgets.values()),
                            layout=widgets.Layout(margin="0 0 0 20px"))

    def option_values(self) -> dict:
        return {k: w.value for k, w in self.option_widgets.items()}

    @abstractmethod
    def run(self, data: np.ndarray, feature_names: list[str], options: dict) -> None: ...

    # ── shared utilities ──────────────────────────────────────────────────────
    @staticmethod
    def _pca_proj(data: np.ndarray, n: int = 2) -> np.ndarray:
        n = min(n, data.shape[1])
        return PCA(n_components=n).fit_transform(data)

    @staticmethod
    def _colour(label: int) -> str:
        return _PALETTE[int(label) % len(_PALETTE)]

    @staticmethod
    def _guard(data: np.ndarray, min_s: int = 3, min_f: int = 2) -> bool:
        if data.shape[0] < min_s or data.shape[1] < min_f:
            print(f"⚠️  Need ≥{min_s} samples and ≥{min_f} features.")
            return False
        return True

    @staticmethod
    def _scatter_spec(use_3d: bool) -> dict:
        return {"type": "scatter3d"} if use_3d else {"type": "scatter"}

    def _add_cluster_scatter(self, fig, proj, labels, row, col, use_3d):
        """Add one Scatter/Scatter3d trace per cluster label."""
        for cid in np.unique(labels):
            m = labels == cid
            nm = "Noise" if cid == -1 else f"Cluster {cid}"
            colour = "lightgrey" if cid == -1 else self._colour(cid)
            if use_3d:
                z = proj[m, 2] if proj.shape[1] > 2 else np.zeros(m.sum())
                fig.add_trace(go.Scatter3d(
                    x=proj[m, 0], y=proj[m, 1], z=z, mode="markers",
                    name=nm, marker=dict(size=4, color=colour, opacity=0.8)),
                    row=row, col=col)
            else:
                fig.add_trace(go.Scatter(
                    x=proj[m, 0], y=proj[m, 1], mode="markers",
                    name=nm, marker=dict(size=6, color=colour, opacity=0.8)),
                    row=row, col=col)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  CONCRETE ANALYSIS CLASSES
# ══════════════════════════════════════════════════════════════════════════════

# ── 3a. K-Means ──────────────────────────────────────────────────────────────
class KMeansAnalysis(AnalysisBase):
    name = "K-Means Clustering"

    def __init__(self):
        super().__init__()
        self.option_widgets = {
            "max_k": widgets.BoundedIntText(
                value=10, min=2, max=30, description="Max K:",
                layout={"width": "160px"}),
            "scatter_dim": widgets.ToggleButtons(
                options=["2D", "3D"], description="Scatter:",
                style={"description_width": "initial"}),
        }

    def run(self, data, feature_names, options):
        if not self._guard(data):
            return
        max_k = min(options["max_k"], data.shape[0] - 1)
        if max_k < 2:
            print("⚠️  Not enough samples for K-Means (need ≥3).")
            return

        inertias, sils = [], []
        for k in range(2, max_k + 1):
            labs = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(data)
            inertias.append(KMeans(n_clusters=k, random_state=42, n_init=10).fit(data).inertia_)
            sils.append(silhouette_score(data, labs))

        k_range = list(range(2, max_k + 1))
        optimal_k = int(np.argmax(sils)) + 2
        labels = KMeans(n_clusters=optimal_k, random_state=42, n_init=10).fit_predict(data)
        print(f"   optimal K={optimal_k}  silhouette={sils[optimal_k-2]:.3f}")

        use_3d = options["scatter_dim"] == "3D"
        scatter_title = f"Coloured Clusters (PCA {'3D' if use_3d else '2D'})"
        specs = [[{"type": "scatter"}, {"type": "scatter"}, self._scatter_spec(use_3d)]]
        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=("Inertia vs K", "Silhouette vs K", scatter_title),
                            specs=specs)

        fig.add_trace(go.Scatter(x=k_range, y=inertias, mode="lines+markers",
                                 name="Inertia", line=dict(color="steelblue")), row=1, col=1)
        fig.add_vline(x=optimal_k, line_dash="dash", line_color="red", row=1, col=1)
        fig.update_xaxes(title_text="K", row=1, col=1)
        fig.update_yaxes(title_text="Inertia", row=1, col=1)

        fig.add_trace(go.Scatter(x=k_range, y=sils, mode="lines+markers",
                                 name="Silhouette", line=dict(color="tomato")), row=1, col=2)
        fig.add_vline(x=optimal_k, line_dash="dash", line_color="red", row=1, col=2)
        fig.update_xaxes(title_text="K", row=1, col=2)
        fig.update_yaxes(title_text="Silhouette", row=1, col=2)

        proj = self._pca_proj(data, 3 if use_3d else 2)
        self._add_cluster_scatter(fig, proj, labels, row=1, col=3, use_3d=use_3d)

        fig.update_layout(height=480, title_text=f"K-Means  (K={optimal_k})", showlegend=True)
        fig.show()


# ── 3b. PCA ──────────────────────────────────────────────────────────────────
class PCAAnalysis(AnalysisBase):
    name = "PCA"

    def __init__(self):
        super().__init__()
        self.option_widgets = {
            "scatter_dim": widgets.ToggleButtons(
                options=["2D", "3D"], description="Scatter:",
                style={"description_width": "initial"}),
            "n_top_features": widgets.BoundedIntText(
                value=10, min=2, max=50, description="Top features:",
                layout={"width": "185px"}),
        }

    def run(self, data, feature_names, options):
        if not self._guard(data):
            return

        n_comp = min(10, data.shape[1])
        pca = PCA(n_components=n_comp).fit(data)
        proj = pca.transform(data)
        ev = pca.explained_variance_ratio_
        cum = np.cumsum(ev)

        n_feat = min(options["n_top_features"], len(feature_names))
        n_pcs_heat = min(5, pca.components_.shape[0])
        contrib = pca.components_[:n_pcs_heat, :n_feat]

        use_3d = options["scatter_dim"] == "3D"
        scatter_title = f"Projection (PCA {'3D' if use_3d else '2D'}, coloured by PC1)"
        specs = [[{"type": "bar"}, {"type": "heatmap"}, self._scatter_spec(use_3d)]]
        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=("Scree Plot", "Feature Contributions", scatter_title),
                            specs=specs)

        # Scree
        fig.add_trace(go.Bar(x=list(range(1, len(ev)+1)), y=ev,
                             name="Variance", marker_color="steelblue"), row=1, col=1)
        fig.add_trace(go.Scatter(x=list(range(1, len(cum)+1)), y=cum, mode="lines+markers",
                                 name="Cumulative", line=dict(color="tomato", dash="dot")),
                      row=1, col=1)
        fig.update_xaxes(title_text="PC", row=1, col=1)
        fig.update_yaxes(title_text="Explained Variance", row=1, col=1)

        # Feature contribution heatmap
        fig.add_trace(go.Heatmap(
            z=contrib, x=feature_names[:n_feat],
            y=[f"PC{i+1}" for i in range(n_pcs_heat)],
            colorscale="RdBu", zmid=0,
            text=[[f"{v:.2f}" for v in row] for row in contrib],
            texttemplate="%{text}", textfont={"size": 8}, showscale=True,
        ), row=1, col=2)

        # Coloured scatter (colour = PC1 value, continuous gradient)
        colours = proj[:, 0]
        marker_base = dict(color=colours, colorscale="Viridis", opacity=0.8,
                           showscale=True, colorbar=dict(title="PC1"))
        if use_3d:
            z = proj[:, 2] if proj.shape[1] > 2 else np.zeros(len(proj))
            fig.add_trace(go.Scatter3d(
                x=proj[:, 0], y=proj[:, 1], z=z, mode="markers",
                marker=dict(size=4, **marker_base), name="PC1 score"), row=1, col=3)
        else:
            fig.add_trace(go.Scatter(
                x=proj[:, 0], y=proj[:, 1], mode="markers",
                marker=dict(size=6, **marker_base), name="PC1 score"), row=1, col=3)
            fig.update_xaxes(title_text=f"PC1 ({ev[0]:.1%})", row=1, col=3)
            fig.update_yaxes(title_text=f"PC2 ({ev[1]:.1%})", row=1, col=3)

        fig.update_layout(height=480, title_text="Principal Component Analysis")
        fig.show()

        print("   Explained variance (first 5 PCs):")
        for i, v in enumerate(ev[:5], 1):
            print(f"      PC{i}: {v:.3f}  ({v*100:.1f}%)  cumulative: {cum[i-1]*100:.1f}%")


# ── 3c. DBSCAN ───────────────────────────────────────────────────────────────
class DBSCANAnalysis(AnalysisBase):
    name = "DBSCAN"

    def __init__(self):
        super().__init__()
        self.option_widgets = {
            "eps": widgets.FloatText(value=0.5, description="eps:",
                                     layout={"width": "150px"}),
            "min_samples": widgets.BoundedIntText(value=5, min=1, max=50,
                                                   description="min_samples:",
                                                   layout={"width": "165px"}),
            "scatter_dim": widgets.ToggleButtons(
                options=["2D", "3D"], description="Scatter:",
                style={"description_width": "initial"}),
        }

    def run(self, data, feature_names, options):
        if not self._guard(data):
            return

        k = max(1, options["min_samples"] - 1)
        nbrs = NearestNeighbors(n_neighbors=k).fit(data)
        dists, _ = nbrs.kneighbors(data)
        k_dists = np.sort(dists[:, -1])[::-1]

        labels = DBSCAN(eps=options["eps"],
                        min_samples=options["min_samples"]).fit_predict(data)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        print(f"   eps={options['eps']}, min_samples={options['min_samples']} "
              f"→ {n_clusters} clusters, {n_noise} noise points")

        use_3d = options["scatter_dim"] == "3D"
        scatter_title = f"Clusters (PCA {'3D' if use_3d else '2D'})"
        specs = [[{"type": "scatter"}, self._scatter_spec(use_3d), {"type": "bar"}]]
        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=(f"k-Dist Elbow (k={k})",
                                            scatter_title, "Reachability Plot"),
                            specs=specs)

        # k-dist elbow
        fig.add_trace(go.Scatter(x=list(range(len(k_dists))), y=k_dists,
                                 mode="lines", name="k-dist",
                                 line=dict(color="steelblue")), row=1, col=1)
        fig.add_hline(y=options["eps"], line_dash="dash", line_color="red", row=1, col=1)
        fig.update_xaxes(title_text="Points (sorted desc)", row=1, col=1)
        fig.update_yaxes(title_text="k-th NN distance", row=1, col=1)

        # Coloured cluster scatter
        proj = self._pca_proj(data, 3 if use_3d else 2)
        self._add_cluster_scatter(fig, proj, labels, row=1, col=2, use_3d=use_3d)

        # Reachability (bar chart of sorted k-dists, coloured by cluster)
        sort_idx = np.argsort(k_dists)[::-1]
        reach_labels = labels[sort_idx] if len(labels) == len(k_dists) else np.zeros(len(k_dists))
        lbl_colours = {lab: ("lightgrey" if lab == -1 else self._colour(lab))
                       for lab in np.unique(labels)}
        for lab in np.unique(labels):
            m = reach_labels == lab
            nm = "Noise" if lab == -1 else f"Cluster {lab}"
            idxs = np.where(m)[0]
            fig.add_trace(go.Bar(x=idxs.tolist(), y=k_dists[idxs].tolist(),
                                 name=nm, marker_color=lbl_colours[lab]), row=1, col=3)
        fig.update_xaxes(title_text="Sample order", row=1, col=3)
        fig.update_yaxes(title_text="Reachability distance", row=1, col=3)

        fig.update_layout(height=480, title_text="DBSCAN Analysis", barmode="overlay")
        fig.show()


# ── 3d. Gaussian Mixture Model ────────────────────────────────────────────────
class GMMAnalysis(AnalysisBase):
    name = "Gaussian Mixture Model"

    def __init__(self):
        super().__init__()
        self.option_widgets = {
            "max_components": widgets.BoundedIntText(
                value=8, min=2, max=20, description="Max components:",
                layout={"width": "200px"}),
            "covariance_type": widgets.Dropdown(
                options=["full", "tied", "diag", "spherical"],
                value="full", description="Cov type:", layout={"width": "200px"}),
        }

    def run(self, data, feature_names, options):
        if not self._guard(data):
            return

        max_c = min(options["max_components"], data.shape[0] - 1)
        cov = options["covariance_type"]
        comp_range = list(range(1, max_c + 1))
        bics, aics = [], []
        for n in comp_range:
            gm = GaussianMixture(n_components=n, covariance_type=cov,
                                 random_state=42, n_init=3).fit(data)
            bics.append(gm.bic(data))
            aics.append(gm.aic(data))

        best_n = comp_range[int(np.argmin(bics))]
        gmm = GaussianMixture(n_components=best_n, covariance_type=cov,
                              random_state=42, n_init=3).fit(data)
        labels = gmm.predict(data)
        print(f"   best n_components={best_n} (BIC),  covariance={cov}")

        # PCA 2D projection for visualisation
        pca2 = PCA(n_components=min(2, data.shape[1])).fit(data)
        proj = pca2.transform(data)

        specs = [[{"type": "scatter"}, {"type": "scatter"}, {"type": "heatmap"}]]
        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=("BIC / AIC vs Components",
                                            "Confidence Ellipsoids (PCA 2D)",
                                            "Probability Density Map"),
                            specs=specs)

        # BIC / AIC
        fig.add_trace(go.Scatter(x=comp_range, y=bics, mode="lines+markers",
                                 name="BIC", line=dict(color="steelblue")), row=1, col=1)
        fig.add_trace(go.Scatter(x=comp_range, y=aics, mode="lines+markers",
                                 name="AIC", line=dict(color="tomato")), row=1, col=1)
        fig.add_vline(x=best_n, line_dash="dash", line_color="green", row=1, col=1)
        fig.update_xaxes(title_text="Components", row=1, col=1)

        # Scatter + ellipsoids
        for cid in range(best_n):
            m = labels == cid
            fig.add_trace(go.Scatter(
                x=proj[m, 0], y=proj[m, 1], mode="markers",
                name=f"Comp {cid}",
                marker=dict(size=6, color=self._colour(cid), opacity=0.7)), row=1, col=2)

        means_2d = pca2.transform(gmm.means_)
        for cid in range(best_n):
            try:
                if cov == "full":
                    c2 = pca2.components_ @ gmm.covariances_[cid] @ pca2.components_.T
                elif cov == "tied":
                    c2 = pca2.components_ @ gmm.covariances_ @ pca2.components_.T
                elif cov == "diag":
                    c2 = np.diag(pca2.components_ @ np.diag(gmm.covariances_[cid]) @ pca2.components_.T)
                    c2 = np.diag(c2)
                else:
                    c2 = np.eye(2) * gmm.covariances_[cid]
                vals, vecs = np.linalg.eigh(c2[:2, :2])
                order = vals.argsort()[::-1]
                vals, vecs = vals[order], vecs[:, order]
                t = np.linspace(0, 2 * np.pi, 100)
                ell = 2 * np.sqrt(np.abs(vals))
                xe = means_2d[cid, 0] + ell[0]*np.cos(t)*vecs[0,0] + ell[1]*np.sin(t)*vecs[0,1]
                ye = means_2d[cid, 1] + ell[0]*np.cos(t)*vecs[1,0] + ell[1]*np.sin(t)*vecs[1,1]
                fig.add_trace(go.Scatter(x=xe, y=ye, mode="lines",
                                         line=dict(color=self._colour(cid), width=2, dash="dot"),
                                         showlegend=False), row=1, col=2)
            except Exception:
                pass

        # Density heatmap (fit GMM on 2D projection for approx density)
        x0, x1 = proj[:, 0].min() - 0.5, proj[:, 0].max() + 0.5
        y0, y1 = proj[:, 1].min() - 0.5, proj[:, 1].max() + 0.5
        gx, gy = np.meshgrid(np.linspace(x0, x1, 60), np.linspace(y0, y1, 60))
        gmm2 = GaussianMixture(n_components=best_n, covariance_type="full",
                                random_state=42).fit(proj)
        Z = np.exp(gmm2.score_samples(np.c_[gx.ravel(), gy.ravel()])).reshape(gx.shape)
        fig.add_trace(go.Heatmap(x=np.linspace(x0, x1, 60), y=np.linspace(y0, y1, 60),
                                  z=Z, colorscale="Viridis", showscale=True,
                                  name="Density"), row=1, col=3)
        fig.add_trace(go.Scatter(x=proj[:, 0], y=proj[:, 1], mode="markers",
                                  marker=dict(size=4, color="white", opacity=0.4),
                                  showlegend=False), row=1, col=3)

        fig.update_layout(height=480, title_text="Gaussian Mixture Model Analysis")
        fig.show()


# ── 3e. Hierarchical Clustering ───────────────────────────────────────────────
class HierarchicalAnalysis(AnalysisBase):
    name = "Hierarchical Clustering"

    def __init__(self):
        super().__init__()
        self.option_widgets = {
            "n_clusters": widgets.BoundedIntText(
                value=3, min=2, max=20, description="n clusters:",
                layout={"width": "170px"}),
            "linkage_method": widgets.Dropdown(
                options=["ward", "complete", "average", "single"],
                value="ward", description="Linkage:", layout={"width": "190px"}),
            "max_dendrogram_leaves": widgets.BoundedIntText(
                value=50, min=10, max=200, description="Dendro leaves:",
                layout={"width": "190px"}),
        }

    def run(self, data, feature_names, options):
        if not self._guard(data):
            return

        method = options["linkage_method"]
        n_clusters = min(options["n_clusters"], data.shape[0] - 1)
        max_leaves = options["max_dendrogram_leaves"]

        labels = AgglomerativeClustering(
            n_clusters=n_clusters, linkage=method).fit_predict(data)
        sil_vals = silhouette_samples(data, labels)
        overall_sil = silhouette_score(data, labels)
        print(f"   n_clusters={n_clusters}, linkage={method}, "
              f"silhouette={overall_sil:.3f}")

        specs = [[{"type": "scatter"}, {"type": "heatmap"}, {"type": "bar"}]]
        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=(f"Dendrogram ({method})",
                                            "Cluster Feature Heatmap",
                                            "Silhouette Plot"),
                            specs=specs)

        # Dendrogram (sub-sample if large)
        sub = (data if len(data) <= max_leaves
               else data[np.random.default_rng(42).choice(len(data), max_leaves, replace=False)])
        Z_sub = linkage(sub, method=method)
        dend = dendrogram(Z_sub, no_plot=True)
        for xs, ys in zip(dend["icoord"], dend["dcoord"]):
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                      line=dict(color="steelblue", width=1),
                                      showlegend=False), row=1, col=1)
        fig.update_xaxes(title_text="Sample", showticklabels=False, row=1, col=1)
        fig.update_yaxes(title_text="Distance", row=1, col=1)

        # Cluster feature heatmap (mean per cluster, top 20 features)
        feat_names_safe = (feature_names if feature_names
                           else [f"f{i}" for i in range(data.shape[1])])
        df_tmp = pd.DataFrame(data, columns=feat_names_safe[:data.shape[1]])
        df_tmp["_cluster"] = labels
        cm = df_tmp.groupby("_cluster").mean().iloc[:, :20]
        fig.add_trace(go.Heatmap(
            z=cm.values, x=cm.columns.tolist(),
            y=[f"Cluster {i}" for i in cm.index],
            colorscale="RdBu", zmid=0,
            text=[[f"{v:.2f}" for v in row] for row in cm.values],
            texttemplate="%{text}", textfont={"size": 8}, showscale=True,
        ), row=1, col=2)

        # Silhouette bars (sorted within each cluster, horizontal)
        y_lower = 0
        for cid in sorted(set(labels)):
            sil_cid = np.sort(sil_vals[labels == cid])
            size = len(sil_cid)
            fig.add_trace(go.Bar(
                x=sil_cid, y=list(range(y_lower, y_lower + size)),
                orientation="h", name=f"Cluster {cid}",
                marker_color=self._colour(cid), showlegend=True,
            ), row=1, col=3)
            y_lower += size + 2
        fig.add_vline(x=overall_sil, line_dash="dash", line_color="red", row=1, col=3)
        fig.update_xaxes(title_text="Silhouette coefficient", row=1, col=3)

        fig.update_layout(
            height=480,
            title_text=f"Hierarchical Clustering  (n={n_clusters}, {method})",
            barmode="overlay",
        )
        fig.show()


# ══════════════════════════════════════════════════════════════════════════════
# 4.  UI BUILDER
# ══════════════════════════════════════════════════════════════════════════════
class UIBuilder:
    """Widget factory — zero business logic."""

    BAND_EDGE_OPTIONS: dict = {
        "None": None,
        "Linear (0-50, step 10)": np.arange(0, 51, 10),
        "Log (0.5-50, 10 pts)": np.logspace(np.log10(0.5), np.log10(50), num=10),
    }

    def __init__(self, df: pd.DataFrame, filter_cols: list[str],
                 analyses: list[AnalysisBase]):
        self.cat_selectors = self._build_cat_selectors(df, filter_cols)

        all_seqs = sorted(df["sequence_id"].unique()) if "sequence_id" in df.columns else []
        self.seq_selector = widgets.SelectMultiple(
            options=all_seqs, value=(), description="Sequence ID",
            layout={"width": "320px", "height": "150px"})
        self.seq_slider = widgets.IntSlider(
            value=0, min=0, max=len(all_seqs), step=1,
            description="Sample N seqs:", layout={"width": "420px"})

        self.tof_mapping, self.tof_checks = self._build_tof_checks(df)
        self.sensor_groups = self._build_sensor_groups(df)
        self.hyperparams = self._build_hyperparams()
        self.analyses = analyses

        self.record_counter = widgets.HTML()
        self.run_btn = widgets.Button(description="▶ RUN EXTRACTION",
                                      button_style="success", layout={"width": "210px"})
        self.reset_btn = widgets.Button(description="🔄 RESET",
                                        button_style="warning", layout={"width": "120px"})
        self.output = widgets.Output()

    def _build_cat_selectors(self, df, cols):
        heights = {"sequence_type": "55px", "orientation": "110px"}
        return {c: widgets.SelectMultiple(
            options=sorted(df[c].dropna().unique()), value=(),
            description=c.replace("_", " ").title(),
            layout={"width": "260px", "height": heights.get(c, "130px")},
        ) for c in cols}

    def _build_tof_checks(self, df):
        mapping = {f"tof_{i}": [c for c in df.columns if c.startswith(f"tof_{i}_")]
                   for i in range(1, 6)}
        checks = {lbl: widgets.Checkbox(value=True, description=lbl, indent=False)
                  for lbl in mapping}
        return mapping, checks

    def _build_sensor_groups(self, df):
        def make(prefix):
            return {c: widgets.Checkbox(value=True, description=c, indent=False,
                                        layout={"width": "auto"})
                    for c in df.columns if c.startswith(prefix)}
        return {"imu_sensor_list":        make("acc"),
                "rotation_sensor_list":   make("rot"),
                "thermopile_sensor_list": make("thm")}

    def _build_hyperparams(self):
        return {
            "sampling_rate":         widgets.IntText(value=100, description="Sample Rate"),
            "window":                widgets.FloatText(value=2.0, description="Window (s)"),
            "step_sec":              widgets.FloatText(value=0.37, description="Step (s)"),
            "imu_domain":            widgets.Dropdown(
                options=["time", "acceleration", "velocity", "displacement"],
                value="acceleration", description="IMU Domain"),
            "rotation_domain":       widgets.Dropdown(
                options=["time", "acceleration"], value="acceleration",
                description="Rot Domain"),
            "combine_imu_axes":      widgets.Checkbox(value=True, description="Combine IMU"),
            "combine_rotation_axes": widgets.Checkbox(value=True, description="Combine Rot"),
            "dc_offset":             widgets.FloatText(value=0.5, description="DC Offset"),
            "band_edges":            widgets.Dropdown(
                options=list(self.BAND_EDGE_OPTIONS.keys()), value="None",
                description="Band Edges"),
            "disable_tqdm":          widgets.Checkbox(value=True, description="Disable tqdm"),
            "category_data":         widgets.Dropdown(options=[True, False], value=True,
                                                       description="Category Data"),
            "segmentation":          widgets.Dropdown(options=[None, "window", "phase"],
                                                       value=None, description="Segmentation"),
            "tof_mode":              widgets.Dropdown(options=[None, "research"],
                                                       value="research", description="TOF Mode"),
        }

    def selected_tof_cols(self) -> list[str]:
        return [c for lbl, box in self.tof_checks.items()
                if box.value for c in self.tof_mapping[lbl]]

    def selected_sensors(self, group: str) -> list[str] | None:
        sel = [n for n, box in self.sensor_groups[group].items() if box.value]
        return sel or None


# ══════════════════════════════════════════════════════════════════════════════
# 5.  MAIN APPLICATION CONTROLLER
# ══════════════════════════════════════════════════════════════════════════════
class DataExplorerApp:
    """
    Controller — wires DataFilterManager, UIBuilder and analysis objects.

    Usage
    ─────
        app = DataExplorerApp(clean_df, train_demo_df)
        app.show()

    Custom analyses
    ───────────────
        app = DataExplorerApp(clean_df, demo_df, analyses=[KMeansAnalysis(), MyCustomAnalysis()])
    """

    def __init__(self, df: pd.DataFrame, subject_df: pd.DataFrame,
                 analyses: list[AnalysisBase] | None = None):
        self.dm = DataFilterManager(df, subject_df)
        self._analyses: list[AnalysisBase] = analyses if analyses is not None else [
            KMeansAnalysis(),
            PCAAnalysis(),
            DBSCANAnalysis(),
            GMMAnalysis(),
            HierarchicalAnalysis(),
        ]
        self.ui = UIBuilder(df, self.dm.existing_filters, self._analyses)
        self._bind()
        self._refresh()

    # ── event binding ─────────────────────────────────────────────────────────
    def _bind(self):
        for sel in self.ui.cat_selectors.values():
            sel.observe(self._on_cat_change, names="value")
        self.ui.seq_slider.observe(self._on_slider_change, names="value")
        self.ui.run_btn.on_click(self._on_run)
        self.ui.reset_btn.on_click(self._on_reset)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _active_filters(self) -> dict:
        return {col: list(sel.value) for col, sel in self.ui.cat_selectors.items()}

    def _resolve_sequences(self) -> np.ndarray:
        candidates = self.dm.get_candidate_sequences(self._active_filters())
        explicit = list(self.ui.seq_selector.value)
        if explicit:
            cset = set(candidates)
            candidates = np.array([s for s in explicit if s in cset])
        return candidates

    def _refresh(self):
        candidates = self.dm.get_candidate_sequences(self._active_filters())
        stats = self.dm.summary(candidates)
        c = ("#28a745" if stats["seq_pct"] >= 75
             else "#ffc107" if stats["seq_pct"] >= 25 else "#dc3545")
        self.ui.record_counter.value = (
            f'<div style="border:2px solid #dee2e6;border-radius:8px;padding:10px;'
            f'background:#f8f9fa;font-family:monospace;font-size:13px;">'
            f'<b>📊 ROWS:</b> <b style="color:{c};">{stats["sel_rows"]:,}</b> / '
            f'{stats["total_rows"]:,} ({stats["row_pct"]:.1f}%) &nbsp;|&nbsp;'
            f'<b>🔢 SEQS:</b> <b style="color:{c};">{stats["sel_seqs"]:,}</b> / '
            f'{stats["total_seqs"]:,} ({stats["seq_pct"]:.1f}%)'
            f'<span style="color:#6c757d;font-size:11px;margin-left:12px;">'
            f'active filters: {sum(1 for v in self._active_filters().values() if v)}'
            f'</span></div>'
        )
        self.ui.seq_slider.max = max(1, len(candidates))
        current = set(self.ui.seq_selector.value)
        self.ui.seq_selector.options = sorted(candidates)
        self.ui.seq_selector.value = tuple(s for s in current if s in set(candidates))

    # ── callbacks ─────────────────────────────────────────────────────────────
    def _on_cat_change(self, _):
        active = self._active_filters()
        mask = pd.Series(True, index=self.dm.raw_df.index)
        for col, vals in active.items():
            if vals and col in self.dm.raw_df.columns:
                mask &= self.dm.raw_df[col].isin(vals)
        narrowed = self.dm.raw_df.loc[mask]
        for col, sel in self.ui.cat_selectors.items():
            if col in narrowed.columns:
                available = sorted(narrowed[col].dropna().unique())
                kept = tuple(v for v in sel.value if v in available)
                sel.unobserve(self._on_cat_change, names="value")
                sel.options = available
                sel.value = kept
                sel.observe(self._on_cat_change, names="value")
        self._refresh()

    def _on_slider_change(self, change):
        n = change["new"]
        candidates = self.dm.get_candidate_sequences(self._active_filters())
        if n <= 0 or len(candidates) == 0:
            self.ui.seq_selector.value = ()
            return
        chosen = np.random.default_rng(seed=n).choice(
            candidates, size=min(n, len(candidates)), replace=False)
        self.ui.seq_selector.options = sorted(candidates)
        self.ui.seq_selector.value = tuple(sorted(chosen))

    def _on_reset(self, _):
        for col, sel in self.ui.cat_selectors.items():
            sel.unobserve(self._on_cat_change, names="value")
            if col in self.dm.raw_df.columns:
                sel.options = sorted(self.dm.raw_df[col].dropna().unique())
            sel.value = ()
            sel.observe(self._on_cat_change, names="value")
        self.ui.seq_selector.value = ()
        self.ui.seq_slider.value = 0
        self._refresh()
        with self.ui.output:
            clear_output(wait=True)
            print("✅ All filters reset.")

    def _on_run(self, _):
        with self.ui.output:
            clear_output(wait=True)
            seq_ids = self._resolve_sequences()
            if len(seq_ids) == 0:
                print("⚠️  No sequences match current filters.")
                return

            extract_df = self.dm.get_rows_for_sequences(seq_ids)
            print(f"📦  {len(seq_ids)} sequences → {len(extract_df):,} rows")

            if not _HAS_EXTRACTOR:
                print("⚠️  utils.ImuExtractor not found — using raw filtered data.")
                result_df = extract_df
            else:
                extractor = ImuExtractor(
                    imu_sensor_list=          self.ui.selected_sensors("imu_sensor_list"),
                    rotation_sensor_list=     self.ui.selected_sensors("rotation_sensor_list"),
                    thermopile_sensor_list=   self.ui.selected_sensors("thermopile_sensor_list"),
                    sampling_rate=            self.ui.hyperparams["sampling_rate"].value,
                    imu_domain=               self.ui.hyperparams["imu_domain"].value,
                    rotation_domain=          self.ui.hyperparams["rotation_domain"].value,
                    dc_offset=                self.ui.hyperparams["dc_offset"].value,
                    band_edges=               self.ui.BAND_EDGE_OPTIONS[
                                                  self.ui.hyperparams["band_edges"].value],
                    subject_df=               self.dm.subject_df,
                    disable_tqdm=             self.ui.hyperparams["disable_tqdm"].value,
                    category_data=            self.ui.hyperparams["category_data"].value,
                    segmentation=             self.ui.hyperparams["segmentation"].value,
                    window=                   self.ui.hyperparams["window"].value,
                    step_sec=                 self.ui.hyperparams["step_sec"].value,
                    combine_imu_axes=         self.ui.hyperparams["combine_imu_axes"].value,
                    combine_rot_axes=         self.ui.hyperparams["combine_rotation_axes"].value,
                    tof_sensor_list=          self.ui.selected_tof_cols() or None,
                    tof_mode=                 self.ui.hyperparams["tof_mode"].value,
                )
                print("🚀 Extracting features…")
                try:
                    result_df = extractor.fit_transform(extract_df)
                except Exception as exc:
                    print(f"❌  Extractor error: {exc}")
                    raise
                print(f"✅  Extraction done — shape: {result_df.shape}")

            # ── analyses ────────────────────────────────────────────────────
            active = [a for a in self._analyses if a.toggle.value]
            if not active:
                return
            data, feat_names = prepare_numeric(result_df)
            print(f"\n🔬 {len(active)} analysis block(s) · "
                  f"{data.shape[0]} samples × {data.shape[1]} features\n" + "─"*60)
            for analysis in active:
                print(f"\n▶ {analysis.name}")
                try:
                    analysis.run(data, feat_names, analysis.option_values())
                except Exception as exc:
                    print(f"  ❌  {analysis.name} failed: {exc}")
            print("\n✅ All analyses complete.")

    # ── layout ────────────────────────────────────────────────────────────────
    def show(self):
        def panel(title, *children, scroll=False):
            lo = {"border": "1px solid #dee2e6", "padding": "10px",
                  "margin": "4px", "border_radius": "6px"}
            if scroll:
                lo.update({"max_height": "380px", "overflow_y": "auto"})
            return widgets.VBox(
                [widgets.HTML(f"<b style='font-size:13px'>{title}</b>")] + list(children),
                layout=widgets.Layout(**lo))

        filter_panel = panel("🔍 Categorical Filters",
                              *self.ui.cat_selectors.values(), scroll=True)

        seq_panel = panel(
            "🔢 Sequences",
            widgets.HTML("<i style='font-size:11px'>Slider = random sample N<br>"
                         "Multi-select = manual pick</i>"),
            self.ui.seq_slider)

        sensor_items = []
        for grp, checks in self.ui.sensor_groups.items():
            sensor_items.append(widgets.HTML(
                f"<u>{grp.replace('_list','').replace('_',' ').title()}</u>"))
            sensor_items.extend(checks.values())
        sensor_panel = panel("📡 Sensors", *sensor_items, scroll=True)

        tof_panel = panel("📏 TOF", *self.ui.tof_checks.values())

        # Each analysis gets its toggle + inline options
        analysis_rows = []
        for a in self._analyses:
            row_items: list[widgets.Widget] = [a.toggle]
            opts = a.build_options_box()
            if opts:
                row_items.append(opts)
            analysis_rows.append(widgets.VBox(
                row_items,
                layout=widgets.Layout(border="1px solid #eee", padding="6px",
                                      margin="3px", border_radius="4px")))
        analysis_panel = panel("📊 Analyses", *analysis_rows, scroll=False)

        param_panel = panel("⚙️ Params", *self.ui.hyperparams.values(), scroll=True)

        top_row = widgets.HBox(
            [filter_panel, seq_panel, sensor_panel, tof_panel, analysis_panel, param_panel],
            layout=widgets.Layout(flex_flow="row wrap"))

        display(widgets.HTML("<h2 style='margin:8px 0'>🎮 IMU Data Explorer</h2>"))
        display(self.ui.record_counter)
        display(top_row)
        display(widgets.HBox([self.ui.run_btn, self.ui.reset_btn],
                              layout=widgets.Layout(margin="8px 0")))
        display(self.ui.output)
