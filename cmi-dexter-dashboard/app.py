"""CMI Dexter experiment dashboard.

Run with: python -m streamlit run app.py
Keep the interface procedural; reusable experiment logic lives in dashboard_utils.
"""
import json
from pprint import pformat
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import dashboard_utils as data_utils


# ============================================================
# Configuration and appearance
# ============================================================
st.set_page_config(page_title="CMI Dexter | Experiments", page_icon="◈", layout="wide")

st.markdown("""
<style>
:root {color-scheme:dark;}
.stApp {background:radial-gradient(ellipse at 85% 0%,#132333 0%,#0b1017 44%);}
.block-container {max-width:1520px;padding-top:2.2rem;padding-bottom:3rem;}
[data-testid="stSidebar"] {background:#0e1620;border-right:1px solid #20303f;}
[data-testid="stHeader"] {background:transparent;}
h1 {font-weight:680!important;letter-spacing:-1.7px!important;font-size:2.7rem!important;}
h2,h3 {letter-spacing:-.4px;}
.eyebrow {color:#65e6bb;font-size:.72rem;font-weight:700;letter-spacing:2px;margin-bottom:6px;}
.subheading {color:#9aaaba;font-size:.96rem;margin-top:-12px;margin-bottom:24px;}
.brand {font-size:1.35rem;font-weight:700;letter-spacing:-.6px;color:#eff6fc;margin-bottom:3px;}
.brand span {color:#65e6bb;}
.muted {color:#8fa1b5;font-size:.8rem;}
[data-testid="stMetric"] {background:#111b27;border:1px solid #233244;border-radius:14px;padding:17px 20px;min-height:113px;}
[data-testid="stMetricLabel"] {color:#9bafc3;font-size:.8rem;}
[data-testid="stMetricValue"] {font-size:1.8rem;font-weight:650;letter-spacing:-.5px;}
[data-testid="stVerticalBlockBorderWrapper"]>div {border-color:#223143!important;border-radius:14px!important;}
button[kind="primary"] {background:#65e6bb!important;color:#0c1c19!important;border:none!important;font-weight:650;}
button {border-radius:8px!important;}
.stTabs [data-baseweb="tab-list"] {gap:26px;border-bottom:1px solid #243142;margin-top:20px;}
.stTabs [data-baseweb="tab"] {font-size:.9rem;padding-bottom:13px;}
.stTabs [aria-selected="true"] {color:#65e6bb!important;}
.stTabs [data-baseweb="tab-highlight"] {background:#65e6bb;}
.stCaption {color:#90a2b5;}
div[data-testid="stDataFrame"] {border-radius:9px;}
.section-note {font-size:.8rem;color:#8fa1b5;margin-top:-10px;margin-bottom:16px;}
@media(max-width:700px) {
 .block-container {padding:1rem .8rem 3rem;}
 h1 {font-size:2.1rem!important;}
 .stTabs [data-baseweb="tab-list"] {gap:14px;overflow-x:auto;}
 [data-testid="stMetric"] {min-height:90px;padding:12px;}
 [data-testid="stMetricValue"] {font-size:1.5rem;}
}
</style>
""", unsafe_allow_html=True)

colours = ["#65E6BB", "#70ADFF", "#D6A5FF", "#FFBE76"]
local_timezone = ZoneInfo("Europe/London")


# ============================================================
# Sidebar: connection and saved experiments
# ============================================================
with st.sidebar:
    st.markdown('<div class="brand"><span>◈</span> CMI Dexter</div>', unsafe_allow_html=True)
    st.caption("EXPERIMENT WORKSPACE")
    st.divider()
    kernel = st.text_input("Kaggle notebook", value=data_utils.DEFAULT_KERNEL)
    refresh = st.button("↻  Refresh from Kaggle", type="primary", width="stretch")
    st.caption("Reads status, results and available training logs through the API.")

    if refresh:
        with st.spinner("Reading Kaggle results and live logs…"):
            try:
                data_utils.refresh_snapshot(kernel)
                st.success("Results refreshed")
            except (Exception, SystemExit) as error:
                st.error(f"Could not refresh Kaggle: {error}")

    saved_runs = data_utils.snapshots()
    if saved_runs:
        selected_run = st.selectbox("Saved snapshot", options=saved_runs,
                                    format_func=lambda p: p.name[:15].replace("_", " · "))
    else:
        selected_run = None
    st.divider()
    st.markdown("**Explore**")
    st.caption("Monitor learning curves, inspect candidate settings, compare experiments, and prepare the next run.")
    st.markdown("**Your existing code**")
    repository = st.text_input("Repository path", value=str(data_utils.DEFAULT_REPO))
    st.caption("Source files are read when preparing a run. Your original notebooks are not rewritten.")


# ============================================================
# Load selected results
# ============================================================
st.markdown('<div class="eyebrow">RESEARCH / EXPERIMENTS</div>', unsafe_allow_html=True)
st.title("Every run. A clearer picture.")
st.markdown('<div class="subheading">Your multi-branch experiments, learning curves and next decisions.</div>', unsafe_allow_html=True)

if selected_run is None:
    st.info("Connect to your notebook with Refresh from Kaggle, or import a saved experiment below.")
    uploaded = st.file_uploader("Import a JSON experiment snapshot", type=["json"])
    if uploaded:
        st.json(json.load(uploaded))
    st.stop()

snapshot = data_utils.load_snapshot(selected_run)
table = data_utils.leaderboard(snapshot)
valid_table = table[pd.to_numeric(table["CV score"], errors="coerce").ge(0)]
valid_trials = [t for t in snapshot["trials"] if pd.notna(t["score"]) and t["score"] >= 0]
valid_trials = sorted(valid_trials, key=lambda t: -t["score"])
state = snapshot["status"].get("status", {})
status_name = state.get("status", "UNKNOWN") if isinstance(state, dict) else str(state)
checked_at = snapshot["status"].get("checked_at")
checked_label = datetime.fromisoformat(checked_at).astimezone(local_timezone).strftime("%d %b, %H:%M %Z") if checked_at else "Time unavailable"
budget = snapshot["budget"]
budget_label = f"{len(snapshot['trials'])} / {budget}" if budget else str(len(snapshot["trials"]))
best_score = valid_trials[0]["score"] if valid_trials else None
gpu_enabled = snapshot["metadata"].get("enable_gpu", False)

st.caption(f"{snapshot['experiment']}  ·  Retrieved {checked_label}")
st.caption(f"Validation protocol: {snapshot['protocol']}. Scores from different protocols are kept separate.")
metrics = st.columns(5)
metrics[0].metric("RUN STATUS", status_name.title())
metrics[1].metric("CANDIDATES SCORED", budget_label)
metrics[2].metric("BEST CV SCORE", f"{best_score:.4f}" if best_score is not None else "—")
metrics[3].metric("HOLDOUT SCORE", f"{snapshot['holdout']:.4f}" if snapshot["holdout"] is not None else "—")
metrics[4].metric("ACCELERATOR", "T4 GPU" if gpu_enabled else "CPU")
if budget and len(snapshot["trials"]):
    st.progress(min(1.0, len(snapshot["trials"]) / budget), text=f"{snapshot['failures']} recorded candidate failures")
if state.get("failureMessage"):
    st.error(state["failureMessage"])
if snapshot["protocols"] and any(p.get("status") == "failed" for p in snapshot["protocols"]):
    st.error("One or more validation protocols failed. Inspect the logs before treating this run as successful.")

monitor_tab, candidates_tab, compare_tab, configure_tab, logs_tab = st.tabs(
    ["Overview", "Candidates", "Compare runs", "Next experiment", "Logs"])


# ============================================================
# Overview: leaders and training histories
# ============================================================
with monitor_tab:
    title_col, option_col = st.columns([2, 1])
    title_col.subheader("Learning curves")
    if valid_trials:
        display_mode = option_col.selectbox("Show", ["Selected candidate", "Leading three"], label_visibility="collapsed")
        candidate_id = st.selectbox("Candidate", options=[t["id"] for t in valid_trials],
                                    format_func=lambda value: f"Trial {value} · CV {next(t['score'] for t in valid_trials if t['id'] == value):.4f}")
        selected_trial = next(t for t in valid_trials if t["id"] == candidate_id)
        plotted_trials = valid_trials[:3] if display_mode == "Leading three" else [selected_trial]
        block_count = max((len(t["histories"]) for t in plotted_trials), default=0)
        block_index = st.selectbox("Fit / stage block", range(block_count),
                                   format_func=lambda i: f"Block {i + 1}") if block_count > 1 else 0
        chart_cols = st.columns(2)
        for chart_index, metric in enumerate(["accuracy", "loss"]):
            figure = go.Figure()
            for trial_index, trial in enumerate(plotted_trials):
                if not trial["histories"] or block_index >= len(trial["histories"]):
                    continue
                history = trial["histories"][block_index]
                for validation in [False, True]:
                    key = ("val_" if validation else "") + metric
                    if metric == "accuracy" and not any(key in r for r in history):
                        key = ("val_" if validation else "") + "predictions_accuracy"
                    rows = [r for r in history if key in r]
                    if rows:
                        figure.add_trace(go.Scatter(x=[r["epoch"] for r in rows], y=[r[key] for r in rows],
                            name=f"Trial {trial['id']} · {'Validation' if validation else 'Train'}",
                            mode="lines", line={"color": colours[trial_index % len(colours)],
                                                "width": 2.5, "dash": "dash" if validation else "solid"},
                            hovertemplate="Epoch %{x}<br>%{y:.4f}<extra>%{fullData.name}</extra>"))
            figure.update_layout(title="Accuracy" if metric == "accuracy" else "Loss",
                                 template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                 plot_bgcolor="#101a25", height=340,
                                 margin={"l": 15, "r": 15, "t": 45, "b": 10},
                                 xaxis_title="Epoch", yaxis_title="Accuracy" if metric == "accuracy" else "Training objective",
                                 legend={"orientation": "h", "y": -0.24}, font={"color": "#aabccd"})
            figure.update_xaxes(gridcolor="#203042")
            figure.update_yaxes(gridcolor="#203042")
            if metric == "accuracy":
                figure.update_yaxes(range=[0, 1.03], tickformat=".0%")
            if figure.data:
                chart_cols[chart_index].plotly_chart(figure, width="stretch", key=f"curve_{metric}")
            else:
                chart_cols[chart_index].info(f"{metric.title()} history is not available for this candidate yet.")
        st.caption("Solid = training · dashed = internal validation. CV competition scores and accuracy measure different things. Fit/stage blocks remain separate.")
        with st.container(border=True):
            st.markdown("**Experiment summary**")
            st.write(data_utils.curve_summary(selected_trial))
            p = selected_trial["params"]
            st.caption(f"Acceleration: {p.get('estimator__extractor__acc_modes', 'not recorded')} · Augmentation probability: {p.get('estimator__augmentor__prob', 'not recorded')}")
    else:
        if status_name == "QUEUED":
            st.info("Waiting for Kaggle compute. Candidate rankings and learning curves will appear when training starts.")
        else:
            st.info("No scored candidates have been published yet. Use Refresh from Kaggle to capture live epoch logs.")
        if snapshot["running_histories"]:
            h = snapshot["running_histories"][-1]
            st.caption(f"Unscored fit: {len(h)} epochs captured. It will join the leaderboard once its CV score is logged.")
            figure = go.Figure()
            for key in ["accuracy", "val_accuracy"]:
                rows = [r for r in h if key in r]
                if rows:
                    figure.add_trace(go.Scatter(x=[r["epoch"] for r in rows], y=[r[key] for r in rows], name=key))
            figure.update_layout(template="plotly_dark", title="Currently fitting · unscored", height=330,
                                 paper_bgcolor="rgba(0,0,0,0)", xaxis_title="Epoch")
            if figure.data:
                st.plotly_chart(figure, width="stretch")
    st.subheader("Leading candidates")
    if not valid_table.empty:
        st.dataframe(valid_table.head(5), hide_index=True, width="stretch",
                     column_config={"CV score": st.column_config.NumberColumn(format="%.4f"),
                                    "Last val. accuracy": st.column_config.NumberColumn(format="%.4f")})
    else:
        st.caption("No candidate scores available in this snapshot.")


# ============================================================
# Full candidate table and parameter inspection
# ============================================================
with candidates_tab:
    st.subheader("Every candidate, in one place")
    show_failures = st.toggle("Include failed candidates", value=False)
    candidate_table = table if show_failures else valid_table
    st.dataframe(candidate_table, hide_index=True, width="stretch")
    st.download_button("Download candidate results", candidate_table.to_csv(index=False),
                       file_name="cmi_candidate_results.csv", mime="text/csv")
    if snapshot["trials"]:
        inspect_id = st.selectbox("Inspect parameters", [t["id"] for t in snapshot["trials"]])
        inspected = next(t for t in snapshot["trials"] if t["id"] == inspect_id)
        st.caption(f"Score source: {inspected['source']}")
        query = st.text_input("Filter parameter names", placeholder="dropout, augmentor, learning_rate…")
        parameters_df = pd.DataFrame([{"Parameter": key, "Value": str(value)}
                                     for key, value in inspected["params"].items() if query.lower() in key.lower()])
        st.dataframe(parameters_df, hide_index=True, width="stretch")


# ============================================================
# Compare saved runs without merging validation protocols
# ============================================================
with compare_tab:
    st.subheader("Compare experiments")
    comparison_paths = st.multiselect("Snapshots", saved_runs, default=saved_runs[:min(3, len(saved_runs))],
                                      format_func=lambda p: p.name[:15])
    comparisons = []
    progress_figure = go.Figure()
    for index, path in enumerate(comparison_paths):
        other = data_utils.load_snapshot(path)
        other_table = data_utils.leaderboard(other)
        good = other_table[pd.to_numeric(other_table["CV score"], errors="coerce").ge(0)]
        comparisons.append({"Snapshot": path.name[:15], "Experiment": other["experiment"],
                            "Candidates": len(other["trials"]), "Best CV": float(good["CV score"].max()) if not good.empty else None,
                            "Holdout": other["holdout"], "Target": other["config"].get("TARGET_COL", "—"),
                            "Validation": ", ".join(p.get("validation_mode", "") for p in other["protocols"]) or "Not published"})
        scores = [t["score"] for t in other["trials"] if pd.notna(t["score"]) and t["score"] >= 0]
        if scores:
            progress_figure.add_trace(go.Scatter(x=list(range(1, len(scores) + 1)),
                                                 y=pd.Series(scores).cummax().tolist(),
                                                 name=path.name[:15], mode="lines+markers"))
    st.dataframe(pd.DataFrame(comparisons), hide_index=True, width="stretch")
    if progress_figure.data:
        progress_figure.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                      title="Best score as search progresses", xaxis_title="Scored candidate", yaxis_title="CV score", height=370)
        st.plotly_chart(progress_figure, width="stretch")
    st.caption("Compare like-for-like targets and validation protocols. A different split can change the meaning of a score.")


# ============================================================
# Next experiment: dictionary-based configuration
# ============================================================
with configure_tab:
    st.subheader("Prepare your next experiment")
    st.caption("Uses the retrieved notebook and an exact snapshot of your selected repository's source modules.")
    name = st.text_input("Experiment name", value="multibranch_dashboard_experiment")
    config_cols = st.columns(3)
    search_mode = config_cols[0].selectbox("Search", ["bayes", "evo"])
    iterations = config_cols[1].number_input("Bayesian candidates", 1, 500, 20)
    epochs = config_cols[2].number_input("Maximum epochs", 1, 5000, 150)
    training_cols = st.columns(3)
    architecture = training_cols[0].selectbox("Architecture", ["bigru_attention", "cnn_attention_pool", "bert", "phase_attention"])
    use_bigru = training_cols[1].checkbox("Enable BiGRU", value=False, disabled=architecture != "bigru_attention")
    target = training_cols[2].selectbox("Target", ["bfrb", "gesture", "gesture_action", "orientation", "gesture_position"])
    sequence_cols = st.columns(2)
    finetuning_source = sequence_cols[0].selectbox("Training sequences", ["target", "all", "non_target"])
    pretraining_source = sequence_cols[1].selectbox("Pretraining source", ["baseline", "non_target", "all", "target"])
    learning_rates_text = st.text_input("Learning-rate candidates", value="0.0001")
    orientations_text = st.text_input("Orientation filter", placeholder="Leave blank for all; separate exact labels with commas")
    acc_modes = st.multiselect("Acceleration candidates",
        ["raw", "smoothed", "velocity", "displacement", "jerk", "raw|smoothed|jerk",
         "smoothed|jerk", "smoothed|velocity|jerk", "smoothed|velocity|displacement|jerk",
         "raw|smoothed|velocity|displacement|jerk"],
        default=["raw", "smoothed|jerk", "raw|smoothed|jerk"])
    with st.expander("Training and regularisation", expanded=False):
        patience = st.number_input("Early-stopping patience · 0 disables", 0, 5000, 30)
        scheduler = st.selectbox("Learning-rate schedule", ["plateau", "cosine", "none"])
        weight_decay_text = st.text_input("Weight-decay candidates", value="0.0")
        cnn_dropout_text = st.text_input("CNN dropout candidates", value="0.05, 0.1, 0.15, 0.2")
        head_dropout_text = st.text_input("Classifier-head dropout candidates", value="0.0, 0.1, 0.2")
        static_dropout_text = st.text_input("Static-branch dropout candidates", value="0.0, 0.1, 0.2")
        st.caption("For bigru_attention with BiGRU and auxiliary heads off, CNN dropout maps to conv_dropout. Other temporal architectures use branch_dropout.")
    with st.expander("Augmentation candidates", expanded=False):
        enabled_augmentations = st.multiselect("Enabled transformations", data_utils.AUGMENTATIONS,
                                               default=["jitter", "scaling", "time_shift"])
        prob_text = st.text_input("Sequence probability", "0.0, 0.25, 0.5, 0.75, 1.0")
        per_aug_text = st.text_input("Per-transform probability", "0.25, 0.5, 0.75, 1.0")
        augmentation_defaults = {
            "sensor_drop_prob": "0.0, 0.05, 0.1, 0.2", "noise_std": "0.0, 0.025, 0.05, 0.1",
            "jitter_sigma": "0.0, 0.005, 0.01, 0.03", "scaling_sigma": "0.0, 0.025, 0.05, 0.1",
            "channel_drop_prob": "0.0, 0.05, 0.1, 0.2", "time_shift_frac": "0.025, 0.05, 0.1",
            "temporal_num_masks": "0, 1, 2", "temporal_mask_frac": "0.025, 0.05, 0.1",
            "warp_sigma": "0.0, 0.05, 0.1, 0.2", "warp_num_knots": "2, 4, 6, 8"}
        augmentation_text = {}
        for key, default in augmentation_defaults.items():
            augmentation_text[key] = st.text_input(key, default)
        crop_text = st.text_area("Crop fraction ranges · JSON", value="[[1.0, 1.0], [0.9, 1.0], [0.8, 1.0]]", height=80)
    config_error = None
    try:
        space = {
            "estimator__learning_rate": [float(v.strip()) for v in learning_rates_text.split(",")],
            "estimator__finetuning_learning_rate": [None], "estimator__epochs": [int(epochs)],
            "estimator__temporal_architecture": [architecture], "estimator__use_bigru": [use_bigru],
            "estimator__use_orientation_aux_head": [False], "estimator__use_phase_aux_head": [False],
            "estimator__custom_aux_heads": [[]], "estimator__primary_target": [target],
            "estimator__extractor__finetuning_label_column": [target],
            "estimator__extractor__finetuning_source": [finetuning_source],
            "estimator__extractor__pretraining_source": [pretraining_source],
            "estimator__extractor__acc_modes": acc_modes,
            "estimator__early_stopping_patience": [int(patience)],
            "estimator__optimizer_name": ["adamw"],
            "estimator__weight_decay": [float(v.strip()) for v in weight_decay_text.split(",")],
            "estimator__use_lr_scheduler": [scheduler != "none"],
            "estimator__lr_scheduler_type": ["plateau" if scheduler == "none" else scheduler],
            "estimator__cosine_t_max": [int(epochs)],
            "estimator__dropout": [float(v.strip()) for v in head_dropout_text.split(",")],
            "estimator__static_dropout": [float(v.strip()) for v in static_dropout_text.split(",")],
            "estimator__augmentor__augmentations": [enabled_augmentations],
            "estimator__augmentor__prob": [float(v.strip()) for v in prob_text.split(",")],
            "estimator__augmentor__per_aug_prob": [float(v.strip()) for v in per_aug_text.split(",")],
            "estimator__augmentor__crop_frac_range": [tuple(v) for v in json.loads(crop_text)],
        }
        dropout_key = "conv_dropout" if architecture == "bigru_attention" and not use_bigru else "branch_dropout"
        space[f"estimator__{dropout_key}"] = [float(v.strip()) for v in cnn_dropout_text.split(",")]
        for key, text in augmentation_text.items():
            converter = int if key in {"warp_num_knots", "temporal_num_masks"} else float
            space[f"estimator__augmentor__{key}"] = [converter(v.strip()) for v in text.split(",")]
        if any(not values for values in space.values()):
            raise ValueError("Each parameter needs at least one candidate.")
        for key, values in space.items():
            if "dropout" in key and "augmentor" not in key and any(not 0 <= v < 1 for v in values):
                raise ValueError(f"{key}: dropout must be at least 0 and below 1.")
            if key.endswith(("__prob", "__per_aug_prob", "__sensor_drop_prob", "__channel_drop_prob")) and any(not 0 <= v <= 1 for v in values):
                raise ValueError(f"{key}: probabilities must be between 0 and 1.")
        if any(v <= 0 for v in space["estimator__learning_rate"]):
            raise ValueError("Learning rates must be positive.")
        settings = {"name": name, "search": search_mode, "iterations": int(iterations), "epochs": int(epochs),
                    "target": target, "orientations": [s.strip() for s in orientations_text.split(",") if s.strip()],
                    "space": space}
    except (ValueError, TypeError) as error:
        config_error = str(error)
        st.error(config_error)
    if not config_error:
        with st.expander("Review parameter dictionary"):
            st.code("GRID_PARAM_SPACE = " + pformat(space, sort_dicts=False, width=110), language="python")
        st.download_button("Download configuration", json.dumps(settings, indent=2),
                           file_name="experiment_configuration.json", mime="application/json")
        action_cols = st.columns(2)
        if action_cols[0].button("Prepare notebook", width="stretch"):
            try:
                build = data_utils.prepare_experiment(snapshot, settings, repository)
                st.session_state["prepared_build"] = str(build)
                st.session_state["prepared_settings"] = json.dumps(settings, sort_keys=True)
                st.success("Notebook prepared with the current source snapshot.")
            except Exception as error:
                st.error(str(error))
        prepared = st.session_state.get("prepared_build")
        matching_settings = st.session_state.get("prepared_settings") == json.dumps(settings, sort_keys=True)
        if prepared and matching_settings:
            st.download_button("Download prepared notebook", (Path(prepared) / "experiment.ipynb").read_bytes(),
                               file_name="cmi_experiment.ipynb", mime="application/x-ipynb+json")
        active_run = status_name in {"RUNNING", "QUEUED", "STARTING"}
        if action_cols[1].button("Launch on Kaggle GPU", type="primary", width="stretch",
                                 disabled=not prepared or not matching_settings or active_run):
            with st.spinner("Submitting the prepared GPU notebook…"):
                try:
                    response = data_utils.submit_experiment(prepared)
                    st.success("Kaggle accepted the new version.")
                    st.json(response)
                except (Exception, SystemExit) as error:
                    st.error(str(error))
        if active_run:
            st.caption("Your notebook currently has an active run. You can prepare the next experiment while it finishes.")


# ============================================================
# Raw logs and provenance
# ============================================================
with logs_tab:
    st.subheader("Logs and run details")
    errors_only = st.toggle("Warnings and errors only", value=False)
    lines = snapshot["log"].splitlines()
    if errors_only:
        lines = [line for line in lines if any(word in line.lower() for word in ["warning", "error", "failed", "traceback"])]
    st.code("\n".join(lines[-400:]) if lines else "No logs published in this snapshot.", language="text", height=440)
    st.download_button("Download full log", snapshot["log"], file_name="execution.log", mime="text/plain")
    with st.expander("Notebook metadata"):
        st.json(snapshot["metadata"])
    st.caption(f"Snapshot retrieved {checked_label}. Refresh to retrieve newer results.")
