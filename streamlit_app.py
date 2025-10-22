
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ========= Imports =========
import json
from pathlib import Path
from io import BytesIO

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import streamlit as st
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report

# ========= Page style =========
st.set_page_config(page_title="Microbe-Driven Spoilage", page_icon="🥩", layout="wide")
st.markdown("""
<style>
  .stApp {
    background: linear-gradient(180deg,#ffeef8 0%,#fff9fc 100%);
    color:#4d004d;
    font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
  }
  h1,h2,h3 { color:#e75480; }
  .stButton button{
    background:#ffb6c1;color:#4d004d;border:none;border-radius:12px;
    padding:.55rem 1.2rem;font-weight:700;box-shadow:0 1px 8px rgba(231,84,128,.25);
  }
  .stButton button:hover{ background:#ffc8d9;color:#000; }
  .small-cap { color:#6b6b6b; font-size:0.9rem; }
  .codechip {
    display:inline-block; padding:2px 6px; background:#f6f6f6; border-radius:6px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    font-size: 12px; color:#4b5563;
  }
</style>
""", unsafe_allow_html=True)

st.title("🥩🍖 Microbe-Driven Spoilage")
st.caption("Upload a CSV to get predictions, performance metrics, sensory guidance, and microbiome summaries.")

# ========= Sidebar (fixed artifacts + RUO) =========
with st.sidebar:
    st.header("📦 Artifacts & Settings")

    artifacts_dir = Path(__file__).parent / "artifacts"
    st.markdown(
        f"**Artifacts folder:**  \n"
        f"<span class='codechip'>{artifacts_dir.as_posix()}</span>",
        unsafe_allow_html=True
    )

    st.markdown("**Classification threshold:** 0.50")
    st.caption("Used for classification, performance summaries, sensory grouping, and microbiome summaries.")

    st.markdown("---")
    st.markdown(
        "⚠️ **Research Use Only (RUO)**  \n"
        "This model was trained on **pork** and **poultry** datasets. "
        "Outputs are research-grade and **not** intended for safety, clinical, or regulatory use."
    )

# ========= Load artifacts =========
need = ["rf_model_tuned.joblib", "model_meta.json"]
if not artifacts_dir.exists() or any(not (artifacts_dir/f).exists() for f in need):
    st.warning(f"Artifacts not found at: {artifacts_dir}\n\nRequired: {need}")
    st.stop()

@st.cache_resource
def load_artifacts(folder: Path):
    model = joblib.load(folder / "rf_model_tuned.joblib")
    meta = json.loads((folder / "model_meta.json").read_text())
    return model, meta

try:
    clf, meta = load_artifacts(artifacts_dir)
except Exception as e:
    st.error(f"Failed to load artifacts: {e}")
    st.stop()

feature_names = meta.get("feature_names", [])
GT_THRESHOLD = float(meta.get("threshold_cfu", 7.0))  # ground-truth late threshold if CFU present

# Aesthetic palette used in Plotly charts
PALETTE = ["#d5b4ab", "#c3b1e1", "#a3c1da", "#a8c69f", "#f4c2c2", "#f7b3c2", "#b9e0ff"]

# ========= Upload data =========
csv_file = st.file_uploader("Upload CSV (same schema as model training)", type=["csv"])
if csv_file is None:
    st.info("Upload your dataset to generate spoilage predictions and sensory insights.")
    st.stop()

try:
    df_raw = pd.read_csv(csv_file)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

# Column helpers
ITEM_COL = "Sample_Name_y" if "Sample_Name_y" in df_raw.columns else None
PTYPE_COL = "EnvType" if "EnvType" in df_raw.columns else None
if "Day_numeric" in df_raw.columns:
    DAY_COL = "Day_numeric"
elif "Days_Numeric" in df_raw.columns:
    DAY_COL = "Days_Numeric"
else:
    DAY_COL = None
LOG_CFU_COL = "Total mesophilic aerobic flora (log10 CFU.g-1)"  # if present

# ========= Build X, run classifier =========
X = df_raw.reindex(columns=feature_names, fill_value=0)
for c in X.columns:
    if not np.issubdtype(X[c].dtype, np.number):
        X[c] = pd.to_numeric(X[c], errors="coerce")
X = X.fillna(0.0)

prob_thr = 0.50
try:
    score_not_safe = clf.predict_proba(X)[:, 1]
except Exception as e:
    st.error(f"Prediction failed: {e}")
    score_not_safe = np.zeros(len(X))

pred_class = np.where(score_not_safe >= prob_thr, "not-safe", "safe")
# confidence away from threshold (0..1)
safe_conf = (prob_thr - score_not_safe) / prob_thr
notsafe_conf = (score_not_safe - prob_thr) / (1.0 - prob_thr)
confidence = np.where(pred_class == "safe", safe_conf, notsafe_conf)
confidence = np.clip(np.nan_to_num(confidence), 0.0, 1.0)

# ========= Tabs =========
tab_pred, tab_perf, tab_micro, tab_sens = st.tabs(
    ["🔮 Predictions", "📊 Performance", "🧬 Microbiome", "👃 Sensory Profile"]
)

# ----- Predictions -----
with tab_pred:
    st.markdown("#### 🔍 Model Predictions")
    st.caption("Classifies each uploaded sample as *safe* or *not-safe* (threshold: 0.50) and reports confidence.")

    disp_cols = {}
    if ITEM_COL: disp_cols["Item"] = df_raw[ITEM_COL]
    if PTYPE_COL: disp_cols["Product Type"] = df_raw[PTYPE_COL]
    if DAY_COL: disp_cols["Days"] = df_raw[DAY_COL]
    disp_cols["Prediction"] = pred_class
    disp_cols["Confidence"] = confidence
    if LOG_CFU_COL in df_raw.columns:
        disp_cols["Log CFU (Input)"] = df_raw[LOG_CFU_COL]
    disp = pd.DataFrame(disp_cols)

    st.dataframe(disp.style.format({"Confidence": "{:.1%}"}), use_container_width=True)

    st.download_button(
        "⬇️ Download predictions",
        disp.to_csv(index=False).encode("utf-8"),
        "spoilage_predictions.csv",
        "text/csv"
    )

    # quick probability histogram
    figp, axp = plt.subplots(figsize=(6.2, 3.6))
    axp.hist(score_not_safe, bins=25, edgecolor="white")
    axp.axvline(prob_thr, color="#e75480", linestyle="--", linewidth=2)
    axp.set_title("Predicted probability of 'not-safe'")
    axp.set_xlabel("Probability")
    axp.set_ylabel("Count")
    st.pyplot(figp)

# ----- Performance -----
with tab_perf:
    st.markdown("#### 📈 Performance vs. Ground Truth")
    st.caption("Compares predicted labels with ground-truth CFU values (≥ 7 log CFU·g⁻¹ = not-safe) when present.")

    if LOG_CFU_COL not in df_raw.columns:
        st.info(f"Ground-truth column “{LOG_CFU_COL}” is not in this CSV, so summary metrics cannot be computed.")
    else:
        y_true = np.where(df_raw[LOG_CFU_COL] >= GT_THRESHOLD, "not-safe", "safe")
        y_pred = pred_class
        labels = ["safe","not-safe"]

        # classification report
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
        rep_df = pd.DataFrame(report).transpose()
        st.dataframe(
            rep_df.style.format({
                "precision": "{:.2f}",
                "recall": "{:.2f}",
                "f1-score": "{:.2f}",
                "support": "{:.0f}"
            }),
            use_container_width=True
        )

        # confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig_cm = px.imshow(
            cm, text_auto=True, x=labels, y=labels,
            labels=dict(x="Predicted", y="True"), color_continuous_scale="Reds"
        )
        fig_cm.update_layout(title="Confusion Matrix", coloraxis_showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)

# ----- Microbiome -----
with tab_micro:
    st.markdown("#### 🧬 Microbiome Relative Abundance")
    st.caption("Shows composition by sample. If the file has no numeric abundance columns, this view is empty.")

    NON_MICROBE_COLS = set([c for c in [ITEM_COL, PTYPE_COL, DAY_COL, LOG_CFU_COL, "Prediction", "Confidence"] if c])
    micro_cols = [
        c for c in df_raw.columns
        if c not in NON_MICROBE_COLS and np.issubdtype(df_raw[c].dtype, np.number)
    ]

    if not micro_cols:
        st.info("No numeric microbial columns detected.")
    else:
        micro_df = df_raw[micro_cols].copy()
        totals = micro_df.sum(axis=1).replace(0, np.nan)
        rel_abund = micro_df.div(totals, axis=0).fillna(0.0)  # 0..1

        sample_labels = (df_raw[ITEM_COL] if ITEM_COL else df_raw.index).astype(str).tolist()
        sel = st.selectbox("Select a sample", sample_labels, index=0)

        # resolve index
        if ITEM_COL:
            idx_series = df_raw.index[df_raw[ITEM_COL].astype(str) == sel]
            idx = int(idx_series[0]) if len(idx_series) else 0
        else:
            idx = int(sel)

        topN = st.slider("Top microbes (N)", 5, 20, 10, 1)
        series = (rel_abund.iloc[idx].sort_values(ascending=False).head(topN) * 100.0)
        df_top = series.rename("Relative abundance (%)").reset_index().rename(columns={"index": "Microbe"})

        fig = px.bar(
            df_top, x="Microbe", y="Relative abundance (%)", color="Microbe",
            color_discrete_sequence=(PALETTE * ((len(df_top)//len(PALETTE))+1))[:len(df_top)],
            title=f"Top {topN} microbes — {sel}"
        )
        fig.update_layout(
            xaxis_title="", yaxis_title="Relative abundance (%)",
            xaxis_tickangle=-30, legend_title_text="Microbe"
        )
        st.plotly_chart(fig, use_container_width=True)

        # keep for potential future use
        st.session_state["_rel_abund"] = rel_abund
        st.session_state["_micro_cols"] = micro_cols

# ----- Sensory -----
with tab_sens:
    st.markdown("#### 👃 Sensory Profile")
    st.caption(
        "If the CSV includes sensory columns, they’re plotted directly. "
        "By default we show **mean intensity by day** (faceted by product type). "
        "Toggle *Show individual traces* to see per-sample trajectories."
    )

    sensory_cols = [c for c in ["Etheral","Fermented","Prickly","Rancid","Sulfurous","Old_cheese"] if c in df_raw.columns]

    if sensory_cols and DAY_COL:
        sens = df_raw[[c for c in [PTYPE_COL, DAY_COL] if c] + sensory_cols].copy()
        for c in sensory_cols:
            sens[c] = pd.to_numeric(sens[c], errors="coerce")
        sens = sens.dropna(subset=[DAY_COL])

        show_spaghetti = st.checkbox("Show individual traces", value=False)

        long = sens.melt(
            id_vars=[c for c in [PTYPE_COL, DAY_COL] if c],
            value_vars=sensory_cols,
            var_name="Descriptor",
            value_name="Intensity"
        ).dropna(subset=["Intensity"])

        facet_arg = dict(facet_col=PTYPE_COL) if PTYPE_COL else {}

        if not show_spaghetti:
            agg = (
                long.groupby([c for c in [PTYPE_COL, DAY_COL, "Descriptor"] if c], as_index=False)
                    .agg(Mean=("Intensity","mean"))
            )
            fig = px.line(
                agg, x=DAY_COL, y="Mean", color="Descriptor",
                markers=True, color_discrete_sequence=PALETTE,
                title="Mean sensory intensity over time", **facet_arg
            )
            fig.update_layout(xaxis_title="Day", yaxis_title="Mean intensity")
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.line(
                long, x=DAY_COL, y="Intensity", color="Descriptor",
                hover_data=[DAY_COL, "Descriptor"], color_discrete_sequence=PALETTE,
                title="Per-sample sensory trajectories", **facet_arg
            )
            fig.update_traces(line=dict(width=1), opacity=0.35)
            fig.update_layout(xaxis_title="Day", yaxis_title="Intensity")
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No sensory columns found. Showing a generalized early → late pattern.")
        trend = pd.DataFrame({
            "Descriptor":["Etheral","Fermented","Prickly","Rancid","Sulfurous","Old_cheese"],
            "Early":[0.4,0.3,0.2,0.2,0.1,0.2],
            "Late":[1.2,1.8,1.5,1.3,0.5,1.0]
        })
        df_plot = trend.melt(id_vars="Descriptor", var_name="Stage", value_name="Intensity")
        fig = px.bar(
            df_plot, x="Descriptor", y="Intensity", color="Stage",
            barmode="group", color_discrete_sequence=["#a3c1da","#e75480"],
            title="Generalized Sensory Profile (Early vs Late)"
        )
        st.plotly_chart(fig, use_container_width=True)
