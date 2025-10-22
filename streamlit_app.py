#!/usr/bin/env python3
import json
from pathlib import Path
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from sklearn.metrics import confusion_matrix, classification_report

# ------------------ PAGE STYLE ------------------
st.set_page_config(page_title="Microbe-Driven Spoilage", page_icon="🥩", layout="wide")
st.markdown("""
<style>
  .stApp {
    background: linear-gradient(180deg,#ffeef8 0%,#fff9fc 100%);
    color:#4d004d;
  }
  h1,h2,h3 { color:#e75480; }
  .stButton button {
    background:#ffb6c1;color:#4d004d;border:none;border-radius:12px;
    padding:.55rem 1.2rem;font-weight:700;box-shadow:0 1px 8px rgba(231,84,128,.25);
  }
  .stButton button:hover { background:#ffc8d9;color:#000; }
</style>
""", unsafe_allow_html=True)

st.title("🥩🍖 Microbe-Driven Spoilage + 👃 Sensory Forecast")
st.caption("Upload a CSV to get spoilage predictions, performance metrics, sensory guidance, and microbiome summaries.")

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.header("🧰 Artifacts & Settings")
    # Always look for artifacts within repo folder
    default_artifacts = str(Path(__file__).parent / "artifacts")
    st.markdown(f"**Artifacts folder**: `{default_artifacts}`")

    st.markdown("**Classification threshold:**  **0.50**")
    st.caption("Used for classification, performance summaries, sensory grouping, and microbiome summaries.")

    st.markdown("""
    ---
    **⚠️ Research Use Only (RUO)**  
    This model was trained on **pork** and **poultry** samples.  
    All outputs are research-grade and **not** intended for safety, clinical, or regulatory use.
    """, unsafe_allow_html=True)

artifacts_dir = Path(default_artifacts)
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
GT_THRESHOLD = float(meta.get("threshold_cfu", 7.0))
PALETTE = ["#d5b4ab","#c3b1e1","#a3c1da","#a8c69f","#f4c2c2"]

# ------------------ DATA UPLOAD ------------------
csv_file = st.file_uploader("Upload CSV (same schema as model training)", type=["csv"])
if csv_file is None:
    st.info("Upload your dataset to generate spoilage predictions, sensory insights, and microbiome views.")
    st.stop()

df_raw = pd.read_csv(csv_file)

# ------------------ PREDICTIONS ------------------
X = df_raw.reindex(columns=feature_names, fill_value=0)
for c in X.columns:
    if not np.issubdtype(X[c].dtype, np.number):
        X[c] = pd.to_numeric(X[c], errors="coerce")
X = X.fillna(0.0)

prob_thr = 0.50

try:
    pred_score = clf.predict_proba(X)[:, 1]  # probability of "not-safe"
    classifier_ok = True
except Exception as e:
    st.error(f"Prediction failed: {e}")
    classifier_ok = False
    pred_score = np.zeros(len(X))

pred_class = np.where(pred_score >= prob_thr, "not-safe", "safe")

# Confidence (distance from threshold, 0..1)
safe_conf = (prob_thr - pred_score) / prob_thr
notsafe_conf = (pred_score - prob_thr) / (1.0 - prob_thr)
confidence = np.where(pred_class == "safe", safe_conf, notsafe_conf)
confidence = np.clip(np.nan_to_num(confidence), 0.0, 1.0)

# Column names used for display/plots if present
ITEM_COL = "Sample_Name_y"
PTYPE_COL = "EnvType"
DAY_COL = "Day_numeric" if "Day_numeric" in df_raw.columns else (
    "Days_Numeric" if "Days_Numeric" in df_raw.columns else None
)
LOG_CFU_COL = "Total mesophilic aerobic flora (log10 CFU.g-1)"

# ------------------ TABS ------------------
tab_pred, tab_perf, tab_micro, tab_sens = st.tabs([
    "🔮 Predictions", "📊 Performance", "🧬 Microbiome", "👃 Sensory Profile"
])

# ---------- PREDICTIONS ----------
with tab_pred:
    st.markdown("### 🔍 Predictions")
    st.caption("Each uploaded sample is classified as **safe** or **not-safe** with a confidence score based on the trained classifier.")

    disp = pd.DataFrame({
        "Item": df_raw.get(ITEM_COL, ""),
        "Product Type": df_raw.get(PTYPE_COL, ""),
        "Days (if provided)": df_raw.get(DAY_COL, ""),
        "Prediction": pred_class,
        "Confidence": confidence
    })
    if LOG_CFU_COL in df_raw.columns:
        disp["Log CFU (Input)"] = df_raw[LOG_CFU_COL]

    st.dataframe(disp.style.format({"Confidence": "{:.1%}"}), use_container_width=True)
    st.download_button(
        "⬇️ Download predictions",
        disp.to_csv(index=False).encode("utf-8"),
        "spoilage_predictions.csv",
        "text/csv"
    )

# ---------- PERFORMANCE ----------
with tab_perf:
    st.markdown("### 📈 Performance")
    st.caption("If your CSV includes ground-truth load (column: **Total mesophilic aerobic flora (log10 CFU.g-1)**), metrics compare predictions to the rule **≥ 7 = not-safe**.")

    if LOG_CFU_COL not in df_raw.columns:
        st.info(f"Ground-truth column **'{LOG_CFU_COL}'** not found — metrics are skipped.")
    else:
        y_true = np.where(df_raw[LOG_CFU_COL] >= GT_THRESHOLD, "not-safe", "safe")
        y_pred = pred_class
        labels = ["safe", "not-safe"]

        # Classification report
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig_cm = px.imshow(
            cm, text_auto=True,
            labels=dict(x="Predicted", y="True"),
            x=labels, y=labels, color_continuous_scale="Reds"
        )
        fig_cm.update_layout(title="Confusion Matrix", coloraxis_showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)

# ---------- MICROBIOME ----------
with tab_micro:
    st.markdown("### 🧫 Microbiome Relative Abundance")
    st.caption("Shows composition by sample. If your file has no numeric abundance columns, this view will be empty.")

    NON_MICROBE_COLS = {LOG_CFU_COL, "Prediction", "Confidence"}
    if ITEM_COL: NON_MICROBE_COLS.add(ITEM_COL)
    if PTYPE_COL: NON_MICROBE_COLS.add(PTYPE_COL)
    if DAY_COL: NON_MICROBE_COLS.add(DAY_COL)

    micro_cols = [
        c for c in df_raw.columns
        if c not in NON_MICROBE_COLS and np.issubdtype(df_raw[c].dtype, np.number)
    ]

    if not micro_cols:
        st.info("No numeric microbial abundance columns detected.")
    else:
        micro_df = df_raw[micro_cols]
        rel_abund = micro_df.div(micro_df.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

        sample_labels = df_raw.get(ITEM_COL, df_raw.index.astype(str))
        sel = st.selectbox("Select a sample", sample_labels, index=0)
        idx = df_raw.index[df_raw.get(ITEM_COL, df_raw.index).astype(str) == str(sel)][0]

        series = (rel_abund.iloc[idx].sort_values(ascending=False).head(10) * 100)
        df_top = series.rename("Relative Abundance (%)").reset_index().rename(columns={"index": "Microbe"})

        fig = px.bar(
            df_top, x="Microbe", y="Relative Abundance (%)",
            color="Microbe", color_discrete_sequence=PALETTE,
            title=f"Top 10 microbes — {sel}"
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------- SENSORY ----------
with tab_sens:
    st.markdown("### 👃 Sensory Profile")
    st.caption("If the CSV includes sensory columns, they’re plotted directly. Otherwise a generalized early→late pattern is shown.")

    sensory_cols = [c for c in ["Etheral","Fermented","Prickly","Rancid","Sulfurous","Old_cheese"] if c in df_raw.columns]

    if sensory_cols and DAY_COL and PTYPE_COL in df_raw.columns:
        melt = df_raw.melt(
            id_vars=[PTYPE_COL, DAY_COL],
            value_vars=sensory_cols,
            var_name="Descriptor", value_name="Intensity"
        )
        fig = px.line(
            melt, x=DAY_COL, y="Intensity", color="Descriptor", facet_col=PTYPE_COL,
            markers=True, color_discrete_sequence=PALETTE,
            title="Sensory Intensity Over Time"
        )
        st.plotly_chart(fig, use_container_width=True)
    elif sensory_cols:
        melt = df_raw.melt(value_vars=sensory_cols, var_name="Descriptor", value_name="Intensity")
        fig = px.box(melt, x="Descriptor", y="Intensity", color="Descriptor",
                     color_discrete_sequence=PALETTE, title="Sensory Intensities (no day grouping)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Generalized early vs late profile (used when no sensory columns provided)
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
