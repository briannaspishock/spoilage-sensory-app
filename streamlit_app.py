#!/usr/bin/env python3
# streamlit_app.py

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

# ===================== PAGE STYLE =====================
st.set_page_config(page_title="Microbe-Driven Spoilage", page_icon="🥩", layout="wide")
st.markdown("""
<style>
  .stApp { background: linear-gradient(180deg,#ffeef8 0%,#fff9fc 100%); color:#4d004d; }
  h1 { font-size: 1.6rem; line-height: 1.2; margin: .25rem 0 .75rem 0; color:#e75480; }
  h2,h3 { color:#e75480; }
  .stButton button{
    background:#ffb6c1;color:#4d004d;border:none;border-radius:12px;
    padding:.55rem 1.2rem;font-weight:700;box-shadow:0 1px 8px rgba(231,84,128,.25);
  }
  .stButton button:hover{ background:#ffc8d9;color:#000; }
  .smallcap { color:#6b4d57; font-size:.88rem; }
  .block-container { padding-top: .75rem; }
</style>
""", unsafe_allow_html=True)

st.title("🥩🍖 Microbe-Driven Spoilage + 👃 Sensory Forecast")
st.caption("Predictions, performance, microbiome summaries, and sensory guidance from your uploaded CSV.")

# ===================== CONSTANTS =====================
PALETTE = ["#d5b4ab","#c3b1e1","#a3c1da","#a8c69f","#f4c2c2"]
PROB_THR = 0.50
ITEM_COL = "Sample_Name_y"
PTYPE_COL = "EnvType"
DAY_COLS = ["Day_numeric","Days_Numeric"]
LOG_CFU_COL = "Total mesophilic aerobic flora (log10 CFU.g-1)"
SENS_COLS = ["Etheral","Fermented","Prickly","Rancid","Sulfurous","Old_cheese"]

# ===================== SIDEBAR =====================
with st.sidebar:
    st.header("📦 Artifacts & Settings")
    artifacts_dir = Path(__file__).parent / "artifacts"
    artifacts_dir_str = st.text_input("Artifacts folder", value=str(artifacts_dir),
                                      help="Folder must contain: rf_model_tuned.joblib, model_meta.json")
    st.markdown(f"**Classification threshold:**  {PROB_THR:.2f}")
    st.caption("Used for classification, performance summaries, sensory grouping, and microbiome summaries.")

    st.markdown("---")
    st.markdown("### 📤 Upload data")
    csv_file = st.file_uploader("CSV (same schema as model training)", type=["csv"],
                                help="Drag & drop or browse a .csv with the training schema.")
    if csv_file:
        st.caption(f"Loaded: **{csv_file.name}**")
    else:
        st.caption("No file selected yet.")

    st.markdown("---")
    st.markdown(
        "### ⚠️ Research Use Only (RUO)\n"
        "<small>This model was trained on **pork** and **poultry** datasets. "
        "All outputs are research-grade and **not** intended for safety, clinical, or regulatory use.</small>",
        unsafe_allow_html=True
    )

# ===================== LOAD ARTIFACTS =====================
need = ["rf_model_tuned.joblib", "model_meta.json"]
artifacts_dir = Path(artifacts_dir_str).expanduser()
if not artifacts_dir.exists() or any(not (artifacts_dir / f).exists() for f in need):
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

# ===================== DATA =====================
if csv_file is None:
    st.info("Upload a CSV from the sidebar to generate predictions and insights.")
    st.stop()

try:
    df_raw = pd.read_csv(csv_file)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

# Choose day column if present
DAY_COL = next((c for c in DAY_COLS if c in df_raw.columns), None)

# ===================== PREDICTIONS =====================
X = df_raw.reindex(columns=feature_names, fill_value=0)
for c in X.columns:
    if not np.issubdtype(X[c].dtype, np.number):
        X[c] = pd.to_numeric(X[c], errors="coerce")
X = X.fillna(0.0)

try:
    prob_not_safe = clf.predict_proba(X)[:, 1]
except Exception as e:
    st.error(f"Prediction failed: {e}")
    prob_not_safe = np.zeros(len(X))

pred_class = np.where(prob_not_safe >= PROB_THR, "not-safe", "safe")

# confidence (distance to threshold)
safe_conf = (PROB_THR - prob_not_safe) / PROB_THR
notsafe_conf = (prob_not_safe - PROB_THR) / (1.0 - PROB_THR)
confidence = np.where(pred_class == "safe", safe_conf, notsafe_conf)
confidence = np.clip(np.nan_to_num(confidence), 0.0, 1.0)

# ===================== HELPERS =====================
def rel_abundance_table(df: pd.DataFrame, exclude_cols: set) -> tuple[pd.DataFrame, list]:
    micro_cols = [c for c in df.columns if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]
    if not micro_cols:
        return pd.DataFrame(index=df.index), micro_cols
    micro_df = df[micro_cols].copy()
    totals = micro_df.sum(axis=1).replace(0, np.nan)
    rel_abund = micro_df.div(totals, axis=0).fillna(0.0)
    return rel_abund, micro_cols

SMELL_GUIDE = {
    "safe": "🧊 Low risk — expect faintly sweet **Etheral** notes, mild **Fermented** aroma.",
    "not-safe": "🔥 High risk — stronger **Fermented/Prickly** with **Rancid** or **Old_cheese** tones."
}

def one_liner_from_sensory(row: pd.Series) -> str:
    # if real sensory is present, bias the wording slightly
    vals = row.get(SENS_COLS, pd.Series(dtype=float))
    if isinstance(vals, pd.Series) and len(vals.dropna()) > 0:
        hi = vals.idxmax()
        if hi in ["Rancid","Old_cheese","Prickly","Fermented"]:
            return "🔥 Sensory suggests late spoilage (sharp fermented/rancid characteristics)."
        return "🧊 Sensory suggests early stage (light, sweet/etheral profile)."
    return ""  # fall back to risk-based text

def highlight_pred(df):
    # style: highlight Item and Prediction columns by class
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    if "Prediction" in df:
        red = "background-color:#fde8e8;color:#9b1c1c;font-weight:600;"
        green = "background-color:#e8f5e9;color:#1b5e20;font-weight:600;"
        for i, v in df["Prediction"].items():
            style = red if v == "not-safe" else green
            for col in ["Item","Prediction"]:
                if col in df.columns:
                    styles.loc[i, col] = style
    return styles

# ===================== TABS =====================
tab_pred, tab_perf, tab_micro, tab_sens = st.tabs(
    ["🔮 Predictions", "📊 Performance", "🧬 Microbiome", "👃 Sensory"]
)

# ---------- PREDICTIONS ----------
with tab_pred:
    st.markdown("### 🔍 Predictions")
    st.caption("Model classifies each sample as **safe** or **not-safe** (threshold 0.50) and shows confidence.")

    disp = pd.DataFrame({
        "Item": df_raw.get(ITEM_COL, ""),
        "Product Type": df_raw.get(PTYPE_COL, ""),
        "Days": df_raw.get(DAY_COL, ""),
        "Prediction": pred_class,
        "Confidence": confidence
    })
    if LOG_CFU_COL in df_raw.columns:
        disp["Log CFU (Input)"] = df_raw[LOG_CFU_COL]

    # Add a compact, risk-based smell hint (not full sensory)
    disp["Smell guidance (1-liner)"] = np.where(
        disp["Prediction"].eq("not-safe"),
        SMELL_GUIDE["not-safe"],
        SMELL_GUIDE["safe"]
    )

    st.dataframe(
        disp.style.apply(highlight_pred, axis=None).format({"Confidence":"{:.1%}"}),
        use_container_width=True, height=420
    )
    st.download_button(
        "⬇️ Download predictions",
        disp.to_csv(index=False).encode("utf-8"),
        "spoilage_predictions.csv",
        "text/csv"
    )

# ---------- PERFORMANCE ----------
with tab_perf:
    st.markdown("### 📈 Performance")
    st.caption("If your CSV contains the CFU ground-truth column (≥ 7 = not-safe), metrics are computed below.")
    if LOG_CFU_COL not in df_raw.columns:
        st.warning(f"Ground truth column **'{LOG_CFU_COL}'** missing — cannot compute metrics.")
    else:
        y_true = np.where(df_raw[LOG_CFU_COL] >= GT_THRESHOLD, "not-safe", "safe")
        y_pred = pred_class
        labels = ["safe","not-safe"]

        # Report
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
        rep_df = pd.DataFrame(report).transpose()
        st.dataframe(
            rep_df.style.format({"precision":"{:.2f}","recall":"{:.2f}","f1-score":"{:.2f}","support":"{:.0f}"}),
            use_container_width=True
        )

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig_cm = px.imshow(
            cm, text_auto=True,
            labels=dict(x="Predicted", y="True"),
            x=labels, y=labels,
            color_continuous_scale="Reds"
        )
        fig_cm.update_layout(title="Confusion Matrix", coloraxis_showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)

# ---------- MICROBIOME ----------
with tab_micro:
    st.markdown("### 🧫 Microbiome Relative Abundance")
    st.caption("Shows composition by sample (top-N bar). Below: **Top taxa by predicted risk** from your upload.")

    non_micro = {ITEM_COL, PTYPE_COL, LOG_CFU_COL, "Prediction", "Confidence"}
    if DAY_COL: non_micro.add(DAY_COL)

    rel_abund, micro_cols = rel_abundance_table(df_raw, non_micro)

    if not micro_cols:
        st.info("No numeric microbial columns detected to compute relative abundance.")
    else:
        # single-sample bar + small table
        c1, c2 = st.columns([2,1], gap="large")

        with c1:
            labels = df_raw.get(ITEM_COL, df_raw.index.astype(str))
            sel = st.selectbox("Select a sample", labels, index=0)
            idx = df_raw.index[labels.astype(str) == str(sel)][0]
            topN = st.slider("Top microbes (N)", 5, 20, 10, 1, key="topN_single")

            series = (rel_abund.iloc[idx].sort_values(ascending=False).head(topN) * 100.0)
            df_top = series.rename("Relative_Abundance_%").reset_index().rename(columns={"index":"Microbe"})

            fig = px.bar(
                df_top, x="Microbe", y="Relative_Abundance_%",
                color="Microbe",
                color_discrete_sequence=(PALETTE * ((len(df_top)//len(PALETTE))+1))[:len(df_top)],
                title=f"Top {topN} microbes — {sel}"
            )
            fig.update_layout(yaxis_title="Relative abundance (%)", xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("**Top microbes (this sample)**")
            st.dataframe(df_top, use_container_width=True, height=330)

        st.markdown("---")

        # Top taxa by risk buckets from predictions
        st.subheader("Top taxa by predicted risk")
        thr = PROB_THR
        high_mask = prob_not_safe >= thr
        low_mask  = prob_not_safe < thr

        left, right = st.columns(2)
        with left:
            st.markdown(f"**Top 10 taxa — High-risk (prob ≥ {thr:.2f})**")
            if high_mask.any():
                top_high = rel_abund[high_mask].mean().sort_values(ascending=False).head(10) * 100.0
                st.dataframe(
                    top_high.rename("Mean_RelAbund_%").reset_index().rename(columns={"index":"Taxon"}),
                    use_container_width=True, height=320
                )
            else:
                st.info("No samples in the high-risk bucket.")

        with right:
            st.markdown(f"**Top 10 taxa — Low-risk (prob < {thr:.2f})**")
            if low_mask.any():
                top_low = rel_abund[low_mask].mean().sort_values(ascending=False).head(10) * 100.0
                st.dataframe(
                    top_low.rename("Mean_RelAbund_%").reset_index().rename(columns={"index":"Taxon"}),
                    use_container_width=True, height=320
                )
            else:
                st.info("No samples in the low-risk bucket.")

# ---------- SENSORY ----------
with tab_sens:
    st.markdown("### 👃 Sensory")
    st.caption(
        "If your CSV includes sensory columns, they’re plotted directly. "
        "Otherwise we show a generalized early➜late pattern and **smell guidance** for each sample."
    )

    has_sens = any(c in df_raw.columns for c in SENS_COLS)

    # A) Per-item smell guidance cards (risk-based, nudged by real sensory if present)
    st.markdown("#### Smell guidance by sample")
    for i, row in df_raw.iterrows():
        item = str(row.get(ITEM_COL, i))
        env  = str(row.get(PTYPE_COL, ""))
        risk = "not-safe" if pred_class[i] == "not-safe" else "safe"
        nudged = one_liner_from_sensory(row)
        summary = nudged if nudged else SMELL_GUIDE[risk]
        st.markdown(f"**{item}** — {env} — *{risk}*  \n{summary}")

    st.markdown("---")

    # B) Sensory plots
    if has_sens and DAY_COL:
        melt = df_raw.melt(id_vars=[c for c in [PTYPE_COL, DAY_COL] if c in df_raw.columns],
                           value_vars=[c for c in SENS_COLS if c in df_raw.columns],
                           var_name="Descriptor", value_name="Intensity")
        if PTYPE_COL in melt.columns:
            fig = px.line(
                melt, x=DAY_COL, y="Intensity", color="Descriptor", facet_col=PTYPE_COL,
                markers=True, color_discrete_sequence=PALETTE, title="Sensory Intensity Over Time (Days)"
            )
        else:
            fig = px.line(
                melt, x=DAY_COL, y="Intensity", color="Descriptor",
                markers=True, color_discrete_sequence=PALETTE, title="Sensory Intensity Over Time (Days)"
            )
        st.plotly_chart(fig, use_container_width=True)
    else:
        # generalized early vs late bar
        st.info("No usable sensory columns found; showing generalized early vs late profile.")
        trend = pd.DataFrame({
            "Descriptor":["Etheral","Fermented","Prickly","Rancid","Sulfurous","Old_cheese"],
            "Early":[0.4,0.3,0.2,0.2,0.1,0.2],
            "Late":[1.2,1.8,1.5,1.3,0.5,1.0],
        })
        df_plot = trend.melt(id_vars="Descriptor", var_name="Stage", value_name="Intensity")
        fig = px.bar(
            df_plot, x="Descriptor", y="Intensity", color="Stage", barmode="group",
            color_discrete_sequence=["#a3c1da","#e75480"], title="Generalized Sensory Profile (Early vs Late)"
        )
        st.plotly_chart(fig, use_container_width=True)
