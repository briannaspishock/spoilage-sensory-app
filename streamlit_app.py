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
import streamlit as st
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report

# ===================== PAGE STYLE =====================
st.set_page_config(page_title="Microbe-Driven Spoilage", page_icon="🥩", layout="wide")
st.markdown("""
<style>
  .stApp { background: linear-gradient(180deg,#ffeef8 0%,#fff9fc 100%); color:#4d004d; }
  h1 { color:#e75480; font-size:1.65rem; line-height:1.15; margin-bottom:.25rem; }
  h2,h3 { color:#e75480; }
  .stButton button{
    background:#ffb6c1;color:#4d004d;border:none;border-radius:12px;
    padding:.55rem 1.2rem;font-weight:700;box-shadow:0 1px 8px rgba(231,84,128,.25);
  }
  .stButton button:hover{ background:#ffc8d9; color:#000; }
  .soft-card { border-radius:12px; padding:12px 14px; margin:.5rem 0; }
</style>
""", unsafe_allow_html=True)

st.title("🥩🍖 Microbe-Driven Spoilage")
st.caption("Upload a CSV to get predictions, performance metrics, microbiome summaries, and per-item sensory guidance.")

# ===================== CONSTANTS =====================
PALETTE = ["#d5b4ab","#c3b1e1","#a3c1da","#a8c69f","#f4c2c2"]
SENS_COLS = ["Etheral","Fermented","Prickly","Rancid","Sulfurous","Old_cheese"]
ITEM_COL = "Sample_Name_y"
PTYPE_COL = "EnvType"
DAY_CANDIDATES = ["Day_numeric","Days_Numeric"]
LOG_CFU_COL = "Total mesophilic aerobic flora (log10 CFU.g-1)"
PROB_THR = 0.50   # fixed; shown in sidebar

# ===================== SIDEBAR =====================
with st.sidebar:
    st.header("🧰 Artifacts & Data")
    default_artifacts = str((Path(__file__).parent / "artifacts").resolve())
    artifacts_dir_str = st.text_input("Artifacts folder", value=default_artifacts,
                                      help="Folder must contain: rf_model_tuned.joblib, model_meta.json")
    csv_file = st.file_uploader("Upload CSV (same schema as model training)", type=["csv"])

    st.markdown("---")
    st.subheader("⚙️ Settings")
    st.markdown(f"**Classification threshold:** `{PROB_THR:.2f}`")

    st.markdown("""
    ---
    **⚠️ RUO Disclaimer**  
    This model was trained on **pork and poultry** datasets. Predictions are for **Research Use Only** and not intended for safety or regulatory decisions.
    """)

# ===================== LOAD ARTIFACTS =====================
artifacts_dir = Path(artifacts_dir_str).expanduser()
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

# ===================== LOAD DATA =====================
if csv_file is None:
    st.info("Upload your dataset in the sidebar to generate spoilage predictions and sensory insights.")
    st.stop()

try:
    df_raw = pd.read_csv(csv_file)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

# Pick a day column if present
DAY_COL = next((c for c in DAY_CANDIDATES if c in df_raw.columns), None)

# ===================== PREDICTIONS =====================
# Align to training features; coerce numerics
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

# Confidence measure (0..1) relative to fixed threshold
safe_conf = (PROB_THR - prob_not_safe) / PROB_THR
notsafe_conf = (prob_not_safe - PROB_THR) / (1.0 - PROB_THR)
confidence = np.where(pred_class == "safe", safe_conf, notsafe_conf)
confidence = np.clip(np.nan_to_num(confidence), 0.0, 1.0)

# ===================== HELPERS =====================
def style_pred_table(row):
    """Color 'Prediction' cell red/green and make Item bold for readability."""
    styles = [''] * len(row)
    try:
        pred_idx = row.index.get_loc("Prediction")
        item_idx = row.index.get_loc("Item")
    except KeyError:
        return styles

    if row["Prediction"] == "not-safe":
        styles[pred_idx] = "background-color:#fde8e8; color:#9b1c1c; font-weight:700;"
    else:
        styles[pred_idx] = "background-color:#e8f5e9; color:#1b5e20; font-weight:700;"

    styles[item_idx] = "font-weight:700;"
    return styles

def smell_tip(prob: float, env: str) -> str:
    """Short, friendly smell guidance based on risk + product type."""
    env_l = (env or "").strip().lower()
    high = prob >= PROB_THR
    if env_l == "pork":
        return ("⚠️ High risk — watch for **fermented**, **rancid**, **cheesy**, or **sulfurous** notes."
                if high else
                "🧊 Low risk — faint **sweet/ethereal**, mild **fermented** aroma.")
    if env_l == "poultry":
        return ("⚠️ High risk — **vinegary/prickly**, **sulfur/egg**, or **cheesy** odor."
                if high else
                "🧊 Low risk — mild **sweet**, slightly **fermented** or neutral.")
    return ("⚠️ High risk — strong **fermented**, **rancid**, or **sulfurous** smell."
            if high else
            "🧊 Low risk — faint **sweet** or **neutral** aroma.")

# ===================== TABS =====================
tab_pred, tab_perf, tab_micro, tab_sens = st.tabs(
    ["🔮 Predictions", "📊 Performance", "🧬 Microbiome", "👃 Sensory"]
)

# ---------- PREDICTIONS ----------
with tab_pred:
    st.markdown("#### 🔍 Model Predictions")
    st.caption("Each sample is classified as **safe / not-safe** at a fixed threshold of 0.50. Confidence is relative to that threshold.")

    disp = pd.DataFrame({
        "Item": df_raw.get(ITEM_COL, ""),
        "Product Type": df_raw.get(PTYPE_COL, ""),
        "Days": df_raw.get(DAY_COL, "") if DAY_COL else "",
        "Prediction": pred_class,
        "Confidence": confidence
    })
    if LOG_CFU_COL in df_raw.columns:
        disp["Log CFU (Input)"] = df_raw[LOG_CFU_COL]

    st.dataframe(
        disp.style.apply(style_pred_table, axis=1).format({"Confidence": "{:.1%}"}),
        use_container_width=True
    )

    st.download_button(
        "⬇️ Download predictions",
        disp.to_csv(index=False).encode("utf-8"),
        "spoilage_predictions.csv",
        "text/csv"
    )

# ---------- PERFORMANCE ----------
with tab_perf:
    st.markdown("#### 📈 Model Performance Metrics")
    st.caption(f"Compares predictions to ground truth if **{LOG_CFU_COL}** is present (≥ {GT_THRESHOLD:g} = not-safe).")

    if LOG_CFU_COL not in df_raw.columns:
        st.info(f"Ground-truth column **'{LOG_CFU_COL}'** not found — metrics unavailable for this upload.")
    else:
        y_true = np.where(df_raw[LOG_CFU_COL] >= GT_THRESHOLD, "not-safe", "safe")
        y_pred = pred_class
        labels = ["safe", "not-safe"]

        try:
            report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate classification report: {e}")

        try:
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            fig_cm = px.imshow(
                cm, text_auto=True,
                labels=dict(x="Predicted", y="True"),
                x=labels, y=labels,
                color_continuous_scale="Reds"
            )
            fig_cm.update_layout(title="Confusion Matrix", coloraxis_showscale=False)
            st.plotly_chart(fig_cm, use_container_width=True, key="cm_plot")
        except Exception as e:
            st.error(f"Could not render confusion matrix: {e}")

# ---------- MICROBIOME ----------
with tab_micro:
    st.markdown("#### 🧫 Microbiome Relative Abundance")
    st.caption("Shows top taxa per sample. If your file lacks abundance columns, this tab will say so and skip plots.")

    # treat all numeric, non-meta columns as microbes
    NON_MICROBE = {ITEM_COL, PTYPE_COL, LOG_CFU_COL, "Prediction", "Confidence"}
    if DAY_COL: NON_MICROBE.add(DAY_COL)

    micro_cols = [c for c in df_raw.columns
                  if c not in NON_MICROBE and np.issubdtype(df_raw[c].dtype, np.number)]

    if not micro_cols:
        st.info("No numeric microbial abundance columns detected in your CSV.")
    else:
        micro_df = df_raw[micro_cols].copy()
        totals = micro_df.sum(axis=1).replace(0, np.nan)
        rel_abund = micro_df.div(totals, axis=0).fillna(0.0)

        sample_labels = df_raw.get(ITEM_COL, df_raw.index.astype(str))
        sel = st.selectbox("Select a sample", sample_labels.astype(str).tolist(), index=0, key="sample_micro")
        if ITEM_COL in df_raw.columns:
            idx = df_raw.index[sample_labels.astype(str) == sel][0]
        else:
            idx = int(sel)

        series = (rel_abund.iloc[idx].sort_values(ascending=False).head(10) * 100.0)
        df_top = series.rename("Relative Abundance %").reset_index().rename(columns={"index": "Microbe"})

        fig = px.bar(
            df_top, x="Microbe", y="Relative Abundance %",
            color="Microbe",
            color_discrete_sequence=(PALETTE * ((len(df_top)//len(PALETTE))+1))[:len(df_top)],
            title=f"Top 10 microbes — {sel}"
        )
        fig.update_layout(yaxis_title="Relative abundance (%)", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True, key=f"micro_{sel}")

        # optional mean-by-day plot
        if DAY_COL:
            rel_w_day = rel_abund.copy()
            rel_w_day[DAY_COL] = pd.to_numeric(df_raw[DAY_COL], errors="coerce")
            top_global = rel_w_day[micro_cols].mean().sort_values(ascending=False).head(7).index.tolist()
            by_day = rel_w_day.groupby(DAY_COL)[top_global].mean().reset_index()
            df_time = by_day.melt(id_vars=DAY_COL, var_name="Genus", value_name="Mean_RelAbund_%")
            df_time["Mean_RelAbund_%"] *= 100.0
            fig_tc = px.line(
                df_time, x=DAY_COL, y="Mean_RelAbund_%", color="Genus",
                markers=True, color_discrete_sequence=PALETTE,
                title="Genus time-course (mean relative abundance by day)"
            )
            fig_tc.update_layout(xaxis_title="Day", yaxis_title="Mean relative abundance (%)")
            st.plotly_chart(fig_tc, use_container_width=True, key="timecourse")

# ---------- SENSORY (PER-ITEM DROPDOWN) ----------
with tab_sens:
    st.markdown("#### 👃 Sensory")
    st.caption(
        "If your CSV includes sensory columns, we plot them. "
        "Otherwise we show a generalized early→late pattern.\n\n"
        "**Use the dropdown below for per-item smell guidance.**"
    )

    # A) Per-item smell guidance (dropdown)
    st.markdown("##### 🧠 Smell guidance (per item)")
    labels_series = df_raw.get(ITEM_COL, pd.Series(df_raw.index.astype(str), index=df_raw.index))
    labels = labels_series.astype(str).tolist()
    selected_label = st.selectbox("Pick a sample", labels, index=0, key="smell_select")

    if ITEM_COL in df_raw.columns:
        row_idx = df_raw.index[labels_series.astype(str) == selected_label][0]
    else:
        row_idx = int(selected_label)

    env = str(df_raw.loc[row_idx].get(PTYPE_COL, ""))
    p = float(prob_not_safe[row_idx])
    pred_lbl = "not-safe" if p >= PROB_THR else "safe"
    guidance = smell_tip(p, env)

    st.markdown(
        f"""
<div class="soft-card" style="background:{'#fde8e8' if pred_lbl=='not-safe' else '#e8f5e9'}; 
                               color:{'#9b1c1c' if pred_lbl=='not-safe' else '#1b5e20'};">
  <b>{selected_label}</b> — {env or '—'}<br>
  <span style="font-weight:700;">Prediction:</span> {pred_lbl}<br>
  <span style="font-weight:700;">Probability (not-safe):</span> {p:.2f}<br>
  {guidance}
</div>
""",
        unsafe_allow_html=True
    )

    # Nudge with actual sensory values if present
    has_sens_cols = any(c in df_raw.columns for c in SENS_COLS)
    if has_sens_cols:
        vals = df_raw.loc[row_idx, [c for c in SENS_COLS if c in df_raw.columns]]
        if hasattr(vals, "notna") and vals.notna().any():
            top = vals.sort_values(ascending=False).head(3)
            st.caption(
                "Top measured sensory intensities (this sample): " +
                ", ".join([f"**{k}** {float(v):.2f}" for k, v in top.items()])
            )

    st.markdown("---")

    # B) Sensory plots
    if has_sens_cols and DAY_COL:
        melt = df_raw.melt(
            id_vars=[c for c in [PTYPE_COL, DAY_COL] if c in df_raw.columns],
            value_vars=[c for c in SENS_COLS if c in df_raw.columns],
            var_name="Descriptor", value_name="Intensity"
        )
        if PTYPE_COL in melt.columns:
            fig = px.line(
                melt, x=DAY_COL, y="Intensity", color="Descriptor", facet_col=PTYPE_COL,
                markers=True, color_discrete_sequence=PALETTE,
                title="Sensory intensity over time (by product type)"
            )
        else:
            fig = px.line(
                melt, x=DAY_COL, y="Intensity", color="Descriptor",
                markers=True, color_discrete_sequence=PALETTE,
                title="Sensory intensity over time"
            )
        st.plotly_chart(fig, use_container_width=True, key="sens_time")
    else:
        # Generalized early vs late bar chart
        trend = pd.DataFrame({
            "Descriptor":["Etheral","Fermented","Prickly","Rancid","Sulfurous","Old_cheese"],
            "Early":[0.4,0.3,0.2,0.2,0.1,0.2],
            "Late":[1.2,1.8,1.5,1.3,0.5,1.0],
        })
        df_plot = trend.melt(id_vars="Descriptor", var_name="Stage", value_name="Intensity")
        fig = px.bar(
            df_plot, x="Descriptor", y="Intensity", color="Stage",
            barmode="group", color_discrete_sequence=["#a3c1da","#e75480"],
            title="Generalized sensory profile (Early vs Late)"
        )
        st.plotly_chart(fig, use_container_width=True, key="sens_general")

    # C) (Optional) Batch guidance (collapsed)
    with st.expander("Show smell guidance for all items"):
        rows = []
        for i, lbl in enumerate(labels):
            env_i = str(df_raw.iloc[i].get(PTYPE_COL, ""))
            p_i = float(prob_not_safe[i])
            pred_i = "not-safe" if p_i >= PROB_THR else "safe"
            rows.append({
                "Item": lbl,
                "Product Type": env_i,
                "Prediction": pred_i,
                "Prob (not-safe)": round(p_i, 3),
                "Guidance": smell_tip(p_i, env_i)
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=320)
