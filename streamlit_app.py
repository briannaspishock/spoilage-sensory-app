#!/usr/bin/env python3
import json
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px

# =========================
# THEME / PAGE SETUP
# =========================
st.set_page_config(page_title="Microbe-Driven Spoilage", page_icon="🥩", layout="wide")
st.markdown("""
<style>
  .stApp { background: linear-gradient(180deg, #ffeef8 0%, #fff9fc 100%); color:#4d004d; }
  h1,h2,h3 { color:#74264d; }
  .stButton button{ background:#ffb6c1; color:#4d004d; border:none; border-radius:12px;
    padding:.55rem 1.2rem; font-weight:700; box-shadow:0 1px 8px rgba(231,84,128,.25);}
  .stButton button:hover{ background:#ffc8d9; color:#000; }
  .small-note { color:#5a3b53; font-size:0.92rem; }
</style>
""", unsafe_allow_html=True)

st.title("🥩🍗 Microbe-Driven Spoilage")
st.caption("Upload a CSV to get predictions, performance, sensory guidance, and microbiome views (if present).")

# =========================
# CONSTANTS & UTILITIES
# =========================
APP_DIR        = Path(__file__).parent.resolve()
ARTIFACTS_DIR  = APP_DIR / "artifacts"      # <— always use repo artifacts
MODEL_FILE     = ARTIFACTS_DIR / "rf_model_tuned.joblib"
META_FILE      = ARTIFACTS_DIR / "model_meta.json"
REQUIRED_FILES = [MODEL_FILE, META_FILE]
PROB_THR       = 0.50                       # fixed threshold (Jed’s convention)
LOG_CFU_COL    = "Total mesophilic aerobic flora (log10 CFU.g-1)"  # exact header
PALETTE        = ["#d5b4ab", "#c3b1e1", "#a3c1da", "#a8c69f", "#f4c2c2"]

def load_artifacts():
    """Load RF classifier + meta safely."""
    for f in REQUIRED_FILES:
        if not f.exists():
            raise FileNotFoundError(f"Missing artifact: {f.name} in {ARTIFACTS_DIR}")
    clf  = joblib.load(MODEL_FILE)
    meta = json.loads(META_FILE.read_text())
    return clf, meta

def as_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if not np.issubdtype(out[c].dtype, np.number):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.fillna(0.0)

def explain_tab(text: str):
    st.markdown(f"<div class='small-note'>ℹ️ {text}</div>", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("🧰 Artifacts & Settings")

    st.markdown(f"**Artifacts folder (fixed):** `{ARTIFACTS_DIR}`")
    if all(p.exists() for p in REQUIRED_FILES):
        st.success("Artifacts detected ✅")
    else:
        st.error("Artifacts missing. Ensure `artifacts/rf_model_tuned.joblib` and `artifacts/model_meta.json` exist.")

    st.markdown("---")
    st.subheader("🔒 Classification threshold")
    st.write(f"Probability for **“not-safe”** is fixed at **{PROB_THR:.2f}** (Jed RF).")

    st.markdown("---")
    st.subheader("⚠️ RUO Notice")
    st.write("This model was trained on **pork & poultry** samples only. All outputs are **Research Use Only** (RUO) — not for clinical/regulatory decisions.")

# =========================
# LOAD MODEL
# =========================
try:
    clf, meta = load_artifacts()
    feature_names = meta.get("feature_names", [])
except Exception as e:
    st.error(f"Failed to load model artifacts: {e}")
    st.stop()

# =========================
# UPLOAD DATA
# =========================
csv = st.file_uploader("Upload CSV (same schema as model training)", type=["csv"])
if csv is None:
    st.info("Upload a CSV to run predictions. The app will align your columns to the model’s feature set.")
    st.stop()

try:
    df_raw = pd.read_csv(csv)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

# =========================
# PREPARE FEATURES
# =========================
# Align strictly to the model’s features (everything else ignored for the classifier)
X = df_raw.reindex(columns=feature_names, fill_value=0)
X = as_numeric(X)

# Useful columns if present
ITEM_COL  = "Sample_Name_y" if "Sample_Name_y" in df_raw.columns else None
PTYPE_COL = "EnvType" if "EnvType" in df_raw.columns else None
DAY_COL   = "Day_numeric" if "Day_numeric" in df_raw.columns else ("Days_Numeric" if "Days_Numeric" in df_raw.columns else None)

# =========================
# PREDICTIONS
# =========================
try:
    prob_not_safe = clf.predict_proba(X)[:, 1]
except Exception as e:
    st.error(f"Model predict_proba failed: {e}")
    st.stop()

pred_label = np.where(prob_not_safe >= PROB_THR, "not-safe", "safe")

# Confidence (0..1) relative to the threshold
safe_conf    = (PROB_THR - prob_not_safe) / PROB_THR
notsafe_conf = (prob_not_safe - PROB_THR) / (1.0 - PROB_THR)
conf = np.where(pred_label == "safe", safe_conf, notsafe_conf)
conf = np.clip(np.nan_to_num(conf, nan=0.0), 0.0, 1.0)

# =========================
# TABS
# =========================
tab_pred, tab_perf, tab_micro, tab_sens = st.tabs(
    ["🔮 Predictions", "📊 Performance", "🧬 Microbiome", "👃 Sensory"]
)

# ---------- PREDICTIONS ----------
with tab_pred:
    explain_tab("This table reflects **Jed’s Random Forest** predictions on your uploaded CSV. "
                "Labels are derived from probability ≥ 0.50 → **not-safe**.")
    disp = pd.DataFrame({
        "Prediction": pred_label,
        "Prob_not_safe": np.round(prob_not_safe, 3),
        "Confidence": np.round(conf, 2),
    })

    if ITEM_COL:  disp.insert(0, "Item", df_raw[ITEM_COL].astype(str))
    if PTYPE_COL: disp.insert(1 if ITEM_COL else 0, "Product Type", df_raw[PTYPE_COL].astype(str))
    if DAY_COL:   disp.insert(2 if ITEM_COL and PTYPE_COL else 1, "Days", df_raw[DAY_COL])

    if LOG_CFU_COL in df_raw.columns:
        disp["Log CFU (input)"] = df_raw[LOG_CFU_COL]

    st.dataframe(disp, use_container_width=True)

    st.download_button(
        "⬇️ Download predictions (.csv)",
        disp.to_csv(index=False).encode("utf-8"),
        "spoilage_predictions.csv",
        "text/csv"
    )

    # Distribution plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(prob_not_safe, bins=25, edgecolor="white")
    ax.axvline(PROB_THR, color="#e75480", linestyle="--", linewidth=2)
    ax.set_xlabel("Predicted probability of 'not-safe'")
    ax.set_ylabel("Count")
    ax.set_title("Prediction probability distribution")
    st.pyplot(fig)

# ---------- PERFORMANCE ----------
with tab_perf:
    explain_tab("If your CSV includes the ground-truth column "
                f"**'{LOG_CFU_COL}'**, we compare predictions to the rule "
                "**log CFU ≥ 7 → not-safe**. If the column is missing, this tab shows guidance only.")
    if LOG_CFU_COL not in df_raw.columns:
        st.info(f"Ground-truth column **'{LOG_CFU_COL}'** not found — performance metrics skipped.")
    else:
        from sklearn.metrics import confusion_matrix, classification_report
        y_true = np.where(df_raw[LOG_CFU_COL] >= 7.0, "not-safe", "safe")
        y_pred = pred_label

        st.markdown("#### Classification report")
        report = classification_report(y_true, y_pred, labels=["safe", "not-safe"], output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(
            report_df.style.format({
                "precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}", "support": "{:.0f}"
            }),
            use_container_width=True
        )

        st.markdown("#### Confusion matrix")
        cm = confusion_matrix(y_true, y_pred, labels=["safe", "not-safe"])
        fig_cm = px.imshow(cm, text_auto=True, x=["safe", "not-safe"], y=["safe", "not-safe"],
                           labels=dict(x="Predicted", y="True"), color_continuous_scale="Reds")
        fig_cm.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)

# ---------- MICROBIOME ----------
with tab_micro:
    explain_tab("Shows **relative abundance** if your CSV contains numeric microbial columns "
                "(all numeric columns excluding common metadata).")
    NON_MICRO = {
        "Sample_Name_y","Sample_Name","EnvType","EnvTyp",
        "Day_numeric","Days_Numeric","Day","Days",
        "SRR_ID","SAMN_ID","SamplingTime",
        LOG_CFU_COL,
        # sensory names we don't want treated as microbes
        "Etheral","Fermented","Prickly","Rancid","Sulfurous","Old_cheese",
    }
    # plus any columns we used for model features (avoid double use if they’re not taxa)
    NON_MICRO.update(feature_names)

    micro_cols = [c for c in df_raw.columns
                  if c not in NON_MICRO and np.issubdtype(df_raw[c].dtype, np.number)]

    if not micro_cols:
        st.info("No numeric microbial columns detected.")
    else:
        micro = df_raw[micro_cols].copy()
        totals = micro.sum(axis=1).replace(0, np.nan)
        rel = (micro.div(totals, axis=0).fillna(0.0) * 100.0)

        # Sample selector
        labels = df_raw[ITEM_COL].astype(str) if ITEM_COL else df_raw.index.astype(str)
        sel = st.selectbox("Select a sample", options=list(labels), index=0, key="micro_sel")
        if ITEM_COL:
            idx = df_raw.index[df_raw[ITEM_COL].astype(str) == sel][0]
        else:
            idx = int(sel)

        topN = st.slider("Top microbes (N)", 5, 20, 10, 1, key="micro_topN")
        series = rel.iloc[idx].sort_values(ascending=False).head(topN)
        df_top = series.rename("Relative_Abundance_%").reset_index().rename(columns={"index": "Microbe"})

        fig = px.bar(df_top, x="Microbe", y="Relative_Abundance_%",
                     color="Microbe",
                     color_discrete_sequence=(PALETTE * ((len(df_top)//len(PALETTE))+1))[:len(df_top)],
                     title=f"Top {topN} microbes — {sel}")
        fig.update_layout(yaxis_title="Relative abundance (%)", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True, key=f"micro_{idx}")

        # Optional time-course by day if available
        if DAY_COL:
            rel2 = rel.copy()
            rel2[DAY_COL] = pd.to_numeric(df_raw[DAY_COL], errors="coerce")
            rel2 = rel2.dropna(subset=[DAY_COL])
            if not rel2.empty:
                top_by_mean = rel[micro_cols].mean().sort_values(ascending=False).head(min(7, len(micro_cols))).index
                df_time = rel2.groupby(DAY_COL)[list(top_by_mean)].mean().reset_index()
                df_long = df_time.melt(id_vars=DAY_COL, var_name="Genus", value_name="Mean_RelAbund_%")
                fig_tc = px.line(df_long, x=DAY_COL, y="Mean_RelAbund_%", color="Genus", markers=True,
                                 color_discrete_sequence=PALETTE, title="Genus time-course (mean by day)")
                fig_tc.update_layout(xaxis_title="Day", yaxis_title="Mean relative abundance (%)")
                st.plotly_chart(fig_tc, use_container_width=True)

# ---------- SENSORY ----------
with tab_sens:
    explain_tab("If your CSV has sensory columns, we summarize them by **safe vs not-safe** "
                "(based on Jed RF). If sensory columns are missing, we show generalized expectations.")
    sensory_cols = [c for c in ["Etheral","Fermented","Prickly","Rancid","Sulfurous","Old_cheese"]
                    if c in df_raw.columns and np.issubdtype(df_raw[c].dtype, np.number)]

    if not sensory_cols:
        st.info("No sensory columns found — showing generalized early/late expectations:\n\n"
                "• Early: lighter **Ethereal**, mild **Fermented**, low **Rancid/Sulfurous**.\n"
                "• Late: higher **Fermented/Prickly**, increasing **Rancid/Sulfurous**, stronger **Old_cheese**.")
    else:
        df_s = df_raw.copy()
        df_s["RiskGroup"] = np.where(prob_not_safe >= PROB_THR, "High (not-safe)", "Low (safe)")
        means = df_s.groupby(["RiskGroup"])[sensory_cols].mean().reset_index()
        df_long = means.melt(id_vars="RiskGroup", var_name="Descriptor", value_name="MeanIntensity")

        fig_bar = px.bar(
            df_long, x="Descriptor", y="MeanIntensity", color="RiskGroup",
            barmode="group", facet_col=None,
            color_discrete_map={
                "Low (safe)": "#a8c69f",
                "High (not-safe)": "#b91c1c"
            },
            title="Sensory profile by risk group"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.dataframe(means, use_container_width=True)

