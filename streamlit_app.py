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
/* Force Streamlit into light mode */
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stSidebar"], .stApp {
  background-color: #fff9fc !important;
  color: #4d004d !important;
}

/* Override dark theme elements */
[data-testid="stSidebar"], [data-testid="stSidebarNav"] {
  background: #ffeef8 !important;
  color: #4d004d !important;
}

[data-testid="stHeader"] {
  background: #fff9fc !important;
}

h1,h2,h3,h4,p,span,div {
  color: #4d004d !important;
}

/* Fix buttons and scrollbars */
.stButton>button {
  background-color: #ffb6c1 !important;
  color: #4d004d !important;
  border:none !important;
  border-radius:12px !important;
  font-weight:700 !important;
}
.stButton>button:hover {
  background-color: #ffc8d9 !important;
  color:#000 !important;
}

::-webkit-scrollbar { width: 10px; }
::-webkit-scrollbar-thumb { background-color: #e75480; border-radius: 10px; }

</style>
""", unsafe_allow_html=True)

st.markdown(
    """
<style>
  .stApp { background: linear-gradient(180deg,#ffeef8 0%,#fff9fc 100%); color:#4d004d; }
  h1,h2,h3 { color:#e75480; }
  h1 { font-size: 1.6rem; margin: .2rem 0 1.0rem 0; }
  .stTabs [data-baseweb="tab-list"] { gap: .25rem; }
  .stTabs [data-baseweb="tab"] { padding: .4rem .7rem; font-weight: 700; }
  .stButton button{
    background:#ffb6c1;color:#4d004d;border:none;border-radius:12px;
    padding:.55rem 1.2rem;font-weight:700;box-shadow:0 1px 8px rgba(231,84,128,.25);
  }
  .stButton button:hover{ background:#ffc8d9;color:#000; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🥩🍖 Microbe-Driven Spoilage")

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.header("📁 Artifacts & Data")

    # Always look for artifacts inside the repo (works on Streamlit Cloud)
    default_artifacts = str((Path(__file__).parent / "artifacts").resolve())
    artifacts_dir_str = st.text_input("Artifacts folder", value=default_artifacts, help="Contains rf_model_tuned.joblib and model_meta.json")

    st.markdown("**Upload CSV (same schema as model training)**")
    csv_file = st.file_uploader(" ", type=["csv"])

    st.markdown("### ⚙️ Settings")
    st.markdown("**Classification threshold:** **0.50**")
    st.caption("Used for classification and metrics; labels are shown only on the Predictions tab.")

    st.markdown(
        """
        ### ⚠️ RUO Disclaimer
        This model was trained on **pork** and **poultry** datasets.  
        Predictions are for **Research Use Only** — **not** intended for safety, clinical, or regulatory use.
        """
    )

# ------------------ LOAD ARTIFACTS ------------------
artifacts_dir = Path(artifacts_dir_str).expanduser()
need = ["rf_model_tuned.joblib", "model_meta.json"]
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
PALETTE = ["#d5b4ab", "#c3b1e1", "#a3c1da", "#a8c69f", "#f4c2c2"]
prob_thr = 0.50

# ------------------ DATA ------------------
if csv_file is None:
    st.info("Upload a CSV in the sidebar to generate predictions, performance metrics, microbiome views, and sensory guidance.")
    st.stop()

try:
    df_raw = pd.read_csv(csv_file)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

# column names we reference
ITEM_COL = "Sample_Name_y"
PTYPE_COL = "EnvType"
DAY_COL = "Day_numeric" if "Day_numeric" in df_raw.columns else ("Days_Numeric" if "Days_Numeric" in df_raw.columns else None)
LOG_CFU_COL = "Total mesophilic aerobic flora (log10 CFU.g-1)"
SENSORY_COLS = [c for c in ["Etheral", "Fermented", "Prickly", "Rancid", "Sulfurous", "Old_cheese"] if c in df_raw.columns]

# ------------------ PREDICTIONS (computed once) ------------------
X = df_raw.reindex(columns=feature_names, fill_value=0)
for c in X.columns:
    if not np.issubdtype(X[c].dtype, np.number):
        X[c] = pd.to_numeric(X[c], errors="coerce")
X = X.fillna(0.0)

try:
    pred_score = clf.predict_proba(X)[:, 1]
except Exception as e:
    st.error(f"Prediction failed: {e}")
    pred_score = np.zeros(len(X))  # fail-safe

# make scores available across tabs (prevents rerun NameError)
st.session_state["pred_score"] = pred_score

pred_class = np.where(pred_score >= prob_thr, "not-safe", "safe")
safe_conf = (prob_thr - pred_score) / prob_thr
notsafe_conf = (pred_score - prob_thr) / (1.0 - prob_thr)
confidence = np.clip(np.nan_to_num(np.where(pred_class == "safe", safe_conf, notsafe_conf)), 0.0, 1.0)

# ------------------ TABS ------------------
tab_pred, tab_perf, tab_micro, tab_sens = st.tabs(["🔮 Predictions", "📊 Performance", "🧬 Microbiome", "👃 Sensory"])

# ---------- PREDICTIONS ----------
def color_pred(val: str):
    v = str(val).lower()
    if v in ("unsafe", "not-safe"):
        return "background-color:#fde8e8; color:#9b1c1c; font-weight:600;"
    if v in ("safe", "low risk"):
        return "background-color:#e8f5e9; color:#1b5e20; font-weight:600;"
    return ""

HEADER_STYLE = [{"selector": "th",
                 "props": [("background-color", "#ffeef8"),
                           ("color", "#4d004d"),
                           ("font-weight", "700")]}]

with tab_pred:
    st.markdown("#### 🔍 Predictions")
    st.caption("Classification of each uploaded sample as *safe* or *not-safe* with confidence (threshold 0.50).")

    disp = pd.DataFrame({
        "Item": df_raw.get(ITEM_COL, ""),
        "Product Type": df_raw.get(PTYPE_COL, ""),
        "Days": df_raw.get(DAY_COL, ""),
        "Prediction": pred_class,
        "Confidence": confidence
    })
    if LOG_CFU_COL in df_raw.columns:
        disp["Log CFU (Input)"] = df_raw[LOG_CFU_COL]

    fmt = {"Confidence": "{:.0%}"} if "Confidence" in disp.columns else {}
    styler = (
        disp.style
            .applymap(color_pred, subset=["Prediction"])
            .format(fmt)
            .set_table_styles(HEADER_STYLE)
    )
    # Note: st.table supports Styler; st.dataframe ignores it. Use st.table for styling.
    st.table(styler)

# ---------- PERFORMANCE ----------
with tab_perf:
    st.markdown("#### 📈 Performance")
    st.caption("Compares predicted labels with ground-truth CFU values (≥ 7 = not-safe). No per-sample labels shown here.")

    if LOG_CFU_COL not in df_raw.columns:
        st.warning(f"Ground truth column '{LOG_CFU_COL}' not found — metrics unavailable.")
    else:
        y_true = np.where(df_raw[LOG_CFU_COL] >= GT_THRESHOLD, "not-safe", "safe")
        y_pred = pred_class
        labels = ["safe", "not-safe"]

        report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="True"),
                           x=labels, y=labels, color_continuous_scale="Reds")
        fig_cm.update_layout(title="Confusion Matrix", coloraxis_showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)

# ---------- MICROBIOME ----------
with tab_micro:
    st.markdown("#### 🧫 Microbiome Relative Abundance")
    st.caption("Shows composition by sample. If your file lacks numeric abundance columns, this view will be empty.")

    NON_MICROBE_COLS = {ITEM_COL, PTYPE_COL, DAY_COL, LOG_CFU_COL, "Prediction", "Confidence"}
    micro_cols = [c for c in df_raw.columns if c not in NON_MICROBE_COLS and np.issubdtype(df_raw[c].dtype, np.number)]

    if not micro_cols:
        st.info("No numeric microbial columns detected.")
    else:
        micro_df = df_raw[micro_cols]
        row_totals = micro_df.sum(axis=1).replace(0, np.nan)
        rel_abund = micro_df.div(row_totals, axis=0).fillna(0.0)

        # sample picker
        labels_series = df_raw.get(ITEM_COL, df_raw.index.astype(str)).astype(str)
        sel = st.selectbox("Select a sample", list(labels_series), index=0)
        match_idx = np.where(labels_series.values == str(sel))[0]
        idx = int(match_idx[0]) if len(match_idx) else 0

        topN = st.slider("Top microbes (N)", 5, 20, 10, 1)
        series = (rel_abund.iloc[idx].sort_values(ascending=False).head(topN) * 100.0)
        df_top = series.rename("Relative_Abundance_%").reset_index().rename(columns={"index": "Microbe"})

        left, right = st.columns([2,1], gap="large")
        with left:
            fig = px.bar(
                df_top, x="Microbe", y="Relative_Abundance_%", color="Microbe",
                color_discrete_sequence=(PALETTE * ((len(df_top)//len(PALETTE))+1))[:len(df_top)],
                title=f"Top {topN} microbes — {sel}"
            )
            fig.update_layout(yaxis_title="Relative abundance (%)", xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.markdown("**Top microbes (this sample)**")
            st.dataframe(df_top, use_container_width=True)

# ---------- SENSORY ----------
with tab_sens:
    st.markdown("#### 👃 Sensory")
    st.caption("If the CSV includes sensory columns, they’re plotted directly. Otherwise a generalized early➜late pattern is shown. No safety labels here.")

    # 1) Smell guidance (per item) — uses probability only (no safe/not-safe wording)
    st.markdown("##### 🧭 Smell guidance (per item)")
    def guidance_from_prob(p: float) -> str:
        if p >= 0.80:
            return "⚠️ High risk — watch for **fermented**, **rancid**, **cheesy**, or **sulfurous** notes."
        if p >= 0.50:
            return "⚠️ Elevated risk — **fermented**/**rancid** may emerge soon."
        return "🧊 Low risk — expect faint **sweet/ethereal**, mild **fermented** aroma."

    scores = st.session_state.get("pred_score", np.zeros(len(df_raw)))
    labels_series = df_raw.get(ITEM_COL, df_raw.index.astype(str)).astype(str)
    pick = st.selectbox("Choose an item", list(labels_series), index=0, key="smell_pick")
    match_idx = np.where(labels_series.values == str(pick))[0]
    idx = int(match_idx[0]) if len(match_idx) else 0
    st.write(guidance_from_prob(float(scores[idx])))

    with st.expander("Show smell guidance for all items"):
        dd = pd.DataFrame({
            "Item": labels_series,
            "Product Type": df_raw.get(PTYPE_COL, ""),
            "Prob (not-safe)": np.round(scores, 3),
            "Guidance": [guidance_from_prob(float(p)) for p in scores]
        })
        # No "Prediction" column here (labels are only on Predictions tab)
        st.dataframe(dd, use_container_width=True)

    # 2) Always show an Early vs Late bar (generalized template)
    st.markdown("##### Early vs Late (generalized)")
    bar_template = pd.DataFrame({
        "Descriptor": ["Etheral","Fermented","Old_cheese","Prickly","Rancid","Sulfurous"],
        "Early":      [0.5, 1.0, 0.2, 0.45, 0.25, 0.10],
        "Late":       [1.4, 1.4, 0.55, 1.4, 0.55, 0.50],
    })
    df_bar = bar_template.melt(id_vars="Descriptor", var_name="Stage", value_name="Mean intensity")
    fig_bar = px.bar(
        df_bar, x="Descriptor", y="Mean intensity", color="Stage",
        barmode="group", color_discrete_sequence=["#a3c1da","#e75480"],
        title=""
    )
    st.plotly_chart(fig_bar, use_container_width=True)
