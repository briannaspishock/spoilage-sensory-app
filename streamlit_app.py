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
    st.caption("10 rows per page. Prediction cells are color-coded; confidence is for the predicted class.")

    # ---- base table (no extra label col) ----
    disp = pd.DataFrame({
        "Item": df_raw.get(ITEM_COL, ""),
        "Product Type": df_raw.get(PTYPE_COL, ""),
        "Days": df_raw.get(DAY_COL, ""),
        "Prediction": pred_class,
        "Confidence": (confidence * 100).round(0).astype(int),  # as %
    })
    if LOG_CFU_COL in df_raw.columns:
        disp["Log CFU (Input)"] = df_raw[LOG_CFU_COL]

    # optional ordering (risk-aware)
    scores = st.session_state.get("pred_score", np.zeros(len(disp)))
    order = st.selectbox(
        "Order",
        ["Original order", "Highest risk first", "Lowest risk first"],
        index=0,
        help="Sort by model probability of not-safe."
    )
    if order == "Highest risk first":
        disp = disp.iloc[np.argsort(-scores)].reset_index(drop=True)
    elif order == "Lowest risk first":
        disp = disp.iloc[np.argsort(scores)].reset_index(drop=True)

    # ---- pagination (10 per page) ----
    PAGE_SIZE = 10
    total_pages = max(1, int(np.ceil(len(disp) / PAGE_SIZE)))
    page = st.selectbox("Page", list(range(1, total_pages + 1)), index=0)
    start, end = (page - 1) * PAGE_SIZE, (page * PAGE_SIZE)
    disp_page = disp.iloc[start:end].reset_index(drop=True)

    st.write(f"Showing items **{start + 1}–{min(end, len(disp))}** of **{len(disp)}**")

    # ---- HTML table with cell coloring + scroll ----
    # NOTE: st.markdown with unsafe_allow_html allows us to color just the Prediction cell,
    # and wrap in a scrollable container for a visible scrollbar.
    def _row_html(row):
        pred = str(row["Prediction"]).lower()
        pred_class = "pred-safe" if pred == "safe" else "pred-unsafe"
        cells = [
            f"<td>{row['Item']}</td>",
            f"<td>{row['Product Type']}</td>",
            f"<td>{'' if pd.isna(row['Days']) else row['Days']}</td>",
            f"<td class='{pred_class}'><b>{row['Prediction']}</b></td>",
            f"<td>{'' if pd.isna(row['Confidence']) else f'{int(row['Confidence'])}%'}</td>",
        ]
        if "Log CFU (Input)" in disp_page.columns:
            cfu_val = row["Log CFU (Input)"]
            cells.append(f"<td>{'' if pd.isna(cfu_val) else round(float(cfu_val), 2)}</td>")
        return "<tr>" + "".join(cells) + "</tr>"

    headers = ["Item", "Product Type", "Days", "Prediction", "Confidence"]
    if "Log CFU (Input)" in disp_page.columns:
        headers.append("Log CFU (Input)")

    table_html = [
        """
        <style>
          .pred-table-wrap {
            max-height: 430px;           /* visible area for ~10 rows */
            overflow-y: auto;            /* <-- explicit scrollbar */
            border: 1px solid #f1d7e1;
            border-radius: 10px;
            background: #fff9fc;
          }
          table.pred-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.95rem;
          }
          table.pred-table th, table.pred-table td {
            padding: 8px 10px;
            border-bottom: 1px solid #f5e7ee;
            text-align: left;
            color: #4d004d;
            background: #ffffff;
          }
          table.pred-table thead th {
            position: sticky;
            top: 0;
            z-index: 1;
            background: #ffeef8;
            font-weight: 700;
          }
          td.pred-safe   { background:#e8f5e9; color:#1b5e20; border-left: 4px solid #1b5e20; }
          td.pred-unsafe { background:#fde8e8; color:#9b1c1c; border-left: 4px solid #9b1c1c; }
        </style>
        <div class="pred-table-wrap">
          <table class="pred-table">
            <thead><tr>
        """
    ]
    table_html += [f"<th>{h}</th>" for h in headers]
    table_html += ["</tr></thead><tbody>"]

    for _, r in disp_page.iterrows():
        table_html.append(_row_html(r))

    table_html += ["</tbody></table></div>"]
    st.markdown("".join(table_html), unsafe_allow_html=True)


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
