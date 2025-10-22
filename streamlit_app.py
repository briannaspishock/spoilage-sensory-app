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

    # Default artifacts path: repo/artifacts (works on Streamlit Cloud)
    default_artifacts = str((Path(__file__).parent / "artifacts").resolve())
    artifacts_dir_str = st.text_input(
        "Artifacts folder",
        value=default_artifacts,
        help="Must contain rf_model_tuned.joblib, model_meta.json, rf_feature_importances.csv",
    )

    st.markdown("**Upload CSV**")
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

# <<< required files for the RF build >>>
need = ["rf_model_tuned.joblib", "model_meta.json", "rf_feature_importances.csv"]
if not artifacts_dir.exists() or any(not (artifacts_dir / f).exists() for f in need):
    st.warning(f"Pick a valid artifacts folder containing: {need}")
    st.stop()

@st.cache_resource
def load_artifacts(folder: Path):
    clf = joblib.load(folder / "rf_model_tuned.joblib")  # RF classifier
    meta = json.loads((folder / "model_meta.json").read_text())
    return clf, meta

try:
    clf, meta = load_artifacts(artifacts_dir)
except Exception as e:
    st.error(f"Failed to load artifacts: {e}")
    st.stop()

# Globals / defaults
LOG_CFU_COL = "Total mesophilic aerobic flora (log10 CFU.g-1)"  # With period
GT_THRESHOLD = float(meta.get("threshold_cfu", 7.0))
PALETTE = ["#d5b4ab", "#c3b1e1", "#a3c1da", "#a8c69f", "#f4c2c2"]
prob_thr = 0.50

# ---------------- DATA INPUT ----------------
if csv_file is None:
    st.info("Upload a CSV in the sidebar to generate predictions, performance metrics, microbiome views, and sensory guidance.")
    st.stop()

try:
    df_raw = pd.read_csv(csv_file)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

# ---------------- FEATURE NAME RESOLUTION ----------------
def resolve_feature_names(meta: dict, clf, df: pd.DataFrame) -> list:
    # 1) From model_meta.json
    if isinstance(meta.get("feature_names", None), (list, tuple)):
        return list(meta["feature_names"])
    # 2) From fitted sklearn model
    if hasattr(clf, "feature_names_in_"):
        return list(clf.feature_names_in_)
    # 3) Fallback: infer numeric columns from CSV (drop obvious non-features)
    drop_cols = {
        "Sample_Name_y", "EnvType", "Item", "Product Type",
        "Day_numeric", "Days_Numeric", LOG_CFU_COL
    }
    cand = [
        c for c in df.columns
        if c not in drop_cols and np.issubdtype(df[c].dtype, np.number)
    ]
    return cand

feature_names = resolve_feature_names(meta, clf, df_raw)
if not feature_names:
    st.error("No usable feature list found in meta/model; couldn’t infer from the CSV.")
    st.stop()

# ---------------- COLUMN ALIASES ----------------
ITEM_COL = "Sample_Name_y"
PTYPE_COL = "EnvType"
DAY_COL = "Day_numeric" if "Day_numeric" in df_raw.columns else ("Days_Numeric" if "Days_Numeric" in df_raw.columns else None)

# ---------------- CORE PREDICTIONS (compute once) ----------------
X = df_raw.reindex(columns=feature_names, fill_value=0)
for c in X.columns:
    if not np.issubdtype(X[c].dtype, np.number):
        X[c] = pd.to_numeric(X[c], errors="coerce")
X = X.fillna(0.0)

try:
    pred_score = clf.predict_proba(X)[:, 1]   # prob of class "not-safe"
except Exception as e:
    st.error(f"Failed to get predictions from your model: {e}")
    st.stop()

pred_class = np.where(pred_score >= prob_thr, "not-safe", "safe")

# Confidence for the predicted class
safe_conf = (prob_thr - pred_score) / prob_thr
notsafe_conf = (pred_score - prob_thr) / (1.0 - prob_thr)
confidence = np.clip(np.nan_to_num(np.where(pred_class == "safe", safe_conf, notsafe_conf)), 0.0, 1.0)

# also expose score for other tabs
st.session_state["pred_score"] = pred_score

# ---------------- TABS ----------------
tab_pred, tab_perf, tab_micro, tab_sens = st.tabs(["🔮 Predictions", "📊 Performance", "🧬 Microbiome", "👃 Sensory"])

# ====== PREDICTIONS ======
# ====== PREDICTIONS (scrollable + colored cells) ======
with tab_pred:
    st.markdown("### 📊 Predictions from Model")

    # -------- base table from computed outputs --------
    disp = pd.DataFrame({
        "Days in Refrigerator": df_raw.get(DAY_COL, ""),
        "Item": df_raw.get(ITEM_COL, ""),
        "Product Type": df_raw.get(PTYPE_COL, ""),
        "Prediction": pred_class,            # "safe" / "not-safe"
        "Confidence": confidence,            # 0–1 float
    })
    if LOG_CFU_COL in df_raw.columns:
        disp["Log CFU (Input)"] = df_raw[LOG_CFU_COL]

    # -------- controls: ordering + search + product type filter --------
    c1, c2, c3 = st.columns([2, 2, 3], gap="small")
    with c1:
        order = st.selectbox(
            "Order",
            ["Original order", "Highest risk first", "Lowest risk first"],
            index=0,
            help="Sort by model probability of not-safe."
        )
    with c2:
        q = st.text_input("Search item", "", placeholder="Type part of an item name…")
    with c3:
        ptype_opts = [str(x) for x in disp["Product Type"].dropna().unique() if str(x).strip()]
        ptype_opts = sorted(ptype_opts)
        ptype_pick = st.multiselect("Filter by Product Type", options=ptype_opts, default=[])

    scores = st.session_state.get("pred_score", np.zeros(len(disp)))
    if order == "Highest risk first":
        disp = disp.iloc[np.argsort(-scores)].reset_index(drop=True)
    elif order == "Lowest risk first":
        disp = disp.iloc[np.argsort(scores)].reset_index(drop=True)

    mask = np.ones(len(disp), dtype=bool)
    if q:
        qq = q.strip().lower()
        mask &= disp["Item"].astype(str).str.lower().str.contains(qq, na=False)
    if ptype_pick:
        mask &= disp["Product Type"].astype(str).isin(ptype_pick)
    disp = disp.loc[mask].reset_index(drop=True)

    st.write(f"Showing **{len(disp)}** items")

    # -------- HTML renderer (keeps colors AND adds scrollbar) --------
    def _row_html(row: pd.Series) -> str:
        pred = str(row.get("Prediction", "")).lower()
        bg = "#e8f5e9" if pred == "safe" else "#fde8e8"
        fg = "#1b5e20" if pred == "safe" else "#9b1c1c"
        # confidence as %, blank if NaN
        conf = "" if pd.isna(row.get("Confidence", np.nan)) else f"{float(row['Confidence'])*100:.1f}%"
        # cfu optional
        cfu_out = ""
        if "Log CFU (Input)" in disp.columns:
            v = row.get("Log CFU (Input)", np.nan)
            cfu_out = "" if pd.isna(v) else f"{float(v):.2f}"
        cells = [
            f"<td>{'' if pd.isna(row.get('Days in Refrigerator', np.nan)) else row['Days in Refrigerator']}</td>",
            f"<td>{'' if pd.isna(row.get('Item', np.nan)) else row['Item']}</td>",
            f"<td>{'' if pd.isna(row.get('Product Type', np.nan)) else row['Product Type']}</td>",
            f"<td style='background:{bg};color:{fg};font-weight:600'>{row.get('Prediction','')}</td>",
            f"<td style='background:{bg};color:{fg};font-weight:600'>{conf}</td>",
        ]
        if "Log CFU (Input)" in disp.columns:
            cells.append(f"<td>{cfu_out}</td>")
        return "<tr>" + "".join(cells) + "</tr>"

    headers = ["Days in Refrigerator", "Item", "Product Type", "Prediction", "Confidence"]
    if "Log CFU (Input)" in disp.columns:
        headers.append("Log CFU (Input)")

    rows_html = "\n".join(_row_html(r) for _, r in disp.iterrows())

    # IMPORTANT: use st.markdown(..., unsafe_allow_html=True) — not st.write/st.code
    table_html = (
        """
<style>
  .pred-table-wrap {
    max-height: 430px;       /* adjust to taste */
    overflow-y: auto;        /* <-- visible scrollbar */
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
</style>
<div class="pred-table-wrap">
  <table class="pred-table">
    <thead><tr>"""
        + "".join(f"<th>{h}</th>" for h in headers)
        + """</tr></thead>
    <tbody>
"""
        + rows_html
        + """
    </tbody>
  </table>
</div>
"""
    )

    st.markdown(table_html, unsafe_allow_html=True)

    # -------- download CSV --------
    st.download_button(
        "⬇️ Download predictions as CSV",
        disp.to_csv(index=False).encode("utf-8"),
        "spoilage_predictions.csv",
        "text/csv"
    )


# ====== PERFORMANCE ======
with tab_perf:
    st.markdown("#### 📈 Performance")
    st.caption("Compares predicted outputs with ground-truth CFU values (≥ 7 = not-safe).")

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

# ====== MICROBIOME ======
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

# ====== SENSORY ======
with tab_sens:
    st.markdown("#### 👃 Sensory")
    st.caption("If the CSV includes sensory columns, they’re plotted directly. Otherwise a generalized early➜late pattern is shown.")

    # Smell guidance (per item) — probability only (no 'safe/not-safe' wording)
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
        st.dataframe(dd, use_container_width=True)

    # Early vs Late (generalized template)
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
