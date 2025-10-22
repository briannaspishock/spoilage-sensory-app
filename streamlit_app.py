#!/usr/bin/env python3
# Spoilage + Sensory — RF classifier app (Desktop artifacts, fixed threshold=0.50)
# - Sidebar: read-only artifacts path + fixed threshold display (no slider)
# - Tabs: Predictions, Performance, Sensory Forecast, Microbiome
# - Each tab begins with an explainer about data source & whether results are predictions vs generalized fallbacks.

import json, itertools
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report

# ---------------- THEME ----------------
st.set_page_config(page_title="Spoilage + Sensory", page_icon="🥩", layout="wide")
st.markdown("""
<style>
  .stApp { background: linear-gradient(180deg, #ffeef8 0%, #fff9fc 100%); color:#4d004d; }
  h1,h2,h3 { color:#e75480; }
  .stButton button{ background:#ffb6c1; color:#4d004d; border:none; border-radius:12px;
    padding:.55rem 1.2rem; font-weight:700; box-shadow:0 1px 8px rgba(231,84,128,.25);}
  .stButton button:hover{ background:#ffc8d9; color:#000; }
  .small-mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
                font-size:.8rem; color:#6b214f; }
</style>
""", unsafe_allow_html=True)

st.title("🥩🍗 Microbe-Driven Spoilage + 👃 Sensory Forecast")
st.caption("Upload a CSV to get predictions, performance, sensory guidance, and microbiome views.")

# ---------------- CONSTANTS ----------------
REQUIRED = ["rf_model_tuned.joblib", "model_meta.json"]    # feature_importances CSV optional
LOG_CFU_COL = "Total mesophilic aerobic flora (log10 CFU.g-1)"  # exact header
PALETTE = ["#d5b4ab", "#c3b1e1", "#a3c1da", "#a8c69f", "#f4c2c2"]
THRESHOLD = 0.50  # fixed probability cutoff for "not-safe"

# ---------------- ARTIFACTS (Desktop auto-detect, read-only) ----------------
def find_artifacts_on_desktop():
    """Find model files somewhere on Desktop (depth ≤2)."""
    home = Path.home()
    desktop = home / "Desktop"
    candidates = [desktop]
    if desktop.exists():
        # also scan first and second level
        candidates += [p for p in itertools.chain(desktop.glob("*/"), desktop.glob("*/*/")) if p.is_dir()]
    for base in candidates:
        if base.exists() and all((base / f).exists() for f in REQUIRED):
            return base
    return None

ARTIFACTS_DIR = find_artifacts_on_desktop()

# ---------------- SIDEBAR (model-style) ----------------
with st.sidebar:
    st.header("⚙️ Artifacts & Settings")
    st.caption("Artifacts folder (auto-detected on Desktop):")
    st.code(str(ARTIFACTS_DIR) if ARTIFACTS_DIR else "NOT FOUND", language="bash")
    st.markdown(f"**Classification threshold:** `{THRESHOLD:.2f}`")
    st.caption("Used for classification, performance, sensory grouping, and microbiome summaries.")

if ARTIFACTS_DIR is None:
    st.error("Could not find **rf_model_tuned.joblib** and **model_meta.json** on your Desktop. "
             "Place both files together in one Desktop folder.")
    st.stop()

@st.cache_resource
def load_artifacts(folder: Path):
    clf = joblib.load(folder / "rf_model_tuned.joblib")     # Random Forest classifier (predict_proba)
    meta = json.loads((folder / "model_meta.json").read_text())
    fi_csv = folder / "rf_feature_importances.csv"          # optional
    return clf, meta, fi_csv

try:
    clf, meta, fi_csv = load_artifacts(ARTIFACTS_DIR)
except Exception as e:
    st.error(f"Failed to load artifacts: {e}")
    st.stop()

feature_names = meta.get("feature_names", [])
GT_THRESHOLD = float(meta.get("threshold_cfu", 7.0))

# ---------------- CSV INPUT ----------------
csv_file = st.file_uploader("Upload CSV (same schema as model training)", type=["csv"])
if csv_file is None:
    st.stop()

try:
    df_raw = pd.read_csv(csv_file)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

# ---------------- BUILD X FOR MODEL ----------------
X = df_raw.reindex(columns=feature_names, fill_value=0)
for c in X.columns:
    if not np.issubdtype(X[c].dtype, np.number):
        X[c] = pd.to_numeric(X[c], errors="coerce")
X = X.fillna(0.0)

# ---------------- PREDICTIONS ----------------
try:
    pred_score = clf.predict_proba(X)[:, 1]  # prob(class=1) → "not-safe"
except Exception as e:
    st.error(f"Model predict_proba failed: {e}")
    st.stop()

pred_class = np.where(pred_score >= THRESHOLD, "not-safe", "safe")
safe_conf = (THRESHOLD - pred_score) / THRESHOLD
notsafe_conf = (pred_score - THRESHOLD) / (1.0 - THRESHOLD)
confidence_score = np.clip(np.nan_to_num(np.where(pred_class == "safe", safe_conf, notsafe_conf), nan=0.0), 0.0, 1.0)

# Helpers
ITEM_COL = "Sample_Name_y"
PTYPE_COL = "EnvType"
DAY_COL = "Day_numeric" if "Day_numeric" in df_raw.columns else ("Days_Numeric" if "Days_Numeric" in df_raw.columns else None)

# ---------------- TABS ----------------
tab_pred, tab_perf, tab_sens, tab_micro = st.tabs(
    ["🔮 Predictions", "📊 Performance", "👃 Sensory Forecast", "🧬 Microbiome"]
)

# ===== PREDICTIONS =====
with tab_pred:
    st.info(
        "This tab shows **per-sample predictions** from the uploaded CSV using the model found on your Desktop. "
        f"Labels use a fixed probability cutoff of **{THRESHOLD:.2f}** for “not-safe”. "
        "Confidence is how far the probability is from the cutoff."
    )
    disp = pd.DataFrame({
        "Item": df_raw[ITEM_COL] if ITEM_COL in df_raw.columns else "",
        "Product Type": df_raw[PTYPE_COL] if PTYPE_COL in df_raw.columns else "",
        "Days in Refrigerator": df_raw[DAY_COL] if DAY_COL else "",
        "Prediction": pred_class,
        "Confidence": confidence_score,
        "Prob_not_safe": pred_score
    })
    if LOG_CFU_COL in df_raw.columns:
        disp["Log CFU (Input)"] = df_raw[LOG_CFU_COL]

    st.dataframe(
        disp.style.format({
            "Confidence": "{:.1%}",
            "Prob_not_safe": "{:.3f}",
            "Log CFU (Input)": "{:.2f}" if LOG_CFU_COL in disp.columns else "{:}"
        }),
        use_container_width=True
    )

    st.download_button(
        "⬇️ Download predictions as CSV",
        disp.to_csv(index=False).encode("utf-8"),
        "spoilage_predictions.csv",
        "text/csv"
    )

# ===== PERFORMANCE =====
with tab_perf:
    st.info(
        "This tab compares **model predictions** to **ground-truth CFU** if your CSV contains "
        f"“{LOG_CFU_COL}”. Ground truth rule: values ≥ **{GT_THRESHOLD:.1f}** are treated as “not-safe”. "
        "If your CSV lacks that column, the metrics can’t be computed."
    )

    if LOG_CFU_COL not in df_raw.columns:
        st.warning(f"Add **'{LOG_CFU_COL}'** to compute performance.")
    else:
        y_true = np.where(df_raw[LOG_CFU_COL] >= GT_THRESHOLD, "not-safe", "safe")
        y_pred = pred_class
        labels = ["safe", "not-safe"]

        # Classification report
        try:
            report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(
                report_df.style.format({
                    "precision": "{:.2f}",
                    "recall": "{:.2f}",
                    "f1-score": "{:.2f}",
                    "support": "{:.0f}"
                }),
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Classification report failed: {e}")

        # Confusion matrix
        try:
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            fig_cm = px.imshow(
                cm, text_auto=True,
                labels=dict(x="Predicted", y="True"),
                x=labels, y=labels,
                color_continuous_scale='Reds'
            )
            fig_cm.update_layout(title="Confusion Matrix", coloraxis_showscale=False)
            st.plotly_chart(fig_cm, use_container_width=True, key="cm_plot")
        except Exception as e:
            st.error(f"Confusion matrix failed: {e}")

        # Optional: feature importances
        if (ARTIFACTS_DIR / "rf_feature_importances.csv").exists():
            st.markdown("#### Top Feature Importances")
            try:
                df_imp = pd.read_csv(ARTIFACTS_DIR / "rf_feature_importances.csv").sort_values("Importance", ascending=False).head(25)
                fig_imp = px.bar(
                    df_imp.sort_values("Importance"),
                    x="Importance", y="Feature", orientation="h",
                    color_discrete_sequence=["#e75480"],
                    title="Top Model Importances"
                )
                st.plotly_chart(fig_imp, use_container_width=True, key="imp_plot")
                st.dataframe(df_imp, use_container_width=True, height=360)
            except Exception as e:
                st.info(f"Couldn’t render importances: {e}")

# ===== SENSORY FORECAST =====
with tab_sens:
    sensory_cols = [c for c in ["Etheral","Fermented","Prickly","Rancid","Sulfurous","Old_cheese"] if c in df_raw.columns]
    has_real_sensory = len(sensory_cols) >= 2

    if has_real_sensory:
        st.info(
            "This tab visualizes **your uploaded sensory columns** per sample, and summary profiles by predicted class. "
            "Bars show intensities as provided in your CSV."
        )
    else:
        st.info(
            "Your CSV did **not** include sensory columns — so this tab **synthesizes expected sensory notes** from the "
            "model’s predicted risk. Treat as guidance, not measurement."
        )

    def synthesize_sensory(p):
        # probability → synthetic sensory (0..1): Etheral ↓, others ↑ with risk
        return {
            "Etheral": float(max(0.0, 1.0 - 1.2*p)),
            "Fermented": float(min(1.0, 0.2 + 0.9*p)),
            "Prickly": float(min(1.0, 0.1 + 1.2*p)),
            "Rancid": float(min(1.0, 0.0 + 1.6*p)),
            "Sulfurous": float(min(1.0, 0.0 + 1.3*p)),
            "Old_cheese": float(min(1.0, 0.1 + 1.1*p)),
        }

    # Per-sample bar
    if "Sample_Name_y" in df_raw.columns:
        sample_labels = df_raw["Sample_Name_y"].astype(str)
        chosen = st.selectbox("Choose a sample", options=list(sample_labels), index=0, key="sens_sample")
        row_idx = int(df_raw.index[df_raw["Sample_Name_y"].astype(str) == chosen][0])
        title_suffix = f" — {chosen}"
    else:
        sample_labels = df_raw.index.astype(str)
        chosen = st.selectbox("Choose a row", options=list(sample_labels), index=0, key="sens_sample")
        row_idx = int(chosen)
        title_suffix = f" — Row {row_idx}"

    if has_real_sensory:
        data = df_raw.loc[row_idx, sensory_cols].astype(float).clip(lower=0)
        df_bar = pd.DataFrame({"Descriptor": sensory_cols, "Intensity": data.values})
        title = f"Sensory (provided){title_suffix}"
    else:
        syn = synthesize_sensory(pred_score[row_idx])
        df_bar = pd.DataFrame({"Descriptor": list(syn.keys()), "Intensity": list(syn.values())})
        title = f"Sensory (synthesized from probability={pred_score[row_idx]:.2f}){title_suffix}"

    fig_sens = px.bar(
        df_bar, x="Descriptor", y="Intensity",
        color="Descriptor",
        color_discrete_sequence=(PALETTE * 3)[:len(df_bar)],
        range_y=[0,1],
        title=title
    )
    st.plotly_chart(fig_sens, use_container_width=True, key=f"sens_bar_{chosen}")

    st.markdown("---")
    st.markdown("### 📈 Average sensory by predicted class")

    # Build sensory matrix for all rows
    if has_real_sensory:
        sens_all = df_raw[sensory_cols].astype(float).clip(lower=0)
    else:
        sens_all = pd.DataFrame([synthesize_sensory(p) for p in pred_score])

    sens_all["pred_class"] = pred_class
    avg = sens_all.groupby("pred_class").mean(numeric_only=True).reset_index()
    avg_long = avg.melt(id_vars="pred_class", var_name="Descriptor", value_name="Mean_Intensity")

    fig_avg = px.bar(
        avg_long, x="Descriptor", y="Mean_Intensity",
        color="pred_class", barmode="group",
        category_orders={"pred_class": ["safe","not-safe"]},
        color_discrete_map={"safe":"#a3c1da","not-safe":"#e75480"},
        range_y=[0,1],
        title="Average sensory (safe vs not-safe)"
    )
    st.plotly_chart(fig_avg, use_container_width=True, key="avg_sens")

# ===== MICROBIOME =====
with tab_micro:
    st.info(
        "This tab summarizes **relative abundances** from numeric taxa columns in your CSV. "
        "Per-sample bars use your data; group tables contrast **high-risk (prob ≥ threshold)** vs **low-risk** samples."
    )
    NON_MICROBE = {
        "Sample_Name_y","EnvType","EnvTyp","Day_numeric","Days_Numeric","Day","Days",
        "SRR_ID","SAMN_ID","SamplingTime",
        LOG_CFU_COL,
        "Prediction","Confidence","Prob_not_safe"
    }
    micro_cols = [c for c in df_raw.columns if c not in NON_MICROBE and np.issubdtype(df_raw[c].dtype, np.number)]

    if not micro_cols:
        st.warning("No numeric microbial columns detected — add genus/feature abundance columns to enable this view.")
    else:
        micro_df = df_raw[micro_cols].copy()
        totals = micro_df.sum(axis=1).replace(0, np.nan)
        rel_abund = micro_df.div(totals, axis=0).fillna(0.0)

        # Single-sample Top-N
        left, right = st.columns([2, 1], gap="large")
        with left:
            if "Sample_Name_y" in df_raw.columns:
                sample_labels_m = df_raw["Sample_Name_y"].astype(str)
                selected_label = st.selectbox("Select a sample", options=list(sample_labels_m), index=0, key="micro_select")
                row_idx_m = df_raw.index[df_raw["Sample_Name_y"].astype(str) == selected_label][0]
                title_item = selected_label
            else:
                sample_labels_m = df_raw.index.astype(str)
                selected_label = st.selectbox("Select a row", options=list(sample_labels_m), index=0, key="micro_select")
                row_idx_m = int(selected_label)
                title_item = f"Row {row_idx_m}"

            topN = st.slider("Top microbes (N)", 5, 20, 10, 1, key="topN_micro")
            series = (rel_abund.iloc[row_idx_m].sort_values(ascending=False).head(topN) * 100.0)
            df_top = series.rename("Relative_Abundance_%").reset_index().rename(columns={"index": "Microbe"})

            fig = px.bar(
                df_top, x="Microbe", y="Relative_Abundance_%",
                color="Microbe",
                color_discrete_sequence=(PALETTE * ((len(df_top)//len(PALETTE))+1))[:len(df_top)],
                title=f"Top {topN} microbes — {title_item}"
            )
            fig.update_layout(yaxis_title="Relative abundance (%)", xaxis_title="")
            st.plotly_chart(fig, use_container_width=True, key=f"micro_bar_{title_item}")

        with right:
            st.write("Top microbes (this sample)")
            st.dataframe(df_top, use_container_width=True, width=500)

        st.markdown("---")
        st.markdown("#### Top taxa by predicted risk")
        high_mask = pred_score >= THRESHOLD
        low_mask  = pred_score < THRESHOLD

        top_high = rel_abund[high_mask].mean().sort_values(ascending=False).head(10) * 100.0 if high_mask.sum() > 0 else pd.Series(dtype=float)
        top_low  = rel_abund[low_mask].mean().sort_values(ascending=False).head(10) * 100.0 if low_mask.sum() > 0 else pd.Series(dtype=float)

        cA, cB = st.columns(2)
        with cA:
            st.markdown(f"**Top 10 taxa — High-risk (prob ≥ {THRESHOLD:.2f})**")
            if not top_high.empty:
                st.dataframe(
                    top_high.rename("Mean_RelAbund_%").reset_index().rename(columns={"index":"Taxon"}),
                    use_container_width=True, height=320
                )
            else:
                st.write("—")
        with cB:
            st.markdown(f"**Top 10 taxa — Low-risk (prob < {THRESHOLD:.2f})**")
            if not top_low.empty:
                st.dataframe(
                    top_low.rename("Mean_RelAbund_%").reset_index().rename(columns={"index":"Taxon"}),
                    use_container_width=True, height=320
                )
            else:
                st.write("—")
