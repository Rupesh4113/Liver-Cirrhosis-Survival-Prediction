"""
Liver Cirrhosis Survival Prediction - Streamlit Research Dashboard.
Production-grade clinical ML portfolio application with 7 dedicated analytical pages.
Loads pre-trained artifacts for instantaneous responsiveness without retraining on startup.
"""

import json
import logging
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from config.config import (
    RAW_DATA_PATH,
    ARTIFACTS_DIR,
    MODEL_METADATA_PATH,
    TARGET_CLASSES,
    MEDICAL_DISCLAIMER,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)
from src.prediction import predict_patient, load_inference_artifacts
from src.data_loader import load_raw_data
from src.validation import normalize_column_names, preprocess_target, ValidationError
from src.visualization import (
    plot_target_distribution,
    plot_missingness_overview,
    plot_biomarker_distribution_by_outcome,
    plot_correlation_heatmap,
    plot_model_comparison,
    plot_interactive_confusion_matrix,
    plot_feature_importance_interactive,
)
from src.explainability import explain_single_patient

# Page Configuration
st.set_page_config(
    page_title="Liver Cirrhosis Survival Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a365d;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4a5568;
        margin-bottom: 1.2rem;
    }
    .disclaimer-banner {
        background-color: #fffaf0;
        border-left: 5px solid #dd6b20;
        padding: 0.8rem 1.2rem;
        border-radius: 4px;
        margin-bottom: 1.5rem;
        font-size: 0.92rem;
        color: #7b341e;
        line-height: 1.4;
    }
    .metric-card {
        background: #f7fafc;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2b6cb0;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_cached_data_and_artifacts():
    """Load and cache raw dataset and generated model artifacts."""
    df_raw = None
    dist_df = None
    if RAW_DATA_PATH.exists():
        df_raw = pd.read_csv(RAW_DATA_PATH)
        df_norm = normalize_column_names(df_raw)
        try:
            _, dist_df = preprocess_target(df_norm)
        except Exception:
            pass

    metrics_data = {}
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

    comp_df = pd.DataFrame()
    comp_path = ARTIFACTS_DIR / "model_comparison.csv"
    if comp_path.exists():
        comp_df = pd.read_csv(comp_path)

    imp_df = pd.DataFrame()
    imp_path = ARTIFACTS_DIR / "feature_importance.csv"
    if imp_path.exists():
        imp_df = pd.read_csv(imp_path)

    metadata = {}
    if MODEL_METADATA_PATH.exists():
        with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return df_raw, dist_df, metrics_data, comp_df, imp_df, metadata


def render_disclaimer():
    """Prominently display research safety disclaimer."""
    st.markdown(
        f"""
        <div class="disclaimer-banner">
            <strong>⚠️ Research & Educational Use Only</strong><br>
            {MEDICAL_DISCLAIMER}
        </div>
        """,
        unsafe_allow_html=True,
    )


# Sidebar Navigation
st.sidebar.title("🩺 Navigation")
st.sidebar.markdown("**Liver Cirrhosis Research Portal**")

page = st.sidebar.radio(
    "Select Module",
    [
        "1. Overview",
        "2. Patient Prediction",
        "3. Model Performance",
        "4. Feature Importance",
        "5. Explainability",
        "6. Data Exploration",
        "7. About & Ethics",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("Mayo Clinic Primary Biliary Cirrhosis Study")
st.sidebar.caption("Evaluated on 5 Machine Learning Algorithms")

# Load artifacts
df_raw, dist_df, metrics_data, comp_df, imp_df, metadata = load_cached_data_and_artifacts()

# Check if model trained
model_exists = (ARTIFACTS_DIR / "metrics.json").exists()

# -----------------------------------------------------------------------------
# PAGE 1: OVERVIEW
# -----------------------------------------------------------------------------
if page == "1. Overview":
    st.markdown('<div class="main-title">Liver Cirrhosis Survival Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Multiclass Machine Learning Research Dashboard</div>', unsafe_allow_html=True)
    render_disclaimer()

    st.markdown("### Research Executive Summary")
    st.markdown(
        """
        Primary Biliary Cirrhosis (PBC) is a chronic autoimmune liver disease causing progressive bile duct destruction,
        cholestasis, and eventual cirrhosis. This application benchmarks **5 machine learning algorithms** to classify
        retrospective patient follow-up outcomes into three distinct categories:
        - **Class 0: Death** (Patient succumbed during the study period)
        - **Class 1: Censored** (Patient survived follow-up without liver transplantation)
        - **Class 2: Transplant** (Patient received a liver transplant)
        """
    )

    if not model_exists:
        st.warning("Model artifacts not yet found in `artifacts/`. Please run `python train.py` in your terminal.")
    else:
        best_metrics = metrics_data.get("best_metrics", {})
        ds_summary = metrics_data.get("dataset_summary", {})

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-val">{ds_summary.get('total_samples', 418)}</div>
                    <div class="metric-lbl">Total Patients</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-val">{len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)}</div>
                    <div class="metric-lbl">Clinical Biomarkers</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-val">{metrics_data.get('best_model', 'N/A')}</div>
                    <div class="metric-lbl">Top Model (Macro F1)</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-val">{best_metrics.get('holdout_macro_f1', 0.0):.3f}</div>
                    <div class="metric-lbl">Holdout Macro F1</div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.subheader("Cohort Outcome Distribution")

        c1, c2 = st.columns([1.2, 1])
        with c1:
            if dist_df is not None:
                fig_dist = plot_target_distribution(dist_df)
                st.plotly_chart(fig_dist, use_container_width=True)
        with c2:
            st.markdown("#### Class Imbalance Context")
            st.markdown(
                """
                - **Class Imbalance**: The **Transplant** class represents a minority subset of the cohort.
                - **Evaluation Strategy**: Optimizing strictly for accuracy would bias the classifier toward majority classes (`Censored` and `Death`).
                - **Primary Metric**: **Macro F1** is utilized as the primary criterion for algorithm selection because it weights all three outcome classes equally.
                - **Zero Data Leakage**: All imputations, scaling, and feature engineering are encapsulated strictly inside cross-validation training folds.
                """
            )
            if dist_df is not None:
                st.dataframe(dist_df, hide_index=True, use_container_width=True)

# -----------------------------------------------------------------------------
# PAGE 2: PATIENT PREDICTION
# -----------------------------------------------------------------------------
elif page == "2. Patient Prediction":
    st.markdown('<div class="main-title">Patient Outcome Risk Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Individual Biomarker Simulation & Inference</div>', unsafe_allow_html=True)
    render_disclaimer()

    st.markdown("Simulate patient physiological profiles to evaluate model-assigned multiclass outcome probabilities.")

    with st.form("patient_prediction_form"):
        st.subheader("Demographic & Clinical Signs")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Age (Years)", min_value=15.0, max_value=85.0, value=52.0, step=0.5)
            sex = st.selectbox("Sex", options=["F", "M"], index=0)
            stage = st.selectbox("Histologic Disease Stage", options=[1.0, 2.0, 3.0, 4.0], index=2, format_func=lambda x: f"Stage {int(x)}")
        with c2:
            ascites = st.selectbox("Ascites (Fluid Accumulation)", options=[0.0, 1.0], index=0, format_func=lambda x: "Present (Yes)" if x == 1.0 else "Absent (No)")
            hepatomegaly = st.selectbox("Hepatomegaly (Enlarged Liver)", options=[0.0, 1.0], index=1, format_func=lambda x: "Present (Yes)" if x == 1.0 else "Absent (No)")
            spiders = st.selectbox("Spiders (Spider Angiomas)", options=[0.0, 1.0], index=0, format_func=lambda x: "Present (Yes)" if x == 1.0 else "Absent (No)")
        with c3:
            edema = st.selectbox("Peripheral Edema", options=[0.0, 0.5, 1.0], index=0, format_func=lambda x: "0.0: No Edema" if x == 0.0 else ("0.5: Untreated / Resolved" if x == 0.5 else "1.0: Edema despite diuretics"))

        st.subheader("Serum Laboratory Biomarkers")
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            bili = st.number_input("Serum Bilirubin (mg/dL)", min_value=0.1, max_value=40.0, value=2.4, step=0.1)
            chol = st.number_input("Serum Cholesterol (mg/dL)", min_value=50.0, max_value=2000.0, value=290.0, step=5.0)
            albumin = st.number_input("Serum Albumin (g/dL)", min_value=1.0, max_value=6.0, value=3.4, step=0.1)
        with b2:
            copper = st.number_input("Urine Copper (ug/day)", min_value=5.0, max_value=1000.0, value=75.0, step=5.0)
            alk_phos = st.number_input("Alkaline Phosphatase (U/L)", min_value=100.0, max_value=15000.0, value=1850.0, step=25.0)
            sgot = st.number_input("SGOT / AST (U/mL)", min_value=10.0, max_value=600.0, value=115.0, step=5.0)
        with b3:
            trig = st.number_input("Triglycerides (mg/dL)", min_value=20.0, max_value=1000.0, value=120.0, step=5.0)
            platelets = st.number_input("Platelets (x1000/uL)", min_value=10.0, max_value=800.0, value=220.0, step=5.0)
            protime = st.number_input("Prothrombin Time (sec)", min_value=8.0, max_value=30.0, value=10.6, step=0.1)
        with b4:
            st.info("💡 **Clinical Tip**: Elevated bilirubin, prolonged prothrombin time, and decreased albumin are established markers of end-stage liver disease.")

        submitted = st.form_submit_button("Run Model Inference", use_container_width=True)

    if submitted:
        patient_dict = {
            "Age": age,
            "Sex": sex,
            "Ascites": ascites,
            "Hepatomegaly": hepatomegaly,
            "Spiders": spiders,
            "Edema": edema,
            "Bilirubin": bili,
            "Cholesterol": chol,
            "Albumin": albumin,
            "Copper": copper,
            "Alk_Phos": alk_phos,
            "SGOT": sgot,
            "Triglycerides": trig,
            "Platelets": platelets,
            "Prothrombin": protime,
            "Stage": stage,
        }

        try:
            res = predict_patient(patient_dict)
            st.success("Model Inference Complete!")

            pred_label = res["predicted_label"]
            probs = res["probabilities"]

            res_c1, res_c2 = st.columns([1, 1.5])
            with res_c1:
                st.markdown(
                    f"""
                    <div style="background:#edf2f7; padding:1.5rem; border-radius:8px; border-left:6px solid #3182ce;">
                        <h4 style="margin:0; color:#2b6cb0;">Predicted Class Assignment:</h4>
                        <h2 style="margin:0.5rem 0; color:#1a202c;">{pred_label} Class</h2>
                        <p style="color:#718096; font-size:0.9rem; margin:0;">
                            Model: {res['model_version']}<br>
                            Evaluated at: {res.get('training_date', 'N/A')}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption("Notice: Model assignment indicates statistical alignment with retrospective outcome categories, not patient destiny.")

            with res_c2:
                prob_df = pd.DataFrame([
                    {"Outcome": k, "Probability": v, "Percentage": f"{v*100:.1f}%"}
                    for k, v in probs.items()
                ])
                fig_prob = px.bar(
                    prob_df,
                    x="Probability",
                    y="Outcome",
                    orientation="h",
                    color="Outcome",
                    text="Percentage",
                    color_discrete_map={"Death": "#e74c3c", "Censored": "#2ecc71", "Transplant": "#f39c12"},
                    title="<b>Model-Assigned Class Probabilities</b>",
                )
                fig_prob.update_layout(xaxis_range=[0, 1.0], template="plotly_white", margin=dict(t=40, b=30, l=30, r=30))
                st.plotly_chart(fig_prob, use_container_width=True)

        except ValidationError as e:
            st.error(f"Input Validation Error: {e}")
        except Exception as e:
            st.error(f"Inference Error: {e}")

# -----------------------------------------------------------------------------
# PAGE 3: MODEL PERFORMANCE
# -----------------------------------------------------------------------------
elif page == "3. Model Performance":
    st.markdown('<div class="main-title">Model Benchmarks & Holdout Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Rigorous Comparison Across 5 Clinical Classifiers</div>', unsafe_allow_html=True)
    render_disclaimer()

    if comp_df.empty:
        st.warning("Model comparison data not found. Please run `python train.py` first.")
    else:
        st.subheader("Holdout Test Set Performance Table")
        st.dataframe(
            comp_df.style.highlight_max(subset=["Holdout Macro F1", "Holdout Accuracy", "CV Macro F1"], color="#d4edda"),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")
        st.subheader("Visual Comparative Benchmarks")
        fig_comp = plot_model_comparison(comp_df)
        st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown("---")
        st.subheader("Confusion Matrix & Per-Class Breakdown")
        col_cm, col_rep = st.columns([1, 1.2])

        best_summary = metrics_data.get("all_models", {}).get(metrics_data.get("best_model", ""), {})
        cm_data = best_summary.get("confusion_matrix")

        with col_cm:
            if cm_data:
                fig_cm = plot_interactive_confusion_matrix(cm_data, list(TARGET_CLASSES.values()))
                st.plotly_chart(fig_cm, use_container_width=True)
            else:
                st.info("Confusion matrix data unavailable.")

        with col_rep:
            st.markdown(f"#### Classification Report ({metrics_data.get('best_model', 'Best Model')})")
            class_metrics = best_summary.get("class_metrics", {})
            if class_metrics:
                report_rows = []
                for cls_name, vals in class_metrics.items():
                    report_rows.append({
                        "Class": cls_name,
                        "Precision": f"{vals.get('precision', 0.0):.3f}",
                        "Recall": f"{vals.get('recall', 0.0):.3f}",
                        "F1-Score": f"{vals.get('f1', 0.0):.3f}",
                    })
                st.dataframe(pd.DataFrame(report_rows), hide_index=True, use_container_width=True)

            st.markdown(
                """
                > [!NOTE]
                > **Minority Class Scrutiny**:
                > Notice the recall for the **Transplant** class. In medical datasets, minority classes often suffer low recall under unweighted algorithms.
                > Balanced class weighting and macro-F1 optimization ensure this outcome is recognized rather than overlooked.
                """
            )

# -----------------------------------------------------------------------------
# PAGE 4: FEATURE IMPORTANCE
# -----------------------------------------------------------------------------
elif page == "4. Feature Importance":
    st.markdown('<div class="main-title">Global Feature Importance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Algorithmic Attributions & Permutation Analysis</div>', unsafe_allow_html=True)
    render_disclaimer()

    st.markdown(
        """
        > [!IMPORTANT]
        > **Methodological Context**: Feature importance indicates which clinical variables the mathematical model
        > relied upon most heavily to distinguish outcomes. It does **NOT** demonstrate clinical causality.
        """
    )

    if imp_df.empty:
        st.warning("Feature importance data not found. Please run `python train.py`.")
    else:
        tab1, tab2 = st.tabs(["Holdout Permutation Importance", "Model Intrinsic Gini Importance"])

        with tab1:
            st.markdown("#### Permutation Feature Importance (Evaluated on Holdout Test Set)")
            st.caption("Measures the degradation in Holdout Macro F1 when the values of each feature are randomly permuted.")
            fig_perm = plot_feature_importance_interactive(
                imp_df[["Feature", "Permutation_Importance_Mean"]].rename(columns={"Permutation_Importance_Mean": "Permutation_Importance"}),
                "Permutation Feature Importance (Drop in Holdout Macro F1)"
            )
            st.plotly_chart(fig_perm, use_container_width=True)

        with tab2:
            st.markdown("#### Gini / MDI Feature Importance")
            st.caption("Calculated from impurity reduction across decision trees. (Note: Gini importance can favor continuous high-cardinality features).")
            if "Best_Model_Gini_Importance" in imp_df.columns:
                fig_gini = plot_feature_importance_interactive(
                    imp_df[["Feature", "Best_Model_Gini_Importance"]].rename(columns={"Best_Model_Gini_Importance": "Gini_Importance"}),
                    "Mean Decrease in Impurity (Gini Importance)"
                )
                st.plotly_chart(fig_gini, use_container_width=True)
            else:
                st.info("Gini importance only available for tree-based estimators.")

        st.markdown("---")
        st.subheader("Feature Importance Table")
        st.dataframe(imp_df, hide_index=True, use_container_width=True)

# -----------------------------------------------------------------------------
# PAGE 5: EXPLAINABILITY
# -----------------------------------------------------------------------------
elif page == "5. Explainability":
    st.markdown('<div class="main-title">Patient-Level Model Explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">SHAP Biomarker Contribution Breakdown</div>', unsafe_allow_html=True)
    render_disclaimer()

    st.markdown(
        """
        Explore why the model predicted a particular outcome for a specific patient profile.
        These attributions describe mathematical risk directionality inside the model.
        """
    )

    if df_raw is None or not model_exists:
        st.warning("Dataset or model artifacts not found. Please run `python train.py`.")
    else:
        st.subheader("Select Cohort Patient or Input Custom Profile")
        patient_idx = st.slider("Select Patient Record from Cohort", min_value=0, max_value=min(100, len(df_raw)-1), value=4)

        sample_row = df_raw.iloc[[patient_idx]]
        norm_row = normalize_column_names(sample_row)
        clean_row = norm_row[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()

        st.write("Patient Profile Biomarkers:")
        st.dataframe(clean_row, hide_index=True, use_container_width=True)

        if st.button("Generate SHAP Explainability Decomposition", use_container_width=True):
            with st.spinner("Computing SHAP explanations..."):
                model, _ = load_inference_artifacts()
                bg_sample = normalize_column_names(df_raw.head(30))[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

                explanation = explain_single_patient(model, clean_row, bg_sample)

                c1, c2 = st.columns([1, 1.5])
                with c1:
                    pred_name = TARGET_CLASSES[explanation["predicted_class"]]
                    st.markdown(
                        f"""
                        <div style="background:#f7fafc; padding:1.2rem; border-radius:6px; border:1px solid #cbd5e0;">
                            <h4>Model Predicted Class:</h4>
                            <h3 style="color:#2b6cb0; margin:0.3rem 0;">{pred_name}</h3>
                            <p style="color:#4a5568; font-size:0.9rem;">
                                Assigned Probability: <strong>{explanation['probabilities'][explanation['predicted_class']]*100:.1f}%</strong>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.caption(explanation["disclaimer"])

                with c2:
                    contribs = explanation["feature_contributions"]
                    if contribs:
                        contrib_df = pd.DataFrame(contribs)
                        fig_sh = px.bar(
                            contrib_df,
                            x="SHAP_Value",
                            y="Feature",
                            orientation="h",
                            color="Direction",
                            color_discrete_map={"Increased Risk": "#e53e3e", "Decreased Risk": "#3182ce"},
                            title=f"<b>Top Biomarker Drivers for {pred_name} Class</b>",
                        )
                        fig_sh.update_layout(template="plotly_white", margin=dict(t=40, b=30, l=30, r=30))
                        st.plotly_chart(fig_sh, use_container_width=True)
                    else:
                        st.info("Individual feature contributions computed via fallback importance.")

# -----------------------------------------------------------------------------
# PAGE 6: DATA EXPLORATION
# -----------------------------------------------------------------------------
elif page == "6. Data Exploration":
    st.markdown('<div class="main-title">Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Biomarker Distributions, Correlations, and Missingness</div>', unsafe_allow_html=True)
    render_disclaimer()

    if df_raw is None:
        st.warning("Raw dataset not found.")
    else:
        df_norm = normalize_column_names(df_raw)

        tab1, tab2, tab3 = st.tabs(["Biomarker Distributions", "Correlation Analysis", "Missing Data Inspection"])

        with tab1:
            st.subheader("Biomarker Stratification by Outcome Status")
            col_sel, col_type = st.columns(2)
            with col_sel:
                selected_feat = st.selectbox("Select Physiological Biomarker", options=NUMERIC_FEATURES, index=1)
            with col_type:
                plot_kind = st.radio("Plot Type", options=["box", "violin", "histogram"], horizontal=True)

            fig_bio = plot_biomarker_distribution_by_outcome(df_norm, biomarker=selected_feat, plot_type=plot_kind)
            st.plotly_chart(fig_bio, use_container_width=True)

            st.caption(f"Note: Observed differences in {selected_feat} across groups reflect retrospective observational distributions.")

        with tab2:
            st.subheader("Correlation Heatmap")
            st.markdown("Spearman rank correlations between continuous clinical biomarkers.")
            fig_corr = plot_correlation_heatmap(df_norm, NUMERIC_FEATURES)
            st.plotly_chart(fig_corr, use_container_width=True)

        with tab3:
            st.subheader("Missingness Assessment")
            st.markdown("Quantification of missing laboratory measurements in the historical trial cohort.")
            fig_miss = plot_missingness_overview(df_norm[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
            st.plotly_chart(fig_miss, use_container_width=True)

# -----------------------------------------------------------------------------
# PAGE 7: ABOUT & ETHICS
# -----------------------------------------------------------------------------
elif page == "7. About & Ethics":
    st.markdown('<div class="main-title">Research Governance, Methodology & Ethics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Clinical Machine Learning Limitations & Ethical Guardrails</div>', unsafe_allow_html=True)
    render_disclaimer()

    st.markdown(
        """
        ### 1. Research Project Framing
        This application represents a **portfolio-grade clinical machine learning study** designed to explore retrospective
        multiclass classification methodology on historical observational data. It evaluates how machine learning classifiers
        handle clinical class imbalance, missing laboratory analytes, and non-linear risk surfaces.

        ### 2. Mandatory Medical Disclaimer
        > **This application is for research and educational purposes only. It is not a medical device, does not provide
        medical advice, and must not be used for diagnosis, treatment, prognosis, organ-allocation priority, or clinical decision-making.**

        ### 3. Critical Methodological & Clinical Limitations
        Any prospective deployment or clinical extrapolation is strictly contraindicated due to the following structural constraints:

        1. **Retrospective Cohort & Small Sample Size**:
           - The cohort comprises only 418 patients collected between 1974 and 1984 at the Mayo Clinic. Small clinical cohorts are vulnerable to sampling noise.
        2. **Historical Dataset Shift**:
           - Clinical treatment protocols for Primary Biliary Cirrhosis have evolved dramatically since the 1970s (e.g., standard introduction of Ursodeoxycholic Acid [UDCA] and Obeticholic Acid). Historical survival distributions do not match modern outcomes.
        3. **Class Imbalance & Transplant Competing Risk**:
           - Liver transplantation is treated here as a static classification state. In rigorous biostatistical survival analysis, transplantation represents a **competing risk** or dependent censoring event, which standard multiclass classifiers approximate but do not fully resolve.
        4. **Observational Association vs. Causality**:
           - Feature importance metrics (Gini, permutation, SHAP) reflect statistical patterns in this specific cohort. They do **not** imply that intervening to lower a biomarker will alter patient mortality.
        5. **No External or Prospective Validation**:
           - While 5-fold cross-validation and holdout validation were performed rigorously without data leakage, the pipeline has not been validated on an independent external healthcare institution.
        6. **Probability Calibration**:
           - Raw softmax or tree output probabilities reflect algorithmic confidence within the model's feature space, not calibrated clinical risk probabilities.

        ### 4. Technical Architecture
        ```
        Raw Mayo Clinic PBC Dataset (418 patients)
                    ↓
        Column Normalization & Alias Mapping
                    ↓
        Stratified 80/20 Train/Test Split
                    ↓
        Leakage-Safe sklearn Pipeline:
          - Median & Most-Frequent Imputers
          - Clinical Feature Engineering (ALBI proxy, APRI proxy)
          - OneHotEncoder & Scalers
                    ↓
        5-Model Stratified 5-Fold Cross-Validation
                    ↓
        Hyperparameter Optimization (Scored on Macro F1)
                    ↓
        Holdout Evaluation & SHAP Interpretability
                    ↓
        Streamlit Research Dashboard & Prediction API
        ```

        ### 5. Citation & Provenance
        - *Dickson, E. R., Grambsch, P. M., Fleming, T. R., Fisher, L. D., & Langworthy, A. (1989). Prognosis in primary biliary cirrhosis: model for decision making. Hepatology, 10(1), 1-7.*
        - *Therneau, T. M., & Grambsch, P. M. (2000). Modeling Survival Data: Extending the Cox Model. Springer.*
        """
    )

st.sidebar.markdown("---")
st.sidebar.info("Developed with scikit-learn, LightGBM, XGBoost, SHAP, and Streamlit.")
