"""
Visualization module providing interactive Plotly figures for Streamlit
and publication-ready static Matplotlib/Seaborn charts for saved artifacts.
"""

from typing import List, Optional, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from config.config import TARGET_CLASSES, ARTIFACTS_DIR


def plot_target_distribution(dist_df: pd.DataFrame) -> go.Figure:
    """Create interactive bar and donut chart of target class distribution."""
    fig = px.bar(
        dist_df,
        x="Class_Name",
        y="Count",
        color="Class_Name",
        text="Count",
        color_discrete_sequence=["#e74c3c", "#2ecc71", "#f39c12"],
        labels={"Class_Name": "Outcome Status", "Count": "Patient Count"},
        title="<b>Target Class Distribution (Mayo Clinic PBC Cohort)</b>",
    )
    fig.update_traces(texttemplate="%{text} (%{customdata[0]:.1f}%)", customdata=dist_df[["Percentage"]])
    fig.update_layout(showlegend=False, template="plotly_white", margin=dict(t=50, b=40, l=40, r=40))
    return fig


def plot_missingness_overview(df: pd.DataFrame) -> go.Figure:
    """Create missingness percentage bar chart across clinical variables."""
    missing_counts = df.isna().sum()
    missing_pct = (missing_counts / len(df)) * 100.0
    miss_df = pd.DataFrame({
        "Feature": missing_counts.index,
        "Missing_Count": missing_counts.values,
        "Missing_Percentage": missing_pct.values,
    }).sort_values(by="Missing_Percentage", ascending=False)
    
    # Filter features that have missing values or show top 15
    miss_df = miss_df[miss_df["Missing_Percentage"] > 0]
    if miss_df.empty:
        miss_df = pd.DataFrame([{"Feature": "No Missing Values", "Missing_Count": 0, "Missing_Percentage": 0.0}])

    fig = px.bar(
        miss_df,
        x="Missing_Percentage",
        y="Feature",
        orientation="h",
        color="Missing_Percentage",
        color_continuous_scale="Reds",
        text="Missing_Percentage",
        title="<b>Missing Data Percentage by Clinical Biomarker</b>",
        labels={"Missing_Percentage": "Missing (%)", "Feature": "Biomarker"},
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(template="plotly_white", margin=dict(t=50, b=40, l=40, r=40))
    return fig


def plot_biomarker_distribution_by_outcome(
    df: pd.DataFrame,
    biomarker: str,
    target_col: str = "Status",
    plot_type: str = "box",
) -> go.Figure:
    """
    Generate biomarker distribution stratified by clinical outcome class.
    Framed as observational associations, NOT causality.
    """
    plot_df = df.copy()
    plot_df["Outcome"] = plot_df[target_col].map(TARGET_CLASSES).fillna(plot_df[target_col].astype(str))

    color_map = {"Death": "#e74c3c", "Censored": "#2ecc71", "Transplant": "#f39c12"}

    if plot_type == "box":
        fig = px.box(
            plot_df,
            x="Outcome",
            y=biomarker,
            color="Outcome",
            color_discrete_map=color_map,
            points="outliers",
            title=f"<b>Distribution of {biomarker} Associated with Clinical Outcome Status</b>",
            labels={"Outcome": "Outcome Status", biomarker: f"{biomarker} Level"},
        )
    elif plot_type == "violin":
        fig = px.violin(
            plot_df,
            x="Outcome",
            y=biomarker,
            color="Outcome",
            color_discrete_map=color_map,
            box=True,
            points="all",
            title=f"<b>Violin Density of {biomarker} Associated with Outcome Status</b>",
        )
    else:
        fig = px.histogram(
            plot_df,
            x=biomarker,
            color="Outcome",
            barmode="overlay",
            opacity=0.6,
            color_discrete_map=color_map,
            title=f"<b>Histogram of {biomarker} Stratified by Outcome Status</b>",
        )

    fig.update_layout(template="plotly_white", margin=dict(t=50, b=40, l=40, r=40))
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: List[str]) -> go.Figure:
    """Interactive Spearman/Pearson correlation heatmap for physiological biomarkers."""
    avail_cols = [c for c in numeric_cols if c in df.columns]
    corr_matrix = df[avail_cols].corr(method="spearman").round(2)

    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="<b>Spearman Correlation Matrix of Clinical Biomarkers</b>",
    )
    fig.update_layout(template="plotly_white", margin=dict(t=50, b=40, l=40, r=40))
    return fig


def plot_model_comparison(comp_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing the 5 models across multiple holdout metrics."""
    melted = comp_df.melt(
        id_vars=["Model"],
        value_vars=["CV Macro F1", "Holdout Macro F1", "Holdout Accuracy", "Transplant Recall"],
        var_name="Metric",
        value_name="Score",
    )

    fig = px.bar(
        melted,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        text="Score",
        title="<b>Comparative Algorithm Performance (Ranked by Macro F1)</b>",
        color_discrete_sequence=["#3498db", "#9b59b6", "#2ecc71", "#e67e22"],
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(
        template="plotly_white",
        yaxis_range=[0, 1.05],
        margin=dict(t=50, b=40, l=40, r=40),
    )
    return fig


def plot_interactive_confusion_matrix(cm: List[List[int]], labels: List[str]) -> go.Figure:
    """Plot interactive confusion matrix heatmap."""
    fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted Outcome", y="Actual Ground Truth", color="Patient Count"),
        title="<b>Holdout Confusion Matrix (Counts)</b>",
    )
    fig.update_layout(template="plotly_white", margin=dict(t=50, b=40, l=40, r=40))
    return fig


def plot_feature_importance_interactive(df_imp: pd.DataFrame, title: str) -> go.Figure:
    """Horizontal bar chart for feature importances."""
    val_col = [c for c in df_imp.columns if "Importance" in c][0]
    sorted_df = df_imp.sort_values(by=val_col, ascending=True).tail(12)

    fig = px.bar(
        sorted_df,
        x=val_col,
        y="Feature",
        orientation="h",
        color=val_col,
        color_continuous_scale="Viridis",
        title=f"<b>{title}</b>",
        labels={val_col: "Importance Score", "Feature": "Biomarker"},
    )
    fig.update_layout(template="plotly_white", margin=dict(t=50, b=40, l=40, r=40))
    return fig


def save_static_confusion_matrix(cm: np.ndarray, labels: List[str], save_path=None) -> str:
    """Save high-resolution confusion matrix plot to disk for artifacts."""
    target_path = save_path or (ARTIFACTS_DIR / "confusion_matrix.png")
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
    )
    plt.title("Holdout Test Confusion Matrix\n(Primary Biliary Cirrhosis Cohort)", fontsize=13, pad=12)
    plt.xlabel("Predicted Outcome Class", fontsize=11)
    plt.ylabel("True Observed Class", fontsize=11)
    plt.tight_layout()
    plt.savefig(target_path, dpi=300)
    plt.close()
    return str(target_path)
