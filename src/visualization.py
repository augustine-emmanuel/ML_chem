"""
visualization.py
----------------
Publication-ready figures for Week 2 results.
All figures saved as high-resolution PNG + PDF for paper submission.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# ── Consistent style across all figures ──────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

MODEL_COLORS = {
    "random_forest": "#2E86AB",
    "xgboost":       "#A23B72",
    "svr":           "#F18F01",
    "mlp":           "#C73E1D"
}

MODEL_LABELS = {
    "random_forest": "Random Forest",
    "xgboost":       "XGBoost",
    "svr":           "SVR",
    "mlp":           "MLP"
}

FEATURE_MARKERS = {
    "descriptors":  "o",
    "fingerprints": "s"
}


def save_figure(fig, name: str, output_dir: str = "figures/"):
    """Saves figure as both PNG and PDF."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        path = Path(output_dir) / f"{name}.{ext}"
        fig.savefig(path)
    logger.info(f"Saved figure: {name}")
    plt.close(fig)


# ─────────────────────────────────────────────
# FIG 1: Performance comparison bar chart
# ─────────────────────────────────────────────

def plot_performance_comparison(
    metrics_df: pd.DataFrame,
    dataset_name: str,
    metric: str = "rmse",
    output_dir: str = "figures/"
):
    """
    Grouped bar chart comparing model performance across feature types.
    Error bars = std across CV folds.
    """
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    fig, ax = plt.subplots(figsize=(9, 5))

    model_names = metrics_df["model"].unique()
    feature_types = metrics_df["feature_type"].unique()
    n_models = len(model_names)
    n_ftypes = len(feature_types)

    bar_width = 0.35
    x = np.arange(n_models)

    for i, ftype in enumerate(feature_types):
        subset = metrics_df[metrics_df["feature_type"] == ftype]
        subset = subset.set_index("model").reindex(model_names)

        offset = (i - (n_ftypes - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset,
            subset[mean_col],
            bar_width,
            yerr=subset[std_col],
            label=ftype.capitalize(),
            color=[MODEL_COLORS[m] for m in model_names],
            alpha=0.85 if ftype == "descriptors" else 0.5,
            capsize=4,
            error_kw={"linewidth": 1.2}
        )

    ax.set_xlabel("Model")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Model Performance Comparison — {dataset_name.upper()}\n(5-fold CV, mean ± std)")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in model_names])
    ax.legend(title="Feature Type")
    ax.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax)

    save_figure(fig, f"{dataset_name}_{metric}_comparison", output_dir)


# ─────────────────────────────────────────────
# FIG 2: Predicted vs True scatter plots
# ─────────────────────────────────────────────

def plot_pred_vs_true(
    results: dict,
    dataset_name: str,
    feature_type: str = "descriptors",
    output_dir: str = "figures/"
):
    """
    2x2 grid of predicted vs. true scatter plots, one per model.
    Uses all fold predictions concatenated.
    """
    model_names = list(results.keys())
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.flatten()

    for i, model_name in enumerate(model_names):
        ax = axes[i]

        if feature_type not in results[model_name]:
            ax.set_visible(False)
            continue

        res = results[model_name][feature_type]
        all_true, all_pred = [], []
        for y_true, y_pred in res["fold_predictions"]:
            all_true.extend(y_true)
            all_pred.extend(y_pred)

        all_true = np.array(all_true)
        all_pred = np.array(all_pred)

        # Scatter
        ax.scatter(
            all_true, all_pred,
            alpha=0.4, s=18,
            color=MODEL_COLORS[model_name],
            edgecolors="none"
        )

        # Identity line
        lims = [min(all_true.min(), all_pred.min()),
                max(all_true.max(), all_pred.max())]
        ax.plot(lims, lims, "k--", linewidth=1, alpha=0.6, label="Ideal")

        # Annotate with R²
        agg = res["aggregated"]
        ax.set_title(
            f"{MODEL_LABELS[model_name]}\n"
            f"R²={agg['r2_mean']:.3f} | RMSE={agg['rmse_mean']:.3f}"
        )
        ax.set_xlabel("True Value")
        ax.set_ylabel("Predicted Value")
        sns.despine(ax=ax)

    fig.suptitle(
        f"Predicted vs. True — {dataset_name.upper()} ({feature_type})",
        fontsize=13, y=1.01
    )
    plt.tight_layout()
    save_figure(fig, f"{dataset_name}_{feature_type}_pred_vs_true", output_dir)


# ─────────────────────────────────────────────
# FIG 3: Error distribution (residuals)
# ─────────────────────────────────────────────

def plot_residual_distributions(
    results: dict,
    dataset_name: str,
    feature_type: str = "descriptors",
    output_dir: str = "figures/"
):
    """
    Overlaid KDE plots of residuals (y_pred - y_true) for all models.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name in results:
        if feature_type not in results[model_name]:
            continue

        res = results[model_name][feature_type]
        residuals = []
        for y_true, y_pred in res["fold_predictions"]:
            residuals.extend((y_pred - y_true).tolist())

        sns.kdeplot(
            residuals,
            ax=ax,
            label=MODEL_LABELS[model_name],
            color=MODEL_COLORS[model_name],
            linewidth=2
        )

    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("Residual (Predicted − True)")
    ax.set_ylabel("Density")
    ax.set_title(f"Residual Distributions — {dataset_name.upper()} ({feature_type})")
    ax.legend()
    sns.despine(ax=ax)

    save_figure(fig, f"{dataset_name}_{feature_type}_residuals", output_dir)


# ─────────────────────────────────────────────
# FIG 4: Training time comparison
# ─────────────────────────────────────────────

def plot_training_times(
    results: dict,
    dataset_name: str,
    output_dir: str = "figures/"
):
    """Horizontal bar chart of training times per model/feature type."""
    rows = []
    for model_name, feat_results in results.items():
        for ftype, res in feat_results.items():
            rows.append({
                "model": MODEL_LABELS[model_name],
                "feature_type": ftype,
                "time": res["training_time"]
            })

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, ftype in enumerate(df["feature_type"].unique()):
        subset = df[df["feature_type"] == ftype]
        y_pos = np.arange(len(subset)) + i * 0.35
        ax.barh(
            y_pos,
            subset["time"],
            height=0.3,
            label=ftype.capitalize(),
            alpha=0.8
        )

    ax.set_xlabel("Training Time (seconds, 5-fold CV)")
    ax.set_title(f"Computational Cost — {dataset_name.upper()}")
    ax.legend()
    sns.despine(ax=ax)

    save_figure(fig, f"{dataset_name}_training_times", output_dir)
