"""
shap_visualization.py
---------------------
Publication-ready SHAP and stability figures for Week 3.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import shap
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

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


def save_figure(fig, name: str, output_dir: str = "figures/"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(Path(output_dir) / f"{name}.{ext}")
    logger.info(f"Saved: {name}")
    plt.close(fig)


# ─────────────────────────────────────────────
# FIG 5: SHAP Summary Plot (beeswarm)
# ─────────────────────────────────────────────

def plot_shap_summary(
    shap_values: np.ndarray,
    X_val: np.ndarray,
    feature_names: list,
    model_name: str,
    dataset_name: str,
    top_n: int = 20,
    output_dir: str = "figures/"
):
    """
    SHAP beeswarm summary plot for top-N features.
    Uses fold 1 SHAP values for illustration.
    """
    # Limit to top_n features by mean |SHAP|
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    top_idx = np.argsort(mean_abs)[-top_n:][::-1]

    shap_subset = shap_values[:, top_idx]
    X_subset = X_val[:, top_idx]
    names_subset = [feature_names[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(9, 7))
    shap.summary_plot(
        shap_subset,
        X_subset,
        feature_names=names_subset,
        show=False,
        plot_size=None,
        color_bar=True
    )
    plt.title(
        f"SHAP Summary — {MODEL_LABELS[model_name]} | {dataset_name.upper()}\n"
        f"(Top {top_n} features, Fold 1)"
    )
    plt.tight_layout()
    save_figure(plt.gcf(), f"{dataset_name}_{model_name}_shap_summary", output_dir)


# ─────────────────────────────────────────────
# FIG 6: Top-K Feature Importance Comparison
# ─────────────────────────────────────────────

def plot_feature_importance_comparison(
    mean_importances: dict,
    dataset_name: str,
    feature_type: str,
    top_k: int = 15,
    output_dir: str = "figures/"
):
    """
    Side-by-side horizontal bar charts of top-k SHAP importances
    for all models on the same feature type.

    Parameters
    ----------
    mean_importances : dict
        {model_name: pd.DataFrame from mean_importance_across_folds()}
    """
    n_models = len(mean_importances)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 7), sharey=False)

    if n_models == 1:
        axes = [axes]

    for ax, (model_name, imp_df) in zip(axes, mean_importances.items()):
        top = imp_df.head(top_k)

        ax.barh(
            range(len(top)),
            top["importance_mean"],
            xerr=top["importance_std"],
            color=MODEL_COLORS[model_name],
            alpha=0.8,
            capsize=3,
            error_kw={"linewidth": 1}
        )
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["feature"], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP|")
        ax.set_title(MODEL_LABELS[model_name])
        sns.despine(ax=ax)

    fig.suptitle(
        f"Top-{top_k} SHAP Feature Importances — "
        f"{dataset_name.upper()} ({feature_type})",
        fontsize=13
    )
    plt.tight_layout()
    save_figure(fig, f"{dataset_name}_{feature_type}_importance_comparison", output_dir)


# ─────────────────────────────────────────────
# FIG 7: Stability Heatmap
# ─────────────────────────────────────────────

def plot_stability_heatmap(
    stability_matrix: np.ndarray,
    model_name: str,
    dataset_name: str,
    feature_type: str,
    metric_name: str = "Spearman",
    output_dir: str = "figures/"
):
    """
    Heatmap of pairwise fold stability scores.
    One heatmap per model × feature type.
    """
    K = stability_matrix.shape[0]
    fold_labels = [f"Fold {i+1}" for i in range(K)]

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        stability_matrix,
        ax=ax,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        vmin=0, vmax=1,
        xticklabels=fold_labels,
        yticklabels=fold_labels,
        linewidths=0.5,
        cbar_kws={"label": metric_name}
    )
    ax.set_title(
        f"{metric_name} Stability — {MODEL_LABELS[model_name]}\n"
        f"{dataset_name.upper()} | {feature_type}"
    )
    plt.tight_layout()
    save_figure(
        fig,
        f"{dataset_name}_{model_name}_{feature_type}_{metric_name.lower()}_heatmap",
        output_dir
    )


# ─────────────────────────────────────────────
# FIG 8: Accuracy vs Stability Trade-off
# ─────────────────────────────────────────────

def plot_accuracy_stability_tradeoff(
    tradeoff_df: pd.DataFrame,
    dataset_name: str,
    output_dir: str = "figures/"
):
    """
    Scatter plot: RMSE (x) vs Jaccard stability (y).
    One point per model × feature type.
    This is the paper's key figure — shows the trade-off finding.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, ftype in zip(axes, ["descriptors", "fingerprints"]):
        subset = tradeoff_df[tradeoff_df["features"] == ftype]

        for _, row in subset.iterrows():
            model = row["model"]
            ax.scatter(
                row["rmse_mean"],
                row["jaccard"],
                color=MODEL_COLORS[model],
                s=120,
                zorder=3,
                label=MODEL_LABELS[model]
            )
            ax.annotate(
                MODEL_LABELS[model],
                xy=(row["rmse_mean"], row["jaccard"]),
                xytext=(5, 3),
                textcoords="offset points",
                fontsize=8,
                color=MODEL_COLORS[model]
            )

        ax.set_xlabel("RMSE (lower = better)")
        ax.set_ylabel("Jaccard Stability (higher = better)")
        ax.set_title(f"{ftype.capitalize()} features")
        ax.grid(alpha=0.25)
        sns.despine(ax=ax)

    # Deduplicated legend
    handles, labels = axes[0].get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    fig.legend(seen.values(), seen.keys(), loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, -0.08))

    fig.suptitle(
        f"Accuracy vs. Explanation Stability Trade-off — {dataset_name.upper()}",
        fontsize=13
    )
    plt.tight_layout()
    save_figure(fig, f"{dataset_name}_accuracy_stability_tradeoff", output_dir)


# ─────────────────────────────────────────────
# FIG 9: Variance per Feature (top unstable)
# ─────────────────────────────────────────────

def plot_per_feature_variance(
    per_feature_variance: np.ndarray,
    feature_names: list,
    model_name: str,
    dataset_name: str,
    feature_type: str,
    top_n: int = 20,
    output_dir: str = "figures/"
):
    """
    Bar chart of per-feature variance across CV folds.
    Highlights which features are most/least stable.
    """
    df = pd.DataFrame({
        "feature": feature_names,
        "variance": per_feature_variance
    }).sort_values("variance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        range(len(df)),
        df["variance"],
        color=MODEL_COLORS[model_name],
        alpha=0.8
    )
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["feature"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Variance of Normalized Importance")
    ax.set_title(
        f"Most Variable Features — {MODEL_LABELS[model_name]}\n"
        f"{dataset_name.upper()} | {feature_type}"
    )
    sns.despine(ax=ax)
    plt.tight_layout()
    save_figure(
        fig,
        f"{dataset_name}_{model_name}_{feature_type}_feature_variance",
        output_dir
    )
