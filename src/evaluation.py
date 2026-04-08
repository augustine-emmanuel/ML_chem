"""
evaluation.py
-------------
Computes and stores performance metrics for regression models.
Includes statistical comparison between models (Wilcoxon signed-rank test).
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. PER-FOLD METRICS
# ─────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Computes regression metrics for a single fold.

    Returns
    -------
    dict with keys: rmse, mae, r2
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {"rmse": rmse, "mae": mae, "r2": r2}


# ─────────────────────────────────────────────
# 2. AGGREGATE ACROSS FOLDS
# ─────────────────────────────────────────────

def aggregate_fold_metrics(fold_metrics: list[dict]) -> dict:
    """
    Aggregates per-fold metrics into mean ± std.

    Parameters
    ----------
    fold_metrics : list of dicts from compute_metrics()

    Returns
    -------
    dict with keys: rmse_mean, rmse_std, mae_mean, mae_std, r2_mean, r2_std
    """
    df = pd.DataFrame(fold_metrics)
    result = {}
    for col in df.columns:
        result[f"{col}_mean"] = df[col].mean()
        result[f"{col}_std"] = df[col].std()
    return result


# ─────────────────────────────────────────────
# 3. STATISTICAL COMPARISON
# ─────────────────────────────────────────────

def wilcoxon_comparison(
    metrics_a: list[dict],
    metrics_b: list[dict],
    metric: str = "rmse"
) -> dict:
    """
    Wilcoxon signed-rank test between two models' per-fold metrics.
    Non-parametric — appropriate for small K (5 folds).

    Parameters
    ----------
    metrics_a, metrics_b : list of per-fold metric dicts
    metric : str — which metric to compare

    Returns
    -------
    dict with: statistic, p_value, significant (bool at alpha=0.05)
    """
    scores_a = [m[metric] for m in metrics_a]
    scores_b = [m[metric] for m in metrics_b]

    # Wilcoxon requires differences to be non-zero
    differences = [a - b for a, b in zip(scores_a, scores_b)]
    if all(d == 0 for d in differences):
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}

    stat, p_val = stats.wilcoxon(scores_a, scores_b)
    return {
        "statistic": stat,
        "p_value": p_val,
        "significant": p_val < 0.05
    }


def pairwise_model_comparison(
    all_fold_metrics: dict,
    metric: str = "rmse"
) -> pd.DataFrame:
    """
    Runs pairwise Wilcoxon tests across all model pairs.

    Parameters
    ----------
    all_fold_metrics : dict
        {model_name: [fold_metric_dict, ...]}
    metric : str

    Returns
    -------
    pd.DataFrame — pairwise p-values matrix
    """
    model_names = list(all_fold_metrics.keys())
    n = len(model_names)
    p_matrix = pd.DataFrame(
        np.ones((n, n)),
        index=model_names,
        columns=model_names
    )

    for i, m_a in enumerate(model_names):
        for j, m_b in enumerate(model_names):
            if i >= j:
                continue
            result = wilcoxon_comparison(
                all_fold_metrics[m_a],
                all_fold_metrics[m_b],
                metric=metric
            )
            p_matrix.loc[m_a, m_b] = result["p_value"]
            p_matrix.loc[m_b, m_a] = result["p_value"]

    return p_matrix


# ─────────────────────────────────────────────
# 4. RESULTS TABLE
# ─────────────────────────────────────────────

def build_results_table(
    all_aggregated: dict,
    dataset_name: str
) -> pd.DataFrame:
    """
    Builds a clean results DataFrame for the paper.

    Parameters
    ----------
    all_aggregated : dict
        {model_name: {feature_type: aggregated_metrics_dict}}
    dataset_name : str

    Returns
    -------
    pd.DataFrame formatted for paper table
    """
    rows = []
    for model_name, feat_results in all_aggregated.items():
        for ftype, metrics in feat_results.items():
            rows.append({
                "Dataset": dataset_name.upper(),
                "Model": model_name,
                "Features": ftype,
                "RMSE": f"{metrics['rmse_mean']:.3f} ± {metrics['rmse_std']:.3f}",
                "MAE": f"{metrics['mae_mean']:.3f} ± {metrics['mae_std']:.3f}",
                "R²": f"{metrics['r2_mean']:.3f} ± {metrics['r2_std']:.3f}",
                # raw values for plotting
                "_rmse_mean": metrics["rmse_mean"],
                "_r2_mean": metrics["r2_mean"],
            })
    return pd.DataFrame(rows)


def save_metrics(metrics_df: pd.DataFrame, path: str = "results/metrics.csv"):
    """Saves metrics table to CSV."""
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(path, index=False)
    logger.info(f"Saved metrics to {path}")
