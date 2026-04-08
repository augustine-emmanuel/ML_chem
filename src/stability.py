"""
stability.py
------------
Explanation stability metrics across CV folds.

Three complementary metrics:
  1. Jaccard similarity    — top-k feature overlap (intuitive, good for figures)
  2. Spearman correlation  — full rank correlation  (statistically principled)
  3. Variance-based        — per-feature consistency (normalized, bounded [0,1])

Mathematical formulations match those in the paper methodology section.
"""

import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import spearmanr
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. JACCARD STABILITY (top-k)
# ─────────────────────────────────────────────

def jaccard_pair(importance_a: np.ndarray, importance_b: np.ndarray, top_k: int) -> float:
    """
    Jaccard similarity between top-k feature sets of two folds.

    J(a,b) = |T(a) ∩ T(b)| / |T(a) ∪ T(b)|
    """
    top_a = set(np.argsort(importance_a)[-top_k:])
    top_b = set(np.argsort(importance_b)[-top_k:])
    intersection = len(top_a & top_b)
    union = len(top_a | top_b)
    return intersection / union if union > 0 else 0.0


def jaccard_stability(fold_importances: list, top_k: int = 10) -> dict:
    """
    Computes mean pairwise Jaccard stability across all fold pairs.

    S_Jaccard = (2 / K(K-1)) * sum_{a<b} J(a,b)

    Parameters
    ----------
    fold_importances : list of np.ndarray, one per fold
    top_k            : int — number of top features to consider

    Returns
    -------
    dict with: score (float), pairwise_scores (list), matrix (np.ndarray)
    """
    K = len(fold_importances)
    pairwise = []
    matrix = np.ones((K, K))

    for (i, imp_a), (j, imp_b) in combinations(enumerate(fold_importances), 2):
        score = jaccard_pair(imp_a, imp_b, top_k)
        pairwise.append(score)
        matrix[i, j] = score
        matrix[j, i] = score

    return {
        "score": float(np.mean(pairwise)),
        "pairwise_scores": pairwise,
        "matrix": matrix
    }


# ─────────────────────────────────────────────
# 2. SPEARMAN STABILITY (full ranking)
# ─────────────────────────────────────────────

def spearman_pair(importance_a: np.ndarray, importance_b: np.ndarray) -> float:
    """
    Spearman rank correlation between full importance vectors of two folds.
    """
    corr, _ = spearmanr(importance_a, importance_b)
    return float(corr)


def spearman_stability(fold_importances: list) -> dict:
    """
    Computes mean pairwise Spearman rank correlation across all fold pairs.

    S_Spearman = (2 / K(K-1)) * sum_{a<b} rho(a,b)

    Returns
    -------
    dict with: score (float), pairwise_scores (list), matrix (np.ndarray)
    """
    K = len(fold_importances)
    pairwise = []
    matrix = np.ones((K, K))

    for (i, imp_a), (j, imp_b) in combinations(enumerate(fold_importances), 2):
        score = spearman_pair(imp_a, imp_b)
        pairwise.append(score)
        matrix[i, j] = score
        matrix[j, i] = score

    return {
        "score": float(np.mean(pairwise)),
        "pairwise_scores": pairwise,
        "matrix": matrix
    }


# ─────────────────────────────────────────────
# 3. VARIANCE-BASED STABILITY (per-feature)
# ─────────────────────────────────────────────

def variance_stability(fold_importances_normalized: list) -> dict:
    """
    Computes variance-based stability using normalized importance vectors.

    Normalization: I~_j^(k) = I_j^(k) / sum_j(I_j^(k))
    ensures importances are in [0,1] and bounded variance.

    sigma_j^2 = (1/K) * sum_k (I~_j^(k) - mean_j)^2

    S_Var = 1 - (1/m) * sum_j sigma_j^2

    Parameters
    ----------
    fold_importances_normalized : list of normalized np.ndarray per fold
        (use normalized_importance() from explainability.py)

    Returns
    -------
    dict with:
        score              : float in [0,1]
        per_feature_variance: np.ndarray — variance per feature
        mean_importance    : np.ndarray — mean normalized importance per feature
    """
    # Stack into (K, m) matrix
    matrix = np.stack(fold_importances_normalized, axis=0)  # (K, n_features)

    mean_imp = matrix.mean(axis=0)                         # (n_features,)
    per_feature_var = matrix.var(axis=0)                   # (n_features,) — ddof=0

    # Bounded: since importances are normalized to sum=1, max variance is bounded
    S_var = 1.0 - per_feature_var.mean()

    return {
        "score": float(S_var),
        "per_feature_variance": per_feature_var,
        "mean_importance": mean_imp
    }


# ─────────────────────────────────────────────
# 4. COMBINED STABILITY REPORT
# ─────────────────────────────────────────────

def compute_all_stability_metrics(
    fold_importances: list,
    fold_importances_normalized: list,
    top_k: int = 10
) -> dict:
    """
    Computes all three stability metrics in one call.

    Returns
    -------
    dict with keys: jaccard, spearman, variance
    Each contains the full result dict from its respective function.
    """
    return {
        "jaccard":  jaccard_stability(fold_importances, top_k=top_k),
        "spearman": spearman_stability(fold_importances),
        "variance": variance_stability(fold_importances_normalized)
    }


# ─────────────────────────────────────────────
# 5. CROSS-MODEL STABILITY COMPARISON
# ─────────────────────────────────────────────

def build_stability_table(
    all_stability: dict,
    dataset_name: str
) -> pd.DataFrame:
    """
    Builds a summary table of stability scores for all models/feature types.

    Parameters
    ----------
    all_stability : dict
        {model_name: {feature_type: stability_result_dict}}

    Returns
    -------
    pd.DataFrame — ready for paper Table 2
    """
    rows = []
    for model_name, feat_results in all_stability.items():
        for ftype, metrics in feat_results.items():
            rows.append({
                "dataset":   dataset_name.upper(),
                "model":     model_name,
                "features":  ftype,
                "jaccard":   round(metrics["jaccard"]["score"], 4),
                "spearman":  round(metrics["spearman"]["score"], 4),
                "variance":  round(metrics["variance"]["score"], 4),
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["features", "jaccard"], ascending=[True, False])
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# 6. ACCURACY VS STABILITY ANALYSIS
# ─────────────────────────────────────────────

def accuracy_stability_tradeoff(
    metrics_df: pd.DataFrame,
    stability_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges performance and stability results for trade-off analysis.
    This is the key finding discussed in the paper.

    Returns
    -------
    pd.DataFrame with both performance and stability columns,
    suitable for scatter plot (Fig. X in paper).
    """
    perf = metrics_df[["model", "feature_type", "rmse_mean", "r2_mean"]].copy()
    perf.columns = ["model", "features", "rmse_mean", "r2_mean"]

    stab = stability_df[["model", "features", "jaccard", "spearman"]].copy()

    merged = pd.merge(perf, stab, on=["model", "features"])
    return merged
