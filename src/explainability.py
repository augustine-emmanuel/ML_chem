"""
explainability.py
-----------------
Computes SHAP values for all models using the correct explainer per model type.
- TreeSHAP  : RandomForest, XGBoost (exact, fast)
- KernelSHAP: SVR, MLP (approximate, uses background sample)

Stores per-fold SHAP values and global importance vectors.
"""

import numpy as np
import pandas as pd
import shap
import logging
import time
from pathlib import Path

from src.models import get_shap_explainer_type
from src.feature_engineering import scale_features

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. SHAP VALUE COMPUTATION
# ─────────────────────────────────────────────

def compute_shap_values(
    model,
    model_name: str,
    X_train: np.ndarray,
    X_val: np.ndarray,
    config: dict
) -> np.ndarray:
    """
    Computes SHAP values for validation set using correct explainer.

    Parameters
    ----------
    model        : fitted sklearn-compatible model
    model_name   : str — used to select explainer type
    X_train      : np.ndarray — scaled training data (background for KernelSHAP)
    X_val        : np.ndarray — scaled validation data to explain
    config       : dict

    Returns
    -------
    shap_values  : np.ndarray of shape (n_val_samples, n_features)
    """
    explainer_type = get_shap_explainer_type(model_name)

    if explainer_type == "tree":
        # TreeSHAP — exact, no background needed
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val)

        # Some tree models return list (multi-output) — take first
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

    elif explainer_type == "kernel":
        # KernelSHAP — uses background sample to estimate expectations
        n_background = config["explainability"]["shap_background_samples"]

        # Subsample background to keep computation tractable
        if X_train.shape[0] > n_background:
            bg_idx = np.random.choice(X_train.shape[0], n_background, replace=False)
            background = shap.kmeans(X_train[bg_idx], min(50, n_background))
        else:
            background = shap.kmeans(X_train, min(50, X_train.shape[0]))

        explainer = shap.KernelExplainer(model.predict, background)

        # KernelSHAP is slow — log progress
        logger.info(
            f"    KernelSHAP on {X_val.shape[0]} samples "
            f"({X_val.shape[1]} features) — this may take a few minutes..."
        )
        shap_values = explainer.shap_values(X_val, silent=True)

    return np.array(shap_values)


def global_importance(shap_values: np.ndarray) -> np.ndarray:
    """
    Computes global feature importance from SHAP values.
    I_j = mean(|phi_ij|) over all samples.

    Returns
    -------
    np.ndarray of shape (n_features,)
    """
    return np.mean(np.abs(shap_values), axis=0)


def normalized_importance(importance: np.ndarray) -> np.ndarray:
    """
    Normalizes importance vector to sum to 1.
    Required for variance-based stability metric.
    """
    total = importance.sum()
    if total == 0:
        return np.zeros_like(importance)
    return importance / total


# ─────────────────────────────────────────────
# 2. FULL SHAP PIPELINE ACROSS CV FOLDS
# ─────────────────────────────────────────────

def compute_fold_shap(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    folds: list,
    fold_models: list,
    config: dict
) -> dict:
    """
    Computes SHAP values for each CV fold using the already-trained fold models.

    Parameters
    ----------
    model_name   : str
    X            : np.ndarray — full unscaled feature matrix
    y            : np.ndarray — targets (unused, kept for API consistency)
    folds        : list of (train_idx, val_idx) tuples
    fold_models  : list of fitted model objects (from Week 2 training)
    config       : dict

    Returns
    -------
    dict with keys:
        fold_shap_values   : list of np.ndarray per fold
        fold_importances   : list of global importance vectors per fold
        fold_importances_n : list of normalized importance vectors per fold
        computation_time   : float (seconds)
    """
    fold_shap_values = []
    fold_importances = []
    fold_importances_n = []

    start = time.time()

    for k, ((train_idx, val_idx), model) in enumerate(zip(folds, fold_models)):
        logger.info(f"  SHAP Fold {k+1}/{len(folds)}...")

        X_train, X_val = X[train_idx], X[val_idx]

        # Scale consistently with training (fit on train, apply to val)
        X_train_s, X_val_s, _ = scale_features(X_train, X_val)

        shap_vals = compute_shap_values(
            model, model_name, X_train_s, X_val_s, config
        )

        importance = global_importance(shap_vals)
        importance_n = normalized_importance(importance)

        fold_shap_values.append(shap_vals)
        fold_importances.append(importance)
        fold_importances_n.append(importance_n)

    computation_time = time.time() - start
    logger.info(f"  SHAP complete in {computation_time:.1f}s")

    return {
        "fold_shap_values": fold_shap_values,
        "fold_importances": fold_importances,
        "fold_importances_n": fold_importances_n,
        "computation_time": computation_time
    }


# ─────────────────────────────────────────────
# 3. MEAN SHAP IMPORTANCE ACROSS FOLDS
# ─────────────────────────────────────────────

def mean_importance_across_folds(
    fold_importances: list,
    feature_names: list
) -> pd.DataFrame:
    """
    Aggregates per-fold importances into mean ± std.

    Returns
    -------
    pd.DataFrame sorted by mean importance descending
    """
    matrix = np.stack(fold_importances, axis=0)  # (K, n_features)
    df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": matrix.mean(axis=0),
        "importance_std": matrix.std(axis=0)
    })
    return df.sort_values("importance_mean", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────
# 4. PERSISTENCE
# ─────────────────────────────────────────────

def save_shap_results(results: dict, dataset_name: str, model_name: str,
                      feature_type: str, output_dir: str = "results/shap_values/"):
    """Saves SHAP results dict to disk."""
    import pickle
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / f"{dataset_name}_{model_name}_{feature_type}_shap.pkl"
    with open(path, "wb") as f:
        pickle.dump(results, f)
    logger.info(f"Saved SHAP results: {path}")


def load_shap_results(dataset_name: str, model_name: str,
                      feature_type: str, output_dir: str = "results/shap_values/") -> dict:
    """Loads saved SHAP results."""
    import pickle
    path = Path(output_dir) / f"{dataset_name}_{model_name}_{feature_type}_shap.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)
