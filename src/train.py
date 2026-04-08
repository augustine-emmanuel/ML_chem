"""
train.py
--------
Cross-validation training loop.
Trains each model on each feature type across all CV folds.
Stores per-fold metrics, predictions, and raw fold importances
(used in Week 3 for SHAP stability analysis).
"""

import numpy as np
import pandas as pd
import pickle
import logging
import time
from pathlib import Path

from src.models import get_model, get_all_model_names
from src.evaluation import compute_metrics, aggregate_fold_metrics
from src.feature_engineering import scale_features

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# CORE TRAINING LOOP
# ─────────────────────────────────────────────

def train_single_model(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    folds: list,
    config: dict,
    feature_names: list = None
) -> dict:
    """
    Trains a single model across all CV folds.

    Parameters
    ----------
    model_name : str
    X : np.ndarray — full feature matrix
    y : np.ndarray — target values
    folds : list of (train_idx, val_idx) tuples
    config : dict
    feature_names : list of str, optional

    Returns
    -------
    dict with keys:
        fold_metrics    : list of per-fold metric dicts
        fold_predictions: list of (y_true, y_pred) tuples
        fold_models     : list of fitted model objects
        aggregated      : aggregated metrics dict
        training_time   : float (seconds)
    """
    fold_metrics = []
    fold_predictions = []
    fold_models = []

    start_time = time.time()

    for k, (train_idx, val_idx) in enumerate(folds):
        logger.info(f"  Fold {k+1}/{len(folds)}...")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Scale inside fold — prevents leakage
        X_train_scaled, X_val_scaled, _ = scale_features(X_train, X_val)

        # Train
        model = get_model(model_name, config)
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred = model.predict(X_val_scaled)

        # Metrics
        metrics = compute_metrics(y_val, y_pred)
        fold_metrics.append(metrics)
        fold_predictions.append((y_val, y_pred))
        fold_models.append(model)

        logger.info(
            f"    RMSE={metrics['rmse']:.4f} | "
            f"MAE={metrics['mae']:.4f} | "
            f"R²={metrics['r2']:.4f}"
        )

    training_time = time.time() - start_time
    aggregated = aggregate_fold_metrics(fold_metrics)

    logger.info(
        f"  [{model_name}] CV complete in {training_time:.1f}s | "
        f"RMSE={aggregated['rmse_mean']:.4f}±{aggregated['rmse_std']:.4f} | "
        f"R²={aggregated['r2_mean']:.4f}±{aggregated['r2_std']:.4f}"
    )

    return {
        "fold_metrics": fold_metrics,
        "fold_predictions": fold_predictions,
        "fold_models": fold_models,
        "aggregated": aggregated,
        "training_time": training_time,
        "feature_names": feature_names
    }


# ─────────────────────────────────────────────
# FULL EXPERIMENT RUNNER
# ─────────────────────────────────────────────

def run_all_models(
    dataset_name: str,
    feature_matrices: dict,
    y: np.ndarray,
    folds: list,
    config: dict
) -> dict:
    """
    Trains all models on all feature types for one dataset.

    Parameters
    ----------
    dataset_name : str
    feature_matrices : dict
        {'descriptors': pd.DataFrame, 'fingerprints': pd.DataFrame}
    y : np.ndarray
    folds : list
    config : dict

    Returns
    -------
    results : dict
        {model_name: {feature_type: training_result_dict}}
    """
    model_names = get_all_model_names()
    results = {m: {} for m in model_names}

    total = len(model_names) * len(feature_matrices)
    done = 0

    for ftype, X_df in feature_matrices.items():
        X = X_df.values.astype(float)
        feature_names = X_df.columns.tolist()

        for model_name in model_names:
            done += 1
            logger.info(
                f"\n[{done}/{total}] {dataset_name.upper()} | "
                f"{model_name} | {ftype}"
            )

            result = train_single_model(
                model_name, X, y, folds, config, feature_names
            )
            results[model_name][ftype] = result

    return results


# ─────────────────────────────────────────────
# PERSISTENCE
# ─────────────────────────────────────────────

def save_results(results: dict, dataset_name: str, output_dir: str = "results/"):
    """Saves full training results (models + metrics) to disk."""
    path = Path(output_dir) / f"{dataset_name}_training_results.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(results, f)
    logger.info(f"Saved training results to {path}")


def load_results(dataset_name: str, output_dir: str = "results/") -> dict:
    """Loads previously saved training results."""
    path = Path(output_dir) / f"{dataset_name}_training_results.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_metrics_summary(results: dict, dataset_name: str) -> pd.DataFrame:
    """
    Extracts a clean metrics DataFrame from results dict.
    Suitable for saving to CSV and reporting in paper.
    """
    rows = []
    for model_name, feat_results in results.items():
        for ftype, res in feat_results.items():
            agg = res["aggregated"]
            rows.append({
                "dataset": dataset_name,
                "model": model_name,
                "feature_type": ftype,
                "rmse_mean": round(agg["rmse_mean"], 4),
                "rmse_std": round(agg["rmse_std"], 4),
                "mae_mean": round(agg["mae_mean"], 4),
                "mae_std": round(agg["mae_std"], 4),
                "r2_mean": round(agg["r2_mean"], 4),
                "r2_std": round(agg["r2_std"], 4),
                "training_time_s": round(res["training_time"], 2)
            })
    return pd.DataFrame(rows)
