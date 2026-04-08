"""
week2_run.py
------------
Runs the complete Week 2 pipeline:
1. Load processed features + fold indices from Week 1
2. Train all 4 models on both feature types for both datasets
3. Compute and save all metrics
4. Run pairwise statistical comparisons
5. Generate Week 2 figures

Prerequisite: Run week1_run.py first.

Run from project root:
    python week2_run.py
"""

import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path

from src.data_loader import load_config
from src.train import run_all_models, save_results, extract_metrics_summary
from src.evaluation import pairwise_model_comparison, save_metrics
from src.visualization import (
    plot_performance_comparison,
    plot_pred_vs_true,
    plot_residual_distributions,
    plot_training_times
)
from src.cv_setup import load_folds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_feature_matrix(dataset_name: str, feature_type: str) -> pd.DataFrame:
    """Loads a saved feature matrix from Week 1."""
    import pandas as pd
    path = f"data/processed/{dataset_name}_{feature_type}.csv"
    return pd.read_csv(path)


def load_targets(dataset_name: str) -> np.ndarray:
    """Loads target values from processed dataset."""
    path = f"data/processed/{dataset_name}_processed.csv"
    df = pd.read_csv(path)
    return df["target"].values


def run_week2(config_path: str = "config.yaml"):
    logger.info("=" * 60)
    logger.info("WEEK 2 PIPELINE START")
    logger.info("=" * 60)

    config = load_config(config_path)
    dataset_names = list(config["datasets"].keys())
    feature_types = ["descriptors", "fingerprints"]
    all_metrics_dfs = []

    for dataset_name in dataset_names:
        logger.info(f"\n{'─'*50}")
        logger.info(f"DATASET: {dataset_name.upper()}")
        logger.info(f"{'─'*50}")

        # ── Load data ───────────────────────────────────────────
        y = load_targets(dataset_name)
        folds = load_folds(dataset_name)

        feature_matrices = {}
        for ftype in feature_types:
            feature_matrices[ftype] = load_feature_matrix(dataset_name, ftype)
            logger.info(f"  Loaded {ftype}: {feature_matrices[ftype].shape}")

        # ── Train all models ────────────────────────────────────
        results = run_all_models(
            dataset_name=dataset_name,
            feature_matrices=feature_matrices,
            y=y,
            folds=folds,
            config=config
        )

        # ── Save results ────────────────────────────────────────
        save_results(results, dataset_name)

        # ── Extract and save metrics ────────────────────────────
        metrics_df = extract_metrics_summary(results, dataset_name)
        all_metrics_dfs.append(metrics_df)

        logger.info(f"\n[{dataset_name.upper()}] RESULTS SUMMARY:")
        print(metrics_df[["model", "feature_type", "rmse_mean", "rmse_std", "r2_mean", "r2_std"]].to_string(index=False))

        # ── Statistical comparisons ─────────────────────────────
        logger.info(f"\n[{dataset_name.upper()}] Pairwise Wilcoxon tests (RMSE):")
        for ftype in feature_types:
            fold_metrics_by_model = {
                m: results[m][ftype]["fold_metrics"]
                for m in results
            }
            p_matrix = pairwise_model_comparison(fold_metrics_by_model, metric="rmse")
            p_path = f"results/{dataset_name}_{ftype}_wilcoxon_pvalues.csv"
            Path("results/").mkdir(exist_ok=True)
            p_matrix.to_csv(p_path)
            logger.info(f"  Saved p-value matrix: {p_path}")

        # ── Figures ─────────────────────────────────────────────
        logger.info(f"\n[{dataset_name.upper()}] Generating figures...")

        for ftype in feature_types:
            ftype_metrics = metrics_df[metrics_df["feature_type"] == ftype]
            plot_performance_comparison(ftype_metrics, dataset_name, metric="rmse")
            plot_pred_vs_true(results, dataset_name, feature_type=ftype)
            plot_residual_distributions(results, dataset_name, feature_type=ftype)

        plot_training_times(results, dataset_name)

    # ── Save combined metrics CSV ───────────────────────────────
    combined = pd.concat(all_metrics_dfs, ignore_index=True)
    save_metrics(combined, "results/metrics.csv")

    # ── Final summary ───────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("WEEK 2 COMPLETE — SUMMARY")
    logger.info("=" * 60)
    print(combined[[
        "dataset", "model", "feature_type",
        "rmse_mean", "rmse_std", "r2_mean", "r2_std",
        "training_time_s"
    ]].to_string(index=False))
    print("\nAll results saved to results/")
    print("All figures saved to figures/")
    print("Ready for Week 3: SHAP explainability\n")


if __name__ == "__main__":
    run_week2()
