"""
week3_run.py
------------
Runs the complete Week 3 pipeline:
1. Load trained fold models from Week 2
2. Compute SHAP values per fold per model per feature type
3. Compute all three stability metrics
4. Generate all SHAP and stability figures
5. Build stability summary table for paper

Prerequisite: Run week1_run.py and week2_run.py first.

Run from project root:
    python week3_run.py
"""

import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path

from src.data_loader import load_config
from src.cv_setup import load_folds
from src.explainability import (
    compute_fold_shap,
    mean_importance_across_folds,
    save_shap_results
)
from src.stability import (
    compute_all_stability_metrics,
    build_stability_table,
    accuracy_stability_tradeoff
)
from src.shap_visualization import (
    plot_shap_summary,
    plot_feature_importance_comparison,
    plot_stability_heatmap,
    plot_accuracy_stability_tradeoff,
    plot_per_feature_variance
)
from src.train import load_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_feature_matrix(dataset_name: str, feature_type: str) -> pd.DataFrame:
    return pd.read_csv(f"data/processed/{dataset_name}_{feature_type}.csv")


def load_targets(dataset_name: str) -> np.ndarray:
    df = pd.read_csv(f"data/processed/{dataset_name}_processed.csv")
    return df["target"].values


def run_week3(config_path: str = "config.yaml"):
    logger.info("=" * 60)
    logger.info("WEEK 3 PIPELINE START")
    logger.info("=" * 60)

    config = load_config(config_path)
    dataset_names = list(config["datasets"].keys())
    feature_types = ["descriptors", "fingerprints"]
    top_k = config["explainability"]["top_k_features"]

    all_stability_dfs = []
    all_metrics_dfs = []

    for dataset_name in dataset_names:
        logger.info(f"\n{'─'*50}")
        logger.info(f"DATASET: {dataset_name.upper()}")
        logger.info(f"{'─'*50}")

        # ── Load data + fold models from Week 2 ─────────────────
        y = load_targets(dataset_name)
        folds = load_folds(dataset_name)
        training_results = load_results(dataset_name)

        all_stability = {}
        mean_importances_store = {}   # for feature importance comparison plots

        for model_name in training_results:
            all_stability[model_name] = {}
            mean_importances_store[model_name] = {}

            for ftype in feature_types:
                logger.info(f"\n  [{model_name}] [{ftype}] Computing SHAP...")

                X_df = load_feature_matrix(dataset_name, ftype)
                X = X_df.values.astype(float)
                feature_names = X_df.columns.tolist()

                # Retrieve trained fold models from Week 2
                fold_models = training_results[model_name][ftype]["fold_models"]

                # ── Compute SHAP values across folds ────────────
                shap_result = compute_fold_shap(
                    model_name=model_name,
                    X=X,
                    y=y,
                    folds=folds,
                    fold_models=fold_models,
                    config=config
                )
                save_shap_results(shap_result, dataset_name, model_name, ftype)

                # ── Stability metrics ────────────────────────────
                stability = compute_all_stability_metrics(
                    fold_importances=shap_result["fold_importances"],
                    fold_importances_normalized=shap_result["fold_importances_n"],
                    top_k=top_k
                )
                all_stability[model_name][ftype] = stability

                logger.info(
                    f"    Jaccard={stability['jaccard']['score']:.4f} | "
                    f"Spearman={stability['spearman']['score']:.4f} | "
                    f"Variance={stability['variance']['score']:.4f}"
                )

                # ── Mean importance for comparison plot ──────────
                mean_imp = mean_importance_across_folds(
                    shap_result["fold_importances"],
                    feature_names
                )
                mean_importances_store[model_name][ftype] = mean_imp

                # ── Per-model stability heatmaps ─────────────────
                plot_stability_heatmap(
                    stability_matrix=stability["spearman"]["matrix"],
                    model_name=model_name,
                    dataset_name=dataset_name,
                    feature_type=ftype,
                    metric_name="Spearman"
                )
                plot_stability_heatmap(
                    stability_matrix=stability["jaccard"]["matrix"],
                    model_name=model_name,
                    dataset_name=dataset_name,
                    feature_type=ftype,
                    metric_name="Jaccard"
                )

                # ── Per-feature variance plot ────────────────────
                plot_per_feature_variance(
                    per_feature_variance=stability["variance"]["per_feature_variance"],
                    feature_names=feature_names,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    feature_type=ftype
                )

                # ── SHAP summary plot (Fold 1) ───────────────────
                plot_shap_summary(
                    shap_values=shap_result["fold_shap_values"][0],
                    X_val=X[folds[0][1]],
                    feature_names=feature_names,
                    model_name=model_name,
                    dataset_name=dataset_name
                )

        # ── Feature importance comparison (all models, per ftype) ─
        for ftype in feature_types:
            ftype_importances = {
                m: mean_importances_store[m][ftype]
                for m in training_results
            }
            plot_feature_importance_comparison(
                mean_importances=ftype_importances,
                dataset_name=dataset_name,
                feature_type=ftype
            )

        # ── Stability summary table ──────────────────────────────
        stability_df = build_stability_table(all_stability, dataset_name)
        all_stability_dfs.append(stability_df)

        stab_path = f"results/stability_results/{dataset_name}_stability.csv"
        Path("results/stability_results/").mkdir(parents=True, exist_ok=True)
        stability_df.to_csv(stab_path, index=False)
        logger.info(f"\nStability table saved: {stab_path}")
        print(stability_df.to_string(index=False))

        # ── Accuracy vs Stability trade-off plot ─────────────────
        metrics_df = pd.read_csv("results/metrics.csv")
        dataset_metrics = metrics_df[metrics_df["dataset"] == dataset_name]
        all_metrics_dfs.append(dataset_metrics)

        tradeoff_df = accuracy_stability_tradeoff(dataset_metrics, stability_df)
        plot_accuracy_stability_tradeoff(tradeoff_df, dataset_name)

    # ── Save combined stability table ────────────────────────────
    combined_stability = pd.concat(all_stability_dfs, ignore_index=True)
    combined_stability.to_csv("results/stability_results/all_stability.csv", index=False)

    # ── Final summary ─────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("WEEK 3 COMPLETE — SUMMARY")
    logger.info("=" * 60)
    print("\nCombined stability results:")
    print(combined_stability.to_string(index=False))
    print("\nAll SHAP values saved to results/shap_values/")
    print("All figures saved to figures/")
    print("Ready for Week 4: writing\n")


if __name__ == "__main__":
    run_week3()
