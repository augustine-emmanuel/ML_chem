"""
week1_run.py
------------
Runs the complete Week 1 pipeline:
1. Load datasets (ESOL + BACE)
2. Extract descriptors and fingerprints
3. Create and validate CV folds
4. Save everything to disk

Run from project root:
    python week1_run.py
"""

import yaml
import logging
import pandas as pd
import numpy as np
from pathlib import Path

from src.data_loader import load_config, load_all_datasets, summarize_dataset, save_processed
from src.feature_engineering import build_feature_matrix, save_feature_matrix
from src.cv_setup import create_cv_folds, validate_fold_consistency, save_folds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_week1(config_path: str = "config.yaml", local_paths: dict = None):
    """
    Full Week 1 pipeline.

    Parameters
    ----------
    local_paths : dict, optional
        If you've already downloaded the CSVs manually:
        {
            'esol': 'data/raw/delaney-processed.csv',
            'bace': 'data/raw/bace.csv'
        }
    """
    logger.info("=" * 60)
    logger.info("WEEK 1 PIPELINE START")
    logger.info("=" * 60)

    # ── 1. Load config ──────────────────────────────────────────
    config = load_config(config_path)
    seed = config["project"]["seed"]
    np.random.seed(seed)

    # ── 2. Load datasets ─────────────────────────────────────────
    logger.info("\n[STEP 1] Loading datasets...")
    datasets = load_all_datasets(config, local_paths=local_paths or {})

    for name, df in datasets.items():
        summarize_dataset(df, name.upper())
        save_processed(df, name)

    # ── 3. Feature extraction ─────────────────────────────────────
    logger.info("\n[STEP 2] Extracting features...")
    feature_store = {}  # {dataset_name: {feature_type: DataFrame}}

    for name, df in datasets.items():
        mols = df["mol"].tolist()
        feature_store[name] = {}

        for ftype in ["descriptors", "fingerprints"]:
            logger.info(f"  [{name.upper()}] {ftype}...")
            X, removed = build_feature_matrix(mols, config, feature_type=ftype)
            feature_store[name][ftype] = X
            save_feature_matrix(X, name, ftype)

            logger.info(
                f"  [{name.upper()}] {ftype}: "
                f"{X.shape[1]} features retained, "
                f"{len(removed)} removed"
            )

    # ── 4. CV fold creation ───────────────────────────────────────
    logger.info("\n[STEP 3] Creating CV folds...")
    fold_store = {}

    for name, df in datasets.items():
        folds = create_cv_folds(len(df), config)
        validate_fold_consistency(folds, len(df))
        save_folds(folds, name)
        fold_store[name] = folds

    # ── 5. Summary report ─────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("WEEK 1 COMPLETE — SUMMARY")
    logger.info("=" * 60)

    for name in datasets:
        df = datasets[name]
        print(f"\n{'─'*40}")
        print(f"Dataset : {name.upper()}")
        print(f"Samples : {len(df)}")
        for ftype in ["descriptors", "fingerprints"]:
            n_feats = feature_store[name][ftype].shape[1]
            print(f"  {ftype:<15}: {n_feats} features")
        print(f"CV folds: {len(fold_store[name])}")

    print("\nAll outputs saved to data/processed/")
    print("Ready for Week 2: modeling\n")


if __name__ == "__main__":
    # ── If you need to use local files, specify paths here ──────
    local_paths = {
        'esol': 'data/raw/esol.csv',
        'bace': 'data/raw/bace.csv'
    }
    run_week1(local_paths=local_paths)

    # run_week1()
