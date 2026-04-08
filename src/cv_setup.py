"""
cv_setup.py
-----------
Creates and saves cross-validation fold indices.
Ensures identical splits are used across all models and feature types
for fair comparison — critical for a research paper.
"""

import numpy as np
import pandas as pd
import yaml
import logging
import pickle
from pathlib import Path
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


def create_cv_folds(n_samples: int, config: dict) -> list[tuple]:
    """
    Creates stratified K-Fold splits.

    Parameters
    ----------
    n_samples : int
    config : dict

    Returns
    -------
    List of (train_indices, val_indices) tuples, one per fold.
    """
    cv_cfg = config["cross_validation"]
    kf = KFold(
        n_splits=cv_cfg["n_splits"],
        shuffle=cv_cfg["shuffle"],
        random_state=config["project"]["seed"]
    )

    folds = []
    indices = np.arange(n_samples)
    for train_idx, val_idx in kf.split(indices):
        folds.append((train_idx, val_idx))

    logger.info(f"Created {len(folds)} CV folds for {n_samples} samples.")
    for i, (tr, vl) in enumerate(folds):
        logger.info(f"  Fold {i+1}: train={len(tr)}, val={len(vl)}")

    return folds


def save_folds(folds: list, dataset_name: str, output_dir: str = "data/processed/"):
    """Saves fold indices to disk for reproducibility."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / f"{dataset_name}_folds.pkl"
    with open(path, "wb") as f:
        pickle.dump(folds, f)
    logger.info(f"Saved CV folds to {path}")


def load_folds(dataset_name: str, output_dir: str = "data/processed/") -> list:
    """Loads previously saved fold indices."""
    path = Path(output_dir) / f"{dataset_name}_folds.pkl"
    with open(path, "rb") as f:
        folds = pickle.load(f)
    logger.info(f"Loaded {len(folds)} folds from {path}")
    return folds


def validate_fold_consistency(folds: list, n_samples: int):
    """
    Sanity checks:
    - No overlap between train and val in any fold
    - All samples appear in validation exactly once
    """
    all_val_indices = []

    for i, (train_idx, val_idx) in enumerate(folds):
        overlap = set(train_idx) & set(val_idx)
        assert len(overlap) == 0, f"Fold {i+1}: train/val overlap detected!"
        all_val_indices.extend(val_idx.tolist())

    all_val_indices = sorted(all_val_indices)
    expected = list(range(n_samples))
    assert all_val_indices == expected, "Not all samples appear in validation exactly once!"

    logger.info("CV fold validation passed — no leakage detected.")


if __name__ == "__main__":
    import yaml
    from data_loader import load_config, load_all_datasets

    config = load_config("config.yaml")
    datasets = load_all_datasets(config)

    for name, df in datasets.items():
        folds = create_cv_folds(len(df), config)
        validate_fold_consistency(folds, len(df))
        save_folds(folds, name)
