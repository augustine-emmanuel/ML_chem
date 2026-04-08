"""
feature_engineering.py
-----------------------
Extracts molecular descriptors and Morgan fingerprints from RDKit Mol objects.
Handles correlation filtering, variance filtering, and feature name tracking.
"""

import numpy as np
import pandas as pd
import logging
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. MOLECULAR DESCRIPTORS
# ─────────────────────────────────────────────

def get_descriptor_names() -> list[str]:
    """Returns list of all available RDKit descriptor names."""
    return [name for name, _ in Descriptors.descList]


def compute_descriptors(mols: list, descriptor_names: list = None) -> pd.DataFrame:
    """
    Computes RDKit molecular descriptors for a list of Mol objects.

    Parameters
    ----------
    mols : list of RDKit Mol objects
    descriptor_names : list of str, optional
        Subset of descriptors to compute. Defaults to all RDKit descriptors.

    Returns
    -------
    pd.DataFrame of shape (n_molecules, n_descriptors)
    """
    if descriptor_names is None:
        descriptor_names = get_descriptor_names()

    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

    rows = []
    for i, mol in enumerate(mols):
        if mol is None:
            logger.warning(f"Mol at index {i} is None — filling with NaN.")
            rows.append([np.nan] * len(descriptor_names))
        else:
            try:
                vals = calculator.CalcDescriptors(mol)
                rows.append(list(vals))
            except Exception as e:
                logger.warning(f"Descriptor calculation failed for mol {i}: {e}")
                rows.append([np.nan] * len(descriptor_names))

    df = pd.DataFrame(rows, columns=descriptor_names)

    # Report NaN rate
    nan_rate = df.isna().mean()
    high_nan = nan_rate[nan_rate > 0.1]
    if not high_nan.empty:
        logger.warning(f"{len(high_nan)} descriptors have >10% NaN rate. Will be handled in cleaning.")

    return df


# ─────────────────────────────────────────────
# 2. MORGAN FINGERPRINTS
# ─────────────────────────────────────────────

def compute_fingerprints(mols: list, radius: int = 2, n_bits: int = 2048) -> pd.DataFrame:
    """
    Computes Morgan (ECFP) fingerprints for a list of Mol objects.

    Parameters
    ----------
    mols : list of RDKit Mol objects
    radius : int
        ECFP radius (2 = ECFP4, 3 = ECFP6)
    n_bits : int
        Fingerprint vector length

    Returns
    -------
    pd.DataFrame of shape (n_molecules, n_bits)
        Columns named 'fp_0', 'fp_1', ..., 'fp_{n_bits-1}'
    """
    col_names = [f"fp_{i}" for i in range(n_bits)]
    rows = []

    for i, mol in enumerate(mols):
        if mol is None:
            logger.warning(f"Mol at index {i} is None — filling fingerprint with zeros.")
            rows.append(np.zeros(n_bits, dtype=int))
        else:
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            rows.append(np.array(fp))

    return pd.DataFrame(rows, columns=col_names)


# ─────────────────────────────────────────────
# 3. CLEANING & FILTERING
# ─────────────────────────────────────────────

def clean_descriptor_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles NaN and infinite values in descriptor matrix.
    - Replaces inf with NaN
    - Fills NaN with column median
    """
    df = df.replace([np.inf, -np.inf], np.nan)

    n_nan_cols = df.isna().any(axis=0).sum()
    if n_nan_cols > 0:
        logger.info(f"Imputing NaN values in {n_nan_cols} descriptor columns using column median.")
        df = df.fillna(df.median())

    return df


def remove_low_variance(df: pd.DataFrame, threshold: float = 0.01) -> tuple[pd.DataFrame, list]:
    """
    Removes near-zero variance features.

    Returns
    -------
    filtered DataFrame, list of removed feature names
    """
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(df)
    kept_mask = selector.get_support()
    removed = df.columns[~kept_mask].tolist()

    if removed:
        logger.info(f"Removed {len(removed)} low-variance features.")

    return df.loc[:, kept_mask], removed


def remove_correlated_features(df: pd.DataFrame, threshold: float = 0.95) -> tuple[pd.DataFrame, list]:
    """
    Removes highly correlated features (keeps first of each correlated pair).

    Parameters
    ----------
    threshold : float
        Pearson correlation threshold above which one feature is dropped.

    Returns
    -------
    filtered DataFrame, list of removed feature names
    """
    corr_matrix = df.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [
        col for col in upper_triangle.columns
        if any(upper_triangle[col] > threshold)
    ]

    logger.info(f"Removed {len(to_drop)} highly correlated features (threshold={threshold}).")
    return df.drop(columns=to_drop), to_drop


def scale_features(X_train: np.ndarray, X_val: np.ndarray) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Fits StandardScaler on training data, transforms both train and validation.
    Call this inside each CV fold to prevent leakage.

    Returns
    -------
    X_train_scaled, X_val_scaled, fitted_scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    return X_train_scaled, X_val_scaled, scaler


# ─────────────────────────────────────────────
# 4. FULL PIPELINE
# ─────────────────────────────────────────────

def build_feature_matrix(
    mols: list,
    config: dict,
    feature_type: str = "descriptors"
) -> tuple[pd.DataFrame, list]:
    """
    Full feature extraction + cleaning pipeline.

    Parameters
    ----------
    mols : list of RDKit Mol objects
    config : dict
    feature_type : str
        One of 'descriptors' or 'fingerprints'

    Returns
    -------
    X : pd.DataFrame — final feature matrix
    removed_features : list — names of features removed during cleaning
    """
    feat_cfg = config["features"]
    removed = []

    if feature_type == "descriptors":
        logger.info("Computing molecular descriptors...")
        X = compute_descriptors(mols)
        X = clean_descriptor_matrix(X)

        X, removed_var = remove_low_variance(X, threshold=feat_cfg["descriptors"]["variance_threshold"])
        removed += removed_var

        X, removed_corr = remove_correlated_features(X, threshold=feat_cfg["descriptors"]["correlation_threshold"])
        removed += removed_corr

        logger.info(f"Final descriptor matrix: {X.shape}")

    elif feature_type == "fingerprints":
        logger.info("Computing Morgan fingerprints...")
        X = compute_fingerprints(
            mols,
            radius=feat_cfg["fingerprints"]["radius"],
            n_bits=feat_cfg["fingerprints"]["n_bits"]
        )
        # Fingerprints are binary — only apply variance filter
        X, removed_var = remove_low_variance(X, threshold=0.001)
        removed += removed_var

        logger.info(f"Final fingerprint matrix: {X.shape}")

    else:
        raise ValueError(f"feature_type must be 'descriptors' or 'fingerprints'. Got: {feature_type}")

    return X, removed


def save_feature_matrix(X: pd.DataFrame, dataset_name: str, feature_type: str, output_dir: str = "data/processed/"):
    """Saves feature matrix to CSV."""
    from pathlib import Path
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = f"{output_dir}/{dataset_name}_{feature_type}.csv"
    X.to_csv(path, index=False)
    logger.info(f"Saved {feature_type} matrix for {dataset_name} to {path}")


if __name__ == "__main__":
    import yaml
    from data_loader import load_config, load_all_datasets

    config = load_config("config.yaml")
    datasets = load_all_datasets(config)

    for name, df in datasets.items():
        mols = df["mol"].tolist()

        for ftype in ["descriptors", "fingerprints"]:
            X, removed = build_feature_matrix(mols, config, feature_type=ftype)
            save_feature_matrix(X, name, ftype)
            print(f"{name.upper()} | {ftype}: {X.shape[1]} features after cleaning")
