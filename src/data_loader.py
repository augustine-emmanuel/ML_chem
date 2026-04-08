"""
data_loader.py
--------------
Handles loading and initial validation of ESOL and BACE datasets.
Performs SMILES validation and basic cleaning.
"""

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from rdkit import Chem

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def validate_smiles(smiles_list: list) -> tuple[list[bool], list]:
    """
    Validates each SMILES string using RDKit.
    Returns a boolean mask and list of RDKit Mol objects (None for invalid).
    """
    valid_mask = []
    mols = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        valid_mask.append(mol is not None)
        mols.append(mol)
    return valid_mask, mols


def load_dataset(dataset_name: str, config: dict, local_path: str = None) -> pd.DataFrame:
    """
    Loads a dataset by name from config. 
    Falls back to local_path if URL is unavailable.

    Parameters
    ----------
    dataset_name : str
        One of 'esol' or 'bace'
    config : dict
        Loaded config.yaml as dict
    local_path : str, optional
        Path to local CSV file if URL loading fails

    Returns
    -------
    pd.DataFrame with columns: ['smiles', 'target', 'mol']
        - smiles: cleaned SMILES string
        - target: property value (float)
        - mol: RDKit Mol object
    """
    cfg = config["datasets"][dataset_name]

    # --- Load raw data ---
    if local_path:
        logger.info(f"Loading {dataset_name.upper()} from local path: {local_path}")
        df = pd.read_csv(local_path)
    else:
        logger.info(f"Loading {dataset_name.upper()} from URL...")
        try:
            df = pd.read_csv(cfg["url"])
        except Exception as e:
            raise RuntimeError(
                f"Failed to load {dataset_name} from URL. "
                f"Download manually from:\n{cfg['url']}\n"
                f"Then pass local_path= argument.\nError: {e}"
            )

    logger.info(f"Raw dataset shape: {df.shape}")

    # --- Extract relevant columns ---
    smiles_col = cfg["smiles_col"]
    target_col = cfg["target_col"]

    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column '{smiles_col}' not found. Available: {df.columns.tolist()}")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available: {df.columns.tolist()}")

    df = df[[smiles_col, target_col]].copy()
    df.columns = ["smiles", "target"]

    # --- Drop missing values ---
    n_before = len(df)
    df.dropna(subset=["smiles", "target"], inplace=True)
    df["smiles"] = df["smiles"].str.strip()
    df = df[df["smiles"] != ""]
    n_after = len(df)
    if n_before != n_after:
        logger.warning(f"Dropped {n_before - n_after} rows with missing SMILES or target.")

    # --- Validate SMILES ---
    logger.info("Validating SMILES...")
    valid_mask, mols = validate_smiles(df["smiles"].tolist())
    n_invalid = sum(1 for v in valid_mask if not v)

    if n_invalid > 0:
        logger.warning(f"Removing {n_invalid} invalid SMILES strings.")

    df["mol"] = mols
    df["valid"] = valid_mask
    df = df[df["valid"]].drop(columns=["valid"]).reset_index(drop=True)

    # --- Target sanity check ---
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    n_non_numeric = df["target"].isna().sum()
    if n_non_numeric > 0:
        logger.warning(f"Dropping {n_non_numeric} rows with non-numeric target values.")
        df.dropna(subset=["target"], inplace=True)

    df.reset_index(drop=True, inplace=True)

    logger.info(f"Final dataset size: {len(df)} molecules")
    logger.info(f"Target stats:\n{df['target'].describe().round(3)}")

    return df


def load_all_datasets(config: dict, local_paths: dict = None) -> dict:
    """
    Loads all datasets defined in config.

    Parameters
    ----------
    config : dict
    local_paths : dict, optional
        e.g. {'esol': 'data/raw/esol.csv', 'bace': 'data/raw/bace.csv'}

    Returns
    -------
    dict of {dataset_name: pd.DataFrame}
    """
    local_paths = local_paths or {}
    datasets = {}
    for name in config["datasets"]:
        datasets[name] = load_dataset(
            name, config,
            local_path=local_paths.get(name)
        )
    return datasets


def save_processed(df: pd.DataFrame, dataset_name: str, output_dir: str = "data/processed/"):
    """Saves processed DataFrame (without mol objects) to CSV."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_df = df.drop(columns=["mol"], errors="ignore")
    path = Path(output_dir) / f"{dataset_name}_processed.csv"
    save_df.to_csv(path, index=False)
    logger.info(f"Saved processed {dataset_name} to {path}")


# --- Quick dataset summary ---
def summarize_dataset(df: pd.DataFrame, name: str):
    print(f"\n{'='*40}")
    print(f"Dataset: {name}")
    print(f"  Molecules : {len(df)}")
    print(f"  Target    : mean={df['target'].mean():.3f}, "
          f"std={df['target'].std():.3f}, "
          f"min={df['target'].min():.3f}, "
          f"max={df['target'].max():.3f}")
    print(f"{'='*40}\n")


if __name__ == "__main__":
    config = load_config("config.yaml")

    datasets = load_all_datasets(config)

    for name, df in datasets.items():
        summarize_dataset(df, name.upper())
        save_processed(df, name)
