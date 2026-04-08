"""
models.py
---------
Model definitions and hyperparameter configurations.
All models follow sklearn API for consistent interface.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import yaml
import logging

logger = logging.getLogger(__name__)


def get_model(model_name: str, config: dict):
    """
    Returns an instantiated model from config parameters.

    Parameters
    ----------
    model_name : str
        One of 'random_forest', 'xgboost', 'svr', 'mlp'
    config : dict
        Loaded config.yaml

    Returns
    -------
    Instantiated sklearn-compatible estimator
    """
    params = config["models"][model_name]
    seed = config["project"]["seed"]

    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params.get("max_depth"),
            min_samples_split=params["min_samples_split"],
            random_state=seed,
            n_jobs=-1
        )

    elif model_name == "xgboost":
        return XGBRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            random_state=seed,
            n_jobs=-1,
            verbosity=0
        )

    elif model_name == "svr":
        return SVR(
            kernel=params["kernel"],
            C=params["C"],
            epsilon=params["epsilon"]
        )

    elif model_name == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=tuple(params["hidden_layer_sizes"]),
            activation=params["activation"],
            max_iter=params["max_iter"],
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.1
        )

    else:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Choose from: random_forest, xgboost, svr, mlp"
        )


def get_all_model_names() -> list[str]:
    return ["random_forest", "xgboost", "svr", "mlp"]


MODEL_DISPLAY_NAMES = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "svr": "SVR",
    "mlp": "MLP"
}


# ─────────────────────────────────────────────
# SHAP EXPLAINER TYPE
# ─────────────────────────────────────────────

def get_shap_explainer_type(model_name: str) -> str:
    """
    Returns which SHAP explainer to use for each model.
    Critical distinction — using KernelSHAP on tree models is wasteful
    and using TreeSHAP on MLP will fail.
    """
    if model_name in ["random_forest", "xgboost"]:
        return "tree"       # TreeSHAP — exact, fast
    elif model_name in ["svr", "mlp"]:
        return "kernel"     # KernelSHAP — approximate, slower
    else:
        raise ValueError(f"Unknown model: {model_name}")
