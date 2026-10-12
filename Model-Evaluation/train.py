"""
train.py — candidate model training, deliberately decoupled from evaluation.

This module knows how to fit models. It knows nothing about metrics, plots,
or dashboards — that's evaluate.py's job. A teammate training models a
completely different way (different notebook, different framework, even a
model trained outside this repo entirely) can skip this file altogether and
just hand evaluate.py a fitted model + X_test + y_test, per the target
architecture:

    Dataset --> [Model A, Model B, Model C, ...] --> Evaluator --> Dashboard

train_candidates() is a convenience for this dashboard's own demo/testing
use, not a required entry point.
"""

import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# name -> (constructor, needs_scaling)
MODEL_REGISTRY = {
    "Linear Regression": (lambda p: LinearRegression(), True),
    "Ridge Regression": (lambda p: Ridge(alpha=p.get("alpha", 1.0), random_state=42), True),
    "Random Forest": (
        lambda p: RandomForestRegressor(
            n_estimators=p.get("n_estimators", 200),
            max_depth=p.get("max_depth", 10),
            min_samples_leaf=p.get("min_samples_leaf", 2),
            random_state=42, n_jobs=-1,
        ),
        False,
    ),
    "Gradient Boosting": (
        lambda p: GradientBoostingRegressor(
            n_estimators=p.get("n_estimators", 200),
            max_depth=p.get("max_depth", 3),
            learning_rate=p.get("learning_rate", 0.1),
            random_state=42,
        ),
        False,
    ),
}


def train_one(model_name, params, X_train, y_train):
    """Train a single named model. Returns (fitted_model, fitted_scaler_or_None, train_seconds)."""
    ctor, needs_scaling = MODEL_REGISTRY[model_name]
    scaler = StandardScaler() if needs_scaling else None
    X_proc = scaler.fit_transform(X_train) if scaler is not None else np.asarray(X_train)

    model = ctor(params)
    t0 = time.time()
    model.fit(X_proc, y_train)
    return model, scaler, time.time() - t0


def train_candidates(model_configs, X_train, y_train):
    """
    Train several candidate models for comparison.

    model_configs: dict of {display_name: (registry_key, params_dict)}
    Returns: dict of {display_name: {"model":, "scaler":, "train_time":}}
    """
    results = {}
    for display_name, (registry_key, params) in model_configs.items():
        model, scaler, train_time = train_one(registry_key, params, X_train, y_train)
        results[display_name] = {"model": model, "scaler": scaler, "train_time": train_time}
    return results