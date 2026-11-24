"""Numerical verification of the derived closed-form results.

``docs/mathematics.md`` derives, for squared-error loss (gradient g_i = F(x_i) - y_i
and Hessian h_i = 1):

* scikit-learn gradient boosting: first-stage leaf value  nu * mean(y - F0)
* XGBoost and LightGBM:           first-stage leaf value  nu * sum(y - F0) / (n_j + lambda)
* XGBoost split gain:             G_L^2/(n_L+lambda) + G_R^2/(n_R+lambda) - G^2/(n+lambda)

Each function fits a single-tree model on the real data and measures the
largest absolute discrepancy to the formula, so the maths and the libraries
are held to one another.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

from .data import DataSplit, TripFeatureBuilder
from .models import NativeCategoryCaster, _onehot_preprocessor

LEARNING_RATE = 0.3
LAMBDA = 5.0


def _leaf_formula(y: np.ndarray, leaf_ids: np.ndarray, base: float, lam: float, lr: float) -> np.ndarray:
    """Predictions implied by  F0 + lr * sum_leaf(y - F0) / (n_leaf + lam)."""
    residual = y - base
    frame = pd.DataFrame({"leaf": leaf_ids, "r": residual})
    grouped = frame.groupby("leaf")["r"].agg(["sum", "count"])
    weights = grouped["sum"] / (grouped["count"] + lam)
    return base + lr * frame["leaf"].map(weights).to_numpy()


def check_sklearn_first_stage(split: DataSplit, random_state: int = 0) -> float:
    """Leaf value is the mean residual (lambda = 0, line search for squared error)."""
    features = TripFeatureBuilder().transform(split.X_train)
    X = _onehot_preprocessor().fit_transform(features)
    y = split.y_train.to_numpy()
    model = GradientBoostingRegressor(
        n_estimators=1,
        learning_rate=LEARNING_RATE,
        max_depth=3,
        subsample=1.0,
        random_state=random_state,
    ).fit(X, y)
    leaf_ids = np.asarray(model.apply(X)).reshape(len(X), -1)[:, 0]
    expected = _leaf_formula(y, leaf_ids, float(y.mean()), 0.0, LEARNING_RATE)
    return float(np.max(np.abs(model.predict(X) - expected)))


def check_xgboost_first_stage(split: DataSplit, random_state: int = 0) -> float:
    """Leaf weight  w* = -G / (H + lambda)  with G = sum(F0 - y), H = n."""
    features = TripFeatureBuilder().transform(split.X_train)
    X = NativeCategoryCaster().fit_transform(features)
    y = split.y_train.to_numpy()
    base = float(y.mean())
    model = XGBRegressor(
        n_estimators=1,
        learning_rate=LEARNING_RATE,
        max_depth=3,
        reg_lambda=LAMBDA,
        reg_alpha=0.0,
        gamma=0.0,
        base_score=base,
        tree_method="hist",
        enable_categorical=True,
        random_state=random_state,
        n_jobs=1,
    ).fit(X, y)
    leaf_ids = np.asarray(model.apply(X)).reshape(len(X), -1)[:, 0]
    expected = _leaf_formula(y, leaf_ids, base, LAMBDA, LEARNING_RATE)
    return float(np.max(np.abs(model.predict(X) - expected)))


def check_lightgbm_first_stage(split: DataSplit, random_state: int = 0) -> float:
    """Same closed form as XGBoost; LightGBM initialises F0 at the label mean."""
    features = TripFeatureBuilder().transform(split.X_train)
    X = NativeCategoryCaster().fit_transform(features)
    y = split.y_train.to_numpy()
    model = LGBMRegressor(
        n_estimators=1,
        learning_rate=LEARNING_RATE,
        num_leaves=8,
        reg_lambda=LAMBDA,
        min_child_samples=20,
        subsample_freq=0,
        random_state=random_state,
        n_jobs=1,
        verbosity=-1,
    ).fit(X, y)
    leaf_ids = model.predict(X, pred_leaf=True)[:, 0]
    expected = _leaf_formula(y, leaf_ids, float(y.mean()), LAMBDA, LEARNING_RATE)
    return float(np.max(np.abs(model.predict(X) - expected)))


def best_split_gain(X: np.ndarray, y: np.ndarray, base: float, lam: float) -> float:
    """Exhaustive search of  G_L^2/(n_L+lam) + G_R^2/(n_R+lam) - G^2/(n+lam)."""
    grad = base - y
    total_g, n = grad.sum(), len(y)
    parent = total_g**2 / (n + lam)
    best = -np.inf
    for j in range(X.shape[1]):
        order = np.argsort(X[:, j], kind="stable")
        xs, gs = X[order, j], grad[order]
        cum_g = np.cumsum(gs)[:-1]
        n_left = np.arange(1, n)
        valid = xs[1:] > xs[:-1]
        if not valid.any():
            continue
        gain = (
            cum_g**2 / (n_left + lam)
            + (total_g - cum_g) ** 2 / (n - n_left + lam)
            - parent
        )
        best = max(best, float(gain[valid].max()))
    return best


def check_xgboost_split_gain(split: DataSplit, random_state: int = 0) -> tuple[float, float]:
    """Compare the brute-force gain with the root gain XGBoost reports.

    Returns:
        ``(formula_gain, reported_gain)`` for a depth-one exact-greedy tree.
    """
    features = TripFeatureBuilder().transform(split.X_train)
    X = _onehot_preprocessor().fit_transform(features)
    y = split.y_train.to_numpy()
    base = float(y.mean())
    model = XGBRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        reg_lambda=LAMBDA,
        gamma=0.0,
        min_child_weight=0.0,
        base_score=base,
        tree_method="exact",
        random_state=random_state,
        n_jobs=1,
    ).fit(X, y)
    table = model.get_booster().trees_to_dataframe()
    reported = float(table.loc[table["Node"] == 0, "Gain"].iloc[0])
    return best_split_gain(X, y, base, LAMBDA), reported


def check_lightgbm_split_gain(split: DataSplit, random_state: int = 0) -> tuple[float, float]:
    """Evaluate the gain formula at LightGBM's chosen root split.

    The histogram algorithm restricts thresholds to bin edges, so this checks
    that the reported gain equals the closed form at the split LightGBM chose,
    rather than re-deriving the search.

    Returns:
        ``(formula_gain, reported_gain)``.
    """
    features = TripFeatureBuilder().transform(split.X_train)
    X = _onehot_preprocessor().fit_transform(features)
    y = split.y_train.to_numpy()
    base = float(y.mean())
    model = LGBMRegressor(
        n_estimators=1,
        num_leaves=2,
        learning_rate=1.0,
        reg_lambda=LAMBDA,
        min_child_samples=20,
        subsample_freq=0,
        random_state=random_state,
        n_jobs=1,
        verbosity=-1,
    ).fit(X, y)
    table = model.booster_.trees_to_dataframe()
    root = table[table["node_depth"] == 1].iloc[0]
    column = int(str(root["split_feature"]).split("_")[1])
    left = X[:, column] <= float(root["threshold"])

    grad = base - y
    g_left, g_right, g_all = grad[left].sum(), grad[~left].sum(), grad.sum()
    n_left, n_right, n_all = int(left.sum()), int((~left).sum()), len(y)
    formula = (
        g_left**2 / (n_left + LAMBDA)
        + g_right**2 / (n_right + LAMBDA)
        - g_all**2 / (n_all + LAMBDA)
    )
    return float(formula), float(root["split_gain"])


def run_derivation_checks(split: DataSplit, random_state: int = 0) -> pd.DataFrame:
    """Run every check and return a tidy report."""
    xgb_formula, xgb_reported = check_xgboost_split_gain(split, random_state)
    lgb_formula, lgb_reported = check_lightgbm_split_gain(split, random_state)
    rows = [
        {
            "check": "first-stage leaf value",
            "library": "scikit-learn",
            "max_abs_error": check_sklearn_first_stage(split, random_state),
            "tolerance": 1e-8,
        },
        {
            "check": "first-stage leaf value",
            "library": "xgboost",
            "max_abs_error": check_xgboost_first_stage(split, random_state),
            "tolerance": 1e-3,
        },
        {
            "check": "first-stage leaf value",
            "library": "lightgbm",
            "max_abs_error": check_lightgbm_first_stage(split, random_state),
            "tolerance": 1e-6,
        },
        {
            "check": "root split gain",
            "library": "xgboost",
            "max_abs_error": abs(xgb_formula - xgb_reported),
            "tolerance": 1e-6 * abs(xgb_formula),
            "formula_value": xgb_formula,
            "library_value": xgb_reported,
        },
        {
            "check": "root split gain",
            "library": "lightgbm",
            "max_abs_error": abs(lgb_formula - lgb_reported),
            "tolerance": 1e-6 * abs(lgb_formula),
            "formula_value": lgb_formula,
            "library_value": lgb_reported,
        },
    ]
    report = pd.DataFrame(rows)
    report["passed"] = report["max_abs_error"] <= report["tolerance"]
    return report
