"""Model diagnostics: staged error curves and feature importance."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


def staged_rmse(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y,
    n_points: int = 40,
) -> pd.DataFrame:
    """Test RMSE as a function of the number of boosting iterations.

    This traces the additive model ``F_m(x) = F_0(x) + sum_{k<=m} nu f_k(x)``
    for ``m`` between 1 and the fitted number of trees.
    """
    transform = pipeline[:-1]
    model = pipeline[-1]
    features = transform.transform(X)
    y_arr = np.asarray(y, dtype=float)
    total = int(model.n_estimators)
    steps = np.unique(np.linspace(1, total, n_points).astype(int))

    if isinstance(model, GradientBoostingRegressor):
        wanted = set(steps.tolist())
        rmse = {
            i: float(np.sqrt(np.mean((pred - y_arr) ** 2)))
            for i, pred in enumerate(model.staged_predict(features), start=1)
            if i in wanted
        }
        values = [rmse[int(s)] for s in steps]
    elif isinstance(model, XGBRegressor):
        values = [
            float(np.sqrt(np.mean((model.predict(features, iteration_range=(0, int(s))) - y_arr) ** 2)))
            for s in steps
        ]
    elif isinstance(model, LGBMRegressor):
        values = [
            float(np.sqrt(np.mean((model.predict(features, num_iteration=int(s)) - y_arr) ** 2)))
            for s in steps
        ]
    else:
        raise TypeError(f"Unsupported regressor for staged evaluation: {type(model).__name__}")

    return pd.DataFrame({"iteration": steps, "rmse": values})


def native_importance(pipeline: Pipeline) -> pd.DataFrame:
    """Library-native importance (impurity or gain), normalised to sum to one."""
    names = pipeline[:-1].get_feature_names_out()
    raw = np.asarray(pipeline[-1].feature_importances_, dtype=float)
    total = raw.sum()
    share = raw / total if total > 0 else raw
    return (
        pd.DataFrame({"feature": names, "importance": share})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def permutation_importance_table(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y,
    n_repeats: int,
    random_state: int,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Model-agnostic importance: increase in RMSE when a raw input is shuffled.

    Computed on the raw input columns of the full pipeline, so every model is
    measured on identical, comparable units regardless of internal encoding.
    """
    result = permutation_importance(
        pipeline,
        X,
        y,
        scoring="neg_root_mean_squared_error",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "rmse_increase_mean": result.importances_mean,
                "rmse_increase_std": result.importances_std,
            }
        )
        .sort_values("rmse_increase_mean", ascending=False)
        .reset_index(drop=True)
    )
