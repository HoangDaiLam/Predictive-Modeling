"""Candidate model definitions, preprocessing and hyperparameter search spaces.

Each candidate is an sklearn ``Pipeline`` of three stages::

    TripFeatureBuilder -> preprocessing -> regressor

so a single logged MLflow model maps raw trip columns to a prediction.

Preprocessing is chosen per library to respect what each implementation
supports natively:

* ``GradientBoostingRegressor`` (scikit-learn) cannot consume NaN or
  pandas categoricals, so it receives median imputation with missing
  indicators and one-hot encoded categories.
* ``XGBRegressor`` and ``LGBMRegressor`` learn a default direction for
  missing values and split categorical features natively, so they receive
  the data with only a fixed-category dtype cast.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from .data import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES, TripFeatureBuilder


class NativeCategoryCaster(BaseEstimator, TransformerMixin):
    """Cast categorical columns to pandas ``category`` with training-set levels.

    Levels unseen during fitting become missing values at inference time,
    which the tree learners route along their learned default direction.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "NativeCategoryCaster":
        self.categories_ = {
            column: sorted(X[column].dropna().unique().tolist())
            for column in CATEGORICAL_FEATURES
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for column in CATEGORICAL_FEATURES:
            levels = self.categories_[column]
            known = out[column].where(out[column].isin(levels))
            out[column] = pd.Categorical(known, categories=levels)
        return out[list(MODEL_FEATURES)]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(MODEL_FEATURES, dtype=object)


def _onehot_preprocessor() -> ColumnTransformer:
    numeric = SimpleImputer(strategy="median", add_indicator=True)
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, list(NUMERIC_FEATURES)),
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
        ],
        verbose_feature_names_out=False,
    )


@dataclass(frozen=True)
class Candidate:
    """A model family together with its search space."""

    key: str
    display_name: str
    build: Callable[[int, int], Pipeline]
    search_space: dict[str, Any]
    preprocessing: str


def _gradient_boosting(random_state: int, n_jobs: int) -> Pipeline:
    return Pipeline(
        [
            ("features", TripFeatureBuilder()),
            ("prep", _onehot_preprocessor()),
            (
                "model",
                GradientBoostingRegressor(loss="squared_error", random_state=random_state),
            ),
        ]
    )


def _xgboost(random_state: int, n_jobs: int) -> Pipeline:
    return Pipeline(
        [
            ("features", TripFeatureBuilder()),
            ("prep", NativeCategoryCaster()),
            (
                "model",
                XGBRegressor(
                    objective="reg:squarederror",
                    tree_method="hist",
                    enable_categorical=True,
                    importance_type="gain",
                    random_state=random_state,
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )


def _lightgbm(random_state: int, n_jobs: int) -> Pipeline:
    return Pipeline(
        [
            ("features", TripFeatureBuilder()),
            ("prep", NativeCategoryCaster()),
            (
                "model",
                LGBMRegressor(
                    objective="regression",
                    subsample_freq=1,
                    importance_type="gain",
                    random_state=random_state,
                    n_jobs=n_jobs,
                    verbosity=-1,
                ),
            ),
        ]
    )


CANDIDATES: dict[str, Candidate] = {
    "gradient_boosting": Candidate(
        key="gradient_boosting",
        display_name="Gradient Boosting (scikit-learn)",
        build=_gradient_boosting,
        search_space={
            "model__n_estimators": randint(100, 400),
            "model__learning_rate": loguniform(0.03, 0.25),
            "model__max_depth": randint(2, 6),
            "model__min_samples_leaf": randint(5, 60),
            "model__subsample": uniform(0.6, 0.4),
            "model__max_features": uniform(0.5, 0.5),
        },
        preprocessing="median imputation + missing indicators, one-hot categories",
    ),
    "xgboost": Candidate(
        key="xgboost",
        display_name="XGBoost",
        build=_xgboost,
        search_space={
            "model__n_estimators": randint(200, 800),
            "model__learning_rate": loguniform(0.02, 0.2),
            "model__max_depth": randint(3, 9),
            "model__min_child_weight": uniform(1.0, 19.0),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.5, 0.5),
            "model__reg_lambda": loguniform(0.1, 30.0),
            "model__reg_alpha": loguniform(1e-3, 5.0),
            "model__gamma": loguniform(1e-3, 2.0),
        },
        preprocessing="native missing-value routing, native categorical splits",
    ),
    "lightgbm": Candidate(
        key="lightgbm",
        display_name="LightGBM",
        build=_lightgbm,
        search_space={
            "model__n_estimators": randint(200, 800),
            "model__learning_rate": loguniform(0.02, 0.2),
            "model__num_leaves": randint(15, 127),
            "model__min_child_samples": randint(10, 80),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.5, 0.5),
            "model__reg_lambda": loguniform(0.1, 30.0),
            "model__reg_alpha": loguniform(1e-3, 5.0),
            "model__min_split_gain": loguniform(1e-3, 2.0),
        },
        preprocessing="native missing-value routing, native categorical splits",
    ),
}


def build_baseline() -> Pipeline:
    """Mean predictor used as a sanity reference, not as a candidate."""
    return Pipeline(
        [
            ("features", TripFeatureBuilder()),
            ("prep", _onehot_preprocessor()),
            ("model", DummyRegressor(strategy="mean")),
        ]
    )


def get_candidate(key: str) -> Candidate:
    """Look up a candidate, raising a clear error for unknown keys."""
    try:
        return CANDIDATES[key]
    except KeyError as exc:
        raise ValueError(f"Unknown candidate '{key}'. Options: {sorted(CANDIDATES)}") from exc
