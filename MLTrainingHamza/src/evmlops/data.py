"""Data loading, schema definition, feature engineering and splitting.

Leakage policy
--------------
Two columns are deliberately excluded from the feature set:

* ``energy_used_kwh``: the target is a deterministic function of it,
  ``consumption = 100 * energy_used_kwh / distance_km`` (verified to within
  rounding error on this dataset), so using it would reduce the task to
  arithmetic.
* ``battery_level_end_pct``: only known after the trip has finished, and it
  encodes the energy that was consumed.

Identifier columns are never used as features.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GroupShuffleSplit

LOGGER = logging.getLogger(__name__)

TARGET = "consumption_kwh_per_100km"
LEAKAGE_COLUMNS = ("energy_used_kwh", "battery_level_end_pct")
IDENTIFIER_COLUMNS = ("observation_id", "trip_id", "vehicle_id")
TIMESTAMP_COLUMN = "timestamp"

RAW_NUMERIC = (
    "distance_km",
    "speed_kmh",
    "outside_temperature_c",
    "battery_temperature_c",
    "battery_level_start_pct",
    "tyre_pressure_bar",
    "slope_grade_pct",
)
CATEGORICAL = (
    "traffic_situation",
    "preconditioning",
    "climate_settings",
    "road_condition",
    "topography",
    "towing",
    "current_driving_pattern",
    "historical_driving_pattern",
)
ENGINEERED_NUMERIC = (
    "hour_sin",
    "hour_cos",
    "month",
    "day_of_week",
    "is_weekend",
    "temp_delta_c",
    "abs_slope_grade_pct",
)

# Columns a caller must supply at inference time.
RAW_INPUT_COLUMNS = RAW_NUMERIC + CATEGORICAL + (TIMESTAMP_COLUMN,)
NUMERIC_FEATURES = RAW_NUMERIC + ENGINEERED_NUMERIC
CATEGORICAL_FEATURES = CATEGORICAL
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class DataSplit:
    """Container for a train/test partition."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    groups_train: pd.Series
    groups_test: pd.Series


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Read the CSV and validate that the expected schema is present.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing or the target is invalid.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    required = set(RAW_INPUT_COLUMNS) | {TARGET} | set(IDENTIFIER_COLUMNS)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if df[TARGET].isna().any():
        raise ValueError(f"Target column '{TARGET}' contains missing values")
    if (df[TARGET] <= 0).any():
        raise ValueError(f"Target column '{TARGET}' must be strictly positive")

    LOGGER.info("Loaded %d rows and %d columns from %s", *df.shape, path)
    return df


class TripFeatureBuilder(BaseEstimator, TransformerMixin):
    """Stateless row-wise feature engineering.

    Every derived feature depends only on the same row, so applying it
    before the train/test split cannot leak information across the split.
    Living inside the model pipeline means the logged MLflow model accepts
    the raw trip columns directly.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "TripFeatureBuilder":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = sorted(set(RAW_INPUT_COLUMNS) - set(X.columns))
        if missing:
            raise ValueError(f"Input is missing required columns: {missing}")

        out = X[list(RAW_INPUT_COLUMNS)].copy()
        ts = pd.to_datetime(out[TIMESTAMP_COLUMN], errors="coerce")

        hour = ts.dt.hour + ts.dt.minute / 60.0
        out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
        out["month"] = ts.dt.month.astype("float64")
        out["day_of_week"] = ts.dt.dayofweek.astype("float64")
        out["is_weekend"] = (ts.dt.dayofweek >= 5).astype("float64")
        out["temp_delta_c"] = out["battery_temperature_c"] - out["outside_temperature_c"]
        out["abs_slope_grade_pct"] = out["slope_grade_pct"].abs()

        for column in NUMERIC_FEATURES:
            out[column] = pd.to_numeric(out[column], errors="coerce").astype("float64")
        for column in CATEGORICAL_FEATURES:
            out[column] = out[column].astype("object")

        return out[list(MODEL_FEATURES)]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(MODEL_FEATURES, dtype=object)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return raw model inputs, target and the group labels."""
    return df[list(RAW_INPUT_COLUMNS)].copy(), df[TARGET].copy(), df["vehicle_id"].copy()


def group_train_test_split(
    df: pd.DataFrame,
    group_column: str,
    test_size: float,
    random_state: int,
) -> DataSplit:
    """Split so that no group appears in both partitions."""
    if group_column not in df.columns:
        raise ValueError(f"Unknown group column: {group_column}")

    X, y, _ = split_features_target(df)
    groups = df[group_column]

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    split = DataSplit(
        X_train=X.iloc[train_idx].reset_index(drop=True),
        X_test=X.iloc[test_idx].reset_index(drop=True),
        y_train=y.iloc[train_idx].reset_index(drop=True),
        y_test=y.iloc[test_idx].reset_index(drop=True),
        groups_train=groups.iloc[train_idx].reset_index(drop=True),
        groups_test=groups.iloc[test_idx].reset_index(drop=True),
    )
    overlap = set(split.groups_train) & set(split.groups_test)
    if overlap:
        raise RuntimeError(f"Group leakage between partitions: {sorted(overlap)[:5]}")

    LOGGER.info(
        "Split: %d train rows (%d groups), %d test rows (%d groups)",
        len(split.y_train),
        split.groups_train.nunique(),
        len(split.y_test),
        split.groups_test.nunique(),
    )
    return split
