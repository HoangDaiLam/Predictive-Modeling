"""Leakage-safe feature/target selection helpers for model training."""
import pandas as pd

TARGET_COLUMNS = ["energy_consumption_kwh", "consumption_kwh_per_100km"]
IDENTIFIER_COLUMNS = ["sample_id", "dataset_version", "random_seed"]
# Audit energy terms are derived inside the label-generating physics model.
# Exclude them to avoid giving a model components of its own target calculation.
AUDIT_COLUMNS = [
    "rolling_energy_wheel_kwh", "aerodynamic_energy_wheel_kwh",
    "grade_energy_wheel_kwh", "road_load_energy_battery_kwh",
    "transient_energy_battery_kwh", "unclipped_energy_kwh",
    "hvac_energy_kwh", "auxiliary_energy_kwh",
]
EXCLUDED_COLUMNS = set(IDENTIFIER_COLUMNS + TARGET_COLUMNS + AUDIT_COLUMNS)


def make_xy(df: pd.DataFrame, target: str = "consumption_kwh_per_100km"):
    if target not in TARGET_COLUMNS:
        raise ValueError(f"target must be one of {TARGET_COLUMNS}")
    if target not in df.columns:
        raise ValueError(f"Missing target column: {target}")
    feature_columns = [c for c in df.columns if c not in EXCLUDED_COLUMNS]
    X = df[feature_columns].copy()
    y = df[target].copy()
    return X, y
