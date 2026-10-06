"""
data.py — synthetic data generation for the battery energy consumption project.

Stands in for the team's real dataset until it's ready. Produces vehicle,
route, and weather features plus a continuous target (energy consumption in
kWh/100km) generated from a physically-plausible formula plus noise, so the
evaluator has real, learnable signal to work with rather than pure random
numbers.

Replace `generate_synthetic_data` with a real data loader (or just start
uploading CSVs through the app) once the team's dataset exists — nothing
downstream (train.py, evaluate.py) needs to change, since they only ever
see a DataFrame of features + a target column.
"""

import numpy as np
import pandas as pd

TARGET_COL = "energy_consumption_kwh_per_100km"

# Features grouped for UI presentation (matches the vehicle / route / weather
# structure in the project spec)
FEATURE_GROUPS = {
    "Vehicle": {
        "curb_mass_kg": (900, 2800, 1650),
        "battery_capacity_kwh": (30, 120, 75),
        "drag_coefficient": (0.20, 0.42, 0.28),
        "frontal_area_m2": (1.8, 3.2, 2.4),
        "tire_rolling_resistance": (0.007, 0.014, 0.010),
    },
    "Route": {
        "distance_km": (1, 300, 42),
        "avg_speed_kmh": (20, 140, 80),
        "elevation_gain_m_per_km": (0, 40, 5),
        "stops_per_km": (0, 5, 0.5),
    },
    "Weather": {
        "temperature_c": (-20, 40, 15),
        "wind_speed_kmh": (0, 60, 10),
        "precipitation_mm": (0, 30, 0),
        "humidity_pct": (10, 100, 55),
    },
}

FEATURE_NAMES = [f for group in FEATURE_GROUPS.values() for f in group]


def _sample_uniform(low, high, n, rng):
    return rng.uniform(low, high, n)


def generate_synthetic_data(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic vehicle/route/weather -> energy consumption dataset."""
    rng = np.random.default_rng(seed)
    data = {}
    for group in FEATURE_GROUPS.values():
        for feat, (lo, hi, _mid) in group.items():
            data[feat] = _sample_uniform(lo, hi, n_samples, rng)
    df = pd.DataFrame(data)

    # --- Physically-motivated synthetic target ------------------------
    base = 14.0  # kWh/100km baseline for a mid-size EV

    mass_term = (df["curb_mass_kg"] - 1500) / 1000 * 2.2
    aero_term = df["drag_coefficient"] * df["frontal_area_m2"] * (df["avg_speed_kmh"] / 100) ** 2 * 6.0
    rolling_term = df["tire_rolling_resistance"] * 300
    elevation_term = df["elevation_gain_m_per_km"] * 0.05
    stops_term = df["stops_per_km"] * 0.6

    # Cold weather hurts EV efficiency (heating + battery chemistry);
    # extreme heat has a smaller penalty (AC load)
    cold_term = np.where(df["temperature_c"] < 15, (15 - df["temperature_c"]) * 0.14, 0)
    heat_term = np.where(df["temperature_c"] > 28, (df["temperature_c"] - 28) * 0.05, 0)
    wind_term = df["wind_speed_kmh"] * 0.015
    precip_term = df["precipitation_mm"] * 0.03

    noise = rng.normal(0, 0.6, n_samples)

    raw_target = (
        base + mass_term + aero_term + rolling_term + elevation_term
        + stops_term + cold_term + heat_term + wind_term + precip_term + noise
    )
    df[TARGET_COL] = np.clip(raw_target, 4.0, None).round(2)

    # Round display columns sensibly
    for col in ["curb_mass_kg", "battery_capacity_kwh", "distance_km", "avg_speed_kmh"]:
        df[col] = df[col].round(0)
    for col in ["drag_coefficient", "tire_rolling_resistance"]:
        df[col] = df[col].round(3)

    return df