#!/usr/bin/env python3
"""
Clean battery_data.csv (output of generate_battery_data.py) using Polars.

Cleaning rules (no imputation here; fit imputers on training data only):
  - cast numeric columns to Float64; unparseable text (e.g. '11.829 x') -> null
  - drop rows missing the target, distance or energy (needed to define the label)
  - drop duplicate observation_id (keep first)
  - drop rows with impossible values; null values are kept for later imputation
  - drop rows below a minimum plausible consumption (default 5.0 kWh/100 km, see
    --min-consumption). The generator clips consumption at 3.0, so many rows sit on
    that floor (e.g. 40 km for 1.2 kWh), which no real car achieves.

Usage:
    python clean_data.py
    python clean_data.py --input data/battery_data.csv --output data/cleaned_data.csv
"""
from pathlib import Path
import argparse
import polars as pl

TARGET = "consumption_kwh_per_100km"
NUMERIC_COLUMNS = [
    "distance_km", "speed_kmh", "outside_temperature_c", "battery_temperature_c",
    "battery_level_start_pct", "battery_level_end_pct", "tyre_pressure_bar",
    "slope_grade_pct", "energy_used_kwh", TARGET,
]
CATEGORICAL_COLUMNS = [
    "traffic_situation", "preconditioning", "climate_settings", "road_condition",
    "topography", "towing", "current_driving_pattern", "historical_driving_pattern",
]
REQUIRED_COLUMNS = ["observation_id", "vehicle_id", "trip_id", "timestamp",
                    "distance_km", "energy_used_kwh", TARGET]


def _ok_or_null(col, cond):
    """Keep a row if the value is missing (imputed later) or satisfies the condition."""
    return pl.col(col).is_null() | cond


def clean_data(file_path, min_consumption=5.0):
    df = pl.read_csv(file_path, infer_schema_length=None)
    df = df.rename({c: c.strip().lower() for c in df.columns})

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    n0 = df.height

    # Types: invalid numbers become null; categoricals are trimmed and lower-cased
    df = df.with_columns(
        [pl.col(c).cast(pl.Float64, strict=False) for c in NUMERIC_COLUMNS if c in df.columns]
        + [pl.col(c).cast(pl.String).str.strip_chars().str.to_lowercase()
           for c in CATEGORICAL_COLUMNS if c in df.columns]
    )

    # Rows without target, distance or energy cannot be used
    df = df.drop_nulls(subset=["distance_km", "energy_used_kwh", TARGET])
    n_null = n0 - df.height

    # Duplicates (the generator repeats rows with the same observation_id)
    before = df.height
    df = df.unique(subset="observation_id", keep="first", maintain_order=True)
    n_dup = before - df.height

    # Impossible values (missing values are kept)
    before = df.height
    checks = [
        pl.col("distance_km") > 0,
        pl.col("energy_used_kwh") > 0,
        pl.col(TARGET) >= min_consumption,
    ]
    if "speed_kmh" in df.columns:
        checks.append(_ok_or_null("speed_kmh", pl.col("speed_kmh") > 0))
    if "tyre_pressure_bar" in df.columns:
        checks.append(_ok_or_null("tyre_pressure_bar", pl.col("tyre_pressure_bar") > 0))
    for c in ("battery_level_start_pct", "battery_level_end_pct"):
        if c in df.columns:
            checks.append(_ok_or_null(c, pl.col(c).is_between(0, 100)))
    df = df.filter(pl.all_horizontal(checks))
    n_bad = before - df.height

    df = df.sort("observation_id")
    print(f"rows in: {n0:,} | dropped missing target/distance/energy: {n_null:,} | "
          f"duplicates: {n_dup:,} | impossible/implausible values: {n_bad:,} | rows out: {df.height:,}")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("data/battery_data.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/cleaned_data.csv"))
    parser.add_argument("--min-consumption", type=float, default=5.0,
                        help="Drop rows below this kWh/100 km (default: 5.0)")
    args = parser.parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Could not find {args.input}. Run generate_battery_data.py first.")
    df = clean_data(args.input, args.min_consumption)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(args.output)
    print(f"Cleaned data saved: {args.output} ({df.height:,} rows)")


if __name__ == "__main__":
    main()