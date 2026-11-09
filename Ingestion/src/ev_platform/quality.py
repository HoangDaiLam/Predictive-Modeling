"""Data integrity, mathematical consistency and range checks."""
from pathlib import Path
import json
import pandas as pd

REQUIRED = {"sample_id", "distance_km", "duration_minutes", "average_speed_kmh", "energy_consumption_kwh", "consumption_kwh_per_100km", "vehicle_mass_kg", "traffic_index", "ambient_temperature_c"}
BOUNDS = {"distance_km": (0, 300), "duration_minutes": (0, 1440), "average_speed_kmh": (0, 150), "vehicle_mass_kg": (500, 5000), "battery_capacity_kwh": (10, 250), "battery_health_fraction": (0, 1), "traffic_index": (0, 1), "energy_consumption_kwh": (0, 200), "consumption_kwh_per_100km": (0, 150)}


def quality_report(df: pd.DataFrame) -> dict:
    missing = sorted(REQUIRED - set(df.columns))
    if missing:
        return {"passed": False, "missing_columns": missing}
    null_counts = df.isna().sum()
    range_violations = {c: int((~df[c].between(lo, hi)).sum()) for c, (lo, hi) in BOUNDS.items() if c in df and (~df[c].between(lo, hi)).any()}
    speed_expected = df.distance_km / (df.duration_minutes / 60.0)
    speed_error = (speed_expected - df.average_speed_kmh).abs()
    intensity_expected = df.energy_consumption_kwh / df.distance_km * 100.0
    intensity_error = (intensity_expected - df.consumption_kwh_per_100km).abs()
    checks = {
        "required_columns_present": not missing,
        "no_nulls": int(null_counts.sum()) == 0,
        "unique_sample_ids": not df.sample_id.duplicated().any(),
        "no_duplicate_rows": not df.duplicated().any(),
        "numeric_ranges_valid": not range_violations,
        "distance_duration_speed_identity": bool((speed_error <= 0.002).all()),
        "energy_intensity_identity": bool((intensity_error <= 0.002).all()),
        "positive_distance_and_duration": bool((df.distance_km.gt(0) & df.duration_minutes.gt(0)).all()),
    }
    return {
        "passed": all(checks.values()), "row_count": int(len(df)), "column_count": int(len(df.columns)),
        "checks": checks, "null_counts": {k: int(v) for k, v in null_counts.items() if v},
        "duplicate_sample_ids": int(df.sample_id.duplicated().sum()), "range_violations": range_violations,
        "speed_identity_max_abs_error_kmh": float(speed_error.max()),
        "speed_identity_mismatch_count": int((speed_error > 0.002).sum()),
        "energy_intensity_max_abs_error": float(intensity_error.max()),
        "energy_intensity_mismatch_count": int((intensity_error > 0.002).sum()),
        "target_summary": {"energy_kwh_mean": float(df.energy_consumption_kwh.mean()), "energy_kwh_min": float(df.energy_consumption_kwh.min()), "energy_kwh_max": float(df.energy_consumption_kwh.max()), "kwh_per_100km_mean": float(df.consumption_kwh_per_100km.mean())},
    }


def write_quality_report(df: pd.DataFrame, output_path: str | Path) -> dict:
    report = quality_report(df)
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
