import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pytest
from src.ev_platform.generator import generate_dataset
from src.ev_platform.quality import quality_report

def test_reproducible():
    assert generate_dataset(100, 7).equals(generate_dataset(100, 7))

def test_count_and_unique_ids():
    df = generate_dataset(250, 3)
    assert len(df) == 250 and df.sample_id.is_unique

def test_speed_distance_duration_identity():
    df = generate_dataset(500, 11)
    calculated = df.distance_km / (df.duration_minutes / 60)
    assert np.max(np.abs(calculated - df.average_speed_kmh)) < 0.002

def test_energy_intensity_identity():
    df = generate_dataset(500, 11)
    calculated = df.energy_consumption_kwh / df.distance_km * 100
    assert np.max(np.abs(calculated - df.consumption_kwh_per_100km)) < 0.002

def test_quality_gate():
    assert quality_report(generate_dataset(1000))["passed"]

def test_zero_rows_rejected():
    with pytest.raises(ValueError):
        generate_dataset(0)

def test_feature_helper_excludes_target_and_audit_columns():
    from src.ev_platform.features import make_xy, EXCLUDED_COLUMNS
    df = generate_dataset(50)
    X, y = make_xy(df)
    assert "consumption_kwh_per_100km" not in X.columns
    assert "energy_consumption_kwh" not in X.columns
    assert not (set(X.columns) & EXCLUDED_COLUMNS)
    assert len(X) == len(y) == 50

def test_energy_target_matches_audit_components():
    df = generate_dataset(500, 19)
    reconstructed = df.road_load_energy_battery_kwh + df.transient_energy_battery_kwh + df.hvac_energy_kwh + df.auxiliary_energy_kwh
    expected = reconstructed.clip(lower=0.05)
    assert np.max(np.abs(expected - df.energy_consumption_kwh)) < 0.00001
