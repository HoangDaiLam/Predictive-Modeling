import numpy as np
import pandas as pd
import pytest

from evmlops.data import (
    LEAKAGE_COLUMNS,
    MODEL_FEATURES,
    RAW_INPUT_COLUMNS,
    TARGET,
    TripFeatureBuilder,
    group_train_test_split,
    load_dataset,
)


def test_leakage_columns_never_enter_the_model(small_df):
    features = TripFeatureBuilder().transform(small_df[list(RAW_INPUT_COLUMNS)])
    assert not set(LEAKAGE_COLUMNS) & set(features.columns)
    assert not set(LEAKAGE_COLUMNS) & set(RAW_INPUT_COLUMNS)
    assert TARGET not in features.columns


def test_target_is_deterministic_function_of_energy_and_distance(full_df):
    reconstructed = 100.0 * full_df["energy_used_kwh"] / full_df["distance_km"]
    assert (reconstructed - full_df[TARGET]).abs().max() < 1e-2


def test_feature_builder_output_schema(small_df):
    out = TripFeatureBuilder().transform(small_df[list(RAW_INPUT_COLUMNS)])
    assert list(out.columns) == list(MODEL_FEATURES)
    assert len(out) == len(small_df)
    assert out["hour_sin"].between(-1, 1).all()
    assert out["hour_cos"].between(-1, 1).all()
    assert out["is_weekend"].isin([0.0, 1.0]).all()
    expected = small_df["battery_temperature_c"] - small_df["outside_temperature_c"]
    pd.testing.assert_series_equal(out["temp_delta_c"], expected, check_names=False)


def test_feature_builder_does_not_mutate_input(small_df):
    raw = small_df[list(RAW_INPUT_COLUMNS)]
    before = raw.copy(deep=True)
    TripFeatureBuilder().transform(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_feature_builder_rejects_missing_columns(small_df):
    with pytest.raises(ValueError, match="missing required columns"):
        TripFeatureBuilder().transform(small_df[["distance_km"]])


def test_group_split_has_no_group_overlap(small_df):
    split = group_train_test_split(small_df, "vehicle_id", test_size=0.3, random_state=1)
    assert not set(split.groups_train) & set(split.groups_test)
    assert len(split.y_train) + len(split.y_test) == len(small_df)


def test_group_split_is_reproducible(small_df):
    a = group_train_test_split(small_df, "vehicle_id", 0.3, 5)
    b = group_train_test_split(small_df, "vehicle_id", 0.3, 5)
    assert set(a.groups_test) == set(b.groups_test)


def test_group_split_unknown_column(small_df):
    with pytest.raises(ValueError, match="Unknown group column"):
        group_train_test_split(small_df, "nope", 0.3, 0)


def test_load_dataset_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "absent.csv")


def test_load_dataset_validates_schema_and_target(small_df, tmp_path):
    missing = tmp_path / "missing_col.csv"
    small_df.drop(columns=["speed_kmh"]).to_csv(missing, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset(missing)

    bad_target = small_df.copy()
    bad_target.loc[0, TARGET] = -1.0
    path = tmp_path / "bad_target.csv"
    bad_target.to_csv(path, index=False)
    with pytest.raises(ValueError, match="strictly positive"):
        load_dataset(path)


def test_dataset_has_expected_missingness(full_df):
    # The file is described as cleaned but still contains NaNs; models must cope.
    assert full_df[list(RAW_INPUT_COLUMNS)].isna().any().any()
    assert not np.isnan(full_df[TARGET]).any()
