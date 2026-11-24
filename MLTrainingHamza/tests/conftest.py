"""Shared fixtures: a small, vehicle-balanced slice of the real dataset."""

from __future__ import annotations

import pytest

from evmlops.config import PROJECT_ROOT
from evmlops.data import DataSplit, group_train_test_split, load_dataset

DATA_PATH = PROJECT_ROOT / "data" / "cleaned_data.csv"
N_VEHICLES = 14


@pytest.fixture(scope="session")
def full_df():
    return load_dataset(DATA_PATH)


@pytest.fixture(scope="session")
def small_df(full_df):
    vehicles = sorted(full_df["vehicle_id"].unique())[:N_VEHICLES]
    return full_df[full_df["vehicle_id"].isin(vehicles)].reset_index(drop=True)


@pytest.fixture(scope="session")
def small_split(small_df) -> DataSplit:
    return group_train_test_split(small_df, "vehicle_id", test_size=0.3, random_state=0)
