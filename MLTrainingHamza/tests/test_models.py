import numpy as np
import pandas as pd
import pytest

from evmlops.data import RAW_INPUT_COLUMNS
from evmlops.models import CANDIDATES, build_baseline, get_candidate


@pytest.mark.parametrize("key", sorted(CANDIDATES))
def test_candidate_fits_and_predicts(key, small_split):
    pipeline = get_candidate(key).build(0, 1)
    pipeline.set_params(model__n_estimators=20)
    pipeline.fit(small_split.X_train, small_split.y_train)
    pred = pipeline.predict(small_split.X_test)
    assert pred.shape == (len(small_split.y_test),)
    assert np.isfinite(pred).all()


@pytest.mark.parametrize("key", sorted(CANDIDATES))
def test_candidate_beats_mean_baseline(key, small_split):
    baseline = build_baseline().fit(small_split.X_train, small_split.y_train)
    pipeline = get_candidate(key).build(0, 1)
    pipeline.set_params(model__n_estimators=60)
    pipeline.fit(small_split.X_train, small_split.y_train)

    def rmse(model):
        return float(np.sqrt(np.mean((model.predict(small_split.X_test) - small_split.y_test) ** 2)))

    assert rmse(pipeline) < rmse(baseline)


@pytest.mark.parametrize("key", ["xgboost", "lightgbm"])
def test_native_models_tolerate_unseen_categories_and_all_missing_rows(key, small_split):
    pipeline = get_candidate(key).build(0, 1)
    pipeline.set_params(model__n_estimators=10)
    pipeline.fit(small_split.X_train, small_split.y_train)

    unseen = small_split.X_test.head(3).copy()
    unseen["road_condition"] = "volcanic_ash"
    unseen.loc[unseen.index[0], list(set(RAW_INPUT_COLUMNS) - {"timestamp"})] = np.nan
    assert np.isfinite(pipeline.predict(unseen)).all()


def test_onehot_model_tolerates_unseen_categories(small_split):
    pipeline = get_candidate("gradient_boosting").build(0, 1)
    pipeline.set_params(model__n_estimators=10)
    pipeline.fit(small_split.X_train, small_split.y_train)
    unseen = small_split.X_test.head(3).copy()
    unseen["road_condition"] = "volcanic_ash"
    assert np.isfinite(pipeline.predict(unseen)).all()


def test_unknown_candidate_raises():
    with pytest.raises(ValueError, match="Unknown candidate"):
        get_candidate("random_forest")


def test_search_spaces_target_the_model_step():
    for candidate in CANDIDATES.values():
        assert candidate.search_space
        assert all(name.startswith("model__") for name in candidate.search_space)
