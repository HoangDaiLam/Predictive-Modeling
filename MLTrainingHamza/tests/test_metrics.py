import numpy as np
import pytest

from evmlops.metrics import cluster_bootstrap, regression_metrics


def test_regression_metrics_known_values():
    y = np.array([10.0, 20.0, 30.0])
    p = np.array([12.0, 18.0, 33.0])
    m = regression_metrics(y, p)
    assert m["rmse"] == pytest.approx(np.sqrt((4 + 4 + 9) / 3))
    assert m["mae"] == pytest.approx((2 + 2 + 3) / 3)
    assert m["bias"] == pytest.approx((2 - 2 + 3) / 3)
    assert m["median_ae"] == pytest.approx(2.0)
    assert m["mape"] == pytest.approx(100 * (0.2 + 0.1 + 0.1) / 3)
    assert m["r2"] == pytest.approx(1 - 17 / 200)


def test_perfect_prediction():
    y = np.array([5.0, 8.0, 13.0])
    m = regression_metrics(y, y)
    assert m["rmse"] == 0.0 and m["r2"] == 1.0


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(3)
    y = rng.gamma(4.0, 5.0, 600)
    groups = rng.integers(0, 12, 600)
    good = y + rng.normal(0, 1.5, 600)
    poor = y + rng.normal(0, 4.0, 600)
    return y, {"good": good, "poor": poor}, groups


def test_bootstrap_intervals_bracket_the_estimate(synthetic):
    y, preds, groups = synthetic
    intervals, _ = cluster_bootstrap(y, preds, groups, n_bootstrap=400, random_state=0)
    assert (intervals["ci_low"] <= intervals["estimate"]).all()
    assert (intervals["estimate"] <= intervals["ci_high"]).all()


def test_bootstrap_point_estimate_matches_direct_metric(synthetic):
    y, preds, groups = synthetic
    intervals, _ = cluster_bootstrap(y, preds, groups, n_bootstrap=50, random_state=0)
    direct = regression_metrics(y, preds["good"])
    row = intervals[(intervals["model"] == "good") & (intervals["metric"] == "rmse")].iloc[0]
    assert row["estimate"] == pytest.approx(direct["rmse"])


def test_bootstrap_identifies_better_model(synthetic):
    y, preds, groups = synthetic
    _, pairwise = cluster_bootstrap(y, preds, groups, n_bootstrap=400, random_state=0)
    row = pairwise.iloc[0]
    assert row["model_a"] == "good"
    assert row["rmse_diff_a_minus_b"] < 0
    assert row["ci_high"] < 0
    assert row["prob_a_better"] > 0.99


def test_bootstrap_is_deterministic(synthetic):
    y, preds, groups = synthetic
    a, _ = cluster_bootstrap(y, preds, groups, n_bootstrap=100, random_state=9)
    b, _ = cluster_bootstrap(y, preds, groups, n_bootstrap=100, random_state=9)
    assert a.equals(b)
