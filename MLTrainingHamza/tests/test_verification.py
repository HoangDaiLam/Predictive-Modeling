"""The closed-form results in docs/mathematics.md must hold for the real libraries."""

import pytest

from evmlops import verification as v


def test_sklearn_first_stage_matches_mean_residual_formula(small_split):
    assert v.check_sklearn_first_stage(small_split) < 1e-8


def test_xgboost_first_stage_matches_regularised_leaf_formula(small_split):
    assert v.check_xgboost_first_stage(small_split) < 1e-3


def test_lightgbm_first_stage_matches_regularised_leaf_formula(small_split):
    assert v.check_lightgbm_first_stage(small_split) < 1e-5


def test_xgboost_reported_gain_equals_closed_form(small_split):
    formula, reported = v.check_xgboost_split_gain(small_split)
    assert reported == pytest.approx(formula, rel=1e-6)


def test_lightgbm_reported_gain_equals_closed_form(small_split):
    formula, reported = v.check_lightgbm_split_gain(small_split)
    assert reported == pytest.approx(formula, rel=1e-6)


def test_report_marks_everything_passed(small_split):
    report = v.run_derivation_checks(small_split)
    assert report["passed"].all()
    assert len(report) == 5
