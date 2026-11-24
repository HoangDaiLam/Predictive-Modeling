"""End-to-end run against a throwaway MLflow store."""

from __future__ import annotations

import dataclasses

import mlflow
import pandas as pd
import pytest

from evmlops.config import ExperimentConfig
from evmlops.data import RAW_INPUT_COLUMNS
from evmlops.pipeline import run_experiment

N_TRIALS = 2


@pytest.fixture(scope="module")
def completed(tmp_path_factory, small_df):
    root = tmp_path_factory.mktemp("mlflow_store")
    data_path = root / "small.csv"
    small_df.to_csv(data_path, index=False)
    config = ExperimentConfig(
        data_path=data_path,
        tracking_uri=f"sqlite:///{root / 'mlflow.db'}",
        artifact_root=root / "artifacts",
        experiment_name="integration-test",
        test_size=0.3,
        cv_folds=2,
        n_search_iter=N_TRIALS,
        n_bootstrap=50,
        permutation_repeats=2,
    )
    summary = run_experiment(config)
    return config, summary


def test_summary_contains_all_candidates(completed):
    _, summary = completed
    assert set(summary.comparison["model"]) == {"gradient_boosting", "xgboost", "lightgbm"}
    assert summary.winner == summary.comparison.loc[0, "model"]
    assert summary.comparison["cv_rmse"].is_monotonic_increasing
    assert summary.derivation_checks["passed"].all()


def test_run_hierarchy_is_logged(completed):
    config, summary = completed
    mlflow.set_tracking_uri(config.tracking_uri)
    client = mlflow.MlflowClient()
    runs = client.search_runs([summary.experiment_id], max_results=200)
    kinds = pd.Series([r.data.tags.get("run_kind") for r in runs]).value_counts().to_dict()
    assert kinds == {"comparison": 1, "candidate": 3, "search_trial": 3 * N_TRIALS}

    parent = client.get_run(summary.parent_run_id)
    assert parent.data.tags["winner"] == summary.winner
    assert parent.data.metrics["derivation_checks_passed"] == 1.0
    for key in ("xgboost", "lightgbm", "gradient_boosting"):
        assert f"{key}_test_rmse" in parent.data.metrics

    artifacts = {a.path for a in client.list_artifacts(summary.parent_run_id, "comparison")}
    assert "comparison/comparison_table.csv" in artifacts
    assert "comparison/test_rmse.png" in artifacts


def test_candidate_runs_log_params_metrics_and_model(completed):
    config, summary = completed
    mlflow.set_tracking_uri(config.tracking_uri)
    client = mlflow.MlflowClient()
    for result in summary.results.values():
        run = client.get_run(result.run_id)
        assert run.data.metrics["test_rmse"] == pytest.approx(result.test_metrics["rmse"])
        assert "cv_rmse" in run.data.metrics
        assert any(k.startswith("best_") for k in run.data.params)
        assert client.search_logged_models(experiment_ids=[summary.experiment_id])


def test_no_test_set_leakage_into_selection(completed):
    _, summary = completed
    # The selected model is the one with the lowest CV RMSE, by construction.
    best_cv = summary.comparison["cv_rmse"].min()
    assert summary.comparison.loc[summary.comparison["model"] == summary.winner, "cv_rmse"].iloc[0] == best_cv


def test_registered_champion_serves_raw_rows(completed, small_df):
    config, summary = completed
    mlflow.set_tracking_uri(config.tracking_uri)
    assert summary.registered_version == "1"
    model = mlflow.pyfunc.load_model(f"models:/{config.registered_model_name}@champion")
    rows = small_df[list(RAW_INPUT_COLUMNS)].head(8)
    predictions = model.predict(rows)
    assert len(predictions) == 8
    assert pd.Series(predictions).notna().all()


def test_registration_can_be_disabled(tmp_path, small_df):
    data_path = tmp_path / "small.csv"
    small_df.to_csv(data_path, index=False)
    config = ExperimentConfig(
        data_path=data_path,
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        artifact_root=tmp_path / "artifacts",
        experiment_name="no-register",
        test_size=0.3,
        cv_folds=2,
        n_search_iter=1,
        n_bootstrap=20,
        permutation_repeats=1,
        register_best_model=False,
        log_search_trials=False,
        candidates=("lightgbm",),
    )
    summary = run_experiment(dataclasses.replace(config))
    assert summary.registered_version is None
    mlflow.set_tracking_uri(config.tracking_uri)
    client = mlflow.MlflowClient()
    runs = client.search_runs([summary.experiment_id])
    assert not any(r.data.tags.get("run_kind") == "search_trial" for r in runs)
