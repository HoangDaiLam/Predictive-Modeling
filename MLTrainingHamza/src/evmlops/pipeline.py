"""MLflow experiment: tune, evaluate and compare the candidate models.

Run layout in the tracking server::

    comparison-<timestamp>                 (parent: data, maths checks, comparison)
    +-- <candidate display name>           (one per model family)
        +-- trial-00 ... trial-NN          (one per random-search configuration)

Model selection uses grouped cross-validation RMSE on the training vehicles
only. The held-out vehicles are touched once per model, to report results.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import mlflow.data
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

from . import plots
from .config import PROJECT_ROOT, ExperimentConfig
from .data import (
    CATEGORICAL_FEATURES,
    LEAKAGE_COLUMNS,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    DataSplit,
    group_train_test_split,
    load_dataset,
)
from .diagnostics import native_importance, permutation_importance_table, staged_rmse
from .metrics import cluster_bootstrap, regression_metrics
from .models import Candidate, build_baseline, get_candidate
from .verification import run_derivation_checks

LOGGER = logging.getLogger(__name__)

SCORING = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}


@dataclass
class CandidateResult:
    """Everything produced while fitting one candidate."""

    key: str
    display_name: str
    run_id: str
    model_uri: str
    pipeline: Any
    best_params: dict[str, Any]
    cv_rmse: float
    cv_rmse_std: float
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    test_predictions: np.ndarray
    search_frame: pd.DataFrame
    staged: pd.DataFrame
    permutation: pd.DataFrame
    fit_seconds: float
    latency_ms_per_1000: float


@dataclass
class ExperimentSummary:
    """Returned to the caller after a completed experiment."""

    parent_run_id: str
    experiment_id: str
    winner: str
    comparison: pd.DataFrame
    intervals: pd.DataFrame
    pairwise: pd.DataFrame
    derivation_checks: pd.DataFrame
    registered_version: str | None = None
    results: dict[str, CandidateResult] = field(default_factory=dict)


def _python_scalar(value: Any) -> Any:
    """Convert numpy scalars to built-in types for MLflow parameter logging."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _configure_tracking(config: ExperimentConfig) -> str:
    """Point MLflow at the configured store and return the experiment id."""
    mlflow.set_tracking_uri(config.tracking_uri)
    config.artifact_root.mkdir(parents=True, exist_ok=True)
    experiment = mlflow.get_experiment_by_name(config.experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            config.experiment_name, artifact_location=config.artifact_root.resolve().as_uri()
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


def _log_figure(fig: plt.Figure, artifact_file: str) -> None:
    mlflow.log_figure(fig, artifact_file)
    plt.close(fig)


def _log_datasets(config: ExperimentConfig, split: DataSplit) -> None:
    """Attach the train and test partitions to the run as MLflow datasets."""
    for name, X, y, context in (
        ("train_vehicles", split.X_train, split.y_train, "training"),
        ("test_vehicles", split.X_test, split.y_test, "testing"),
    ):
        frame = pd.concat([X, y.rename(TARGET)], axis=1)
        dataset = mlflow.data.from_pandas(frame, source=str(config.data_path), name=name, targets=TARGET)
        mlflow.log_input(dataset, context=context)


def _log_trials(candidate: Candidate, search: RandomizedSearchCV) -> pd.DataFrame:
    """Record each search configuration as a nested run and return a summary table."""
    results = search.cv_results_
    rows = []
    for i, params in enumerate(results["params"]):
        row = {k.replace("model__", ""): _python_scalar(v) for k, v in params.items()}
        row.update(
            mean_cv_rmse=float(-results["mean_test_rmse"][i]),
            std_cv_rmse=float(results["std_test_rmse"][i]),
            mean_cv_mae=float(-results["mean_test_mae"][i]),
            mean_cv_r2=float(results["mean_test_r2"][i]),
            mean_fit_seconds=float(results["mean_fit_time"][i]),
            rank=int(results["rank_test_rmse"][i]),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _emit_trial_runs(frame: pd.DataFrame, param_names: list[str]) -> None:
    for i, row in frame.iterrows():
        with mlflow.start_run(run_name=f"trial-{i:02d}", nested=True):
            mlflow.set_tag("run_kind", "search_trial")
            mlflow.log_params({name: row[name] for name in param_names})
            mlflow.log_metrics(
                {
                    "cv_rmse": row["mean_cv_rmse"],
                    "cv_rmse_std": row["std_cv_rmse"],
                    "cv_mae": row["mean_cv_mae"],
                    "cv_r2": row["mean_cv_r2"],
                    "fit_seconds": row["mean_fit_seconds"],
                    "rank": row["rank"],
                }
            )


def _time_prediction(pipeline: Any, X: pd.DataFrame, repeats: int = 3) -> float:
    """Best-of-N inference latency in milliseconds per 1000 rows."""
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        pipeline.predict(X)
        timings.append(time.perf_counter() - start)
    return min(timings) / len(X) * 1000.0 * 1000.0


def fit_candidate(candidate: Candidate, split: DataSplit, config: ExperimentConfig) -> CandidateResult:
    """Tune, refit, evaluate and log a single candidate inside a nested run."""
    LOGGER.info("Tuning %s (%d trials x %d folds)", candidate.display_name, config.n_search_iter, config.cv_folds)

    with mlflow.start_run(run_name=candidate.display_name, nested=True) as run:
        mlflow.set_tags(
            {
                "run_kind": "candidate",
                "model_family": candidate.key,
                "preprocessing": candidate.preprocessing,
            }
        )

        search = RandomizedSearchCV(
            estimator=candidate.build(config.random_state, config.n_jobs),
            param_distributions=candidate.search_space,
            n_iter=config.n_search_iter,
            scoring=SCORING,
            refit="rmse",
            cv=GroupKFold(n_splits=config.cv_folds),
            random_state=config.random_state,
            n_jobs=config.n_jobs,
            error_score="raise",
        )
        search.fit(split.X_train, split.y_train, groups=split.groups_train)

        trials = _log_trials(candidate, search)
        param_names = [k.replace("model__", "") for k in candidate.search_space]
        if config.log_search_trials:
            _emit_trial_runs(trials, param_names)

        best = search.best_estimator_
        best_params = {k.replace("model__", ""): _python_scalar(v) for k, v in search.best_params_.items()}
        best_row = trials.loc[trials["rank"] == 1].iloc[0]

        train_pred = best.predict(split.X_train)
        test_pred = best.predict(split.X_test)
        train_metrics = regression_metrics(split.y_train, train_pred)
        test_metrics = regression_metrics(split.y_test, test_pred)
        latency = _time_prediction(best, split.X_test)

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_metrics(
            {
                "cv_rmse": float(best_row["mean_cv_rmse"]),
                "cv_rmse_std": float(best_row["std_cv_rmse"]),
                "cv_mae": float(best_row["mean_cv_mae"]),
                "cv_r2": float(best_row["mean_cv_r2"]),
                "refit_seconds": float(search.refit_time_),
                "predict_ms_per_1000_rows": latency,
                "generalisation_gap_rmse": test_metrics["rmse"] - train_metrics["rmse"],
            }
        )

        staged = staged_rmse(best, split.X_test, split.y_test)
        permutation = permutation_importance_table(
            best, split.X_test, split.y_test, config.permutation_repeats, config.random_state, config.n_jobs
        )
        mlflow.log_table(trials, "search/trials.json")
        mlflow.log_text(trials.to_csv(index=False), "search/trials.csv")
        mlflow.log_text(native_importance(best).to_csv(index=False), "importance/native.csv")
        mlflow.log_text(permutation.to_csv(index=False), "importance/permutation.csv")
        mlflow.log_text(staged.to_csv(index=False), "diagnostics/staged_rmse.csv")

        signature = infer_signature(split.X_train.head(200), best.predict(split.X_train.head(200)))
        model_info = mlflow.sklearn.log_model(
            best,
            name="model",
            signature=signature,
            input_example=split.X_train.head(5),
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

        LOGGER.info(
            "%s: CV RMSE %.3f, test RMSE %.3f, test R2 %.3f",
            candidate.display_name,
            best_row["mean_cv_rmse"],
            test_metrics["rmse"],
            test_metrics["r2"],
        )
        return CandidateResult(
            key=candidate.key,
            display_name=candidate.display_name,
            run_id=run.info.run_id,
            model_uri=model_info.model_uri,
            pipeline=best,
            best_params=best_params,
            cv_rmse=float(best_row["mean_cv_rmse"]),
            cv_rmse_std=float(best_row["std_cv_rmse"]),
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            test_predictions=test_pred,
            search_frame=trials,
            staged=staged,
            permutation=permutation,
            fit_seconds=float(search.refit_time_),
            latency_ms_per_1000=latency,
        )


def _comparison_table(results: dict[str, CandidateResult], baseline: dict[str, float]) -> pd.DataFrame:
    rows = []
    for key, res in results.items():
        rows.append(
            {
                "model": key,
                "cv_rmse": res.cv_rmse,
                "cv_rmse_std": res.cv_rmse_std,
                "test_rmse": res.test_metrics["rmse"],
                "test_mae": res.test_metrics["mae"],
                "test_r2": res.test_metrics["r2"],
                "test_mape_pct": res.test_metrics["mape"],
                "test_bias": res.test_metrics["bias"],
                "train_rmse": res.train_metrics["rmse"],
                "refit_seconds": res.fit_seconds,
                "predict_ms_per_1000_rows": res.latency_ms_per_1000,
                "rmse_vs_baseline_pct": 100.0 * (1.0 - res.test_metrics["rmse"] / baseline["rmse"]),
            }
        )
    return pd.DataFrame(rows).sort_values("cv_rmse").reset_index(drop=True)


def _describe_run(summary_table: pd.DataFrame, winner: str, baseline: dict[str, float]) -> str:
    lines = [
        "## Gradient-boosted EV consumption models",
        "",
        "Target: `consumption_kwh_per_100km`. Held-out evaluation uses vehicles never seen in training.",
        f"Selection criterion: grouped cross-validation RMSE. **Selected model: `{winner}`**.",
        f"Mean-predictor baseline test RMSE: {baseline['rmse']:.3f}.",
        "",
        "| Model | CV RMSE | Test RMSE | Test MAE | Test R2 |",
        "|---|---|---|---|---|",
    ]
    for _, row in summary_table.iterrows():
        lines.append(
            f"| {row['model']} | {row['cv_rmse']:.3f} | {row['test_rmse']:.3f} | "
            f"{row['test_mae']:.3f} | {row['test_r2']:.3f} |"
        )
    return "\n".join(lines)


def run_experiment(config: ExperimentConfig | None = None) -> ExperimentSummary:
    """Execute the full comparison and return a structured summary."""
    config = config or ExperimentConfig()
    experiment_id = _configure_tracking(config)

    df = load_dataset(config.data_path)
    split = group_train_test_split(df, config.group_column, config.test_size, config.random_state)
    candidates = [get_candidate(key) for key in config.candidates]

    run_name = f"comparison-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    with mlflow.start_run(run_name=run_name) as parent:
        mlflow.set_tags(
            {
                "run_kind": "comparison",
                "target": TARGET,
                "leakage_columns_excluded": ",".join(LEAKAGE_COLUMNS),
                "split_strategy": f"GroupShuffleSplit by {config.group_column}",
            }
        )
        mlflow.log_params({f"config_{k}": v for k, v in config.to_dict().items()})
        mlflow.log_params(
            {
                "n_numeric_features": len(NUMERIC_FEATURES),
                "n_categorical_features": len(CATEGORICAL_FEATURES),
                "n_train_rows": len(split.y_train),
                "n_test_rows": len(split.y_test),
                "n_train_groups": split.groups_train.nunique(),
                "n_test_groups": split.groups_test.nunique(),
            }
        )
        mlflow.log_dict(
            {"numeric": list(NUMERIC_FEATURES), "categorical": list(CATEGORICAL_FEATURES), "all": list(MODEL_FEATURES)},
            "data/features.json",
        )
        _log_datasets(config, split)

        math_doc = PROJECT_ROOT / "docs" / "mathematics.md"
        if math_doc.is_file():
            mlflow.log_artifact(str(math_doc), artifact_path="documentation")

        checks = run_derivation_checks(split, config.random_state)
        mlflow.log_text(checks.to_csv(index=False), "documentation/derivation_checks.csv")
        mlflow.log_metric("derivation_checks_passed", float(checks["passed"].all()))
        LOGGER.info("Derivation checks:\n%s", checks.to_string(index=False))

        baseline_model = build_baseline().fit(split.X_train, split.y_train)
        baseline = regression_metrics(split.y_test, baseline_model.predict(split.X_test))
        mlflow.log_metrics({f"baseline_test_{k}": v for k, v in baseline.items()})

        results = {c.key: fit_candidate(c, split, config) for c in candidates}

        comparison = _comparison_table(results, baseline)
        winner = str(comparison.loc[0, "model"])
        predictions = {k: r.test_predictions for k, r in results.items()}
        intervals, pairwise = cluster_bootstrap(
            split.y_test, predictions, split.groups_test, config.n_bootstrap, config.random_state
        )

        for key, res in results.items():
            mlflow.log_metrics(
                {
                    f"{key}_cv_rmse": res.cv_rmse,
                    f"{key}_test_rmse": res.test_metrics["rmse"],
                    f"{key}_test_mae": res.test_metrics["mae"],
                    f"{key}_test_r2": res.test_metrics["r2"],
                }
            )
        mlflow.log_table(comparison, "comparison/comparison_table.json")
        mlflow.log_text(comparison.to_csv(index=False), "comparison/comparison_table.csv")
        mlflow.log_text(intervals.to_csv(index=False), "comparison/bootstrap_intervals.csv")
        mlflow.log_text(pairwise.to_csv(index=False), "comparison/pairwise_rmse_differences.csv")
        mlflow.log_dict(
            {k: {"best_params": r.best_params, "run_id": r.run_id} for k, r in results.items()},
            "comparison/best_params.json",
        )

        _log_figure(plots.metric_comparison(intervals, "rmse"), "comparison/test_rmse.png")
        _log_figure(plots.metric_comparison(intervals, "mae"), "comparison/test_mae.png")
        _log_figure(plots.actual_vs_predicted(split.y_test, predictions), "comparison/actual_vs_predicted.png")
        _log_figure(plots.residual_distributions(split.y_test, predictions), "comparison/residuals.png")
        _log_figure(plots.staged_curves({k: r.staged for k, r in results.items()}), "comparison/staged_error.png")
        _log_figure(plots.importance_comparison({k: r.permutation for k, r in results.items()}), "comparison/permutation_importance.png")
        _log_figure(plots.search_trials({k: r.search_frame for k, r in results.items()}), "comparison/search_trials.png")

        mlflow.set_tag("winner", winner)
        mlflow.set_tag("mlflow.note.content", _describe_run(comparison, winner, baseline))

        version = None
        if config.register_best_model:
            registered = mlflow.register_model(results[winner].model_uri, config.registered_model_name)
            version = str(registered.version)
            client = MlflowClient()
            client.set_registered_model_alias(config.registered_model_name, "champion", version)
            client.set_model_version_tag(
                config.registered_model_name, version, "selection_metric", "grouped_cv_rmse"
            )
            mlflow.set_tag("registered_model_version", version)
            LOGGER.info("Registered %s v%s as champion", config.registered_model_name, version)

        summary = ExperimentSummary(
            parent_run_id=parent.info.run_id,
            experiment_id=experiment_id,
            winner=winner,
            comparison=comparison,
            intervals=intervals,
            pairwise=pairwise,
            derivation_checks=checks,
            registered_version=version,
            results=results,
        )

    LOGGER.info("Comparison table:\n%s", comparison.round(4).to_string(index=False))
    return summary
