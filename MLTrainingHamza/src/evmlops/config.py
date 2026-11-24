"""Central configuration for the experiment.

All tunable values live here so that a run is fully described by one
serialisable object, which is also logged to MLflow.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ExperimentConfig:
    """Run configuration.

    Attributes:
        data_path: Location of the cleaned trip-level CSV file.
        tracking_uri: MLflow tracking URI (SQLite by default so that the
            model registry and the UI both work out of the box).
        artifact_root: Directory where run artifacts (models, plots) are stored.
        experiment_name: MLflow experiment under which runs are stored.
        group_column: Column used to keep related rows on the same side of
            every split. Splitting by vehicle measures generalisation to
            vehicles never seen during training.
        test_size: Fraction of groups held out as the final test set.
        cv_folds: Number of GroupKFold folds used during hyperparameter search.
        n_search_iter: Random search budget per candidate model.
        random_state: Seed used by every stochastic component.
        n_bootstrap: Cluster bootstrap replicates for confidence intervals.
        n_jobs: Parallelism for search and permutation importance.
        permutation_repeats: Repeats for permutation importance.
        register_best_model: Register the winning model in the MLflow registry.
        registered_model_name: Registry name used when registering.
        log_search_trials: Log every search trial as its own nested run.
        candidates: Candidate model keys to train and compare.
    """

    data_path: Path = PROJECT_ROOT / "data" / "cleaned_data.csv"
    tracking_uri: str = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
    artifact_root: Path = PROJECT_ROOT / "mlartifacts"
    experiment_name: str = "ev-energy-consumption"
    group_column: str = "vehicle_id"
    test_size: float = 0.2
    cv_folds: int = 5
    n_search_iter: int = 12
    random_state: int = 42
    n_bootstrap: int = 1000
    n_jobs: int = 1
    permutation_repeats: int = 5
    register_best_model: bool = True
    registered_model_name: str = "ev-consumption-predictor"
    log_search_trials: bool = True
    candidates: tuple[str, ...] = field(
        default=("gradient_boosting", "xgboost", "lightgbm")
    )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML friendly representation."""
        out = asdict(self)
        out["data_path"] = str(self.data_path)
        out["artifact_root"] = str(self.artifact_root)
        out["candidates"] = list(self.candidates)
        return out

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load overrides from a YAML file; unspecified keys keep defaults."""
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if "data_path" in raw:
            raw["data_path"] = Path(raw["data_path"])
        if "artifact_root" in raw:
            raw["artifact_root"] = Path(raw["artifact_root"])
        if "candidates" in raw:
            raw["candidates"] = tuple(raw["candidates"])
        return cls(**raw)
