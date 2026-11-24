"""Command line interface.

Examples::

    evmlops train                      # tune, evaluate and compare the three models
    evmlops train --n-search-iter 25   # larger search budget
    evmlops ui                         # themed MLflow UI on http://127.0.0.1:5000
    evmlops ui --plain                 # stock MLflow UI
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import subprocess
import sys
from pathlib import Path

from .config import ExperimentConfig

LOGGER = logging.getLogger("evmlops")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evmlops",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="run the model comparison experiment")
    train.add_argument("--config", type=Path, help="YAML file with ExperimentConfig overrides")
    train.add_argument("--data-path", type=Path)
    train.add_argument("--tracking-uri")
    train.add_argument("--artifact-root", type=Path)
    train.add_argument("--experiment-name")
    train.add_argument("--n-search-iter", type=int)
    train.add_argument("--cv-folds", type=int)
    train.add_argument("--n-bootstrap", type=int)
    train.add_argument("--n-jobs", type=int)
    train.add_argument("--seed", type=int, dest="random_state")
    train.add_argument("--no-register", action="store_true", help="skip model registry registration")
    train.add_argument(
        "--no-trial-runs",
        action="store_true",
        help="do not log each search trial as a nested run",
    )

    ui = sub.add_parser("ui", help="serve the MLflow UI")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=5000)
    ui.add_argument("--workers", type=int, default=1)
    ui.add_argument("--backend-store-uri")
    ui.add_argument("--artifact-root", type=Path)
    ui.add_argument("--plain", action="store_true", help="serve the stock MLflow theme")
    return parser


def _config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    config = ExperimentConfig.from_yaml(args.config) if args.config else ExperimentConfig()
    overrides = {
        name: getattr(args, name)
        for name in (
            "data_path",
            "tracking_uri",
            "artifact_root",
            "experiment_name",
            "n_search_iter",
            "cv_folds",
            "n_bootstrap",
            "n_jobs",
            "random_state",
        )
        if getattr(args, name, None) is not None
    }
    if args.no_register:
        overrides["register_best_model"] = False
    if args.no_trial_runs:
        overrides["log_search_trials"] = False
    return dataclasses.replace(config, **overrides)


def _run_train(args: argparse.Namespace) -> int:
    from .pipeline import run_experiment

    config = _config_from_args(args)
    summary = run_experiment(config)
    LOGGER.info("Selected model: %s (parent run %s)", summary.winner, summary.parent_run_id)
    LOGGER.info("View the results with: evmlops ui --backend-store-uri %s", config.tracking_uri)
    return 0


def _run_ui(args: argparse.Namespace) -> int:
    defaults = ExperimentConfig()
    store = args.backend_store_uri or defaults.tracking_uri
    artifact_root = (args.artifact_root or defaults.artifact_root).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--backend-store-uri",
        store,
        "--default-artifact-root",
        str(artifact_root),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--workers",
        str(args.workers),
    ]
    if not args.plain:
        from .ui.app import prepare_assets

        # Build once here so server workers only ever read the cache.
        prepare_assets()
        command += ["--app-name", "prussian-gold"]

    LOGGER.info("Starting MLflow UI at http://%s:%d", args.host, args.port)
    return subprocess.call(command)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(argv)
    if args.command == "train":
        return _run_train(args)
    return _run_ui(args)


if __name__ == "__main__":
    raise SystemExit(main())
