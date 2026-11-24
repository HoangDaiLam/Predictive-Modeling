"""Application factory registered as the ``prussian-gold`` MLflow app.

Usage::

    mlflow server --app-name prussian-gold --backend-store-uri sqlite:///mlflow.db

The factory swaps the Flask app's static directory for a recoloured copy of
MLflow's UI (built once and cached) and then returns the same kind of app the
stock server would: a FastAPI wrapper under uvicorn, or the Flask app under
gunicorn or waitress.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow
from flask import Flask
from mlflow.environment_variables import _MLFLOW_SGI_NAME
from mlflow.server import app as default_flask_app

from .palette import build_themed_static

LOGGER = logging.getLogger(__name__)


def prepare_assets() -> Path:
    """Build or locate the themed static directory for the installed MLflow."""
    source = Path(default_flask_app.static_folder)
    return build_themed_static(source, mlflow.__version__)


def create_app(app: Flask = default_flask_app):
    """Return the MLflow server application with the theme applied."""
    app.static_folder = str(prepare_assets())
    LOGGER.info("Serving themed MLflow UI from %s", app.static_folder)

    if _MLFLOW_SGI_NAME.get() == "uvicorn":
        from mlflow.server.fastapi_app import create_fastapi_app

        return create_fastapi_app(app)
    return app
