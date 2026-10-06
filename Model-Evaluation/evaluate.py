"""
evaluate.py — the standardized ML evaluation layer.

This is the piece meant to outlive any particular model or training method.
It accepts:

    model    : any object with a .predict(X) method
    X_test   : feature matrix (unscaled — this module applies `scaler` itself)
    y_test   : true target values
    scaler   : optional fitted scaler (None if the model was trained on raw features)

...and produces predictions, regression metrics, and diagnostic plots. It
never trains anything and never assumes how the model got fitted, so a
teammate's model — trained in a notebook, a different framework, or by a
completely different person — plugs in exactly the same way as a model
trained by train.py in this repo.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error,
)


def predict(model, X_test, scaler=None):
    X_proc = scaler.transform(X_test) if scaler is not None else np.asarray(X_test)
    return model.predict(X_proc)


def compute_metrics(y_true, y_pred, safe_mape=True):
    """Core regression metrics. MAPE is guarded against near-zero targets."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    metrics = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred),
    }

    if safe_mape and np.min(np.abs(y_true)) < 1e-6:
        metrics["MAPE"] = None  # target crosses zero — MAPE is unreliable, flag rather than fake it
    else:
        metrics["MAPE"] = mean_absolute_percentage_error(y_true, y_pred) * 100

    return metrics


def evaluate(model, X_test, y_test, scaler=None):
    """One-stop evaluation: returns (y_pred, metrics_dict, residuals)."""
    y_pred = predict(model, X_test, scaler)
    metrics = compute_metrics(y_test, y_pred)
    residuals = np.asarray(y_test) - y_pred
    return y_pred, metrics, residuals


def compare_models(evaluated: dict) -> pd.DataFrame:
    """
    Build a comparison table from a dict of already-evaluated models.

    evaluated: {display_name: {"metrics": {...}, "train_time": float}}
    Not hard-coded — built entirely from whatever evaluation runs are passed in.
    """
    rows = []
    for name, info in evaluated.items():
        row = {"Model": name, **info["metrics"]}
        if "train_time" in info:
            row["Train time (s)"] = round(info["train_time"], 3)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Model")
    return df.sort_values("RMSE")


# ------------------------------------------------------------------
# Diagnostic plots
# ------------------------------------------------------------------

def plot_predicted_vs_actual(y_true, y_pred, unit="kWh/100km"):
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred, mode="markers", name="Predictions",
        marker=dict(opacity=0.55, size=7),
    ))
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction (y = x)",
        line=dict(dash="dash", color="gray"),
    ))
    fig.update_layout(
        xaxis_title=f"Actual ({unit})", yaxis_title=f"Predicted ({unit})",
        title="Predicted vs. Actual",
    )
    return fig


def plot_residuals(y_pred, residuals, unit="kWh/100km"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_pred, y=residuals, mode="markers", name="Residual",
        marker=dict(opacity=0.55, size=7),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        xaxis_title=f"Predicted ({unit})", yaxis_title=f"Residual = Actual - Predicted ({unit})",
        title="Residual Plot",
    )
    return fig


def plot_error_distribution(residuals, unit="kWh/100km"):
    fig = px.histogram(residuals, nbins=40, labels={"value": f"Residual ({unit})"})
    fig.update_layout(title="Error Distribution", showlegend=False)
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    return fig


def plot_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importance = pd.Series(model.feature_importances_, index=feature_names)
    elif hasattr(model, "coef_"):
        importance = pd.Series(np.abs(model.coef_), index=feature_names)
    else:
        return None
    importance = importance.sort_values(ascending=True).tail(15)
    fig = px.bar(importance, orientation="h", labels={"index": "Feature", "value": "Importance"})
    fig.update_layout(title="Feature Importance", showlegend=False)
    return fig