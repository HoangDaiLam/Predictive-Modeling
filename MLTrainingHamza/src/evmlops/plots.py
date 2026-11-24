"""Matplotlib figures in the project palette (Prussian blue and gold)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

PRUSSIAN = "#003153"
PRUSSIAN_MID = "#1D4E7A"
PRUSSIAN_LIGHT = "#5B84B1"
GOLD = "#C9A227"
GOLD_LIGHT = "#E6C85C"
PAPER = "#FBFAF6"
INK = "#0B1F33"
GRID = "#D9D4C3"

MODEL_COLORS = {
    "gradient_boosting": PRUSSIAN_LIGHT,
    "xgboost": PRUSSIAN,
    "lightgbm": GOLD,
}
MODEL_LABELS = {
    "gradient_boosting": "Gradient Boosting",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "axes.edgecolor": PRUSSIAN,
            "axes.labelcolor": INK,
            "axes.titlecolor": PRUSSIAN,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "legend.frameon": False,
            "font.size": 10,
        }
    )


def _color(key: str) -> str:
    return MODEL_COLORS.get(key, PRUSSIAN_MID)


def _label(key: str) -> str:
    return MODEL_LABELS.get(key, key)


def actual_vs_predicted(y_true, predictions: dict[str, np.ndarray]) -> Figure:
    """One scatter panel per model against the identity line."""
    _style()
    y = np.asarray(y_true)
    fig, axes = plt.subplots(1, len(predictions), figsize=(5 * len(predictions), 4.6), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    upper = float(max(y.max(), *(np.max(p) for p in predictions.values())))
    for ax, (key, pred) in zip(axes, predictions.items()):
        ax.scatter(y, pred, s=8, alpha=0.35, color=_color(key), edgecolor="none")
        ax.plot([0, upper], [0, upper], color=GOLD if key != "lightgbm" else PRUSSIAN, lw=1.4)
        ax.set_title(_label(key))
        ax.set_xlabel("Observed (kWh/100 km)")
    axes[0].set_ylabel("Predicted (kWh/100 km)")
    fig.tight_layout()
    return fig


def residual_distributions(y_true, predictions: dict[str, np.ndarray]) -> Figure:
    """Overlaid residual histograms."""
    _style()
    y = np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    limit = np.percentile(np.abs(np.concatenate([p - y for p in predictions.values()])), 99)
    bins = np.linspace(-limit, limit, 60)
    for key, pred in predictions.items():
        ax.hist(pred - y, bins=bins, histtype="stepfilled", alpha=0.45, color=_color(key), label=_label(key))
        ax.hist(pred - y, bins=bins, histtype="step", color=_color(key), lw=1.2)
    ax.axvline(0.0, color=INK, lw=1.0)
    ax.set_xlabel("Residual: predicted - observed (kWh/100 km)")
    ax.set_ylabel("Trips")
    ax.set_title("Residual distribution on the held-out vehicles")
    ax.legend()
    fig.tight_layout()
    return fig


def metric_comparison(intervals: pd.DataFrame, metric: str = "rmse") -> Figure:
    """Point estimates with bootstrap intervals for one metric."""
    _style()
    data = intervals[intervals["metric"] == metric].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    label_x = float(data["ci_high"].max()) * 1.03
    for i, row in data.iterrows():
        ax.barh(i, row["estimate"], color=_color(row["model"]), height=0.55)
        err = [[row["estimate"] - row["ci_low"]], [row["ci_high"] - row["estimate"]]]
        # Light halo under the whisker keeps it visible on dark bars.
        ax.errorbar(row["estimate"], i, xerr=err, color=PAPER, capsize=5, lw=3.6, capthick=3.6, zorder=3)
        ax.errorbar(row["estimate"], i, xerr=err, color=INK, capsize=4, lw=1.4, zorder=4)
        ax.text(label_x, i, f"{row['estimate']:.3f}", va="center", fontsize=9, fontweight="bold")
    ax.set_xlim(0, label_x * 1.1)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([_label(m) for m in data["model"]])
    ax.invert_yaxis()
    ax.set_xlabel(f"{metric.upper()} (95% cluster bootstrap interval)")
    ax.set_title("Held-out test performance")
    fig.tight_layout()
    return fig


def staged_curves(curves: dict[str, pd.DataFrame]) -> Figure:
    """Test RMSE against number of boosting iterations."""
    _style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, frame in curves.items():
        ax.plot(frame["iteration"], frame["rmse"], color=_color(key), lw=2.0, label=_label(key))
    ax.set_xlabel("Boosting iterations (m)")
    ax.set_ylabel("Test RMSE (kWh/100 km)")
    ax.set_title("Staged error of the additive model F_m")
    ax.legend()
    fig.tight_layout()
    return fig


def importance_comparison(tables: dict[str, pd.DataFrame], top: int = 10) -> Figure:
    """Side-by-side permutation importance for the most important inputs."""
    _style()
    merged = None
    for key, table in tables.items():
        part = table[["feature", "rmse_increase_mean"]].rename(columns={"rmse_increase_mean": key})
        merged = part if merged is None else merged.merge(part, on="feature", how="outer")
    merged = merged.fillna(0.0)
    merged["mean"] = merged[list(tables)].mean(axis=1)
    merged = merged.sort_values("mean", ascending=False).head(top).iloc[::-1]

    height = 0.8 / len(tables)
    y = np.arange(len(merged))
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for i, key in enumerate(tables):
        ax.barh(y + i * height, merged[key], height=height, color=_color(key), label=_label(key))
    ax.set_yticks(y + height * (len(tables) - 1) / 2)
    ax.set_yticklabels(merged["feature"])
    ax.set_xlabel("Increase in test RMSE when shuffled (kWh/100 km)")
    ax.set_title("Permutation importance (raw inputs)")
    ax.legend()
    fig.tight_layout()
    return fig


def search_trials(frames: dict[str, pd.DataFrame]) -> Figure:
    """Cross-validated RMSE of each random-search trial, sorted per model."""
    _style()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for key, frame in frames.items():
        scores = np.sort(frame["mean_cv_rmse"].to_numpy())
        ax.plot(np.arange(1, len(scores) + 1), scores, marker="o", color=_color(key), lw=1.8, label=_label(key))
    ax.set_xlabel("Search trial (ranked by CV RMSE)")
    ax.set_ylabel("Grouped CV RMSE (kWh/100 km)")
    ax.set_title("Hyperparameter search outcomes")
    ax.legend()
    fig.tight_layout()
    return fig
