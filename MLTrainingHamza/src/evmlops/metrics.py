"""Regression metrics and cluster-bootstrap uncertainty estimates.

Rows from one vehicle are correlated, so a plain row-level bootstrap would
understate uncertainty. Resampling whole groups (vehicles) with replacement
respects that dependence.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Point metrics on one prediction set.

    Returns:
        rmse, mae, r2, mape (percent), median_ae and bias (mean signed error,
        positive means over-prediction).
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    err = p - y
    sse = float(np.sum(err**2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(sse / len(y))),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1.0 - sse / sst) if sst > 0 else float("nan"),
        "mape": float(np.mean(np.abs(err) / np.abs(y)) * 100.0),
        "median_ae": float(np.median(np.abs(err))),
        "bias": float(np.mean(err)),
    }


@dataclass(frozen=True)
class _GroupSums:
    """Per-group sufficient statistics for RMSE, MAE and R-squared."""

    n: np.ndarray
    sum_sq_err: np.ndarray
    sum_abs_err: np.ndarray
    sum_y: np.ndarray
    sum_y2: np.ndarray


def _group_sums(y: np.ndarray, pred: np.ndarray, groups: np.ndarray) -> _GroupSums:
    frame = pd.DataFrame(
        {
            "g": groups,
            "n": 1.0,
            "se": (pred - y) ** 2,
            "ae": np.abs(pred - y),
            "y": y,
            "y2": y**2,
        }
    )
    agg = frame.groupby("g", sort=True).sum()
    return _GroupSums(
        n=agg["n"].to_numpy(),
        sum_sq_err=agg["se"].to_numpy(),
        sum_abs_err=agg["ae"].to_numpy(),
        sum_y=agg["y"].to_numpy(),
        sum_y2=agg["y2"].to_numpy(),
    )


def _metrics_from_counts(sums: _GroupSums, counts: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorised metrics for B bootstrap replicates given a (B, G) count matrix."""
    n = counts @ sums.n
    sse = counts @ sums.sum_sq_err
    sae = counts @ sums.sum_abs_err
    sy = counts @ sums.sum_y
    sy2 = counts @ sums.sum_y2
    sst = sy2 - sy**2 / n
    return {"rmse": np.sqrt(sse / n), "mae": sae / n, "r2": 1.0 - sse / sst}


def cluster_bootstrap(
    y_true,
    predictions: dict[str, np.ndarray],
    groups,
    n_bootstrap: int = 1000,
    random_state: int = 0,
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group-level bootstrap intervals for each model and for RMSE differences.

    The same resampled groups are used for every model in a replicate, so the
    pairwise differences are paired comparisons.

    Args:
        y_true: Observed targets on the test set.
        predictions: Mapping from model name to its test predictions.
        groups: Group label (vehicle) for each test row.
        n_bootstrap: Number of replicates.
        random_state: Seed.
        alpha: Two-sided level; 0.05 gives 95 percent intervals.

    Returns:
        ``(intervals, pairwise)`` data frames. ``intervals`` has one row per
        (model, metric) with the point estimate and percentile interval.
        ``pairwise`` has one row per model pair with the RMSE difference
        ``rmse_a - rmse_b``, its interval, and the share of replicates in
        which model ``a`` had the lower RMSE.
    """
    y = np.asarray(y_true, dtype=float)
    group_arr = np.asarray(groups)
    n_groups = len(np.unique(group_arr))
    rng = np.random.default_rng(random_state)
    counts = rng.multinomial(n_groups, np.full(n_groups, 1.0 / n_groups), size=n_bootstrap)

    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    boot: dict[str, dict[str, np.ndarray]] = {}
    rows = []
    for name, pred in predictions.items():
        sums = _group_sums(y, np.asarray(pred, dtype=float), group_arr)
        boot[name] = _metrics_from_counts(sums, counts.astype(float))
        point = _metrics_from_counts(sums, np.ones((1, n_groups)))
        for metric, values in boot[name].items():
            rows.append(
                {
                    "model": name,
                    "metric": metric,
                    "estimate": float(point[metric][0]),
                    "ci_low": float(np.percentile(values, lo_q)),
                    "ci_high": float(np.percentile(values, hi_q)),
                }
            )

    pair_rows = []
    for a, b in combinations(predictions, 2):
        diff = boot[a]["rmse"] - boot[b]["rmse"]
        pair_rows.append(
            {
                "model_a": a,
                "model_b": b,
                "rmse_diff_a_minus_b": float(np.mean(diff)),
                "ci_low": float(np.percentile(diff, lo_q)),
                "ci_high": float(np.percentile(diff, hi_q)),
                "prob_a_better": float(np.mean(diff < 0)),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(pair_rows)
