"""Out-of-fold residual analysis on training vehicles only (held-out vehicles untouched).

Usage: python residuals.py [--use-current-pattern]
bias > 0 means the model under-predicts consumption (risky for battery planning).
"""
import sys
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_predict
from train_model import build_features, make_models, TARGET, GROUP

use_cur = "--use-current-pattern" in sys.argv
df = pl.read_csv("data/cleaned_data.csv", infer_schema_length=None)
X, names = build_features(df, use_current_pattern=use_cur)
y = df[TARGET].to_numpy().astype(float)
g = df[GROUP].to_numpy()

tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=42).split(X, y, g))
oof = cross_val_predict(make_models(42)["hist_gbm"], X[tr], y[tr],
                        groups=g[tr], cv=GroupKFold(5))
res = y[tr] - oof
d = df[tr]

groups = {
    "all": np.ones(len(tr), dtype=bool),
    "towing": (d["towing"] == "yes").fill_null(False).to_numpy(),
    "snow_or_ice": (d["road_condition"] == "snow_or_ice").fill_null(False).to_numpy(),
    "cold (<0C)": (d["outside_temperature_c"] < 0).fill_null(False).to_numpy(),
    "downhill (<-1.5%)": (d["slope_grade_pct"] < -1.5).fill_null(False).to_numpy(),
}
for k, m in groups.items():
    if m.sum():
        print(f"{k:20s} n={m.sum():5d} MAE {abs(res[m]).mean():.3f} bias {res[m].mean():+.3f}")