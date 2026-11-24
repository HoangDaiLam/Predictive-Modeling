"""Read-only check of the tuned planning model on the held-out vehicles.
Run from the project root (battery_eda_project): python check_bias.py
"""
import sys
from pathlib import Path
import joblib
import numpy as np
import polars as pl
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "data"))  # where train_model.py lives
from train_model import build_features, TARGET, GROUP

df = pl.read_csv(ROOT / "data" / "cleaned_data.csv", infer_schema_length=None)
X, _ = build_features(df, False)
y = df[TARGET].to_numpy().astype(float)
g = df[GROUP].to_numpy()
tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=42).split(X, y, g))

model = joblib.load(ROOT / "models_plan_tuned" / "model.joblib")["model"]
r = y[te] - model.predict(X[te])  # positive = model under-predicts consumption
print(f"held-out MAE {abs(r).mean():.3f} | bias {r.mean():+.3f}")
print(f"under-prediction percentiles 75/90/95: "
      f"{np.percentile(r, 75):.2f} / {np.percentile(r, 90):.2f} / {np.percentile(r, 95):.2f} kWh/100km")

tow = df.filter(pl.col("towing") == "yes")
print(f"towing: {tow.height} rows from {tow[GROUP].n_unique()} vehicles "
      f"({tow.filter(pl.col(GROUP).is_in(list(set(g[tr]))))[GROUP].n_unique()} in training vehicles)")