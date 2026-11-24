"""Test the tuned planning model. Run from anywhere: python test_model.py
1) What-if sanity checks: does the prediction move in a physically sensible direction?
2) Energy/battery error on held-out vehicles (never used for training or tuning).
"""
import sys
from pathlib import Path
import joblib
import numpy as np
import polars as pl
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "data"))
from train_model import build_features, NUMERIC, CATEGORIES, TARGET, GROUP

df = pl.read_csv(ROOT / "data" / "cleaned_data.csv", infer_schema_length=None)
bundle = joblib.load(ROOT / "models_plan_tuned" / "model.joblib")
model, names = bundle["model"], bundle["features"]
CATS = [c for c in CATEGORIES if c != "current_driving_pattern"]


def predict(rows):
    X, n = build_features(pl.DataFrame(rows), use_current_pattern=False)
    assert n == names, "feature mismatch"
    return model.predict(X)


# ---- 1) what-if sanity: base row = median numerics / most common categories from the data
base = {c: float(df[c].median()) for c in NUMERIC}
base.update({c: df[c].drop_nulls().mode()[0] for c in CATS})
print("Base row:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in base.items()})
print(f"Base prediction: {predict([base])[0]:.2f} kWh/100km\n")

tests = [("speed_kmh", [50, 80, 110, 130]), ("outside_temperature_c", [-15, -5, 5, 20]),
         ("slope_grade_pct", [-3, 0, 3]), ("towing", ["no", "yes"]),
         ("road_condition", ["dry", "wet", "snow_or_ice"]), ("climate_settings", ["off", "low", "high"])]
for col, vals in tests:
    rows = [{**base, col: (float(v) if col in NUMERIC else v)} for v in vals]
    print(f"{col:24s}", "  ".join(f"{v}->{p:.1f}" for v, p in zip(vals, predict(rows))))
print("Expect: higher speed, colder, uphill, towing, snow, strong climate -> higher consumption.\n")

# ---- 2) held-out vehicles: energy error per segment and per vehicle
y = df[TARGET].to_numpy().astype(float)
g = df[GROUP].to_numpy()
X, _ = build_features(df, use_current_pattern=False)
_, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=42).split(X, y, g))
t = df[te].with_columns(pred_c=pl.Series(model.predict(X[te]))).drop_nulls(["energy_used_kwh", "distance_km"])
t = t.with_columns(pred_kwh=pl.col("pred_c") * pl.col("distance_km") / 100)

err = (t["pred_kwh"] - t["energy_used_kwh"]).to_numpy()
print(f"Segments: n={t.height} | energy MAE {abs(err).mean():.3f} kWh | bias {err.mean():+.3f} kWh (negative = under-predict)")

v = t.group_by(GROUP).agg(pl.col("pred_kwh").sum(), pl.col("energy_used_kwh").sum(), pl.col("distance_km").sum())
v = v.with_columns(((pl.col("pred_kwh") - pl.col("energy_used_kwh")) / pl.col("energy_used_kwh") * 100).alias("err_pct"))
print("\nPer held-out vehicle, total over all its segments (cumulative planning error):")
print(v.select([GROUP, "distance_km", "energy_used_kwh", "pred_kwh", "err_pct"]).sort(GROUP)
      .with_columns(pl.all().exclude(GROUP).round(1)))
print(f"\nMean |cumulative error| {v['err_pct'].abs().mean():.1f}% | worst {v['err_pct'].abs().max():.1f}%")