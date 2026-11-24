#!/usr/bin/env python3
"""
Baseline vs Ridge vs gradient boosting for consumption_kwh_per_100km.

- Features exclude leaky/identifier columns (energy_used_kwh, battery_level_end_pct, ids, timestamp).
- Final test = whole vehicles never used for training or model selection.
- Model selection = GroupKFold by vehicle_id on the training vehicles only.
- Preprocessing (imputation, scaling) is fitted inside each training fold via Pipeline.
- --tune adds a RandomizedSearchCV-tuned HistGBM (search uses training vehicles only).

Usage:
    python train_model.py
    python train_model.py --tune --model-dir models_plan
    python train_model.py --input data/cleaned_data.csv --use-current-pattern
"""
from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import polars as pl
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "consumption_kwh_per_100km"
GROUP = "vehicle_id"
LEAKY = {"energy_used_kwh", "battery_level_end_pct"}
NUMERIC = ["distance_km", "speed_kmh", "outside_temperature_c", "battery_temperature_c",
           "battery_level_start_pct", "tyre_pressure_bar", "slope_grade_pct"]
CATEGORIES = {  # fixed schema vocabulary; a missing value becomes all zeros
    "traffic_situation": ["low", "medium", "high"],
    "preconditioning": ["off", "on"],
    "climate_settings": ["off", "low", "medium", "high"],
    "road_condition": ["dry", "wet", "snow_or_ice"],
    "towing": ["no", "yes"],
    "historical_driving_pattern": ["efficient", "typical", "energy_intensive"],
    "current_driving_pattern": ["steady", "mixed", "stop_start"],
}
SEARCH_SPACE = {
    "m__learning_rate": [0.03, 0.05, 0.1],
    "m__max_iter": [200, 400, 800],
    "m__max_depth": [None, 4, 6, 8],
    "m__min_samples_leaf": [20, 50, 100],
    "m__l2_regularization": [0, 1, 10],
    "m__loss": ["squared_error", "absolute_error"],
}


def build_features(df, use_current_pattern):
    cats = {k: v for k, v in CATEGORIES.items() if use_current_pattern or k != "current_driving_pattern"}
    cols, names = [df.select(NUMERIC).to_numpy().astype(float)], list(NUMERIC)
    for c, levels in cats.items():
        vals = df[c].to_numpy()
        for lv in levels:
            cols.append((vals == lv).astype(float)[:, None])
            names.append(f"{c}={lv}")
    X = np.hstack(cols)
    assert not (set(names) & LEAKY), "leaky column in features"
    return X, names


def make_models(seed):
    imp = lambda: SimpleImputer(strategy="median", add_indicator=True)  # noqa: E731
    return {
        "median_baseline": Pipeline([("imp", imp()), ("m", DummyRegressor(strategy="median"))]),
        "ridge": Pipeline([("imp", imp()), ("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]),
        "hist_gbm": Pipeline([("imp", imp()), ("m", HistGradientBoostingRegressor(random_state=seed))]),
    }


def metrics(y, p):
    return {"mae": float(mean_absolute_error(y, p)), "rmse": float(np.sqrt(mean_squared_error(y, p)))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=Path("data/cleaned_data.csv"))
    ap.add_argument("--model-dir", type=Path, default=Path("models"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--use-current-pattern", action="store_true",
                    help="Include current_driving_pattern (only if known at prediction time)")
    ap.add_argument("--tune", action="store_true", help="Add a tuned HistGBM (GroupKFold random search)")
    ap.add_argument("--n-iter", type=int, default=30, help="Random search candidates for --tune")
    ap.add_argument("--final", action="store_true",
                    help="After evaluation, refit the selected model on ALL vehicles and save that one")
    args = ap.parse_args()

    df = pl.read_csv(args.input, infer_schema_length=None)
    if df["observation_id"].n_unique() != df.height:
        raise ValueError("Duplicate observation_id found; run clean_data.py first")
    X, names = build_features(df, args.use_current_pattern)
    y = df[TARGET].to_numpy().astype(float)
    g = df[GROUP].to_numpy()

    tr, te = next(GroupShuffleSplit(1, test_size=args.test_size, random_state=args.seed).split(X, y, g))
    assert not set(g[tr]) & set(g[te])
    print(f"rows {len(y):,} | features {len(names)} | train vehicles {len(set(g[tr]))} "
          f"| test vehicles {len(set(g[te]))} ({len(te):,} rows)")

    models = make_models(args.seed)

    if args.tune:
        rs = RandomizedSearchCV(models["hist_gbm"], SEARCH_SPACE, n_iter=args.n_iter,
                                scoring="neg_mean_absolute_error", cv=GroupKFold(args.folds),
                                random_state=args.seed, n_jobs=-1)
        rs.fit(X[tr], y[tr], groups=g[tr])
        print(f"\nTuning (training vehicles only): best params {rs.best_params_}")
        models["hist_gbm_tuned"] = clone(rs.best_estimator_)  # unfitted copy, refit below

    cv = {}
    for name, model in models.items():
        res = []
        for a, b in GroupKFold(args.folds).split(X[tr], y[tr], g[tr]):
            model.fit(X[tr][a], y[tr][a])
            res.append(metrics(y[tr][b], model.predict(X[tr][b])))
        cv[name] = {k: (float(np.mean([r[k] for r in res])), float(np.std([r[k] for r in res])))
                    for k in ("mae", "rmse")}
    print("\nGroupKFold CV on training vehicles (mean +- std over folds):")
    for n, r in cv.items():
        print(f"  {n:16s} MAE {r['mae'][0]:.3f} +- {r['mae'][1]:.3f} | RMSE {r['rmse'][0]:.3f} +- {r['rmse'][1]:.3f}")
    if args.tune:
        print("  (hist_gbm_tuned CV is slightly optimistic: params were picked on these same folds;"
              " the held-out result below is the fair comparison)")

    best = min((n for n in cv if n != "median_baseline"), key=lambda n: cv[n]["mae"][0])
    print(f"\nSelected by CV MAE: {best}")

    test = {}
    for name, model in models.items():
        model.fit(X[tr], y[tr])
        test[name] = metrics(y[te], model.predict(X[te]))
    print("Held-out vehicles (single evaluation, no tuning on it):")
    for n, r in test.items():
        print(f"  {n:16s} MAE {r['mae']:.3f} | RMSE {r['rmse']:.3f}")

    pred = models[best].predict(X[te])
    d = df[te]
    groups = {
        "snow_or_ice": (d["road_condition"] == "snow_or_ice").fill_null(False).to_numpy(),
        "towing": (d["towing"] == "yes").fill_null(False).to_numpy(),
        "cold (<0 C)": (d["outside_temperature_c"] < 0).fill_null(False).to_numpy(),
        "steep downhill (<-1.5%)": (d["slope_grade_pct"] < -1.5).fill_null(False).to_numpy(),
    }
    sub = {}
    print(f"\nSubgroup MAE on held-out vehicles ({best}):")
    for k, m in groups.items():
        if m.sum():
            sub[k] = {"n": int(m.sum()), "mae": float(mean_absolute_error(y[te][m], pred[m]))}
            print(f"  {k:24s} n={m.sum():4d} MAE {sub[k]['mae']:.3f}")

    args.model_dir.mkdir(parents=True, exist_ok=True)
    final_model = models[best]
    if args.final:
        # Metrics above stay the honest estimate (40/10 split); the saved model sees all vehicles.
        final_model = clone(models[best]).fit(X, y)
        print(f"\n--final: refitted '{best}' on all {len(set(g))} vehicles ({len(y):,} rows). "
              f"Expected error is the held-out MAE above; it cannot be re-measured for this model.")
    joblib.dump({"model": final_model, "features": names, "target": TARGET}, args.model_dir / "model.joblib")
    (args.model_dir / "metrics.json").write_text(json.dumps(
        {"selected": best, "cv": cv, "test": test, "subgroups": sub, "seed": args.seed,
         "use_current_pattern": args.use_current_pattern, "tuned": args.tune,
         "final_fit_on_all_vehicles": args.final}, indent=2))
    print(f"\nSaved: {args.model_dir / 'model.joblib'}, {args.model_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()