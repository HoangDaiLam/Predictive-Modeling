# -*- coding: utf-8 -*-
"""
train_ml.py — Train Machine Learning model to predict
energy consumption (kWh/100km) from trip data.

Workflow:
    1. Load cleaned_data.csv
    2. Drop leaky columns (energy_used_kwh, battery_level_end_pct, target)
    3. Encode categorical + normalize numeric
    4. Split by vehicle_id (80/10/10) — NO random split (time-series safety)
    5. Train baseline (Linear Regression) + main model (XGBoost)
    6. Evaluate: MAE, RMSE, R2, P90 error
    7. Feature importance plot
    8. Save model to disk

Usage:
    python train_ml.py
    python train_ml.py --input data/cleaned_data.csv --output-dir reports/ml
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] xgboost not installed. Falling back to RandomForest.")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib


TARGET = "consumption_kwh_per_100km"
LEAKY = {"energy_used_kwh", "battery_level_end_pct", TARGET}
ID_COLS = {"observation_id", "vehicle_id", "trip_id", "timestamp"}

NUMERIC_FEATURES = [
    "distance_km", "speed_kmh", "outside_temperature_c",
    "battery_temperature_c", "battery_level_start_pct",
    "tyre_pressure_bar", "slope_grade_pct",
]
CATEGORICAL_FEATURES = [
    "traffic_situation", "preconditioning", "climate_settings",
    "road_condition", "topography", "towing",
    "current_driving_pattern", "historical_driving_pattern",
]


def load_data(path: Path) -> pd.DataFrame:
    print(f"Loading {path} ...")
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df


def split_by_vehicle(df: pd.DataFrame, train_frac=0.7, val_frac=0.15, seed=42):
    """Split by vehicle_id — prevents data leakage."""
    vehicles = df["vehicle_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(vehicles)

    n = len(vehicles)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_v = set(vehicles[:n_train])
    val_v = set(vehicles[n_train:n_train + n_val])
    test_v = set(vehicles[n_train + n_val:])

    train = df[df["vehicle_id"].isin(train_v)].copy()
    val = df[df["vehicle_id"].isin(val_v)].copy()
    test = df[df["vehicle_id"].isin(test_v)].copy()

    print(f"\nSplit by vehicle_id ({n} vehicles):")
    print(f"  Train: {len(train):,} rows ({len(train_v)} vehicles)")
    print(f"  Val:   {len(val):,} rows ({len(val_v)} vehicles)")
    print(f"  Test:  {len(test):,} rows ({len(test_v)} vehicles)")
    return train, val, test


def build_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def evaluate(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    p90 = np.percentile(np.abs(y_true - y_pred), 90)
    print(f"\n{label}:")
    print(f"  MAE:  {mae:7.4f} kWh/100km")
    print(f"  RMSE: {rmse:7.4f} kWh/100km")
    print(f"  R²:   {r2:7.4f}")
    print(f"  P90 error: {p90:7.4f} kWh/100km")
    return {"mae": mae, "rmse": rmse, "r2": r2, "p90": p90}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("data/cleaned_data.csv"),
                        help="Path to cleaned CSV (default: data/cleaned_data.csv)")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/ml"),
                        help="Where to save model + reports (default: reports/ml)")
    parser.add_argument("--no-eda", action="store_true", help="Skip feature importance plot")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Could not find {args.input}. Run clean_data.py first."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load
    df = load_data(args.input)

    # 2. Validate columns
    missing = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # 3. Drop rows with missing target
    before = len(df)
    df = df.dropna(subset=[TARGET])
    print(f"  Dropped {before - len(df)} rows with missing target")

    # 4. Split by vehicle_id
    train, val, test = split_by_vehicle(df)

    X_train = train[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train[TARGET].values
    X_val = val[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_val = val[TARGET].values
    X_test = test[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test[TARGET].values

    # 5. Preprocess
    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)
    print(f"\nFeature matrix: {X_train_t.shape[1]} columns after encoding")

    # 6. Train baseline (mean predictor)
    baseline_pred = np.full(len(y_test), y_train.mean())
    baseline_metrics = evaluate(y_test, baseline_pred, "BASELINE (predict mean)")

    # 7. Train Linear Regression
    print("\n--- Training Linear Regression ---")
    lr = LinearRegression()
    lr.fit(X_train_t, y_train)
    lr_pred = lr.predict(X_test_t)
    lr_metrics = evaluate(y_test, lr_pred, "LINEAR REGRESSION")

    # 8. Train main model (XGBoost or RandomForest)
    if HAS_XGB:
        print("\n--- Training XGBoost ---")
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=20,
        )
        model.fit(
            X_train_t, y_train,
            eval_set=[(X_val_t, y_val)],
            verbose=False,
        )
    else:
        print("\n--- Training Random Forest ---")
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train_t, y_train)

    model_pred = model.predict(X_test_t)
    model_metrics = evaluate(y_test, model_pred, "MAIN MODEL")

    # 9. Improvement over baseline
    print("\nImprovement over baseline:")
    print(f"  MAE:  {(baseline_metrics['mae'] - model_metrics['mae']) / baseline_metrics['mae'] * 100:.1f}%")
    print(f"  R²:   {baseline_metrics['r2']:.3f} → {model_metrics['r2']:.3f}")

    # 10. Save model + preprocessor
    model_path = args.output_dir / "model.joblib"
    joblib.dump({
        "preprocessor": preprocessor,
        "model": model,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
    }, model_path)
    print(f"\nModel saved: {model_path}")

    # 11. Save metrics
    report = {
        "baseline": baseline_metrics,
        "linear_regression": lr_metrics,
        "main_model": model_metrics,
        "main_model_type": "XGBoost" if HAS_XGB else "RandomForest",
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "n_features": int(X_train_t.shape[1]),
    }
    report_path = args.output_dir / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Metrics saved: {report_path}")

    # 12. Feature importance plot
    if not args.no_eda:
        try:
            plot_feature_importance(model, preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURES, args.output_dir)
        except Exception as e:
            print(f"[WARN] Feature importance plot failed: {e}")

        plot_predictions(y_test, model_pred, args.output_dir)


def plot_feature_importance(model, preprocessor, numeric, categorical, output_dir):
    """Plot top-20 most important features."""
    num_names = numeric
    cat_names = preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(categorical)
    all_names = np.concatenate([num_names, cat_names])

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = np.abs(model.coef_)

    idx = np.argsort(importances)[-20:]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(range(len(idx)), importances[idx])
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([all_names[i] for i in idx], fontsize=9)
    ax.set_xlabel("Importance")
    ax.set_title("Top-20 feature importance")
    fig.tight_layout()
    fig.savefig(output_dir / "01_feature_importance.png", dpi=150)
    plt.close(fig)
    print(f"  Feature importance saved: {output_dir / '01_feature_importance.png'}")


def plot_predictions(y_true, y_pred, output_dir):
    """Predicted vs actual scatter + residual histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Plot 1: Predicted vs Actual ---
    axes[0].scatter(y_true, y_pred, alpha=0.3, s=10, color="tab:blue")
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    axes[0].plot([lo, hi], [lo, hi], "r--", label="Perfect prediction (y=x)")
    axes[0].set_xlabel("ACTUAL Consumption (kWh/100km)")
    axes[0].set_ylabel("PREDICTED Consumption (kWh/100km)")
    axes[0].set_title("Predicted vs Actual\n(Each dot represents one test trip)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Plot 2: Error Histogram ---
    residuals = y_true - y_pred
    axes[1].hist(residuals, bins=50, edgecolor="black", color="tab:orange", alpha=0.8)
    
    # Add vertical line at 0 (Perfect prediction)
    axes[1].axvline(x=0, color="red", linestyle="--", linewidth=1.5, label="Zero error")
    # Add vertical line for mean error
    axes[1].axvline(x=residuals.mean(), color="blue", linestyle="-", linewidth=1.5, label=f"Mean error = {residuals.mean():.2f}")

    axes[1].set_xlabel("Prediction Error (Actual - Predicted) [kWh/100km]")
    # CHANGED: "Count" to "Number of Test Trips"
    axes[1].set_ylabel("Number of Test Trips")
    axes[1].set_title("Distribution of Prediction Errors\n(Most errors center around 0, indicating good performance)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "02_predictions.png", dpi=150)
    plt.close(fig)
    print(f"  Prediction plot saved: {output_dir / '02_predictions.png'}")


if __name__ == "__main__":
    main()