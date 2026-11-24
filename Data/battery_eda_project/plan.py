"""Trip battery planner for Polestar 4 Dual Motor.

Usage (from anywhere):
    python plan.py trip.json --start-pct 90
    python plan.py trip.json --start-pct 90 --model models_plan_tuned/model.joblib

trip.json = list of segments. Each segment needs (battery_level_start_pct is filled automatically):
  distance_km, speed_kmh, outside_temperature_c, battery_temperature_c, tyre_pressure_bar,
  slope_grade_pct, traffic_situation, preconditioning, climate_settings, road_condition,
  towing, historical_driving_pattern
Split long routes into segments (e.g. every 10-30 km or when slope/speed changes).
"""
import argparse
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "data"))
from train_model import build_features, NUMERIC, CATEGORIES  # noqa: E402

CATS = {k: v for k, v in CATEGORIES.items() if k != "current_driving_pattern"}
REQUIRED = [c for c in NUMERIC if c != "battery_level_start_pct"] + list(CATS)
BIAS = 0.03      # added to predicted energy: tuned model under-predicted ~2.5-3% on held-out vehicles
RESERVE = 0.15   # extra safety margin (worst held-out vehicle was off by 13.5%)


def data_stats(data_path):
    d = pl.read_csv(data_path, infer_schema_length=None)
    u = d.drop_nulls(["energy_used_kwh", "battery_level_start_pct", "battery_level_end_pct"])
    e = u["energy_used_kwh"].to_numpy().astype(float)
    drop = (u["battery_level_start_pct"] - u["battery_level_end_pct"]).to_numpy().astype(float)
    ok = (e > 0) & (drop >= 0)
    slope = float((drop[ok] * e[ok]).sum() / (e[ok] ** 2).sum())  # % per kWh, line through origin
    return 100.0 / slope, float(d["distance_km"].quantile(0.99)), float(d["distance_km"].max())


def validate(seg, i):
    miss = [c for c in REQUIRED if seg.get(c) is None]
    if miss:
        raise ValueError(f"segment {i}: missing {miss}")
    for c, levels in CATS.items():
        if seg[c] not in levels:
            raise ValueError(f"segment {i}: {c}='{seg[c]}' invalid, use one of {levels}")


def plan(segments, start_pct, model, names, usable_kwh, max_km):
    pct, total_kwh, total_km, out = start_pct, 0.0, 0.0, []
    for i, s in enumerate(segments, 1):
        validate(s, i)
        s = {**{c: float(s[c]) for c in NUMERIC if c in s and c != "battery_level_start_pct"},
             **{c: s[c] for c in CATS}, "battery_level_start_pct": float(pct)}
        if s["distance_km"] > max_km:
            print(f"WARNING segment {i}: {s['distance_km']:.0f} km is longer than training data; split it.")
        X, n = build_features(pl.DataFrame([s]), use_current_pattern=False)
        assert n == names, "feature mismatch with saved model"
        c = float(model.predict(X)[0])                    # kWh/100 km
        kwh = max(c, 0.0) * s["distance_km"] / 100 * (1 + BIAS)
        pct -= kwh / usable_kwh * 100
        total_kwh += kwh
        total_km += s["distance_km"]
        out.append((i, s["distance_km"], c, kwh, pct))
    return out, total_kwh, total_km


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trip", type=Path)
    ap.add_argument("--start-pct", type=float, required=True)
    ap.add_argument("--model", type=Path, default=ROOT / "models_plan_tuned" / "model.joblib")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "cleaned_data.csv")
    ap.add_argument("--usable-kwh", type=float, default=None, help="override usable battery capacity")
    a = ap.parse_args()

    bundle = joblib.load(a.model)
    model, names = bundle["model"], bundle["features"]
    est_kwh, p99, mx = data_stats(a.data)
    usable = a.usable_kwh or est_kwh
    segs = json.loads(a.trip.read_text())

    rows, kwh, km = plan(segs, a.start_pct, model, names, usable, mx)
    print(f"Usable battery: {usable:.1f} kWh ({'given' if a.usable_kwh else 'estimated from your data'})")
    print(f"Assumptions: +{BIAS:.0%} bias correction, {RESERVE:.0%} reserve on energy\n")
    print(f"{'seg':>3} {'km':>7} {'kWh/100km':>10} {'kWh':>7} {'battery end %':>14}")
    for i, d, c, e, p in rows:
        print(f"{i:>3} {d:7.1f} {c:10.1f} {e:7.1f} {p:14.1f}")
    cons_pct = a.start_pct - kwh * (1 + RESERVE) / usable * 100
    print(f"\nTotal {km:.0f} km, {kwh:.1f} kWh ({kwh / km * 100:.1f} kWh/100km)")
    print(f"Expected battery at arrival:     {rows[-1][4]:.0f}%")
    print(f"Conservative (with reserve):     {cons_pct:.0f}%")
    if cons_pct < 10:
        print("WARNING: conservative estimate is below 10% - plan a charging stop.")
    if any(s.get("towing") == "yes" for s in segs):
        print("NOTE: towing is the weakest case (MAE ~6 kWh/100km, few examples). Add extra margin.")


if __name__ == "__main__":
    main()