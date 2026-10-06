# Battery Energy Consumption — Model Evaluator

A standardized evaluation dashboard for the team's battery-energy-consumption
models, built around one core idea:

```
Dataset --> [Model A, Model B, Model C, ...] --> Evaluator --> Dashboard
```

The dashboard doesn't care how a model was trained. `evaluate.py` only ever
receives a fitted model, test features, and true targets, and produces
predictions, regression metrics, and diagnostic plots from them. Anyone on
the team can hand this evaluator their model without touching the dashboard.

## Project structure

```
.
├── app.py               # Streamlit UI — wires data/train/evaluate together
├── data.py               # synthetic vehicle/route/weather dataset (stand-in for real data)
├── train.py               # candidate model training — decoupled from evaluation
├── evaluate.py             # THE evaluation layer: metrics + diagnostic plots
├── requirements.txt
└── README.md
```

This mirrors the Week 2 architecture directly: `data.py` is the dataset,
`train.py` produces Model A / B / C, and `evaluate.py` is the standalone
evaluator that the dashboard (`app.py`) is just a thin interface onto.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## What's implemented (Week 2 checklist)

- ✅ **Regression, not classification** — target is a continuous
  `energy_consumption_kwh_per_100km`, driven by vehicle, route, and weather
  features
- ✅ **MAE / RMSE / R² / MAPE** — computed in `evaluate.compute_metrics`,
  with MAPE automatically flagged `N/A` rather than faked when the target
  can approach zero
- ✅ **Predicted-vs-actual plot** with a `y = x` reference line
- ✅ **Residual plot** (residual vs. predicted) and an error-distribution
  histogram, to catch systematic under/over-prediction at particular
  consumption levels
- ✅ **Model comparison** — select any subset of candidate regressors in the
  sidebar; the comparison table and bar chart are generated live from actual
  evaluation runs, not hard-coded
- ✅ **Training/evaluation separation** — `train.py` and `evaluate.py` are
  independent modules; `evaluate.py` has no knowledge of how a model was
  fit, so a teammate's model (trained anywhere, any framework) plugs in the
  same way a candidate from `train.py` does
- ✅ **Live prediction**, reframed around vehicle/route/weather sliders
  producing a predicted kWh/100km figure
- ✅ **Production model hand-off** — `production_model.joblib` concept
  carried over and updated for regression (see below)

## Swapping in the team's real model

Once a teammate has a trained model, save a bundle next to `app.py`:

```python
import joblib

joblib.dump({
    "model": fitted_model,              # anything with .predict(X)
    "feature_names": [...],              # must match your X columns, in order
    "scaler": fitted_scaler,             # optional — omit if not used
    "target_name": "energy_consumption_kwh_per_100km",
    "target_unit": "kWh/100km",
}, "production_model.joblib")
```

Restart the app. It detects the file automatically, shows a "production
model detected" banner, and switches straight to evaluating it — synthetic
training is skipped. Upload a CSV of held-out features + true targets in the
app to populate the diagnostic plots and metrics for that model.

## Using a real dataset instead of synthetic data

In the sidebar, choose **"Upload my own CSV"**. Pick the continuous target
column from the dropdown; every downstream tab (exploration, diagnostics,
comparison, live prediction) recomputes against your data automatically.

## Notes on the synthetic dataset

`data.py` generates vehicle (mass, drag coefficient, frontal area, rolling
resistance, battery capacity), route (distance, speed, elevation gain, stop
frequency), and weather (temperature, wind, precipitation, humidity)
features, and derives energy consumption from a physically-motivated formula
(aerodynamic drag scales with speed², cold weather increases consumption,
etc.) plus noise — so the evaluator has real, learnable signal rather than
pure randomness. This is only a stand-in: swap in the team's real dataset or
model at any time without changing `evaluate.py`.

## Deploying to Streamlit Community Cloud

1. Push this folder to a GitHub repo.
2. Go to https://share.streamlit.io, sign in with GitHub.
3. **New app** → select the repo/branch → set **Main file path** to `app.py`.
4. Deploy. No secrets or external services are required.