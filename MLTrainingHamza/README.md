# EV energy consumption: gradient boosting, XGBoost and LightGBM with MLflow

A module that tunes, tests and compares three gradient-boosted tree
regressors on the trip-level EV dataset (`data/cleaned_data.csv`, 9,501 rows,
50 vehicles), tracks everything in MLflow, derives the underlying mathematics,
and serves the MLflow UI in a Prussian blue and gold theme.

Target: `consumption_kwh_per_100km`.

## Quick start

```bash
pip install -e ".[dev]"
pytest                 # 47 tests, about 2 minutes
evmlops train          # about 5 minutes on one CPU core
evmlops ui             # Open ze link at http://127.0.0.1:5000
```

`evmlops train --help` lists overrides (search budget, folds, seed, paths).
`evmlops ui --plain` serves the stock MLflow theme.

## What it does

1. **Loads and validates** the CSV, then splits **by vehicle** so the held-out
   test set contains vehicles never seen in training.
2. **Engineers row-wise features** inside the model pipeline (hour of day as
   sine and cosine, month, weekday, battery-minus-outside temperature, absolute
   slope), so the logged model accepts raw trip columns.
3. **Tunes each candidate** with random search over `GroupKFold` cross-validation
   (12 trials by default), logging every trial as a nested MLflow run.
4. **Selects the winner on cross-validation RMSE only**, then reports every
   model once on the held-out vehicles.
5. **Quantifies uncertainty** with a vehicle-level cluster bootstrap, including
   paired RMSE differences between models.
6. **Verifies the derivations**: single-tree fits are compared with the
   closed-form formulas in `docs/mathematics.md` and the outcome is logged.
7. **Registers the winner** in the MLflow model registry with a `champion` alias.

### Run layout in MLflow

```
comparison-<timestamp>            data, config, derivation checks, comparison tables and plots
  Gradient Boosting (scikit-learn)   best params, CV/train/test metrics, importances, model
    trial-00 ... trial-11            one search configuration each
  XGBoost
  LightGBM
```

## Results (default settings, seed 42)

Held-out test set: 10 vehicles, 1,826 trips. Mean-predictor baseline RMSE: 9.17.

| Model | CV RMSE | Test RMSE (95% interval) | Test MAE | Test R2 |
|---|---|---|---|---|
| XGBoost | 3.163 | 3.283 (2.74 to 3.75) | 2.342 | 0.870 |
| Gradient Boosting | 3.174 | 3.250 (2.77 to 3.66) | 2.343 | 0.873 |
| LightGBM | 3.239 | 3.387 (2.86 to 3.83) | 2.431 | 0.862 |

How to read this:

* All three cut baseline RMSE by about 63 to 65 percent.
* XGBoost is selected because it has the lowest CV RMSE. Its margin over
  scikit-learn gradient boosting (0.011) is far below the fold-to-fold standard
  deviation (about 0.22).
* **XGBoost and scikit-learn gradient boosting are statistically
  indistinguishable here**: the paired test RMSE difference is -0.03 with a 95
  percent interval of (-0.13, 0.06).
* LightGBM is modestly but consistently behind both (differences of about 0.11
  to 0.13 RMSE, intervals excluding zero). Its best search trial used 125
  leaves, near the top of the search range, so a larger search budget or a
  tighter `num_leaves` range may close some of the gap. This was not explored.
* With only 10 test vehicles the intervals are wide. Treat rankings as
  provisional and rerun with other seeds (`--seed`) before relying on them.

Snapshots of the tables and figures from this run are in `reports/`.

## Design decisions

* **Leakage removed.** `energy_used_kwh` reconstructs the target exactly
  (`100 * energy / distance`), and `battery_level_end_pct` is only known after
  the trip. Both are excluded; a test enforces it.
* **Grouped splits.** Rows from one vehicle are correlated (tyre pressure acts
  almost like a vehicle identifier), so random splits would flatter flexible
  models. Change `group_column` in the config to group by `trip_id` instead.
* **Native handling per library.** XGBoost and LightGBM receive NaNs and
  categorical dtypes directly. scikit-learn's `GradientBoostingRegressor`
  cannot, so it gets median imputation with missing indicators and one-hot
  encoding. The data has missing values in nine columns.
* **Model format.** Pipelines are saved with cloudpickle because they contain a
  custom transformer. Load models only from stores you trust, and keep the
  `evmlops` package installed in any environment that loads them.

## Mathematics

`docs/mathematics.md` derives, with the same notation throughout:
functional gradient descent and Friedman's split criterion; XGBoost's
second-order regularised objective, optimal leaf weights and split gain;
LightGBM's leaf-wise growth, histogram subtraction, GOSS and Fisher's
categorical result; the mapping of every symbol to a hyperparameter; and the
evaluation statistics including the cluster bootstrap. The document is logged
to each comparison run. Five of its results are checked numerically against the
libraries (`verification.py`, `tests/test_verification.py`).

Two conventions confirmed by those checks: for squared error all three
libraries use gradient `F - y` and Hessian 1, and both XGBoost and LightGBM
report split gain as the bracketed term without the factor one half.
LightGBM's GOSS is derived but not enabled; the candidate uses standard
bagging.

## The Prussian blue and gold UI

MLflow has no supported theming hook, so `evmlops.ui` does two things:

1. `palette.py` copies MLflow's compiled UI once (cached under
   `~/.cache/evmlops/theme`, keyed by MLflow version) and remaps blues in the
   brand-blue hue band to Prussian blue (`#003153` family), in both the
   stylesheets and the JavaScript design tokens. Greys, reds, greens and chart
   colours are untouched so status colours keep their meaning.
2. `prussian-gold.css` adds gold accents: active-tab underline, focus rings,
   primary-button base line, table header rule, text selection and scrollbars.

It is registered as an official MLflow app, so you can also run
`mlflow server --app-name prussian-gold ...`. To tune the look, edit
`src/evmlops/ui/prussian-gold.css` or the constants in `palette.py`; the cache
rebuilds automatically when either changes.

**Verification limit.** I could confirm that the themed server starts, serves
the recoloured bundles and stylesheet, and passes the REST API through, and the
palette logic is unit-tested. I could not render the UI in a browser in my
environment, so the final appearance has not been seen. If any component looks
off, the gold rules use attribute selectors on MLflow's class names and are
easy to adjust. Pinned to MLflow 3.x; other major versions may need tweaks.

## Layout

```
src/evmlops/
  config.py        experiment configuration (dataclass, YAML overrides)
  data.py          schema, leakage policy, feature builder, grouped split
  models.py        candidates, preprocessing, search spaces
  metrics.py       point metrics and cluster bootstrap
  diagnostics.py   staged error curves, native and permutation importance
  verification.py  numerical checks of the derived formulas
  plots.py         figures in the project palette
  pipeline.py      MLflow experiment orchestration
  cli.py           `evmlops train` and `evmlops ui`
  ui/              palette remapper, gold stylesheet, app factory
docs/mathematics.md
tests/             unit and end-to-end tests
reports/           snapshot of results from the default run
```
