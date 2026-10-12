"""
app.py — Battery Energy Consumption Model Evaluator

The dashboard is the *interface* to a standardized evaluation system, not
the system itself — the real logic lives in data.py / train.py / evaluate.py
so any teammate's model can be dropped in without touching this file's core
evaluation path.

SWAPPING IN THE TEAM'S REAL MODEL
----------------------------------
Save a joblib bundle named `production_model.joblib` next to this script:

    import joblib
    joblib.dump({
        "model": fitted_model,            # anything with .predict(X)
        "feature_names": [...],            # must match X_test columns, in order
        "scaler": fitted_scaler,           # optional, omit if not used
        "target_name": "energy_consumption_kwh_per_100km",
        "target_unit": "kWh/100km",
    }, "production_model.joblib")

The app detects it automatically and evaluates it directly — synthetic
training is skipped entirely. You can also upload a CSV of X_test + y_test
to evaluate that model against real held-out data instead of the synthetic
split.
"""
# Programming Logic Imports
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split

# Modules imports
from data import generate_synthetic_data, FEATURE_GROUPS, FEATURE_NAMES, TARGET_COL
from train import MODEL_REGISTRY, train_candidates
from evaluate import (
    evaluate, compare_models, plot_predicted_vs_actual,
    plot_residuals, plot_error_distribution, plot_feature_importance,
)

st.set_page_config(page_title="Battery Energy Model Evaluator -|--|--", layout="wide")

PRODUCTION_MODEL_PATH = os.path.join(os.path.dirname(__file__), "production_model.joblib")


@st.cache_data(show_spinner=False)
def _generate_data(n_samples, seed):
    return generate_synthetic_data(n_samples=n_samples, seed=seed)


@st.cache_resource(show_spinner=False)
def _load_production_model():
    if os.path.exists(PRODUCTION_MODEL_PATH):
        try:
            return joblib.load(PRODUCTION_MODEL_PATH)
        except Exception:
            return None
    return None


@st.cache_resource(show_spinner=False)
def _train_candidates(model_configs, X_train, y_train):
    return train_candidates(model_configs, X_train, y_train)


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
st.sidebar.title("⚙️ Configuration")
prod_bundle = _load_production_model()

if prod_bundle is not None:
    st.sidebar.success("DING DING DING Production model detected — evaluating your real model.")
    mode = "production"
else:
    st.sidebar.info(
        "No `production_model.joblib` yet. Evaluating candidate regressors trained "
        "live on synthetic vehicle/route/weather data, standing in for the team's "
        "real dataset."
    )
    mode = st.sidebar.radio("Data source", ["Synthetic demo data", "Upload my own CSV"])

target_col = TARGET_COL
target_unit = "kWh/100km"

if mode == "Synthetic demo data":
    n_samples = st.sidebar.slider("Number of synthetic samples", 500, 5000, 2000, step=250)
    df = _generate_data(n_samples, seed=42)
    feature_names = FEATURE_NAMES
elif mode == "Upload my own CSV":
    uploaded = st.sidebar.file_uploader("Upload CSV (features + a continuous target column)", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        cols = list(df.columns)
        target_col = st.sidebar.selectbox("Target column", cols, index=len(cols) - 1)
        target_unit = st.sidebar.text_input("Target unit (for axis labels)", value="units")
        feature_names = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]
    else:
        st.sidebar.warning("No file uploaded yet — showing synthetic data meanwhile.")
        df = _generate_data(2000, seed=42)
        feature_names = FEATURE_NAMES
else:
    df, feature_names = None, None

if mode != "production":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Candidate models to compare")
    available_models = list(MODEL_REGISTRY.keys())
    chosen_models = st.sidebar.multiselect(
        "Select 2+ models", available_models, default=["Random Forest", "Gradient Boosting", "Linear Regression"]
    )
    test_size = st.sidebar.slider("Test set size", 0.1, 0.4, 0.2, step=0.05)
    focus_model = st.sidebar.selectbox(
        "Model to inspect in detail", chosen_models if chosen_models else available_models
    )

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("Polestar 4 Battery Consumption — ML Model Evaluator")
st.caption(
    "Standardized evaluation layer for the team's battery-energy models. "
    + ("Currently evaluating your production model." if mode == "production"
       else "Currently training and comparing candidate regressors on demo data — "
            "swap in the team's real model any time (see app.py docstring).")
)

tab_overview, tab_eda, tab_perf, tab_compare, tab_predict, tab_about = st.tabs(
    ["🗂 Data", "🔍 Exploration", "📈 Diagnostics", "⚖️ Model Comparison", "🎯 Live Prediction", "ℹ️ About"]
)

# ----------------------------------------------------------------------
# PRODUCTION MODEL PATH
# ----------------------------------------------------------------------
if mode == "production":
    model = prod_bundle["model"]
    feature_names = prod_bundle["feature_names"]
    scaler = prod_bundle.get("scaler")
    target_unit = prod_bundle.get("target_unit", "units")

    with tab_overview:
        st.write("Running your saved production model bundle.")
        st.json({"features": feature_names, "target_unit": target_unit})
        st.info("Upload a CSV with these feature columns plus the true target to evaluate held-out performance.")

    eval_file = st.file_uploader(
        "Upload evaluation CSV (feature columns + true target) to see diagnostics", type=["csv"], key="prod_eval"
    )

    if eval_file is not None:
        eval_df = pd.read_csv(eval_file)
        eval_target_col = st.selectbox("True target column", list(eval_df.columns), index=len(eval_df.columns) - 1)
        X_eval = eval_df[feature_names]
        y_eval = eval_df[eval_target_col]
        y_pred, metrics, residuals = evaluate(model, X_eval, y_eval, scaler)

        with tab_perf:
            cols = st.columns(4)
            for c, (name, val) in zip(cols, metrics.items()):
                c.metric(name, f"{val:.3f}" if val is not None else "N/A")
            st.plotly_chart(plot_predicted_vs_actual(y_eval, y_pred, target_unit), use_container_width=True)
            st.plotly_chart(plot_residuals(y_pred, residuals, target_unit), use_container_width=True)
            st.plotly_chart(plot_error_distribution(residuals, target_unit), use_container_width=True)
            imp_fig = plot_feature_importance(model, feature_names)
            if imp_fig:
                st.plotly_chart(imp_fig, use_container_width=True)
    else:
        with tab_perf:
            st.info("Upload evaluation data above to see MAE / RMSE / R² / MAPE and diagnostic plots.")

    with tab_predict:
        st.subheader("Live prediction")
        inputs = {}
        cols = st.columns(2)
        for i, feat in enumerate(feature_names):
            with cols[i % 2]:
                inputs[feat] = st.number_input(feat, value=0.0)
        if st.button("Predict", type="primary"):
            from evaluate import predict as _predict
            X_input = pd.DataFrame([inputs])[feature_names]
            pred = _predict(model, X_input, scaler)[0]
            st.success(f"Predicted: **{pred:.2f} {target_unit}**")

    with tab_eda, tab_compare, tab_about:
        st.info("These tabs are most useful in demo mode with synthetic/candidate models — "
                "upload evaluation data above to populate diagnostics for your production model.")

    st.stop()

# ----------------------------------------------------------------------
# DEMO / CANDIDATE-MODEL PATH
# ----------------------------------------------------------------------
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Features", f"{len(feature_names)}")
    c3.metric(f"Mean {target_col}", f"{df[target_col].mean():.2f} {target_unit}")
    c4.metric("Missing values", f"{df.isna().sum().sum()}")
    st.markdown("#### Sample of the data")
    st.dataframe(df.head(20), use_container_width=True)
    st.markdown("#### Summary statistics")
    st.dataframe(df.describe().T, use_container_width=True)

with tab_eda:
    left, right = st.columns(2)
    with left:
        st.markdown("#### Target distribution")
        st.plotly_chart(px.histogram(df, x=target_col, nbins=40), use_container_width=True)
    with right:
        st.markdown("#### Feature vs. target")
        feat_choice = st.selectbox("Feature", feature_names)
        st.plotly_chart(px.scatter(df, x=feat_choice, y=target_col, opacity=0.5, trendline="ols"),
                         use_container_width=True)
    st.markdown("#### Correlation heatmap")
    numeric_df = df[feature_names + [target_col]]
    fig = px.imshow(numeric_df.corr(), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    st.plotly_chart(fig, use_container_width=True)

if not chosen_models or len(chosen_models) < 1:
    st.warning("Select at least one model in the sidebar to see diagnostics and comparisons.")
    st.stop()

X = df[feature_names]
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

model_configs = {name: (name, {}) for name in chosen_models}
trained = _train_candidates(model_configs, X_train, y_train)

evaluated = {}
for name, info in trained.items():
    y_pred, metrics, residuals = evaluate(info["model"], X_test, y_test, info["scaler"])
    evaluated[name] = {
        "model": info["model"], "scaler": info["scaler"], "y_pred": y_pred,
        "metrics": metrics, "residuals": residuals, "train_time": info["train_time"],
    }

# ----------------------------------------------------------------------
# Diagnostics tab (focus model)
# ----------------------------------------------------------------------
with tab_perf:
    st.markdown(f"#### {focus_model} — trained in {evaluated[focus_model]['train_time']:.2f}s "
                f"on {len(X_train):,} rows, tested on {len(X_test):,} rows")

    m = evaluated[focus_model]["metrics"]
    cols = st.columns(4)
    cols[0].metric("MAE", f"{m['MAE']:.3f} {target_unit}")
    cols[1].metric("RMSE", f"{m['RMSE']:.3f} {target_unit}")
    cols[2].metric("R²", f"{m['R2']:.3f}")
    cols[3].metric("MAPE", f"{m['MAPE']:.1f}%" if m["MAPE"] is not None else "N/A (target near zero)")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            plot_predicted_vs_actual(y_test, evaluated[focus_model]["y_pred"], target_unit),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            plot_residuals(evaluated[focus_model]["y_pred"], evaluated[focus_model]["residuals"], target_unit),
            use_container_width=True,
        )

    st.plotly_chart(plot_error_distribution(evaluated[focus_model]["residuals"], target_unit), use_container_width=True)

    imp_fig = plot_feature_importance(evaluated[focus_model]["model"], feature_names)
    if imp_fig is not None:
        st.plotly_chart(imp_fig, use_container_width=True)

    st.caption(
        "Read the residual plot for systematic bias: a cloud centered on zero with no slope means "
        "errors don't depend on the prediction level. A visible trend (e.g. increasingly negative "
        "residuals at high predictions) means the model systematically under- or over-predicts in "
        "that range — exactly the kind of issue that matters for a battery-range estimate."
    )

# ----------------------------------------------------------------------
# Model comparison tab
# ----------------------------------------------------------------------
with tab_compare:
    st.markdown("#### Comparison across selected candidate models")
    comparison_input = {name: {"metrics": info["metrics"], "train_time": info["train_time"]}
                         for name, info in evaluated.items()}
    comp_df = compare_models(comparison_input)
    st.dataframe(comp_df.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R2": "{:.3f}",
                                        "MAPE": "{:.1f}", "Train time (s)": "{:.3f}"}, na_rep="N/A"),
                 use_container_width=True)

    metric_choice = st.selectbox("Metric to chart", ["MAE", "RMSE", "R2", "MAPE"])
    chart_df = comp_df.reset_index().dropna(subset=[metric_choice])
    st.plotly_chart(
        px.bar(chart_df, x="Model", y=metric_choice, title=f"{metric_choice} by model"),
        use_container_width=True,
    )
    st.caption("This table is generated from live evaluation runs above, not hard-coded — "
               "add or remove candidate models in the sidebar to update it.")

# ----------------------------------------------------------------------
# Live prediction tab
# ----------------------------------------------------------------------
with tab_predict:
    st.markdown(f"#### Predict energy consumption with **{focus_model}**")
    inputs = {}
    for group_name, group_feats in FEATURE_GROUPS.items():
        if not all(f in feature_names for f in group_feats):
            continue
        st.markdown(f"**{group_name}**")
        cols = st.columns(len(group_feats))
        for c, (feat, (lo, hi, mid)) in zip(cols, group_feats.items()):
            with c:
                inputs[feat] = st.slider(feat, float(lo), float(hi), float(mid))

    if st.button("Predict", type="primary"):
        from evaluate import predict as _predict
        X_input = pd.DataFrame([inputs])[feature_names]
        model = evaluated[focus_model]["model"]
        scaler = evaluated[focus_model]["scaler"]
        pred = _predict(model, X_input, scaler)[0]
        st.success(f"Predicted energy consumption: **{pred:.2f} {target_unit}**")

# ----------------------------------------------------------------------
# About tab
# ----------------------------------------------------------------------
with tab_about:
    st.markdown(
        f"""
        ### About this evaluator

        This dashboard is the **interface to a standardized evaluation system**, built
        around the separation:

        ```
        Dataset --> [Model A, Model B, Model C, ...] --> Evaluator --> Dashboard
        ```

        `evaluate.py` never trains anything — it only ever consumes a fitted model plus
        test features and true targets, so a model trained by anyone on the team, in any
        framework, plugs in the same way as the candidates trained in this demo.

        **Current status:** no production model yet, so this page trains and compares
        candidate regressors ({', '.join(MODEL_REGISTRY.keys())}) on synthetic
        vehicle/route/weather data shaped like the real project inputs.

        **To plug in the team's real model:** save a `production_model.joblib` bundle
        next to `app.py` (see the docstring at the top of the file). The dashboard
        detects it automatically and switches to evaluating it directly.

        **Metrics used:**
        - **MAE** — average absolute prediction error, in {target_unit}
        - **RMSE** — like MAE but penalizes large errors more heavily
        - **R²** — share of variance in the target explained by the model
        - **MAPE** — average percentage error (flagged as N/A if the target can be ~0,
          where percentage error is unstable)
        """
    )