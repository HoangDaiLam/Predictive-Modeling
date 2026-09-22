"""
Predictive Modelling Gang: Demo Dashboard
==========================================
A deployable Streamlit dashboard for demoing a classification model.

No production model exists yet, so this app trains real candidate models
on the fly (Logistic Regression / Random Forest / Gradient Boosting) using
either a built-in synthetic "customer churn" dataset or a CSV the user
uploads. This keeps every number on screen (accuracy, ROC-AUC, confusion
matrix, feature importance, live predictions) genuine and reproducible,
rather than hard-coded placeholders.

SWAPPING IN A REAL MODEL LATER
-------------------------------
Once you have a trained production model, drop a joblib file named
`production_model.joblib` next to this script (see `load_production_model`
below) containing a dict: {"model": <fitted sklearn estimator>,
"feature_names": [...], "target_name": "...", "class_names": [...]}.
The app will automatically detect it, skip synthetic-data training, and
demo your real model instead -- no other code changes required.
"""

import os
import io
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.datasets import make_classification

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Predictive Modelling Gang: Demo Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRODUCTION_MODEL_PATH = os.path.join(os.path.dirname(__file__), "production_model.joblib")

# ----------------------------------------------------------------------
# Data generation / loading
# ----------------------------------------------------------------------
FEATURE_NAMES = [
    "tenure_months", "monthly_charges", "total_charges", "num_support_tickets",
    "contract_length_score", "num_products", "satisfaction_score",
    "avg_session_minutes", "late_payments", "discount_pct",
]

@st.cache_data(show_spinner=False)
def generate_synthetic_data(n_samples=2000, seed=42):
    """Generate a synthetic, but realistically-shaped, churn dataset."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=len(FEATURE_NAMES),
        n_informative=7,
        n_redundant=1,
        n_clusters_per_class=2,
        weights=[0.75, 0.25],  # imbalanced, like real churn data
        flip_y=0.03,
        class_sep=1.1,
        random_state=seed,
    )
    df = pd.DataFrame(X, columns=FEATURE_NAMES)

    # Rescale columns into human-plausible ranges so the dashboard reads naturally
    df["tenure_months"] = np.clip((df["tenure_months"] * 8 + 24), 0, 72).round(0)
    df["monthly_charges"] = np.clip((df["monthly_charges"] * 20 + 65), 15, 150).round(2)
    df["total_charges"] = (df["tenure_months"] * df["monthly_charges"] * np.random.uniform(0.9, 1.0, n_samples)).round(2)
    df["num_support_tickets"] = np.clip((df["num_support_tickets"] * 1.5 + 2), 0, 15).round(0)
    df["contract_length_score"] = np.clip((df["contract_length_score"] + 2) / 4, 0, 1).round(2)
    df["num_products"] = np.clip((df["num_products"] * 1.2 + 2.5), 1, 6).round(0)
    df["satisfaction_score"] = np.clip((df["satisfaction_score"] + 2) * 2.5, 0, 10).round(1)
    df["avg_session_minutes"] = np.clip((df["avg_session_minutes"] * 5 + 20), 1, 60).round(1)
    df["late_payments"] = np.clip((df["late_payments"] + 2) * 1.5, 0, 10).round(0)
    df["discount_pct"] = np.clip((df["discount_pct"] + 2) * 5, 0, 40).round(1)

    df["churned"] = y
    return df


def load_uploaded_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        return df, None
    except Exception as e:
        return None, str(e)


@st.cache_resource(show_spinner=False)
def load_production_model():
    """Load a real trained model if one has been dropped in, else None."""
    if os.path.exists(PRODUCTION_MODEL_PATH):
        try:
            bundle = joblib.load(PRODUCTION_MODEL_PATH)
            return bundle
        except Exception:
            return None
    return None


# ----------------------------------------------------------------------
# Model training
# ----------------------------------------------------------------------
MODEL_REGISTRY = {
    "Logistic Regression": lambda p: LogisticRegression(
        C=p["C"], max_iter=1000, random_state=42
    ),
    "Random Forest": lambda p: RandomForestClassifier(
        n_estimators=p["n_estimators"], max_depth=p["max_depth"],
        min_samples_leaf=p["min_samples_leaf"], random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": lambda p: GradientBoostingClassifier(
        n_estimators=p["n_estimators"], max_depth=p["gb_max_depth"],
        learning_rate=p["learning_rate"], random_state=42
    ),
}


@st.cache_resource(show_spinner=False)
def train_model(model_name, params, X_train, y_train, use_scaling):
    scaler = StandardScaler() if use_scaling else None
    X_train_proc = scaler.fit_transform(X_train) if scaler is not None else X_train.values

    model = MODEL_REGISTRY[model_name](params)
    t0 = time.time()
    model.fit(X_train_proc, y_train)
    train_time = time.time() - t0
    return model, scaler, train_time


def evaluate_model(model, scaler, X_test, y_test):
    X_proc = scaler.transform(X_test) if scaler is not None else X_test.values
    y_pred = model.predict(X_proc)
    y_proba = model.predict_proba(X_proc)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["ROC-AUC"] = roc_auc_score(y_test, y_proba)

    cm = confusion_matrix(y_test, y_pred)
    return metrics, cm, y_pred, y_proba


def get_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    if hasattr(model, "coef_"):
        return pd.Series(np.abs(model.coef_[0]), index=feature_names).sort_values(ascending=False)
    return None


# ----------------------------------------------------------------------
# Sidebar — data & model configuration
# ----------------------------------------------------------------------
st.sidebar.title("⚙️ Configuration")

prod_bundle = load_production_model()

if prod_bundle is not None:
    st.sidebar.success("✅ Production model detected — demoing your real model.")
    data_mode = "production"
else:
    st.sidebar.info(
        "No `production_model.joblib` found yet, so this dashboard trains "
        "real candidate models live on synthetic (or your uploaded) data. "
        "Swap in your trained model any time — see the docstring in app.py."
    )
    data_mode = st.sidebar.radio(
        "Data source",
        ["Synthetic demo data", "Upload my own CSV"],
        help="Synthetic data lets you demo the full dashboard immediately.",
    )

target_col = "churned"

if data_mode == "Synthetic demo data":
    n_samples = st.sidebar.slider("Number of synthetic samples", 500, 5000, 2000, step=250)
    df = generate_synthetic_data(n_samples=n_samples)
elif data_mode == "Upload my own CSV":
    uploaded = st.sidebar.file_uploader("Upload CSV (must include a binary target column)", type=["csv"])
    if uploaded is not None:
        df, err = load_uploaded_data(uploaded)
        if err:
            st.sidebar.error(f"Could not read file: {err}")
            df = generate_synthetic_data()
        else:
            cols = list(df.columns)
            target_col = st.sidebar.selectbox("Target column (0/1)", cols, index=len(cols) - 1)
    else:
        st.sidebar.warning("No file uploaded yet — showing synthetic data meanwhile.")
        df = generate_synthetic_data()
else:
    df = None  # production mode uses its own data story

if data_mode != "production":
    feature_cols = [c for c in df.columns if c != target_col]

    st.sidebar.markdown("---")
    st.sidebar.subheader("Model")
    model_name = st.sidebar.selectbox("Algorithm", list(MODEL_REGISTRY.keys()))

    params = {}
    if model_name == "Logistic Regression":
        params["C"] = st.sidebar.slider("Regularization strength (C)", 0.01, 10.0, 1.0)
        use_scaling = True
    elif model_name == "Random Forest":
        params["n_estimators"] = st.sidebar.slider("Number of trees", 50, 500, 200, step=50)
        params["max_depth"] = st.sidebar.slider("Max depth", 2, 20, 8)
        params["min_samples_leaf"] = st.sidebar.slider("Min samples per leaf", 1, 20, 3)
        use_scaling = False
    else:  # Gradient Boosting
        params["n_estimators"] = st.sidebar.slider("Number of estimators", 50, 500, 150, step=50)
        params["gb_max_depth"] = st.sidebar.slider("Max depth", 1, 10, 3)
        params["learning_rate"] = st.sidebar.slider("Learning rate", 0.01, 0.5, 0.1)
        use_scaling = False

    test_size = st.sidebar.slider("Test set size", 0.1, 0.4, 0.2, step=0.05)
    st.sidebar.markdown("---")
    st.sidebar.caption("Adjust settings above, then results update automatically.")

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("Predictive Modelling: Dashboard Prototype Model")
st.caption(
    "Interactive demo environment for a churn-prediction classifier. "
    "Currently running on " + ("your production model." if data_mode == "production"
                                else "a live-trained candidate model, so every metric below is real.")
)

tab_overview, tab_eda, tab_perf, tab_predict, tab_about = st.tabs(
    ["🗂 Data Overview", "🔍 Exploration", "📈 Model Performance", "🎯 Live Prediction", "ℹ️ About"]
)

# ----------------------------------------------------------------------
# Production-model-only simplified path
# ----------------------------------------------------------------------
if data_mode == "production":
    model = prod_bundle["model"]
    feature_names = prod_bundle["feature_names"]
    class_names = prod_bundle.get("class_names", ["Negative", "Positive"])
    scaler = prod_bundle.get("scaler")

    with tab_overview:
        st.write("This dashboard is running your saved production model bundle.")
        st.json({"features": feature_names, "classes": class_names})

    with tab_predict:
        st.subheader("Try a live prediction")
        inputs = {}
        cols = st.columns(2)
        for i, feat in enumerate(feature_names):
            with cols[i % 2]:
                inputs[feat] = st.number_input(feat, value=0.0)
        if st.button("Predict", type="primary"):
            X_input = pd.DataFrame([inputs])[feature_names]
            X_proc = scaler.transform(X_input) if scaler is not None else X_input.values
            pred = model.predict(X_proc)[0]
            proba = model.predict_proba(X_proc)[0] if hasattr(model, "predict_proba") else None
            st.success(f"Predicted class: **{class_names[int(pred)]}**")
            if proba is not None:
                st.write(pd.DataFrame({"class": class_names, "probability": proba}))

    with tab_eda, tab_perf, tab_about:
        st.info("Upload evaluation data or extend this section to show held-out metrics for your production model.")

    st.stop()

# ----------------------------------------------------------------------
# Overview tab
# ----------------------------------------------------------------------
with tab_overview:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Features", f"{len(feature_cols)}")
    churn_rate = df[target_col].mean()
    col3.metric("Positive class rate", f"{churn_rate:.1%}")
    col4.metric("Missing values", f"{df.isna().sum().sum()}")

    st.markdown("#### Sample of the data")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("#### Summary statistics")
    st.dataframe(df.describe().T, use_container_width=True)

# ----------------------------------------------------------------------
# EDA tab
# ----------------------------------------------------------------------
with tab_eda:
    left, right = st.columns([1, 1])

    with left:
        st.markdown("#### Target distribution")
        target_counts = df[target_col].value_counts().rename(index={0: "No", 1: "Yes"})
        fig = px.pie(values=target_counts.values, names=target_counts.index,
                     title="Churned?", hole=0.45)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### Feature to explore")
        explore_feat = st.selectbox("Choose a feature", feature_cols)
        fig2 = px.histogram(df, x=explore_feat, color=df[target_col].map({0: "No", 1: "Yes"}),
                             barmode="overlay", nbins=30, opacity=0.7,
                             labels={"color": "Churned"})
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Correlation heatmap")
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig3 = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    st.plotly_chart(fig3, use_container_width=True)

# ----------------------------------------------------------------------
# Train / test split (shared by Performance + Prediction tabs)
# ----------------------------------------------------------------------
X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
y = df[target_col].astype(int)
numeric_feature_cols = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y if y.nunique() == 2 else None
)

with st.spinner(f"Training {model_name}..."):
    model, scaler, train_time = train_model(model_name, params, X_train, y_train, use_scaling)

metrics, cm, y_pred, y_proba = evaluate_model(model, scaler, X_test, y_test)
importance = get_feature_importance(model, numeric_feature_cols)

# ----------------------------------------------------------------------
# Performance tab
# ----------------------------------------------------------------------
with tab_perf:
    st.markdown(f"#### {model_name} — trained in {train_time:.2f}s on {len(X_train):,} rows")

    mcols = st.columns(len(metrics))
    for c, (name, val) in zip(mcols, metrics.items()):
        c.metric(name, f"{val:.3f}")

    left, right = st.columns(2)

    with left:
        st.markdown("##### Confusion Matrix")
        cm_fig = px.imshow(
            cm, text_auto=True, color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual"),
            x=["No", "Yes"], y=["No", "Yes"],
        )
        st.plotly_chart(cm_fig, use_container_width=True)

    with right:
        if y_proba is not None:
            st.markdown("##### ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_fig = go.Figure()
            roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {metrics['ROC-AUC']:.3f}"))
            roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash")))
            roc_fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(roc_fig, use_container_width=True)

    if importance is not None:
        st.markdown("##### Feature Importance")
        imp_fig = px.bar(importance.head(15).sort_values(), orientation="h",
                          labels={"index": "Feature", "value": "Importance"})
        st.plotly_chart(imp_fig, use_container_width=True)

    with st.expander("Full classification report"):
        st.text(classification_report(y_test, y_pred, zero_division=0))

# ----------------------------------------------------------------------
# Live prediction tab
# ----------------------------------------------------------------------
with tab_predict:
    st.markdown("#### Try the model on a hypothetical record")
    st.caption("Sliders default to the median of the training data.")

    input_vals = {}
    cols = st.columns(2)
    for i, feat in enumerate(numeric_feature_cols):
        col_min, col_max, col_med = float(X[feat].min()), float(X[feat].max()), float(X[feat].median())
        with cols[i % 2]:
            input_vals[feat] = st.slider(feat, col_min, col_max, col_med)

    if st.button("Predict", type="primary"):
        X_input = pd.DataFrame([input_vals])[numeric_feature_cols]
        X_proc = scaler.transform(X_input) if scaler is not None else X_input.values
        pred = model.predict(X_proc)[0]
        proba = model.predict_proba(X_proc)[0][1] if hasattr(model, "predict_proba") else None

        if pred == 1:
            st.error(f"⚠️ Predicted: **Will churn** (probability: {proba:.1%})" if proba is not None else "⚠️ Predicted: Will churn")
        else:
            st.success(f"✅ Predicted: **Will stay** (probability of churn: {proba:.1%})" if proba is not None else "✅ Predicted: Will stay")

# ----------------------------------------------------------------------
# About tab
# ----------------------------------------------------------------------
with tab_about:
    st.markdown(
        """
        ### About this dashboard

        This is a **fully functional demo environment**, not a mockup: the model shown
        is genuinely trained on the data selected in the sidebar, and every metric,
        chart and prediction reflects that real model.

        **Current status:** no production model has been supplied yet, so the app
        trains a candidate classifier (Logistic Regression / Random Forest / Gradient
        Boosting) live on synthetic churn-shaped data (or a CSV you upload).

        **To plug in your real model:** save a `production_model.joblib` file
        (a dict with `model`, `feature_names`, `class_names`, and optionally `scaler`)
        in the same folder as `app.py`. The dashboard will detect it automatically on
        next load and switch to demoing it directly, skipping synthetic training.

        **Deployment:** see `README.md` for one-click Streamlit Community Cloud
        deployment instructions.
        """
    )