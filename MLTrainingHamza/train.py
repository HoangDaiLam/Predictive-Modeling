import os
import shutil
import warnings
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from dataset import FEATURES
from energy_features import engineer_energy_features

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
mlflow.set_tracking_uri(Path("mlruns").resolve().as_uri())

TARGET = "energy_consumed_kwh"
data = pd.read_csv("data/incoming/route_telemetry.csv")
X = data[FEATURES]
y = data[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")