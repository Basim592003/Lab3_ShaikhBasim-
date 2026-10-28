# File: src/model_development.py
import json
import os
import pickle
import re
import shutil

import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Mounted to ./outputs on the host (see docker-compose.yaml)
OUTPUT_DIR = "/opt/airflow/outputs"
RUNS_DIR = os.path.join(OUTPUT_DIR, "runs")
PUBLISHED_DIR = os.path.join(OUTPUT_DIR, "published")

CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "advertising.csv",
)

FEATURE_COLUMNS = [
    "Daily Time Spent on Site",
    "Age",
    "Area Income",
    "Daily Internet Usage",
    "Male",
]
NUMERIC_COLUMNS = FEATURE_COLUMNS[:4]  # "Male" is already 0/1, so it isn't scaled
TARGET_COLUMN = "Clicked on Ad"
MIN_ROWS = 100

MODELS = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000),
    "random_forest": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
}


def run_dir_for(run_id: str) -> str:
    """
    Folder holding one DAG run's files. Run ids contain ':' and '+', so those are replaced.
    """
    return os.path.join(RUNS_DIR, re.sub(r"[^A-Za-z0-9._-]", "_", run_id))


def read_metrics(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "metrics.json")) as f:
        return json.load(f)


def write_metrics(run_dir: str, metrics: dict) -> None:
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def validate_data(run_id: str) -> str:
    """
    Check the CSV before anything is trained on it, then save it to this run's folder.
    Raises ValueError, failing the run, if the data is unusable.
    Returns the run folder.
    """
    df = pd.read_csv(CSV_PATH)

    missing_columns = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing_columns:
        raise ValueError(f"CSV is missing columns: {missing_columns}")

    missing_values = df[FEATURE_COLUMNS + [TARGET_COLUMN]].isna().sum()
    if missing_values.any():
        raise ValueError(f"CSV has missing values: {missing_values[missing_values > 0].to_dict()}")

    if len(df) < MIN_ROWS:
        raise ValueError(f"CSV has {len(df)} rows; at least {MIN_ROWS} are needed")

    labels = set(df[TARGET_COLUMN].unique())
    if not labels <= {0, 1}:
        raise ValueError(f"'{TARGET_COLUMN}' should only contain 0 and 1, found {sorted(labels)}")

    print(f"Data check passed: {len(df)} rows, no missing values")

    run_dir = run_dir_for(run_id)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "raw.pkl"), "wb") as f:
        pickle.dump(df, f)
    return run_dir


def data_preprocessing(run_dir: str) -> str:
    """
    Keep the model inputs and split them 70/30 into train and test sets.
    Scaling happens inside each model's pipeline (see train_model), so every column
    is scaled once and the saved model works on raw data.
    Returns the run folder.
    """
    with open(os.path.join(run_dir, "raw.pkl"), "rb") as f:
        df = pickle.load(f)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    with open(os.path.join(run_dir, "split.pkl"), "wb") as f:
        pickle.dump((X_train, X_test, y_train, y_test), f)
    return run_dir


def train_model(run_dir: str, model_name: str) -> dict:
    """
    Train one model (scaler + classifier), score it on the test set and save it.
    Returns the model name and its test accuracy.
    """
    with open(os.path.join(run_dir, "split.pkl"), "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)

    model = make_pipeline(
        make_column_transformer(
            (StandardScaler(), NUMERIC_COLUMNS),
            remainder="passthrough",
        ),
        MODELS[model_name](),
    )
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"{model_name} test accuracy: {accuracy:.4f}")

    with open(os.path.join(run_dir, f"{model_name}.pkl"), "wb") as f:
        pickle.dump(model, f)
    return {"model": model_name, "accuracy": float(accuracy)}


def select_best_model(run_dir: str, *results: dict) -> str:
    """
    Keep the model with the highest test accuracy and write the run's metrics.json.
    Returns the best model's name.
    """
    best = max(results, key=lambda r: r["accuracy"])

    write_metrics(run_dir, {
        "run": os.path.basename(run_dir),
        "accuracy": {r["model"]: round(r["accuracy"], 4) for r in results},
        "best_model": best["model"],
        "best_accuracy": round(best["accuracy"], 4),
    })
    print(f"Best model: {best['model']} ({best['accuracy']:.4f})")
    return best["model"]


def quality_check(run_dir: str, min_accuracy: str) -> None:
    """
    Fail the run if the best model is below the minimum accuracy, so it isn't published.
    min_accuracy comes from the DAG's params, rendered by Jinja as a string.
    """
    min_accuracy = float(min_accuracy)
    metrics = read_metrics(run_dir)
    metrics["min_accuracy"] = min_accuracy
    metrics["passed"] = metrics["best_accuracy"] >= min_accuracy
    write_metrics(run_dir, metrics)

    if not metrics["passed"]:
        raise ValueError(
            f"Best accuracy {metrics['best_accuracy']} is below the minimum {min_accuracy}; "
            "model not published"
        )
    print(f"Quality check passed: {metrics['best_accuracy']} >= {min_accuracy}")


def publish_model(run_dir: str) -> str:
    """
    Copy the winning model and its metrics to outputs/published, replacing the previous ones.
    Returns the published model's path.
    """
    metrics = read_metrics(run_dir)

    os.makedirs(PUBLISHED_DIR, exist_ok=True)
    model_path = os.path.join(PUBLISHED_DIR, "model.pkl")
    shutil.copy(os.path.join(run_dir, f"{metrics['best_model']}.pkl"), model_path)
    shutil.copy(os.path.join(run_dir, "metrics.json"), os.path.join(PUBLISHED_DIR, "metrics.json"))

    print(f"Published {metrics['best_model']} to {model_path}")
    return model_path
