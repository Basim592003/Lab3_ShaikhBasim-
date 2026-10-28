from __future__ import annotations

import os
import time
import base64
import socket
import pendulum
import logging
import requests
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from flask import Flask, redirect, render_template

from src.model_development import read_metrics, run_dir_for


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set the lowest threshold for the logge

# ---------- Config ----------
# Ensure WEBSERVER points to 'airflow-apiserver' as defined in your compose
WEBSERVER = os.getenv("AIRFLOW_WEBSERVER", "http://airflow-apiserver:8080")
AF_USER   = os.getenv("AIRFLOW_USERNAME", "airflow")
AF_PASS   = os.getenv("AIRFLOW_PASSWORD", "airflow")
TARGET_DAG_ID = os.getenv("TARGET_DAG_ID", "Airflow_Lab2")

# Set up robust paths for Docker Workers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

# ---------- Flask app ----------
app = Flask(__name__, template_folder=TEMPLATE_DIR)

# Cached token to avoid requesting a new token on every HTTP request
_CACHED_TOKEN = None
_TOKEN_EXPIRY = 0

def get_auth_token():
    global _CACHED_TOKEN, _TOKEN_EXPIRY
    # Reuse valid token if available
    if _CACHED_TOKEN and time.time() < _TOKEN_EXPIRY - 60:
        return _CACHED_TOKEN

    token_url = f"{WEBSERVER}/auth/token"
    resp = requests.post(
        token_url,
        json={"username": AF_USER, "password": AF_PASS},
        timeout=10,
    )
    if resp.status_code in (200, 201):
        data = resp.json()
        _CACHED_TOKEN = data.get("access_token")
        # Token valid for 24h by default; refresh after 1 hour safely
        _TOKEN_EXPIRY = time.time() + 3600
        return _CACHED_TOKEN
    
    logger.error(f"Failed to authenticate with Airflow: {resp.status_code} {resp.text}")
    return None

def get_run_metrics(run_id):
    """
    Model scores for a run, from its metrics.json. Empty if the run stopped before scoring.
    """
    try:
        return read_metrics(run_dir_for(run_id))
    except (OSError, ValueError):
        return {}

def get_latest_run_info():
    url = f"{WEBSERVER}/api/v2/dags/{TARGET_DAG_ID}/dagRuns?order_by=-run_after&limit=1"

    info = {
        "state": "unknown",
        "run_id": "N/A",
        "logical_date": "N/A",
        "start_date": "N/A",
        "end_date": "N/A",
        "note": "",
        "metrics": {},
    }

    token = get_auth_token()
    if not token:
        info["note"] = "Authentication failed (could not get token)"
        return False, info

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
    }

    try:
        r = requests.get(url, headers=headers, timeout=10)
        logger.info(f"response was : {r}")
        if r.status_code != 200:
            info["note"] = f"API Error {r.status_code}"
            return False, info

        data = r.json()
        runs = data.get("dag_runs", [])
        
        if not runs:
            info["note"] = "Waiting for DAG run to appear..."
            return False, info

        run = runs[0]
        state = run.get("state", "unknown")

        logger.info(f"state was {state}")

        info.update({
            "state": state,
            "run_id": run.get("dag_run_id", "N/A"),
            "logical_date": run.get("logical_date") or run.get("run_after", "N/A"),
            "start_date": run.get("start_date", "N/A"),
            "end_date": run.get("end_date", "N/A"),
        })
        info["metrics"] = get_run_metrics(info["run_id"])

        if state == "queued" or state == "running":
            info["note"] = "Model pipeline is still in progress..."
            return False, info

        if state != "success":
            metrics = info["metrics"]
            if metrics.get("passed") is False:
                info["note"] = (
                    f"Best accuracy {metrics['best_accuracy']} ({metrics['best_model']}) is below "
                    f"the minimum {metrics['min_accuracy']}, so the model was not published"
                )
            else:
                info["note"] = "A task failed; check the task logs in Airflow"

        return state == "success", info

    except Exception as e:
        info["note"] = f"Network issue: {str(e)}"
        return False, info


@app.route("/")
def index():
    ok, _ = get_latest_run_info()
    return redirect("/success" if ok else "/failure")

# Each page sends you to the other one if it doesn't match the latest run,
# so a bookmarked or refreshed page never shows an old result.
@app.route("/success")
def success():
    ok, info = get_latest_run_info()
    if not ok:
        return redirect("/failure")
    return render_template("success.html", **info)

@app.route("/failure")
def failure():
    ok, info = get_latest_run_info()
    if ok:
        return redirect("/success")
    return render_template("failure.html", **info)

@app.route("/health")
def health():
    return "ok", 200

def start_flask_app():
    """
    Starts the Flask server, unless an earlier run already started it. The running
    server always shows the latest run, so a second one isn't needed (and couldn't
    bind the port anyway).
    Note: Blocks the worker slot. Disable reloader for Docker.
    """
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", 5555)) == 0:
            print("Status page is already running on port 5555; nothing to start.", flush=True)
            return

    print(f"Starting Flask on 0.0.0.0:5555 checking {WEBSERVER}...", flush=True)
    app.run(host="0.0.0.0", port=5555, debug=False, use_reloader=False)

# ---------- DAG Definition ----------
default_args = {
    "start_date": pendulum.datetime(2024, 1, 1, tz="UTC"),
    "retries": 0,
}

dag = DAG(
    dag_id="Airflow_Lab2_Flask",
    default_args=default_args,
    schedule=None,
    catchup=False,
    is_paused_upon_creation=False,
)

PythonOperator(
    task_id="start_Flask_API",
    python_callable=start_flask_app,
    dag=dag,
)