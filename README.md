# Airflow Lab: Ad-Click Model Pipeline with a Quality Check

An Apache Airflow 3 pipeline that checks the advertising dataset, trains two models side by side,
keeps the more accurate one, and publishes it only if it reaches a minimum accuracy. A small Flask
status page shows the latest run's result and each model's score.

Based on [Airflow_Labs/Lab_2](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Airflow_Labs/Lab_2)
from the MLOps course repository, which trains a single logistic regression and starts a Flask page
when it finishes.

## What I changed

| | Original lab | This version |
|---|---|---|
| Data check | None | `validate_data_task` checks the CSV has the expected columns, no missing values, enough rows and 0/1 labels, and stops the run if not |
| Preprocessing | Each column scaled twice (`MinMaxScaler` and `StandardScaler` on the same columns) | Each column scaled once, inside each model's pipeline, so the saved model works on raw data |
| Training | One `LogisticRegression` | Logistic regression and random forest, each trained in its own task at the same time |
| Pick the best | None | `select_best_model_task` keeps the model with the higher test accuracy |
| Minimum score | None | `quality_check_task` fails the run if the best accuracy is below `min_accuracy`, so the model isn't published. `min_accuracy` is a DAG parameter (default 0.90) you can change when triggering the run |
| Saved results | Pickles written inside the container and lost when it is removed | Every run gets its own folder under `./outputs/runs/` with both models and a `metrics.json`; passing runs also update `./outputs/published/` |
| Status page | Always said "FAILED" when the run wasn't a success, and a second run crashed because port 5555 was already in use | Shows the real state (refreshing while the run is in progress), each model's accuracy, the winner, and why a run failed. A second run reuses the server that is already running |

Smaller changes:

- **Run results are now honest.** The original started the status page last, with `TriggerRule.ALL_DONE`. Airflow decides a run's result from its last tasks (["leaf nodes"](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/dag-run.html)), so every run counted as a success even when training failed. The status page is now started first.
- Removed the commented-out email task, the pass-through `separate_data_outputs` task, and the `requirements.txt` that listed `apache-airflow` (installing it would replace Airflow inside the image). `scikit-learn` is installed through `docker-compose.yaml` instead.
- Added the official Airflow 3.3.2 `docker-compose.yaml`, with example DAGs turned off, an `./outputs` folder shared with the host, and port 5555 published for the status page.

## Pipeline

```text
task_using_linked_owner
├── my_trigger_task ............ starts Airflow_Lab2_Flask (the status page)
└── validate_data_task
    └── data_preprocessing_task
        ├── train_logistic_regression_task ──┐
        └── train_random_forest_task ────────┴── select_best_model_task
                                                  └── quality_check_task
                                                      └── publish_model_task
```

## The model

Predicts whether a website visitor clicks on an ad, using `dags/data/advertising.csv`
(1,000 visitors: 500 clicked, 500 didn't).

| Input | Range |
|---|---|
| Daily Time Spent on Site | about 33–91 minutes |
| Age | 19–61 |
| Area Income | about $14k–$79k |
| Daily Internet Usage | about 105–270 minutes |
| Male | 0 or 1 |

Both models train on 70% of the rows and are scored on the other 30% (300 visitors).

## Project structure

```text
.
├── docker-compose.yaml        # Airflow 3.3.2 (CeleryExecutor), with the changes above
├── setup.sh                   # resets the environment and writes .env and config/airflow.cfg
├── dags/
│   ├── main.py                # Airflow_Lab2: the training pipeline
│   ├── Flask_API.py           # Airflow_Lab2_Flask: the status page on port 5555
│   ├── src/model_development.py
│   ├── templates/             # success.html and failure.html for the status page
│   └── data/advertising.csv
└── outputs/                   # created at run time (not committed)
    ├── runs/<run id>/         # raw.pkl, split.pkl, both models, metrics.json
    └── published/             # model.pkl and metrics.json from the latest passing run
```

## Running it

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/) with at least 4 GB of
memory (8 GB is better).

```bash
bash setup.sh                    # resets everything; first run downloads the images
docker compose up airflow-init   # creates the database and the airflow/airflow login
docker compose up -d             # starts Airflow in the background
```

The first start takes a few minutes, because each container installs scikit-learn.

1. Open <http://localhost:8080> and log in with **airflow / airflow**.
2. Turn on **Airflow_Lab2**. It is scheduled daily, so it starts one run straight away using the
   default minimum accuracy of 0.90.
3. Open the status page at <http://localhost:5555>.
4. To see the failure path, click **Trigger** on Airflow_Lab2 and set `min_accuracy` higher than
   the best model's score (for example 0.99). The run fails at `quality_check_task`, nothing is
   published, and the status page explains why.

Stop everything with:

```bash
docker compose down
```

## Sample results

From runs on this setup (Airflow 3.3.2, scikit-learn 1.9.1):

| Run | `min_accuracy` | Logistic regression | Random forest | Result |
|---|---|---|---|---|
| Scheduled | 0.90 (default) | **0.9700** | 0.9533 | Success: logistic regression published |
| Triggered manually | 0.99 | **0.9700** | 0.9533 | Failed at `quality_check_task`; nothing published |

`outputs/runs/<run id>/metrics.json` for the passing run:

```json
{
  "run": "scheduled__2026-10-28T00_00_00_00_00",
  "accuracy": {
    "logistic_regression": 0.97,
    "random_forest": 0.9533
  },
  "best_model": "logistic_regression",
  "best_accuracy": 0.97,
  "min_accuracy": 0.9,
  "passed": true
}
```

For the failing run, the status page gives the reason: *"Best accuracy 0.97 (logistic_regression) is below
the minimum 0.99, so the model was not published"*.

## Using the published model

The published model includes its scaler, so it takes the raw input columns:

```python
import pickle
import pandas as pd

model = pickle.load(open("outputs/published/model.pkl", "rb"))
visitor = pd.DataFrame([{
    "Daily Time Spent on Site": 40.0, "Age": 50, "Area Income": 30000.0,
    "Daily Internet Usage": 120.0, "Male": 1,
}])
print(model.predict(visitor))  # [1] = predicted to click
```

Load it with the same scikit-learn version it was trained with (1.9.1, pinned in `docker-compose.yaml`).
