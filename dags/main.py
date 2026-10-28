# File: main.py
from __future__ import annotations

import pendulum
from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sdk import Param

from src.model_development import (
    validate_data,
    data_preprocessing,
    train_model,
    select_best_model,
    quality_check,
    publish_model,
)

# ---------- Default args ----------
default_args = {
    "start_date": pendulum.datetime(2024, 1, 1, tz="UTC"),
    "retries": 0,
}

# ---------- DAG ----------
dag = DAG(
    dag_id="Airflow_Lab2",
    default_args=default_args,
    description="Check the ad-click data, train two models and publish the better one if it is accurate enough",
    schedule="@daily",
    catchup=False,
    tags=["example"],
    owner_links={"Shaikh Basim": "https://github.com/Basim592003"},
    max_active_runs=1,
    # Shown as a form when you trigger the DAG from the UI
    params={
        "min_accuracy": Param(
            0.90,
            type="number",
            minimum=0,
            maximum=1,
            description="The best model must reach this test accuracy to be published",
        ),
    },
)

# ---------- Tasks ----------
owner_task = BashOperator(
    task_id="task_using_linked_owner",
    bash_command="echo 1",
    owner="Shaikh Basim",
    dag=dag,
)

# Start the status page first, so it can show this run while it is in progress.
# (It used to run last with ALL_DONE, which made every run count as a success.)
trigger_dag_task = TriggerDagRunOperator(
    task_id="my_trigger_task",
    trigger_dag_id="Airflow_Lab2_Flask",
    conf={"message": "Data from upstream DAG"},
    reset_dag_run=False,
    wait_for_completion=False,          # don't block
    dag=dag,
)

validate_data_task = PythonOperator(
    task_id="validate_data_task",
    python_callable=validate_data,
    op_kwargs={"run_id": "{{ run_id }}"},
    dag=dag,
)

data_preprocessing_task = PythonOperator(
    task_id="data_preprocessing_task",
    python_callable=data_preprocessing,
    op_args=[validate_data_task.output],
    dag=dag,
)

# The two models train at the same time, each in its own task
train_logistic_regression_task = PythonOperator(
    task_id="train_logistic_regression_task",
    python_callable=train_model,
    op_args=[data_preprocessing_task.output, "logistic_regression"],
    dag=dag,
)

train_random_forest_task = PythonOperator(
    task_id="train_random_forest_task",
    python_callable=train_model,
    op_args=[data_preprocessing_task.output, "random_forest"],
    dag=dag,
)

select_best_model_task = PythonOperator(
    task_id="select_best_model_task",
    python_callable=select_best_model,
    op_args=[
        data_preprocessing_task.output,
        train_logistic_regression_task.output,
        train_random_forest_task.output,
    ],
    dag=dag,
)

quality_check_task = PythonOperator(
    task_id="quality_check_task",
    python_callable=quality_check,
    op_args=[data_preprocessing_task.output, "{{ params.min_accuracy }}"],
    dag=dag,
)

publish_model_task = PythonOperator(
    task_id="publish_model_task",
    python_callable=publish_model,
    op_args=[data_preprocessing_task.output],
    dag=dag,
)

# ---------- Dependencies ----------
owner_task >> [trigger_dag_task, validate_data_task]
validate_data_task >> data_preprocessing_task >> \
    [train_logistic_regression_task, train_random_forest_task] >> \
    select_best_model_task >> quality_check_task >> publish_model_task
