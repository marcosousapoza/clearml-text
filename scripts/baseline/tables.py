import sys
from contextlib import redirect_stdout
from typing import Any, cast

import pandas as pd
from relbench.base import TaskType

from .core import (
    LEAKY_BASELINES,
    configure_baseline_environment,
    default_dataset_names,
    evaluate_dataset,
)


DATASET_LABELS = {
    "bpi2017": "BPI2017",
    "bpi2019": "BPI2019",
    "container_logistics": "Container logistics",
    "order_management": "Order management",
}

REGRESSION_BASELINES = ("global_zero", "global_mean", "global_median")
CLASSIFICATION_BASELINES = ("random", "majority")


def latex_escape(value: str) -> str:
    return value.replace("_", r"\_")


def format_metric(value: float, *, decimals: int) -> str:
    if value != value:
        return "--"
    if decimals == 0:
        return f"{value:,.0f}"
    return f"{value:.{decimals}f}"


def maybe_bold(value: float, best: float, text: str) -> str:
    if value == value and abs(value - best) <= 1e-12:
        return rf"\textbf{{{text}}}"
    return text


def task_type_name(task_result: dict[str, Any]) -> str:
    return task_result["task_type"].removeprefix("TaskType.")


def task_metric(task_result: dict[str, Any], baseline_name: str, metric: str) -> float:
    return float(task_result["baselines"][baseline_name]["test"][metric])


def build_regression_table(results: list[dict[str, Any]]) -> str:
    rows: list[dict[str, str]] = []
    for dataset_result in results:
        dataset_name = cast(str, dataset_result["dataset"])
        dataset_label = DATASET_LABELS.get(dataset_name, dataset_name)
        tasks = cast(dict[str, dict[str, Any]], dataset_result["tasks"])
        for task_name, task_result in tasks.items():
            if task_type_name(task_result) != TaskType.REGRESSION.name:
                continue
            values = {
                name: task_metric(task_result, name, "mae")
                for name in REGRESSION_BASELINES
                if name not in LEAKY_BASELINES
            }
            best = min(values.values())
            metric_cols = [
                maybe_bold(value, best, format_metric(value, decimals=0))
                for value in values.values()
            ]
            rows.append(
                {
                    "Dataset": dataset_label,
                    "Task": rf"\texttt{{{latex_escape(task_name)}}}",
                    "Global zero": metric_cols[0],
                    "Global mean": metric_cols[1],
                    "Global median": metric_cols[2],
                }
            )

    return cast(str, pd.DataFrame(rows).to_latex(
        index=False,
        escape=False,
        column_format="llrrr",
    ))


def build_classification_table(results: list[dict[str, Any]]) -> str:
    rows: list[dict[str, str]] = []
    for dataset_result in results:
        dataset_name = cast(str, dataset_result["dataset"])
        dataset_label = DATASET_LABELS.get(dataset_name, dataset_name)
        tasks = cast(dict[str, dict[str, Any]], dataset_result["tasks"])
        for task_name, task_result in tasks.items():
            if task_type_name(task_result) not in {
                TaskType.BINARY_CLASSIFICATION.name,
                TaskType.MULTICLASS_CLASSIFICATION.name,
            }:
                continue
            values = {
                name: task_metric(task_result, name, "roc_auc")
                for name in CLASSIFICATION_BASELINES
                if name not in LEAKY_BASELINES
            }
            best = max(values.values())
            metric_cols = [
                maybe_bold(value, best, format_metric(value, decimals=3))
                for value in values.values()
            ]
            rows.append(
                {
                    "Dataset": dataset_label,
                    "Task": rf"\texttt{{{latex_escape(task_name)}}}",
                    "Random": metric_cols[0],
                    "Majority": metric_cols[1],
                }
            )

    return cast(str, pd.DataFrame(rows).to_latex(
        index=False,
        escape=False,
        column_format="llrr",
    ))


def generate_baseline_tables(
    *,
    dataset_names: list[str] | None = None,
    task_names: list[str] | None = None,
    seed: int = 42,
) -> tuple[str, str]:
    selected_datasets = dataset_names or default_dataset_names()
    results = [
        evaluate_dataset(
            dataset_name,
            task_names=task_names,
            seed=seed,
        )
        for dataset_name in selected_datasets
    ]
    return build_regression_table(results), build_classification_table(results)


def print_baseline_tables(
    *,
    dataset_names: list[str] | None = None,
    task_names: list[str] | None = None,
    seed: int = 42,
) -> None:
    configure_baseline_environment(seed=seed)
    with redirect_stdout(sys.stderr):
        regression_table, classification_table = generate_baseline_tables(
            dataset_names=dataset_names,
            task_names=task_names,
            seed=seed,
        )
    print(regression_table)
    print(classification_table)
