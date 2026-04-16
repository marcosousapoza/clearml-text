import argparse
from pathlib import Path
from typing import Any

import pandas as pd

import scripts
scripts.load_env()
from relbench.base import TaskType

from .core import configure_baseline_environment, default_dataset_names, evaluate_dataset


CLASSIFICATION_METRICS = (
    "accuracy",
    "f1",
    "auprc",
    "roc_auc",
    "multiclass_f1",
    "multiclass_roc_auc",
)
REGRESSION_METRICS = ("mae", "mse", "rmse", "r2")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run seeded baseline evaluations and export flat CSV summaries."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset to include. Repeat to select multiple datasets. Defaults to all registered datasets.",
    )
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Task name to evaluate. Repeat to select multiple tasks. Defaults to all registered tasks.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="Seeds to evaluate. Defaults to 1 2 3 4 5.",
    )
    parser.add_argument(
        "--classification-out",
        type=Path,
        default=Path("classification.csv"),
        help="Output CSV path for classification tasks.",
    )
    parser.add_argument(
        "--regression-out",
        type=Path,
        default=Path("regression.csv"),
        help="Output CSV path for regression tasks.",
    )
    return parser


def task_type_name(task_result: dict[str, Any]) -> str:
    return str(task_result["task_type"]).removeprefix("TaskType.")


def flatten_rows(
    dataset_result: dict[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    classification_rows: list[dict[str, Any]] = []
    regression_rows: list[dict[str, Any]] = []

    dataset_name = str(dataset_result["dataset"])
    tasks = dict(dataset_result["tasks"])
    for task_name, task_result in tasks.items():
        task_type = task_type_name(task_result)
        common = {
            "dataset": dataset_name,
            "task": task_name,
            "task_type": task_type,
            "seed": seed,
            "train_rows": int(task_result["train_rows"]),
            "val_rows": int(task_result["val_rows"]),
            "test_rows": int(task_result["test_rows"]),
        }
        baselines = dict(task_result["baselines"])
        for baseline_name, baseline_result in baselines.items():
            for split_name, metrics in dict(baseline_result).items():
                row = {
                    **common,
                    "baseline": baseline_name,
                    "split": split_name,
                }
                if task_type == TaskType.REGRESSION.name:
                    for metric in REGRESSION_METRICS:
                        row[metric] = metrics.get(metric)
                    regression_rows.append(row)
                else:
                    for metric in CLASSIFICATION_METRICS:
                        row[metric] = metrics.get(metric)
                    classification_rows.append(row)

    return classification_rows, regression_rows


def export_results(
    *,
    dataset_names: list[str] | None = None,
    task_names: list[str] | None = None,
    seeds: list[int] | None = None,
    classification_out: Path = Path("classification.csv"),
    regression_out: Path = Path("regression.csv"),
) -> tuple[Path, Path]:
    selected_datasets = dataset_names or default_dataset_names()
    selected_seeds = seeds or [1, 2, 3, 4, 5]

    classification_rows: list[dict[str, Any]] = []
    regression_rows: list[dict[str, Any]] = []

    for seed in selected_seeds:
        configure_baseline_environment(seed=seed)
        for dataset_name in selected_datasets:
            dataset_result = evaluate_dataset(
                dataset_name,
                task_names=task_names,
                seed=seed,
            )
            cls_rows, reg_rows = flatten_rows(dataset_result, seed)
            classification_rows.extend(cls_rows)
            regression_rows.extend(reg_rows)

    classification_df = pd.DataFrame(classification_rows).sort_values(
        by=["dataset", "task", "seed", "baseline", "split"]
    )
    regression_df = pd.DataFrame(regression_rows).sort_values(
        by=["dataset", "task", "seed", "baseline", "split"]
    )

    classification_out.parent.mkdir(parents=True, exist_ok=True)
    regression_out.parent.mkdir(parents=True, exist_ok=True)
    classification_df.to_csv(classification_out, index=False)
    regression_df.to_csv(regression_out, index=False)
    return classification_out.resolve(), regression_out.resolve()


def main() -> None:
    args = build_parser().parse_args()
    classification_out, regression_out = export_results(
        dataset_names=args.dataset or None,
        task_names=args.task or None,
        seeds=args.seeds,
        classification_out=args.classification_out,
        regression_out=args.regression_out,
    )
    print(f"Wrote classification results to {classification_out}")
    print(f"Wrote regression results to {regression_out}")


if __name__ == "__main__":
    main()
