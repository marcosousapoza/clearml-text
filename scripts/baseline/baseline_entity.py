import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.seed import seed_everything

from relbench.base import EntityTask, Table, TaskType
from relbench.datasets import get_dataset
from relbench.tasks import get_task, get_task_names

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import data.dataset  # noqa: F401
import task  # noqa: F401
from data.dataset._utils import DEFAULT_CACHE_ROOT, configure_cache_environment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run simple entity-task baselines for one or more registered RelBench tasks."
        )
    )
    parser.add_argument("--dataset", type=str, default="order_management")
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Task name to evaluate. Repeat to select multiple tasks.",
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Evaluate all registered entity tasks for the dataset.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Use RelBench download mode instead of local cache generation.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
        help="Shared cache root. Defaults to $HOME/scratch/relbench.",
    )
    return parser


def configure_environment(cache_root: Path | None) -> None:
    configure_cache_environment(cache_root)


def resolve_task_names(args: argparse.Namespace, dataset_name: str) -> list[str]:
    registered_tasks = list(get_task_names(dataset_name))
    if args.all_tasks:
        return registered_tasks
    if not args.task:
        return registered_tasks

    unknown = sorted(set(args.task) - set(registered_tasks))
    if unknown:
        raise ValueError(
            f"Unknown task(s) for dataset {dataset_name!r}: {', '.join(unknown)}"
        )
    return args.task


def combine_train_val_table(train_table: Table, val_table: Table) -> Table:
    return Table(
        df=pd.concat([train_table.df, val_table.df], axis=0, ignore_index=True),
        fkey_col_to_pkey_table=train_table.fkey_col_to_pkey_table,
        pkey_col=train_table.pkey_col,
        time_col=train_table.time_col,
    )


def fkey_col(table: Table) -> str | None:
    keys = list(table.fkey_col_to_pkey_table.keys())
    return keys[0] if keys else None


def scalar_majority(values: pd.Series) -> Any:
    mode_values = values.mode(dropna=True)
    if not mode_values.empty:
        return mode_values.iloc[0]
    return values.iloc[0]


def predict_baseline(task: EntityTask, train_table: Table, pred_table: Table, name: str) -> np.ndarray:
    target = train_table.df[task.target_col]
    key = fkey_col(train_table)

    if name == "global_zero":
        return np.zeros(len(pred_table.df), dtype=float)

    if name == "global_mean":
        return np.full(len(pred_table.df), float(target.astype(float).mean()))

    if name == "global_median":
        return np.full(len(pred_table.df), float(np.median(target.astype(float).to_numpy())))

    if name == "entity_mean":
        if key is None:
            return np.zeros(len(pred_table.df), dtype=float)
        grouped = (
            train_table.df.groupby(key, observed=True)[task.target_col]
            .mean()
            .rename("__target__")
            .reset_index()
        )
        merged = pred_table.df.merge(grouped, how="left", on=key)
        fallback = float(target.astype(float).mean())
        return merged["__target__"].fillna(fallback).astype(float).to_numpy()

    if name == "entity_median":
        if key is None:
            return np.zeros(len(pred_table.df), dtype=float)
        grouped = (
            train_table.df.groupby(key, observed=True)[task.target_col]
            .median()
            .rename("__target__")
            .reset_index()
        )
        merged = pred_table.df.merge(grouped, how="left", on=key)
        fallback = float(np.median(target.astype(float).to_numpy()))
        return merged["__target__"].fillna(fallback).astype(float).to_numpy()

    if name == "random":
        if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            pred = np.random.rand(len(pred_table.df), task.num_classes)
            row_sums = pred.sum(axis=1, keepdims=True)
            return pred / row_sums
        return np.random.rand(len(pred_table.df))

    if name == "majority":
        majority_value = scalar_majority(target)
        if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            pred = np.zeros((len(pred_table.df), task.num_classes), dtype=float)
            pred[:, int(majority_value)] = 1.0
            return pred
        if isinstance(majority_value, (bool, np.bool_)):
            fill_value = float(bool(majority_value))
        else:
            fill_value = float(majority_value)
        return np.full(len(pred_table.df), fill_value)

    if name == "entity_majority_multiclass":
        if key is None:
            return predict_baseline(task, train_table, pred_table, "majority")
        grouped = (
            train_table.df.groupby(key, observed=True)[task.target_col]
            .agg(scalar_majority)
            .rename("__target__")
            .reset_index()
        )
        merged = pred_table.df.merge(grouped, how="left", on=key)
        default_label = int(scalar_majority(target))
        labels = merged["__target__"].fillna(default_label).astype(int).to_numpy()
        pred = np.zeros((len(pred_table.df), task.num_classes), dtype=float)
        pred[np.arange(len(pred_table.df)), labels] = 1.0
        return pred

    raise ValueError(f"Unknown baseline {name!r}.")


def evaluate_split(
    task: EntityTask,
    train_table: Table,
    pred_table: Table,
    baseline_name: str,
) -> dict[str, float]:
    pred = predict_baseline(task, train_table, pred_table, baseline_name)
    eval_table = None if task.target_col not in pred_table.df.columns else pred_table
    return task.evaluate(pred, eval_table)


def baseline_names(task: EntityTask) -> list[str]:
    if task.task_type == TaskType.REGRESSION:
        return [
            "global_zero",
            "global_mean",
            "global_median",
            "entity_mean",
            "entity_median",
        ]
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        return ["random", "majority", "entity_mean", "entity_median"]
    if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        return ["random", "majority", "entity_majority_multiclass"]
    raise ValueError(
        f"Task {task.__class__.__name__} has unsupported type {task.task_type} for this baseline runner."
    )


def evaluate_task(dataset_name: str, task_name: str, download: bool) -> dict[str, Any]:
    task = get_task(dataset_name, task_name, download=download)
    if not isinstance(task, EntityTask):
        raise TypeError(f"Task {task_name!r} is not an EntityTask.")

    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test")
    trainval_table = combine_train_val_table(train_table, val_table)

    results: dict[str, Any] = {
        "task_type": str(task.task_type),
        "train_rows": len(train_table.df),
        "val_rows": len(val_table.df),
        "test_rows": len(test_table.df),
        "baselines": {},
    }

    for name in baseline_names(task):
        results["baselines"][name] = {
            "train": evaluate_split(task, train_table, train_table, name),
            "val": evaluate_split(task, train_table, val_table, name),
            "test": evaluate_split(task, trainval_table, test_table, name),
        }

    return results


def main() -> None:
    args = build_parser().parse_args()
    configure_environment(args.cache_root)
    seed_everything(args.seed)
    np.random.seed(args.seed)

    dataset_name = args.dataset
    task_names = resolve_task_names(args, dataset_name)
    dataset = get_dataset(dataset_name, download=args.download)
    dataset.get_db()

    summary: dict[str, Any] = {
        "dataset": dataset_name,
        "seed": args.seed,
        "download": args.download,
        "tasks": {},
    }

    for task_name in task_names:
        print(f"Running baseline for {dataset_name}/{task_name}...")
        summary["tasks"][task_name] = evaluate_task(
            dataset_name=dataset_name,
            task_name=task_name,
            download=args.download,
        )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
