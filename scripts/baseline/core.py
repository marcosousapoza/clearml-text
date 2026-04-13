import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn.metrics as skm
from torch_geometric.seed import seed_everything

from relbench.base import Table, TaskType
from relbench.datasets import get_dataset
from relbench.tasks import get_task, get_task_names

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.cache import configure_cache_environment
from data.dataset import register_all_datasets
from task import TASK_SPECS, register_tasks
from task.utils.custom import MEntityTask


LEAKY_BASELINES = {
    "entity_mean",
    "entity_median",
    "entity_majority_multiclass",
}


def default_dataset_names() -> list[str]:
    return sorted({dataset_name for dataset_name, _task_name, _task_cls in TASK_SPECS})


def configure_baseline_environment(
    *,
    seed: int = 42,
) -> Path:
    resolved_cache_root = configure_cache_environment()
    register_all_datasets(resolved_cache_root)
    register_tasks(resolved_cache_root)
    seed_everything(seed)
    np.random.seed(seed)
    return resolved_cache_root


def resolve_task_names(dataset_name: str, task_names: list[str], all_tasks: bool = False) -> list[str]:
    registered_tasks = list(get_task_names(dataset_name))
    if all_tasks or not task_names:
        return registered_tasks

    unknown = sorted(set(task_names) - set(registered_tasks))
    if unknown:
        raise ValueError(
            f"Unknown task(s) for dataset {dataset_name!r}: {', '.join(unknown)}"
        )
    return task_names


def combine_train_val_table(train_table: Table, val_table: Table) -> Table:
    return Table(
        df=pd.concat([train_table.df, val_table.df], axis=0, ignore_index=True),
        fkey_col_to_pkey_table=train_table.fkey_col_to_pkey_table,
        pkey_col=train_table.pkey_col,
        time_col=train_table.time_col,
    )


def scalar_majority(values: pd.Series) -> Any:
    mode_values = values.mode(dropna=True)
    if not mode_values.empty:
        return mode_values.iloc[0]
    return values.iloc[0]


def multiclass_width(task: MEntityTask, train_table: Table, pred_table: Table) -> int:
    width = int(getattr(task, "num_classes", 0) or 0)
    for table in (train_table, pred_table):
        if task.target_col not in table.df.columns or table.df.empty:
            continue
        max_label = table.df[task.target_col].max()
        if pd.notna(max_label):
            width = max(width, int(max_label) + 1)
    return width


def predict_baseline(
    task: MEntityTask,
    train_table: Table,
    pred_table: Table,
    name: str,
) -> np.ndarray:
    target = train_table.df[task.target_col]
    keys = list(train_table.fkey_col_to_pkey_table.keys())
    key = keys[0] if keys else None

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
            pred = np.random.rand(len(pred_table.df), multiclass_width(task, train_table, pred_table))
            row_sums = pred.sum(axis=1, keepdims=True)
            return pred / row_sums
        return np.random.rand(len(pred_table.df))

    if name == "majority":
        majority_value = scalar_majority(target)
        if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            pred = np.zeros((len(pred_table.df), multiclass_width(task, train_table, pred_table)), dtype=float)
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
        pred = np.zeros((len(pred_table.df), multiclass_width(task, train_table, pred_table)), dtype=float)
        pred[np.arange(len(pred_table.df)), labels] = 1.0
        return pred

    raise ValueError(f"Unknown baseline {name!r}.")


def classification_roc_auc(target: np.ndarray, pred: np.ndarray) -> float:
    if pred.ndim == 1 or pred.shape[1] == 1:
        return float(skm.roc_auc_score(target, pred))

    target = target.astype(int)
    max_label = int(target.max()) if len(target) else -1
    if max_label >= pred.shape[1]:
        pred = np.pad(pred, ((0, 0), (0, max_label + 1 - pred.shape[1])))
    labels = np.array([label for label in np.unique(target) if 0 <= label < pred.shape[1]])
    if labels.size < 2:
        return float("nan")
    indexed_labels = {label: idx for idx, label in enumerate(labels)}
    remapped_target = np.array([indexed_labels[label] for label in target], dtype=int)
    observed_pred = pred[:, labels]
    row_sums = observed_pred.sum(axis=1, keepdims=True)
    observed_pred = np.divide(
        observed_pred,
        row_sums,
        out=np.full_like(observed_pred, 1.0 / observed_pred.shape[1]),
        where=row_sums != 0,
    )
    return float(
        skm.roc_auc_score(
            remapped_target,
            observed_pred,
            average="macro",
            multi_class="ovr",
        )
    )


def evaluate_split(
    task: MEntityTask,
    train_table: Table,
    pred_table: Table,
    baseline_name: str,
) -> dict[str, float]:
    pred = predict_baseline(task, train_table, pred_table, baseline_name)
    eval_table = None if task.target_col not in pred_table.df.columns else pred_table
    scores = task.evaluate(pred, eval_table)
    if task.task_type in {
        TaskType.BINARY_CLASSIFICATION,
        TaskType.MULTICLASS_CLASSIFICATION,
    }:
        target_table = task.get_table("test", mask_input_cols=False) if eval_table is None else eval_table
        scores["roc_auc"] = classification_roc_auc(
            target_table.df[task.target_col].to_numpy(),
            pred,
        )
    return scores


def baseline_names(task: MEntityTask) -> list[str]:
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


def evaluate_task(
    dataset_name: str,
    task_name: str,
) -> dict[str, Any]:
    task = get_task(dataset_name, task_name, download=False)
    if not isinstance(task, MEntityTask):
        raise TypeError(f"Task {task_name!r} is not an MEntityTask.")

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
            "train": evaluate_split(
                task,
                train_table,
                train_table,
                name,
            ),
            "val": evaluate_split(
                task,
                train_table,
                val_table,
                name,
            ),
            "test": evaluate_split(
                task,
                trainval_table,
                test_table,
                name,
            ),
        }

    return results


def evaluate_dataset(
    dataset_name: str,
    task_names: list[str] | None = None,
    *,
    all_tasks: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    resolved_task_names = resolve_task_names(dataset_name, task_names or [], all_tasks=all_tasks)
    dataset = get_dataset(dataset_name, download=False)
    dataset.get_db()

    summary: dict[str, Any] = {
        "dataset": dataset_name,
        "seed": seed,
        "tasks": {},
    }

    for task_name in resolved_task_names:
        print(f"Running baseline for {dataset_name}/{task_name}...", file=sys.stderr)
        summary["tasks"][task_name] = evaluate_task(
            dataset_name=dataset_name,
            task_name=task_name,
        )
    return summary
