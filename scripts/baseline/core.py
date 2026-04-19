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
DEFAULT_BASELINE_SEEDS = [1, 2, 3, 4, 5]


def default_dataset_names() -> list[str]:
    return sorted({dataset_name for dataset_name, _task_name, _task_cls in TASK_SPECS})


def configure_baseline_environment(
    *,
    seed: int = 42,
) -> Path:
    resolved_cache_root = configure_cache_environment()
    register_all_datasets(resolved_cache_root)
    register_tasks(resolved_cache_root)
    set_baseline_seed(seed)
    return resolved_cache_root


def set_baseline_seed(seed: int) -> None:
    seed_everything(seed)
    np.random.seed(seed)


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


def metric_name_for_logging(task_type: TaskType, metric_name: str) -> str:
    if task_type == TaskType.MULTICLASS_CLASSIFICATION:
        return {
            "f1": "multiclass_f1",
            "roc_auc": "multiclass_roc_auc",
        }.get(metric_name, metric_name)
    if task_type == TaskType.MULTILABEL_CLASSIFICATION:
        return {
            "f1": "multilabel_f1_macro",
            "accuracy": "multilabel_accuracy",
        }.get(metric_name, metric_name)
    return metric_name


def metric_names_for_logging(task: MEntityTask) -> list[str]:
    return [metric_name_for_logging(task.task_type, fn.__name__) for fn in task.metrics]


def log_baseline_to_wandb(
    *,
    cache_root: Path,
    dataset_name: str,
    task_name: str,
    baseline_name: str,
    seed: int,
    task: MEntityTask,
    val_scores: dict[str, float],
    test_scores: dict[str, float],
    wandb_project: str,
) -> None:
    import wandb

    run_name = f"{dataset_name}_{task_name}_{baseline_name}_seed{seed}"
    run_dir = cache_root / dataset_name / task_name / "baseline" / baseline_name
    run_dir.mkdir(parents=True, exist_ok=True)

    metric_names = metric_names_for_logging(task)
    metrics = {
        **{f"val/{name}": float(val_scores[name]) for name in metric_names if name in val_scores},
        **{f"test/{name}": float(test_scores[name]) for name in metric_names if name in test_scores},
    }

    run = wandb.init(
        project=wandb_project,
        name=run_name,
        group=f"{dataset_name}_{task_name}_{baseline_name}",
        job_type="baseline",
        dir=str(run_dir),
        config={
            "dataset": dataset_name,
            "task": task_name,
            "baseline": baseline_name,
            "seed": seed,
            "task_type": str(task.task_type),
        },
        tags=["baseline", dataset_name, task_name, baseline_name],
    )
    try:
        run.log(metrics)
        run.summary.update(metrics)
    finally:
        run.finish()


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


def split_tables(task: MEntityTask) -> dict[str, tuple[Table, Table]]:
    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test")
    return {
        "train": (train_table, train_table),
        "val": (train_table, val_table),
        "test": (combine_train_val_table(train_table, val_table), test_table),
    }


def classification_roc_auc(target: np.ndarray, pred: np.ndarray) -> float:
    if pred.ndim == 1 or pred.shape[1] == 1:
        return float(skm.roc_auc_score(target, pred))

    target = target.astype(int)
    observed_labels = np.array([label for label in np.unique(target) if 0 <= label < pred.shape[1]])
    if observed_labels.size < 2:
        return float("nan")
    if observed_labels.size == 2:
        positive_label = int(observed_labels[-1])
        return float(
            skm.roc_auc_score(
                (target == positive_label).astype(int),
                pred[:, positive_label],
            )
        )

    max_label = int(target.max()) if len(target) else -1
    if max_label >= pred.shape[1]:
        pred = np.pad(pred, ((0, 0), (0, max_label + 1 - pred.shape[1])))
    labels = np.array([label for label in np.unique(target) if 0 <= label < pred.shape[1]])
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
    *,
    seed: int = 42,
    cache_root: Path | None = None,
    wandb_project: str | None = None,
) -> dict[str, Any]:
    task = get_task(dataset_name, task_name, download=False)
    if not isinstance(task, MEntityTask):
        raise TypeError(f"Task {task_name!r} is not an MEntityTask.")

    splits = split_tables(task)
    train_table, val_table = splits["train"][0], splits["val"][1]
    test_table = splits["test"][1]

    results: dict[str, Any] = {
        "task_type": str(task.task_type),
        "train_rows": len(train_table.df),
        "val_rows": len(val_table.df),
        "test_rows": len(test_table.df),
        "baselines": {},
    }

    for name in baseline_names(task):
        baseline_scores = {
            split: evaluate_split(task, split_train, split_pred, name)
            for split, (split_train, split_pred) in splits.items()
        }
        results["baselines"][name] = baseline_scores
        if wandb_project is not None:
            if cache_root is None:
                raise ValueError("cache_root is required when wandb logging is enabled.")
            log_baseline_to_wandb(
                cache_root=cache_root,
                dataset_name=dataset_name,
                task_name=task_name,
                baseline_name=name,
                seed=seed,
                task=task,
                val_scores=baseline_scores["val"],
                test_scores=baseline_scores["test"],
                wandb_project=wandb_project,
            )

    return results


def aggregate_metric_runs(score_runs: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    metric_names = sorted({metric_name for scores in score_runs for metric_name in scores})
    aggregated: dict[str, dict[str, float]] = {}
    for metric_name in metric_names:
        values = np.array(
            [float(scores[metric_name]) for scores in score_runs if metric_name in scores],
            dtype=float,
        )
        aggregated[metric_name] = {
            "mean": float(np.nanmean(values)) if values.size else float("nan"),
            "std": float(np.nanstd(values)) if values.size else float("nan"),
        }
    return aggregated


def evaluate_task_across_seeds(
    dataset_name: str,
    task_name: str,
    *,
    seeds: list[int],
    cache_root: Path | None = None,
    wandb_project: str | None = None,
) -> dict[str, Any]:
    if not seeds:
        raise ValueError("seeds must contain at least one value.")

    task_runs = []
    for seed in seeds:
        set_baseline_seed(seed)
        task_runs.append({
            "seed": seed,
            **evaluate_task(
                dataset_name=dataset_name,
                task_name=task_name,
                seed=seed,
                cache_root=cache_root,
                wandb_project=wandb_project,
            ),
        })

    first_run = task_runs[0]
    aggregated_baselines = {}
    for baseline_name in first_run["baselines"]:
        baseline_runs = [{
            "seed": int(task_run["seed"]),
            **{
                split: dict(task_run["baselines"][baseline_name][split])
                for split in ("train", "val", "test")
            },
        } for task_run in task_runs]
        aggregated_baselines[baseline_name] = {
            "runs": baseline_runs,
            "aggregate": {
                split: aggregate_metric_runs([dict(run[split]) for run in baseline_runs])
                for split in ("train", "val", "test")
            },
        }

    return {
        "task_type": first_run["task_type"],
        "train_rows": int(first_run["train_rows"]),
        "val_rows": int(first_run["val_rows"]),
        "test_rows": int(first_run["test_rows"]),
        "baselines": aggregated_baselines,
    }


def prepare_dataset_evaluation(
    dataset_name: str,
    task_names: list[str] | None = None,
    *,
    all_tasks: bool = False,
) -> tuple[list[str], Path]:
    resolved_task_names = resolve_task_names(dataset_name, task_names or [], all_tasks=all_tasks)
    dataset = get_dataset(dataset_name, download=False)
    dataset.get_db()
    return resolved_task_names, configure_cache_environment()


def evaluate_dataset(
    dataset_name: str,
    task_names: list[str] | None = None,
    *,
    all_tasks: bool = False,
    seed: int = 42,
    wandb_project: str | None = None,
) -> dict[str, Any]:
    resolved_task_names, cache_root = prepare_dataset_evaluation(
        dataset_name,
        task_names,
        all_tasks=all_tasks,
    )

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
            seed=seed,
            cache_root=cache_root,
            wandb_project=wandb_project,
        )
    return summary


def evaluate_dataset_across_seeds(
    dataset_name: str,
    task_names: list[str] | None = None,
    *,
    all_tasks: bool = False,
    seeds: list[int] | None = None,
    wandb_project: str | None = None,
) -> dict[str, Any]:
    resolved_task_names, cache_root = prepare_dataset_evaluation(
        dataset_name,
        task_names,
        all_tasks=all_tasks,
    )
    selected_seeds = seeds or DEFAULT_BASELINE_SEEDS

    summary: dict[str, Any] = {
        "dataset": dataset_name,
        "seeds": selected_seeds,
        "tasks": {},
    }

    for task_name in resolved_task_names:
        print(
            f"Running baseline for {dataset_name}/{task_name} across seeds {selected_seeds}...",
            file=sys.stderr,
        )
        summary["tasks"][task_name] = evaluate_task_across_seeds(
            dataset_name=dataset_name,
            task_name=task_name,
            seeds=selected_seeds,
            cache_root=cache_root,
            wandb_project=wandb_project,
        )
    return summary
