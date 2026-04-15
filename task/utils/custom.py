from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import sklearn.metrics as skm
from numpy.typing import NDArray
from relbench.base import BaseTask, Database, Table, TaskType
from relbench.modeling.graph import (
    AttachTargetTransform,
    NodeTrainTableInput,
    to_unix_time,
)

from data.const import OBJECT_ID_COL, OBJECT_TABLE, TIME_COL
from task.metrics import roc_auc
from .transform import TargetTransform


# ---------------------------------------------------------------------------
# Per-task-type statistics helpers
# Each function fills the `stats` dict for one task type. Keeping them small
# and separate makes it easy to add a new task type without touching the rest.
# ---------------------------------------------------------------------------

def _stats_binary(df: pd.DataFrame, col: str, stats: dict) -> None:
    # For binary tasks we mostly care about class balance.
    stats["num_positives"] = int((df[col] == 1).sum())
    stats["num_negatives"] = int((df[col] == 0).sum())


def _stats_regression(df: pd.DataFrame, col: str, stats: dict) -> None:
    # Regression targets: distribution shape matters more than raw counts.
    quantiles = df[col].quantile([0.25, 0.5, 0.75])
    stats["min_target"]         = df[col].min()
    stats["max_target"]         = df[col].max()
    stats["mean_target"]        = df[col].mean()
    stats["quantile_25_target"] = quantiles.iloc[0]
    stats["median_target"]      = quantiles.iloc[1]
    stats["quantile_75_target"] = quantiles.iloc[2]


def _stats_multilabel(df: pd.DataFrame, col: str, stats: dict) -> None:
    # Multilabel: track how many labels each entity has and per-class frequency.
    arr       = np.array([row for row in df[col]])
    arr_row   = arr.sum(1)
    arr_class = arr.sum(0)
    max_idx   = int(arr_class.argmax())
    min_idx   = int(arr_class.argmin())
    stats["mean_num_classes_per_entity"] = round(float(arr_row.mean()), 4)
    stats["max_num_classes_per_entity"]  = arr_row.max()
    stats["min_num_classes_per_entity"]  = arr_row.min()
    stats["max_num_class_idx"]           = max_idx
    stats["max_num_class_num"]           = arr_class[max_idx]
    stats["min_num_class_idx"]           = min_idx
    stats["min_num_class_num"]           = arr_class[min_idx]


_STATS_HANDLERS: dict[TaskType, Callable] = {
    TaskType.BINARY_CLASSIFICATION:      _stats_binary,
    TaskType.REGRESSION:                 _stats_regression,
    TaskType.MULTILABEL_CLASSIFICATION:  _stats_multilabel,
}


# ---------------------------------------------------------------------------
# Base task class
# ---------------------------------------------------------------------------

class MEntityTask(BaseTask):
    """Entity-style task with one or more foreign-key entity columns.

    Subclasses declare task_type, object_types, timedelta, metrics, and
    implement make_table(). Everything else has sensible defaults for the
    common single-entity case; pair-entity tasks override entity_cols and
    entity_tables.
    """

    # Defaults for single-entity tasks (12 of 16 concrete classes use these).
    # Pair-entity tasks that operate on O2O object pairs must override both.
    entity_cols:   tuple[str, ...] = (OBJECT_ID_COL,)
    entity_tables: tuple[str, ...] = (OBJECT_TABLE,)
    time_col:   str = TIME_COL
    target_col: str = "target"

    task_type: TaskType
    object_types: tuple[str, ...]
    num_eval_timestamps: int = 1000 # very large number to generate all validation observations

    def make_target_transform(self) -> TargetTransform | None:
        return None

    def _make_table(self, df: pd.DataFrame) -> Table:
        """Wraps a result DataFrame in the standard RelBench Table shape.

        All tasks share the same structure: no primary key, entity columns
        pointing into the shared object table, and a shared time column.
        """
        return Table(
            df=df,
            fkey_col_to_pkey_table=dict(zip(self.entity_cols, self.entity_tables)),
            pkey_col=None,
            time_col=self.time_col,
        )

    def filter_dangling_entities(self, table: Table) -> Table:
        db = self.dataset.get_db()
        filter_mask = pd.Series(False, index=table.df.index)
        for entity_col, entity_table in zip(self.entity_cols, self.entity_tables):
            num_entities = len(db.table_dict[entity_table])
            filter_mask |= table.df[entity_col] >= num_entities

        if filter_mask.any():
            table.df = table.df[~filter_mask].reset_index(drop=True)

        return table

    def evaluate(
        self,
        pred: NDArray,
        target_table: Table | None = None,
        metrics=None,
    ) -> dict[str, float]:
        """Evaluate predictions against ground truth.

        Handles multiclass classification specially: f1 needs argmax first,
        and roc_auc needs raw class scores rather than hard predictions.
        """
        if metrics is None:
            metrics = self.metrics

        if target_table is None:
            target_table = self.get_table("test", mask_input_cols=False)

        target = target_table.df[self.target_col].to_numpy()
        if len(pred) != len(target):
            raise ValueError(
                f"The length of pred and target must be the same (got "
                f"{len(pred)} and {len(target)}, respectively)."
            )

        results: dict[str, float] = {}
        for fn in metrics:
            metric_name = fn.__name__
            if self.task_type == TaskType.MULTICLASS_CLASSIFICATION and metric_name == "f1":
                results["multiclass_f1"] = float(skm.f1_score(
                    target,
                    pred.argmax(axis=1),
                    average="macro",
                ))
                continue
            if self.task_type == TaskType.MULTICLASS_CLASSIFICATION and metric_name == "roc_auc":
                results["multiclass_roc_auc"] = roc_auc(target, pred)
                continue
            results[metric_name] = fn(target, pred)

        return results

    def stats(self) -> dict[str, dict[str, Any]]:
        res = {}
        for split in ["train", "val", "test"]:
            table = self.get_table(split, mask_input_cols=False)
            timestamps = table.df[self.time_col].unique()
            split_stats = {}
            for timestamp in timestamps:
                temp_df = table.df[table.df[self.time_col] == timestamp]
                s = {
                    "num_rows": len(temp_df),
                    "num_unique_entity_tuples": temp_df[list(self.entity_cols)].drop_duplicates().shape[0],
                }
                for entity_col in self.entity_cols:
                    s[f"num_unique_{entity_col}"] = temp_df[entity_col].nunique()
                self._set_stats(temp_df, s)
                split_stats[str(timestamp)] = s

            split_stats["total"] = {
                "num_rows": len(table.df),
                "num_unique_entity_tuples": table.df[list(self.entity_cols)].drop_duplicates().shape[0],
            }
            for entity_col in self.entity_cols:
                split_stats["total"][f"num_unique_{entity_col}"] = table.df[entity_col].nunique()
            self._set_stats(table.df, split_stats["total"])
            res[split] = split_stats

        total_df = pd.concat(
            [
                table.df
                for table in [
                    self.get_table(split, mask_input_cols=False)
                    for split in ["train", "val", "test"]
                ]
                if table is not None
            ]
        )
        res["total"] = {}
        self._set_stats(total_df, res["total"])

        train_uniques = set(map(tuple, self.get_table("train").df[list(self.entity_cols)].drop_duplicates().to_numpy()))
        test_uniques = set(
            map(
                tuple,
                self.get_table("test", mask_input_cols=False).df[list(self.entity_cols)].drop_duplicates().to_numpy(),
            )
        )
        res["total"]["ratio_train_test_entity_overlap"] = (
            len(train_uniques.intersection(test_uniques)) / len(test_uniques)
            if test_uniques
            else float("nan")
        )
        return res

    def _set_stats(self, df: pd.DataFrame, stats: dict[str, Any]) -> None:
        handler = _STATS_HANDLERS.get(self.task_type)
        if handler is None:
            raise ValueError(f"Unsupported task type {self.task_type}")
        handler(df, self.target_col, stats)


# ---------------------------------------------------------------------------
# Training table helper
# ---------------------------------------------------------------------------

def add_task_to_database(
    db: Any,
    task: "MEntityTask",
    task_name: str,
    col_to_stype_dict: dict,
) -> tuple[Any, dict[str, NodeTrainTableInput], TargetTransform | None]:
    """Add one MEntityTask's label table to the database and return split inputs.

    Concatenates train/val/test DataFrames into a single label table (without
    the target column), registers it in the database, then builds per-split
    NodeTrainTableInput objects with time-shifted indices and optional target
    normalization.
    """

    def target_tensor(df: pd.DataFrame) -> torch.Tensor:
        # Each task type needs labels packed differently for the loss function.
        if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            return torch.from_numpy(df[task.target_col].to_numpy(dtype=int, copy=True))
        if task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
            return torch.from_numpy(np.stack(df[task.target_col].values))  # type: ignore
        return torch.from_numpy(df[task.target_col].to_numpy(dtype=float, copy=True))

    labels_table_name = f"{task_name}_labels"
    split_frames: dict[str, pd.DataFrame] = {}

    for split in ["train", "val", "test"]:
        split_df = task.get_table(split, mask_input_cols=False).df.copy()
        split_df[task.time_col] = split_df[task.time_col] + task.timedelta
        split_df = split_df.sort_values(task.time_col, kind="stable").reset_index(drop=True)
        split_frames[split] = split_df

    label_df = pd.concat(
        [split_frames["train"], split_frames["val"], split_frames["test"]],
        ignore_index=True,
    )
    label_feature_df = label_df.drop(columns=[task.target_col])

    fkey_col_to_pkey_table = dict(zip(task.entity_cols, task.entity_tables))
    db.table_dict[labels_table_name] = Table(
        df=label_feature_df,
        fkey_col_to_pkey_table=fkey_col_to_pkey_table,
        pkey_col=None,
        time_col=task.time_col,
    )

    # Task labels must not be part of the node features.
    col_to_stype_dict[labels_table_name] = {}
    target_transform = task.make_target_transform()
    if target_transform is not None:
        target_transform.fit(target_tensor(split_frames["train"]))

    split_inputs: dict[str, NodeTrainTableInput] = {}
    start = 0
    for split in ["train", "val", "test"]:
        split_df = split_frames[split]
        stop = start + len(split_df)
        node_ids = torch.arange(start, stop, dtype=torch.long)
        target = target_tensor(split_df)
        if target_transform is not None:
            target = target_transform.transform(target)
        split_inputs[split] = NodeTrainTableInput(
            nodes=(labels_table_name, node_ids),
            time=torch.from_numpy(to_unix_time(split_df[task.time_col])),
            target=target,
            transform=AttachTargetTransform(labels_table_name, target),
        )
        start = stop

    return db, split_inputs, target_transform
