from dataclasses import dataclass
from typing import Any, cast

import torch
import numpy as np
import torchmetrics
from relbench.base import Table, TaskType
from torch import Tensor
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, L1Loss

from task.utils import MEntityTask
from task.utils.transform import TargetTransform


@dataclass
class TaskSetup:
    task_type: TaskType
    out_channels: int
    loss_fn: nn.Module
    tune_metric: str
    higher_is_better: bool


def _class_weights(task: MEntityTask) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss, computed from the training table."""
    train_table = task.get_table("train")
    targets = train_table.df[task.target_col].to_numpy()
    num_classes = task.num_classes  # type: ignore[attr-defined]
    counts = np.bincount(targets.astype(int), minlength=num_classes).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    weights = 1.0 / counts
    weights /= weights.sum()
    return torch.tensor(weights, dtype=torch.float32)


def build_task_setup(task: MEntityTask) -> TaskSetup:
    """Derive task-type-specific training and evaluation configuration."""
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        out_channels, loss_fn, tune_metric, higher_is_better = 1, BCEWithLogitsLoss(), "roc_auc", True
    elif task.task_type == TaskType.REGRESSION:
        out_channels, loss_fn, tune_metric, higher_is_better = 1, L1Loss(), "mae", False
    elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
        out_channels = task.num_classes  # type: ignore[attr-defined]
        loss_fn, tune_metric, higher_is_better = BCEWithLogitsLoss(), "multilabel_f1_macro", True
    elif task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        out_channels = task.num_classes  # type: ignore[attr-defined]
        loss_fn = CrossEntropyLoss(weight=_class_weights(task))
        tune_metric, higher_is_better = "multiclass_f1", True
    else:
        raise ValueError(f"Task type {task.task_type} is unsupported")

    return TaskSetup(
        task_type=task.task_type,
        out_channels=out_channels,
        loss_fn=loss_fn,
        tune_metric=tune_metric,
        higher_is_better=higher_is_better,
    )


def postprocess_pred(raw_pred: Tensor, task_setup: TaskSetup, target_transform: TargetTransform | None) -> Tensor:
    """Apply task-type-specific postprocessing to a raw model output tensor."""
    if task_setup.task_type == TaskType.REGRESSION:
        if target_transform is not None:
            raw_pred = target_transform.inverse_transform(raw_pred)
        return raw_pred

    if task_setup.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTILABEL_CLASSIFICATION):
        return torch.sigmoid(raw_pred)

    if task_setup.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        return torch.softmax(raw_pred, dim=1)

    return raw_pred


class RelbenchEvalMetric(torchmetrics.Metric):
    """Accumulates per-batch predictions and targets, then evaluates via task.evaluate()."""

    full_state_update = False

    def __init__(
        self,
        task: MEntityTask,
        split: str,
        task_setup: TaskSetup,
        target_transform: TargetTransform | None,
    ) -> None:
        super().__init__()
        self.task = task
        self.split = split
        self.task_setup = task_setup
        self.target_transform = target_transform
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, raw_pred: Tensor, target: Tensor) -> None:
        preds = cast(list[Tensor], self.preds)
        targets = cast(list[Tensor], self.targets)
        preds.append(postprocess_pred(raw_pred, self.task_setup, self.target_transform).detach().cpu())
        targets.append(target.detach().cpu())

    def compute(self) -> dict[str, Any]:
        preds = cast(list[Tensor], self.preds)
        targets = cast(list[Tensor], self.targets)
        if not preds:
            return {}

        pred = torch.cat(preds, dim=0).numpy()

        # Inverse-transform regression targets if needed
        raw_targets = torch.cat(targets, dim=0)
        if self.task_setup.task_type == TaskType.REGRESSION and self.target_transform is not None:
            raw_targets = self.target_transform.inverse_transform(raw_targets)
        target = raw_targets.numpy()

        target_table = self.task.get_table(self.split, mask_input_cols=False)
        if len(pred) < len(target_table):
            target_table = Table(
                df=target_table.df.iloc[: len(pred)].copy(),
                fkey_col_to_pkey_table=target_table.fkey_col_to_pkey_table,
                pkey_col=target_table.pkey_col,
                time_col=target_table.time_col,
            )
        target_table.df[self.task.target_col] = list(target) if target.ndim > 1 else target
        return self.task.evaluate(pred, target_table)
