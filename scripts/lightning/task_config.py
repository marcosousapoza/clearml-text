from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from relbench.base import TaskType
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, L1Loss

from task.utils import MEntityTask


@dataclass
class TaskSetup:
    task_type: TaskType
    out_channels: int
    loss_fn: nn.Module
    tune_metric: str
    higher_is_better: bool
    clamp_min: float | None
    clamp_max: float | None


def build_task_setup(task: MEntityTask) -> TaskSetup:
    """Derive all task-type-specific training configuration from a task object."""
    clamp_min, clamp_max = None, None

    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        out_channels, loss_fn, tune_metric, higher_is_better = 1, BCEWithLogitsLoss(), "roc_auc", True

    elif task.task_type == TaskType.REGRESSION:
        out_channels, loss_fn, tune_metric, higher_is_better = 1, L1Loss(), "mae", False
        train_table = task.get_table("train")
        clamp_min, clamp_max = (
            float(v)
            for v in np.percentile(train_table.df[task.target_col].to_numpy(), [2, 98])
        )

    elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
        out_channels = task.num_labels  # type: ignore[attr-defined]
        loss_fn, tune_metric, higher_is_better = BCEWithLogitsLoss(), "multilabel_auprc_macro", True

    elif task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        out_channels = task.num_classes  # type: ignore[attr-defined]
        loss_fn, tune_metric, higher_is_better = CrossEntropyLoss(), "multiclass_f1", True

    else:
        raise ValueError(f"Task type {task.task_type} is unsupported")

    return TaskSetup(
        task_type=task.task_type,
        out_channels=out_channels,
        loss_fn=loss_fn,
        tune_metric=tune_metric,
        higher_is_better=higher_is_better,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
    )
