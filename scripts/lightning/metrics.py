from typing import Any, cast

import torch
import torchmetrics
from relbench.base import Table, TaskType
from torch import Tensor

from task.utils import MEntityTask
from task.utils.transform import TargetTransform

from .task_config import TaskSetup


def postprocess_pred(raw_pred: Tensor, task_setup: TaskSetup, target_transform: TargetTransform | None) -> Tensor:
    """Apply task-type-specific postprocessing to a raw model output tensor."""
    if task_setup.task_type == TaskType.REGRESSION:
        if target_transform is not None:
            raw_pred = target_transform.inverse_transform(raw_pred)
        return torch.clamp(raw_pred, task_setup.clamp_min, task_setup.clamp_max)

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
        if self.task_setup.clamp_min is not None and self.target_transform is not None:
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
