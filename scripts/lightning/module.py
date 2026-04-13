from typing import Any

import numpy as np
import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from relbench.base import TaskType
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, L1Loss
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from scripts.model import Model
from task.utils import MEntityTask
from task.utils.transform import TargetTransform


class EntityGNNLightningModule(LightningModule):
    def __init__(
        self,
        *,
        task: MEntityTask,
        data: Any,
        col_stats_dict: dict,
        split_inputs: dict[str, Any],
        task_node_type: str,
        target_transform: TargetTransform | None,
        num_layers: int,
        channels: int,
        aggr: str,
        lr: float,
        epochs: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=["task", "data", "col_stats_dict", "split_inputs", "target_transform"],
        )

        self.task = task
        self.task_node_type = task_node_type
        self.target_transform = target_transform
        self.lr = lr
        self.epochs = epochs

        self.out_channels, self.loss_fn, self.tune_metric, self.higher_is_better = self._build_task_setup(
            task=task,
            split_inputs=split_inputs,
        )
        self.clamp_min, self.clamp_max = self._get_regression_clamp(task)

        self.model = Model(
            data=data,
            col_stats_dict=col_stats_dict,
            num_layers=num_layers,
            channels=channels,
            out_channels=self.out_channels,
            aggr=aggr,
            norm="batch_norm",
        )

        self._val_preds: list[Tensor] = []
        self._val_targets: list[Tensor] = []
        self._test_preds: list[Tensor] = []
        self._test_targets: list[Tensor] = []

    @property
    def checkpoint_monitor(self) -> str:
        return f"val/{self.tune_metric}"

    @property
    def checkpoint_mode(self) -> str:
        return "max" if self.higher_is_better else "min"

    def forward(self, batch: Any) -> Tensor:
        return self.model(batch, self.task_node_type)

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        pred = self._reshape_prediction(self(batch))
        target = batch[self.task_node_type].y
        if self.task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            loss = self.loss_fn(pred, target.long())
        else:
            loss = self.loss_fn(pred.float(), target.float())

        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=target.size(0))
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        raw_pred = self._reshape_prediction(self(batch))
        target = batch[self.task_node_type].y
        if self.task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            loss = self.loss_fn(raw_pred, target.long())
        else:
            loss = self.loss_fn(raw_pred.float(), target.float())
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=target.size(0))
        self._val_preds.append(self._postprocess_for_eval(raw_pred).detach().cpu())
        self._val_targets.append(target.detach().cpu())

    def on_validation_epoch_start(self) -> None:
        self._val_preds = []
        self._val_targets = []

    def on_validation_epoch_end(self) -> None:
        self._log_eval_metrics("val", self._val_preds, self.task.get_table("val", mask_input_cols=False))

    def test_step(self, batch: Any, batch_idx: int) -> None:
        raw_pred = self._reshape_prediction(self(batch))
        target = batch[self.task_node_type].y
        if self.task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            loss = self.loss_fn(raw_pred, target.long())
        else:
            loss = self.loss_fn(raw_pred.float(), target.float())
        self.log("test/loss", loss, on_step=False, on_epoch=True, batch_size=target.size(0))
        self._test_preds.append(self._postprocess_for_eval(raw_pred).detach().cpu())
        self._test_targets.append(target.detach().cpu())

    def on_test_epoch_start(self) -> None:
        self._test_preds = []
        self._test_targets = []

    def on_test_epoch_end(self) -> None:
        self._log_eval_metrics("test", self._test_preds, self.task.get_table("test", mask_input_cols=False))

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def _reshape_prediction(self, pred: Tensor) -> Tensor:
        return pred.view(-1) if pred.dim() > 1 and pred.size(1) == 1 else pred

    def _postprocess_for_eval(self, pred: Tensor) -> Tensor:
        if self.task.task_type == TaskType.REGRESSION:
            assert self.clamp_min is not None
            assert self.clamp_max is not None
            if self.target_transform is not None:
                pred = self.target_transform.inverse_transform(pred)
            pred = torch.clamp(pred, self.clamp_min, self.clamp_max)

        if self.task.task_type in (
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTILABEL_CLASSIFICATION,
        ):
            pred = torch.sigmoid(pred)
        elif self.task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            pred = torch.softmax(pred, dim=1)

        return pred

    def _log_eval_metrics(
        self,
        split: str,
        pred_list: list[Tensor],
        target_table: Any,
    ) -> None:
        if not pred_list:
            return

        pred = torch.cat(pred_list, dim=0).numpy()
        metrics = self.task.evaluate(pred, target_table)
        for name, value in metrics.items():
            self.log(
                f"{split}/{name}",
                float(value),
                prog_bar=(split == "val" and name == self.tune_metric),
                on_step=False,
                on_epoch=True,
                logger=True,
            )

    @staticmethod
    def _build_task_setup(
        task: MEntityTask,
        split_inputs: dict[str, Any],
    ) -> tuple[int, torch.nn.Module, str, bool]:
        if task.task_type == TaskType.BINARY_CLASSIFICATION:
            return 1, BCEWithLogitsLoss(), "roc_auc", True

        if task.task_type == TaskType.REGRESSION:
            return 1, L1Loss(), "mae", False

        if task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
            return task.num_labels, BCEWithLogitsLoss(), "multilabel_auprc_macro", True  # type: ignore[attr-defined]

        if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            out_channels = task.num_classes  # type: ignore[attr-defined]
            train_target = split_inputs["train"].target.long()
            class_counts = torch.bincount(train_target, minlength=out_channels).float()
            class_weights = class_counts.sum() / torch.clamp(class_counts, min=1.0)
            class_weights = class_weights / class_weights.mean()
            return out_channels, CrossEntropyLoss(weight=class_weights), "multiclass_f1", True

        raise ValueError(f"Task type {task.task_type} is unsupported")

    @staticmethod
    def _get_regression_clamp(task: MEntityTask) -> tuple[float | None, float | None]:
        if task.task_type != TaskType.REGRESSION:
            return None, None

        train_table = task.get_table("train")
        clamp_min, clamp_max = np.percentile(
            train_table.df[task.target_col].to_numpy(),
            [2, 98],
        )
        return float(clamp_min), float(clamp_max)
