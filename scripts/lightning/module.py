from typing import Any

from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from relbench.base import TaskType
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from scripts.model import Model
from task.utils import MEntityTask
from task.utils.transform import TargetTransform

from .data import DataArtifacts
from .metrics import RelbenchEvalMetric
from .task_config import TaskSetup, build_task_setup


class EntityGNNLightningModule(LightningModule):
    def __init__(
        self,
        *,
        num_layers: int,
        channels: int,
        aggr: str,
        lr: float,
        epochs: int,
        # Artifact args — provided directly (HPO / old CLI) or via configure_from_artifacts() (LightningCLI)
        task: MEntityTask | None = None,
        data: Any = None,
        col_stats_dict: dict | None = None,
        split_inputs: dict[str, Any] | None = None,
        task_node_type: str | None = None,
        target_transform: TargetTransform | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=["task", "data", "col_stats_dict", "split_inputs", "target_transform"],
        )
        self.lr = lr
        self.epochs = epochs

        if task is not None:
            assert data is not None and col_stats_dict is not None and task_node_type is not None
            self._init_from_artifacts(task, data, col_stats_dict, task_node_type, target_transform)

    def configure_from_artifacts(self, artifacts: DataArtifacts) -> None:
        """Deferred initialisation called by LightningCLI after datamodule.setup()."""
        self._init_from_artifacts(
            artifacts.task,
            artifacts.data,
            artifacts.col_stats_dict,
            artifacts.task_node_type,
            artifacts.target_transform,
        )

    def _init_from_artifacts(
        self,
        task: MEntityTask,
        data: Any,
        col_stats_dict: dict,
        task_node_type: str,
        target_transform: TargetTransform | None,
    ) -> None:
        hparams = self.hparams
        self.task = task
        self.task_node_type = task_node_type

        task_setup: TaskSetup = build_task_setup(task)
        self.task_setup = task_setup

        self.model = Model(
            data=data,
            col_stats_dict=col_stats_dict,
            num_layers=hparams["num_layers"],
            channels=hparams["channels"],
            out_channels=task_setup.out_channels,
            aggr=hparams["aggr"],
            norm="batch_norm",
        )

        self.val_metric = RelbenchEvalMetric(task, "val", task_setup, target_transform)
        self.test_metric = RelbenchEvalMetric(task, "test", task_setup, target_transform)

        # Register loss weight as a buffer so Lightning moves it to the correct device automatically
        loss_weight = getattr(task_setup.loss_fn, "weight", None)
        self.register_buffer("_loss_weight", loss_weight)  # None registers as None (no-op)

    def on_fit_start(self) -> None:
        # Sync loss weight to the device Lightning chose (buffer was moved by .to(device))
        if self._loss_weight is not None:
            self.task_setup.loss_fn.weight = self._loss_weight

    @property
    def checkpoint_monitor(self) -> str:
        return f"val/{self.task_setup.tune_metric}"

    @property
    def checkpoint_mode(self) -> str:
        return "max" if self.task_setup.higher_is_better else "min"

    def forward(self, batch: Any) -> Tensor:
        return self.model(batch, self.task_node_type)

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        raw_pred = self(batch)
        pred = raw_pred.view(-1) if raw_pred.dim() > 1 and raw_pred.size(1) == 1 else raw_pred
        target = batch[self.task_node_type].y
        if self.task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            loss = self.task_setup.loss_fn(pred, target.long())
        else:
            loss = self.task_setup.loss_fn(pred.float(), target.float())
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=target.size(0))
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        raw_pred = self(batch)
        pred = raw_pred.view(-1) if raw_pred.dim() > 1 and raw_pred.size(1) == 1 else raw_pred
        target = batch[self.task_node_type].y
        if self.task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            loss = self.task_setup.loss_fn(pred, target.long())
        else:
            loss = self.task_setup.loss_fn(pred.float(), target.float())
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=target.size(0))
        self.val_metric.update(pred, target)

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metric.compute()
        tune = self.task_setup.tune_metric
        if tune in metrics:
            self.log(f"val/{tune}", float(metrics[tune]), prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.log_dict(
            {f"val/{k}": float(v) for k, v in metrics.items() if k != tune},
            on_step=False, on_epoch=True, logger=True,
        )
        self.val_metric.reset()

    def test_step(self, batch: Any, batch_idx: int) -> None:
        raw_pred = self(batch)
        pred = raw_pred.view(-1) if raw_pred.dim() > 1 and raw_pred.size(1) == 1 else raw_pred
        target = batch[self.task_node_type].y
        if self.task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            loss = self.task_setup.loss_fn(pred, target.long())
        else:
            loss = self.task_setup.loss_fn(pred.float(), target.float())
        self.log("test/loss", loss, on_step=False, on_epoch=True, batch_size=target.size(0))
        self.test_metric.update(pred, target)

    def on_test_epoch_end(self) -> None:
        metrics = self.test_metric.compute()
        tune = self.task_setup.tune_metric
        if tune in metrics:
            self.log(f"test/{tune}", float(metrics[tune]), on_step=False, on_epoch=True, logger=True)
        self.log_dict(
            {f"test/{k}": float(v) for k, v in metrics.items() if k != tune},
            on_step=False, on_epoch=True, logger=True,
        )
        self.test_metric.reset()

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
