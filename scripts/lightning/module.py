from typing import Any

from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from relbench.base import TaskType
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from scripts.model import Model
from task.utils import MEntityTask

from .data import DataArtifacts
from .metrics import RelbenchEvalMetric, TaskSetup, build_task_setup
from .tuple import TupleConcatPredictor


class EntityGNNLightningModule(LightningModule):
    def __init__(
        self,
        *,
        num_layers: int,
        channels: int,
        lr: float,
        patience: int = 10,
        min_lr: float = 1e-6,
        aggr: str = "sum",
        gnn_type: str = "sage",
        hgt_heads: int = 4,
        dropout: float = 0.3,
        weight_decay: float = 1e-4,
        # Artifact args — provided directly (HPO / old CLI) or via configure_from_artifacts() (LightningCLI)
        task: MEntityTask | None = None,
        data: Any = None,
        col_stats_dict: dict | None = None,
        entity_table: str | None = None,
        tuple_arity: int | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=["task", "data", "col_stats_dict"],
        )
        self.lr = lr
        self.patience = patience
        self.min_lr = min_lr
        self.weight_decay = weight_decay

        if task is not None:
            assert (
                data is not None
                and col_stats_dict is not None
                and entity_table is not None
                and tuple_arity is not None
            )
            self._init_from_artifacts(task, data, col_stats_dict, entity_table, tuple_arity)

    def configure_from_artifacts(self, artifacts: DataArtifacts) -> None:
        """Deferred initialisation called by LightningCLI after datamodule.setup()."""
        self._init_from_artifacts(
            artifacts.task,
            artifacts.data,
            artifacts.col_stats_dict,
            artifacts.entity_table,
            artifacts.tuple_arity,
        )

    def _init_from_artifacts(
        self,
        task: MEntityTask,
        data: Any,
        col_stats_dict: dict,
        entity_table: str,
        tuple_arity: int,
    ) -> None:
        hparams = self.hparams
        self.task = task
        self.entity_table = entity_table
        self.tuple_arity = tuple_arity

        task_setup: TaskSetup = build_task_setup(task)
        self.task_setup = task_setup

        channels = hparams["channels"]
        self.model = Model(
            data=data,
            col_stats_dict=col_stats_dict,
            num_layers=hparams["num_layers"],
            channels=channels,
            aggr=hparams["aggr"],
            gnn_type=hparams["gnn_type"],
            hgt_heads=hparams["hgt_heads"],
            dropout=hparams["dropout"],
        )
        self.predictor = TupleConcatPredictor(
            hidden_dim=channels,
            tuple_arity=tuple_arity,
            out_dim=task_setup.out_channels,
            hidden_dims=(channels,),
            dropout=hparams["dropout"],
            layer_norm=True,
        )

        self.val_metric = RelbenchEvalMetric(task, "val", task_setup)
        self.test_metric = RelbenchEvalMetric(task, "test", task_setup)

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
        node_embeddings = self.model(batch, self.entity_table)
        return self.predictor(node_embeddings, batch[self.entity_table])

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        raw_pred = self(batch)
        pred = raw_pred.view(-1) if raw_pred.dim() > 1 and raw_pred.size(1) == 1 else raw_pred
        target = batch[self.entity_table].tuple_y
        if self.task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            loss = self.task_setup.loss_fn(pred, target.long())
        else:
            loss = self.task_setup.loss_fn(pred.float(), target.float())
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=target.size(0))
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        raw_pred = self(batch)
        pred = raw_pred.view(-1) if raw_pred.dim() > 1 and raw_pred.size(1) == 1 else raw_pred
        target = batch[self.entity_table].tuple_y
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
        target = batch[self.entity_table].tuple_y
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
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=self.checkpoint_mode,  # type: ignore[arg-type]
            factor=0.5,
            patience=self.patience,
            min_lr=self.min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.checkpoint_monitor,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def lr_scheduler_step(self, scheduler: Any, metric: Any) -> None:
        scheduler.step(metric)
        optimizer = scheduler.optimizer
        if all(group["lr"] <= self.min_lr for group in optimizer.param_groups):
            self.trainer.should_stop = True
