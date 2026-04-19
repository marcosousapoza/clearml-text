import argparse
from pathlib import Path

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger, Logger
from torch_geometric.seed import seed_everything

from .data import RelbenchLightningDataModule
from .module import EntityGNNLightningModule
from .warnings import configure_training_warnings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train entity-task GNNs with PyTorch Lightning.")
    parser.add_argument("--dataset", type=str, default="rel-event")
    parser.add_argument("--task", type=str, default="user-attendance")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--channels", type=int, default=48)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-neighbors", type=int, default=16)
    parser.add_argument("--gnn-type", type=str, default="sage", choices=["sage", "hgt"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--default-root-dir", type=str, default=None)
    parser.add_argument("--flatten", action="store_true")
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Also log metrics to Weights & Biases.")
    parser.add_argument("--wandb-project", type=str, default="ocel-ocp")
    return parser


def main(argv: list[str] | None = None) -> None:
    configure_training_warnings()
    args = build_parser().parse_args(argv)

    if torch.cuda.is_available():
        torch.set_num_threads(1)
    seed_everything(args.seed)

    datamodule = RelbenchLightningDataModule(
        dataset_name=args.dataset,
        task_name=args.task,
        batch_size=args.batch_size,
        num_layers=args.num_layers,
        num_neighbors=args.num_neighbors,
        cache_dir=args.cache_dir,
        flatten=args.flatten,
    )
    datamodule.setup("fit")
    assert datamodule.artifacts is not None

    module = EntityGNNLightningModule(
        num_layers=args.num_layers,
        channels=args.channels,
        gnn_type=args.gnn_type,
        lr=args.lr,
        patience=args.patience,
        task=datamodule.artifacts.task,
        data=datamodule.artifacts.data,
        col_stats_dict=datamodule.artifacts.col_stats_dict,
        entity_table=datamodule.artifacts.entity_table,
        tuple_arity=datamodule.artifacts.tuple_arity,
    )

    run_name = f"{args.dataset}_{args.task}{'_flat' if args.flatten else ''}"
    root_dir = Path(args.default_root_dir) if args.default_root_dir else (
        datamodule.artifacts.cache_root / run_name / "lightning"
    )
    loggers: list[Logger] = [
        CSVLogger(save_dir=str(root_dir), name="logs"),
    ]
    if args.wandb:
        loggers.append(
            WandbLogger(
                project=args.wandb_project,
                name=run_name,
                config=vars(args),
                save_dir=str(root_dir),
            )
        )
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(root_dir / "checkpoints"),
        filename="epoch={epoch:02d}-{" + module.checkpoint_monitor.replace("/", "_") + ":.4f}",
        monitor=module.checkpoint_monitor,
        mode=module.checkpoint_mode,
        save_top_k=1,
        save_last=True,
    )
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision="32-true",
        max_epochs=args.epochs,
        limit_train_batches=2000,
        num_sanity_val_steps=0,
        default_root_dir=str(root_dir),
        logger=loggers,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval="epoch"),
        ],
        fast_dev_run=args.fast_dev_run,
        inference_mode=False,
        log_every_n_steps=1,
    )

    trainer.fit(module, datamodule=datamodule)

    best_path = checkpoint_callback.best_model_path or None
    if best_path:
        print(f"Best checkpoint: {best_path}")
    if not args.fast_dev_run:
        trainer.test(module, datamodule=datamodule, ckpt_path="best" if best_path else None)
