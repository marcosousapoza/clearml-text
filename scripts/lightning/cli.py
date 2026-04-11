import argparse
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch_geometric.seed import seed_everything

from .data import RelbenchLightningDataModule
from .module import EntityGNNLightningModule


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train entity-task GNNs with PyTorch Lightning.")
    parser.add_argument("--dataset", type=str, default="rel-event")
    parser.add_argument("--task", type=str, default="user-attendance")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--aggr", type=str, default="sum")
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_neighbors", type=int, default=128)
    parser.add_argument("--temporal_strategy", type=str, default="last")
    parser.add_argument("--max_steps_per_epoch", type=int, default=2000)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="32-true")
    parser.add_argument("--default_root_dir", type=str, default=None)
    parser.add_argument("--num_sanity_val_steps", type=int, default=0)
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to a saved Lightning checkpoint to test without training.",
    )
    parser.add_argument("--fast_dev_run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
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
        temporal_strategy=args.temporal_strategy,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
    )
    datamodule.setup("fit")
    assert datamodule.artifacts is not None

    module = EntityGNNLightningModule(
        task=datamodule.artifacts.task,
        data=datamodule.artifacts.data,
        col_stats_dict=datamodule.artifacts.col_stats_dict,
        split_inputs=datamodule.artifacts.split_inputs,
        task_node_type=datamodule.artifacts.task_node_type,
        num_layers=args.num_layers,
        channels=args.channels,
        aggr=args.aggr,
        lr=args.lr,
        epochs=args.epochs,
    )

    root_dir = Path(args.default_root_dir) if args.default_root_dir else (
        datamodule.artifacts.cache_root / f"{args.dataset}_{args.task}" / "lightning"
    )
    checkpoint_dir = root_dir / "checkpoints"
    logger = CSVLogger(save_dir=str(root_dir), name="logs")
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="epoch={epoch:02d}-{" + module.checkpoint_monitor.replace("/", "_") + ":.4f}",
        monitor=module.checkpoint_monitor,
        mode=module.checkpoint_mode,
        save_top_k=1,
        save_last=True,
    )

    trainer = Trainer(
        accelerator=args.accelerator,
        devices=_parse_devices(args.devices),
        precision=args.precision,
        max_epochs=args.epochs,
        limit_train_batches=args.max_steps_per_epoch,
        num_sanity_val_steps=args.num_sanity_val_steps,
        default_root_dir=str(root_dir),
        logger=logger,
        callbacks=[checkpoint_callback],
        fast_dev_run=args.fast_dev_run,
    )

    if args.ckpt_path is not None:
        trainer.test(module, datamodule=datamodule, ckpt_path=args.ckpt_path)
        return

    trainer.fit(module, datamodule=datamodule)

    best_path = checkpoint_callback.best_model_path or None
    if best_path:
        print(f"Best checkpoint: {best_path}")
    if not args.fast_dev_run:
        trainer.test(module, datamodule=datamodule, ckpt_path="best" if best_path else None)


def _parse_devices(value: str) -> str | int | list[int]:
    if value == "auto":
        return value
    if "," in value:
        return [int(item) for item in value.split(",") if item]
    return int(value)
