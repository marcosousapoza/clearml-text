from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HpoConfig:
    action: str
    dataset: str
    task: str
    study_name: str
    n_trials: int
    cache_dir: str | None
    seed: int
    timeout: float | None
    accelerator: str
    devices: str
    precision: str
    num_workers: int
    pruner_startup_trials: int
    pruner_warmup_steps: int
    num_sanity_val_steps: int
    fixed_params: dict[str, Any]


def build_config(args: Any) -> HpoConfig:
    return HpoConfig(
        action=args.action,
        dataset=args.dataset,
        task=args.task,
        study_name=args.study_name or f"{args.dataset}:{args.task}",
        n_trials=args.n_trials,
        cache_dir=args.cache_dir,
        seed=args.seed,
        timeout=args.timeout,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        num_workers=args.num_workers,
        pruner_startup_trials=args.pruner_startup_trials,
        pruner_warmup_steps=args.pruner_warmup_steps,
        num_sanity_val_steps=args.num_sanity_val_steps,
        fixed_params=_fixed_params(args),
    )


def study_root(cache_root: Path, config: HpoConfig) -> Path:
    return cache_root / "optuna" / f"{config.dataset}_{config.task}" / _slugify(config.study_name)


def storage_uri(study_dir: Path) -> str:
    return f"sqlite:///{(study_dir / 'optuna.sqlite3').resolve()}"


def _fixed_params(args: Any) -> dict[str, Any]:
    return {
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "channels": args.channels,
        "aggr": args.aggr,
        "num_layers": args.num_layers,
        "num_neighbors": args.num_neighbors,
        "temporal_strategy": args.temporal_strategy,
        "max_steps_per_epoch": args.max_steps_per_epoch,
    }


def _slugify(value: str) -> str:
    chars = []
    for char in value.lower():
        if char.isalnum():
            chars.append(char)
        elif chars and chars[-1] != "-":
            chars.append("-")
    return "".join(chars).strip("-") or "study"
