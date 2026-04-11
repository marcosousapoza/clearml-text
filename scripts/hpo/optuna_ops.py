import json
import shutil
import warnings
from pathlib import Path
from typing import Any

import lightning as L
import optuna
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from relbench.base import TaskType
from relbench.tasks import get_task
from torch import Tensor
from torch_geometric.seed import seed_everything
from unique_names_generator import get_random_name

from data.cache import configure_cache_environment
from data.dataset import register_all_datasets
from scripts.lightning.data import RelbenchLightningDataModule
from scripts.lightning.module import EntityGNNLightningModule
from task import register_tasks

from .config import HpoConfig, storage_uri, study_root
from .search_space import TrialParams, suggest_trial_params


class GracefulOptunaPruningCallback(Callback):
    def __init__(self, trial: optuna.Trial, monitor: str) -> None:
        super().__init__()
        self.trial = trial
        self.monitor = monitor
        self.should_prune = False
        self.pruned_epoch: int | None = None

    def on_validation_end(self, trainer: Trainer, pl_module: L.LightningModule) -> None:
        if trainer.sanity_checking:
            return

        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            warnings.warn(
                f"The metric {self.monitor!r} is not in Lightning callback_metrics for pruning.",
                stacklevel=2,
            )
            return

        epoch = int(pl_module.current_epoch)
        should_prune = False
        if trainer.is_global_zero:
            self.trial.report(_metric_value(current_score), step=epoch)
            should_prune = self.trial.should_prune()

        should_prune = bool(trainer.strategy.broadcast(should_prune))
        if not should_prune:
            return

        self.should_prune = True
        self.pruned_epoch = epoch
        trainer.should_stop = True


def run_hpo(config: HpoConfig) -> None:
    cache_root = configure_cache_environment(config.cache_dir)
    study_dir = study_root(cache_root, config)
    study_dir.mkdir(parents=True, exist_ok=True)

    register_all_datasets()
    register_tasks()
    task = get_task(config.dataset, config.task, download=False)
    direction = "minimize" if task.task_type == TaskType.REGRESSION else "maximize"
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=config.pruner_startup_trials,
        n_warmup_steps=config.pruner_warmup_steps,
    )
    study = optuna.create_study(
        study_name=config.study_name,
        storage=storage_uri(study_dir),
        direction=direction,
        pruner=pruner,
        load_if_exists=True,
    )

    print(f"[hpo] study={config.study_name!r} direction={direction} storage={storage_uri(study_dir)}")
    print(f"[hpo] outputs={study_dir}")

    objective = _Objective(config=config, study_dir=study_dir, direction=direction)
    study.optimize(objective, n_trials=config.n_trials, timeout=config.timeout, n_jobs=1)

    try:
        best_trial = study.best_trial
    except ValueError:
        print("[hpo] no completed trials")
    else:
        print(
            "[hpo] best completed trial="
            f"{best_trial.number} value={study.best_value} "
            f"checkpoint={study.user_attrs.get('best_checkpoint_path')}"
        )


class _Objective:
    def __init__(self, *, config: HpoConfig, study_dir: Path, direction: str) -> None:
        self.config = config
        self.study_dir = study_dir
        self.direction = direction

    def __call__(self, trial: optuna.Trial) -> float:
        params = suggest_trial_params(trial, self.config.fixed_params)
        trial_dir = self.study_dir / f"trial_{trial.number:04d}_{_slugify(get_random_name())}"
        checkpoint_dir = trial_dir / "checkpoints"
        trial_dir.mkdir(parents=True, exist_ok=True)
        _write_json(trial_dir / "params.json", params.to_dict())

        seed_everything(self.config.seed + trial.number)
        if torch.cuda.is_available():
            torch.set_num_threads(1)

        datamodule = RelbenchLightningDataModule(
            dataset_name=self.config.dataset,
            task_name=self.config.task,
            batch_size=params.batch_size,
            num_layers=params.num_layers,
            num_neighbors=params.num_neighbors,
            temporal_strategy=params.temporal_strategy,
            num_workers=self.config.num_workers,
            cache_dir=self.config.cache_dir,
        )
        datamodule.setup("fit")
        assert datamodule.artifacts is not None

        module = EntityGNNLightningModule(
            task=datamodule.artifacts.task,
            data=datamodule.artifacts.data,
            col_stats_dict=datamodule.artifacts.col_stats_dict,
            split_inputs=datamodule.artifacts.split_inputs,
            task_node_type=datamodule.artifacts.task_node_type,
            num_layers=params.num_layers,
            channels=params.channels,
            aggr=params.aggr,
            lr=params.lr,
            epochs=params.epochs,
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="best",
            monitor=module.checkpoint_monitor,
            mode=module.checkpoint_mode,
            save_top_k=1,
            save_last=True,
        )
        pruning_callback = GracefulOptunaPruningCallback(trial, module.checkpoint_monitor)
        logger = CSVLogger(save_dir=str(trial_dir), name="logs", version="")
        trainer = Trainer(
            accelerator=self.config.accelerator,
            devices=_parse_devices(self.config.devices),
            precision=self.config.precision, # type: ignore
            max_epochs=params.epochs,
            limit_train_batches=params.max_steps_per_epoch,
            num_sanity_val_steps=self.config.num_sanity_val_steps,
            default_root_dir=str(trial_dir),
            logger=logger,
            callbacks=[checkpoint_callback, pruning_callback],
        )

        status = "complete"
        trainer.fit(module, datamodule=datamodule)

        best_path = checkpoint_callback.best_model_path or ""
        test_metrics = {}
        if best_path:
            results = trainer.test(module, datamodule=datamodule, ckpt_path=best_path)
            test_metrics = {key: float(value) for key, value in (results[0] if results else {}).items()}
        score = checkpoint_callback.best_model_score
        if score is None:
            raise RuntimeError("[hpo] no validation checkpoint score was recorded")
        val_score = _metric_value(score)

        metrics = {
            "status": "pruned" if pruning_callback.should_prune else status,
            "monitor": module.checkpoint_monitor,
            "mode": module.checkpoint_mode,
            "best_validation_score": val_score,
            "evaluated_checkpoint_path": best_path or None,
            "retained_checkpoint_path": None,
            "checkpoint_retained": False,
            "pruned_epoch": pruning_callback.pruned_epoch,
            "test": test_metrics,
        }

        if pruning_callback.should_prune:
            _delete_dir(checkpoint_dir)
            _write_json(trial_dir / "metrics.json", metrics)
            _set_trial_attrs(trial, params, trial_dir, metrics)
            raise optuna.TrialPruned(f"Trial was pruned at epoch {pruning_callback.pruned_epoch}.")

        checkpoint_retained = self._retain_completed_best_checkpoint(
            trial,
            val_score,
            best_path,
            checkpoint_dir,
            trial_dir,
        )
        metrics["checkpoint_retained"] = checkpoint_retained
        metrics["retained_checkpoint_path"] = best_path if checkpoint_retained and best_path else None
        _write_json(trial_dir / "metrics.json", metrics)
        _set_trial_attrs(trial, params, trial_dir, metrics)
        return val_score

    def _retain_completed_best_checkpoint(
        self,
        trial: optuna.Trial,
        val_score: float,
        best_path: str,
        checkpoint_dir: Path,
        trial_dir: Path,
    ) -> bool:
        study = trial.study
        previous_best_number = study.user_attrs.get("best_trial_number")
        previous_best_value = study.user_attrs.get("best_validation_score")
        previous_best_dir = study.user_attrs.get("best_checkpoint_dir")
        previous_best_trial_dir = study.user_attrs.get("best_trial_dir")

        if previous_best_value is None:
            is_better = True
        else:
            previous_score = float(previous_best_value)
            is_better = (
                val_score < previous_score
                if self.direction == "minimize"
                else val_score > previous_score
            )
        if is_better:
            if previous_best_dir and previous_best_number != trial.number:
                _delete_dir(Path(previous_best_dir))
                if previous_best_trial_dir:
                    _mark_checkpoint_not_retained(Path(previous_best_trial_dir))
            study.set_user_attr("best_trial_number", trial.number)
            study.set_user_attr("best_validation_score", val_score)
            study.set_user_attr("best_checkpoint_path", best_path or None)
            study.set_user_attr("best_checkpoint_dir", str(checkpoint_dir))
            study.set_user_attr("best_trial_dir", str(trial_dir))
            return True

        _delete_dir(checkpoint_dir)
        return False


def _metric_value(value: Any) -> float:
    if isinstance(value, Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _set_trial_attrs(trial: optuna.Trial, params: TrialParams, trial_dir: Path, metrics: dict[str, Any]) -> None:
    trial.set_user_attr("trial_dir", str(trial_dir))
    trial.set_user_attr("params", params.to_dict())
    for name, value in metrics.items():
        trial.set_user_attr(name, value)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _mark_checkpoint_not_retained(trial_dir: Path) -> None:
    metrics_path = trial_dir / "metrics.json"
    if not metrics_path.exists():
        return
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["checkpoint_retained"] = False
    metrics["retained_checkpoint_path"] = None
    _write_json(metrics_path, metrics)


def _delete_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _parse_devices(value: str) -> str | int | list[int]:
    if value == "auto":
        return value
    if "," in value:
        return [int(item) for item in value.split(",") if item]
    return int(value)


def _slugify(value: str) -> str:
    chars = []
    for char in value.lower():
        if char.isalnum():
            chars.append(char)
        elif chars and chars[-1] != "-":
            chars.append("-")
    return "".join(chars).strip("-") or "trial"
