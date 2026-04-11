from dataclasses import asdict, dataclass
from typing import Any

import optuna


@dataclass(frozen=True)
class TrialParams:
    lr: float
    epochs: int
    batch_size: int
    channels: int
    aggr: str
    num_layers: int
    num_neighbors: int
    temporal_strategy: str
    max_steps_per_epoch: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def suggest_trial_params(trial: optuna.Trial, overrides: dict[str, Any]) -> TrialParams:
    """Search space for tunable CLI parameters exposed by scripts/gnn_entity.py."""

    return TrialParams(
        lr=_value_or_suggest(overrides, "lr", lambda: trial.suggest_float("lr", 1e-4, 5e-2, log=True)),
        epochs=_value_or_suggest(overrides, "epochs", lambda: trial.suggest_categorical("epochs", [5, 10, 20])),
        batch_size=_value_or_suggest(
            overrides,
            "batch_size",
            lambda: trial.suggest_categorical("batch_size", [128, 256, 512, 1024]),
        ),
        channels=_value_or_suggest(
            overrides,
            "channels",
            lambda: trial.suggest_categorical("channels", [64, 128, 256]),
        ),
        aggr=_value_or_suggest(overrides, "aggr", lambda: trial.suggest_categorical("aggr", ["sum", "mean", "max"])),
        num_layers=_value_or_suggest(overrides, "num_layers", lambda: trial.suggest_int("num_layers", 1, 4)),
        num_neighbors=_value_or_suggest(
            overrides,
            "num_neighbors",
            lambda: trial.suggest_categorical("num_neighbors", [32, 64, 128, 256]),
        ),
        temporal_strategy=_value_or_suggest(
            overrides,
            "temporal_strategy",
            lambda: trial.suggest_categorical("temporal_strategy", ["last", "uniform"]),
        ),
        max_steps_per_epoch=_value_or_suggest(
            overrides,
            "max_steps_per_epoch",
            lambda: trial.suggest_categorical("max_steps_per_epoch", [250, 500, 1000, 2000]),
        ),
    )


def _value_or_suggest(overrides: dict[str, Any], name: str, suggest: Any) -> Any:
    value = overrides.get(name)
    return suggest() if value is None else value
