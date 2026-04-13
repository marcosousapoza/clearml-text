import argparse
import os

from .config import build_config
from scripts.lightning.warnings import configure_training_warnings


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local Optuna HPO for Lightning GNN entity tasks.")
    subparsers = parser.add_subparsers(dest="action", required=True)

    run_parser = subparsers.add_parser("run", help="Run an Optuna HPO study.")
    _add_common_args(run_parser)
    run_parser.add_argument("--n-trials", type=_positive_int, default=20)
    run_parser.add_argument("--timeout", type=float, default=None, help="Optional HPO wall-clock limit in seconds.")
    _add_fixed_param_args(run_parser)

    delete_parser = subparsers.add_parser("delete-study", help="Delete an Optuna study from local storage.")
    _add_common_args(delete_parser)
    delete_parser.add_argument("--n-trials", type=_positive_int, default=1)
    delete_parser.add_argument("--timeout", type=float, default=None)
    _add_fixed_param_args(delete_parser)

    return parser.parse_args()


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--study-name", default=None, help="Defaults to '<dataset>:<task>'.")
    parser.add_argument("--storage-backend", choices=("journal", "sqlite"), default="journal")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="32-true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-sanity-val-steps", type=int, default=0)
    parser.add_argument("--pruner-startup-trials", type=_non_negative_int, default=5)
    parser.add_argument("--pruner-warmup-steps", type=_non_negative_int, default=5)


def _add_fixed_param_args(parser: argparse.ArgumentParser) -> None:
    fixed = parser.add_argument_group("fixed hyperparameter overrides")
    fixed.add_argument("--lr", type=float, default=None)
    fixed.add_argument("--epochs", type=int, default=None)
    fixed.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    fixed.add_argument("--channels", type=int, default=None)
    fixed.add_argument("--aggr", choices=("sum", "mean", "max"), default=None)
    fixed.add_argument("--num-layers", dest="num_layers", type=int, default=None)
    fixed.add_argument("--num-neighbors", dest="num_neighbors", type=int, default=None)
    fixed.add_argument("--temporal-strategy", dest="temporal_strategy", choices=("last", "uniform"), default=None)
    fixed.add_argument("--max-steps-per-epoch", dest="max_steps_per_epoch", type=int, default=None)


def main() -> None:
    configure_training_warnings()
    args = parse_args()
    config = build_config(args)

    if config.action == "run":
        if config.accelerator == "cpu":
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        from .optuna_ops import run_hpo

        run_hpo(config)
        return

    if config.action == "delete-study":
        if config.accelerator == "cpu":
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        from .storage import delete_study

        delete_study(config)
        return

    raise SystemExit(f"[hpo] unsupported action: {config.action}")
