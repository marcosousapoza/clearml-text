import argparse
import json

from .core import (
    DEFAULT_BASELINE_SEEDS,
    configure_baseline_environment,
    evaluate_dataset_across_seeds,
)
from .tables import print_baseline_tables


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run entity-task baselines or print baseline LaTeX tables."
    )
    parser.add_argument(
        "--tables",
        action="store_true",
        help="Print regression and classification LaTeX tables instead of JSON results.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help=(
            "Dataset to include. Repeat to select multiple datasets. "
            "Defaults to all registered datasets for --tables and order_management otherwise."
        ),
    )
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Task name to evaluate. Repeat to select multiple tasks. Defaults to all registered tasks.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_BASELINE_SEEDS,
        help="Seeds to evaluate for baseline runs. Defaults to 1 2 3 4 5.",
    )
    parser.add_argument("--wandb", action="store_true", help="Also log metrics to Weights & Biases.")
    parser.add_argument("--wandb-project", type=str, default="ocel-ocp")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.tables:
        print_baseline_tables(
            dataset_names=args.dataset or None,
            task_names=args.task or None,
            seed=args.seed,
        )
        return

    configure_baseline_environment(seed=args.seed)
    dataset_names = args.dataset or ["order_management"]
    summaries = [
        evaluate_dataset_across_seeds(
            dataset_name,
            args.task or None,
            seeds=args.seeds,
            wandb_project=args.wandb_project if args.wandb else None,
        )
        for dataset_name in dataset_names
    ]
    summary = summaries[0] if len(summaries) == 1 else {"seeds": args.seeds, "datasets": summaries}
    print(json.dumps(summary, indent=2, sort_keys=True))
