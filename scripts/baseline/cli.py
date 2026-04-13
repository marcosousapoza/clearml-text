import argparse
import json

from .core import configure_baseline_environment, evaluate_dataset
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
        evaluate_dataset(
            dataset_name,
            args.task,
            seed=args.seed,
        )
        for dataset_name in dataset_names
    ]
    summary = summaries[0] if len(summaries) == 1 else {"seed": args.seed, "datasets": summaries}
    print(json.dumps(summary, indent=2, sort_keys=True))
