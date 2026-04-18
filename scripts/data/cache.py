import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts
from data.cache import configure_cache_environment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build or clear RelBench dataset/task caches for this project. "
            "Caches default to CACHE_ROOT from the environment or repo .env."
        )
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help=(
            "Dataset name to cache or clear. Repeat the flag to select multiple datasets. "
            "Defaults to all registered datasets when omitted."
        ),
    )
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Task name to cache or clear. Repeat the flag to select multiple tasks.",
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Cache or clear all registered tasks for each selected dataset.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Use RelBench download mode instead of computing caches locally.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Optional cache root override. Defaults to CACHE_ROOT from .env.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List each selected dataset's registered tasks and exit.",
    )
    parser.add_argument(
        "--clear-db",
        action="store_true",
        help="Remove the cached database at <cache-root>/<dataset>/db for each selected dataset.",
    )
    parser.add_argument(
        "--clear-tasks",
        action="store_true",
        help=(
            "Remove cached task tables. With --task, clears only those tasks. "
            "Without --task, clears all tasks for each selected dataset."
        ),
    )
    parser.add_argument(
        "--clear-all",
        action="store_true",
        help="Remove the entire cache directory for each selected dataset.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help=(
            "Remove the processed cache directory for each selected dataset, then "
            "rebuild the dataset database and all registered tasks. Raw OCEL data "
            "is reused when present and downloaded only if missing."
        ),
    )
    return parser


def remove_path(path: Path) -> None:
    if not path.exists():
        print(f"Cache not found, skipping: {path}")
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    print(f"Removed cache: {path}")


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.datasets = args.dataset or []
    delattr(args, "dataset")

    if args.task and len(args.datasets) > 1:
        raise ValueError("--task can only be used with a single --dataset.")
    if args.download and args.rebuild:
        raise ValueError("--download cannot be combined with --rebuild.")

    return args


def resolve_task_names(args: argparse.Namespace, dataset_name: str, registered_tasks: list[str]) -> list[str]:
    if args.rebuild:
        return registered_tasks
    if args.all_tasks:
        return registered_tasks
    if not args.task:
        return []

    unknown = sorted(set(args.task) - set(registered_tasks))
    if unknown:
        raise ValueError(
            f"Unknown task(s) for dataset {dataset_name!r}: {', '.join(unknown)}"
        )
    return args.task


def process_dataset(args: argparse.Namespace, dataset_name: str, cache_root: Path) -> None:
    dataset_dir = cache_root / dataset_name

    from relbench.datasets import get_dataset
    from relbench.tasks import get_task, get_task_names

    registered_tasks = get_task_names(dataset_name)

    if args.list:
        print(f"Dataset: {dataset_name}")
        print(f"Cache root: {cache_root}")
        if registered_tasks:
            print("Registered tasks:")
            for task_name in registered_tasks:
                print(f"  - {task_name}")
        else:
            print("Registered tasks: none")
        return

    task_names = resolve_task_names(args, dataset_name, registered_tasks)
    if args.clear_all:
        remove_path(dataset_dir)
        return

    if args.rebuild:
        remove_path(dataset_dir)

    if args.clear_db:
        remove_path(dataset_dir / "db")
    if args.clear_tasks:
        paths = (
            [dataset_dir / "tasks" / task_name for task_name in task_names]
            if task_names
            else [dataset_dir / "tasks"]
        )
        for path in paths:
            remove_path(path)
    if args.clear_db or args.clear_tasks:
        return

    cache_root.mkdir(parents=True, exist_ok=True)
    dataset = get_dataset(dataset_name, download=args.download)
    db = dataset.get_db()
    print(
        f"Cached dataset {dataset_name!r} with tables: "
        f"{', '.join(sorted(db.table_dict.keys()))}"
    )

    for task_name in task_names:
        task = get_task(dataset_name, task_name, download=args.download)
        for split in ["train", "val", "test"]:
            table = task.get_table(split)
            print(
                f"Cached task {task_name!r} split {split!r} "
                f"with {len(table.df)} rows."
            )

    print(f"Dataset cache root: {dataset_dir}")
    print(f"Raw OCEL cache root: {cache_root / 'raw_ocel'}")
    if task_names:
        print(f"Task cache root: {dataset_dir / 'tasks'}")


def main() -> None:
    scripts.load_env()
    args = normalize_args(build_parser().parse_args())
    cache_root = configure_cache_environment(args.cache_root)

    from data.dataset import DATASET_NAMES
    import task  # noqa: F401

    available_datasets = list(DATASET_NAMES)
    if not args.datasets:
        args.datasets = sorted(available_datasets)

    unknown = sorted(set(args.datasets) - set(available_datasets))
    if unknown:
        raise ValueError(
            f"Unknown dataset(s): {', '.join(unknown)}. Available datasets: "
            f"{', '.join(sorted(available_datasets))}"
        )

    for index, dataset_name in enumerate(args.datasets):
        if index:
            print()
        process_dataset(args, dataset_name, cache_root)


if __name__ == "__main__":
    main()
