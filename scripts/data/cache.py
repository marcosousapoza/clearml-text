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
    parser.add_argument("--dataset", required=True, help="Dataset name to cache or clear.")
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Task name to cache or clear. Repeat the flag to select multiple tasks.",
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Cache or clear all registered tasks for the dataset.",
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
    parser.add_argument("--list", action="store_true", help="List the dataset's registered tasks and exit.")
    parser.add_argument(
        "--clear-db",
        action="store_true",
        help="Remove the cached database at <cache-root>/<dataset>/db.",
    )
    parser.add_argument(
        "--clear-tasks",
        action="store_true",
        help=(
            "Remove cached task tables. With --task, clears only those tasks. "
            "Without --task, clears all tasks for the dataset."
        ),
    )
    parser.add_argument(
        "--clear-all",
        action="store_true",
        help="Remove the entire dataset cache directory.",
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


def resolve_task_names(args: argparse.Namespace, registered_tasks: list[str]) -> list[str]:
    if args.all_tasks:
        return registered_tasks
    if not args.task:
        return []

    unknown = sorted(set(args.task) - set(registered_tasks))
    if unknown:
        raise ValueError(
            f"Unknown task(s) for dataset {args.dataset!r}: {', '.join(unknown)}"
        )
    return args.task
def main() -> None:
    scripts.load_env()
    args = build_parser().parse_args()
    cache_root = configure_cache_environment(args.cache_root)
    dataset_dir = cache_root / args.dataset

    import data.dataset  # noqa: F401
    import task  # noqa: F401
    from relbench.datasets import get_dataset, get_dataset_names
    from relbench.tasks import get_task, get_task_names

    available_datasets = get_dataset_names()
    if args.dataset not in available_datasets:
        raise ValueError(
            f"Unknown dataset {args.dataset!r}. Available datasets: "
            f"{', '.join(sorted(available_datasets))}"
        )

    registered_tasks = get_task_names(args.dataset)
    if args.list:
        print(f"Dataset: {args.dataset}")
        print(f"Cache root: {cache_root}")
        if registered_tasks:
            print("Registered tasks:")
            for task_name in registered_tasks:
                print(f"  - {task_name}")
        else:
            print("Registered tasks: none")
        return

    task_names = resolve_task_names(args, registered_tasks)
    if args.clear_all:
        remove_path(dataset_dir)
        return
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
    dataset = get_dataset(args.dataset, download=args.download)
    db = dataset.get_db()
    print(
        f"Cached dataset {args.dataset!r} with tables: "
        f"{', '.join(sorted(db.table_dict.keys()))}"
    )

    for task_name in task_names:
        task = get_task(args.dataset, task_name, download=args.download)
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


if __name__ == "__main__":
    main()
