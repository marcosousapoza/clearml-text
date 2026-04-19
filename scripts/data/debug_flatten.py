import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

import scripts

scripts.load_env()

from data.cache import configure_cache_environment
from data.const import OBJECT_TABLE, TIME_COL
from data.dataset import register_all_datasets
from data.flat import flatten as flatten_db
from data.graph import make_ocel_graph
from relbench.base import Database, Table
from relbench.datasets import get_dataset
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task
from task import register_tasks
from task.utils import MEntityTask


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Construct and inspect flattened OCEL tables for a dataset/task."
    )
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--task", required=True, type=str)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument(
        "--write-dir",
        type=Path,
        default=None,
        help="Optional directory to write flattened tables as parquet files.",
    )
    parser.add_argument(
        "--skip-reindex",
        action="store_true",
        help="Do not reindex pkeys/fkeys after flattening.",
    )
    parser.add_argument(
        "--build-graph",
        action="store_true",
        help="Also build the hetero graph and validate temporal edge ordering.",
    )
    return parser


def capture_pkey_mapping(db: Database) -> dict[str, pd.Series]:
    original_pkeys: dict[str, pd.Series] = {}
    for table_name, table in db.table_dict.items():
        if table.pkey_col is None:
            continue
        original_pkeys[table_name] = table.df[table.pkey_col].copy()

    db.reindex_pkeys_and_fkeys()

    mappings: dict[str, pd.Series] = {}
    for table_name, old_values in original_pkeys.items():
        table = db.table_dict[table_name]
        assert table.pkey_col is not None
        new_values = table.df[table.pkey_col]
        mappings[table_name] = pd.Series(new_values.to_numpy(), index=old_values.to_numpy())
    return mappings


def summarize_table(name: str, table: Table) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "rows": len(table.df),
        "columns": list(table.df.columns),
        "pkey_col": table.pkey_col,
        "time_col": table.time_col,
        "fkeys": dict(table.fkey_col_to_pkey_table),
    }
    if table.pkey_col is not None:
        summary["duplicate_pkeys"] = int(table.df[table.pkey_col].duplicated().sum())
    if table.time_col is not None and table.time_col in table.df.columns:
        summary["null_times"] = int(table.df[table.time_col].isna().sum())
        summary["min_time"] = str(table.df[table.time_col].min())
        summary["max_time"] = str(table.df[table.time_col].max())
    return summary


def validate_foreign_keys(db: Database) -> list[str]:
    errors: list[str] = []
    for table_name, table in db.table_dict.items():
        for fkey_col, target_table_name in table.fkey_col_to_pkey_table.items():
            if fkey_col not in table.df.columns:
                errors.append(f"{table_name}.{fkey_col}: missing foreign-key column")
                continue
            target_table = db.table_dict[target_table_name]
            if target_table.pkey_col is None:
                errors.append(f"{target_table_name}: missing primary key for FK validation")
                continue
            missing = pd.Index(table.df[fkey_col].dropna().unique()).difference(
                pd.Index(target_table.df[target_table.pkey_col].dropna().unique())
            )
            if not missing.empty:
                errors.append(
                    f"{table_name}.{fkey_col} -> {target_table_name}.{target_table.pkey_col}: "
                    f"{len(missing)} dangling refs, e.g. {missing[:5].tolist()}"
                )
    return errors


def validate_task_entity_ids(
    task: MEntityTask,
    flat_db: Database,
    pkey_mappings: dict[str, pd.Series],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        split_df = task.get_table(split, mask_input_cols=False).df
        split_result: dict[str, Any] = {}
        for entity_col, entity_table in zip(task.entity_cols, task.entity_tables):
            table = flat_db.table_dict[entity_table]
            assert table.pkey_col is not None
            existing_ids = pd.Index(table.df[table.pkey_col].dropna().unique())
            raw_ids = pd.Index(split_df[entity_col].dropna().unique())
            raw_missing = raw_ids.difference(existing_ids)
            mapped = split_df[entity_col].map(pkey_mappings.get(entity_table, pd.Series(dtype="int64")))
            mapped_missing = int(mapped.isna().sum()) if entity_table in pkey_mappings else 0
            split_result[entity_col] = {
                "unique_ids": int(raw_ids.size),
                "raw_missing_after_flatten": int(raw_missing.size),
                "raw_missing_examples": raw_missing[:5].tolist(),
                "mapped_missing_after_reindex": mapped_missing,
            }
        out[split] = split_result
    return out


def validate_temporal_edge_order(data) -> list[str]:
    issues: list[str] = []
    for edge_type in data.edge_types:
        edge_index = data[edge_type].edge_index
        if edge_index.numel() == 0:
            continue
        src_type, _, _ = edge_type
        if "time" not in data[src_type]:
            continue

        row, col = edge_index
        src_time = data[src_type].time[row]

        if bool((col[1:] < col[:-1]).any()):
            issues.append(f"{edge_type}: destinations are not globally nondecreasing")
            continue

        start = 0
        for i in range(1, col.numel() + 1):
            if i == col.numel() or col[i] != col[start]:
                local_times = src_time[start:i]
                if local_times.numel() > 1 and bool((local_times[1:] < local_times[:-1]).any()):
                    issues.append(
                        f"{edge_type}: destination {int(col[start])} has non-monotone source times "
                        f"{local_times[:8].tolist()}"
                    )
                    break
                start = i
    return issues


def write_tables(db: Database, write_dir: Path) -> None:
    write_dir.mkdir(parents=True, exist_ok=True)
    for table_name, table in db.table_dict.items():
        table.df.to_parquet(write_dir / f"{table_name}.parquet", index=False)


def main() -> None:
    args = build_parser().parse_args()

    cache_root = configure_cache_environment(args.cache_dir)
    register_all_datasets(cache_root)
    register_tasks(cache_root)

    dataset = get_dataset(args.dataset, download=False)
    task = get_task(args.dataset, args.task, download=False)
    if not isinstance(task, MEntityTask):
        raise TypeError(f"Task {args.task!r} is not an MEntityTask.")

    flat_db = flatten_db(dataset.get_db(upto_test_timestamp=False), task.object_types)
    pkey_mappings = {} if args.skip_reindex else capture_pkey_mapping(flat_db)

    report: dict[str, Any] = {
        "dataset": args.dataset,
        "task": args.task,
        "object_types": list(task.object_types),
        "reindexed": not args.skip_reindex,
        "tables": {
            table_name: summarize_table(table_name, table)
            for table_name, table in flat_db.table_dict.items()
        },
        "foreign_key_errors": validate_foreign_keys(flat_db),
        "task_entity_id_check": validate_task_entity_ids(task, flat_db, pkey_mappings),
    }

    if args.build_graph:
        col_to_stype_dict = get_stype_proposal(flat_db)
        col_to_stype_dict = {
            table_name: {
                col: col_to_stype[col]
                for col in flat_db.table_dict[table_name].df.columns
                if col in col_to_stype
            }
            for table_name, col_to_stype in col_to_stype_dict.items()
            if table_name in flat_db.table_dict
        }
        if hasattr(dataset, "set_stype"):
            col_to_stype_dict = dataset.set_stype(col_to_stype_dict)  # type: ignore[attr-defined]
        data, _ = make_ocel_graph(flat_db, col_to_stype_dict=col_to_stype_dict, cache_dir=None)
        report["graph"] = {
            "node_types": list(data.node_types),
            "edge_types": [list(edge_type) for edge_type in data.edge_types],
            "temporal_edge_order_issues": validate_temporal_edge_order(data),
        }

    if args.write_dir is not None:
        write_tables(flat_db, args.write_dir)
        report["write_dir"] = str(args.write_dir.resolve())

    object_table = flat_db.table_dict[OBJECT_TABLE]
    if object_table.pkey_col is not None:
        report["object_id_density"] = {
            "min": int(object_table.df[object_table.pkey_col].min()),
            "max": int(object_table.df[object_table.pkey_col].max()),
            "num_objects": int(len(object_table.df)),
        }
    if TIME_COL in flat_db.table_dict[OBJECT_TABLE].df.columns:
        report["object_time_nulls"] = int(flat_db.table_dict[OBJECT_TABLE].df[TIME_COL].isna().sum())

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
