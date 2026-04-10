"""Influence diagnostics for long-running objects in the event-object graph."""

import duckdb
import pandas as pd
from relbench.base.database import Database

from data.const import E2O_TABLE, EVENT_ID_COL, OBJECT_ID_COL, OBJECT_TABLE, OBJECT_TYPE_COL
from data.wrapper import check_dbs


@check_dbs
def compute_object_trace_lengths(db: Database) -> pd.DataFrame:
    """Compute the number of distinct linked events for each object.

    The trace length is defined purely from the e2o relation as the count of
    distinct event ids attached to each object id.
    """

    e2o_df = db.table_dict[E2O_TABLE].df[[EVENT_ID_COL, OBJECT_ID_COL]].dropna(
        subset=[EVENT_ID_COL, OBJECT_ID_COL]
    )
    con = duckdb.connect()
    try:
        con.register("e2o_df", e2o_df)
        lengths = con.execute(
            f"""
            SELECT
                {OBJECT_ID_COL} AS object_id,
                COUNT(DISTINCT {EVENT_ID_COL}) AS object_trace_length
            FROM e2o_df
            GROUP BY {OBJECT_ID_COL}
            ORDER BY object_trace_length, object_id
            """
        ).df()
    finally:
        con.close()
    return lengths


@check_dbs
def event_share_by_min_object_trace_length(db: Database) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure event and object coverage as the minimum object length rises.

    An event is covered at threshold ``k`` if it participates with at least one
    object whose total trace length is at least ``k``. An object is covered at
    threshold ``k`` if its own trace length is at least ``k``. The full curves
    are derived from reverse cumulative histograms after one event-side and one
    object-side aggregation.
    """

    object_lengths = compute_object_trace_lengths(db)
    empty = pd.DataFrame(
        columns=[
            "min_object_trace_length",
            "event_count",
            "total_event_count",
            "event_share",
            "event_share_pct",
            "object_count",
            "total_object_count",
            "object_share",
            "object_share_pct",
        ]
    )
    if object_lengths.empty:
        return object_lengths, empty

    e2o_df = db.table_dict[E2O_TABLE].df[[EVENT_ID_COL, OBJECT_ID_COL]].dropna(
        subset=[EVENT_ID_COL, OBJECT_ID_COL]
    )
    con = duckdb.connect()
    try:
        con.register("e2o_df", e2o_df)
        con.register("object_lengths", object_lengths)
        event_max_lengths = con.execute(
            f"""
            SELECT
                e.{EVENT_ID_COL} AS event_id,
                MAX(o.object_trace_length) AS max_object_trace_length
            FROM e2o_df AS e
            INNER JOIN object_lengths AS o
                ON e.{OBJECT_ID_COL} = o.object_id
            GROUP BY e.{EVENT_ID_COL}
            ORDER BY event_id
            """
        ).df()
    finally:
        con.close()

    if event_max_lengths.empty:
        return object_lengths, empty

    total_event_count = int(len(event_max_lengths))
    total_object_count = int(len(object_lengths))
    max_length = int(
        max(
            event_max_lengths["max_object_trace_length"].max(),
            object_lengths["object_trace_length"].max(),
        )
    )

    event_histogram = (
        event_max_lengths["max_object_trace_length"]
        .value_counts(sort=False)
        .rename_axis("trace_length")
        .reset_index(name="num_events")
    )
    event_histogram["trace_length"] = event_histogram["trace_length"].astype(int)
    event_counts_by_length = dict(zip(event_histogram["trace_length"], event_histogram["num_events"]))

    object_histogram = (
        object_lengths["object_trace_length"]
        .value_counts(sort=False)
        .rename_axis("trace_length")
        .reset_index(name="num_objects")
    )
    object_histogram["trace_length"] = object_histogram["trace_length"].astype(int)
    object_counts_by_length = dict(zip(object_histogram["trace_length"], object_histogram["num_objects"]))

    rows = []
    running_event_count = 0
    running_object_count = 0
    for min_length in range(max_length, -1, -1):
        running_event_count += int(event_counts_by_length.get(min_length, 0))
        running_object_count += int(object_counts_by_length.get(min_length, 0))
        event_share = (running_event_count / total_event_count) if total_event_count else 0.0
        object_share = (running_object_count / total_object_count) if total_object_count else 0.0
        rows.append(
            {
                "min_object_trace_length": min_length,
                "event_count": int(running_event_count),
                "total_event_count": int(total_event_count),
                "event_share": float(event_share),
                "event_share_pct": float(event_share * 100.0),
                "object_count": int(running_object_count),
                "total_object_count": int(total_object_count),
                "object_share": float(object_share),
                "object_share_pct": float(object_share * 100.0),
            }
        )

    influence = (
        pd.DataFrame(rows)
        .sort_values("min_object_trace_length", kind="stable")
        .reset_index(drop=True)
    )
    return object_lengths, influence

@check_dbs
def object_share_above_trace_length(db: Database) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure the share of objects whose trace length is strictly above a threshold."""

    object_lengths = compute_object_trace_lengths(db)
    empty = pd.DataFrame(
        columns=[
            "trace_length_threshold",
            "object_count_above_threshold",
            "total_object_count",
            "object_share_above_threshold",
            "object_share_above_threshold_pct",
        ]
    )
    if object_lengths.empty:
        return object_lengths, empty

    total_object_count = int(len(object_lengths))
    max_length = int(object_lengths["object_trace_length"].max())
    object_histogram = (
        object_lengths["object_trace_length"]
        .value_counts(sort=False)
        .rename_axis("trace_length")
        .reset_index(name="num_objects")
    )
    object_histogram["trace_length"] = object_histogram["trace_length"].astype(int)
    object_counts_by_length = dict(zip(object_histogram["trace_length"], object_histogram["num_objects"]))

    rows = []
    excluded_object_count = 0
    for threshold in range(0, max_length + 1):
        excluded_object_count += int(object_counts_by_length.get(threshold, 0))
        object_count_above_threshold = total_object_count - excluded_object_count
        object_share_above_threshold = (
            object_count_above_threshold / total_object_count if total_object_count else 0.0
        )
        rows.append(
            {
                "trace_length_threshold": threshold,
                "object_count_above_threshold": int(object_count_above_threshold),
                "total_object_count": int(total_object_count),
                "object_share_above_threshold": float(object_share_above_threshold),
                "object_share_above_threshold_pct": float(object_share_above_threshold * 100.0),
            }
        )

    object_share = pd.DataFrame(rows)
    return object_lengths, object_share


@check_dbs
def object_type_occupation_above_trace_length(db: Database) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure object-type composition among objects longer than each threshold."""

    object_lengths = compute_object_trace_lengths(db)
    empty = pd.DataFrame(
        columns=[
            "trace_length_threshold",
            "object_type",
            "object_count_above_threshold",
            "total_object_count_above_threshold",
            "type_share_within_threshold",
            "type_share_within_threshold_pct",
            "global_object_share_pct",
        ]
    )
    if object_lengths.empty:
        return object_lengths, empty

    object_df = db.table_dict[OBJECT_TABLE].df[[OBJECT_ID_COL, OBJECT_TYPE_COL]].dropna(
        subset=[OBJECT_ID_COL]
    )
    object_df[OBJECT_TYPE_COL] = object_df[OBJECT_TYPE_COL].astype(str)

    typed_lengths = object_lengths.merge(
        object_df,
        how="left",
        left_on="object_id",
        right_on=OBJECT_ID_COL,
    )
    typed_lengths[OBJECT_TYPE_COL] = typed_lengths[OBJECT_TYPE_COL].fillna("unknown")

    total_object_count = int(len(typed_lengths))
    max_length = int(typed_lengths["object_trace_length"].max())
    type_totals = typed_lengths[OBJECT_TYPE_COL].value_counts(sort=False).to_dict()

    counts_by_length_and_type = (
        typed_lengths.groupby(["object_trace_length", OBJECT_TYPE_COL], dropna=False, observed=False)
        .size()
        .rename("num_objects")
        .reset_index()
    )
    counts_by_length_and_type["object_trace_length"] = counts_by_length_and_type["object_trace_length"].astype(int)

    excluded_type_counts: dict[str, int] = {str(object_type): 0 for object_type in type_totals}
    rows = []
    for threshold in range(0, max_length + 1):
        threshold_slice = counts_by_length_and_type[
            counts_by_length_and_type["object_trace_length"] == threshold
        ]
        for record in threshold_slice.itertuples(index=False):
            excluded_type_counts[str(record[1])] = excluded_type_counts.get(str(record[1]), 0) + int(
                record[2]
            )

        remaining_type_counts = {
            str(object_type): int(type_totals[object_type]) - int(excluded_type_counts.get(str(object_type), 0))
            for object_type in type_totals
        }
        total_remaining = int(sum(remaining_type_counts.values()))
        global_object_share = (total_remaining / total_object_count) if total_object_count else 0.0
        for object_type in sorted(type_totals):
            type_count = int(remaining_type_counts[object_type])
            within_threshold_share = (type_count / total_remaining) if total_remaining else 0.0
            rows.append(
                {
                    "trace_length_threshold": threshold,
                    "object_type": str(object_type),
                    "object_count_above_threshold": type_count,
                    "total_object_count_above_threshold": total_remaining,
                    "type_share_within_threshold": float(within_threshold_share),
                    "type_share_within_threshold_pct": float(within_threshold_share * 100.0),
                    "global_object_share_pct": float(global_object_share * 100.0),
                }
            )

    occupation = pd.DataFrame(rows)
    return typed_lengths, occupation


__all__ = [
    "compute_object_trace_lengths",
    "event_share_by_min_object_trace_length",
    "object_share_above_trace_length",
    "object_type_occupation_above_trace_length",
]
