import duckdb
import pandas as pd
from relbench.base import Database

from data.const import (
    E2O_TABLE,
    EVENT_ID_COL,
    EVENT_TABLE,
    EVENT_TYPE_COL,
    OBJECT_ID_COL,
    OBJECT_TABLE,
    OBJECT_TYPE_COL,
    O2O_DST_COL,
    O2O_SRC_COL,
    TIME_COL,
)
from data.wrapper import check_dbs


@check_dbs
def build_event_within_table(
    db: Database,
    object_type: str,
    event_type: str,
    times: pd.Series,
    delta: pd.Timedelta,
) -> pd.DataFrame:
    event = db.table_dict[EVENT_TABLE].df
    obj = db.table_dict[OBJECT_TABLE].df
    e2o = db.table_dict[E2O_TABLE].df
    times_df = pd.DataFrame({"obs_time": pd.to_datetime(times).sort_values().unique()})
    future_window = f"{int(delta.total_seconds())} seconds"

    con = duckdb.connect()
    con.register("event", event)
    con.register("obj", obj)
    con.register("e2o", e2o)
    con.register("times_df", times_df)
    try:
        return con.execute(
            f"""
            WITH typed_objects AS (
                SELECT {OBJECT_ID_COL}
                FROM obj
                WHERE {OBJECT_TYPE_COL} = '{object_type}'
            ),
            object_events AS (
                SELECT
                    eo.{OBJECT_ID_COL},
                    e.{EVENT_TYPE_COL},
                    e.{TIME_COL}
                FROM e2o eo
                JOIN event e
                  ON e.{EVENT_ID_COL} = eo.{EVENT_ID_COL}
                JOIN typed_objects o
                  ON o.{OBJECT_ID_COL} = eo.{OBJECT_ID_COL}
            ),
            first_seen AS (
                SELECT
                    {OBJECT_ID_COL},
                    MIN({TIME_COL}) AS first_seen_time
                FROM object_events
                GROUP BY {OBJECT_ID_COL}
            ),
            first_target AS (
                SELECT
                    {OBJECT_ID_COL},
                    MIN({TIME_COL}) AS first_target_time
                FROM object_events
                WHERE {EVENT_TYPE_COL} = '{event_type}'
                GROUP BY {OBJECT_ID_COL}
            ),
            candidates AS (
                SELECT
                    t.obs_time AS {TIME_COL},
                    fs.{OBJECT_ID_COL},
                    ft.first_target_time
                FROM times_df t
                JOIN first_seen fs
                  ON fs.first_seen_time <= t.obs_time
                LEFT JOIN first_target ft
                  ON ft.{OBJECT_ID_COL} = fs.{OBJECT_ID_COL}
                WHERE ft.first_target_time IS NULL
                   OR ft.first_target_time > t.obs_time
            )
            SELECT
                {OBJECT_ID_COL},
                {TIME_COL},
                CAST(
                    COALESCE(
                        first_target_time > {TIME_COL}
                        AND first_target_time <= {TIME_COL} + INTERVAL '{future_window}',
                        FALSE
                    )
                    AS INTEGER
                ) AS target
            FROM candidates
            ORDER BY {TIME_COL}, {OBJECT_ID_COL}
            """
        ).df()
    finally:
        con.close()


@check_dbs
def build_pair_event_within_table(
    db: Database,
    object_types: tuple[str, str],
    event_type: str,
    times: pd.Series,
    delta: pd.Timedelta,
) -> pd.DataFrame:
    event = db.table_dict[EVENT_TABLE].df
    obj = db.table_dict[OBJECT_TABLE].df
    e2o = db.table_dict[E2O_TABLE].df
    times_df = pd.DataFrame({"obs_time": pd.to_datetime(times).sort_values().unique()})
    future_window = f"{int(delta.total_seconds())} seconds"

    con = duckdb.connect()
    con.register("event", event)
    con.register("obj", obj)
    con.register("e2o", e2o)
    con.register("times_df", times_df)
    try:
        return con.execute(
            f"""
            WITH linked AS (
                SELECT
                    e.{EVENT_ID_COL},
                    e.{EVENT_TYPE_COL} AS event_type,
                    e.{TIME_COL},
                    o.{OBJECT_TYPE_COL} AS object_type,
                    eo.{OBJECT_ID_COL}
                FROM event e
                JOIN e2o eo
                  ON eo.{EVENT_ID_COL} = e.{EVENT_ID_COL}
                JOIN obj o
                  ON o.{OBJECT_ID_COL} = eo.{OBJECT_ID_COL}
                WHERE o.{OBJECT_TYPE_COL} IN ('{object_types[0]}', '{object_types[1]}')
            ),
            pairs AS (
                SELECT
                    left_link.{OBJECT_ID_COL} AS {O2O_SRC_COL},
                    right_link.{OBJECT_ID_COL} AS {O2O_DST_COL},
                    left_link.event_type,
                    left_link.{TIME_COL}
                FROM linked left_link
                JOIN linked right_link
                  ON right_link.{EVENT_ID_COL} = left_link.{EVENT_ID_COL}
                WHERE left_link.object_type = '{object_types[0]}'
                  AND right_link.object_type = '{object_types[1]}'
                  AND left_link.{OBJECT_ID_COL} <> right_link.{OBJECT_ID_COL}
            ),
            first_seen AS (
                SELECT
                    {O2O_SRC_COL},
                    {O2O_DST_COL},
                    MIN({TIME_COL}) AS first_seen_time
                FROM pairs
                GROUP BY {O2O_SRC_COL}, {O2O_DST_COL}
            ),
            first_target AS (
                SELECT
                    {O2O_SRC_COL},
                    {O2O_DST_COL},
                    MIN({TIME_COL}) AS first_target_time
                FROM pairs
                WHERE event_type = '{event_type}'
                GROUP BY {O2O_SRC_COL}, {O2O_DST_COL}
            ),
            candidates AS (
                SELECT
                    t.obs_time AS {TIME_COL},
                    fs.{O2O_SRC_COL},
                    fs.{O2O_DST_COL},
                    ft.first_target_time
                FROM times_df t
                JOIN first_seen fs
                  ON fs.first_seen_time <= t.obs_time
                LEFT JOIN first_target ft
                  ON ft.{O2O_SRC_COL} = fs.{O2O_SRC_COL}
                 AND ft.{O2O_DST_COL} = fs.{O2O_DST_COL}
                WHERE ft.first_target_time IS NULL
                   OR ft.first_target_time > t.obs_time
            )
            SELECT
                {O2O_SRC_COL},
                {O2O_DST_COL},
                {TIME_COL},
                CAST(
                    COALESCE(
                        first_target_time > {TIME_COL}
                        AND first_target_time <= {TIME_COL} + INTERVAL '{future_window}',
                        FALSE
                    )
                    AS INTEGER
                ) AS target
            FROM candidates
            ORDER BY {TIME_COL}, {O2O_SRC_COL}, {O2O_DST_COL}
            """
        ).df()
    finally:
        con.close()


@check_dbs
def build_complete_pair_event_within_table(
    db: Database,
    object_types: tuple[str, str],
    event_type: str,
    times: pd.Series,
    delta: pd.Timedelta,
) -> pd.DataFrame:
    event = db.table_dict[EVENT_TABLE].df
    obj = db.table_dict[OBJECT_TABLE].df
    e2o = db.table_dict[E2O_TABLE].df
    times_df = pd.DataFrame({"obs_time": pd.to_datetime(times).sort_values().unique()})
    future_window = f"{int(delta.total_seconds())} seconds"

    con = duckdb.connect()
    con.register("event", event)
    con.register("obj", obj)
    con.register("e2o", e2o)
    con.register("times_df", times_df)
    try:
        return con.execute(
            f"""
            WITH src_objects AS (
                SELECT {OBJECT_ID_COL} AS {O2O_SRC_COL}
                FROM obj
                WHERE {OBJECT_TYPE_COL} = '{object_types[0]}'
            ),
            dst_objects AS (
                SELECT {OBJECT_ID_COL} AS {O2O_DST_COL}
                FROM obj
                WHERE {OBJECT_TYPE_COL} = '{object_types[1]}'
            ),
            target_events AS (
                SELECT
                    src_eo.{OBJECT_ID_COL} AS {O2O_SRC_COL},
                    dst_eo.{OBJECT_ID_COL} AS {O2O_DST_COL},
                    e.{TIME_COL}
                FROM event e
                JOIN e2o src_eo
                  ON src_eo.{EVENT_ID_COL} = e.{EVENT_ID_COL}
                JOIN obj src_obj
                  ON src_obj.{OBJECT_ID_COL} = src_eo.{OBJECT_ID_COL}
                 AND src_obj.{OBJECT_TYPE_COL} = '{object_types[0]}'
                JOIN e2o dst_eo
                  ON dst_eo.{EVENT_ID_COL} = e.{EVENT_ID_COL}
                JOIN obj dst_obj
                  ON dst_obj.{OBJECT_ID_COL} = dst_eo.{OBJECT_ID_COL}
                 AND dst_obj.{OBJECT_TYPE_COL} = '{object_types[1]}'
                WHERE e.{EVENT_TYPE_COL} = '{event_type}'
                  AND src_eo.{OBJECT_ID_COL} <> dst_eo.{OBJECT_ID_COL}
            ),
            candidates AS (
                SELECT
                    t.obs_time AS {TIME_COL},
                    src.{O2O_SRC_COL},
                    dst.{O2O_DST_COL}
                FROM times_df t
                CROSS JOIN src_objects src
                CROSS JOIN dst_objects dst
                WHERE src.{O2O_SRC_COL} <> dst.{O2O_DST_COL}
            ),
            labeled AS (
                SELECT
                    c.{O2O_SRC_COL},
                    c.{O2O_DST_COL},
                    c.{TIME_COL},
                    COUNT(t.{TIME_COL}) > 0 AS target
                FROM candidates c
                LEFT JOIN target_events t
                  ON t.{O2O_SRC_COL} = c.{O2O_SRC_COL}
                 AND t.{O2O_DST_COL} = c.{O2O_DST_COL}
                 AND t.{TIME_COL} > c.{TIME_COL}
                 AND t.{TIME_COL} <= c.{TIME_COL} + INTERVAL '{future_window}'
                GROUP BY c.{O2O_SRC_COL}, c.{O2O_DST_COL}, c.{TIME_COL}
            )
            SELECT
                {O2O_SRC_COL},
                {O2O_DST_COL},
                {TIME_COL},
                CAST(target AS INTEGER) AS target
            FROM labeled
            ORDER BY {TIME_COL}, {O2O_SRC_COL}, {O2O_DST_COL}
            """
        ).df()
    finally:
        con.close()
