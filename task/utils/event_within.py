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
