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
    object_types: tuple[str, str],
    event_type: str,
    times: pd.Series,
    delta: pd.Timedelta,
    lookback: pd.Timedelta,
) -> pd.DataFrame:
    event = db.table_dict[EVENT_TABLE].df
    obj = db.table_dict[OBJECT_TABLE].df
    e2o = db.table_dict[E2O_TABLE].df
    times_df = pd.DataFrame({"obs_time": pd.to_datetime(times).sort_values().unique()})
    future_window = f"{int(delta.total_seconds())} seconds"
    lookback_window = f"{int(lookback.total_seconds())} seconds"
    con = duckdb.connect()
    con.register("event", event)
    con.register("obj", obj)
    con.register("e2o", e2o)
    con.register("times_df", times_df)
    try:
        return con.execute(
            f"""
            WITH src_object AS (
                SELECT {OBJECT_ID_COL} AS src, {TIME_COL} AS src_time
                FROM obj
                WHERE {OBJECT_TYPE_COL} = '{object_types[0]}'
            ),
            dst_object AS (
                SELECT {OBJECT_ID_COL} AS dst, {TIME_COL} AS dst_time
                FROM obj
                WHERE {OBJECT_TYPE_COL} = '{object_types[1]}'
            ),
            candidate_pairs AS (
                SELECT
                    t.obs_time AS {TIME_COL},
                    s.src,
                    d.dst
                FROM times_df t
                JOIN src_object s
                  ON s.src_time BETWEEN t.obs_time - INTERVAL '{lookback_window}' AND t.obs_time
                JOIN dst_object d
                  ON d.dst_time BETWEEN t.obs_time - INTERVAL '{lookback_window}' AND t.obs_time
                WHERE s.src <> d.dst
            ),
            future_links AS (
                SELECT
                    e.{EVENT_ID_COL} AS event_id,
                    e.{TIME_COL} AS event_time,
                    le.{OBJECT_ID_COL} AS src,
                    re.{OBJECT_ID_COL} AS dst
                FROM event e
                JOIN e2o le
                  ON e.{EVENT_ID_COL} = le.{EVENT_ID_COL}
                JOIN e2o re
                  ON e.{EVENT_ID_COL} = re.{EVENT_ID_COL}
                JOIN src_object s
                  ON s.src = le.{OBJECT_ID_COL}
                JOIN dst_object d
                  ON d.dst = re.{OBJECT_ID_COL}
                WHERE e.{EVENT_TYPE_COL} = '{event_type}'
                  AND le.{OBJECT_ID_COL} <> re.{OBJECT_ID_COL}
            ),
            labeled AS (
                SELECT
                    c.src,
                    c.dst,
                    c.{TIME_COL},
                    COUNT(f.event_id) > 0 AS target
                FROM candidate_pairs c
                LEFT JOIN future_links f
                  ON f.src = c.src
                 AND f.dst = c.dst
                 AND f.event_time > c.{TIME_COL}
                 AND f.event_time <= c.{TIME_COL} + INTERVAL '{future_window}'
                GROUP BY c.src, c.dst, c.{TIME_COL}
            )
            SELECT
                src,
                dst,
                {TIME_COL},
                target
            FROM labeled
            ORDER BY {TIME_COL}, src, dst
            """
        ).df()
    finally:
        con.close()
