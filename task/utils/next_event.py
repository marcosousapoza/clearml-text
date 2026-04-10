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
def build_next_event_table(db: Database, object_type: str, times: pd.Series) -> pd.DataFrame:
    event = db.table_dict[EVENT_TABLE].df
    obj = db.table_dict[OBJECT_TABLE].df
    e2o = db.table_dict[E2O_TABLE].df
    times_df = pd.DataFrame({"obs_time": pd.to_datetime(times)})

    # Build label encoding from all event types for this object type
    event_types = (
        e2o.merge(obj[[OBJECT_ID_COL, OBJECT_TYPE_COL]], on=OBJECT_ID_COL)
        .loc[lambda d: d[OBJECT_TYPE_COL] == object_type, EVENT_ID_COL]
        .pipe(lambda ids: event.loc[event[EVENT_ID_COL].isin(ids), EVENT_TYPE_COL])
        .unique()
    )
    label_map = {name: idx for idx, name in enumerate(sorted(event_types))}

    con = duckdb.connect()
    con.register("event", event)
    con.register("obj", obj)
    con.register("e2o", e2o)
    con.register("times_df", times_df)
    try:
        df = con.execute(
            f"""
            WITH typed_object AS (
                SELECT {OBJECT_ID_COL}, {TIME_COL} AS object_time
                FROM obj
                WHERE {OBJECT_TYPE_COL} = '{object_type}'
            ),
            obs AS (
                SELECT
                    t.obs_time AS {TIME_COL},
                    o.{OBJECT_ID_COL}
                FROM times_df t
                JOIN typed_object o
                  ON o.object_time <= t.obs_time
            ),
            ranked AS (
                SELECT
                    obs.{OBJECT_ID_COL},
                    obs.{TIME_COL},
                    event.{EVENT_ID_COL},
                    event.{EVENT_TYPE_COL} AS target,
                    ROW_NUMBER() OVER (
                        PARTITION BY obs.{OBJECT_ID_COL}, obs.{TIME_COL}
                        ORDER BY event.{TIME_COL}, event.{EVENT_ID_COL}
                    ) AS rn
                FROM obs
                JOIN e2o
                  ON e2o.{OBJECT_ID_COL} = obs.{OBJECT_ID_COL}
                JOIN event
                  ON event.{EVENT_ID_COL} = e2o.{EVENT_ID_COL}
                WHERE event.{TIME_COL} > obs.{TIME_COL}
            )
            SELECT
                {OBJECT_ID_COL},
                {TIME_COL},
                target
            FROM ranked
            WHERE rn = 1
            ORDER BY {OBJECT_ID_COL}, {TIME_COL}, {EVENT_ID_COL}
            """
        ).df()
    finally:
        con.close()

    df["target"] = df["target"].map(label_map)
    return df
