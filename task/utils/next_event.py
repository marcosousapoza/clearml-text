import pandas as pd
from relbench.base import Database

from data.const import (
    EVENT_ID_COL,
    EVENT_TYPE_COL,
    OBJECT_ID_COL,
    OBJECT_TYPE_COL,
    TIME_COL,
)
from data.wrapper import check_dbs
from .db_utils import ocel_connection


@check_dbs
def build_next_event_table(
    db: Database,
    object_type: str,
    times: pd.Series,
    event_types: list[str] | None = None,
) -> pd.DataFrame:
    """Build the next-event classification target for a given object type.

    For each (object, observation_time) pair, finds the earliest future event
    that the object is linked to, then encodes it as an integer class label.
    If event_types is not provided, classes are inferred from the data and
    sorted alphabetically.
    """
    with ocel_connection(db, times) as con:
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
            ORDER BY {OBJECT_ID_COL}, {TIME_COL}
            """
        ).df()

    if event_types is None:
        event_types = sorted(df["target"].dropna().unique())

    label_map = {name: idx for idx, name in enumerate(event_types)}
    encoded_target = df["target"].map(label_map)
    if encoded_target.isna().any():
        unknown = sorted(df.loc[encoded_target.isna(), "target"].dropna().unique())
        raise ValueError(
            f"Unknown next-event target(s) for object type {object_type!r}: {unknown}"
        )
    df["target"] = encoded_target.astype("int64")
    return df
