import pandas as pd
from relbench.base import Database

from data.const import (
    EVENT_ID_COL,
    OBJECT_ID_COL,
    OBJECT_TYPE_COL,
    TIME_COL,
)
from data.wrapper import check_dbs
from .db_utils import ocel_connection


@check_dbs
def build_remaining_time_table(db: Database, object_type: str, times: pd.Series) -> pd.DataFrame:
    """Build the remaining-time regression target for a given object type.

    For each (object, observation_time) pair, returns the elapsed seconds
    until the object's final linked event — i.e. how much of its lifecycle
    is left. Rows with no future events are excluded.
    """
    with ocel_connection(db, times) as con:
        return con.execute(
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
                    MAX(event.{TIME_COL}) OVER (
                        PARTITION BY obs.{OBJECT_ID_COL}, obs.{TIME_COL}
                    ) AS last_event_time,
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
                CAST(EXTRACT(epoch FROM (last_event_time - {TIME_COL})) AS DOUBLE) AS target
            FROM ranked
            WHERE rn = 1
            ORDER BY {OBJECT_ID_COL}, {TIME_COL}
            """
        ).df()
