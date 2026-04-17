from __future__ import annotations

import pandas as pd
from relbench.base import Database

from data.const import OBJECT_ID_COL, TIME_COL
from data.wrapper import check_dbs
from .db_utils import ocel_connection


@check_dbs
def build_stage_transition_binary_table(
    db: Database,
    object_type: str,
    timestamps: pd.Series,
    delta: pd.Timedelta,
    source_event_type: str,
    target_event_type: str,
    source_max_age: pd.Timedelta | None = None,
) -> pd.DataFrame:
    """Label whether `target_event_type` happens in `(t_obs, t_obs + delta]`.

    Candidates are objects of `object_type` whose latest observed event at
    `t_obs` is `source_event_type`, with that source event itself occurring in
    `(t_obs - delta, t_obs]`. This yields stage-aware snapshots that satisfy
    the task invariants:

    - features are history-only because all candidate state is derived from
      events with `time <= t_obs`
    - objects are candidates only after they have been observed
    - labels come strictly from the future interval `(t_obs, t_obs + delta]`
    """
    delta_seconds = int(delta.total_seconds())
    source_max_age_seconds = (
        None if source_max_age is None else int(source_max_age.total_seconds())
    )
    object_type_sql = object_type.replace("'", "''")
    source_sql = source_event_type.replace("'", "''")
    target_sql = target_event_type.replace("'", "''")
    times_sorted = pd.to_datetime(timestamps).sort_values().unique()

    recency_clause = ""
    if source_max_age_seconds is not None:
        recency_clause = (
            f"AND lbo.last_time > lbo.{TIME_COL} - INTERVAL {source_max_age_seconds} SECOND"
        )

    query = f"""
    WITH typed_events AS (
        SELECT
            eo.{OBJECT_ID_COL},
            e.type,
            e.{TIME_COL},
            e.event_id
        FROM event e
        JOIN e2o eo ON eo.event_id = e.event_id
        JOIN obj o ON o.{OBJECT_ID_COL} = eo.{OBJECT_ID_COL}
        WHERE o.type = '{object_type_sql}'
    ),
    latest_before_obs AS (
        SELECT
            t.obs_time AS {TIME_COL},
            te.{OBJECT_ID_COL},
            te.type AS last_type,
            te.{TIME_COL} AS last_time,
            ROW_NUMBER() OVER (
                PARTITION BY t.obs_time, te.{OBJECT_ID_COL}
                ORDER BY te.{TIME_COL} DESC, te.event_id DESC
            ) AS rn
        FROM times_df t
        JOIN typed_events te
          ON te.{TIME_COL} <= t.obs_time
    ),
    obs AS (
        SELECT
            lbo.{OBJECT_ID_COL},
            lbo.{TIME_COL}
        FROM latest_before_obs lbo
        WHERE lbo.rn = 1
          AND lbo.last_type = '{source_sql}'
          {recency_clause}
    ),
    future_hits AS (
        SELECT
            obs.{OBJECT_ID_COL},
            obs.{TIME_COL},
            MAX(CASE WHEN te.type = '{target_sql}' THEN 1 ELSE 0 END) AS target
        FROM obs
        LEFT JOIN typed_events te
          ON te.{OBJECT_ID_COL} = obs.{OBJECT_ID_COL}
         AND te.{TIME_COL} > obs.{TIME_COL}
         AND te.{TIME_COL} <= obs.{TIME_COL} + INTERVAL {delta_seconds} SECOND
        GROUP BY obs.{OBJECT_ID_COL}, obs.{TIME_COL}
    )
    SELECT
        {OBJECT_ID_COL},
        {TIME_COL},
        CAST(target AS INTEGER) AS target
    FROM future_hits
    ORDER BY {TIME_COL}, {OBJECT_ID_COL}
    """

    with ocel_connection(db, times_sorted) as con:
        return con.execute(query).df()
