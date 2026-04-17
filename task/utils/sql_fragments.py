"""Reusable SQL fragment helpers for window-based task builders.

All helpers return SQL strings (CTEs or clause fragments) that compose into
the standard DuckDB query used by build_window_event_counts.

Assumed context when the SQL is executed:
  - `times_df`  : table with column `obs_time` (one row per observation timestamp)
  - `event`     : RelBench event table (event_id, type, time, ...)
  - `obj`       : RelBench object table (object_id, type, time, ...)
  - `e2o`       : RelBench event-to-object table (event_id, object_id)
"""

from __future__ import annotations

from data.const import (
    EVENT_ID_COL,
    EVENT_TYPE_COL,
    OBJECT_ID_COL,
    OBJECT_TYPE_COL,
    TIME_COL,
)


def sql_event_type_filter(event_types: str | list[str] | None, alias: str = "e") -> str:
    """Return an AND-clause that filters to the given event type(s), or '' if None."""
    if event_types is None:
        return ""
    if isinstance(event_types, str):
        event_types = [event_types]
    quoted = ", ".join("'" + t.replace("'", "''") + "'" for t in event_types)
    return f"AND {alias}.{EVENT_TYPE_COL} IN ({quoted})"


def sql_single_obs(object_type: str, delta_seconds: int) -> str:
    """CTEs for single-entity observation candidates.

    Produces:
      - typed_object: objects of the requested type with their first-seen timestamp
      - active_object: (object_id, obs_time) pairs where the object had at least one
        event in (obs_time - Δ, obs_time] — one row per (entity, timestamp) so the
        activity check is timestamp-aware rather than a flat entity set
      - obs: active objects that were also first seen before obs_time

    Returns a string of comma-separated CTEs (no leading WITH).
    """
    ot = object_type.replace("'", "''")
    return f"""
    typed_object AS (
        SELECT {OBJECT_ID_COL}, MIN({TIME_COL}) AS first_seen
        FROM obj
        WHERE {OBJECT_TYPE_COL} = '{ot}'
        GROUP BY {OBJECT_ID_COL}
    ),
    active_object AS (
        SELECT DISTINCT eo.{OBJECT_ID_COL}, t.obs_time
        FROM times_df t
        JOIN event e  ON e.{TIME_COL}  >  t.obs_time - INTERVAL {delta_seconds} SECOND
                     AND e.{TIME_COL}  <= t.obs_time
        JOIN e2o eo   ON eo.{EVENT_ID_COL} = e.{EVENT_ID_COL}
        JOIN obj o    ON o.{OBJECT_ID_COL}  = eo.{OBJECT_ID_COL}
                     AND o.{OBJECT_TYPE_COL} = '{ot}'
    ),
    obs AS (
        SELECT ao.obs_time AS {TIME_COL}, ao.{OBJECT_ID_COL}
        FROM active_object ao
        JOIN typed_object  tp ON tp.{OBJECT_ID_COL} = ao.{OBJECT_ID_COL}
                              AND tp.first_seen      <= ao.obs_time
    )"""


def sql_pair_obs_observed(src_type: str, dst_type: str, delta_seconds: int) -> str:
    """CTEs for observed-pair observation candidates.

    A pair (src, dst) is a candidate at obs_time if:
      - both objects were first seen before obs_time
      - they co-appeared in at least one event before obs_time
      - at least one of them was active in (obs_time - Δ, obs_time]

    active_src/dst produce (entity_id, obs_time) pairs so the activity check
    is timestamp-aware. co_pairs deduplicates to (src_id, dst_id) with the
    earliest co-occurrence time only, avoiding repeated rows in the obs join.

    Returns a string of comma-separated CTEs (no leading WITH).
    """
    src = src_type.replace("'", "''")
    dst = dst_type.replace("'", "''")
    return f"""
    src_obj AS (
        SELECT {OBJECT_ID_COL} AS src_id, MIN({TIME_COL}) AS first_seen
        FROM obj WHERE {OBJECT_TYPE_COL} = '{src}'
        GROUP BY {OBJECT_ID_COL}
    ),
    dst_obj AS (
        SELECT {OBJECT_ID_COL} AS dst_id, MIN({TIME_COL}) AS first_seen
        FROM obj WHERE {OBJECT_TYPE_COL} = '{dst}'
        GROUP BY {OBJECT_ID_COL}
    ),
    co_pairs AS (
        SELECT
            src_eo.{OBJECT_ID_COL} AS src_id,
            dst_eo.{OBJECT_ID_COL} AS dst_id,
            MIN(e.{TIME_COL})       AS first_co_time
        FROM event e
        JOIN e2o src_eo ON src_eo.{EVENT_ID_COL} = e.{EVENT_ID_COL}
        JOIN obj  src_o ON src_o.{OBJECT_ID_COL}  = src_eo.{OBJECT_ID_COL}
                       AND src_o.{OBJECT_TYPE_COL} = '{src}'
        JOIN e2o dst_eo ON dst_eo.{EVENT_ID_COL} = e.{EVENT_ID_COL}
        JOIN obj  dst_o ON dst_o.{OBJECT_ID_COL}  = dst_eo.{OBJECT_ID_COL}
                       AND dst_o.{OBJECT_TYPE_COL} = '{dst}'
        WHERE src_eo.{OBJECT_ID_COL} <> dst_eo.{OBJECT_ID_COL}
        GROUP BY src_eo.{OBJECT_ID_COL}, dst_eo.{OBJECT_ID_COL}
    ),
    active_src AS (
        SELECT DISTINCT eo.{OBJECT_ID_COL} AS src_id, t.obs_time
        FROM times_df t
        JOIN event e  ON e.{TIME_COL}  >  t.obs_time - INTERVAL {delta_seconds} SECOND
                     AND e.{TIME_COL}  <= t.obs_time
        JOIN e2o eo   ON eo.{EVENT_ID_COL} = e.{EVENT_ID_COL}
        JOIN obj o    ON o.{OBJECT_ID_COL}  = eo.{OBJECT_ID_COL}
                     AND o.{OBJECT_TYPE_COL} = '{src}'
    ),
    active_dst AS (
        SELECT DISTINCT eo.{OBJECT_ID_COL} AS dst_id, t.obs_time
        FROM times_df t
        JOIN event e  ON e.{TIME_COL}  >  t.obs_time - INTERVAL {delta_seconds} SECOND
                     AND e.{TIME_COL}  <= t.obs_time
        JOIN e2o eo   ON eo.{EVENT_ID_COL} = e.{EVENT_ID_COL}
        JOIN obj o    ON o.{OBJECT_ID_COL}  = eo.{OBJECT_ID_COL}
                     AND o.{OBJECT_TYPE_COL} = '{dst}'
    ),
    obs AS (
        SELECT t.obs_time AS {TIME_COL}, cp.src_id, cp.dst_id
        FROM times_df t
        JOIN co_pairs cp ON cp.first_co_time < t.obs_time
        JOIN src_obj  so ON so.src_id         = cp.src_id
                        AND so.first_seen     <= t.obs_time
        JOIN dst_obj  do_ ON do_.dst_id        = cp.dst_id
                         AND do_.first_seen   <= t.obs_time
        WHERE EXISTS (SELECT 1 FROM active_src a WHERE a.src_id = cp.src_id AND a.obs_time = t.obs_time)
           OR EXISTS (SELECT 1 FROM active_dst a WHERE a.dst_id = cp.dst_id AND a.obs_time = t.obs_time)
    )"""


def sql_pair_obs_cartesian(src_type: str, dst_type: str, delta_seconds: int) -> str:
    """CTEs for cartesian-pair observation candidates.

    Every (src, dst) combination where both were first seen before obs_time and at
    least one was active in (obs_time - Δ, obs_time].

    active_src/dst produce (entity_id, obs_time) pairs so the activity check is
    timestamp-aware. The obs filter uses EXISTS rather than a flat IN-subquery.

    Returns a string of comma-separated CTEs (no leading WITH).
    """
    src = src_type.replace("'", "''")
    dst = dst_type.replace("'", "''")
    return f"""
    src_obj AS (
        SELECT {OBJECT_ID_COL} AS src_id, MIN({TIME_COL}) AS first_seen
        FROM obj WHERE {OBJECT_TYPE_COL} = '{src}'
        GROUP BY {OBJECT_ID_COL}
    ),
    dst_obj AS (
        SELECT {OBJECT_ID_COL} AS dst_id, MIN({TIME_COL}) AS first_seen
        FROM obj WHERE {OBJECT_TYPE_COL} = '{dst}'
        GROUP BY {OBJECT_ID_COL}
    ),
    active_src AS (
        SELECT DISTINCT eo.{OBJECT_ID_COL} AS src_id, t.obs_time
        FROM times_df t
        JOIN event e  ON e.{TIME_COL}  >  t.obs_time - INTERVAL {delta_seconds} SECOND
                     AND e.{TIME_COL}  <= t.obs_time
        JOIN e2o eo   ON eo.{EVENT_ID_COL} = e.{EVENT_ID_COL}
        JOIN obj o    ON o.{OBJECT_ID_COL}  = eo.{OBJECT_ID_COL}
                     AND o.{OBJECT_TYPE_COL} = '{src}'
    ),
    active_dst AS (
        SELECT DISTINCT eo.{OBJECT_ID_COL} AS dst_id, t.obs_time
        FROM times_df t
        JOIN event e  ON e.{TIME_COL}  >  t.obs_time - INTERVAL {delta_seconds} SECOND
                     AND e.{TIME_COL}  <= t.obs_time
        JOIN e2o eo   ON eo.{EVENT_ID_COL} = e.{EVENT_ID_COL}
        JOIN obj o    ON o.{OBJECT_ID_COL}  = eo.{OBJECT_ID_COL}
                     AND o.{OBJECT_TYPE_COL} = '{dst}'
    ),
    obs AS (
        SELECT t.obs_time AS {TIME_COL}, so.src_id, do_.dst_id
        FROM times_df t
        JOIN src_obj so  ON so.first_seen  <= t.obs_time
        JOIN dst_obj do_ ON do_.first_seen <= t.obs_time
        WHERE so.src_id <> do_.dst_id
          AND (
              EXISTS (SELECT 1 FROM active_src a WHERE a.src_id = so.src_id AND a.obs_time = t.obs_time)
           OR EXISTS (SELECT 1 FROM active_dst a WHERE a.dst_id = do_.dst_id AND a.obs_time = t.obs_time)
          )
    )"""


def sql_single_window_events(delta_seconds: int, event_type_filter: str = "") -> str:
    """CTE: future events in (obs_time, obs_time+Δ] linked to the single entity in obs."""
    return f"""
    window_events AS (
        SELECT obs.{OBJECT_ID_COL}, obs.{TIME_COL} AS obs_time,
               e.{EVENT_ID_COL}, e.{EVENT_TYPE_COL}, e.{TIME_COL} AS event_time
        FROM obs
        JOIN e2o ON e2o.{OBJECT_ID_COL} = obs.{OBJECT_ID_COL}
        JOIN event e ON e.{EVENT_ID_COL} = e2o.{EVENT_ID_COL}
        WHERE e.{TIME_COL} >  obs.{TIME_COL}
          AND e.{TIME_COL} <= obs.{TIME_COL} + INTERVAL {delta_seconds} SECOND
          {event_type_filter}
    )"""


def sql_pair_window_events(delta_seconds: int, event_type_filter: str = "") -> str:
    """CTE: future events in (obs_time, obs_time+Δ] jointly linking (src, dst) in obs."""
    return f"""
    window_events AS (
        SELECT obs.src_id, obs.dst_id, obs.{TIME_COL} AS obs_time,
               e.{EVENT_ID_COL}, e.{EVENT_TYPE_COL}, e.{TIME_COL} AS event_time
        FROM obs
        JOIN e2o src_link ON src_link.{OBJECT_ID_COL} = obs.src_id
        JOIN event e      ON e.{EVENT_ID_COL} = src_link.{EVENT_ID_COL}
        JOIN e2o dst_link ON dst_link.{EVENT_ID_COL} = e.{EVENT_ID_COL}
                         AND dst_link.{OBJECT_ID_COL} = obs.dst_id
        WHERE e.{TIME_COL} >  obs.{TIME_COL}
          AND e.{TIME_COL} <= obs.{TIME_COL} + INTERVAL {delta_seconds} SECOND
          {event_type_filter}
    )"""
