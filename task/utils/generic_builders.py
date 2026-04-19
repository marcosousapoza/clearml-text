"""Generic task table builders with look-back window activity filtering.

All builders process timestamps in batches so each DuckDB query works over a
small times_df, keeping intermediate join sizes bounded.

Single-entity builders:
  build_generic_next_event_table      -> multiclass classification
  build_generic_next_time_table       -> regression (days to next event)
  build_generic_remaining_time_table  -> regression (weeks to last future event)

Pair builders (observed co-occurrences):
  build_generic_pair_next_event_table -> multiclass classification
  build_generic_pair_next_time_table  -> regression (days to next shared event)
"""

import pandas as pd
from tqdm import tqdm
from relbench.base import Database

from data.const import (
    EVENT_ID_COL,
    EVENT_TYPE_COL,
    OBJECT_ID_COL,
    OBJECT_TYPE_COL,
    O2O_DST_COL,
    O2O_SRC_COL,
    TIME_COL,
)
from data.wrapper import check_dbs
from .db_utils import ocel_connection

# Number of timestamps per DuckDB query batch. Smaller = less memory per query.
_BATCH_SIZE = 50


def _batched(times_sorted, batch_size: int):
    """Yield successive chunks of times_sorted."""
    n = len(times_sorted)
    for i in range(0, n, batch_size):
        yield times_sorted[i : i + batch_size]


# ---------------------------------------------------------------------------
# Single-entity CTEs  (no range join on full event table)
# ---------------------------------------------------------------------------

def _single_obs_cte(object_type: str, back_seconds: int) -> str:
    ot = object_type.replace("'", "''")
    return f"""
    typed_events AS (
        SELECT eo.{OBJECT_ID_COL},
               e.{EVENT_ID_COL},
               e.{TIME_COL}
        FROM e2o eo
        JOIN event e ON e.{EVENT_ID_COL}   = eo.{EVENT_ID_COL}
        JOIN obj   o ON o.{OBJECT_ID_COL}  = eo.{OBJECT_ID_COL}
                    AND o.{OBJECT_TYPE_COL} = '{ot}'
    ),
    latest_before_obs AS (
        SELECT t.obs_time AS {TIME_COL},
               te.{OBJECT_ID_COL},
               te.{TIME_COL} AS last_time,
               ROW_NUMBER() OVER (
                   PARTITION BY t.obs_time, te.{OBJECT_ID_COL}
                   ORDER BY te.{TIME_COL} DESC, te.{EVENT_ID_COL} DESC
               ) AS rn
        FROM times_df t
        JOIN typed_events te
          ON te.{TIME_COL} <= t.obs_time
    ),
    obs AS (
        SELECT lbo.{TIME_COL}, lbo.{OBJECT_ID_COL}
        FROM latest_before_obs lbo
        WHERE lbo.rn = 1
          AND lbo.last_time > lbo.{TIME_COL} - INTERVAL {back_seconds} SECOND
    )"""


# ---------------------------------------------------------------------------
# Public single-entity builders
# ---------------------------------------------------------------------------

@check_dbs
def build_generic_next_event_table(
    db: Database,
    object_type: str,
    times: pd.Series,
    event_types: list[str],
    delta_back: pd.Timedelta,
    delta_fwd: pd.Timedelta,
) -> pd.DataFrame:
    """Multiclass next-event label for active objects, batched over timestamps."""
    back_sec = int(delta_back.total_seconds())
    fwd_sec  = int(delta_fwd.total_seconds())
    times_sorted = pd.to_datetime(times).sort_values().unique()

    in_list   = ", ".join(f"'{t.replace(chr(39), chr(39)*2)}'" for t in event_types)
    case_expr = " ".join(
        f"WHEN '{t.replace(chr(39), chr(39)*2)}' THEN {i}"
        for i, t in enumerate(event_types)
    )
    obs_ctes = _single_obs_cte(object_type, back_sec)

    query = f"""
    WITH
    {obs_ctes},
    future_ranked AS (
        SELECT obs.{OBJECT_ID_COL}, obs.{TIME_COL},
               e.{EVENT_TYPE_COL} AS etype,
               ROW_NUMBER() OVER (
                   PARTITION BY obs.{OBJECT_ID_COL}, obs.{TIME_COL}
                   ORDER BY e.{TIME_COL}, e.{EVENT_ID_COL}
               ) AS rn
        FROM obs
        JOIN e2o   ON e2o.{OBJECT_ID_COL} = obs.{OBJECT_ID_COL}
        JOIN event e ON e.{EVENT_ID_COL}   = e2o.{EVENT_ID_COL}
        WHERE e.{TIME_COL} >  obs.{TIME_COL}
          AND e.{TIME_COL} <= obs.{TIME_COL} + INTERVAL {fwd_sec} SECOND
          AND e.{EVENT_TYPE_COL} IN ({in_list})
    )
    SELECT {OBJECT_ID_COL}, {TIME_COL},
           CAST(CASE etype {case_expr} END AS INTEGER) AS target
    FROM future_ranked WHERE rn = 1
    ORDER BY {TIME_COL}, {OBJECT_ID_COL}
    """

    chunks = []
    batches = list(_batched(times_sorted, _BATCH_SIZE))
    for batch in tqdm(batches, desc=f"next_event({object_type})", leave=False):
        with ocel_connection(db, batch) as con:
            chunks.append(con.execute(query).df())
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


@check_dbs
def build_generic_next_time_table(
    db: Database,
    object_type: str,
    times: pd.Series,
    delta_back: pd.Timedelta,
) -> pd.DataFrame:
    """Regression: days until the next event, batched over timestamps."""
    back_sec = int(delta_back.total_seconds())
    times_sorted = pd.to_datetime(times).sort_values().unique()
    obs_ctes = _single_obs_cte(object_type, back_sec)

    query = f"""
    WITH
    {obs_ctes},
    next_event AS (
        SELECT obs.{OBJECT_ID_COL}, obs.{TIME_COL},
               MIN(e.{TIME_COL}) AS next_time
        FROM obs
        JOIN e2o   ON e2o.{OBJECT_ID_COL} = obs.{OBJECT_ID_COL}
        JOIN event e ON e.{EVENT_ID_COL}   = e2o.{EVENT_ID_COL}
        WHERE e.{TIME_COL} > obs.{TIME_COL}
        GROUP BY obs.{OBJECT_ID_COL}, obs.{TIME_COL}
    )
    SELECT {OBJECT_ID_COL}, {TIME_COL},
           CAST((epoch(next_time) - epoch({TIME_COL})) / 86400.0 AS DOUBLE) AS target
    FROM next_event
    ORDER BY {TIME_COL}, {OBJECT_ID_COL}
    """

    chunks = []
    batches = list(_batched(times_sorted, _BATCH_SIZE))
    for batch in tqdm(batches, desc=f"next_time({object_type})", leave=False):
        with ocel_connection(db, batch) as con:
            chunks.append(con.execute(query).df())
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


@check_dbs
def build_generic_remaining_time_table(
    db: Database,
    object_type: str,
    times: pd.Series,
    delta_back: pd.Timedelta,
) -> pd.DataFrame:
    """Regression: weeks until the last future event, batched over timestamps."""
    back_sec = int(delta_back.total_seconds())
    times_sorted = pd.to_datetime(times).sort_values().unique()
    obs_ctes = _single_obs_cte(object_type, back_sec)

    query = f"""
    WITH
    {obs_ctes},
    last_event AS (
        SELECT obs.{OBJECT_ID_COL}, obs.{TIME_COL},
               MAX(e.{TIME_COL}) AS last_time
        FROM obs
        JOIN e2o   ON e2o.{OBJECT_ID_COL} = obs.{OBJECT_ID_COL}
        JOIN event e ON e.{EVENT_ID_COL}   = e2o.{EVENT_ID_COL}
        WHERE e.{TIME_COL} > obs.{TIME_COL}
        GROUP BY obs.{OBJECT_ID_COL}, obs.{TIME_COL}
    )
    SELECT {OBJECT_ID_COL}, {TIME_COL},
           CAST((epoch(last_time) - epoch({TIME_COL})) / 604800.0 AS DOUBLE) AS target
    FROM last_event
    ORDER BY {TIME_COL}, {OBJECT_ID_COL}
    """

    chunks = []
    batches = list(_batched(times_sorted, _BATCH_SIZE))
    for batch in tqdm(batches, desc=f"remaining_time({object_type})", leave=False):
        with ocel_connection(db, batch) as con:
            chunks.append(con.execute(query).df())
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


# ---------------------------------------------------------------------------
# Pair CTEs
# ---------------------------------------------------------------------------

def _pair_obs_cte(src_type: str, dst_type: str, back_seconds: int) -> str:
    src = src_type.replace("'", "''")
    dst = dst_type.replace("'", "''")
    return f"""
    src_events AS (
        SELECT eo.{OBJECT_ID_COL} AS src_id,
               e.{EVENT_ID_COL},
               e.{TIME_COL}
        FROM e2o eo
        JOIN event e ON e.{EVENT_ID_COL}   = eo.{EVENT_ID_COL}
        JOIN obj   o ON o.{OBJECT_ID_COL}  = eo.{OBJECT_ID_COL}
                    AND o.{OBJECT_TYPE_COL} = '{src}'
    ),
    dst_events AS (
        SELECT eo.{OBJECT_ID_COL} AS dst_id,
               e.{EVENT_ID_COL},
               e.{TIME_COL}
        FROM e2o eo
        JOIN event e ON e.{EVENT_ID_COL}   = eo.{EVENT_ID_COL}
        JOIN obj   o ON o.{OBJECT_ID_COL}  = eo.{OBJECT_ID_COL}
                    AND o.{OBJECT_TYPE_COL} = '{dst}'
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
    src_latest_before_obs AS (
        SELECT t.obs_time AS {TIME_COL},
               se.src_id,
               se.{TIME_COL} AS last_time,
               ROW_NUMBER() OVER (
                   PARTITION BY t.obs_time, se.src_id
                   ORDER BY se.{TIME_COL} DESC, se.{EVENT_ID_COL} DESC
               ) AS rn
        FROM times_df t
        JOIN src_events se
          ON se.{TIME_COL} <= t.obs_time
    ),
    dst_latest_before_obs AS (
        SELECT t.obs_time AS {TIME_COL},
               de.dst_id,
               de.{TIME_COL} AS last_time,
               ROW_NUMBER() OVER (
                   PARTITION BY t.obs_time, de.dst_id
                   ORDER BY de.{TIME_COL} DESC, de.{EVENT_ID_COL} DESC
               ) AS rn
        FROM times_df t
        JOIN dst_events de
          ON de.{TIME_COL} <= t.obs_time
    ),
    obs AS (
        SELECT slbo.{TIME_COL}, cp.src_id, cp.dst_id
        FROM co_pairs cp
        JOIN src_latest_before_obs slbo
          ON slbo.src_id = cp.src_id
         AND slbo.rn = 1
        JOIN dst_latest_before_obs dlbo
          ON dlbo.dst_id = cp.dst_id
         AND dlbo.{TIME_COL} = slbo.{TIME_COL}
         AND dlbo.rn = 1
        WHERE cp.first_co_time < slbo.{TIME_COL}
          AND slbo.last_time > slbo.{TIME_COL} - INTERVAL {back_seconds} SECOND
          AND dlbo.last_time > dlbo.{TIME_COL} - INTERVAL {back_seconds} SECOND
    )"""


# ---------------------------------------------------------------------------
# Public pair builders
# ---------------------------------------------------------------------------

@check_dbs
def build_generic_pair_next_event_table(
    db: Database,
    src_type: str,
    dst_type: str,
    times: pd.Series,
    event_types: list[str],
    delta_back: pd.Timedelta,
    delta_fwd: pd.Timedelta,
) -> pd.DataFrame:
    """Multiclass next shared-event label for observed pairs, batched."""
    back_sec = int(delta_back.total_seconds())
    fwd_sec  = int(delta_fwd.total_seconds())
    times_sorted = pd.to_datetime(times).sort_values().unique()

    in_list   = ", ".join(f"'{t.replace(chr(39), chr(39)*2)}'" for t in event_types)
    case_expr = " ".join(
        f"WHEN '{t.replace(chr(39), chr(39)*2)}' THEN {i}"
        for i, t in enumerate(event_types)
    )
    obs_ctes = _pair_obs_cte(src_type, dst_type, back_sec)

    query = f"""
    WITH
    {obs_ctes},
    future_ranked AS (
        SELECT obs.src_id, obs.dst_id, obs.{TIME_COL},
               e.{EVENT_TYPE_COL} AS etype,
               ROW_NUMBER() OVER (
                   PARTITION BY obs.src_id, obs.dst_id, obs.{TIME_COL}
                   ORDER BY e.{TIME_COL}, e.{EVENT_ID_COL}
               ) AS rn
        FROM obs
        JOIN e2o src_link ON src_link.{OBJECT_ID_COL} = obs.src_id
        JOIN event e       ON e.{EVENT_ID_COL}          = src_link.{EVENT_ID_COL}
        JOIN e2o dst_link  ON dst_link.{EVENT_ID_COL}   = e.{EVENT_ID_COL}
                          AND dst_link.{OBJECT_ID_COL}  = obs.dst_id
        WHERE e.{TIME_COL} >  obs.{TIME_COL}
          AND e.{TIME_COL} <= obs.{TIME_COL} + INTERVAL {fwd_sec} SECOND
          AND e.{EVENT_TYPE_COL} IN ({in_list})
    )
    SELECT src_id AS {O2O_SRC_COL}, dst_id AS {O2O_DST_COL}, {TIME_COL},
           CAST(CASE etype {case_expr} END AS INTEGER) AS target
    FROM future_ranked WHERE rn = 1
    ORDER BY {TIME_COL}, src_id, dst_id
    """

    chunks = []
    batches = list(_batched(times_sorted, _BATCH_SIZE))
    for batch in tqdm(batches, desc=f"pair_next_event({src_type[:6]},{dst_type[:6]})", leave=False):
        with ocel_connection(db, batch) as con:
            chunks.append(con.execute(query).df())
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


@check_dbs
def build_generic_pair_next_time_table(
    db: Database,
    src_type: str,
    dst_type: str,
    times: pd.Series,
    delta_back: pd.Timedelta,
) -> pd.DataFrame:
    """Regression: days until next shared event for observed pairs, batched."""
    back_sec = int(delta_back.total_seconds())
    times_sorted = pd.to_datetime(times).sort_values().unique()
    obs_ctes = _pair_obs_cte(src_type, dst_type, back_sec)

    query = f"""
    WITH
    {obs_ctes},
    next_shared AS (
        SELECT obs.src_id, obs.dst_id, obs.{TIME_COL},
               MIN(e.{TIME_COL}) AS next_time
        FROM obs
        JOIN e2o src_link ON src_link.{OBJECT_ID_COL} = obs.src_id
        JOIN event e       ON e.{EVENT_ID_COL}          = src_link.{EVENT_ID_COL}
        JOIN e2o dst_link  ON dst_link.{EVENT_ID_COL}   = e.{EVENT_ID_COL}
                          AND dst_link.{OBJECT_ID_COL}  = obs.dst_id
        WHERE e.{TIME_COL} > obs.{TIME_COL}
        GROUP BY obs.src_id, obs.dst_id, obs.{TIME_COL}
    )
    SELECT src_id AS {O2O_SRC_COL}, dst_id AS {O2O_DST_COL}, {TIME_COL},
           CAST((epoch(next_time) - epoch({TIME_COL})) / 86400.0 AS DOUBLE) AS target
    FROM next_shared
    ORDER BY {TIME_COL}, src_id, dst_id
    """

    chunks = []
    batches = list(_batched(times_sorted, _BATCH_SIZE))
    for batch in tqdm(batches, desc=f"pair_next_time({src_type[:6]},{dst_type[:6]})", leave=False):
        with ocel_connection(db, batch) as con:
            chunks.append(con.execute(query).df())
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
