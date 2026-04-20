"""Reusable DuckDB-backed label builders for MEntityTask subclasses.

Every public function returns a plain DataFrame with columns:
  (entity_col, time_col, "target")

ready to be wrapped in a relbench Table by the calling task.

Design principles
-----------------
* One function per label family; each is parameterised so tasks only supply
  the domain-specific constants (object type, event names, window sizes).
* Queries are batched over ``times`` in slices of BATCH_SIZE to keep
  intermediate join cardinalities bounded.
* All window boundaries are converted to integer seconds before the SQL
  so DuckDB can use interval arithmetic without Python datetime overhead.
"""
from typing import Sequence

import pandas as pd
from relbench.base import Database

from data.const import OBJECT_ID_COL, TIME_COL
from task.utils.db_utils import ocel_connection

BATCH_SIZE = 50  # timestamps per DuckDB query batch


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _batched(times: pd.Series, size: int = BATCH_SIZE):
    """Yield successive slices of a timestamp Series."""
    arr = list(times)
    for i in range(0, len(arr), size):
        yield arr[i : i + size]


def _register_batch(con, batch: list) -> None:
    """(Re-)register the current observation-time slice as times_df."""
    import pandas as _pd
    con.register("times_df", _pd.DataFrame({"obs_time": _pd.to_datetime(batch)}))


def _case_expr(event_types: Sequence[str]) -> str:
    """Build a SQL CASE expression mapping event type → integer class index."""
    branches = "\n        ".join(
        f"WHEN etype = {et!r} THEN {i}" for i, et in enumerate(event_types)
    )
    return f"CASE\n        {branches}\n        ELSE NULL\n      END"


# ---------------------------------------------------------------------------
# 1. Next-event classification
#    For each (object, obs_time): what is the type of the very next event
#    involving this object within [obs_time, obs_time + fwd]?
# ---------------------------------------------------------------------------

def build_next_event_table(
    db: Database,
    object_type: str,
    times: pd.Series,
    event_types: Sequence[str],
    *,
    delta_back: pd.Timedelta = pd.Timedelta(days=30),
    delta_fwd: pd.Timedelta = pd.Timedelta(days=14),
) -> pd.DataFrame:
    back_s = int(delta_back.total_seconds())
    fwd_s  = int(delta_fwd.total_seconds())
    case   = _case_expr(event_types)
    parts: list[pd.DataFrame] = []

    with ocel_connection(db) as con:
        for batch in _batched(times):
            _register_batch(con, batch)
            df = con.execute(f"""
                WITH
                  typed_obj AS (
                    SELECT object_id FROM obj WHERE type = {object_type!r}
                  ),
                  obs AS (
                    SELECT t.obs_time AS time, o.object_id
                    FROM times_df t
                    CROSS JOIN typed_obj o
                    WHERE EXISTS (
                      SELECT 1 FROM e2o eo
                      JOIN event e ON e.event_id = eo.event_id
                      WHERE eo.object_id = o.object_id
                        AND e.time > t.obs_time - INTERVAL ({back_s}) SECOND
                        AND e.time <= t.obs_time
                    )
                  ),
                  future AS (
                    SELECT obs.object_id, obs.time,
                           e.type AS etype,
                           ROW_NUMBER() OVER (
                             PARTITION BY obs.object_id, obs.time
                             ORDER BY e.time ASC
                           ) AS rn
                    FROM obs
                    JOIN e2o eo ON eo.object_id = obs.object_id
                    JOIN event e ON e.event_id = eo.event_id
                    WHERE e.time > obs.time
                      AND e.time <= obs.time + INTERVAL ({fwd_s}) SECOND
                  )
                SELECT object_id, time,
                       CAST({case} AS INTEGER) AS target
                FROM future
                WHERE rn = 1
                  AND target IS NOT NULL
            """).df()
            if not df.empty:
                parts.append(df)

    if not parts:
        return pd.DataFrame(columns=[OBJECT_ID_COL, TIME_COL, "target"])
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# 2. Next-time regression
#    For each (object, obs_time): days until the next event involving
#    this object within the look-back-active window.
# ---------------------------------------------------------------------------

def build_next_time_table(
    db: Database,
    object_type: str,
    times: pd.Series,
    *,
    delta_back: pd.Timedelta = pd.Timedelta(days=30),
    delta_fwd: pd.Timedelta = pd.Timedelta(days=30),
) -> pd.DataFrame:
    back_s = int(delta_back.total_seconds())
    fwd_s  = int(delta_fwd.total_seconds())
    parts: list[pd.DataFrame] = []

    with ocel_connection(db) as con:
        for batch in _batched(times):
            _register_batch(con, batch)
            df = con.execute(f"""
                WITH
                  typed_obj AS (
                    SELECT object_id FROM obj WHERE type = {object_type!r}
                  ),
                  obs AS (
                    SELECT t.obs_time AS time, o.object_id
                    FROM times_df t
                    CROSS JOIN typed_obj o
                    WHERE EXISTS (
                      SELECT 1 FROM e2o eo
                      JOIN event e ON e.event_id = eo.event_id
                      WHERE eo.object_id = o.object_id
                        AND e.time > t.obs_time - INTERVAL ({back_s}) SECOND
                        AND e.time <= t.obs_time
                    )
                  ),
                  future AS (
                    SELECT obs.object_id, obs.time,
                           MIN(e.time) AS next_event_time
                    FROM obs
                    JOIN e2o eo ON eo.object_id = obs.object_id
                    JOIN event e ON e.event_id = eo.event_id
                    WHERE e.time > obs.time
                      AND e.time <= obs.time + INTERVAL ({fwd_s}) SECOND
                    GROUP BY obs.object_id, obs.time
                  )
                SELECT object_id, time,
                       CAST(EXTRACT(EPOCH FROM (next_event_time - time)) / 86400.0 AS DOUBLE) AS target
                FROM future
                WHERE target IS NOT NULL
            """).df()
            if not df.empty:
                parts.append(df)

    if not parts:
        return pd.DataFrame(columns=[OBJECT_ID_COL, TIME_COL, "target"])
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# 3. Remaining-time regression
#    For each (object, obs_time): days until the LAST future event for
#    this object (case completion / total remaining cycle time).
# ---------------------------------------------------------------------------

def build_remaining_time_table(
    db: Database,
    object_type: str,
    times: pd.Series,
    *,
    delta_back: pd.Timedelta = pd.Timedelta(days=30),
) -> pd.DataFrame:
    back_s = int(delta_back.total_seconds())
    parts: list[pd.DataFrame] = []

    with ocel_connection(db) as con:
        for batch in _batched(times):
            _register_batch(con, batch)
            df = con.execute(f"""
                WITH
                  typed_obj AS (
                    SELECT object_id FROM obj WHERE type = {object_type!r}
                  ),
                  obs AS (
                    SELECT t.obs_time AS time, o.object_id
                    FROM times_df t
                    CROSS JOIN typed_obj o
                    WHERE EXISTS (
                      SELECT 1 FROM e2o eo
                      JOIN event e ON e.event_id = eo.event_id
                      WHERE eo.object_id = o.object_id
                        AND e.time > t.obs_time - INTERVAL ({back_s}) SECOND
                        AND e.time <= t.obs_time
                    )
                  ),
                  future AS (
                    SELECT obs.object_id, obs.time,
                           MAX(e.time) AS last_event_time
                    FROM obs
                    JOIN e2o eo ON eo.object_id = obs.object_id
                    JOIN event e ON e.event_id = eo.event_id
                    WHERE e.time > obs.time
                    GROUP BY obs.object_id, obs.time
                  )
                SELECT object_id, time,
                       CAST(EXTRACT(EPOCH FROM (last_event_time - time)) / 86400.0 AS DOUBLE) AS target
                FROM future
                WHERE target IS NOT NULL
            """).df()
            if not df.empty:
                parts.append(df)

    if not parts:
        return pd.DataFrame(columns=[OBJECT_ID_COL, TIME_COL, "target"])
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# 4. Binary event-within-window classification
#    For each (object, obs_time): does this object experience *any* event
#    from ``target_event_types`` within ``delta_fwd`` of obs_time?
#    Optionally conditioned on a prior trigger event within ``delta_back``.
# ---------------------------------------------------------------------------

def build_event_within_table(
    db: Database,
    object_type: str,
    times: pd.Series,
    target_event_types: Sequence[str],
    *,
    delta_back: pd.Timedelta = pd.Timedelta(days=30),
    delta_fwd: pd.Timedelta = pd.Timedelta(days=14),
    trigger_event_types: Sequence[str] | None = None,
) -> pd.DataFrame:
    back_s = int(delta_back.total_seconds())
    fwd_s  = int(delta_fwd.total_seconds())
    target_list = ", ".join(f"{et!r}" for et in target_event_types)
    parts: list[pd.DataFrame] = []

    trigger_filter = ""
    if trigger_event_types:
        trig_list = ", ".join(f"{et!r}" for et in trigger_event_types)
        trigger_filter = f"""
                    AND EXISTS (
                      SELECT 1 FROM e2o eo2
                      JOIN event e2 ON e2.event_id = eo2.event_id
                      WHERE eo2.object_id = o.object_id
                        AND e2.type IN ({trig_list})
                        AND e2.time > t.obs_time - INTERVAL ({back_s}) SECOND
                        AND e2.time <= t.obs_time
                    )"""

    with ocel_connection(db) as con:
        for batch in _batched(times):
            _register_batch(con, batch)
            df = con.execute(f"""
                WITH
                  typed_obj AS (
                    SELECT object_id FROM obj WHERE type = {object_type!r}
                  ),
                  obs AS (
                    SELECT t.obs_time AS time, o.object_id
                    FROM times_df t
                    CROSS JOIN typed_obj o
                    WHERE EXISTS (
                      SELECT 1 FROM e2o eo
                      JOIN event e ON e.event_id = eo.event_id
                      WHERE eo.object_id = o.object_id
                        AND e.time > t.obs_time - INTERVAL ({back_s}) SECOND
                        AND e.time <= t.obs_time
                    ){trigger_filter}
                  )
                SELECT obs.object_id, obs.time,
                       CAST(CASE WHEN EXISTS (
                         SELECT 1 FROM e2o eo
                         JOIN event e ON e.event_id = eo.event_id
                         WHERE eo.object_id = obs.object_id
                           AND e.type IN ({target_list})
                           AND e.time > obs.time
                           AND e.time <= obs.time + INTERVAL ({fwd_s}) SECOND
                       ) THEN 1 ELSE 0 END AS INTEGER) AS target
                FROM obs
            """).df()
            if not df.empty:
                parts.append(df)

    if not parts:
        return pd.DataFrame(columns=[OBJECT_ID_COL, TIME_COL, "target"])
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# 5. Multi-entity pair — binary: will src interact with ANY dst of dst_type
#    via a specific event within delta_fwd after obs_time?
#    Entity columns: (object_id [src], object_id_partner [dst])
# ---------------------------------------------------------------------------

def build_pair_interaction_table(
    db: Database,
    src_type: str,
    dst_type: str,
    times: pd.Series,
    interaction_event_types: Sequence[str],
    *,
    delta_back: pd.Timedelta = pd.Timedelta(days=30),
    delta_fwd: pd.Timedelta = pd.Timedelta(days=14),
    pair_col: str = "object_id_partner",
    max_negatives_per_positive: int | None = None,
) -> pd.DataFrame:
    """Enumerate observed (src, dst) pairs active at obs_time; label = 1 if
    they co-appear in a future interaction event within delta_fwd."""
    back_s  = int(delta_back.total_seconds())
    fwd_s   = int(delta_fwd.total_seconds())
    ev_list = ", ".join(f"{et!r}" for et in interaction_event_types)
    parts: list[pd.DataFrame] = []

    with ocel_connection(db) as con:
        for batch in _batched(times):
            _register_batch(con, batch)
            df = con.execute(f"""
                WITH
                  src_obj AS (SELECT object_id FROM obj WHERE type = {src_type!r}),
                  dst_obj AS (SELECT object_id AS dst_id FROM obj WHERE type = {dst_type!r}),
                  -- observed (src, dst) pairs: both appeared in same event in lookback
                  observed_pairs AS (
                    SELECT DISTINCT eo_s.object_id AS object_id,
                                    eo_d.object_id AS {pair_col},
                                    t.obs_time AS time
                    FROM times_df t
                    JOIN e2o eo_s ON TRUE
                    JOIN event e  ON e.event_id = eo_s.event_id
                                  AND e.time > t.obs_time - INTERVAL ({back_s}) SECOND
                                  AND e.time <= t.obs_time
                    JOIN e2o eo_d ON eo_d.event_id = eo_s.event_id
                    JOIN src_obj  ON src_obj.object_id  = eo_s.object_id
                    JOIN dst_obj  ON dst_obj.dst_id      = eo_d.object_id
                    WHERE eo_s.object_id <> eo_d.object_id
                  ),
                  labelled AS (
                    SELECT op.object_id, op.{pair_col}, op.time,
                           CAST(CASE WHEN EXISTS (
                             SELECT 1 FROM e2o eos2
                             JOIN event e2 ON e2.event_id = eos2.event_id
                             JOIN e2o eod2 ON eod2.event_id = eos2.event_id
                             WHERE eos2.object_id = op.object_id
                               AND eod2.object_id = op.{pair_col}
                               AND e2.type IN ({ev_list})
                               AND e2.time > op.time
                               AND e2.time <= op.time + INTERVAL ({fwd_s}) SECOND
                           ) THEN 1 ELSE 0 END AS INTEGER) AS target
                    FROM observed_pairs op
                  )
                SELECT object_id, {pair_col}, time, target FROM labelled
            """).df()
            if not df.empty:
                parts.append(df)

    if not parts:
        return pd.DataFrame(columns=[OBJECT_ID_COL, pair_col, TIME_COL, "target"])
    df = pd.concat(parts, ignore_index=True)
    if max_negatives_per_positive is None:
        return df
    return _cap_binary_negatives(
        df,
        max_negatives_per_positive=max_negatives_per_positive,
        key_cols=[OBJECT_ID_COL, pair_col, TIME_COL],
    )


def _cap_binary_negatives(
    df: pd.DataFrame,
    *,
    max_negatives_per_positive: int,
    key_cols: list[str],
) -> pd.DataFrame:
    if max_negatives_per_positive < 1:
        raise ValueError("max_negatives_per_positive must be at least 1")

    positives = df[df["target"] == 1]
    negatives = df[df["target"] == 0]
    max_negatives = len(positives) * max_negatives_per_positive
    if positives.empty or len(negatives) <= max_negatives:
        return df

    hash_cols = [col for col in key_cols if col in negatives.columns]
    sampled_negatives = (
        negatives.assign(
            __sample_hash__=pd.util.hash_pandas_object(
                negatives[hash_cols],
                index=False,
            )
        )
        .sort_values("__sample_hash__", kind="stable")
        .head(max_negatives)
        .drop(columns="__sample_hash__")
    )
    return (
        pd.concat([positives, sampled_negatives], ignore_index=True)
        .sort_values(key_cols, kind="stable")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 6. Next co-object classification
#    For each (object, obs_time): which object of dst_type appears together
#    with this object in the very next future event (of any or specific type)?
#    Returns the dst object_id as an integer class target (label-encoded).
# ---------------------------------------------------------------------------

def build_next_coobject_table(
    db: Database,
    src_type: str,
    dst_type: str,
    times: pd.Series,
    *,
    delta_back: pd.Timedelta = pd.Timedelta(days=30),
    delta_fwd: pd.Timedelta = pd.Timedelta(days=14),
    event_types: Sequence[str] | None = None,
    pair_col: str = "object_id_partner",
) -> pd.DataFrame:
    """Target = dst object_id (encoded as int64 RelBench entity index)."""
    back_s  = int(delta_back.total_seconds())
    fwd_s   = int(delta_fwd.total_seconds())
    ev_filter = ""
    if event_types:
        ev_list   = ", ".join(f"{et!r}" for et in event_types)
        ev_filter = f"AND e.type IN ({ev_list})"
    parts: list[pd.DataFrame] = []

    with ocel_connection(db) as con:
        for batch in _batched(times):
            _register_batch(con, batch)
            df = con.execute(f"""
                WITH
                  src_obj AS (SELECT object_id FROM obj WHERE type = {src_type!r}),
                  obs AS (
                    SELECT t.obs_time AS time, o.object_id
                    FROM times_df t
                    CROSS JOIN src_obj o
                    WHERE EXISTS (
                      SELECT 1 FROM e2o eo
                      JOIN event e ON e.event_id = eo.event_id
                      WHERE eo.object_id = o.object_id
                        AND e.time > t.obs_time - INTERVAL ({back_s}) SECOND
                        AND e.time <= t.obs_time
                    )
                  ),
                  future AS (
                    SELECT obs.object_id, obs.time,
                           eo_d.object_id AS {pair_col},
                           e.time AS event_time,
                           ROW_NUMBER() OVER (
                             PARTITION BY obs.object_id, obs.time
                             ORDER BY e.time ASC
                           ) AS rn
                    FROM obs
                    JOIN e2o eo_s ON eo_s.object_id = obs.object_id
                    JOIN event e  ON e.event_id = eo_s.event_id
                                  AND e.time > obs.time
                                  AND e.time <= obs.time + INTERVAL ({fwd_s}) SECOND
                                  {ev_filter}
                    JOIN e2o eo_d ON eo_d.event_id = eo_s.event_id
                    JOIN obj dst  ON dst.object_id = eo_d.object_id
                                  AND dst.type = {dst_type!r}
                    WHERE eo_d.object_id <> obs.object_id
                  )
                SELECT object_id, {pair_col} AS target_entity, time
                FROM future
                WHERE rn = 1
            """).df()
            if not df.empty:
                # rename target_entity → pair_col and keep raw int as target
                df = df.rename(columns={"target_entity": pair_col})
                df["target"] = df[pair_col].astype("int64")
                parts.append(df)

    if not parts:
        return pd.DataFrame(columns=[OBJECT_ID_COL, pair_col, TIME_COL, "target"])
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Convenience: wrap a builder DataFrame in a relbench Table
# ---------------------------------------------------------------------------

def to_relbench_table(
    df: pd.DataFrame,
    entity_cols: tuple[str, ...],
    entity_tables: tuple[str, ...],
    time_col: str = TIME_COL,
):
    """Wrap a label DataFrame in a relbench Table with correct FK metadata."""
    from relbench.base import Table
    return Table(
        df=df,
        fkey_col_to_pkey_table=dict(zip(entity_cols, entity_tables)),
        pkey_col=None,
        time_col=time_col,
    )
