from __future__ import annotations

import pandas as pd
from relbench.base import Database

from data.const import (
    EVENT_ID_COL,
    OBJECT_ID_COL,
    OBJECT_TYPE_COL,
    O2O_DST_COL,
    O2O_SRC_COL,
    TIME_COL,
)
from data.wrapper import check_dbs
from .db_utils import ocel_connection


def _sql_quote(value: str) -> str:
    return value.replace("'", "''")


def _recency_clause(alias: str, time_col: str, source_max_age: pd.Timedelta | None) -> str:
    if source_max_age is None:
        return ""
    seconds = int(source_max_age.total_seconds())
    return f"AND {alias}.{time_col} > {alias}.{TIME_COL} - INTERVAL {seconds} SECOND"


def _class_case_expr(column: str, class_values: list[str]) -> str:
    clauses = [
        f"WHEN '{_sql_quote(value)}' THEN {idx}"
        for idx, value in enumerate(class_values)
    ]
    return f"CASE {column} " + " ".join(clauses) + " END"


@check_dbs
def build_stage_multiclass_next_event_table(
    db: Database,
    object_type: str,
    timestamps: pd.Series,
    delta: pd.Timedelta,
    source_event_type: str,
    next_event_types: list[str],
    source_max_age: pd.Timedelta | None = None,
) -> pd.DataFrame:
    delta_seconds = int(delta.total_seconds())
    object_type_sql = _sql_quote(object_type)
    source_sql = _sql_quote(source_event_type)
    next_list_sql = ", ".join(f"'{_sql_quote(value)}'" for value in next_event_types)
    class_expr = _class_case_expr("future_ranked.future_type", next_event_types)
    times_sorted = pd.to_datetime(timestamps).sort_values().unique()
    recency_clause = _recency_clause("lbo", "last_time", source_max_age)

    query = f"""
    WITH typed_events AS (
        SELECT
            eo.{OBJECT_ID_COL},
            e.{EVENT_ID_COL},
            e.type,
            e.{TIME_COL}
        FROM event e
        JOIN e2o eo ON eo.{EVENT_ID_COL} = e.{EVENT_ID_COL}
        JOIN obj o ON o.{OBJECT_ID_COL} = eo.{OBJECT_ID_COL}
        WHERE o.{OBJECT_TYPE_COL} = '{object_type_sql}'
    ),
    latest_before_obs AS (
        SELECT
            t.obs_time AS {TIME_COL},
            te.{OBJECT_ID_COL},
            te.type AS last_type,
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
        SELECT lbo.{OBJECT_ID_COL}, lbo.{TIME_COL}
        FROM latest_before_obs lbo
        WHERE lbo.rn = 1
          AND lbo.last_type = '{source_sql}'
          {recency_clause}
    ),
    future_ranked AS (
        SELECT
            obs.{OBJECT_ID_COL},
            obs.{TIME_COL},
            te.type AS future_type,
            ROW_NUMBER() OVER (
                PARTITION BY obs.{OBJECT_ID_COL}, obs.{TIME_COL}
                ORDER BY te.{TIME_COL}, te.{EVENT_ID_COL}
            ) AS rn
        FROM obs
        JOIN typed_events te
          ON te.{OBJECT_ID_COL} = obs.{OBJECT_ID_COL}
         AND te.{TIME_COL} > obs.{TIME_COL}
         AND te.{TIME_COL} <= obs.{TIME_COL} + INTERVAL {delta_seconds} SECOND
    )
    SELECT
        future_ranked.{OBJECT_ID_COL},
        future_ranked.{TIME_COL},
        CAST({class_expr} AS INTEGER) AS target
    FROM future_ranked
    WHERE future_ranked.rn = 1
      AND future_ranked.future_type IN ({next_list_sql})
    ORDER BY future_ranked.{TIME_COL}, future_ranked.{OBJECT_ID_COL}
    """

    with ocel_connection(db, times_sorted) as con:
        return con.execute(query).df()


@check_dbs
def build_stage_future_event_count_table(
    db: Database,
    object_type: str,
    timestamps: pd.Series,
    delta: pd.Timedelta,
    source_event_type: str,
    target_event_type: str,
    source_max_age: pd.Timedelta | None = None,
) -> pd.DataFrame:
    delta_seconds = int(delta.total_seconds())
    object_type_sql = _sql_quote(object_type)
    source_sql = _sql_quote(source_event_type)
    target_sql = _sql_quote(target_event_type)
    times_sorted = pd.to_datetime(timestamps).sort_values().unique()
    recency_clause = _recency_clause("lbo", "last_time", source_max_age)

    query = f"""
    WITH typed_events AS (
        SELECT
            eo.{OBJECT_ID_COL},
            e.{EVENT_ID_COL},
            e.type,
            e.{TIME_COL}
        FROM event e
        JOIN e2o eo ON eo.{EVENT_ID_COL} = e.{EVENT_ID_COL}
        JOIN obj o ON o.{OBJECT_ID_COL} = eo.{OBJECT_ID_COL}
        WHERE o.{OBJECT_TYPE_COL} = '{object_type_sql}'
    ),
    latest_before_obs AS (
        SELECT
            t.obs_time AS {TIME_COL},
            te.{OBJECT_ID_COL},
            te.type AS last_type,
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
        SELECT lbo.{OBJECT_ID_COL}, lbo.{TIME_COL}
        FROM latest_before_obs lbo
        WHERE lbo.rn = 1
          AND lbo.last_type = '{source_sql}'
          {recency_clause}
    ),
    future_counts AS (
        SELECT
            obs.{OBJECT_ID_COL},
            obs.{TIME_COL},
            COUNT(*) FILTER (WHERE te.type = '{target_sql}') AS target
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
        CAST(target AS DOUBLE) AS target
    FROM future_counts
    ORDER BY {TIME_COL}, {OBJECT_ID_COL}
    """

    with ocel_connection(db, times_sorted) as con:
        return con.execute(query).df()


@check_dbs
def build_stage_future_distinct_related_count_table(
    db: Database,
    object_type: str,
    related_object_type: str,
    timestamps: pd.Series,
    delta: pd.Timedelta,
    source_event_type: str,
    target_event_type: str,
    source_max_age: pd.Timedelta | None = None,
) -> pd.DataFrame:
    delta_seconds = int(delta.total_seconds())
    object_type_sql = _sql_quote(object_type)
    related_type_sql = _sql_quote(related_object_type)
    source_sql = _sql_quote(source_event_type)
    target_sql = _sql_quote(target_event_type)
    times_sorted = pd.to_datetime(timestamps).sort_values().unique()
    recency_clause = _recency_clause("lbo", "last_time", source_max_age)

    query = f"""
    WITH typed_events AS (
        SELECT
            eo.{OBJECT_ID_COL},
            e.{EVENT_ID_COL},
            e.type,
            e.{TIME_COL}
        FROM event e
        JOIN e2o eo ON eo.{EVENT_ID_COL} = e.{EVENT_ID_COL}
        JOIN obj o ON o.{OBJECT_ID_COL} = eo.{OBJECT_ID_COL}
        WHERE o.{OBJECT_TYPE_COL} = '{object_type_sql}'
    ),
    latest_before_obs AS (
        SELECT
            t.obs_time AS {TIME_COL},
            te.{OBJECT_ID_COL},
            te.type AS last_type,
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
        SELECT lbo.{OBJECT_ID_COL}, lbo.{TIME_COL}
        FROM latest_before_obs lbo
        WHERE lbo.rn = 1
          AND lbo.last_type = '{source_sql}'
          {recency_clause}
    ),
    future_related AS (
        SELECT DISTINCT
            obs.{OBJECT_ID_COL},
            obs.{TIME_COL},
            rel_obj.{OBJECT_ID_COL} AS related_id
        FROM obs
        JOIN e2o own_link
          ON own_link.{OBJECT_ID_COL} = obs.{OBJECT_ID_COL}
        JOIN event e
          ON e.{EVENT_ID_COL} = own_link.{EVENT_ID_COL}
         AND e.type = '{target_sql}'
         AND e.{TIME_COL} > obs.{TIME_COL}
         AND e.{TIME_COL} <= obs.{TIME_COL} + INTERVAL {delta_seconds} SECOND
        JOIN e2o rel_link
          ON rel_link.{EVENT_ID_COL} = e.{EVENT_ID_COL}
         AND rel_link.{OBJECT_ID_COL} <> obs.{OBJECT_ID_COL}
        JOIN obj rel_obj
          ON rel_obj.{OBJECT_ID_COL} = rel_link.{OBJECT_ID_COL}
         AND rel_obj.{OBJECT_TYPE_COL} = '{related_type_sql}'
    ),
    future_counts AS (
        SELECT
            obs.{OBJECT_ID_COL},
            obs.{TIME_COL},
            COUNT(DISTINCT fr.related_id) AS target
        FROM obs
        LEFT JOIN future_related fr
          ON fr.{OBJECT_ID_COL} = obs.{OBJECT_ID_COL}
         AND fr.{TIME_COL} = obs.{TIME_COL}
        GROUP BY obs.{OBJECT_ID_COL}, obs.{TIME_COL}
    )
    SELECT
        {OBJECT_ID_COL},
        {TIME_COL},
        CAST(target AS DOUBLE) AS target
    FROM future_counts
    ORDER BY {TIME_COL}, {OBJECT_ID_COL}
    """

    with ocel_connection(db, times_sorted) as con:
        return con.execute(query).df()


@check_dbs
def build_observed_pair_event_within_table(
    db: Database,
    object_types: tuple[str, str],
    timestamps: pd.Series,
    delta: pd.Timedelta,
    source_event_type: str,
    target_event_type: str,
    source_max_age: pd.Timedelta | None = None,
) -> pd.DataFrame:
    delta_seconds = int(delta.total_seconds())
    src_type_sql = _sql_quote(object_types[0])
    dst_type_sql = _sql_quote(object_types[1])
    source_sql = _sql_quote(source_event_type)
    target_sql = _sql_quote(target_event_type)
    times_sorted = pd.to_datetime(timestamps).sort_values().unique()
    recency_clause = _recency_clause("lbo", "last_time", source_max_age)

    query = f"""
    WITH pair_events AS (
        SELECT DISTINCT
            src.{OBJECT_ID_COL} AS src_id,
            dst.{OBJECT_ID_COL} AS dst_id,
            e.{EVENT_ID_COL},
            e.type,
            e.{TIME_COL}
        FROM event e
        JOIN e2o src_eo ON src_eo.{EVENT_ID_COL} = e.{EVENT_ID_COL}
        JOIN obj src
          ON src.{OBJECT_ID_COL} = src_eo.{OBJECT_ID_COL}
         AND src.{OBJECT_TYPE_COL} = '{src_type_sql}'
        JOIN e2o dst_eo
          ON dst_eo.{EVENT_ID_COL} = e.{EVENT_ID_COL}
         AND dst_eo.{OBJECT_ID_COL} <> src_eo.{OBJECT_ID_COL}
        JOIN obj dst
          ON dst.{OBJECT_ID_COL} = dst_eo.{OBJECT_ID_COL}
         AND dst.{OBJECT_TYPE_COL} = '{dst_type_sql}'
    ),
    latest_before_obs AS (
        SELECT
            t.obs_time AS {TIME_COL},
            pe.src_id,
            pe.dst_id,
            pe.type AS last_type,
            pe.{TIME_COL} AS last_time,
            ROW_NUMBER() OVER (
                PARTITION BY t.obs_time, pe.src_id, pe.dst_id
                ORDER BY pe.{TIME_COL} DESC, pe.{EVENT_ID_COL} DESC
            ) AS rn
        FROM times_df t
        JOIN pair_events pe
          ON pe.{TIME_COL} <= t.obs_time
    ),
    obs AS (
        SELECT lbo.src_id, lbo.dst_id, lbo.{TIME_COL}
        FROM latest_before_obs lbo
        WHERE lbo.rn = 1
          AND lbo.last_type = '{source_sql}'
          {recency_clause}
    ),
    future_hits AS (
        SELECT
            obs.src_id,
            obs.dst_id,
            obs.{TIME_COL},
            MAX(CASE WHEN pe.type = '{target_sql}' THEN 1 ELSE 0 END) AS target
        FROM obs
        LEFT JOIN pair_events pe
          ON pe.src_id = obs.src_id
         AND pe.dst_id = obs.dst_id
         AND pe.{TIME_COL} > obs.{TIME_COL}
         AND pe.{TIME_COL} <= obs.{TIME_COL} + INTERVAL {delta_seconds} SECOND
        GROUP BY obs.src_id, obs.dst_id, obs.{TIME_COL}
    )
    SELECT
        src_id AS {O2O_SRC_COL},
        dst_id AS {O2O_DST_COL},
        {TIME_COL},
        CAST(target AS INTEGER) AS target
    FROM future_hits
    ORDER BY {TIME_COL}, src_id, dst_id
    """

    with ocel_connection(db, times_sorted) as con:
        return con.execute(query).df()


@check_dbs
def build_observed_pair_future_event_count_table(
    db: Database,
    object_types: tuple[str, str],
    timestamps: pd.Series,
    delta: pd.Timedelta,
    source_event_type: str,
    target_event_type: str,
    source_max_age: pd.Timedelta | None = None,
) -> pd.DataFrame:
    delta_seconds = int(delta.total_seconds())
    src_type_sql = _sql_quote(object_types[0])
    dst_type_sql = _sql_quote(object_types[1])
    source_sql = _sql_quote(source_event_type)
    target_sql = _sql_quote(target_event_type)
    times_sorted = pd.to_datetime(timestamps).sort_values().unique()
    recency_clause = _recency_clause("lbo", "last_time", source_max_age)

    query = f"""
    WITH pair_events AS (
        SELECT DISTINCT
            src.{OBJECT_ID_COL} AS src_id,
            dst.{OBJECT_ID_COL} AS dst_id,
            e.{EVENT_ID_COL},
            e.type,
            e.{TIME_COL}
        FROM event e
        JOIN e2o src_eo ON src_eo.{EVENT_ID_COL} = e.{EVENT_ID_COL}
        JOIN obj src
          ON src.{OBJECT_ID_COL} = src_eo.{OBJECT_ID_COL}
         AND src.{OBJECT_TYPE_COL} = '{src_type_sql}'
        JOIN e2o dst_eo
          ON dst_eo.{EVENT_ID_COL} = e.{EVENT_ID_COL}
         AND dst_eo.{OBJECT_ID_COL} <> src_eo.{OBJECT_ID_COL}
        JOIN obj dst
          ON dst.{OBJECT_ID_COL} = dst_eo.{OBJECT_ID_COL}
         AND dst.{OBJECT_TYPE_COL} = '{dst_type_sql}'
    ),
    latest_before_obs AS (
        SELECT
            t.obs_time AS {TIME_COL},
            pe.src_id,
            pe.dst_id,
            pe.type AS last_type,
            pe.{TIME_COL} AS last_time,
            ROW_NUMBER() OVER (
                PARTITION BY t.obs_time, pe.src_id, pe.dst_id
                ORDER BY pe.{TIME_COL} DESC, pe.{EVENT_ID_COL} DESC
            ) AS rn
        FROM times_df t
        JOIN pair_events pe
          ON pe.{TIME_COL} <= t.obs_time
    ),
    obs AS (
        SELECT lbo.src_id, lbo.dst_id, lbo.{TIME_COL}
        FROM latest_before_obs lbo
        WHERE lbo.rn = 1
          AND lbo.last_type = '{source_sql}'
          {recency_clause}
    ),
    future_counts AS (
        SELECT
            obs.src_id,
            obs.dst_id,
            obs.{TIME_COL},
            COUNT(*) FILTER (WHERE pe.type = '{target_sql}') AS target
        FROM obs
        LEFT JOIN pair_events pe
          ON pe.src_id = obs.src_id
         AND pe.dst_id = obs.dst_id
         AND pe.{TIME_COL} > obs.{TIME_COL}
         AND pe.{TIME_COL} <= obs.{TIME_COL} + INTERVAL {delta_seconds} SECOND
        GROUP BY obs.src_id, obs.dst_id, obs.{TIME_COL}
    )
    SELECT
        src_id AS {O2O_SRC_COL},
        dst_id AS {O2O_DST_COL},
        {TIME_COL},
        CAST(target AS DOUBLE) AS target
    FROM future_counts
    ORDER BY {TIME_COL}, src_id, dst_id
    """

    with ocel_connection(db, times_sorted) as con:
        return con.execute(query).df()


@check_dbs
def build_stage_horizon_attribute_multiclass_table(
    db: Database,
    object_type: str,
    attribute_table_name: str,
    attribute_col: str,
    class_values: list[str],
    timestamps: pd.Series,
    delta: pd.Timedelta,
    source_event_type: str,
    source_max_age: pd.Timedelta | None = None,
) -> pd.DataFrame:
    delta_seconds = int(delta.total_seconds())
    object_type_sql = _sql_quote(object_type)
    source_sql = _sql_quote(source_event_type)
    class_list_sql = ", ".join(f"'{_sql_quote(value)}'" for value in class_values)
    class_expr = _class_case_expr(f"horizon_attr.{attribute_col}", class_values)
    times_sorted = pd.to_datetime(timestamps).sort_values().unique()
    recency_clause = _recency_clause("lbo", "last_time", source_max_age)

    query = f"""
    WITH typed_events AS (
        SELECT
            eo.{OBJECT_ID_COL},
            e.{EVENT_ID_COL},
            e.type,
            e.{TIME_COL}
        FROM event e
        JOIN e2o eo ON eo.{EVENT_ID_COL} = e.{EVENT_ID_COL}
        JOIN obj o ON o.{OBJECT_ID_COL} = eo.{OBJECT_ID_COL}
        WHERE o.{OBJECT_TYPE_COL} = '{object_type_sql}'
    ),
    latest_before_obs AS (
        SELECT
            t.obs_time AS {TIME_COL},
            te.{OBJECT_ID_COL},
            te.type AS last_type,
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
        SELECT lbo.{OBJECT_ID_COL}, lbo.{TIME_COL}
        FROM latest_before_obs lbo
        WHERE lbo.rn = 1
          AND lbo.last_type = '{source_sql}'
          {recency_clause}
    ),
    horizon_attr AS (
        SELECT
            obs.{OBJECT_ID_COL},
            obs.{TIME_COL},
            attr.{attribute_col},
            ROW_NUMBER() OVER (
                PARTITION BY obs.{OBJECT_ID_COL}, obs.{TIME_COL}
                ORDER BY attr.{TIME_COL} DESC
            ) AS rn
        FROM obs
        JOIN object_attr_values attr
          ON attr.{OBJECT_ID_COL} = obs.{OBJECT_ID_COL}
         AND attr.{TIME_COL} <= obs.{TIME_COL} + INTERVAL {delta_seconds} SECOND
        WHERE attr.{attribute_col} IN ({class_list_sql})
    )
    SELECT
        horizon_attr.{OBJECT_ID_COL},
        horizon_attr.{TIME_COL},
        CAST({class_expr} AS INTEGER) AS target
    FROM horizon_attr
    WHERE horizon_attr.rn = 1
    ORDER BY horizon_attr.{TIME_COL}, horizon_attr.{OBJECT_ID_COL}
    """

    with ocel_connection(db, times_sorted) as con:
        con.register("object_attr_values", db.table_dict[attribute_table_name].df)
        return con.execute(query).df()
