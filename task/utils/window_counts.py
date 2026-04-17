"""Primitive builder for window-bounded event-type count tables.

build_window_event_counts is the single base constructor for all task targets
in this project. All other task targets are derived from its output via
counts_to_target (pure Python/pandas).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from relbench.base import Database

from data.const import EVENT_TYPE_COL, OBJECT_ID_COL, O2O_SRC_COL, O2O_DST_COL, TIME_COL
from data.wrapper import check_dbs
from .db_utils import ocel_connection
from .sql_fragments import (
    sql_event_type_filter,
    sql_pair_obs_cartesian,
    sql_pair_obs_observed,
    sql_pair_window_events,
    sql_single_obs,
    sql_single_window_events,
)


@check_dbs
def build_window_event_counts(
    db: Database,
    object_types: str | tuple[str, str],
    times: pd.Series,
    delta: pd.Timedelta,
    event_types: list[str],
    pair_mode: Literal["observed", "cartesian"] = "observed",
) -> pd.DataFrame:
    """Return per-(entity, obs_time) counts of each event type in (t_obs, t_obs+Δ].

    Parameters
    ----------
    object_types:
        A single object type string for single-entity tasks, or a
        ``(src_type, dst_type)`` tuple for pair-entity tasks.
    times:
        Observation timestamps (passed through to the DuckDB context).
    delta:
        Length of the future prediction window.
    event_types:
        Explicit list of event type names. Defines the column order in the
        returned DataFrame. Must cover every event type you care about.
    pair_mode:
        Only used when ``object_types`` is a tuple.
        ``"observed"``: restrict to pairs that co-appeared before ``t_obs``.
        ``"cartesian"``: all src × dst combinations active in the window.

    Returns
    -------
    Single-entity:
        DataFrame with columns ``(object_id, time, <type_0>, ..., <type_k>)``.
    Pair:
        DataFrame with columns ``(object_id_src, object_id_dst, time, <type_0>, ..., <type_k>)``.

    All count columns are integers (≥ 0). Rows with zero total events ARE
    included so that downstream ``counts_to_target`` can distinguish "nothing
    happened" from "missing data".  Active-entity filtering ensures only
    objects that did something in ``(t_obs-Δ, t_obs]`` are candidates.
    """
    delta_seconds = int(delta.total_seconds())
    times_sorted = pd.to_datetime(times).sort_values().unique()

    # Build the conditional-aggregation SELECT clauses for each event type.
    count_cols = ", ".join(
        f"COUNT(CASE WHEN we.{EVENT_TYPE_COL} = '{t.replace(chr(39), chr(39)*2)}' THEN 1 END) "
        f"AS {_quote_col(t)}"
        for t in event_types
    )

    is_pair = isinstance(object_types, tuple)

    if not is_pair:
        obs_ctes   = sql_single_obs(object_types, delta_seconds)
        window_cte = sql_single_window_events(delta_seconds)
        entity_select = f"obs.{OBJECT_ID_COL}"
        entity_group  = f"obs.{OBJECT_ID_COL}"
        join_clause = f"LEFT JOIN window_events we ON we.{OBJECT_ID_COL} = obs.{OBJECT_ID_COL} AND we.obs_time = obs.{TIME_COL}"
        order_clause  = f"ORDER BY obs.{TIME_COL}, obs.{OBJECT_ID_COL}"
    else:
        src_type, dst_type = object_types
        if pair_mode == "observed":
            obs_ctes = sql_pair_obs_observed(src_type, dst_type, delta_seconds)
        else:
            obs_ctes = sql_pair_obs_cartesian(src_type, dst_type, delta_seconds)
        window_cte    = sql_pair_window_events(delta_seconds)
        entity_select = f"obs.src_id AS {O2O_SRC_COL}, obs.dst_id AS {O2O_DST_COL}"
        entity_group  = f"obs.src_id, obs.dst_id"
        join_clause   = f"LEFT JOIN window_events we ON we.src_id = obs.src_id AND we.dst_id = obs.dst_id AND we.obs_time = obs.{TIME_COL}"
        order_clause  = f"ORDER BY obs.{TIME_COL}, obs.src_id, obs.dst_id"

    query = f"""
    WITH
    {obs_ctes},
    {window_cte}
    SELECT
        {entity_select},
        obs.{TIME_COL},
        {count_cols}
    FROM obs
    {join_clause}
    GROUP BY {entity_group}, obs.{TIME_COL}
    {order_clause}
    """

    with ocel_connection(db, times_sorted) as con:
        df = con.execute(query).df()

    # Cast count columns to int (DuckDB returns BIGINT, which is fine, but be explicit).
    for t in event_types:
        col = _dequote_col(t)
        df[col] = df[col].fillna(0).astype(int)

    return df


def counts_to_target(
    df: pd.DataFrame,
    event_types: list[str],
    mode: Literal["total_count", "any", "specific_count", "specific_any", "multilabel"],
    specific_type: str | None = None,
) -> pd.DataFrame:
    """Derive a single ``target`` column from a wide counts DataFrame.

    Parameters
    ----------
    df:
        Output of ``build_window_event_counts``.
    event_types:
        Same list passed to ``build_window_event_counts``.
    mode:
        ``"total_count"``   — sum of all type columns → float (regression).
        ``"any"``           — total_count > 0 → int 0/1 (binary).
        ``"specific_count"``— count[specific_type] → float (regression).
        ``"specific_any"``  — count[specific_type] > 0 → int 0/1 (binary).
        ``"multilabel"``    — array [count[t]>0 for t in event_types]
                              stored as object column of np.ndarray (multilabel).
    specific_type:
        Required when mode is ``"specific_count"`` or ``"specific_any"``.

    Returns
    -------
    DataFrame with entity/time columns and a single ``target`` column.
    The wide count columns are dropped.
    """
    count_cols = [_dequote_col(t) for t in event_types]
    entity_cols = [c for c in df.columns if c not in count_cols and c != TIME_COL]

    out = df[entity_cols + [TIME_COL]].copy()

    if mode == "total_count":
        out["target"] = df[count_cols].sum(axis=1).astype(float)

    elif mode == "any":
        out["target"] = (df[count_cols].sum(axis=1) > 0).astype(int)

    elif mode == "specific_count":
        if specific_type is None:
            raise ValueError("specific_type required for mode='specific_count'")
        col = _dequote_col(specific_type)
        out["target"] = df[col].astype(float)

    elif mode == "specific_any":
        if specific_type is None:
            raise ValueError("specific_type required for mode='specific_any'")
        col = _dequote_col(specific_type)
        out["target"] = (df[col] > 0).astype(int)

    elif mode == "multilabel":
        arr = df[count_cols].values > 0          # (N, K) bool
        out["target"] = list(arr.astype(np.float32))

    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    return out


# ---------------------------------------------------------------------------
# Column name helpers
# ---------------------------------------------------------------------------

def _quote_col(event_type: str) -> str:
    """DuckDB double-quoted identifier for an event type used as a column name."""
    escaped = event_type.replace('"', '""')
    return f'"{escaped}"'


def _dequote_col(event_type: str) -> str:
    """Python-side column name matching what DuckDB stores after _quote_col."""
    return event_type
