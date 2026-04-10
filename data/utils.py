from fnmatch import fnmatch
from typing import Any, Literal

import duckdb
import numpy as np
import pandas as pd
from relbench.base import Database
from relbench.base.table import Table
from relbench.modeling.utils import to_unix_time

from .const import (
    E2O_TABLE,
    EVENT_ID_COL,
    EVENT_TABLE,
    EVENT_TYPE_COL,
    OBJECT_ID_COL,
    OBJECT_TABLE,
    OBJECT_TYPE_COL,
    QUALIFIER_COL,
    TIME_COL,
)
from .wrapper import check_dbs


def _resolve_stype(stype_mod: Any, value: Any) -> Any:
    if isinstance(value, str):
        return getattr(stype_mod, value, value)
    return value


Entity = Literal["event", "object"]


def _entity_specs(entity: Entity) -> tuple[str, str, str]:
    if entity == "event":
        return EVENT_TABLE, EVENT_ID_COL, EVENT_TYPE_COL
    if entity == "object":
        return OBJECT_TABLE, OBJECT_ID_COL, OBJECT_TYPE_COL
    raise ValueError(f"Unsupported entity {entity!r}.")


def _normalize_filter(values: str | list[str] | tuple[str, ...] | set[str] | None) -> list[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        normalized = [values]
    else:
        normalized = [str(value) for value in values]
    return sorted(set(normalized))


@check_dbs
def count(db: Database, entity: Entity) -> int:
    table_name, id_col, _ = _entity_specs(entity)
    return int(db.table_dict[table_name].df[id_col].nunique())


@check_dbs
def types(
    db: Database,
    entity: Entity,
    *,
    linked_to: str | list[str] | tuple[str, ...] | set[str] | None = None,
) -> list[str]:
    table_name, _id_col, type_col = _entity_specs(entity)
    table = db.table_dict[table_name].df

    selected = _normalize_filter(linked_to)
    if selected is None:
        return sorted(table[type_col].dropna().astype(str).unique().tolist())
    if not selected:
        return []

    con = duckdb.connect()
    try:
        con.register("evts", db.table_dict[EVENT_TABLE].df)
        con.register("objs", db.table_dict[OBJECT_TABLE].df)
        con.register("e2o", db.table_dict[E2O_TABLE].df)
        if entity == "event":
            con.register("selected", pd.DataFrame({"linked_type": selected}))
            result = con.execute(
                f"""
                SELECT DISTINCT CAST(ev.{EVENT_TYPE_COL} AS VARCHAR) AS type
                FROM e2o AS eo
                INNER JOIN objs AS ob
                    ON eo.{OBJECT_ID_COL} = ob.{OBJECT_ID_COL}
                INNER JOIN selected AS sel
                    ON CAST(ob.{OBJECT_TYPE_COL} AS VARCHAR) = sel.linked_type
                INNER JOIN evts AS ev
                    ON eo.{EVENT_ID_COL} = ev.{EVENT_ID_COL}
                WHERE ev.{EVENT_TYPE_COL} IS NOT NULL
                ORDER BY type
                """
            ).df()
        else:
            con.register("selected", pd.DataFrame({"linked_type": selected}))
            result = con.execute(
                f"""
                SELECT DISTINCT CAST(ob.{OBJECT_TYPE_COL} AS VARCHAR) AS type
                FROM e2o AS eo
                INNER JOIN evts AS ev
                    ON eo.{EVENT_ID_COL} = ev.{EVENT_ID_COL}
                INNER JOIN selected AS sel
                    ON CAST(ev.{EVENT_TYPE_COL} AS VARCHAR) = sel.linked_type
                INNER JOIN objs AS ob
                    ON eo.{OBJECT_ID_COL} = ob.{OBJECT_ID_COL}
                WHERE ob.{OBJECT_TYPE_COL} IS NOT NULL
                ORDER BY type
                """
            ).df()
    finally:
        con.close()
    return result["type"].tolist()


@check_dbs
def num_objects(db: Database):
    return count(db, "object")


@check_dbs
def num_events(db: Database):
    return count(db, "event")


@check_dbs
def event_types(db: Database, object_types: str | list[str] | tuple[str, ...] | set[str] | None = None):
    return types(db, "event", linked_to=object_types)


@check_dbs
def object_types(db: Database, event_types: str | list[str] | tuple[str, ...] | set[str] | None = None):
    return types(db, "object", linked_to=event_types)


def _max_duration_for_objects(db: Database, object_type: str, *, mode: Literal["lifetime", "interevent"]) -> pd.Timedelta:
    event = db.table_dict[EVENT_TABLE].df
    obj = db.table_dict[OBJECT_TABLE].df
    e2o = db.table_dict[E2O_TABLE].df
    con = duckdb.connect()
    con.register("event", event)
    con.register("obj", obj)
    con.register("e2o", e2o)
    try:
        if mode == "lifetime":
            query = f"""
                WITH typed_object AS (
                    SELECT {OBJECT_ID_COL}, {TIME_COL} AS object_time
                    FROM obj
                    WHERE {OBJECT_TYPE_COL} = '{object_type}'
                ),
                object_event_span AS (
                    SELECT
                        o.{OBJECT_ID_COL},
                        o.object_time,
                        MAX(event.{TIME_COL}) AS last_event_time
                    FROM typed_object AS o
                    JOIN e2o
                      ON e2o.{OBJECT_ID_COL} = o.{OBJECT_ID_COL}
                    JOIN event
                      ON event.{EVENT_ID_COL} = e2o.{EVENT_ID_COL}
                    GROUP BY o.{OBJECT_ID_COL}, o.object_time
                )
                SELECT MAX(EXTRACT(epoch FROM (last_event_time - object_time))) AS max_duration_seconds
                FROM object_event_span
            """
        else:
            query = f"""
                WITH typed_object AS (
                    SELECT {OBJECT_ID_COL}
                    FROM obj
                    WHERE {OBJECT_TYPE_COL} = '{object_type}'
                ),
                ordered_events AS (
                    SELECT
                        e2o.{OBJECT_ID_COL},
                        event.{TIME_COL} AS event_time,
                        LAG(event.{TIME_COL}) OVER (
                            PARTITION BY e2o.{OBJECT_ID_COL}
                            ORDER BY event.{TIME_COL}, event.{EVENT_ID_COL}
                        ) AS previous_event_time
                    FROM typed_object AS o
                    JOIN e2o
                      ON e2o.{OBJECT_ID_COL} = o.{OBJECT_ID_COL}
                    JOIN event
                      ON event.{EVENT_ID_COL} = e2o.{EVENT_ID_COL}
                )
                SELECT MAX(EXTRACT(epoch FROM (event_time - previous_event_time))) AS max_duration_seconds
                FROM ordered_events
                WHERE previous_event_time IS NOT NULL
            """
        result = con.execute(query).fetchone()
    finally:
        con.close()

    max_duration_seconds = result[0] if result is not None else None
    if max_duration_seconds is None:
        raise RuntimeError(f"Could not derive max {mode} for object type {object_type!r}.")
    return pd.to_timedelta(float(max_duration_seconds), unit="s")


@check_dbs
def max_lifetime(db: Database, object_type: str) -> pd.Timedelta:
    return _max_duration_for_objects(db, object_type, mode="lifetime")


@check_dbs
def max_interevent(db: Database, object_type: str) -> pd.Timedelta:
    return _max_duration_for_objects(db, object_type, mode="interevent")


def series_to_unix_seconds(values: pd.Series) -> np.ndarray:
    """Convert a timestamp-like Series to UNIX seconds.

    NaT values are kept as the pandas int64 sentinel (minimum int64), matching
    existing behavior in the codebase for downstream masking.
    """
    ts = pd.to_datetime(values, errors="coerce")
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert(None)

    out = np.full(shape=(len(ts),), fill_value=np.iinfo(np.int64).min, dtype=np.int64)
    mask = ~ts.isna()
    if bool(mask.any()):
        out[mask.to_numpy()] = to_unix_time(ts[mask].astype("datetime64[ns]"))
    return out


def get_stype_proposal(
    db: Any, stype_suggestion: dict[str, dict[str, Any]] | None = None
) -> Any:
    """
    OCEL-specific wrapper around `relbench.modeling.utils.get_stype_proposal`,
    with optional overrides via `stype_suggestion`.

    RelBench/torch-frame will often treat string columns as text by default. For
    OCEL tables we only force:
      - `time`: timestamp

    Structural string columns such as `type`, `qualifier`, and
    `edge_object_type` are left on the default RelBench inference path so they
    are not hard-coded to `stype.categorical`.

    Override format:
      stype_suggestion = {
          "event_attr_*": {"resource": "categorical"},
          "object_attr_items": {"price": "numerical"},
      }
    or a callable:
      stype_suggestion(db) -> dict[table_name, dict[col, stype]]
    """
    from relbench.modeling.utils import get_stype_proposal as _get_stype_proposal

    col_to_stype = _get_stype_proposal(db)
    if not isinstance(col_to_stype, dict) or not hasattr(db, "table_dict"):
        return col_to_stype

    try:
        from torch_frame import stype as stype_mod
    except Exception:
        return col_to_stype

    categorical = getattr(stype_mod, "categorical", None)
    timestamp = getattr(stype_mod, "timestamp", None)

    raw_suggestion: Any
    if callable(stype_suggestion):
        raw_suggestion = stype_suggestion(db)
    else:
        raw_suggestion = stype_suggestion
    overrides: dict[str, dict[str, Any]] = (
        raw_suggestion if isinstance(raw_suggestion, dict) else {}
    )

    out: dict[str, Any] = {}
    for table_name, table in db.table_dict.items():
        base = col_to_stype.get(table_name, {})
        if not isinstance(base, dict):
            continue
        table_cols = set(table.df.columns)
        table_map = {col: st for col, st in base.items() if col in table_cols}
        if timestamp is not None and TIME_COL in table_cols:
            table_map[TIME_COL] = timestamp

        for pattern, col_map in overrides.items():
            if not isinstance(col_map, dict):
                continue
            if pattern != table_name and not fnmatch(table_name, pattern):
                continue
            for col, st in col_map.items():
                if col in table_cols:
                    resolved = _resolve_stype(stype_mod, st)
                    if (
                        resolved is categorical
                        and table.df[col].notna().sum() == 0
                    ):
                        continue
                    table_map[col] = resolved
        out[table_name] = table_map
    return out
