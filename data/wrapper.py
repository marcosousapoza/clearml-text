from __future__ import annotations

from functools import wraps
import inspect
from typing import Any, Callable, Iterable, ParamSpec, TypeVar, get_args, get_origin, get_type_hints

import pandas as pd
from relbench.base.database import Database

from .const import (
    E2O_EVENT_ID_COL,
    E2O_OBJECT_ID_COL,
    E2O_TABLE,
    EVENT_ID_COL,
    EVENT_TABLE,
    EVENT_TYPE_COL,
    O2O_DST_COL,
    O2O_SRC_COL,
    O2O_TABLE,
    OBJECT_ID_COL,
    OBJECT_TABLE,
    OBJECT_TYPE_COL,
    QUALIFIER_COL,
    TIME_COL,
)


_MAIN_TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    EVENT_TABLE: (EVENT_ID_COL, EVENT_TYPE_COL, TIME_COL),
    OBJECT_TABLE: (OBJECT_ID_COL, OBJECT_TYPE_COL, TIME_COL),
    E2O_TABLE: (E2O_EVENT_ID_COL, E2O_OBJECT_ID_COL),
}

_NON_NULL_COLUMNS: dict[str, tuple[str, ...]] = {
    EVENT_TABLE: (EVENT_ID_COL, EVENT_TYPE_COL),
    OBJECT_TABLE: (OBJECT_ID_COL, OBJECT_TYPE_COL),
    E2O_TABLE: (E2O_EVENT_ID_COL, E2O_OBJECT_ID_COL),
    O2O_TABLE: (O2O_SRC_COL, O2O_DST_COL),
}

P = ParamSpec("P")
R = TypeVar("R")


def _require_tables(db: Database, names: Iterable[str]) -> None:
    missing = [name for name in names if name not in db.table_dict]
    if missing:
        raise ValueError(f"Database missing required table(s): {', '.join(repr(name) for name in missing)}.")


def _require_columns(df: pd.DataFrame, columns: Iterable[str], table_name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"Table {table_name!r} missing required column(s): {', '.join(repr(column) for column in missing)}."
        )


def _require_non_null(df: pd.DataFrame, columns: Iterable[str], table_name: str) -> None:
    bad = [column for column in columns if column in df.columns and df[column].isna().any()]
    if bad:
        raise ValueError(
            f"Table {table_name!r} contains null values in required column(s): {', '.join(repr(column) for column in bad)}."
        )


def _require_unique(df: pd.DataFrame, column: str, table_name: str) -> None:
    if df[column].duplicated().any():
        raise ValueError(f"Table {table_name!r} must have unique values in {column!r}.")


def _require_references(
    left: pd.Series,
    right: pd.Series,
    *,
    source_table: str,
    source_col: str,
    target_table: str,
    target_col: str,
) -> None:
    missing = pd.Index(left.dropna().unique()).difference(pd.Index(right.dropna().unique()))
    if not missing.empty:
        preview = ", ".join(repr(str(value)) for value in missing[:5].tolist())
        raise ValueError(
            f"Dangling reference(s) from {source_table!r}.{source_col!r} to {target_table!r}.{target_col!r}: {preview}."
        )


def check_db(db: Database) -> Database:
    """Validate the core OCEL/RelBench tables and return the database.

    The checks are intentionally structural:
    - required core tables exist
    - required columns exist
    - primary and link identifiers are non-null
    - primary identifiers are unique
    - link tables do not contain dangling references
    """

    _require_tables(db, (EVENT_TABLE, OBJECT_TABLE, E2O_TABLE))

    event_df = db.table_dict[EVENT_TABLE].df
    object_df = db.table_dict[OBJECT_TABLE].df
    e2o_df = db.table_dict[E2O_TABLE].df
    o2o_df = db.table_dict[O2O_TABLE].df if O2O_TABLE in db.table_dict else None

    for table_name, df in (
        (EVENT_TABLE, event_df),
        (OBJECT_TABLE, object_df),
        (E2O_TABLE, e2o_df),
    ):
        _require_columns(df, _MAIN_TABLE_COLUMNS[table_name], table_name)
        _require_non_null(df, _NON_NULL_COLUMNS[table_name], table_name)

    _require_unique(event_df, EVENT_ID_COL, EVENT_TABLE)
    _require_unique(object_df, OBJECT_ID_COL, OBJECT_TABLE)

    if QUALIFIER_COL in e2o_df.columns and e2o_df[QUALIFIER_COL].isna().any():
        raise ValueError(f"Table {E2O_TABLE!r} contains null values in required column {QUALIFIER_COL!r}.")

    _require_references(
        e2o_df[E2O_EVENT_ID_COL],
        event_df[EVENT_ID_COL],
        source_table=E2O_TABLE,
        source_col=E2O_EVENT_ID_COL,
        target_table=EVENT_TABLE,
        target_col=EVENT_ID_COL,
    )
    _require_references(
        e2o_df[E2O_OBJECT_ID_COL],
        object_df[OBJECT_ID_COL],
        source_table=E2O_TABLE,
        source_col=E2O_OBJECT_ID_COL,
        target_table=OBJECT_TABLE,
        target_col=OBJECT_ID_COL,
    )

    if o2o_df is not None:
        _require_columns(o2o_df, (O2O_SRC_COL, O2O_DST_COL), O2O_TABLE)
        _require_non_null(o2o_df, _NON_NULL_COLUMNS[O2O_TABLE], O2O_TABLE)
        _require_references(
            o2o_df[O2O_SRC_COL],
            object_df[OBJECT_ID_COL],
            source_table=O2O_TABLE,
            source_col=O2O_SRC_COL,
            target_table=OBJECT_TABLE,
            target_col=OBJECT_ID_COL,
        )
        _require_references(
            o2o_df[O2O_DST_COL],
            object_df[OBJECT_ID_COL],
            source_table=O2O_TABLE,
            source_col=O2O_DST_COL,
            target_table=OBJECT_TABLE,
            target_col=OBJECT_ID_COL,
        )

    return db


def _is_database_annotation(annotation: Any) -> bool:
    if annotation is inspect._empty:
        return False
    if annotation is Database:
        return True
    origin = get_origin(annotation)
    if origin is None:
        return isinstance(annotation, type) and issubclass(annotation, Database)
    return any(_is_database_annotation(arg) for arg in get_args(annotation))


def check_dbs(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator that validates all `Database`-typed arguments via annotations."""
    signature = inspect.signature(func)
    hints = get_type_hints(func)
    db_params = [
        name
        for name, param in signature.parameters.items()
        if _is_database_annotation(hints.get(name, param.annotation))
    ]

    @wraps(func)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        bound = signature.bind_partial(*args, **kwargs)
        seen: set[int] = set()
        for name in db_params:
            if name not in bound.arguments:
                continue
            value = bound.arguments[name]
            if not isinstance(value, Database):
                raise TypeError(
                    f"Argument {name!r} to {func.__qualname__} must be a Database, got {type(value).__name__}."
                )
            value_id = id(value)
            if value_id not in seen:
                check_db(value)
                seen.add(value_id)
        return func(*args, **kwargs)

    return wrapped
