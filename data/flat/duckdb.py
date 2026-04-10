from __future__ import annotations

import duckdb
import pandas as pd

from relbench.base.database import Database
from relbench.base.table import Table

from data.const import (
    E2O_EVENT_ID_COL,
    E2O_OBJECT_ID_COL,
    E2O_TABLE,
    EVENT_ATTR_TABLE_PREFIX,
    EVENT_ID_COL,
    EVENT_TABLE,
    O2O_DST_COL,
    O2O_SRC_COL,
    O2O_TABLE,
    OBJECT_ATTR_TABLE_PREFIX,
    OBJECT_ID_COL,
    OBJECT_TABLE,
    OBJECT_TYPE_COL,
    QUALIFIER_COL,
    TIME_COL,
)
from data.datareader.relbench_tables import apply_default_column_dtypes_to_df
from data.wrapper import check_dbs


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _select_columns(columns: list[str], *, exclude: set[str] | None = None, alias: str) -> str:
    exclude = exclude or set()
    selected = [f'{alias}.{_quote(column)}' for column in columns if column not in exclude]
    return ", ".join(selected)


def _copy_table(
    source: Table,
    df: pd.DataFrame,
    *,
    fkey_col_to_pkey_table: dict[str, str] | None = None,
    pkey_col: str | None = None,
) -> Table:
    return Table(
        df=apply_default_column_dtypes_to_df(df),
        time_col=source.time_col,
        pkey_col=source.pkey_col if pkey_col is None else pkey_col,
        fkey_col_to_pkey_table=(
            dict(source.fkey_col_to_pkey_table)
            if fkey_col_to_pkey_table is None
            else fkey_col_to_pkey_table
        ),
    )


def _order_by(columns: list[str], *, alias: str, id_col: str) -> str:
    order_cols: list[str] = []
    if TIME_COL in columns:
        order_cols.append(f"{alias}.{_quote(TIME_COL)}")
    order_cols.append(f"{alias}.{_quote(id_col)}")
    return ", ".join(order_cols)


@check_dbs
def flatten(db: Database, object_type: str) -> Database:
    """
    Flatten an OCEL-style RelBench database with respect to a single object type.

    The transform is executed in DuckDB and performs two operations:
    1. Remove all objects that are not of the requested type.
    2. Duplicate events so each flattened event is associated with at most one object.

    Flattened event identifiers are deterministic composite ids of the form
    `<original_event_id>::<object_id>::<ordinal>`.
    """
    object_df = db.table_dict[OBJECT_TABLE].df
    if OBJECT_TYPE_COL not in object_df.columns:
        raise ValueError(f"{OBJECT_TABLE!r} missing required column {OBJECT_TYPE_COL!r}.")
    if object_type not in set(object_df[OBJECT_TYPE_COL].astype(str)):
        raise ValueError(f"Unknown object type {object_type!r}.")

    con = duckdb.connect()
    try:
        for name, table in db.table_dict.items():
            con.register(name, table.df)

        e2o_order_expr = (
            f"COALESCE(CAST(e2o.{_quote(QUALIFIER_COL)} AS VARCHAR), '')"
            if QUALIFIER_COL in db.table_dict[E2O_TABLE].df.columns
            else f"CAST(e2o.{_quote(E2O_OBJECT_ID_COL)} AS VARCHAR)"
        )

        con.execute(
            f"""
            CREATE TEMP TABLE target_object AS
            SELECT *
            FROM {OBJECT_TABLE}
            WHERE {_quote(OBJECT_TYPE_COL)} = ?
            """,
            [object_type],
        )
        con.execute(
            f"""
            CREATE TEMP TABLE flat_event_map AS
            SELECT
                e2o.{_quote(E2O_EVENT_ID_COL)} AS original_event_id,
                e2o.{_quote(E2O_OBJECT_ID_COL)} AS {_quote(OBJECT_ID_COL)},
                e2o.{_quote(E2O_EVENT_ID_COL)}
                    || '::'
                    || e2o.{_quote(E2O_OBJECT_ID_COL)}
                    || '::'
                    || CAST(
                        ROW_NUMBER() OVER (
                            PARTITION BY e2o.{_quote(E2O_EVENT_ID_COL)}, e2o.{_quote(E2O_OBJECT_ID_COL)}
                            ORDER BY {e2o_order_expr}
                        ) AS VARCHAR
                    ) AS {_quote(EVENT_ID_COL)}
            FROM {E2O_TABLE} AS e2o
            INNER JOIN target_object AS obj
                ON obj.{_quote(OBJECT_ID_COL)} = e2o.{_quote(E2O_OBJECT_ID_COL)}
            """
        )

        flat_object = con.execute("SELECT * FROM target_object").df()

        event_cols = db.table_dict[EVENT_TABLE].df.columns.tolist()
        event_passthrough = _select_columns(
            event_cols,
            exclude={EVENT_ID_COL},
            alias="event",
        )
        flat_event = con.execute(
            f"""
            SELECT
                map.{_quote(EVENT_ID_COL)} AS {_quote(EVENT_ID_COL)}
                {", " if event_passthrough else ""}{event_passthrough}
            FROM {EVENT_TABLE} AS event
            INNER JOIN flat_event_map AS map
                ON map.original_event_id = event.{_quote(EVENT_ID_COL)}
            ORDER BY {_order_by(event_cols, alias="event", id_col=EVENT_ID_COL)}, map.{_quote(EVENT_ID_COL)}
            """
        ).df()

        e2o_cols = db.table_dict[E2O_TABLE].df.columns.tolist()
        e2o_passthrough = _select_columns(
            e2o_cols,
            exclude={E2O_EVENT_ID_COL, E2O_OBJECT_ID_COL},
            alias="e2o",
        )
        flat_e2o = con.execute(
            f"""
            SELECT
                map.{_quote(EVENT_ID_COL)} AS {_quote(E2O_EVENT_ID_COL)},
                map.{_quote(OBJECT_ID_COL)} AS {_quote(E2O_OBJECT_ID_COL)}
                {", " if e2o_passthrough else ""}{e2o_passthrough}
            FROM {E2O_TABLE} AS e2o
            INNER JOIN flat_event_map AS map
                ON map.original_event_id = e2o.{_quote(E2O_EVENT_ID_COL)}
               AND map.{_quote(OBJECT_ID_COL)} = e2o.{_quote(E2O_OBJECT_ID_COL)}
            ORDER BY {_order_by(e2o_cols, alias="e2o", id_col=E2O_EVENT_ID_COL)}, map.{_quote(OBJECT_ID_COL)}
            """
        ).df()

        tables: dict[str, Table] = {
            EVENT_TABLE: _copy_table(db.table_dict[EVENT_TABLE], flat_event, pkey_col=EVENT_ID_COL),
            OBJECT_TABLE: _copy_table(db.table_dict[OBJECT_TABLE], flat_object, pkey_col=OBJECT_ID_COL),
            E2O_TABLE: _copy_table(
                db.table_dict[E2O_TABLE],
                flat_e2o,
                fkey_col_to_pkey_table={
                    E2O_EVENT_ID_COL: EVENT_TABLE,
                    E2O_OBJECT_ID_COL: OBJECT_TABLE,
                },
            ),
        }

        if O2O_TABLE in db.table_dict:
            flat_o2o = con.execute(
                f"""
                SELECT o2o.*
                FROM {O2O_TABLE} AS o2o
                INNER JOIN target_object AS src
                    ON src.{_quote(OBJECT_ID_COL)} = o2o.{_quote(O2O_SRC_COL)}
                INNER JOIN target_object AS dst
                    ON dst.{_quote(OBJECT_ID_COL)} = o2o.{_quote(O2O_DST_COL)}
                """
            ).df()
            tables[O2O_TABLE] = _copy_table(
                db.table_dict[O2O_TABLE],
                flat_o2o,
                fkey_col_to_pkey_table={
                    O2O_SRC_COL: OBJECT_TABLE,
                    O2O_DST_COL: OBJECT_TABLE,
                },
            )

        for name, table in db.table_dict.items():
            if name.startswith(EVENT_ATTR_TABLE_PREFIX):
                attr_cols = table.df.columns.tolist()
                attr_passthrough = _select_columns(attr_cols, exclude={EVENT_ID_COL}, alias="attr")
                flat_attr = con.execute(
                    f"""
                    SELECT
                        map.{_quote(EVENT_ID_COL)} AS {_quote(EVENT_ID_COL)}
                        {", " if attr_passthrough else ""}{attr_passthrough}
                    FROM {name} AS attr
                    INNER JOIN flat_event_map AS map
                        ON map.original_event_id = attr.{_quote(EVENT_ID_COL)}
                    ORDER BY {_order_by(attr_cols, alias="attr", id_col=EVENT_ID_COL)}, map.{_quote(EVENT_ID_COL)}
                    """
                ).df()
                tables[name] = _copy_table(
                    table,
                    flat_attr,
                    fkey_col_to_pkey_table={EVENT_ID_COL: EVENT_TABLE},
                )
            elif name == f"{OBJECT_ATTR_TABLE_PREFIX}{object_type}":
                flat_attr = con.execute(
                    f"""
                    SELECT attr.*
                    FROM {name} AS attr
                    INNER JOIN target_object AS obj
                        ON obj.{_quote(OBJECT_ID_COL)} = attr.{_quote(OBJECT_ID_COL)}
                    """
                ).df()
                tables[name] = _copy_table(
                    table,
                    flat_attr,
                    fkey_col_to_pkey_table={OBJECT_ID_COL: OBJECT_TABLE},
                )

        return Database(table_dict=tables)
    finally:
        con.close()
