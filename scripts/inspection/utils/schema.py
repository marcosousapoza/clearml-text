"""Schema-level inspection helpers."""

import pandas as pd
from relbench.base.database import Database

from data.const import TIME_COL
from data.wrapper import check_dbs


@check_dbs
def table_row_counts(db: Database) -> pd.DataFrame:
    """Summarize row and column counts for every table in the database."""

    rows = []
    for name, table in db.table_dict.items():
        rows.append({"table": name, "rows": len(table.df), "columns": len(table.df.columns)})
    return pd.DataFrame(rows).sort_values("table", kind="stable").reset_index(drop=True)


@check_dbs
def describe_tables(db: Database) -> dict[str, pd.DataFrame]:
    """Return a compact schema report for table sizes and column dtypes.

    The result contains two DataFrames:
    ``table_overview`` with one row per table and ``column_dtypes`` with one
    row per column.
    """

    rows: list[dict[str, object]] = []
    for name, table in db.table_dict.items():
        for column, dtype in table.df.dtypes.astype(str).items():
            rows.append({"table": name, "column": column, "dtype": dtype})
    schema = pd.DataFrame(rows)
    return {
        "table_overview": table_row_counts(db).assign(
            num_columns=lambda frame: frame["table"].map(
                {name: len(table.df.columns) for name, table in db.table_dict.items()}
            )
        ),
        "column_dtypes": schema.sort_values(["table", "column"], kind="stable").reset_index(drop=True),
    }


@check_dbs
def inspect_attribute_dtypes(
    db: Database,
    columns: set[str] | None = None,
    preview_rows: int = 10,
) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Inspect selected attribute columns across all tables.

    The function searches case-insensitively for the requested column names and
    returns, for each matching table, a dtype summary plus a small preview of
    the raw values.
    """

    targets = {c.lower() for c in (columns or {"weight", "status"})}
    out: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    for table_name, table in db.table_dict.items():
        selected = [c for c in table.df.columns if str(c).lower() in targets]
        if not selected:
            continue
        dtypes = (
            table.df[selected]
            .dtypes.astype(str)
            .rename("dtype")
            .reset_index()
            .rename(columns={"index": "column"})
        )
        preview = table.df[selected].head(preview_rows).copy()
        if TIME_COL in preview.columns:
            preview[TIME_COL] = pd.to_datetime(preview[TIME_COL], errors="coerce")
        out.append((table_name, dtypes, preview))
    return out


__all__ = ["describe_tables", "inspect_attribute_dtypes", "table_row_counts"]
