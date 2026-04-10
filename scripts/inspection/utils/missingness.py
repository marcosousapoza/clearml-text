"""Attribute completeness summaries for event and object attribute tables."""

import pandas as pd
from relbench.base.database import Database

from data.const import EVENT_ATTR_TABLE_PREFIX, EVENT_ID_COL, OBJECT_ATTR_TABLE_PREFIX, OBJECT_ID_COL, TIME_COL
from data.wrapper import check_dbs


@check_dbs
def attribute_non_null_summary(db: Database, kind: str) -> pd.DataFrame:
    """Summarize non-null coverage for event-side or object-side attributes.

    ``kind`` must be ``"event"`` or ``"object"``. Identifier and timestamp
    columns are skipped because they are structural rather than predictive
    attributes.
    """

    prefix = EVENT_ATTR_TABLE_PREFIX if kind == "event" else OBJECT_ATTR_TABLE_PREFIX
    rows = []
    for table_name, table in db.table_dict.items():
        if not table_name.startswith(prefix):
            continue
        for col in table.df.columns:
            if col in {EVENT_ID_COL, OBJECT_ID_COL, TIME_COL}:
                continue
            non_null = int(table.df[col].notna().sum())
            total = int(len(table.df))
            rows.append(
                {
                    "table": table_name,
                    "column": col,
                    "rows": total,
                    "non_null": non_null,
                    "non_null_rate": (non_null / total) if total else float("nan"),
                }
            )
    columns = ["table", "column", "rows", "non_null", "non_null_rate"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["table", "column"], kind="stable")
        .reset_index(drop=True)
    )


__all__ = ["attribute_non_null_summary"]
