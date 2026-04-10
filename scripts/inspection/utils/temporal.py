"""Temporal aggregation tables for event and object activity."""

from typing import Literal

import pandas as pd
from relbench.base.database import Database

from data.const import TIME_COL
from data.wrapper import check_dbs

Entity = Literal["event", "object"]


def time_histogram(
    df: pd.DataFrame,
    type_col: str,
    freq: str,
    by_type: bool,
) -> pd.DataFrame:
    """Bucket timestamped rows into a time series table.

    When ``by_type`` is true and the type column is available, the summary is
    split into one line per type. Otherwise a single aggregate series is returned.
    """

    work = df.copy()
    if TIME_COL in work.columns:
        work[TIME_COL] = pd.to_datetime(work[TIME_COL], errors="coerce")
    work = work.dropna(subset=[TIME_COL]).copy()
    work["bucket"] = work[TIME_COL].dt.floor(freq)
    group_cols = ["bucket", type_col] if by_type and type_col in work.columns else ["bucket"]
    summary = (
        work.groupby(group_cols, observed=False)
        .size()
        .reset_index(name="count")
        .sort_values(group_cols, kind="stable")
        .reset_index(drop=True)
    )
    return summary


@check_dbs
def hist(
    db: Database,
    entity: Entity,
    freq: str = "D",
    by_type: bool = True,
) -> pd.DataFrame:
    """Return event or object counts over time."""

    if entity == "event":
        return time_histogram(db.table_dict["event"].df.copy(), "type", freq, by_type)
    if entity == "object":
        return time_histogram(db.table_dict["object"].df.copy(), "type", freq, by_type)
    raise ValueError(f"Unsupported entity {entity!r}.")


@check_dbs
def event_histogram(db: Database, freq: str = "D", by_type: bool = True) -> pd.DataFrame:
    return hist(db, "event", freq=freq, by_type=by_type)


@check_dbs
def object_histogram(db: Database, freq: str = "D", by_type: bool = True) -> pd.DataFrame:
    return hist(db, "object", freq=freq, by_type=by_type)


__all__ = ["event_histogram", "hist", "object_histogram", "time_histogram"]
