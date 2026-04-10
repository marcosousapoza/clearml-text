"""Timing analyses over event-object relationships."""

import pandas as pd
from relbench.base.database import Database

from data.const import EVENT_ID_COL, EVENT_TYPE_COL, OBJECT_ID_COL, OBJECT_TYPE_COL, TIME_COL
from data.wrapper import check_dbs
from ._shared import (
    summarize_delta,
)


@check_dbs
def event_object_join(db: Database) -> pd.DataFrame:
    """Join event, object, and event-object tables into one analysis frame.

    The result keeps separate event, object, and link timestamps so downstream
    timing summaries can choose the most appropriate notion of activity time.
    """

    event_df = db.table_dict["event"].df.copy()
    object_df = db.table_dict["object"].df.copy()
    e2o_df = db.table_dict["e2o"].df.copy()
    for frame in (event_df, object_df, e2o_df):
        if TIME_COL in frame.columns:
            frame[TIME_COL] = pd.to_datetime(frame[TIME_COL], errors="coerce")
    if TIME_COL in e2o_df.columns:
        join_time = e2o_df[[EVENT_ID_COL, OBJECT_ID_COL, TIME_COL]].copy()
    else:
        join_time = e2o_df[[EVENT_ID_COL, OBJECT_ID_COL]].copy()
    joined = (
        join_time.merge(
            event_df[[EVENT_ID_COL, TIME_COL, EVENT_TYPE_COL]].rename(
                columns={EVENT_TYPE_COL: "event_type"}
            ),
            on=EVENT_ID_COL,
            how="left",
            suffixes=("", "_event"),
        )
        .merge(
            object_df[[OBJECT_ID_COL, TIME_COL, OBJECT_TYPE_COL]].rename(
                columns={OBJECT_TYPE_COL: "object_type"}
            ),
            on=OBJECT_ID_COL,
            how="left",
            suffixes=("_e2o", "_object"),
        )
    )
    if f"{TIME_COL}_e2o" in joined.columns:
        joined["activity_time"] = joined[f"{TIME_COL}_e2o"]
    else:
        joined["activity_time"] = joined[TIME_COL]
    joined["event_time"] = (
        joined[f"{TIME_COL}_event"] if f"{TIME_COL}_event" in joined.columns else joined[TIME_COL]
    )
    joined["object_time"] = (
        joined[f"{TIME_COL}_object"] if f"{TIME_COL}_object" in joined.columns else pd.NaT
    )
    return joined


@check_dbs
def event_object_matrix(db: Database) -> pd.DataFrame:
    """Summarize how long objects existed before each event type touched them."""

    joined = event_object_join(db)
    joined["delta_hours"] = (joined["event_time"] - joined["object_time"]).dt.total_seconds() / 3600.0
    return summarize_delta(
        joined,
        ["event_type", "object_type"],
        "delta_hours",
        "delta",
    )


@check_dbs
def event_oldest_k_matrix(db: Database, k: int = 5) -> pd.DataFrame:
    """Measure event-to-object age gaps for the oldest linked objects per event.

    Objects are ranked by creation time within each event, then the first ``k``
    ranks are summarized separately.
    """

    joined = event_object_join(db)
    work = joined.dropna(subset=["event_time", "object_time"]).copy()
    work = work.sort_values([EVENT_ID_COL, "object_time"], kind="stable")
    work["rank"] = work.groupby(EVENT_ID_COL).cumcount() + 1
    work = work[work["rank"] <= k].copy()
    work["delta_hours"] = (work["event_time"] - work["object_time"]).dt.total_seconds() / 3600.0
    return summarize_delta(work, ["event_type", "rank"], "delta_hours", "delta")


@check_dbs
def event_object_recent_matrix(db: Database) -> pd.DataFrame:
    """Summarize recency gaps between object activities and the current event.

    For each object, the function looks one activity backward in time and
    aggregates the elapsed hours by event and object type.
    """

    joined = event_object_join(db)
    work = joined.dropna(subset=["activity_time"]).copy()
    work = work.sort_values([OBJECT_ID_COL, "activity_time", EVENT_ID_COL], kind="stable")
    work["prev_activity_time"] = work.groupby(OBJECT_ID_COL)["activity_time"].shift(1)
    work["prev_delta_hours"] = (
        work["activity_time"] - work["prev_activity_time"]
    ).dt.total_seconds() / 3600.0
    return summarize_delta(
        work,
        ["event_type", "object_type"],
        "prev_delta_hours",
        "prev_delta",
    )


@check_dbs
def event_recent_k_matrix(db: Database, k: int = 5) -> pd.DataFrame:
    """Extend recency analysis to multiple previous object activities.

    The function computes lagged activity timestamps from 1 through ``k`` and
    stacks the resulting gaps into a rank-indexed summary.
    """

    joined = event_object_join(db)
    work = joined.dropna(subset=["activity_time"]).copy()
    work = work.sort_values([OBJECT_ID_COL, "activity_time", EVENT_ID_COL], kind="stable")
    pieces = []
    for lag in range(1, k + 1):
        lagged = work.copy()
        lagged["lag_time"] = lagged.groupby(OBJECT_ID_COL)["activity_time"].shift(lag)
        lagged["lag_rank"] = lag
        lagged["prev_delta_hours"] = (
            lagged["activity_time"] - lagged["lag_time"]
        ).dt.total_seconds() / 3600.0
        pieces.append(lagged[["event_type", "lag_rank", "prev_delta_hours"]])
    stacked = pd.concat(pieces, ignore_index=True)
    return summarize_delta(
        stacked,
        ["event_type", "lag_rank"],
        "prev_delta_hours",
        "prev_delta",
    )


__all__ = [
    "event_object_join",
    "event_object_matrix",
    "event_object_recent_matrix",
    "event_oldest_k_matrix",
    "event_recent_k_matrix",
]
