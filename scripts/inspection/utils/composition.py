"""Composition summaries for events, objects, and their links."""

from typing import Literal

import pandas as pd
from relbench.base.database import Database

from data.const import E2O_TABLE, EVENT_ID_COL, OBJECT_ID_COL
from data.wrapper import check_dbs

Entity = Literal["event", "object"]


@check_dbs
def type_counts(db: Database, entity: Entity) -> pd.DataFrame:
    """Count rows by event or object type."""

    if entity == "event":
        table = db.table_dict["event"].df
        type_col = "type"
    elif entity == "object":
        table = db.table_dict["object"].df
        type_col = "type"
    else:
        raise ValueError(f"Unsupported entity {entity!r}.")
    label = f"{entity}_type"
    return (
        table[type_col]
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis(label)
        .reset_index(name="count")
    )


@check_dbs
def degree_dist(db: Database, entity: Entity) -> pd.DataFrame:
    """Count event-object degrees on the requested side of the graph."""

    e2o = db.table_dict[E2O_TABLE].df.copy()
    if entity == "event":
        value_name = "objects_per_event"
        count_name = "num_events"
        per_entity = e2o.groupby(EVENT_ID_COL)[OBJECT_ID_COL].nunique().rename(value_name)
    elif entity == "object":
        value_name = "events_per_object"
        count_name = "num_objects"
        per_entity = e2o.groupby(OBJECT_ID_COL)[EVENT_ID_COL].nunique().rename(value_name)
    else:
        raise ValueError(f"Unsupported entity {entity!r}.")
    return (
        per_entity.value_counts()
        .rename_axis(value_name)
        .reset_index(name=count_name)
        .sort_values(value_name, kind="stable")
        .reset_index(drop=True)
    )


@check_dbs
def e2o_degree_summary(db: Database) -> pd.DataFrame:
    """Summarize the degree distributions of the event-object graph.

    The output compares event-side degree (objects per event) and object-side
    degree (events per object) using count, mean, standard deviation, and key
    quantiles.
    """

    e2o_df = db.table_dict[E2O_TABLE].df
    event_degree = e2o_df.groupby(EVENT_ID_COL)[OBJECT_ID_COL].nunique()
    object_degree = e2o_df.groupby(OBJECT_ID_COL)[EVENT_ID_COL].nunique()
    rows = []
    for name, series in (("objects_per_event", event_degree), ("events_per_object", object_degree)):
        rows.append(
            {
                "metric": name,
                "count": int(series.shape[0]),
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": int(series.min()),
                "p50": float(series.quantile(0.5)),
                "p90": float(series.quantile(0.9)),
                "max": int(series.max()),
            }
        )
    return pd.DataFrame(rows)


@check_dbs
def event_type_counts(db: Database) -> pd.DataFrame:
    return type_counts(db, "event")


@check_dbs
def object_type_counts(db: Database) -> pd.DataFrame:
    return type_counts(db, "object")


@check_dbs
def objects_per_event_distribution(db: Database) -> pd.DataFrame:
    return degree_dist(db, "event")


@check_dbs
def events_per_object_distribution(db: Database) -> pd.DataFrame:
    return degree_dist(db, "object")


__all__ = [
    "degree_dist",
    "e2o_degree_summary",
    "event_type_counts",
    "events_per_object_distribution",
    "object_type_counts",
    "objects_per_event_distribution",
    "type_counts",
]
