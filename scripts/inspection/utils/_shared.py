"""Shared internal helpers for inspection data transformations."""

import pandas as pd

from data.const import (
    EVENT_ATTR_TABLE_PREFIX,
    OBJECT_ATTR_TABLE_PREFIX,
)


def safe_quantile(series: pd.Series, q: float) -> float:
    """Compute a quantile after numeric coercion and null removal."""

    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float(clean.quantile(q))


def summarize_delta(
    frame: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    value_name: str,
) -> pd.DataFrame:
    """Aggregate duration-like values into count, mean, and percentile summaries.

    The input values are coerced to numeric hours, invalid values are dropped,
    and the grouped summary is returned in a deterministic sort order.
    """

    work = frame[group_cols + [value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[value_col])
    if work.empty:
        return pd.DataFrame(
            columns=group_cols
            + [
                "n",
                f"{value_name}_mean_hours",
                f"{value_name}_p50_hours",
                f"{value_name}_p90_hours",
            ]
        )
    return (
        work.groupby(group_cols, dropna=False, observed=False)[value_col]
        .agg(
            n="size",
            **{
                f"{value_name}_mean_hours": "mean",
                f"{value_name}_p50_hours": lambda s: safe_quantile(s, 0.5),
                f"{value_name}_p90_hours": lambda s: safe_quantile(s, 0.9),
            },
        )
        .reset_index()
        .sort_values(group_cols, kind="stable")
        .reset_index(drop=True)
    )
