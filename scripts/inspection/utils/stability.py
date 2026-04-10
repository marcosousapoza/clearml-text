"""Stability diagnostics for temporal activity binning."""

import numpy as np
import pandas as pd

from data.const import TIME_COL


def summarize_interval_stability(
    events: pd.DataFrame,
    hours: int,
    min_events: int = 200,
) -> dict[str, float | int]:
    """Evaluate how stable weekly activity profiles are for a given bin width.

    Events are bucketed by weekday and hour interval, normalized into weekly
    probability vectors, and compared with pairwise correlations across weeks.
    Only weeks with at least ``min_events`` events are kept so sparse weeks do
    not dominate the estimate.
    """

    if hours <= 0 or 24 % hours != 0:
        raise ValueError(f"hours must divide 24 cleanly, got {hours}.")

    work = events[[TIME_COL]].copy()
    work["weekday_num"] = work[TIME_COL].dt.dayofweek
    work["bucket_start"] = (work[TIME_COL].dt.hour // hours) * hours
    work["week_start"] = work[TIME_COL].dt.to_period("W-MON").dt.start_time

    weekly_total = work.groupby("week_start").size().rename("n")
    work = work.merge(weekly_total, on="week_start", how="left")
    work = work[work["n"] >= min_events].copy()
    if work.empty:
        return {
            "interval_hours": hours,
            "weeks_ge_min_events": 0,
            "corr_mean": np.nan,
            "corr_std": np.nan,
            "corr_min": np.nan,
            "corr_p10": np.nan,
            "corr_p50": np.nan,
            "corr_p90": np.nan,
            "active_cells": 0,
        }

    weekly = work.groupby(["week_start", "weekday_num", "bucket_start"]).size().reset_index(name="count")
    full_index = pd.MultiIndex.from_product(
        [
            sorted(weekly["week_start"].unique()),
            range(7),
            range(0, 24, hours),
        ],
        names=["week_start", "weekday_num", "bucket_start"],
    )
    weekly = (
        weekly.set_index(["week_start", "weekday_num", "bucket_start"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )
    weekly["total"] = weekly.groupby("week_start")["count"].transform("sum")
    weekly = weekly[weekly["total"] > 0].copy()
    weekly["prob"] = weekly["count"] / weekly["total"]

    matrix = weekly.pivot(index="week_start", columns=["weekday_num", "bucket_start"], values="prob").fillna(0.0)
    active_cells = int((weekly.groupby(["weekday_num", "bucket_start"])["count"].sum() > 0).sum())
    if len(matrix) < 2:
        return {
            "interval_hours": hours,
            "weeks_ge_min_events": int(len(matrix)),
            "corr_mean": np.nan,
            "corr_std": np.nan,
            "corr_min": np.nan,
            "corr_p10": np.nan,
            "corr_p50": np.nan,
            "corr_p90": np.nan,
            "active_cells": active_cells,
        }

    corr = matrix.T.corr()
    mask = np.eye(len(corr), dtype=bool)
    pairwise = corr.where(~mask).stack()
    return {
        "interval_hours": hours,
        "weeks_ge_min_events": int(len(matrix)),
        "corr_mean": float(pairwise.mean()),
        "corr_std": float(pairwise.std()),
        "corr_min": float(pairwise.min()),
        "corr_p10": float(pairwise.quantile(0.10)),
        "corr_p50": float(pairwise.quantile(0.50)),
        "corr_p90": float(pairwise.quantile(0.90)),
        "active_cells": active_cells,
    }


__all__ = ["summarize_interval_stability"]
