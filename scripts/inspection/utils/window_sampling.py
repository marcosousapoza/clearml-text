"""Sampled window-size diagnostics for approximate OCEL subgraph sizes."""

from collections.abc import Sequence
from statistics import NormalDist

import duckdb
import numpy as np
import pandas as pd
from relbench.base.database import Database

from data.const import E2O_TABLE, EVENT_ID_COL, OBJECT_ID_COL, TIME_COL
from data.wrapper import check_dbs


def _normalize_deltas(deltas: Sequence[str | pd.Timedelta]) -> list[pd.Timedelta]:
    """Convert a sequence of user-provided delta values into timedeltas."""

    normalized = [pd.Timedelta(delta) for delta in deltas]
    if not normalized:
        raise ValueError("deltas must contain at least one value.")
    if any(delta <= pd.Timedelta(0) for delta in normalized):
        raise ValueError("all deltas must be strictly positive.")
    return sorted(set(normalized))


def _sample_root_times(
    relation_df: pd.DataFrame,
    *,
    num_roots: int,
    random_state: int | None,
) -> pd.Series:
    """Sample root timestamps from observed e2o relation times."""

    times = pd.to_datetime(relation_df[TIME_COL], errors="coerce").dropna().sort_values().reset_index(drop=True)
    if times.empty:
        raise ValueError("Cannot sample root times from an e2o table without timestamps.")
    sample_size = min(num_roots, len(times))
    sampled = times.sample(n=sample_size, replace=False, random_state=random_state)
    return sampled.sort_values(ignore_index=True)


@check_dbs
def sample_window_graph_sizes(
    db: Database,
    deltas: Sequence[str | pd.Timedelta],
    *,
    num_roots: int = 64,
    random_state: int | None = 0,
) -> pd.DataFrame:
    """Estimate graph sizes from sampled e2o time windows.

    For each sampled root time ``t`` and each delta ``d``, the function filters
    e2o rows whose relation timestamp lies in ``[t, t + d]`` and counts the
    distinct event ids plus the distinct object ids in that filtered relation.
    """

    normalized_deltas = _normalize_deltas(deltas)
    e2o_df = db.table_dict[E2O_TABLE].df.copy()
    if TIME_COL in e2o_df.columns:
        e2o_df[TIME_COL] = pd.to_datetime(e2o_df[TIME_COL], errors="coerce")
    relation_df = e2o_df[[EVENT_ID_COL, OBJECT_ID_COL, TIME_COL]].dropna(
        subset=[EVENT_ID_COL, OBJECT_ID_COL, TIME_COL]
    )
    root_times = _sample_root_times(relation_df, num_roots=num_roots, random_state=random_state)

    windows = pd.DataFrame(
        [
            {
                "root_index": root_idx,
                "root_time": root_time,
                "delta": delta,
                "delta_days": delta / pd.Timedelta(days=1),
                "window_end": root_time + delta,
            }
            for root_idx, root_time in enumerate(root_times)
            for delta in normalized_deltas
        ]
    )

    con = duckdb.connect()
    try:
        con.register("windows", windows)
        con.register("e2o_df", relation_df)
        sampled = con.execute(
            f"""
            SELECT
                w.root_index,
                w.root_time,
                w.delta,
                w.delta_days,
                COUNT(DISTINCT eo.{EVENT_ID_COL}) AS event_count,
                COUNT(DISTINCT eo.{OBJECT_ID_COL}) AS object_count
            FROM windows AS w
            LEFT JOIN e2o_df AS eo
                ON eo.{TIME_COL} >= w.root_time
               AND eo.{TIME_COL} <= w.window_end
            GROUP BY
                w.root_index,
                w.root_time,
                w.delta,
                w.delta_days
            ORDER BY
                w.delta,
                w.root_time
            """
        ).df()
    finally:
        con.close()
    return sampled.reset_index(drop=True)


def summarize_sampled_window_graph_sizes(
    samples: pd.DataFrame,
    *,
    ci: float = 0.95,
) -> pd.DataFrame:
    """Aggregate sampled window sizes into mean curves and confidence bands."""

    if samples.empty:
        return pd.DataFrame(
            columns=[
                "delta",
                "delta_days",
                "metric",
                "n",
                "mean",
                "std",
                "sem",
                "ci_lower",
                "ci_upper",
            ]
        )
    if not 0 < ci < 1:
        raise ValueError(f"ci must be between 0 and 1, got {ci}.")

    z_value = NormalDist().inv_cdf((1 + ci) / 2)
    rows: list[dict[str, object]] = []
    for metric in ("event_count", "object_count"):
        grouped = (
            samples.groupby(["delta", "delta_days"], observed=False)[metric]
            .agg(n="size", mean="mean", std="std")
            .reset_index()
            .sort_values("delta", kind="stable")
        )
        grouped["std"] = grouped["std"].fillna(0.0)
        grouped["sem"] = grouped["std"] / np.sqrt(grouped["n"].clip(lower=1))
        grouped["ci_lower"] = grouped["mean"] - z_value * grouped["sem"]
        grouped["ci_upper"] = grouped["mean"] + z_value * grouped["sem"]
        grouped["metric"] = metric.removesuffix("_count")
        rows.extend(grouped.to_dict("records")) # type: ignore
    return pd.DataFrame(rows).sort_values(["metric", "delta"], kind="stable").reset_index(drop=True)


__all__ = [
    "sample_window_graph_sizes",
    "summarize_sampled_window_graph_sizes",
]
