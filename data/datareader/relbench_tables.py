from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from relbench.base.database import Database
from relbench.base.table import Table

from data.const import (
    E2O_EVENT_ID_COL, E2O_OBJECT_ID_COL, E2O_TABLE, EVENT_ATTR_TABLE_PREFIX, EVENT_ID_COL,
    EVENT_TABLE, EVENT_TYPE_COL, O2O_DST_COL, O2O_SRC_COL, O2O_TABLE,
    OBJECT_ATTR_TABLE_PREFIX, OBJECT_ID_COL, OBJECT_TABLE, OBJECT_TYPE_COL, QUALIFIER_COL,
    TIME_COL,
)
from data.wrapper import check_dbs

_TIME_FORMATS = ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f",
                 "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S",
                 "%Y/%m/%d %H:%M:%S", "%d-%m-%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S",
                 "%m/%d/%Y %H:%M:%S", "%Y-%m-%d")
_STRING_COLUMNS = {EVENT_TYPE_COL, OBJECT_TYPE_COL, QUALIFIER_COL, "edge_object_type", "role"}
_KEY_COLUMNS = {EVENT_ID_COL, OBJECT_ID_COL, E2O_EVENT_ID_COL, E2O_OBJECT_ID_COL, O2O_SRC_COL, O2O_DST_COL}
_NESTED_TYPES = (dict, list, tuple, set)


def _parse_time_column(series: pd.Series, time_format: str | None = None) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        out = pd.to_datetime(series, errors="coerce")
    elif time_format is not None:
        out = pd.to_datetime(series, format=time_format, errors="coerce")
    else:
        required = series.notna().sum()
        for fmt in _TIME_FORMATS:
            out = pd.to_datetime(series, format=fmt, errors="coerce")
            if out.notna().sum() == required:
                break
        else:
            out = pd.to_datetime(series, format="mixed", errors="coerce")
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_localize(None)
    return out.astype("datetime64[ns]")


def _cast_integer(series: pd.Series, *, unsigned: bool = False) -> pd.Series:
    numeric = pd.to_numeric(series, errors="raise")
    if numeric.isna().any():
        raise ValueError("Missing integers are not supported by the allowed dtype set.")
    if unsigned:
        if len(numeric) and numeric.min() < 0:
            raise ValueError("Unsigned integer column contains negative values.")
        if len(numeric) and numeric.max() > 4294967295:
            raise ValueError("Unsigned integer column exceeds uint32 range.")
        return numeric.astype("uint32")
    if len(numeric):
        lo, hi = int(numeric.min()), int(numeric.max())
        if -(2**31) <= lo and hi <= 2**31 - 1:
            return numeric.astype("int32")
    return numeric.astype("int64")


def _cast_key(series: pd.Series) -> pd.Series:
    if pd.api.types.is_integer_dtype(series.dtype):
        if series.isna().any():
            raise ValueError("Key columns cannot contain missing values.")
        return pd.to_numeric(series, errors="raise").astype("int64")
    numeric = pd.to_numeric(series, errors="coerce")
    if series.notna().any() and not numeric[series.notna()].notna().all():
        return series.astype("string")
    if series.notna().any() and not (numeric[series.notna()] % 1 == 0).all():
        raise ValueError("Key columns must be integer-like.")
    if numeric.isna().any():
        raise ValueError("Key columns cannot contain missing values.")
    return numeric.astype("int64")


def _cast_object(series: pd.Series, column_name: str) -> pd.Series:
    non_null = series.dropna()
    if non_null.empty or column_name in _STRING_COLUMNS:
        return series.astype("string")
    if non_null.map(lambda value: isinstance(value, _NESTED_TYPES)).any():
        raise ValueError(f"Column {column_name!r} contains nested values.")
    if non_null.map(lambda value: isinstance(value, (bytes, bytearray))).any():
        return series.map(lambda value: value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else value).astype("string")
    inferred = pd.api.types.infer_dtype(non_null, skipna=True)
    if inferred == "boolean":
        if series.isna().any():
            raise ValueError(f"Column {column_name!r} contains missing booleans.")
        return series.astype("bool")
    if inferred in {"integer", "mixed-integer"}:
        return _cast_key(series) if column_name in _KEY_COLUMNS else _cast_integer(series)
    if inferred in {"floating", "mixed-integer-float"}:
        return pd.to_numeric(series, errors="raise").astype("float64")
    if inferred in {"datetime64", "datetime", "date"}:
        return _parse_time_column(series)
    return series.astype("string")


def _normalize_series(series: pd.Series, *, column_name: str, time_col: str | None) -> pd.Series:
    if column_name == time_col:
        return _parse_time_column(series)
    if column_name in _STRING_COLUMNS:
        return series.astype("string")
    if column_name in _KEY_COLUMNS:
        return _cast_key(series)
    if isinstance(series.dtype, (pd.StringDtype, pd.CategoricalDtype)):
        return series.astype("string")
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return _parse_time_column(series)
    if pd.api.types.is_bool_dtype(series.dtype):
        if series.isna().any():
            raise ValueError(f"Column {column_name!r} contains missing booleans.")
        return series.astype("bool")
    if pd.api.types.is_integer_dtype(series.dtype):
        return _cast_integer(series)
    if pd.api.types.is_float_dtype(series.dtype):
        return series.astype("float64" if str(series.dtype) == "float64" else "float32")
    return _cast_object(series.astype(object), column_name)


def apply_default_column_dtypes_to_df(df: pd.DataFrame, time_col: str | None = None) -> pd.DataFrame:
    out = df.copy()
    for column in out.columns:
        out[column] = _normalize_series(out[column], column_name=str(column), time_col=time_col)
    return out


@check_dbs
def apply_default_column_dtypes(db: Database) -> Database:
    return Database(table_dict={
        name: Table(
            df=apply_default_column_dtypes_to_df(table.df, table.time_col),
            time_col=table.time_col,
            pkey_col=table.pkey_col,
            fkey_col_to_pkey_table=dict(table.fkey_col_to_pkey_table),
        )
        for name, table in db.table_dict.items()
    })


def _require_columns(df: pd.DataFrame, columns: Iterable[str], table_name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{table_name} missing required columns {missing}.")


def _join_time(df: pd.DataFrame, lookup: pd.DataFrame, *, left_on: str, right_on: str, table_name: str) -> pd.DataFrame:
    out = df.merge(lookup[[right_on, TIME_COL]], left_on=left_on, right_on=right_on, how="left", validate="many_to_one")
    time_col = TIME_COL if TIME_COL in out.columns else f"{TIME_COL}_y"
    if time_col != TIME_COL:
        out = out.drop(columns=[f"{TIME_COL}_x"]).rename(columns={time_col: TIME_COL})
    if out[TIME_COL].isna().any():
        missing = out.loc[out[TIME_COL].isna(), left_on].astype(str).unique()[:10].tolist()
        raise ValueError(f"{table_name} references missing rows. Example ids: {missing}")
    return out.drop(columns=[right_on]) if right_on != left_on else out


def _standardize_attr_tables(attr_tables: dict[str, pd.DataFrame] | None, *, id_col: str, time_format: str | None) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for name, df in (attr_tables or {}).items():
        work = df.copy()
        _require_columns(work, [id_col], name)
        work[id_col] = work[id_col].astype(str)
        if TIME_COL in work.columns:
            work[TIME_COL] = _parse_time_column(work[TIME_COL], time_format)
        out[name] = work
    return out


def _infer_object_time(object_df: pd.DataFrame, event_df: pd.DataFrame, e2o_df: pd.DataFrame,
                       object_attr_by_type: dict[str, pd.DataFrame], *, time_format: str | None) -> pd.DataFrame:
    out = object_df.copy()
    out[TIME_COL] = _parse_time_column(out[TIME_COL], time_format) if TIME_COL in out.columns else pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
    event_min_time = event_df[TIME_COL].min() if not event_df.empty else pd.NaT
    sources: list[pd.Series] = []
    fallback_sources: list[pd.Series] = []
    if not e2o_df.empty:
        sources.append(
            e2o_df.merge(event_df[[EVENT_ID_COL, TIME_COL]], left_on=E2O_EVENT_ID_COL, right_on=EVENT_ID_COL,
                         how="left", validate="many_to_one").groupby(E2O_OBJECT_ID_COL, sort=False)[TIME_COL].min()
        )
    for df in object_attr_by_type.values():
        if TIME_COL not in df.columns or df.empty:
            continue
        fallback_sources.append(df.groupby(OBJECT_ID_COL, sort=False)[TIME_COL].min())
        attr_df = df if pd.isna(event_min_time) else df[df[TIME_COL] >= event_min_time]
        if not attr_df.empty:
            sources.append(attr_df.groupby(OBJECT_ID_COL, sort=False)[TIME_COL].min())
    inferred = pd.concat(sources, axis=1).min(axis=1) if sources else pd.Series(dtype="datetime64[ns]")
    out = out.set_index(OBJECT_ID_COL, drop=False)
    missing = out[TIME_COL].isna()
    if missing.any():
        out.loc[missing, TIME_COL] = inferred.reindex(out.index)[missing]
    missing = out[TIME_COL].isna()
    if missing.any() and fallback_sources:
        fallback_inferred = pd.concat(fallback_sources, axis=1).min(axis=1)
        out.loc[missing, TIME_COL] = fallback_inferred.reindex(out.index)[missing]
    if out[TIME_COL].isna().any():
        raise ValueError(f"Unable to infer {TIME_COL!r} for object_id(s): {out.loc[out[TIME_COL].isna(), OBJECT_ID_COL].astype(str).head(10).tolist()}")
    return out.reset_index(drop=True)


def _fill_object_attr_time(object_attr_by_type: dict[str, pd.DataFrame], object_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    object_time = object_df.set_index(OBJECT_ID_COL)[TIME_COL]
    out: dict[str, pd.DataFrame] = {}
    for name, df in object_attr_by_type.items():
        work = df.copy()
        work[TIME_COL] = work[OBJECT_ID_COL].map(object_time) if TIME_COL not in work.columns else work[TIME_COL].where(work[TIME_COL].notna(), work[OBJECT_ID_COL].map(object_time))
        value_cols = [column for column in work.columns if column not in {OBJECT_ID_COL, TIME_COL}]
        if value_cols:
            work = work.sort_values([OBJECT_ID_COL, TIME_COL], kind="stable")
            work[value_cols] = work.groupby(OBJECT_ID_COL, sort=False)[value_cols].ffill()
        out[name] = work
    return out


def tables_to_relbench_database(event: pd.DataFrame, object: pd.DataFrame, e2o: pd.DataFrame, o2o: pd.DataFrame | None = None,
                                event_attr_by_type: dict[str, pd.DataFrame] | None = None,
                                object_attr_by_type: dict[str, pd.DataFrame] | None = None, *,
                                time_format: str | None = None) -> Database:
    event_df, object_df, e2o_df = event.copy(), object.copy(), e2o.copy()
    o2o_df = None if o2o is None else o2o.copy()
    _require_columns(event_df, [EVENT_ID_COL, EVENT_TYPE_COL, TIME_COL], EVENT_TABLE)
    _require_columns(object_df, [OBJECT_ID_COL, OBJECT_TYPE_COL], OBJECT_TABLE)
    _require_columns(e2o_df, [E2O_EVENT_ID_COL, E2O_OBJECT_ID_COL], E2O_TABLE)
    if o2o_df is not None and not o2o_df.empty:
        _require_columns(o2o_df, [O2O_SRC_COL, O2O_DST_COL], O2O_TABLE)

    event_df[EVENT_ID_COL] = event_df[EVENT_ID_COL].astype(str)
    event_df[TIME_COL] = _parse_time_column(event_df[TIME_COL], time_format)
    event_df = event_df.sort_values(TIME_COL, kind="stable").reset_index(drop=True)
    object_df[OBJECT_ID_COL] = object_df[OBJECT_ID_COL].astype(str)
    e2o_df[E2O_EVENT_ID_COL] = e2o_df[E2O_EVENT_ID_COL].astype(str)
    e2o_df[E2O_OBJECT_ID_COL] = e2o_df[E2O_OBJECT_ID_COL].astype(str)
    if o2o_df is not None and not o2o_df.empty:
        o2o_df[O2O_SRC_COL] = o2o_df[O2O_SRC_COL].astype(str)
        o2o_df[O2O_DST_COL] = o2o_df[O2O_DST_COL].astype(str)

    event_attr = _standardize_attr_tables(event_attr_by_type, id_col=EVENT_ID_COL, time_format=time_format)
    if any(TIME_COL in df.columns for df in event_attr.values()):
        raise ValueError("Event attribute tables must not include explicit time columns.")
    object_attr = _standardize_attr_tables(object_attr_by_type, id_col=OBJECT_ID_COL, time_format=time_format)
    object_df = _infer_object_time(object_df, event_df, e2o_df, object_attr, time_format=time_format)
    object_attr = _fill_object_attr_time(object_attr, object_df)

    if event_df[EVENT_ID_COL].duplicated().any():
        raise ValueError(f"{EVENT_TABLE}.{EVENT_ID_COL} must be unique.")
    if object_df[OBJECT_ID_COL].duplicated().any():
        raise ValueError(f"{OBJECT_TABLE}.{OBJECT_ID_COL} must be unique.")

    e2o_df = _join_time(e2o_df, event_df, left_on=E2O_EVENT_ID_COL, right_on=EVENT_ID_COL, table_name=E2O_TABLE)
    tables: dict[str, Table] = {
        EVENT_TABLE: Table(event_df, {}, pkey_col=EVENT_ID_COL, time_col=TIME_COL),
        OBJECT_TABLE: Table(object_df, {}, pkey_col=OBJECT_ID_COL, time_col=TIME_COL),
        E2O_TABLE: Table(e2o_df, {E2O_EVENT_ID_COL: EVENT_TABLE, E2O_OBJECT_ID_COL: OBJECT_TABLE}, time_col=TIME_COL),
    }
    if o2o_df is not None and not o2o_df.empty:
        o2o_df = _join_time(o2o_df, object_df, left_on=O2O_SRC_COL, right_on=OBJECT_ID_COL, table_name=O2O_TABLE).rename(columns={TIME_COL: "_src_time"})
        o2o_df = _join_time(o2o_df, object_df, left_on=O2O_DST_COL, right_on=OBJECT_ID_COL, table_name=O2O_TABLE).rename(columns={TIME_COL: "_dst_time"})
        o2o_df[TIME_COL] = o2o_df[["_src_time", "_dst_time"]].max(axis=1)
        tables[O2O_TABLE] = Table(o2o_df.drop(columns=["_src_time", "_dst_time"]), {O2O_SRC_COL: OBJECT_TABLE, O2O_DST_COL: OBJECT_TABLE}, time_col=TIME_COL)
    for name, df in event_attr.items():
        if set(df.columns) != {EVENT_ID_COL}:
            tables[f"{EVENT_ATTR_TABLE_PREFIX}{name}"] = Table(_join_time(df, event_df, left_on=EVENT_ID_COL, right_on=EVENT_ID_COL, table_name=name), {EVENT_ID_COL: EVENT_TABLE}, time_col=TIME_COL)
    for name, df in object_attr.items():
        if set(df.columns) != {OBJECT_ID_COL, TIME_COL}:
            tables[f"{OBJECT_ATTR_TABLE_PREFIX}{name}"] = Table(df, {OBJECT_ID_COL: OBJECT_TABLE}, time_col=TIME_COL)
    return apply_default_column_dtypes(Database(table_dict=tables))
