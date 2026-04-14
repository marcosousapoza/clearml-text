import os
from typing import Callable, Optional
from urllib.parse import urlparse
import zipfile

import pandas as pd
import requests
from relbench.base.database import Database
from relbench.base.table import Table

from ..cache import RAW_OCEL_DIRNAME, get_cache_root
from ..const import (
    E2O_EVENT_ID_COL,
    E2O_OBJECT_ID_COL,
    E2O_TABLE,
    EVENT_ATTR_TABLE_PREFIX,
    EVENT_ID_COL,
    EVENT_TABLE,
    OBJECT_ATTR_TABLE_PREFIX,
    OBJECT_ID_COL,
    OBJECT_TABLE,
    O2O_DST_COL,
    O2O_SRC_COL,
    O2O_TABLE,
    TIME_COL,
)
from ..datareader.json._reader import JSONDataReader
from ..datareader.base import BaseDataReader


def from_event_time(db: Database, timestamp: pd.Timestamp) -> Database:
    """
    Return a database containing rows linked to events from timestamp onwards.

    Unlike `Database.from_`, this only uses the `event` table timestamp to choose
    the temporal window. Object rows are kept when they are referenced by a
    surviving event-object link, regardless of the object's own timestamp.
    """
    return _filter_by_event_time(db, timestamp, op="from")


def upto_event_time(db: Database, timestamp: pd.Timestamp) -> Database:
    """
    Return a database containing rows linked to events up to timestamp.

    Unlike `Database.upto`, this only uses the `event` table timestamp to choose
    the temporal window. Object rows are kept when they are referenced by a
    surviving event-object link, regardless of the object's own timestamp.
    """
    return _filter_by_event_time(db, timestamp, op="upto")


def _filter_by_event_time(
    db: Database,
    timestamp: pd.Timestamp,
    op: str,
) -> Database:
    event_table = db.table_dict[EVENT_TABLE]
    event_df = event_table.df

    if op == "from":
        event_mask = event_df[TIME_COL] >= timestamp
    elif op == "upto":
        event_mask = event_df[TIME_COL] <= timestamp
    else:
        raise ValueError(f"Unsupported event time filter operation: {op}")

    kept_event_ids = set(event_df.loc[event_mask, EVENT_ID_COL].dropna())
    kept_object_ids = _object_ids_linked_to_events(db, kept_event_ids)

    out: dict[str, Table] = {}
    for name, table in db.table_dict.items():
        df = table.df
        if name == EVENT_TABLE:
            filtered_df = df.loc[event_mask]
        elif name == E2O_TABLE:
            filtered_df = df[df[E2O_EVENT_ID_COL].isin(kept_event_ids)]
        elif name == O2O_TABLE:
            filtered_df = df[
                df[O2O_SRC_COL].isin(kept_object_ids)
                & df[O2O_DST_COL].isin(kept_object_ids)
            ]
        elif name == OBJECT_TABLE:
            filtered_df = df[df[OBJECT_ID_COL].isin(kept_object_ids)]
        elif name.startswith(EVENT_ATTR_TABLE_PREFIX) and EVENT_ID_COL in df.columns:
            filtered_df = df[df[EVENT_ID_COL].isin(kept_event_ids)]
        elif name.startswith(OBJECT_ATTR_TABLE_PREFIX) and OBJECT_ID_COL in df.columns:
            filtered_df = df[df[OBJECT_ID_COL].isin(kept_object_ids)]
        else:
            filtered_df = df

        out[name] = Table(
            df=filtered_df,
            time_col=table.time_col,
            pkey_col=table.pkey_col,
            fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,
        )

    return Database(table_dict=out)


def _object_ids_linked_to_events(
    db: Database,
    event_ids: set[object],
) -> set[object]:
    e2o_table = db.table_dict.get(E2O_TABLE)
    if e2o_table is None:
        return set()

    e2o_df = e2o_table.df
    return set(
        e2o_df.loc[
            e2o_df[E2O_EVENT_ID_COL].isin(event_ids),
            E2O_OBJECT_ID_COL,
        ].dropna()
    )


def unzip_file(zip_path: str, extract_to: Optional[str] = None) -> None:
    """
    Unzip `zip_path` into `extract_to`.
    If `extract_to` is None, extracts into the zip's parent folder.
    """
    if extract_to is None:
        extract_to = os.path.dirname(zip_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


def download_file(uri: str, cache_dir: str) -> str:
    """
    Download file from URI to cache directory.
    Returns the path to the downloaded file.
    """
    os.makedirs(cache_dir, exist_ok=True)

    parsed = urlparse(uri)
    filename = os.path.basename(parsed.path) or "downloaded_file"
    download_path = os.path.join(cache_dir, filename)

    # Skip download if already cached
    if os.path.exists(download_path):
        return download_path

    # Stream download
    response = requests.get(uri, stream=True)
    response.raise_for_status()
    with open(download_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    return download_path


def get_parser(path: str, file_format: str) -> BaseDataReader:
    """
    Get the appropriate parser for the given file format.

    Args:
        path: Path to the file to parse
        file_format: Format of the file ('json', 'xml', 'csv')

    Returns:
        Appropriate DataReader instance
    """
    if file_format == "xml":
        raise NotImplementedError("Not implemented for XML format")
    elif file_format == "json":
        return JSONDataReader(path)
    elif file_format == "csv":
        raise NotImplementedError("Not implemented for CSV format")
    raise NotImplementedError(f"Unsupported file format: {file_format}")


def parse_ocel_to_database(
    uri: str,
    file_format: str,
    dataset_name: str,
    pre_parse_fn: Optional[Callable[[str], str]] = None,
):
    """
    Download and parse an OCEL file into a relbench Database.

    Args:
        uri: URL to download the OCEL file from
        file_format: Format of the file ('json', 'xml', 'csv')
        cache_dir: Directory to cache downloaded files
        dataset_name: Name of the dataset (for logging)
        pre_parse_fn: Optional function to run before parsing (e.g., unzip).
                      Takes file_path as input and returns the path to parse.

    Returns:
        relbench Database instance
    """
    cache_dir = str(get_cache_root() / RAW_OCEL_DIRNAME)

    # Download file
    download_path = download_file(uri, cache_dir)

    # Pre-parse if needed (e.g., unzip)
    if pre_parse_fn is not None:
        parse_path = pre_parse_fn(download_path)
    else:
        parse_path = download_path

    # Parse to database
    parser = get_parser(parse_path, file_format)
    return parser.parse_tables(
        dataset_name=dataset_name,
    )
