import os
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse
import zipfile

import requests

from ..datareader.json._reader import JSONDataReader
from ..datareader.base import BaseDataReader

DEFAULT_CACHE_ROOT = Path.home() / "scratch" / "relbench"
RAW_OCEL_CACHE_DIRNAME = "raw_ocel"


def get_cache_root() -> Path:
    """Return the shared project cache root."""
    return Path(os.environ.get("RELBENCH_CACHE_DIR", DEFAULT_CACHE_ROOT)).expanduser().resolve()


def configure_cache_environment(cache_root: str | Path | None = None) -> Path:
    """
    Configure the shared cache environment for both RelBench artifacts and raw OCEL downloads.
    """
    resolved_cache_root = (
        Path(cache_root).expanduser().resolve()
        if cache_root is not None
        else get_cache_root()
    )
    os.environ["RELBENCH_CACHE_DIR"] = str(resolved_cache_root)
    os.environ["OCEL_CACHE"] = str(resolved_cache_root / RAW_OCEL_CACHE_DIRNAME)
    return resolved_cache_root


def get_raw_ocel_cache_dir() -> str:
    """
    Return the cache directory for raw OCEL downloads.

    `OCEL_CACHE` overrides the default location. Otherwise, raw downloads live under
    the shared RelBench cache root.
    """
    if "OCEL_CACHE" in os.environ:
        return str(Path(os.environ["OCEL_CACHE"]).expanduser().resolve())
    return str(get_cache_root() / RAW_OCEL_CACHE_DIRNAME)


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
    cache_dir = get_raw_ocel_cache_dir()

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
