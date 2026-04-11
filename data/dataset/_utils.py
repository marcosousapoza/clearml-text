import os
from typing import Callable, Optional
from urllib.parse import urlparse
import zipfile

import requests

from ..cache import RAW_OCEL_DIRNAME, get_cache_root
from ..datareader.json._reader import JSONDataReader
from ..datareader.base import BaseDataReader


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
