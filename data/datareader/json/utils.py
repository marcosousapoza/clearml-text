import ijson


def detect_ocel_version(path: str) -> str:
    """
    Detect the OCEL version from the JSON file at `path`.

    Args:
        path (str): Path to the OCEL JSON file.

    Returns:
        str: "1.0" if v1-format, "2.0" if v2-format.

    Raises:
        ValueError: if neither signature is found.
    """
    seen = set()
    with open(path, "rb") as f:
        for prefix, event, value in ijson.parse(f):
            if prefix != "" or event != "map_key":
                continue
            if value == "ocel:global-log":
                return "1.0"
            seen.add(value)
            if {"eventTypes", "objectTypes"}.issubset(seen):
                return "2.0"
    raise ValueError(f"Unrecognized OCEL JSON format in {path!r}")
