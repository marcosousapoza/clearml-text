import pandas as pd


def normalize_token(x, *, empty_token: str = "default") -> str:
    """
    Normalize a token by stripping whitespace and replacing empty/pipe chars.

    Args:
        x: Value to normalize
        empty_token: Token to use for empty/None values

    Returns:
        Normalized string token
    """
    s = "" if x is None else str(x).strip()
    return empty_token if s == "" else s.replace("|", "_")
