import os
from pathlib import Path


RAW_OCEL_DIRNAME = "raw_ocel"
OPTUNA_CACHE_DIRNAME = "optuna"
OPTUNA_DB_FILENAME = "optuna.sqlite3"


def get_cache_root() -> Path:
    """Return the shared cache root configured through CACHE_ROOT."""
    cache_root = os.environ.get("CACHE_ROOT")
    if not cache_root:
        raise RuntimeError(
            "CACHE_ROOT is not configured. Set it in the environment or in the repo .env file."
        )
    return Path(cache_root).expanduser().resolve()


def configure_cache_environment(cache_root: str | Path | None = None) -> Path:
    """Configure environment variables for all project caches."""
    resolved_cache_root = (
        Path(cache_root).expanduser().resolve()
        if cache_root is not None
        else get_cache_root()
    )
    os.environ["CACHE_ROOT"] = str(resolved_cache_root)
    from relbench import datasets, tasks

    datasets.DOWNLOAD_REGISTRY.path = resolved_cache_root
    tasks.DOWNLOAD_REGISTRY.path = resolved_cache_root
    return resolved_cache_root


def get_optuna_storage_path() -> Path:
    return get_cache_root() / OPTUNA_CACHE_DIRNAME / OPTUNA_DB_FILENAME
