import os
from pathlib import Path


RAW_OCEL_DIRNAME = "raw_ocel"
OPTUNA_CACHE_DIRNAME = "optuna"
OPTUNA_DB_FILENAME = "optuna.sqlite3"
ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT_DIR / ".env"


def load_local_env(env_path: Path = ENV_PATH) -> None:
    """Load CACHE_ROOT from the repo .env file without overwriting the process env."""
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key == "CACHE_ROOT" and key not in os.environ:
            os.environ[key] = value


def get_cache_root() -> Path:
    """Return the shared cache root configured through CACHE_ROOT."""
    load_local_env()
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
