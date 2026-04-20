import fcntl
import os
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Callable, ParamSpec, TypeVar


RAW_OCEL_DIRNAME = "raw_ocel"
OPTUNA_CACHE_DIRNAME = "optuna"
OPTUNA_DB_FILENAME = "optuna.sqlite3"

P = ParamSpec("P")
T = TypeVar("T")


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


def _cache_lock_path(cache_dir: str | Path) -> Path:
    """Return a sibling lock file for a cache directory.

    The lock file deliberately lives next to the cache directory rather than
    inside it. Some cache readers treat any file in a cache directory as
    evidence that the cache is ready to load, so the lock must not create
    content inside a directory that may still be empty or partially written.
    """
    cache_path = Path(cache_dir).expanduser().resolve()
    return cache_path.parent / f".{cache_path.name}.lock"


@contextmanager
def cache_lock(cache_dir: str | Path | None):
    """Serialize access to a shared cache directory across processes."""
    if cache_dir is None:
        yield
        return

    lock_path = _cache_lock_path(cache_dir)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def cache_locked(
    cache_dir: str | Path | None | Callable[P, str | Path | None],
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Wrap a function so it runs under a cache-directory lock.

    ``cache_dir`` may be a concrete directory or a callable that receives the
    wrapped function's arguments and returns the directory to lock.
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            resolved_cache_dir = (
                cache_dir(*args, **kwargs) if callable(cache_dir) else cache_dir
            )
            with cache_lock(resolved_cache_dir):
                return func(*args, **kwargs)

        return wrapper

    return decorator
