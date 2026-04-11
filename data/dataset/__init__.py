from pathlib import Path
from typing import cast

from ..cache import configure_cache_environment

from relbench.base import Dataset
from relbench.datasets import get_dataset, register_dataset

from ._datasets import (
    OrderManagementDataset,
    ContainerLogisticsDataset,
    BPI2019,
    BPI2017,
)


def register_all_datasets(cache_root: str | Path | None = None) -> None:
    resolved_cache_root = configure_cache_environment(cache_root)
    get_dataset.cache_clear()
    register_dataset(
        "bpi2017",
        cast(Dataset, BPI2017),
        cache_dir=str(resolved_cache_root / "bpi2017"),
    )
    register_dataset(
        "bpi2019",
        cast(Dataset, BPI2019),
        cache_dir=str(resolved_cache_root / "bpi2019"),
    )
    register_dataset(
        "order_management",
        cast(Dataset, OrderManagementDataset),
        cache_dir=str(resolved_cache_root / "order_management"),
    )
    register_dataset(
        "container_logistics",
        cast(Dataset, ContainerLogisticsDataset),
        cache_dir=str(resolved_cache_root / "container_logistics"),
    )


register_all_datasets()
