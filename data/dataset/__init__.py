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

DATASET_NAMES = (
    "bpi2017",
    "bpi2019",
    "order_management",
    "container_logistics",
)


def register_all_datasets(cache_root: str | Path | None = None) -> None:
    resolved_cache_root = configure_cache_environment(cache_root)
    get_dataset.cache_clear()
    register_dataset(
        DATASET_NAMES[0],
        cast(Dataset, BPI2017),
        cache_dir=str(resolved_cache_root / DATASET_NAMES[0]),
    )
    register_dataset(
        DATASET_NAMES[1],
        cast(Dataset, BPI2019),
        cache_dir=str(resolved_cache_root / DATASET_NAMES[1]),
    )
    register_dataset(
        DATASET_NAMES[2],
        cast(Dataset, OrderManagementDataset),
        cache_dir=str(resolved_cache_root / DATASET_NAMES[2]),
    )
    register_dataset(
        DATASET_NAMES[3],
        cast(Dataset, ContainerLogisticsDataset),
        cache_dir=str(resolved_cache_root / DATASET_NAMES[3]),
    )


register_all_datasets()
