import os
from typing import cast

from ._utils import DEFAULT_CACHE_ROOT, configure_cache_environment

# Set RELBENCH_CACHE_DIR before registering datasets so that
# register_dataset captures the correct cache_dir at import time.
if "RELBENCH_CACHE_DIR" not in os.environ:
    os.environ["RELBENCH_CACHE_DIR"] = str(DEFAULT_CACHE_ROOT)

from relbench.base import Dataset
from relbench.datasets import register_dataset

from ._datasets import (
    OrderManagementDataset,
    ContainerLogisticsDataset,
    BPI2019,
    BPI2017,
)


def register_all_datasets() -> None:
    register_dataset("bpi2017", cast(Dataset, BPI2017))
    register_dataset("bpi2019", cast(Dataset, BPI2019))
    register_dataset("order_management", cast(Dataset, OrderManagementDataset))
    register_dataset("container_logistics", cast(Dataset, ContainerLogisticsDataset))


register_all_datasets()
