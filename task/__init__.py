from pathlib import Path
from typing import cast

from tqdm import tqdm

from data.cache import configure_cache_environment
from relbench.base import BaseTask
from relbench.tasks import get_task, register_task

from .container_logistics import (
    ContainerNextEvent,
    ContainerNextTime,
    ContainerRemainingTime,
    ContainerTruckPairNextEvent,
    ContainerTruckPairNextTime,
    ContainerTruckPairRemainingTime,
)


TASK_SPECS = (
    ("container_logistics", "container_next_event", ContainerNextEvent),
    ("container_logistics", "container_next_time", ContainerNextTime),
    ("container_logistics", "container_remaining_time", ContainerRemainingTime),
    ("container_logistics", "container_truck_pair_next_event", ContainerTruckPairNextEvent),
    ("container_logistics", "container_truck_pair_next_time", ContainerTruckPairNextTime),
    ("container_logistics", "container_truck_pair_remaining_time", ContainerTruckPairRemainingTime),
)


def register_tasks(cache_root: str | Path | None = None) -> None:
    resolved_cache_root = configure_cache_environment(cache_root)
    get_task.cache_clear()

    for dataset_name, task_name, task_cls in tqdm(TASK_SPECS):
        register_task(
            dataset_name,
            task_name,
            cast(BaseTask, task_cls),
            cache_dir=str(resolved_cache_root / dataset_name / "tasks" / task_name),
        )


register_tasks()
