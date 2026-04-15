from pathlib import Path
from typing import cast

from data.cache import configure_cache_environment
from relbench.base import BaseTask
from relbench.tasks import get_task, register_task

from .bpi2017 import (
    ApplicationNextEvent,
    CaseRNextTime,
    CaseRRemainingTime,
    OfferCancelledWithin30Days,
)
from .bpi2019 import (
    POItemNextEvent,
    POItemNextTime,
    POItemRemainingTime,
    POItemVendorClearInvoiceWithin30Days,
)
from .container_logistics import (
    ContainerNextEvent,
    ContainerNextTime,
    ContainerRemainingTime,
    TransportDocumentVehicleDepartWithin7Days,
)
from .order_management import (
    CustomerProductPlaceOrderWithin14Days,
    OrderNextEvent,
    OrderNextTime,
    OrderRemainingTime,
)


TASK_SPECS = (
    ("bpi2017", "next_event_cases", ApplicationNextEvent),
    ("bpi2017", "next_time_cases", CaseRNextTime),
    ("bpi2017", "remaining_time_cases", CaseRRemainingTime),
    ("bpi2017", "event_within", OfferCancelledWithin30Days),
    ("bpi2019", "next_event_po_items", POItemNextEvent),
    ("bpi2019", "next_time_po_items", POItemNextTime),
    ("bpi2019", "remaining_time_po_items", POItemRemainingTime),
    ("bpi2019", "event_within", POItemVendorClearInvoiceWithin30Days),
    ("order_management", "next_event_orders", OrderNextEvent),
    ("order_management", "next_time_orders", OrderNextTime),
    ("order_management", "remaining_time_orders", OrderRemainingTime),
    ("order_management", "event_within", CustomerProductPlaceOrderWithin14Days),
    ("container_logistics", "next_event_containers", ContainerNextEvent),
    ("container_logistics", "next_time_containers", ContainerNextTime),
    ("container_logistics", "remaining_time_containers", ContainerRemainingTime),
    ("container_logistics", "event_within", TransportDocumentVehicleDepartWithin7Days),
)


def register_tasks(cache_root: str | Path | None = None) -> None:
    resolved_cache_root = configure_cache_environment(cache_root)
    get_task.cache_clear()

    for dataset_name, task_name, task_cls in TASK_SPECS:
        register_task(
            dataset_name,
            task_name,
            cast(BaseTask, task_cls),
            cache_dir=str(resolved_cache_root / dataset_name / "tasks" / task_name),
        )


register_tasks()
