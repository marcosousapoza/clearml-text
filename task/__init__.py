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
    ContainerRescheduledWithin14d,
    ContainerTruckPairInteraction,
    CustomerOrderNextTime,
    CustomerOrderRemainingTime,
    TransportDocNextEvent,
)
from .bpi2017 import (
    ApplicationDeniedWithin14d,
    ApplicationNextEvent,
    ApplicationNextTime,
    ApplicationOfferPairInteraction,
    ApplicationRemainingTime,
    CaseNextTime,
    OfferNextEvent,
    OfferRemainingTime,
)
from .bpi2019 import (
    POItemNextEvent,
    PONextSRMEvent,
    POItemNextTime,
    PONextTime,
    POItemRemainingTime,
    PORemainingTime,
    POItemBlockedWithin14d,
    POItemVendorPairInteraction,
)
from .order_management import (
    ItemNextTime,
    OrderEmployeePairInteraction,
    OrderNextEvent,
    OrderNextTime,
    OrderRemainingTime,
    OrderStockoutWithin14d,
    PackageNextDeliveryEvent,
    PackageRemainingTime,
)

TASK_SPECS = [
    # container_logistics (8 tasks)
    ("container_logistics", "container_next_event",          ContainerNextEvent),
    ("container_logistics", "transport_doc_next_event",      TransportDocNextEvent),
    ("container_logistics", "container_next_time",           ContainerNextTime),
    ("container_logistics", "customer_order_next_time",      CustomerOrderNextTime),
    ("container_logistics", "container_remaining_time",      ContainerRemainingTime),
    ("container_logistics", "customer_order_remaining_time", CustomerOrderRemainingTime),
    ("container_logistics", "container_rescheduled_14d",     ContainerRescheduledWithin14d),
    ("container_logistics", "container_truck_pair",          ContainerTruckPairInteraction),
    # bpi2017 (8 tasks)
    ("bpi2017", "application_next_event",       ApplicationNextEvent),
    ("bpi2017", "offer_next_event",             OfferNextEvent),
    ("bpi2017", "application_next_time",        ApplicationNextTime),
    ("bpi2017", "case_next_time",               CaseNextTime),
    ("bpi2017", "application_remaining_time",   ApplicationRemainingTime),
    ("bpi2017", "offer_remaining_time",         OfferRemainingTime),
    ("bpi2017", "application_denied_14d",       ApplicationDeniedWithin14d),
    ("bpi2017", "application_offer_pair",       ApplicationOfferPairInteraction),
    # bpi2019 (8 tasks)
    ("bpi2019", "po_item_next_event",       POItemNextEvent),
    ("bpi2019", "po_next_srm_event",        PONextSRMEvent),
    ("bpi2019", "po_item_next_time",        POItemNextTime),
    ("bpi2019", "po_next_time",             PONextTime),
    ("bpi2019", "po_item_remaining_time",   POItemRemainingTime),
    ("bpi2019", "po_remaining_time",        PORemainingTime),
    ("bpi2019", "po_item_blocked_14d",      POItemBlockedWithin14d),
    ("bpi2019", "po_item_vendor_pair",      POItemVendorPairInteraction),
    # order_management (8 tasks)
    ("order_management", "order_next_event",            OrderNextEvent),
    ("order_management", "package_next_delivery_event", PackageNextDeliveryEvent),
    ("order_management", "order_next_time",             OrderNextTime),
    ("order_management", "item_next_time",              ItemNextTime),
    ("order_management", "order_remaining_time",        OrderRemainingTime),
    ("order_management", "package_remaining_time",      PackageRemainingTime),
    ("order_management", "order_stockout_14d",          OrderStockoutWithin14d),
    ("order_management", "order_employee_pair",         OrderEmployeePairInteraction),
]


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
