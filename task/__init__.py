from pathlib import Path
from typing import cast
from tqdm import tqdm

from data.cache import configure_cache_environment
from relbench.base import BaseTask
from relbench.tasks import get_task, register_task

from .bpi2017 import (
    ApplicationNextEvent,
    ApplicationNextTime,
    ApplicationRemainingTime,
    OfferNextEvent,
    OfferNextTime,
    OfferRemainingTime,
)
from .bpi2019 import (
    POItemNextEvent,
    POItemNextTime,
    POItemRemainingTime,
    POItemVendorPairNextEvent,
    POItemVendorPairNextTime,
)
from .container_logistics import (
    ContainerNextEvent,
    ContainerNextTime,
    ContainerRemainingTime,
    TransportDocumentNextEvent,
    TransportDocumentNextTime,
    TransportDocumentRemainingTime,
    ContainerTDPairNextEvent,
    ContainerTDPairNextTime,
)
from .order_management import (
    OrderNextEvent,
    OrderNextTime,
    OrderRemainingTime,
    ProductNextEvent,
    ProductNextTime,
    ProductRemainingTime,
    CustomerProductPairNextEvent,
    CustomerProductPairNextTime,
)


TASK_SPECS = (
    # bpi2017 — Application single-entity
    ("bpi2017", "application_next_event",       ApplicationNextEvent),
    ("bpi2017", "application_next_time",        ApplicationNextTime),
    ("bpi2017", "application_remaining_time",   ApplicationRemainingTime),
    # bpi2017 — Offer single-entity
    ("bpi2017", "offer_next_event",             OfferNextEvent),
    ("bpi2017", "offer_next_time",              OfferNextTime),
    ("bpi2017", "offer_remaining_time",         OfferRemainingTime),
    # bpi2019 — POItem single-entity
    ("bpi2019", "po_item_next_event",           POItemNextEvent),
    ("bpi2019", "po_item_next_time",            POItemNextTime),
    ("bpi2019", "po_item_remaining_time",       POItemRemainingTime),
    # bpi2019 — POItem × Vendor pair
    ("bpi2019", "po_item_vendor_pair_next_event", POItemVendorPairNextEvent),
    ("bpi2019", "po_item_vendor_pair_next_time",  POItemVendorPairNextTime),

    # order_management — orders single-entity
    ("order_management", "order_next_event",          OrderNextEvent),
    ("order_management", "order_next_time",           OrderNextTime),
    ("order_management", "order_remaining_time",      OrderRemainingTime),
    # order_management — products single-entity
    ("order_management", "product_next_event",        ProductNextEvent),
    ("order_management", "product_next_time",         ProductNextTime),
    ("order_management", "product_remaining_time",    ProductRemainingTime),
    # order_management — customers × products pair
    ("order_management", "customer_product_pair_next_event", CustomerProductPairNextEvent),
    ("order_management", "customer_product_pair_next_time",  CustomerProductPairNextTime),

    # container_logistics — Container single-entity
    ("container_logistics", "container_next_event",       ContainerNextEvent),
    ("container_logistics", "container_next_time",        ContainerNextTime),
    ("container_logistics", "container_remaining_time",   ContainerRemainingTime),
    # container_logistics — Transport Document single-entity
    ("container_logistics", "transport_document_next_event",       TransportDocumentNextEvent),
    ("container_logistics", "transport_document_next_time",        TransportDocumentNextTime),
    ("container_logistics", "transport_document_remaining_time",   TransportDocumentRemainingTime),
    # container_logistics — Container × Transport Document pair
    ("container_logistics", "container_td_pair_next_event", ContainerTDPairNextEvent),
    ("container_logistics", "container_td_pair_next_time",  ContainerTDPairNextTime),
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
