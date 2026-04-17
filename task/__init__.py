from pathlib import Path
from typing import cast
from tqdm import tqdm

from data.cache import configure_cache_environment
from relbench.base import BaseTask
from relbench.tasks import get_task, register_task

from .bpi2017 import (
    ApplicationFutureValidationCount19Days,
    ApplicationIncompleteOutcome19Days,
    ApplicationValidationOutcome14Days,
    OfferSentOutcome19Days,
    OfferReturnedOutcome19Days,
)
from .bpi2019 import (
    POItemCreationOutcome14Days,
    POItemInvoiceReceiptOutcome14Days,
    VendorFutureClearInvoiceItemCount14Days,
)
from .container_logistics import (
    ContainerLoadPhaseNextEvent4Hours,
    ContainerRemainingLoadTruckCount4Hours,
    TransportDocumentContainerDepartWithin14Days,
    TransportDocumentFutureDepartContainerCount14Days,
    TransportDocumentStatusAfterOrder14Days,
    TransportDocumentVehicleDepartWithin14Days,
    VehicleBookingNextEvent14Days,
    VehicleFutureContainerLoadCount14Days,
)
from .order_management import (
    CustomerProductFutureOrderCount30Days,
    CustomerProductRepeatOrderWithin14Days,
    OrderFutureReminderCount30Days,
    OrderPaymentOutcome30Days,
)


TASK_SPECS = (
    # bpi2017 — Application tasks
    ("bpi2017", "application_validation_outcome_14d",      ApplicationValidationOutcome14Days),
    ("bpi2017", "application_incomplete_outcome_19d",      ApplicationIncompleteOutcome19Days),
    ("bpi2017", "application_future_validation_count_19d", ApplicationFutureValidationCount19Days),
    # bpi2017 — Offer tasks
    ("bpi2017", "offer_sent_outcome_19d",                  OfferSentOutcome19Days),
    ("bpi2017", "offer_returned_outcome_19d",              OfferReturnedOutcome19Days),

    # bpi2019 — POItem / Vendor tasks
    ("bpi2019", "po_item_creation_outcome_14d",              POItemCreationOutcome14Days),
    ("bpi2019", "po_item_invoice_receipt_outcome_14d",       POItemInvoiceReceiptOutcome14Days),
    ("bpi2019", "vendor_future_clear_invoice_item_count_14d", VendorFutureClearInvoiceItemCount14Days),

    # order_management — Order tasks
    ("order_management", "order_payment_outcome_30d",               OrderPaymentOutcome30Days),
    ("order_management", "order_future_reminder_count_30d",         OrderFutureReminderCount30Days),
    # order_management — Pair tasks
    ("order_management", "customer_product_repeat_order_14d",       CustomerProductRepeatOrderWithin14Days),
    ("order_management", "customer_product_future_order_count_30d", CustomerProductFutureOrderCount30Days),

    # container_logistics — Container tasks
    ("container_logistics", "container_load_phase_next_event_4h",         ContainerLoadPhaseNextEvent4Hours),
    ("container_logistics", "container_remaining_load_truck_count_4h",    ContainerRemainingLoadTruckCount4Hours),
    # container_logistics — Vehicle / Transport Document tasks
    ("container_logistics", "vehicle_booking_next_event_14d",             VehicleBookingNextEvent14Days),
    ("container_logistics", "vehicle_future_container_load_count_14d",    VehicleFutureContainerLoadCount14Days),
    ("container_logistics", "transport_document_future_depart_container_count_14d", TransportDocumentFutureDepartContainerCount14Days),
    ("container_logistics", "transport_document_status_after_order_14d",  TransportDocumentStatusAfterOrder14Days),
    # container_logistics — Pair tasks
    ("container_logistics", "transport_document_vehicle_depart_14d",      TransportDocumentVehicleDepartWithin14Days),
    ("container_logistics", "transport_document_container_depart_14d",    TransportDocumentContainerDepartWithin14Days),
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
