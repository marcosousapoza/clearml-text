from pathlib import Path
from typing import cast
from tqdm import tqdm

from data.cache import configure_cache_environment
from relbench.base import BaseTask
from relbench.tasks import get_task, register_task

from .bpi2017 import (
    ApplicationFutureValidationCount10Days,
    ApplicationIncompleteOutcome10Days,
    ApplicationValidationOutcome7Days,
    OfferSentOutcome10Days,
    OfferReturnedOutcome10Days,
)
from .bpi2019 import (
    POItemCreationOutcome7Days,
    POItemInvoiceReceiptOutcome7Days,
    POItemInvoicedNetWorth30Days,
    VendorFutureClearInvoiceItemCount7Days,
)
from .container_logistics import (
    ContainerLoadPhaseNextEvent4Hours,
    ContainerRemainingLoadTruckCount4Hours,
    TransportDocumentContainerDepartWithin7Days,
    TransportDocumentFutureDepartContainerCount7Days,
    TransportDocumentStatusAfterOrder7Days,
    TransportDocumentVehicleDepartWithin7Days,
    VehicleBookingNextEvent7Days,
    VehicleFutureContainerLoadCount7Days,
)
from .order_management import (
    CustomerProductFutureOrderCount14Days,
    CustomerProductRepeatOrderWithin7Days,
    OrderFutureReminderCount14Days,
    OrderPaymentOutcome14Days,
)


TASK_SPECS = (
    # bpi2017 — Application tasks
    ("bpi2017", "application_validation_outcome_7d",       ApplicationValidationOutcome7Days),
    ("bpi2017", "application_incomplete_outcome_10d",      ApplicationIncompleteOutcome10Days),
    ("bpi2017", "application_future_validation_count_10d", ApplicationFutureValidationCount10Days),
    # bpi2017 — Offer tasks
    ("bpi2017", "offer_sent_outcome_10d",                  OfferSentOutcome10Days),
    ("bpi2017", "offer_returned_outcome_10d",              OfferReturnedOutcome10Days),

    # bpi2019 — POItem / Vendor tasks
    ("bpi2019", "po_item_creation_outcome_7d",               POItemCreationOutcome7Days),
    ("bpi2019", "po_item_invoice_receipt_outcome_7d",        POItemInvoiceReceiptOutcome7Days),
    ("bpi2019", "po_item_invoiced_net_worth_30d",            POItemInvoicedNetWorth30Days),
    ("bpi2019", "vendor_future_clear_invoice_item_count_7d", VendorFutureClearInvoiceItemCount7Days),

    # order_management — Order tasks
    ("order_management", "order_payment_outcome_14d",               OrderPaymentOutcome14Days),
    ("order_management", "order_future_reminder_count_14d",         OrderFutureReminderCount14Days),
    # order_management — Pair tasks
    ("order_management", "customer_product_repeat_order_7d",        CustomerProductRepeatOrderWithin7Days),
    ("order_management", "customer_product_future_order_count_14d", CustomerProductFutureOrderCount14Days),

    # container_logistics — Container tasks
    ("container_logistics", "container_load_phase_next_event_4h",         ContainerLoadPhaseNextEvent4Hours),
    ("container_logistics", "container_remaining_load_truck_count_4h",    ContainerRemainingLoadTruckCount4Hours),
    # container_logistics — Vehicle / Transport Document tasks
    ("container_logistics", "vehicle_booking_next_event_7d",              VehicleBookingNextEvent7Days),
    ("container_logistics", "vehicle_future_container_load_count_7d",     VehicleFutureContainerLoadCount7Days),
    ("container_logistics", "transport_document_future_depart_container_count_7d", TransportDocumentFutureDepartContainerCount7Days),
    ("container_logistics", "transport_document_status_after_order_7d",   TransportDocumentStatusAfterOrder7Days),
    # container_logistics — Pair tasks
    ("container_logistics", "transport_document_vehicle_depart_7d",       TransportDocumentVehicleDepartWithin7Days),
    ("container_logistics", "transport_document_container_depart_7d",     TransportDocumentContainerDepartWithin7Days),
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
