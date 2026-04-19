"""Predictive tasks for the order_management dataset.

8 tasks across 4 label families:
  next-event classification  (2 tasks)
  next-time regression       (2 tasks)
  remaining-time regression  (2 tasks)
  binary event-within-window (1 task)
  multi-entity pair binary   (1 task)

Object types in this log
------------------------
  orders, items, packages, employees, products, customers

Event types (11)
----------------
  place order, confirm order, pick item, item out of stock, reorder item,
  create package, send package, failed delivery, package delivered,
  pay order, payment reminder
"""
import pandas as pd
from relbench.base import Database, Table, TaskType
from relbench.metrics import accuracy, f1, mae, mse, r2, rmse

from data.const import OBJECT_ID_COL, OBJECT_TABLE
from task.metrics import roc_auc
from task.utils import MEntityTask
from task.utils.builders import (
    build_event_within_table,
    build_next_event_table,
    build_next_time_table,
    build_pair_interaction_table,
    build_remaining_time_table,
    to_relbench_table,
)

# ---------------------------------------------------------------------------
# Domain vocabulary
# ---------------------------------------------------------------------------

_ORDER_EVENTS = [
    "place order",
    "confirm order",
    "pick item",
    "item out of stock",
    "reorder item",
    "create package",
    "send package",
    "failed delivery",
    "package delivered",
    "pay order",
    "payment reminder",
]

# Post-confirmation fulfilment events for orders
_ORDER_FULFILMENT_EVENTS = [
    "confirm order",
    "pick item",
    "item out of stock",
    "create package",
    "send package",
    "failed delivery",
    "package delivered",
    "pay order",
    "payment reminder",
]

# Delivery-outcome events for packages
_PACKAGE_DELIVERY_EVENTS = [
    "send package",
    "failed delivery",
    "package delivered",
]


# ---------------------------------------------------------------------------
# Shared window presets
# ---------------------------------------------------------------------------

_BACK_30 = pd.Timedelta(days=30)
_FWD_14  = pd.Timedelta(days=14)
_FWD_30  = pd.Timedelta(days=30)
_DELTA   = pd.Timedelta(days=7)


# ---------------------------------------------------------------------------
# Task 1 — Order: next fulfilment event classification
#
# Business meaning : After recent order activity, which of the 9 fulfilment
#                    events (confirm → pay) happens next for this order?
# Signal           : Orders progress through a largely sequential pipeline;
#                    the last observed event is a strong predictor of what
#                    comes next (e.g. confirm → pick, send → delivered).
# ---------------------------------------------------------------------------

class OrderNextEvent(MEntityTask):
    """Next fulfilment event for an active order (9-class)."""

    timedelta    = _DELTA
    task_type    = TaskType.MULTICLASS_CLASSIFICATION
    object_types = ("orders",)
    num_classes  = len(_ORDER_FULFILMENT_EVENTS)
    metrics      = [accuracy, f1, roc_auc]

    def make_table(self, db: Database, timestamps: pd.Series) -> Table:
        df = build_next_event_table(
            db,
            object_type = "orders",
            times       = timestamps,
            event_types = _ORDER_FULFILMENT_EVENTS,
            delta_back  = _BACK_30,
            delta_fwd   = _FWD_14,
        )
        return to_relbench_table(df, self.entity_cols, self.entity_tables)


# ---------------------------------------------------------------------------
# Task 2 — Package: next delivery event classification
#
# Business meaning : After a package is in transit, will the next delivery
#                    event be a successful delivery, a failure, or a send?
# Signal           : Package history (prior failed deliveries) strongly
#                    predicts future delivery outcome.
# ---------------------------------------------------------------------------

class PackageNextDeliveryEvent(MEntityTask):
    """Next delivery-phase event for an active package (3-class)."""

    timedelta    = _DELTA
    task_type    = TaskType.MULTICLASS_CLASSIFICATION
    object_types = ("packages",)
    num_classes  = len(_PACKAGE_DELIVERY_EVENTS)
    metrics      = [accuracy, f1, roc_auc]

    def make_table(self, db: Database, timestamps: pd.Series) -> Table:
        df = build_next_event_table(
            db,
            object_type = "packages",
            times       = timestamps,
            event_types = _PACKAGE_DELIVERY_EVENTS,
            delta_back  = _BACK_30,
            delta_fwd   = _FWD_14,
        )
        return to_relbench_table(df, self.entity_cols, self.entity_tables)


# ---------------------------------------------------------------------------
# Task 3 — Order: next-event time regression
#
# Business meaning : How many seconds until the next activity for this order?
#                    Useful for customer-facing SLA estimation.
# Signal           : Orders awaiting confirmation are typically fast; those
#                    stuck on stock-outs can stall for days.
# ---------------------------------------------------------------------------

class OrderNextTime(MEntityTask):
    """Days until the next event for an active order."""

    timedelta    = _DELTA
    task_type    = TaskType.REGRESSION
    object_types = ("orders",)
    metrics      = [mae, rmse, mse, r2]

    def make_table(self, db: Database, timestamps: pd.Series) -> Table:
        df = build_next_time_table(
            db,
            object_type = "orders",
            times       = timestamps,
            delta_back  = _BACK_30,
            delta_fwd   = _FWD_30,
        )
        return to_relbench_table(df, self.entity_cols, self.entity_tables)


# ---------------------------------------------------------------------------
# Task 4 — Item: next-event time regression
#
# Business meaning : How many seconds until the next activity for this item?
#                    Helps warehouse staff prioritise picking queues.
# Signal           : Items recently picked move quickly to packaging; items
#                    blocked by a stock-out event stall noticeably longer.
# ---------------------------------------------------------------------------

class ItemNextTime(MEntityTask):
    """Days until the next event for an active item."""

    timedelta    = _DELTA
    task_type    = TaskType.REGRESSION
    object_types = ("items",)
    metrics      = [mae, rmse, mse, r2]

    def make_table(self, db: Database, timestamps: pd.Series) -> Table:
        df = build_next_time_table(
            db,
            object_type = "items",
            times       = timestamps,
            delta_back  = _BACK_30,
            delta_fwd   = _FWD_30,
        )
        return to_relbench_table(df, self.entity_cols, self.entity_tables)


# ---------------------------------------------------------------------------
# Task 5 — Order: remaining time regression
#
# Business meaning : How many seconds until the last future event for this
#                    order (complete fulfilment)?  Classic end-to-end cycle
#                    time prediction.
# Signal           : Early-stage orders have long tails; orders near payment
#                    or delivery are almost complete.
# ---------------------------------------------------------------------------

class OrderRemainingTime(MEntityTask):
    """Days until the final event for an active order."""

    timedelta    = _DELTA
    task_type    = TaskType.REGRESSION
    object_types = ("orders",)
    metrics      = [mae, rmse, mse, r2]

    def make_table(self, db: Database, timestamps: pd.Series) -> Table:
        df = build_remaining_time_table(
            db,
            object_type = "orders",
            times       = timestamps,
            delta_back  = _BACK_30,
        )
        return to_relbench_table(df, self.entity_cols, self.entity_tables)


# ---------------------------------------------------------------------------
# Task 6 — Package: remaining time regression
#
# Business meaning : How many seconds until the final event for this package?
#                    Useful for delivery-time promises and logistics planning.
# Signal           : Packages with prior failed deliveries have longer
#                    remaining times; newly sent packages cluster tightly.
# ---------------------------------------------------------------------------

class PackageRemainingTime(MEntityTask):
    """Days until the final event for an active package."""

    timedelta    = _DELTA
    task_type    = TaskType.REGRESSION
    object_types = ("packages",)
    metrics      = [mae, rmse, mse, r2]

    def make_table(self, db: Database, timestamps: pd.Series) -> Table:
        df = build_remaining_time_table(
            db,
            object_type = "packages",
            times       = timestamps,
            delta_back  = _BACK_30,
        )
        return to_relbench_table(df, self.entity_cols, self.entity_tables)


# ---------------------------------------------------------------------------
# Task 7 — Order: will it experience a stock-out within 14 days?
#
# Business meaning : Will this order hit an "item out of stock" event in the
#                    next two weeks?  Early detection allows replenishment or
#                    order splitting before customer impact.
# Signal           : Orders containing products with low reorder frequency or
#                    recent prior stock-outs are significantly more at risk.
# ---------------------------------------------------------------------------

class OrderStockoutWithin14d(MEntityTask):
    """Binary: order hits a stock-out event within 14 days of observation."""

    timedelta    = _DELTA
    task_type    = TaskType.BINARY_CLASSIFICATION
    object_types = ("orders",)
    metrics      = [accuracy, f1, roc_auc]

    def make_table(self, db: Database, timestamps: pd.Series) -> Table:
        df = build_event_within_table(
            db,
            object_type        = "orders",
            times              = timestamps,
            target_event_types = ["item out of stock"],
            delta_back         = _BACK_30,
            delta_fwd          = _FWD_14,
        )
        return to_relbench_table(df, self.entity_cols, self.entity_tables)


# ---------------------------------------------------------------------------
# Task 8 — Order × Employee pair: will they interact again within 14 days?
#
# Business meaning : Given an order and employee observed together recently,
#                    will they co-appear in a future event within two weeks?
#                    Useful for predicting employee workload reassignments.
# Entity columns   : object_id (orders), object_id_partner (employees)
# Signal           : Employees repeatedly handle the same accounts/products;
#                    historical co-occurrence is a strong reassignment signal.
# ---------------------------------------------------------------------------

_PAIR_COL = "object_id_partner"

_ORDER_EMPLOYEE_EVENTS = [
    "confirm order",
    "pick item",
    "create package",
    "send package",
]


class OrderEmployeePairInteraction(MEntityTask):
    """Binary: order–employee pair co-appears in a future work event."""

    timedelta      = _DELTA
    task_type      = TaskType.BINARY_CLASSIFICATION
    entity_cols    = (OBJECT_ID_COL, _PAIR_COL)
    entity_tables  = (OBJECT_TABLE, OBJECT_TABLE)
    object_types   = ("orders", "employees")
    metrics        = [accuracy, f1, roc_auc]

    def make_table(self, db: Database, timestamps: pd.Series) -> Table:
        df = build_pair_interaction_table(
            db,
            src_type                = "orders",
            dst_type                = "employees",
            times                   = timestamps,
            interaction_event_types = _ORDER_EMPLOYEE_EVENTS,
            delta_back              = _BACK_30,
            delta_fwd               = _FWD_14,
            pair_col                = _PAIR_COL,
        )
        return to_relbench_table(df, self.entity_cols, self.entity_tables)
