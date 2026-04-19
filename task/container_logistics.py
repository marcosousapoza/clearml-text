"""Predictive tasks for the container_logistics dataset.

8 tasks across 4 label families:
  next-event classification  (2 tasks)
  next-time regression       (2 tasks)
  remaining-time regression  (2 tasks)
  binary event-within-window (1 task)
  multi-entity pair binary   (1 task)

Object types in this log
------------------------
  Container, Customer Order, Transport Document, Vehicle, Truck, Forklift,
  Handling Unit

Event types (14)
----------------
  Register Customer Order, Order Empty Containers, Pick Up Empty Container,
  Collect Goods, Weigh, Load Truck, Drive to Terminal, Place in Stock,
  Bring to Loading Bay, Load to Vehicle, Book Vehicles,
  Create Transport Document, Reschedule Container, Depart
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

_CONTAINER_TRANSIT_EVENTS = [
    "Pick Up Empty Container",
    "Weigh",
    "Load Truck",
    "Drive to Terminal",
    "Place in Stock",
    "Bring to Loading Bay",
    "Load to Vehicle",
    "Reschedule Container",
    "Depart",
]

_TRANSPORT_DOC_EVENTS = [
    "Create Transport Document",
    "Book Vehicles",
    "Bring to Loading Bay",
    "Load to Vehicle",
    "Depart",
]


# ---------------------------------------------------------------------------
# Shared timedelta / window presets
# ---------------------------------------------------------------------------

_BACK_30  = pd.Timedelta(days=30)
_FWD_14   = pd.Timedelta(days=14)
_FWD_30   = pd.Timedelta(days=30)
_DELTA    = pd.Timedelta(days=7)


# ---------------------------------------------------------------------------
# Task 1 — Container: next event classification
#
# Business meaning : After any recent container activity, which of the 9
#                    transit milestones occurs next for this container?
# Signal           : Container routing follows a largely sequential process
#                    (pick-up → weigh → load → drive → depart), so the
#                    immediately preceding event is highly predictive.
# ---------------------------------------------------------------------------

class ContainerNextEvent(MEntityTask):
    """Next transit event for a container (9-class)."""

    timedelta           = _DELTA
    task_type           = TaskType.MULTICLASS_CLASSIFICATION
    object_types        = ("Container",)
    num_classes         = len(_CONTAINER_TRANSIT_EVENTS)
    metrics             = [accuracy, f1, roc_auc]

    def make_table(self, db: Database, timestamps: pd.Series) -> Table:
        df = build_next_event_table(
            db,
            object_type = "Container",
            times       = timestamps,
            event_types = _CONTAINER_TRANSIT_EVENTS,
            delta_back  = _BACK_30,
            delta_fwd   = _FWD_14,
        )
        return to_relbench_table(df, self.entity_cols, self.entity_tables)


# ---------------------------------------------------------------------------
# Task 2 — Transport Document: next event classification
#
# Business meaning : After recent transport-document activity, which of the
#                    5 document-lifecycle events occurs next?
# Signal           : Document lifecycle is strictly ordered (Create → Book →
#                    Bring to Bay → Load to Vehicle → Depart), making current
#                    stage a strong predictor of the next step.
# ---------------------------------------------------------------------------

class TransportDocNextEvent(MEntityTask):
    """Next lifecycle event for a transport document (5-class)."""

    timedelta           = _DELTA
    task_type           = TaskType.MULTICLASS_CLASSIFICATION
    object_types        = ("Transport Document",)
    num_classes         = len(_TRANSPORT_DOC_EVENTS)
    metrics             = [accuracy, f1, roc_auc]

    def make_table(self, db: Database, timestamps: pd.Series) -> Table:
        df = build_next_event_table(
            db,
            object_type = "Transport Document",
            times       = timestamps,
            event_types = _TRANSPORT_DOC_EVENTS,
            delta_back  = _BACK_30,
            delta_fwd   = _FWD_14,
        )
        return to_relbench_table(df, self.entity_cols, self.entity_tables)


# ---------------------------------------------------------------------------
# Task 3 — Container: next-event time regression
#
# Business meaning : How many seconds until the next activity for this
#                    container?  Useful for scheduling and yard management.
# Signal           : Containers that have just been picked up or weighed tend
#                    to advance quickly; those in stock can wait for days.
# ---------------------------------------------------------------------------

class ContainerNextTime(MEntityTask):
    """Days until the next event for an active container."""

    timedelta    = _DELTA
    task_type    = TaskType.REGRESSION
    object_types = ("Container",)
    metrics      = [mae, rmse, mse, r2]

    def make_table(self, db: Database, timestamps: pd.Series) -> Table:
        df = build_next_time_table(
            db,
            object_type = "Container",
            times       = timestamps,
            delta_back  = _BACK_30,
            delta_fwd   = _FWD_30,
        )
        return to_relbench_table(df, self.entity_cols, self.entity_tables)


# ---------------------------------------------------------------------------
# Task 4 — Customer Order: next-event time regression
#
# Business meaning : How long until the next event for this customer order?
#                    Helps operations teams prioritise follow-up actions.
# Signal           : Order age and completeness of the current stage predict
#                    processing lag well.
# ---------------------------------------------------------------------------

class CustomerOrderNextTime(MEntityTask):
    """Days until the next event for an active customer order."""

    timedelta    = _DELTA
    task_type    = TaskType.REGRESSION
    object_types = ("Customer Order",)
    metrics      = [mae, rmse, mse, r2]

    def make_table(self, db: Database, timestamps: pd.Series) -> Table:
        df = build_next_time_table(
            db,
            object_type = "Customer Order",
            times       = timestamps,
            delta_back  = _BACK_30,
            delta_fwd   = _FWD_30,
        )
        return to_relbench_table(df, self.entity_cols, self.entity_tables)


# ---------------------------------------------------------------------------
# Task 5 — Container: remaining time regression
#
# Business meaning : How many seconds until the last future event for this
#                    container (i.e. until departure)?  Classic cycle-time
#                    prediction for logistics SLA monitoring.
# Signal           : Containers close to departure have few remaining steps;
#                    those just picked up have a long sequence ahead.
# ---------------------------------------------------------------------------

class ContainerRemainingTime(MEntityTask):
    """Days until the final event (departure) for an active container."""

    timedelta    = _DELTA
    task_type    = TaskType.REGRESSION
    object_types = ("Container",)
    metrics      = [mae, rmse, mse, r2]

    def make_table(self, db: Database, timestamps: pd.Series) -> Table:
        df = build_remaining_time_table(
            db,
            object_type = "Container",
            times       = timestamps,
            delta_back  = _BACK_30,
        )
        return to_relbench_table(df, self.entity_cols, self.entity_tables)


# ---------------------------------------------------------------------------
# Task 6 — Customer Order: remaining time regression
#
# Business meaning : How many seconds until the last future event for this
#                    customer order (complete fulfilment / departure)?
# Signal           : Order maturity and open container count drive remaining
#                    processing time.
# ---------------------------------------------------------------------------

class CustomerOrderRemainingTime(MEntityTask):
    """Days until the final event for an active customer order."""

    timedelta    = _DELTA
    task_type    = TaskType.REGRESSION
    object_types = ("Customer Order",)
    metrics      = [mae, rmse, mse, r2]

    def make_table(self, db: Database, timestamps: pd.Series) -> Table:
        df = build_remaining_time_table(
            db,
            object_type = "Customer Order",
            times       = timestamps,
            delta_back  = _BACK_30,
        )
        return to_relbench_table(df, self.entity_cols, self.entity_tables)


# ---------------------------------------------------------------------------
# Task 7 — Container: will it be rescheduled within 14 days?
#
# Business meaning : Is this container at risk of rescheduling in the next
#                    two weeks?  Early warning allows planners to reallocate
#                    vehicles before delays cascade.
# Signal           : Containers whose transport documents are heavily loaded
#                    or that have already been weighed/placed in stock are
#                    more prone to rescheduling.
# ---------------------------------------------------------------------------

class ContainerRescheduledWithin14d(MEntityTask):
    """Binary: container rescheduled within 14 days of observation."""

    timedelta    = _DELTA
    task_type    = TaskType.BINARY_CLASSIFICATION
    object_types = ("Container",)
    metrics      = [accuracy, f1, roc_auc]

    def make_table(self, db: Database, timestamps: pd.Series) -> Table:
        df = build_event_within_table(
            db,
            object_type         = "Container",
            times               = timestamps,
            target_event_types  = ["Reschedule Container"],
            delta_back          = _BACK_30,
            delta_fwd           = _FWD_14,
        )
        return to_relbench_table(df, self.entity_cols, self.entity_tables)


# ---------------------------------------------------------------------------
# Task 8 — Container × Truck pair: will they interact again within 14 days?
#
# Business meaning : Given a container and truck observed together recently,
#                    will they co-appear in a Load Truck or Drive to Terminal
#                    event within the next two weeks?  Useful for scheduling
#                    truck assignments.
# Entity columns   : object_id (Container), object_id_partner (Truck)
# Signal           : Trucks repeatedly serve the same container routes;
#                    historical co-occurrence predicts future assignment.
# ---------------------------------------------------------------------------

_CONTAINER_TRUCK_EVENTS = ["Load Truck", "Drive to Terminal"]
_PAIR_COL = "object_id_partner"


class ContainerTruckPairInteraction(MEntityTask):
    """Binary: container–truck pair co-appears in a future loading event."""

    timedelta      = _DELTA
    task_type      = TaskType.BINARY_CLASSIFICATION
    entity_cols    = (OBJECT_ID_COL, _PAIR_COL)
    entity_tables  = (OBJECT_TABLE, OBJECT_TABLE)
    object_types   = ("Container", "Truck")
    metrics        = [accuracy, f1, roc_auc]

    def make_table(self, db: Database, timestamps: pd.Series) -> Table:
        df = build_pair_interaction_table(
            db,
            src_type                 = "Container",
            dst_type                 = "Truck",
            times                    = timestamps,
            interaction_event_types  = _CONTAINER_TRUCK_EVENTS,
            delta_back               = _BACK_30,
            delta_fwd                = _FWD_14,
            pair_col                 = _PAIR_COL,
        )
        return to_relbench_table(df, self.entity_cols, self.entity_tables)
