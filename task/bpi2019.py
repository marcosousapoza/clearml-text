"""Predictive tasks for the bpi2019 dataset.

8 tasks across 4 label families:
  next-event classification  (2 tasks)
  next-time regression       (2 tasks)
  remaining-time regression  (2 tasks)
  binary event-within-window (1 task)
  multi-entity pair binary   (1 task)

Object types in this log
------------------------
  PO, POItem, Resource, Vendor

Event types (36)
----------------
  Operational:
    Create Purchase Order Item, Create Purchase Requisition Item,
    Release Purchase Order, Receive Order Confirmation,
    Update Order Confirmation, Record Goods Receipt, Cancel Goods Receipt,
    Record Invoice Receipt, Cancel Invoice Receipt, Clear Invoice,
    Record Service Entry Sheet, Record Subsequent Invoice,
    Cancel Subsequent Invoice, Vendor creates invoice,
    Vendor creates debit memo

  Change events:
    Change Approval for Purchase Order, Change Currency,
    Change Delivery Indicator, Change Final Invoice Indicator,
    Change Price, Change Quantity, Change Rejection Indicator,
    Change Storage Location, Change payment term

  Block / lifecycle events:
    Block Purchase Order Item, Reactivate Purchase Order Item,
    Delete Purchase Order Item, Set Payment Block, Remove Payment Block

  SRM events:
    SRM: Awaiting Approval, SRM: Change was Transmitted, SRM: Complete,
    SRM: Created, SRM: Deleted, SRM: Document Completed, SRM: Held,
    SRM: In Transfer to Execution Syst., SRM: Incomplete, SRM: Ordered,
    SRM: Transaction Completed

Dataset splits
--------------
  val_timestamp  = 2018-08-31 15:00:00
  test_timestamp = 2018-10-29 15:00:00
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

# Core operational progression events for a POItem
_POITEM_OPERATIONAL_EVENTS = [
    "Create Purchase Order Item",
    "Create Purchase Requisition Item",
    "Release Purchase Order",
    "Receive Order Confirmation",
    "Record Goods Receipt",
    "Record Invoice Receipt",
    "Clear Invoice",
    "Record Service Entry Sheet",
]

# Change/amendment events for a POItem — frequent, dense, good signal
_POITEM_CHANGE_EVENTS = [
    "Create Purchase Order Item",
    "Change Price",
    "Change Quantity",
    "Change Delivery Indicator",
    "Change Final Invoice Indicator",
    "Change Rejection Indicator",
    "Change Storage Location",
    "Change payment term",
    "Change Approval for Purchase Order",
]

# Blocking / cancellation events used for the binary within-window task
_BLOCK_CANCEL_EVENTS = [
    "Block Purchase Order Item",
    "Delete Purchase Order Item",
    "Cancel Goods Receipt",
    "Cancel Invoice Receipt",
]

# Events where a POItem and Vendor co-appear (goods/invoice exchange)
_POITEM_VENDOR_EVENTS = [
    "Record Goods Receipt",
    "Record Invoice Receipt",
    "Receive Order Confirmation",
    "Update Order Confirmation",
    "Vendor creates invoice",
    "Vendor creates debit memo",
    "Record Service Entry Sheet",
]

# ---------------------------------------------------------------------------
# Shared window presets
# ---------------------------------------------------------------------------

_BACK_7   = pd.Timedelta(days=7)
_FWD_14   = pd.Timedelta(days=14)
_FWD_30   = pd.Timedelta(days=30)
_DELTA    = pd.Timedelta(days=7)
_PAIR_COL = "object_id_partner"


# ---------------------------------------------------------------------------
# Task 1 — POItem: next operational event classification
#
# Business meaning : After recent activity on a purchase order item, which of
#                    the 8 key operational milestones (from creation through
#                    invoice clearing) occurs next?
# Signal           : PO item processing follows a well-defined sequence
#                    (create → release → receive confirmation → goods receipt
#                    → invoice → clear), making the current stage a strong
#                    predictor of the next event.
# ---------------------------------------------------------------------------

class POItemNextEvent(MEntityTask):
    """Next operational event for an active PO item (8-class)."""

    timedelta    = _DELTA
    task_type    = TaskType.MULTICLASS_CLASSIFICATION
    object_types = ("POItem",)
    num_classes  = len(_POITEM_OPERATIONAL_EVENTS)
    metrics      = [accuracy, f1, roc_auc]

    def make_table(self, db: Database, timestamps: pd.Series) -> Table:
        df = build_next_event_table(
            db,
            object_type = "POItem",
            times       = timestamps,
            event_types = _POITEM_OPERATIONAL_EVENTS,
            delta_back  = _BACK_7,
            delta_fwd   = _FWD_14,
        )
        return to_relbench_table(df, self.entity_cols, self.entity_tables)


# ---------------------------------------------------------------------------
# Task 2 — POItem: next change/amendment event classification
#
# Business meaning : After recent activity on a PO item, which of the 9
#                    change or amendment events (price, quantity, approval,
#                    etc.) occurs next?  Flags items mid-amendment so
#                    procurement staff can anticipate the next modification.
# Signal           : Items that just had a price change are more likely to
#                    receive a quantity or approval change next; the sequence
#                    of amendment types is predictable.
# ---------------------------------------------------------------------------

class POItemNextChangeEvent(MEntityTask):
    """Next change/amendment event for an active PO item (9-class)."""

    timedelta    = _DELTA
    task_type    = TaskType.MULTICLASS_CLASSIFICATION
    object_types = ("POItem",)
    num_classes  = len(_POITEM_CHANGE_EVENTS)
    metrics      = [accuracy, f1, roc_auc]

    def make_table(self, db: Database, timestamps: pd.Series) -> Table:
        df = build_next_event_table(
            db,
            object_type = "POItem",
            times       = timestamps,
            event_types = _POITEM_CHANGE_EVENTS,
            delta_back  = _BACK_7,
            delta_fwd   = _FWD_14,
        )
        return to_relbench_table(df, self.entity_cols, self.entity_tables)


# ---------------------------------------------------------------------------
# Task 3 — POItem: next-event time regression
#
# Business meaning : How many days until the next activity on this PO item?
#                    Useful for tracking procurement lead times and identifying
#                    stalled items before they breach SLA.
# Signal           : Items awaiting goods receipt after confirmation tend to
#                    have longer gaps than items in the invoicing phase.
# ---------------------------------------------------------------------------

class POItemNextTime(MEntityTask):
    """Days until the next event for an active PO item."""

    timedelta    = _DELTA
    task_type    = TaskType.REGRESSION
    object_types = ("POItem",)
    metrics      = [mae, rmse, mse, r2]

    def make_table(self, db: Database, timestamps: pd.Series) -> Table:
        df = build_next_time_table(
            db,
            object_type = "POItem",
            times       = timestamps,
            delta_back  = _BACK_7,
            delta_fwd   = _FWD_30,
        )
        return to_relbench_table(df, self.entity_cols, self.entity_tables)


# ---------------------------------------------------------------------------
# Task 4 — PO: next-event time regression
#
# Business meaning : How many days until the next event on this purchase order?
#                    Helps procurement managers prioritise follow-up on orders
#                    that may be delayed in SRM approval or execution.
# Signal           : POs in early SRM states (Created, Awaiting Approval)
#                    advance quickly; those held or incomplete stall longer.
# ---------------------------------------------------------------------------

class PONextTime(MEntityTask):
    """Days until the next event for an active PO."""

    timedelta    = _DELTA
    task_type    = TaskType.REGRESSION
    object_types = ("PO",)
    metrics      = [mae, rmse, mse, r2]

    def make_table(self, db: Database, timestamps: pd.Series) -> Table:
        df = build_next_time_table(
            db,
            object_type = "PO",
            times       = timestamps,
            delta_back  = _BACK_7,
            delta_fwd   = _FWD_30,
        )
        return to_relbench_table(df, self.entity_cols, self.entity_tables)


# ---------------------------------------------------------------------------
# Task 5 — POItem: remaining time regression
#
# Business meaning : How many days until the final event for this PO item
#                    (end of procurement lifecycle)?  Classic cycle-time
#                    prediction for procurement process mining.
# Signal           : Items near invoice clearing have little remaining time;
#                    newly created items have a long sequence ahead.
# ---------------------------------------------------------------------------

class POItemRemainingTime(MEntityTask):
    """Days until the final event for an active PO item."""

    timedelta    = _DELTA
    task_type    = TaskType.REGRESSION
    object_types = ("POItem",)
    metrics      = [mae, rmse, mse, r2]

    def make_table(self, db: Database, timestamps: pd.Series) -> Table:
        df = build_remaining_time_table(
            db,
            object_type = "POItem",
            times       = timestamps,
            delta_back  = _BACK_7,
        )
        return to_relbench_table(df, self.entity_cols, self.entity_tables)


# ---------------------------------------------------------------------------
# Task 6 — PO: remaining time regression
#
# Business meaning : How many days until the final event for this purchase
#                    order?  Useful for estimating when procurement contracts
#                    will be fully settled.
# Signal           : POs with all items received and invoiced are close to
#                    completion; those still in SRM approval have longer tails.
# ---------------------------------------------------------------------------

class PORemainingTime(MEntityTask):
    """Days until the final event for an active PO."""

    timedelta    = _DELTA
    task_type    = TaskType.REGRESSION
    object_types = ("PO",)
    metrics      = [mae, rmse, mse, r2]

    def make_table(self, db: Database, timestamps: pd.Series) -> Table:
        df = build_remaining_time_table(
            db,
            object_type = "PO",
            times       = timestamps,
            delta_back  = _BACK_7,
        )
        return to_relbench_table(df, self.entity_cols, self.entity_tables)


# ---------------------------------------------------------------------------
# Task 7 — POItem: will it be blocked or cancelled within 14 days?
#
# Business meaning : Is this PO item at risk of being blocked, deleted, or
#                    having a receipt cancelled in the next two weeks?  Early
#                    warning allows procurement staff to intervene before
#                    downstream invoicing is affected.
# Signal           : Items with outstanding payment blocks, change events, or
#                    rejection indicators are significantly more likely to be
#                    cancelled or blocked shortly after.
# ---------------------------------------------------------------------------

class POItemBlockedWithin14d(MEntityTask):
    """Binary: PO item blocked or cancelled within 14 days of observation."""

    timedelta    = _DELTA
    task_type    = TaskType.BINARY_CLASSIFICATION
    object_types = ("POItem",)
    metrics      = [accuracy, f1, roc_auc]

    def make_table(self, db: Database, timestamps: pd.Series) -> Table:
        df = build_event_within_table(
            db,
            object_type        = "POItem",
            times              = timestamps,
            target_event_types = _BLOCK_CANCEL_EVENTS,
            delta_back         = _BACK_7,
            delta_fwd          = _FWD_14,
        )
        return to_relbench_table(df, self.entity_cols, self.entity_tables)


# ---------------------------------------------------------------------------
# Task 8 — POItem × Vendor pair: will they co-appear within 14 days?
#
# Business meaning : Given a PO item and vendor observed together recently,
#                    will they co-appear in a future goods or invoice exchange
#                    event within the next two weeks?  Useful for predicting
#                    which vendor relationships will be active next in the
#                    procurement cycle.
# Entity columns   : object_id (POItem), object_id_partner (Vendor)
# Signal           : Vendors repeatedly supply the same PO items; historical
#                    co-occurrence in receipt/invoice events predicts future
#                    activity on the same item-vendor pair.
# ---------------------------------------------------------------------------

class POItemVendorPairInteraction(MEntityTask):
    """Binary: POItem–Vendor pair co-appears in a future exchange event."""

    timedelta      = _DELTA
    task_type      = TaskType.BINARY_CLASSIFICATION
    entity_cols    = (OBJECT_ID_COL, _PAIR_COL)
    entity_tables  = (OBJECT_TABLE, OBJECT_TABLE)
    object_types   = ("POItem", "Vendor")
    metrics        = [accuracy, f1, roc_auc]

    def make_table(self, db: Database, timestamps: pd.Series) -> Table:
        df = build_pair_interaction_table(
            db,
            src_type                = "POItem",
            dst_type                = "Vendor",
            times                   = timestamps,
            interaction_event_types = _POITEM_VENDOR_EVENTS,
            delta_back              = _BACK_7,
            delta_fwd               = _FWD_14,
            pair_col                = _PAIR_COL,
        )
        return to_relbench_table(df, self.entity_cols, self.entity_tables)
