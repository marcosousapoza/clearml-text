"""BPI2019 task definitions.

Single-entity tasks on POItem objects (dominant lifecycle entity).
Multi-entity tasks on POItem × Vendor observed pairs.

Dataset stats: ~3400 active POItems/day, 382-day span, lifecycle median 64 days.
Look-back 3 days keeps only freshly active items (~500–1000 active per timestamp).
num_eval_timestamps=15 yields ~50k train rows per task.
"""

import pandas as pd
from pandas import Series
from relbench.base import Database, Table, TaskType
from relbench.metrics import accuracy, f1, mae, mse, r2, rmse, roc_auc

from data.const import O2O_DST_COL, O2O_SRC_COL, OBJECT_TABLE
from data.wrapper import check_dbs
from .utils import (
    MEntityTask,
    build_generic_next_event_table,
    build_generic_next_time_table,
    build_generic_remaining_time_table,
    build_generic_pair_next_event_table,
    build_generic_pair_next_time_table,
)

POITEM_EVENT_TYPES = [
    "Block Purchase Order Item",
    "Cancel Goods Receipt",
    "Cancel Invoice Receipt",
    "Cancel Subsequent Invoice",
    "Change Approval for Purchase Order",
    "Change Delivery Indicator",
    "Change Price",
    "Change Quantity",
    "Change Storage Location",
    "Change payment term",
    "Clear Invoice",
    "Create Purchase Order Item",
    "Create Purchase Requisition Item",
    "Delete Purchase Order Item",
    "Reactivate Purchase Order Item",
    "Receive Order Confirmation",
    "Record Goods Receipt",
    "Record Invoice Receipt",
    "Record Service Entry Sheet",
    "Record Subsequent Invoice",
    "Release Purchase Order",
    "Release Purchase Requisition",
    "Remove Payment Block",
    "SRM: Created",
    "SRM: Deleted",
    "SRM: Document Completed",
    "SRM: In Transfer to Execution Syst.",
    "SRM: Ordered",
    "SRM: Transaction Completed",
    "SRM: Transfer Failed (E.Sys.)",
    "Set Payment Block",
    "Update Order Confirmation",
    "Vendor creates debit memo",
    "Vendor creates invoice",
]

# Events that jointly involve a POItem and Vendor in the same event.
POITEM_VENDOR_PAIR_EVENT_TYPES = [
    "Clear Invoice",
    "Record Invoice Receipt",
    "Record Subsequent Invoice",
    "Vendor creates debit memo",
    "Vendor creates invoice",
]

_LOOKBACK = pd.Timedelta(days=3)
_FWD_NEXT = pd.Timedelta(days=21, hours=8)
_N_TIMESTAMPS = 15


# ---------------------------------------------------------------------------
# Single-entity: POItem
# ---------------------------------------------------------------------------

class POItemNextEvent(MEntityTask):
    """Next event type for an active purchase order item."""

    timedelta            = _FWD_NEXT
    num_eval_timestamps  = _N_TIMESTAMPS
    task_type            = TaskType.MULTICLASS_CLASSIFICATION
    object_types         = ("POItem",)
    num_classes          = len(POITEM_EVENT_TYPES)
    metrics              = [accuracy, f1, roc_auc]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_generic_next_event_table(
            db=db,
            object_type="POItem",
            times=timestamps,
            event_types=POITEM_EVENT_TYPES,
            delta_back=_LOOKBACK,
            delta_fwd=self.timedelta,
        )
        return self._make_table(df)


class POItemNextTime(MEntityTask):
    """Seconds until the next event for an active purchase order item."""

    timedelta            = _FWD_NEXT
    num_eval_timestamps  = _N_TIMESTAMPS
    task_type            = TaskType.REGRESSION
    object_types         = ("POItem",)
    metrics              = [mae, mse, rmse, r2]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_generic_next_time_table(
            db=db,
            object_type="POItem",
            times=timestamps,
            delta_back=_LOOKBACK,
        )
        return self._make_table(df)


class POItemRemainingTime(MEntityTask):
    """Days until the final event in an active purchase order item's lifecycle."""

    timedelta            = _FWD_NEXT
    num_eval_timestamps  = _N_TIMESTAMPS
    task_type            = TaskType.REGRESSION
    object_types         = ("POItem",)
    metrics              = [mae, mse, rmse, r2]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_generic_remaining_time_table(
            db=db,
            object_type="POItem",
            times=timestamps,
            delta_back=_LOOKBACK,
        )
        return self._make_table(df)


# ---------------------------------------------------------------------------
# Multi-entity: POItem × Vendor observed pairs
# ---------------------------------------------------------------------------

class POItemVendorPairNextEvent(MEntityTask):
    """Next shared event type for an observed POItem–Vendor pair."""

    timedelta            = _FWD_NEXT
    num_eval_timestamps  = _N_TIMESTAMPS
    task_type            = TaskType.MULTICLASS_CLASSIFICATION
    object_types         = ("POItem", "Vendor")
    entity_cols          = (O2O_SRC_COL, O2O_DST_COL)
    entity_tables        = (OBJECT_TABLE, OBJECT_TABLE)
    num_classes          = len(POITEM_VENDOR_PAIR_EVENT_TYPES)
    metrics              = [accuracy, f1, roc_auc]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_generic_pair_next_event_table(
            db=db,
            src_type="POItem",
            dst_type="Vendor",
            times=timestamps,
            event_types=POITEM_VENDOR_PAIR_EVENT_TYPES,
            delta_back=_LOOKBACK,
            delta_fwd=self.timedelta,
        )
        return self._make_table(df)


class POItemVendorPairNextTime(MEntityTask):
    """Seconds until the next shared event for an observed POItem–Vendor pair."""

    timedelta            = _FWD_NEXT
    num_eval_timestamps  = _N_TIMESTAMPS
    task_type            = TaskType.REGRESSION
    object_types         = ("POItem", "Vendor")
    entity_cols          = (O2O_SRC_COL, O2O_DST_COL)
    entity_tables        = (OBJECT_TABLE, OBJECT_TABLE)
    metrics              = [mae, mse, rmse, r2]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_generic_pair_next_time_table(
            db=db,
            src_type="POItem",
            dst_type="Vendor",
            times=timestamps,
            delta_back=_LOOKBACK,
        )
        return self._make_table(df)
