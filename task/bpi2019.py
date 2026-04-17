import pandas as pd
from pandas import Series
from relbench.base import Database, Table, TaskType
from relbench.metrics import accuracy, f1, mae, mse, r2, rmse, roc_auc

from data.wrapper import check_dbs
from .utils import (
    Log1pZScoreTargetTransform,
    MEntityTask,
    build_stage_future_distinct_related_count_table,
    build_stage_multiclass_next_event_table,
)


POITEM_CREATION_OUTCOMES = [
    "Vendor creates invoice",
    "Record Goods Receipt",
    "Change Quantity",
]

POITEM_INVOICE_RECEIPT_OUTCOMES = [
    "Clear Invoice",
    "Remove Payment Block",
    "Record Goods Receipt",
    "Cancel Invoice Receipt",
]


class POItemCreationOutcome14Days(MEntityTask):
    """Branch after item creation: invoice, receipt, or quantity-change follow-up."""

    timedelta = pd.Timedelta(days=14)
    task_type = TaskType.MULTICLASS_CLASSIFICATION
    object_types = ("POItem",)
    metrics = [accuracy, f1, roc_auc]
    num_classes = len(POITEM_CREATION_OUTCOMES)
    source_event_type = "Create Purchase Order Item"
    next_event_types = POITEM_CREATION_OUTCOMES
    source_max_age = timedelta

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_stage_multiclass_next_event_table(
            db=db,
            object_type="POItem",
            timestamps=timestamps,
            delta=self.timedelta,
            source_event_type=self.source_event_type,
            next_event_types=self.next_event_types,
            source_max_age=self.source_max_age,
        )
        return self._make_table(df)


class POItemInvoiceReceiptOutcome14Days(MEntityTask):
    """Operational follow-up after invoice receipt: clear, unblock, wait for goods, or cancel."""

    timedelta = pd.Timedelta(days=14)
    task_type = TaskType.MULTICLASS_CLASSIFICATION
    object_types = ("POItem",)
    metrics = [accuracy, f1, roc_auc]
    num_classes = len(POITEM_INVOICE_RECEIPT_OUTCOMES)
    source_event_type = "Record Invoice Receipt"
    next_event_types = POITEM_INVOICE_RECEIPT_OUTCOMES
    source_max_age = timedelta

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_stage_multiclass_next_event_table(
            db=db,
            object_type="POItem",
            timestamps=timestamps,
            delta=self.timedelta,
            source_event_type=self.source_event_type,
            next_event_types=self.next_event_types,
            source_max_age=self.source_max_age,
        )
        return self._make_table(df)


class VendorFutureClearInvoiceItemCount14Days(MEntityTask):
    """How many distinct PO items from this vendor will clear invoices soon?"""

    timedelta = pd.Timedelta(days=14)
    task_type = TaskType.REGRESSION
    object_types = ("Vendor",)
    metrics = [mae, mse, rmse, r2]
    source_event_type = "Vendor creates invoice"
    target_event_type = "Clear Invoice"
    related_object_type = "POItem"
    source_max_age = timedelta

    def make_target_transform(self):
        return Log1pZScoreTargetTransform()

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_stage_future_distinct_related_count_table(
            db=db,
            object_type="Vendor",
            related_object_type=self.related_object_type,
            timestamps=timestamps,
            delta=self.timedelta,
            source_event_type=self.source_event_type,
            target_event_type=self.target_event_type,
            source_max_age=self.source_max_age,
        )
        return self._make_table(df)
