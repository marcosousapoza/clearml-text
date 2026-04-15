import pandas as pd
from pandas import Series
from relbench.base import Database, Table, TaskType
from relbench.metrics import accuracy, auprc, f1, mae, mse, r2, rmse, roc_auc

from data.const import O2O_DST_COL, O2O_SRC_COL, OBJECT_TABLE
from data.wrapper import check_dbs
from .utils import (
    QuantileTargetTransform,
    MEntityTask,
    build_next_event_table,
    build_next_time_table,
    build_pair_event_within_table,
    build_remaining_time_table,
)


class POItemNextEvent(MEntityTask):
    timedelta = pd.Timedelta(days=14)
    num_eval_timestamps = 40
    task_type = TaskType.MULTICLASS_CLASSIFICATION
    object_types = ("POItem",)
    event_types = [
        "Block Purchase Order Item",
        "Cancel Goods Receipt",
        "Cancel Invoice Receipt",
        "Cancel Subsequent Invoice",
        "Change Approval for Purchase Order",
        "Change Currency",
        "Change Delivery Indicator",
        "Change Final Invoice Indicator",
        "Change Price",
        "Change Quantity",
        "Change Rejection Indicator",
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
        "Remove Payment Block",
        "SRM: Awaiting Approval",
        "SRM: Change was Transmitted",
        "SRM: Complete",
        "SRM: Created",
        "SRM: Deleted",
        "SRM: Document Completed",
        "SRM: Held",
        "SRM: In Transfer to Execution Syst.",
        "SRM: Incomplete",
        "SRM: Ordered",
        "SRM: Transaction Completed",
        "Set Payment Block",
        "Update Order Confirmation",
        "Vendor creates debit memo",
        "Vendor creates invoice",
    ]
    num_classes = 40
    metrics = [accuracy, f1, roc_auc]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        return self._make_table(
            build_next_event_table(db, self.object_types[0], timestamps, self.event_types)
        )


class POItemNextTime(MEntityTask):
    timedelta = pd.Timedelta(days=14)
    num_eval_timestamps = 40
    task_type = TaskType.REGRESSION
    object_types = ("POItem",)
    metrics = [mae, mse, rmse, r2]

    def make_target_transform(self): return QuantileTargetTransform()

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        return self._make_table(
            build_next_time_table(db, self.object_types[0], timestamps)
        )


class POItemRemainingTime(MEntityTask):
    timedelta = pd.Timedelta(days=14)
    num_eval_timestamps = 40
    task_type = TaskType.REGRESSION
    object_types = ("POItem",)
    metrics = [mae, mse, rmse, r2]

    def make_target_transform(self): return QuantileTargetTransform()

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        return self._make_table(
            build_remaining_time_table(db, self.object_types[0], timestamps)
        )


class POItemVendorClearInvoiceWithin30Days(MEntityTask):
    timedelta = pd.Timedelta(days=30)
    num_eval_timestamps = 40
    task_type = TaskType.BINARY_CLASSIFICATION
    # Pair-entity task: src = POItem, dst = Vendor
    entity_cols = (O2O_SRC_COL, O2O_DST_COL)
    entity_tables = (OBJECT_TABLE, OBJECT_TABLE)
    object_types = ("POItem", "Vendor")
    metrics = [accuracy, f1, auprc, roc_auc]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        return self._make_table(
            build_pair_event_within_table(
                db=db,
                object_types=("POItem", "Vendor"),
                event_type="Clear Invoice",
                times=timestamps,
                delta=self.timedelta,
            )
        )
