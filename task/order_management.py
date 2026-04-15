import pandas as pd
from pandas import Series
from relbench.base import Database, Table, TaskType
from relbench.metrics import accuracy, auprc, f1, mae, mse, r2, rmse, roc_auc

from data.const import O2O_DST_COL, O2O_SRC_COL, OBJECT_TABLE
from data.wrapper import check_dbs
from .utils import (
    MEntityTask,
    build_complete_pair_event_within_table,
    build_next_event_table,
    build_next_time_table,
    build_remaining_time_table,
)


class OrderNextEvent(MEntityTask):
    timedelta = pd.Timedelta(days=7)
    num_eval_timestamps = 40
    task_type = TaskType.MULTICLASS_CLASSIFICATION
    object_types = ("orders",)
    event_types = (
        "confirm order",
        "pay order",
        "payment reminder",
    )
    num_classes = 3
    metrics = [accuracy, f1]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        return self._make_table(
            build_next_event_table(db, self.object_types[0], timestamps, self.event_types)
        )


class OrderNextTime(MEntityTask):
    timedelta = pd.Timedelta(days=7)
    num_eval_timestamps = 40
    task_type = TaskType.REGRESSION
    object_types = ("orders",)
    metrics = [mae, mse, rmse, r2]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        return self._make_table(
            build_next_time_table(db, self.object_types[0], timestamps)
        )


class OrderRemainingTime(MEntityTask):
    timedelta = pd.Timedelta(days=7)
    num_eval_timestamps = 40
    task_type = TaskType.REGRESSION
    object_types = ("orders",)
    metrics = [mae, mse, rmse, r2]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        return self._make_table(
            build_remaining_time_table(db, self.object_types[0], timestamps)
        )


class CustomerProductPlaceOrderWithin14Days(MEntityTask):
    timedelta = pd.Timedelta(days=14)
    num_eval_timestamps = 40
    task_type = TaskType.BINARY_CLASSIFICATION
    # Complete cartesian pair task: every customer × every product combination
    entity_cols = (O2O_SRC_COL, O2O_DST_COL)
    entity_tables = (OBJECT_TABLE, OBJECT_TABLE)
    object_types = ("customers", "products")
    metrics = [accuracy, f1, auprc, roc_auc]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        return self._make_table(
            build_complete_pair_event_within_table(
                db=db,
                object_types=("customers", "products"),
                event_type="place order",
                times=timestamps,
                delta=self.timedelta,
            )
        )
