import pandas as pd
from pandas import Series
from relbench.base import Database, Table, TaskType
from relbench.metrics import accuracy, auprc, f1, mae, mse, r2, rmse, roc_auc

from data.const import OBJECT_ID_COL, OBJECT_TABLE, O2O_DST_COL, O2O_SRC_COL, TIME_COL
from data.wrapper import check_dbs
from .utils.custom import MEntityTask
from .utils import (
    build_next_event_table,
    build_next_time_table,
    build_pair_event_within_table,
    build_remaining_time_table,
)


class POItemNextEvent(MEntityTask):
    timedelta = pd.Timedelta(days=14)
    num_eval_timestamps = 40
    task_type = TaskType.MULTICLASS_CLASSIFICATION
    entity_cols = (OBJECT_ID_COL,)
    entity_tables = (OBJECT_TABLE,)
    time_col = TIME_COL
    target_col = "target"
    object_type = "POItem"
    num_classes = 30
    metrics = [accuracy, f1]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        return Table(
            df=build_next_event_table(db, self.object_type, timestamps),
            fkey_col_to_pkey_table={self.entity_cols[0]: self.entity_tables[0]},
            pkey_col=None,
            time_col=self.time_col,
        )


class POItemNextTime(MEntityTask):
    timedelta = pd.Timedelta(days=14)
    num_eval_timestamps = 40
    task_type = TaskType.REGRESSION
    entity_cols = (OBJECT_ID_COL,)
    entity_tables = (OBJECT_TABLE,)
    time_col = TIME_COL
    target_col = "target"
    object_type = "POItem"
    metrics = [mae, mse, rmse, r2]

    # def make_target_transform(self) -> Log1pZScoreTargetTransform:
    #     return Log1pZScoreTargetTransform()

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        return Table(
            df=build_next_time_table(db, self.object_type, timestamps),
            fkey_col_to_pkey_table={self.entity_cols[0]: self.entity_tables[0]},
            pkey_col=None,
            time_col=self.time_col,
        )


class POItemRemainingTime(MEntityTask):
    timedelta = pd.Timedelta(days=14)
    num_eval_timestamps = 40
    task_type = TaskType.REGRESSION
    entity_cols = (OBJECT_ID_COL,)
    entity_tables = (OBJECT_TABLE,)
    time_col = TIME_COL
    target_col = "target"
    object_type = "POItem"
    metrics = [mae, mse, rmse, r2]

    # def make_target_transform(self) -> Log1pZScoreTargetTransform:
    #     return Log1pZScoreTargetTransform()

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        return Table(
            df=build_remaining_time_table(db, self.object_type, timestamps),
            fkey_col_to_pkey_table={self.entity_cols[0]: self.entity_tables[0]},
            pkey_col=None,
            time_col=self.time_col,
        )


class POItemVendorClearInvoiceWithin30Days(MEntityTask):
    timedelta = pd.Timedelta(days=30)
    num_eval_timestamps = 40
    task_type = TaskType.BINARY_CLASSIFICATION
    entity_cols = (O2O_SRC_COL, O2O_DST_COL)
    entity_tables = (OBJECT_TABLE, OBJECT_TABLE)
    time_col = TIME_COL
    target_col = "target"
    metrics = [accuracy, f1, auprc, roc_auc]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        return Table(
            df=build_pair_event_within_table(
                db=db,
                object_types=("POItem", "Vendor"),
                event_type="Clear Invoice",
                times=timestamps,
                delta=self.timedelta,
            ),
            fkey_col_to_pkey_table={
                self.entity_cols[0]: self.entity_tables[0],
                self.entity_cols[1]: self.entity_tables[1],
            },
            pkey_col=None,
            time_col=self.time_col,
        )
