import pandas as pd
from pandas import Series
from relbench.base import Database, Table, TaskType
from relbench.metrics import accuracy, auprc, f1, mae, mse, r2, rmse, roc_auc

from data.const import OBJECT_ID_COL, OBJECT_TABLE, TIME_COL
from data.wrapper import check_dbs
from .utils.custom import MEntityTask
from .utils import (
    build_next_event_table,
    build_next_time_table,
    build_event_within_table,
    build_remaining_time_table,
)


class CaseRNextEvent(MEntityTask):
    timedelta = pd.Timedelta(days=7)
    num_eval_timestamps = 40
    task_type = TaskType.MULTICLASS_CLASSIFICATION
    entity_cols = (OBJECT_ID_COL,)
    entity_tables = (OBJECT_TABLE,)
    time_col = TIME_COL
    target_col = "target"
    object_types = ("Case_R",)
    event_types = [
        "A_Accepted",
        "A_Cancelled",
        "A_Create Application",
        "A_Denied",
        "O_Accepted",
        "O_Cancelled",
        "O_Create Offer",
        "O_Returned",
        "O_Sent (mail and online)",
        "W_Assess potential fraud",
        "W_Call after offers",
        "W_Call incomplete files",
        "W_Complete application",
        "W_Handle leads",
        "W_Validate application",
    ]
    num_classes = 15
    metrics = [accuracy, f1]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        return Table(
            df=build_next_event_table(db, self.object_types[0], timestamps, self.event_types),
            fkey_col_to_pkey_table={self.entity_cols[0]: self.entity_tables[0]},
            pkey_col=None,
            time_col=self.time_col,
        )


class CaseRNextTime(MEntityTask):
    timedelta = pd.Timedelta(days=7)
    num_eval_timestamps = 40
    task_type = TaskType.REGRESSION
    entity_cols = (OBJECT_ID_COL,)
    entity_tables = (OBJECT_TABLE,)
    time_col = TIME_COL
    target_col = "target"
    object_types = ("Case_R",)
    metrics = [mae, mse, rmse, r2]

    # def make_target_transform(self) -> Log1pZScoreTargetTransform:
    #     return Log1pZScoreTargetTransform()

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        return Table(
            df=build_next_time_table(db, self.object_types[0], timestamps),
            fkey_col_to_pkey_table={self.entity_cols[0]: self.entity_tables[0]},
            pkey_col=None,
            time_col=self.time_col,
        )


class CaseRRemainingTime(MEntityTask):
    timedelta = pd.Timedelta(days=7)
    num_eval_timestamps = 40
    task_type = TaskType.REGRESSION
    entity_cols = (OBJECT_ID_COL,)
    entity_tables = (OBJECT_TABLE,)
    time_col = TIME_COL
    target_col = "target"
    object_types = ("Case_R",)
    metrics = [mae, mse, rmse, r2]

    # def make_target_transform(self) -> ZScoreTargetTransform:
    #     return ZScoreTargetTransform()

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        return Table(
            df=build_remaining_time_table(db, self.object_types[0], timestamps),
            fkey_col_to_pkey_table={self.entity_cols[0]: self.entity_tables[0]},
            pkey_col=None,
            time_col=self.time_col,
        )


class OfferCancelledWithin30Days(MEntityTask):
    timedelta = pd.Timedelta(days=30)
    num_eval_timestamps = 1
    task_type = TaskType.BINARY_CLASSIFICATION
    entity_cols = (OBJECT_ID_COL,)
    entity_tables = (OBJECT_TABLE,)
    time_col = TIME_COL
    target_col = "target"
    object_types = ("Offer",)
    metrics = [accuracy, f1, auprc, roc_auc]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        return Table(
            df=build_event_within_table(
                db=db,
                object_type="Offer",
                event_type="O_Cancelled",
                times=timestamps,
                delta=self.timedelta,
            ),
            fkey_col_to_pkey_table={self.entity_cols[0]: self.entity_tables[0]},
            pkey_col=None,
            time_col=self.time_col,
        )
