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


class ContainerNextEvent(MEntityTask):
    timedelta = pd.Timedelta(hours=12)
    num_eval_timestamps = 40
    task_type = TaskType.MULTICLASS_CLASSIFICATION
    entity_cols = (OBJECT_ID_COL,)
    entity_tables = (OBJECT_TABLE,)
    time_col = TIME_COL
    target_col = "target"
    object_types = ("Container",)
    event_types = (
        "Bring to Loading Bay",
        "Depart",
        "Drive to Terminal",
        "Load Truck",
        "Load to Vehicle",
        "Pick Up Empty Container",
        "Place in Stock",
        "Reschedule Container",
        "Weigh",
    )
    num_classes = 9
    metrics = [accuracy, f1, roc_auc]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        return Table(
            df=build_next_event_table(db, self.object_types[0], timestamps, self.event_types),
            fkey_col_to_pkey_table={self.entity_cols[0]: self.entity_tables[0]},
            pkey_col=None,
            time_col=self.time_col,
        )


class ContainerNextTime(MEntityTask):
    timedelta = pd.Timedelta(hours=12)
    num_eval_timestamps = 40
    task_type = TaskType.REGRESSION
    entity_cols = (OBJECT_ID_COL,)
    entity_tables = (OBJECT_TABLE,)
    time_col = TIME_COL
    target_col = "target"
    object_types = ("Container",)
    metrics = [mae, mse, rmse, r2]

    # def make_target_transform(self) -> ZScoreTargetTransform:
    #     return ZScoreTargetTransform()

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        return Table(
            df=build_next_time_table(db, self.object_types[0], timestamps),
            fkey_col_to_pkey_table={self.entity_cols[0]: self.entity_tables[0]},
            pkey_col=None,
            time_col=self.time_col,
        )


class ContainerRemainingTime(MEntityTask):
    timedelta = pd.Timedelta(hours=12)
    num_eval_timestamps = 40
    task_type = TaskType.REGRESSION
    entity_cols = (OBJECT_ID_COL,)
    entity_tables = (OBJECT_TABLE,)
    time_col = TIME_COL
    target_col = "target"
    object_types = ("Container",)
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


class TransportDocumentVehicleDepartWithin7Days(MEntityTask):
    timedelta = pd.Timedelta(days=7)
    num_eval_timestamps = 40
    task_type = TaskType.BINARY_CLASSIFICATION
    entity_cols = (O2O_SRC_COL, O2O_DST_COL)
    entity_tables = (OBJECT_TABLE, OBJECT_TABLE)
    time_col = TIME_COL
    target_col = "target"
    object_types = ("Transport Document", "Vehicle")
    metrics = [accuracy, f1, auprc, roc_auc]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        return Table(
            df=build_pair_event_within_table(
                db=db,
                object_types=("Transport Document", "Vehicle"),
                event_type="Depart",
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
