import pandas as pd
from pandas import Series
from relbench.base import Database, Table, TaskType
from relbench.metrics import accuracy, f1, mae, mse, r2, rmse

from data.const import OBJECT_ID_COL, OBJECT_TABLE, TIME_COL
from data.wrapper import check_dbs
from .utils.custom import MEntityTask
from .utils import build_next_event_table, build_next_time_table, build_remaining_time_table


class ContainerNextEvent(MEntityTask):
    timedelta = pd.Timedelta(hours=12)
    num_eval_timestamps = 40
    task_type = TaskType.MULTICLASS_CLASSIFICATION
    entity_cols = (OBJECT_ID_COL,)
    entity_tables = (OBJECT_TABLE,)
    time_col = TIME_COL
    target_col = "target"
    object_type = "Container"
    num_classes = 10
    metrics = [accuracy, f1]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        return Table(
            df=build_next_event_table(db, self.object_type, timestamps),
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
    object_type = "Container"
    metrics = [mae, mse, rmse, r2]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        return Table(
            df=build_next_time_table(db, self.object_type, timestamps),
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
    object_type = "Container"
    metrics = [mae, mse, rmse, r2]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        return Table(
            df=build_remaining_time_table(db, self.object_type, timestamps),
            fkey_col_to_pkey_table={self.entity_cols[0]: self.entity_tables[0]},
            pkey_col=None,
            time_col=self.time_col,
        )
