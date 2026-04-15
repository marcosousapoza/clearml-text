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


class ContainerNextEvent(MEntityTask):
    timedelta = pd.Timedelta(hours=12)

    task_type = TaskType.MULTICLASS_CLASSIFICATION
    object_types = ("Container",)
    event_types = [
        "Bring to Loading Bay",
        "Depart",
        "Drive to Terminal",
        "Load Truck",
        "Load to Vehicle",
        "Pick Up Empty Container",
        "Place in Stock",
        "Reschedule Container",
        "Weigh",
    ]
    num_classes = 9
    metrics = [accuracy, f1, roc_auc]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        return self._make_table(
            build_next_event_table(db, self.object_types[0], timestamps, self.event_types)
        )


class ContainerNextTime(MEntityTask):
    timedelta = pd.Timedelta(hours=12)

    task_type = TaskType.REGRESSION
    object_types = ("Container",)
    metrics = [mae, mse, rmse, r2]

    def make_target_transform(self): return QuantileTargetTransform()

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        return self._make_table(
            build_next_time_table(db, self.object_types[0], timestamps)
        )


class ContainerRemainingTime(MEntityTask):
    timedelta = pd.Timedelta(hours=12)

    task_type = TaskType.REGRESSION
    object_types = ("Container",)
    metrics = [mae, mse, rmse, r2]

    def make_target_transform(self): return QuantileTargetTransform()

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        return self._make_table(
            build_remaining_time_table(db, self.object_types[0], timestamps)
        )


class TransportDocumentVehicleDepartWithin7Days(MEntityTask):
    timedelta = pd.Timedelta(days=7)

    task_type = TaskType.BINARY_CLASSIFICATION
    # Pair-entity task: src = Transport Document, dst = Vehicle
    entity_cols = (O2O_SRC_COL, O2O_DST_COL)
    entity_tables = (OBJECT_TABLE, OBJECT_TABLE)
    object_types = ("Transport Document", "Vehicle")
    metrics = [accuracy, f1, auprc, roc_auc]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        return self._make_table(
            build_pair_event_within_table(
                db=db,
                object_types=("Transport Document", "Vehicle"),
                event_type="Depart",
                times=timestamps,
                delta=self.timedelta,
            )
        )
