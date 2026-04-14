import pandas as pd
from pandas import Series
from relbench.base import Database, Table, TaskType
from relbench.metrics import accuracy, auprc, f1, mae, mse, r2, rmse, roc_auc

from data.wrapper import check_dbs
from .utils import (
    MEntityTask,
    build_event_within_table,
    build_next_event_table,
    build_next_time_table,
    build_remaining_time_table,
)


class ApplicationNextEvent(MEntityTask):
    timedelta = pd.Timedelta(days=7)
    num_eval_timestamps = 40
    task_type = TaskType.MULTICLASS_CLASSIFICATION
    object_types = ("Application",)
    event_types = [
        "A_Accepted",
        "A_Cancelled",
        "A_Complete",
        "A_Concept",
        "A_Create Application",
        "A_Denied",
        "A_Incomplete",
        "A_Pending",
        "A_Submitted",
        "A_Validating",
    ]
    num_classes = 10
    metrics = [accuracy, f1]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        return self._make_table(
            build_next_event_table(db, self.object_types[0], timestamps, self.event_types)
        )


class CaseRNextTime(MEntityTask):
    timedelta = pd.Timedelta(days=7)
    num_eval_timestamps = 40
    task_type = TaskType.REGRESSION
    object_types = ("Case_R",)
    metrics = [mae, mse, rmse, r2]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        return self._make_table(
            build_next_time_table(db, self.object_types[0], timestamps)
        )


class CaseRRemainingTime(MEntityTask):
    timedelta = pd.Timedelta(days=7)
    num_eval_timestamps = 40
    task_type = TaskType.REGRESSION
    object_types = ("Case_R",)
    metrics = [mae, mse, rmse, r2]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        return self._make_table(
            build_remaining_time_table(db, self.object_types[0], timestamps)
        )


class OfferCancelledWithin30Days(MEntityTask):
    timedelta = pd.Timedelta(days=30)
    num_eval_timestamps = 1
    task_type = TaskType.BINARY_CLASSIFICATION
    object_types = ("Offer",)
    metrics = [accuracy, f1, auprc, roc_auc]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        return self._make_table(
            build_event_within_table(
                db=db,
                object_type="Offer",
                event_type="O_Cancelled",
                times=timestamps,
                delta=self.timedelta,
            )
        )
