import pandas as pd
from pandas import Series
from relbench.base import Database, Table, TaskType
from relbench.metrics import accuracy, auprc, f1, mae, mse, r2, rmse, roc_auc

from data.const import O2O_DST_COL, O2O_SRC_COL, OBJECT_TABLE
from data.wrapper import check_dbs
from .utils import (
    MEntityTask,
    build_observed_pair_event_within_table,
    build_observed_pair_future_event_count_table,
    build_stage_future_event_count_table,
    build_stage_multiclass_next_event_table,
)

ORDER_PAYMENT_OUTCOMES = ["pay order", "payment reminder"]


class OrderPaymentOutcome14Days(MEntityTask):
    """After confirmation, does the order get paid or need a reminder next?"""

    timedelta = pd.Timedelta(days=14)
    task_type = TaskType.MULTICLASS_CLASSIFICATION
    object_types = ("orders",)
    metrics = [accuracy, f1, roc_auc]
    num_classes = len(ORDER_PAYMENT_OUTCOMES)
    source_event_type = "confirm order"
    next_event_types = ORDER_PAYMENT_OUTCOMES
    source_max_age = timedelta

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_stage_multiclass_next_event_table(
            db=db,
            object_type="orders",
            timestamps=timestamps,
            delta=self.timedelta,
            source_event_type=self.source_event_type,
            next_event_types=self.next_event_types,
            source_max_age=self.source_max_age,
        )
        return self._make_table(df)


class OrderFutureReminderCount14Days(MEntityTask):
    """How many payment reminders remain for a confirmed order over 14 days?"""

    timedelta = pd.Timedelta(days=14)
    task_type = TaskType.REGRESSION
    object_types = ("orders",)
    metrics = [mae, mse, rmse, r2]
    source_event_type = "confirm order"
    target_event_type = "payment reminder"
    source_max_age = timedelta

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_stage_future_event_count_table(
            db=db,
            object_type="orders",
            timestamps=timestamps,
            delta=self.timedelta,
            source_event_type=self.source_event_type,
            target_event_type=self.target_event_type,
            source_max_age=self.source_max_age,
        )
        return self._make_table(df)


class CustomerProductRepeatOrderWithin7Days(MEntityTask):
    """Observed customer-product pair: will it place another order within 7 days?"""

    timedelta = pd.Timedelta(days=7)
    source_max_age = pd.Timedelta(weeks=4)
    task_type = TaskType.BINARY_CLASSIFICATION
    object_types = ("customers", "products")
    entity_cols = (O2O_SRC_COL, O2O_DST_COL)
    entity_tables = (OBJECT_TABLE, OBJECT_TABLE)
    metrics = [accuracy, f1, auprc, roc_auc]
    source_event_type = "confirm order"
    target_event_type = "place order"

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_observed_pair_event_within_table(
            db=db,
            object_types=self.object_types,  # type: ignore[arg-type]
            timestamps=timestamps,
            delta=self.timedelta,
            source_event_type=self.source_event_type,
            target_event_type=self.target_event_type,
            source_max_age=self.source_max_age,
        )
        return self._make_table(df)


class CustomerProductFutureOrderCount14Days(MEntityTask):
    """Observed customer-product pair: how many repeat orders arrive in 14 days?"""

    timedelta = pd.Timedelta(days=14)
    source_max_age = pd.Timedelta(weeks=4)
    task_type = TaskType.REGRESSION
    object_types = ("customers", "products")
    entity_cols = (O2O_SRC_COL, O2O_DST_COL)
    entity_tables = (OBJECT_TABLE, OBJECT_TABLE)
    metrics = [mae, mse, rmse, r2]
    source_event_type = "confirm order"
    target_event_type = "place order"

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_observed_pair_future_event_count_table(
            db=db,
            object_types=self.object_types,  # type: ignore[arg-type]
            timestamps=timestamps,
            delta=self.timedelta,
            source_event_type=self.source_event_type,
            target_event_type=self.target_event_type,
            source_max_age=self.source_max_age,
        )
        return self._make_table(df)
