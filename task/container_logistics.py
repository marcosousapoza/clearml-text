import pandas as pd
from pandas import Series
from relbench.base import Database, Table, TaskType
from relbench.metrics import accuracy, auprc, f1, mae, mse, r2, rmse, roc_auc

from data.const import O2O_DST_COL, O2O_SRC_COL, OBJECT_TABLE
from data.wrapper import check_dbs
from .utils import (
    MEntityTask,
    build_observed_pair_event_within_table,
    build_stage_future_distinct_related_count_table,
    build_stage_future_event_count_table,
    build_stage_horizon_attribute_multiclass_table,
    build_stage_multiclass_next_event_table,
)

TD_STATUS_CLASSES = ["null", "in transit", "shipped"]


class ContainerLoadPhaseNextEvent4Hours(MEntityTask):
    """Multiclass branch task inside the loading phase."""

    timedelta = pd.Timedelta(hours=4)
    task_type = TaskType.MULTICLASS_CLASSIFICATION
    object_types = ("Container",)
    metrics = [accuracy, f1, roc_auc]
    num_classes = 2
    source_event_type = "Load Truck"
    next_event_types = ["Load Truck", "Drive to Terminal"]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_stage_multiclass_next_event_table(
            db=db,
            object_type="Container",
            timestamps=timestamps,
            delta=self.timedelta,
            source_event_type=self.source_event_type,
            next_event_types=self.next_event_types,
            source_max_age=self.timedelta,
        )
        return self._make_table(df)


class VehicleBookingNextEvent7Days(MEntityTask):
    """Multiclass vehicle booking progression task."""

    timedelta = pd.Timedelta(days=7)
    task_type = TaskType.MULTICLASS_CLASSIFICATION
    object_types = ("Vehicle",)
    metrics = [accuracy, f1, roc_auc]
    num_classes = 3
    source_event_type = "Book Vehicles"
    next_event_types = ["Book Vehicles", "Load to Vehicle", "Reschedule Container"]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_stage_multiclass_next_event_table(
            db=db,
            object_type="Vehicle",
            timestamps=timestamps,
            delta=self.timedelta,
            source_event_type=self.source_event_type,
            next_event_types=self.next_event_types,
            source_max_age=None,
        )
        return self._make_table(df)


class ContainerRemainingLoadTruckCount4Hours(MEntityTask):
    """How many more truck-load events will this container see soon?"""

    timedelta = pd.Timedelta(hours=4)
    task_type = TaskType.REGRESSION
    object_types = ("Container",)
    metrics = [mae, mse, rmse, r2]
    source_event_type = "Load Truck"
    source_max_age = timedelta
    target_event_type = "Load Truck"

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_stage_future_event_count_table(
            db=db,
            object_type="Container",
            timestamps=timestamps,
            delta=self.timedelta,
            source_event_type=self.source_event_type,
            target_event_type=self.target_event_type,
            source_max_age=self.source_max_age,
        )
        return self._make_table(df)


class VehicleFutureContainerLoadCount7Days(MEntityTask):
    """How many distinct containers will this booked vehicle load soon?"""

    timedelta = pd.Timedelta(days=7)
    task_type = TaskType.REGRESSION
    object_types = ("Vehicle",)
    metrics = [mae, mse, rmse, r2]
    source_event_type = "Book Vehicles"
    target_event_type = "Load to Vehicle"
    related_object_type = "Container"

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_stage_future_distinct_related_count_table(
            db=db,
            object_type="Vehicle",
            related_object_type=self.related_object_type,
            timestamps=timestamps,
            delta=self.timedelta,
            source_event_type=self.source_event_type,
            target_event_type=self.target_event_type,
            source_max_age=None,
        )
        return self._make_table(df)


class TransportDocumentFutureDepartContainerCount7Days(MEntityTask):
    """How many distinct containers will depart under this document soon?"""

    timedelta = pd.Timedelta(days=7)
    task_type = TaskType.REGRESSION
    object_types = ("Transport Document",)
    metrics = [mae, mse, rmse, r2]
    source_event_type = "Order Empty Containers"
    target_event_type = "Depart"
    related_object_type = "Container"

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_stage_future_distinct_related_count_table(
            db=db,
            object_type="Transport Document",
            related_object_type=self.related_object_type,
            timestamps=timestamps,
            delta=self.timedelta,
            source_event_type=self.source_event_type,
            target_event_type=self.target_event_type,
            source_max_age=None,
        )
        return self._make_table(df)


class TransportDocumentStatusAfterOrder7Days(MEntityTask):
    """Predict the transport document status reached by the 7-day horizon."""

    timedelta = pd.Timedelta(days=7)
    task_type = TaskType.MULTICLASS_CLASSIFICATION
    object_types = ("Transport Document",)
    metrics = [accuracy, f1, roc_auc]
    num_classes = len(TD_STATUS_CLASSES)
    source_event_type = "Order Empty Containers"

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_stage_horizon_attribute_multiclass_table(
            db=db,
            object_type="Transport Document",
            attribute_table_name="object_attr_Transport Document",
            attribute_col="Status",
            class_values=TD_STATUS_CLASSES,
            timestamps=timestamps,
            source_event_type=self.source_event_type,
            source_max_age=None,
        )
        return self._make_table(df)


class TransportDocumentVehicleDepartWithin7Days(MEntityTask):
    """Observed TD x Vehicle pair: will it co-depart within 7 days?"""

    timedelta = pd.Timedelta(days=7)
    source_max_age = pd.Timedelta(weeks=4)
    task_type = TaskType.BINARY_CLASSIFICATION
    object_types = ("Transport Document", "Vehicle")
    entity_cols = (O2O_SRC_COL, O2O_DST_COL)
    entity_tables = (OBJECT_TABLE, OBJECT_TABLE)
    metrics = [accuracy, f1, auprc, roc_auc]
    source_event_type = "Book Vehicles"
    target_event_type = "Depart"

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_observed_pair_event_within_table(
            db=db,
            object_types=self.object_types, # type: ignore
            timestamps=timestamps,
            delta=self.timedelta,
            source_event_type=self.source_event_type,
            target_event_type=self.target_event_type,
            source_max_age=self.source_max_age,
        )
        return self._make_table(df)


class TransportDocumentContainerDepartWithin7Days(MEntityTask):
    """Observed TD x Container pair: will it co-depart within 7 days?"""

    timedelta = pd.Timedelta(days=7)
    source_max_age = pd.Timedelta(weeks=4)
    task_type = TaskType.BINARY_CLASSIFICATION
    object_types = ("Transport Document", "Container")
    entity_cols = (O2O_SRC_COL, O2O_DST_COL)
    entity_tables = (OBJECT_TABLE, OBJECT_TABLE)
    metrics = [accuracy, f1, auprc, roc_auc]
    source_event_type = "Order Empty Containers"
    target_event_type = "Depart"

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_observed_pair_event_within_table(
            db=db,
            object_types=self.object_types, # type: ignore
            timestamps=timestamps,
            delta=self.timedelta,
            source_event_type=self.source_event_type,
            target_event_type=self.target_event_type,
            source_max_age=self.source_max_age,
        )
        return self._make_table(df)
