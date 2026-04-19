"""Container Logistics task definitions.

Single-entity tasks on Container and Transport Document objects.
Multi-entity tasks on Container × Transport Document observed pairs.

Dataset stats: ~23 active Containers/day, 457-day span, lifecycle median 14 days.
Look-back 3 days (Container), 14 days (Transport Document).
num_eval_timestamps=100 yields ~2k–5k train rows per task.
"""

import pandas as pd
from pandas import Series
from relbench.base import Database, Table, TaskType
from relbench.metrics import accuracy, f1, mae, mse, r2, rmse, roc_auc

from data.const import O2O_DST_COL, O2O_SRC_COL, OBJECT_TABLE
from data.wrapper import check_dbs
from .utils import (
    MEntityTask,
    build_generic_next_event_table,
    build_generic_next_time_table,
    build_generic_remaining_time_table,
    build_generic_pair_next_event_table,
    build_generic_pair_next_time_table,
)

CONTAINER_EVENT_TYPES = [
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

TRANSPORT_DOC_EVENT_TYPES = [
    "Book Vehicles",
    "Depart",
    "Load to Vehicle",
    "Order Empty Containers",
    "Reschedule Container",
]

CONTAINER_TD_PAIR_EVENT_TYPES = [
    "Depart",
    "Load to Vehicle",
    "Order Empty Containers",
    "Reschedule Container",
]

_N_TIMESTAMPS = 100


# ---------------------------------------------------------------------------
# Single-entity: Container
# ---------------------------------------------------------------------------

class ContainerNextEvent(MEntityTask):
    """Next event type for an active container."""

    timedelta           = pd.Timedelta(hours=8)
    num_eval_timestamps = _N_TIMESTAMPS
    task_type           = TaskType.MULTICLASS_CLASSIFICATION
    object_types        = ("Container",)
    num_classes         = len(CONTAINER_EVENT_TYPES)
    metrics             = [accuracy, f1, roc_auc]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_generic_next_event_table(
            db=db,
            object_type="Container",
            times=timestamps,
            event_types=CONTAINER_EVENT_TYPES,
            delta_back=pd.Timedelta(days=3),
            delta_fwd=self.timedelta,
        )
        return self._make_table(df)


class ContainerNextTime(MEntityTask):
    """Days until the next event for an active container."""

    timedelta           = pd.Timedelta(hours=8)
    num_eval_timestamps = _N_TIMESTAMPS
    task_type           = TaskType.REGRESSION
    object_types        = ("Container",)
    metrics             = [mae, mse, rmse, r2]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_generic_next_time_table(
            db=db,
            object_type="Container",
            times=timestamps,
            delta_back=pd.Timedelta(days=3),
        )
        return self._make_table(df)


class ContainerRemainingTime(MEntityTask):
    """Weeks until the final event in an active container's lifecycle."""

    timedelta           = pd.Timedelta(hours=8)
    num_eval_timestamps = _N_TIMESTAMPS
    task_type           = TaskType.REGRESSION
    object_types        = ("Container",)
    metrics             = [mae, mse, rmse, r2]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_generic_remaining_time_table(
            db=db,
            object_type="Container",
            times=timestamps,
            delta_back=pd.Timedelta(days=3),
        )
        return self._make_table(df)


# ---------------------------------------------------------------------------
# Single-entity: Transport Document
# ---------------------------------------------------------------------------

class TransportDocumentNextEvent(MEntityTask):
    """Next event type for an active transport document."""

    timedelta           = pd.Timedelta(days=1, hours=8)
    num_eval_timestamps = _N_TIMESTAMPS
    task_type           = TaskType.MULTICLASS_CLASSIFICATION
    object_types        = ("Transport Document",)
    num_classes         = len(TRANSPORT_DOC_EVENT_TYPES)
    metrics             = [accuracy, f1, roc_auc]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_generic_next_event_table(
            db=db,
            object_type="Transport Document",
            times=timestamps,
            event_types=TRANSPORT_DOC_EVENT_TYPES,
            delta_back=pd.Timedelta(days=14),
            delta_fwd=self.timedelta,
        )
        return self._make_table(df)


class TransportDocumentNextTime(MEntityTask):
    """Days until the next event for an active transport document."""

    timedelta           = pd.Timedelta(days=1, hours=1)
    num_eval_timestamps = _N_TIMESTAMPS
    task_type           = TaskType.REGRESSION
    object_types        = ("Transport Document",)
    metrics             = [mae, mse, rmse, r2]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_generic_next_time_table(
            db=db,
            object_type="Transport Document",
            times=timestamps,
            delta_back=pd.Timedelta(days=14),
        )
        return self._make_table(df)


class TransportDocumentRemainingTime(MEntityTask):
    """Weeks until the final event in an active transport document's lifecycle."""

    timedelta           = pd.Timedelta(days=1, hours=8)
    num_eval_timestamps = _N_TIMESTAMPS
    task_type           = TaskType.REGRESSION
    object_types        = ("Transport Document",)
    metrics             = [mae, mse, rmse, r2]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_generic_remaining_time_table(
            db=db,
            object_type="Transport Document",
            times=timestamps,
            delta_back=pd.Timedelta(days=14),
        )
        return self._make_table(df)


# ---------------------------------------------------------------------------
# Multi-entity: Container × Transport Document observed pairs
# ---------------------------------------------------------------------------

class ContainerTDPairNextEvent(MEntityTask):
    """Next shared event type for an observed Container–Transport Document pair."""

    timedelta           = pd.Timedelta(days=1, hours=8)
    num_eval_timestamps = _N_TIMESTAMPS
    task_type           = TaskType.MULTICLASS_CLASSIFICATION
    object_types        = ("Container", "Transport Document")
    entity_cols         = (O2O_SRC_COL, O2O_DST_COL)
    entity_tables       = (OBJECT_TABLE, OBJECT_TABLE)
    num_classes         = len(CONTAINER_TD_PAIR_EVENT_TYPES)
    metrics             = [accuracy, f1, roc_auc]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_generic_pair_next_event_table(
            db=db,
            src_type="Container",
            dst_type="Transport Document",
            times=timestamps,
            event_types=CONTAINER_TD_PAIR_EVENT_TYPES,
            delta_back=pd.Timedelta(days=14),
            delta_fwd=self.timedelta,
        )
        return self._make_table(df)


class ContainerTDPairNextTime(MEntityTask):
    """Days until the next shared event for an observed Container–Transport Document pair."""

    timedelta           = pd.Timedelta(days=1, hours=8)
    num_eval_timestamps = _N_TIMESTAMPS
    task_type           = TaskType.REGRESSION
    object_types        = ("Container", "Transport Document")
    entity_cols         = (O2O_SRC_COL, O2O_DST_COL)
    entity_tables       = (OBJECT_TABLE, OBJECT_TABLE)
    metrics             = [mae, mse, rmse, r2]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_generic_pair_next_time_table(
            db=db,
            src_type="Container",
            dst_type="Transport Document",
            times=timestamps,
            delta_back=pd.Timedelta(days=14),
        )
        return self._make_table(df)
