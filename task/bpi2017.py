"""BPI2017 task definitions.

Single-entity tasks on Application and Offer objects.
"""

import pandas as pd
from pandas import Series
from relbench.base import Database, Table, TaskType
from relbench.metrics import accuracy, f1, mae, mse, r2, rmse, roc_auc

from data.wrapper import check_dbs
from .utils import (
    MEntityTask,
    build_generic_next_event_table,
    build_generic_next_time_table,
    build_generic_remaining_time_table,
)

# All event types in BPI2017 that involve Application objects.
APPLICATION_EVENT_TYPES = [
    "A_Accepted",
    "A_Cancelled",
    "A_Create Application",
    "A_Denied",
    "A_Incomplete",
    "A_Pending",
    "A_Validating",
    "W_Assess potential fraud",
    "W_Call after offers",
    "W_Call incomplete files",
    "W_Complete application",
    "W_Handle leads",
    "W_Validate application",
]

# Event types for Offer objects.
OFFER_EVENT_TYPES = [
    "O_Accepted",
    "O_Cancelled",
    "O_Create Offer",
    "O_Refused",
    "O_Returned",
    "O_Sent (mail and online)",
    "O_Sent (online only)",
]

_LOOKBACK = pd.Timedelta(days=7)
_FWD_NEXT = pd.Timedelta(days=7, hours=8)
_N_TIMESTAMPS = 40


# ---------------------------------------------------------------------------
# Single-entity: Application
# ---------------------------------------------------------------------------

class ApplicationNextEvent(MEntityTask):
    """Next event type for an active loan application."""

    timedelta           = _FWD_NEXT
    num_eval_timestamps = _N_TIMESTAMPS
    task_type           = TaskType.MULTICLASS_CLASSIFICATION
    object_types        = ("Application",)
    num_classes         = len(APPLICATION_EVENT_TYPES)
    metrics             = [accuracy, f1, roc_auc]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_generic_next_event_table(
            db=db,
            object_type="Application",
            times=timestamps,
            event_types=APPLICATION_EVENT_TYPES,
            delta_back=_LOOKBACK,
            delta_fwd=self.timedelta,
        )
        return self._make_table(df)


class ApplicationNextTime(MEntityTask):
    """Seconds until the next event for an active loan application."""

    timedelta           = _FWD_NEXT
    num_eval_timestamps = _N_TIMESTAMPS
    task_type           = TaskType.REGRESSION
    object_types        = ("Application",)
    metrics             = [mae, mse, rmse, r2]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_generic_next_time_table(
            db=db,
            object_type="Application",
            times=timestamps,
            delta_back=_LOOKBACK,
        )
        return self._make_table(df)


class ApplicationRemainingTime(MEntityTask):
    """Days until the final event in an active loan application's lifecycle."""

    timedelta           = _FWD_NEXT
    num_eval_timestamps = _N_TIMESTAMPS
    task_type           = TaskType.REGRESSION
    object_types        = ("Application",)
    metrics             = [mae, mse, rmse, r2]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_generic_remaining_time_table(
            db=db,
            object_type="Application",
            times=timestamps,
            delta_back=_LOOKBACK,
        )
        return self._make_table(df)


# ---------------------------------------------------------------------------
# Single-entity: Offer
# ---------------------------------------------------------------------------

class OfferNextEvent(MEntityTask):
    """Next event type for an active loan offer."""

    timedelta           = _FWD_NEXT
    num_eval_timestamps = _N_TIMESTAMPS
    task_type           = TaskType.MULTICLASS_CLASSIFICATION
    object_types        = ("Offer",)
    num_classes         = len(OFFER_EVENT_TYPES)
    metrics             = [accuracy, f1, roc_auc]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_generic_next_event_table(
            db=db,
            object_type="Offer",
            times=timestamps,
            event_types=OFFER_EVENT_TYPES,
            delta_back=_LOOKBACK,
            delta_fwd=self.timedelta,
        )
        return self._make_table(df)


class OfferNextTime(MEntityTask):
    """Seconds until the next event for an active loan offer."""

    timedelta           = _FWD_NEXT
    num_eval_timestamps = _N_TIMESTAMPS
    task_type           = TaskType.REGRESSION
    object_types        = ("Offer",)
    metrics             = [mae, mse, rmse, r2]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_generic_next_time_table(
            db=db,
            object_type="Offer",
            times=timestamps,
            delta_back=_LOOKBACK,
        )
        return self._make_table(df)


class OfferRemainingTime(MEntityTask):
    """Days until the final event in an active loan offer's lifecycle."""

    timedelta           = _FWD_NEXT
    num_eval_timestamps = _N_TIMESTAMPS
    task_type           = TaskType.REGRESSION
    object_types        = ("Offer",)
    metrics             = [mae, mse, rmse, r2]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_generic_remaining_time_table(
            db=db,
            object_type="Offer",
            times=timestamps,
            delta_back=_LOOKBACK,
        )
        return self._make_table(df)
