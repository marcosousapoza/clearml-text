"""BPI2017 task definitions.

Single-entity tasks on Application and Offer objects.
Multi-entity task on Application × Offer observed pairs.

Look-back: 14 days (dataset spans ~1 year; many applications go quiet for weeks).
Temporal sampling: RelBench default (num_eval_timestamps=1000 covers all splits).
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

# Shared events that can link an Application and Offer in the same event.
APP_OFFER_PAIR_EVENT_TYPES = [
    "O_Accepted",
    "O_Cancelled",
    "O_Create Offer",
    "O_Refused",
    "O_Returned",
    "O_Sent (mail and online)",
    "O_Sent (online only)",
]

_LOOKBACK = pd.Timedelta(days=7)
_FWD_NEXT = pd.Timedelta(days=7)
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


# ---------------------------------------------------------------------------
# Multi-entity: Application × Offer observed pairs
# ---------------------------------------------------------------------------

class ApplicationOfferPairNextEvent(MEntityTask):
    """Next shared event type for an observed Application–Offer pair."""

    timedelta           = _FWD_NEXT
    num_eval_timestamps = _N_TIMESTAMPS
    task_type           = TaskType.MULTICLASS_CLASSIFICATION
    object_types        = ("Application", "Offer")
    entity_cols         = (O2O_SRC_COL, O2O_DST_COL)
    entity_tables       = (OBJECT_TABLE, OBJECT_TABLE)
    num_classes         = len(APP_OFFER_PAIR_EVENT_TYPES)
    metrics             = [accuracy, f1, roc_auc]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_generic_pair_next_event_table(
            db=db,
            src_type="Application",
            dst_type="Offer",
            times=timestamps,
            event_types=APP_OFFER_PAIR_EVENT_TYPES,
            delta_back=_LOOKBACK,
            delta_fwd=self.timedelta,
        )
        return self._make_table(df)


class ApplicationOfferPairNextTime(MEntityTask):
    """Seconds until the next shared event for an observed Application–Offer pair."""

    timedelta           = _FWD_NEXT
    num_eval_timestamps = _N_TIMESTAMPS
    task_type           = TaskType.REGRESSION
    object_types        = ("Application", "Offer")
    entity_cols         = (O2O_SRC_COL, O2O_DST_COL)
    entity_tables       = (OBJECT_TABLE, OBJECT_TABLE)
    metrics             = [mae, mse, rmse, r2]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_generic_pair_next_time_table(
            db=db,
            src_type="Application",
            dst_type="Offer",
            times=timestamps,
            delta_back=_LOOKBACK,
        )
        return self._make_table(df)
