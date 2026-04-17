import pandas as pd
from pandas import Series
from relbench.base import Database, Table, TaskType
from relbench.metrics import accuracy, f1, mae, mse, r2, rmse, roc_auc

from data.wrapper import check_dbs
from .utils import (
    Log1pZScoreTargetTransform,
    MEntityTask,
    build_stage_future_event_count_table,
    build_stage_multiclass_next_event_table,
)


APPLICATION_VALIDATION_OUTCOMES = [
    "A_Incomplete",
    "A_Pending",
    "A_Denied",
]

APPLICATION_INCOMPLETE_OUTCOMES = [
    "A_Validating",
    "A_Pending",
    "A_Cancelled",
]

OFFER_SENT_OUTCOMES = [
    "O_Returned",
    "O_Cancelled",
    "O_Refused",
]

OFFER_RETURNED_OUTCOMES = [
    "O_Accepted",
    "O_Cancelled",
    "O_Refused",
]


class ApplicationValidationOutcome14Days(MEntityTask):
    """Branch after validation: pending, incomplete, denied, or cancelled."""

    timedelta = pd.Timedelta(days=14)
    task_type = TaskType.MULTICLASS_CLASSIFICATION
    object_types = ("Application",)
    metrics = [accuracy, f1, roc_auc]
    num_classes = len(APPLICATION_VALIDATION_OUTCOMES)
    source_event_type = "A_Validating"
    next_event_types = APPLICATION_VALIDATION_OUTCOMES
    source_max_age = timedelta

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_stage_multiclass_next_event_table(
            db=db,
            object_type="Application",
            timestamps=timestamps,
            delta=self.timedelta,
            source_event_type=self.source_event_type,
            next_event_types=self.next_event_types,
            source_max_age=self.source_max_age,
        )
        return self._make_table(df)


class ApplicationIncompleteOutcome19Days(MEntityTask):
    """What happens after an incomplete-file state in the next 19 days?"""

    timedelta = pd.Timedelta(days=19)
    task_type = TaskType.MULTICLASS_CLASSIFICATION
    object_types = ("Application",)
    metrics = [accuracy, f1, roc_auc]
    num_classes = len(APPLICATION_INCOMPLETE_OUTCOMES)
    source_event_type = "A_Incomplete"
    next_event_types = APPLICATION_INCOMPLETE_OUTCOMES
    source_max_age = timedelta

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_stage_multiclass_next_event_table(
            db=db,
            object_type="Application",
            timestamps=timestamps,
            delta=self.timedelta,
            source_event_type=self.source_event_type,
            next_event_types=self.next_event_types,
            source_max_age=self.source_max_age,
        )
        return self._make_table(df)


class ApplicationFutureValidationCount19Days(MEntityTask):
    """How many more validation passes will this incomplete application need soon?"""

    timedelta = pd.Timedelta(days=19)
    task_type = TaskType.REGRESSION
    object_types = ("Application",)
    metrics = [mae, mse, rmse, r2]
    source_event_type = "A_Incomplete"
    target_event_type = "A_Validating"
    source_max_age = timedelta

    def make_target_transform(self):
        return Log1pZScoreTargetTransform()

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_stage_future_event_count_table(
            db=db,
            object_type="Application",
            timestamps=timestamps,
            delta=self.timedelta,
            source_event_type=self.source_event_type,
            target_event_type=self.target_event_type,
            source_max_age=self.source_max_age,
        )
        return self._make_table(df)


class OfferSentOutcome19Days(MEntityTask):
    """After an offer is sent, does it return, cancel, or get refused?"""

    timedelta = pd.Timedelta(days=19)
    task_type = TaskType.MULTICLASS_CLASSIFICATION
    object_types = ("Offer",)
    metrics = [accuracy, f1, roc_auc]
    num_classes = len(OFFER_SENT_OUTCOMES)
    source_event_type = "O_Sent (mail and online)"
    next_event_types = OFFER_SENT_OUTCOMES
    source_max_age = timedelta

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_stage_multiclass_next_event_table(
            db=db,
            object_type="Offer",
            timestamps=timestamps,
            delta=self.timedelta,
            source_event_type=self.source_event_type,
            next_event_types=self.next_event_types,
            source_max_age=self.source_max_age,
        )
        return self._make_table(df)


class OfferReturnedOutcome19Days(MEntityTask):
    """Customer response after an offer is returned: accept, cancel, or refuse."""

    timedelta = pd.Timedelta(days=19)
    task_type = TaskType.MULTICLASS_CLASSIFICATION
    object_types = ("Offer",)
    metrics = [accuracy, f1, roc_auc]
    num_classes = len(OFFER_RETURNED_OUTCOMES)
    source_event_type = "O_Returned"
    next_event_types = OFFER_RETURNED_OUTCOMES
    source_max_age = timedelta

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_stage_multiclass_next_event_table(
            db=db,
            object_type="Offer",
            timestamps=timestamps,
            delta=self.timedelta,
            source_event_type=self.source_event_type,
            next_event_types=self.next_event_types,
            source_max_age=self.source_max_age,
        )
        return self._make_table(df)
