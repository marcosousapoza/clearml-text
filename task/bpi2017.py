"""Predictive tasks for the bpi2017 dataset.

8 tasks across 4 label families:
  next-event classification  (2 tasks)
  next-time regression       (2 tasks)
  remaining-time regression  (2 tasks)
  binary event-within-window (1 task)
  multi-entity pair binary   (1 task)

Object types in this log
------------------------
  Application, Case_R, Offer, Workflow

Event types (26)
----------------
  Application events (A_):
    A_Accepted, A_Cancelled, A_Complete, A_Concept, A_Create Application,
    A_Denied, A_Incomplete, A_Pending, A_Submitted, A_Validating

  Offer events (O_):
    O_Accepted, O_Cancelled, O_Create Offer, O_Created, O_Refused,
    O_Returned, O_Sent (mail and online), O_Sent (online only)

  Workflow events (W_):
    W_Assess potential fraud, W_Call after offers, W_Call incomplete files,
    W_Complete application, W_Handle leads, W_Personal Loan collection,
    W_Shortened completion, W_Validate application

Dataset splits
--------------
  val_timestamp  = 2016-09-25 12:00:00
  test_timestamp = 2016-11-14 12:00:00
"""
import pandas as pd
from relbench.base import Database, Table, TaskType
from relbench.metrics import accuracy, f1, mae, mse, r2, rmse

from data.const import OBJECT_ID_COL, OBJECT_TABLE
from task.metrics import roc_auc
from task.utils import MEntityTask
from task.utils.builders import (
    build_event_within_table,
    build_next_event_table,
    build_next_time_table,
    build_pair_interaction_table,
    build_remaining_time_table,
    to_relbench_table,
)

# ---------------------------------------------------------------------------
# Domain vocabulary
# ---------------------------------------------------------------------------

_APPLICATION_EVENTS = [
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

_OFFER_EVENTS = [
    "O_Accepted",
    "O_Cancelled",
    "O_Create Offer",
    "O_Created",
    "O_Refused",
    "O_Returned",
    "O_Sent (mail and online)",
    "O_Sent (online only)",
]

# Negative-outcome events used for the binary within-window task
_APPLICATION_DENIAL_EVENTS = ["A_Denied", "A_Cancelled"]

# Events where a Case_R and Offer co-appear (used for pair task). BPI2017 does
# not directly attach Application and Offer objects to the same events; offers
# co-occur with their Case_R object instead.
_CASE_OFFER_EVENTS = [
    "O_Create Offer",
    "O_Created",
    "O_Sent (mail and online)",
    "O_Sent (online only)",
    "O_Accepted",
    "O_Refused",
    "O_Cancelled",
    "O_Returned",
]


# ---------------------------------------------------------------------------
# Shared window presets
# ---------------------------------------------------------------------------

_BACK_30 = pd.Timedelta(days=30)
_FWD_14  = pd.Timedelta(days=14)
_FWD_30  = pd.Timedelta(days=30)
_DELTA   = pd.Timedelta(days=7)
_PAIR_COL = "object_id_partner"


# ---------------------------------------------------------------------------
# Task 1 — Application: next event classification
#
# Business meaning : After recent application activity, which of the 10 A_
#                    lifecycle events occurs next for this application?
# Signal           : Loan applications progress through a structured intake
#                    pipeline (Create → Submitted → Validating → Accepted/
#                    Denied), making the current stage highly predictive of
#                    the next event.
# ---------------------------------------------------------------------------

class ApplicationNextEvent(MEntityTask):
    """Next A_ lifecycle event for an active application (10-class)."""

    timedelta    = _DELTA
    task_type    = TaskType.MULTICLASS_CLASSIFICATION
    object_types = ("Application",)
    num_classes  = len(_APPLICATION_EVENTS)
    metrics      = [accuracy, f1, roc_auc]

    def make_table(self, db: Database, timestamps: pd.Series) -> Table:
        df = build_next_event_table(
            db,
            object_type = "Application",
            times       = timestamps,
            event_types = _APPLICATION_EVENTS,
            delta_back  = _BACK_30,
            delta_fwd   = _FWD_14,
        )
        return to_relbench_table(df, self.entity_cols, self.entity_tables)


# ---------------------------------------------------------------------------
# Task 2 — Offer: next event classification
#
# Business meaning : After recent offer activity, which of the 8 O_ events
#                    occurs next for this offer?
# Signal           : Offer lifecycle follows a clear path (Create Offer →
#                    Created → Sent → Accepted/Refused/Cancelled), so the
#                    most recent offer event strongly predicts the next one.
# ---------------------------------------------------------------------------

class OfferNextEvent(MEntityTask):
    """Next O_ lifecycle event for an active offer (8-class)."""

    timedelta    = _DELTA
    task_type    = TaskType.MULTICLASS_CLASSIFICATION
    object_types = ("Offer",)
    num_classes  = len(_OFFER_EVENTS)
    metrics      = [accuracy, f1, roc_auc]

    def make_table(self, db: Database, timestamps: pd.Series) -> Table:
        df = build_next_event_table(
            db,
            object_type = "Offer",
            times       = timestamps,
            event_types = _OFFER_EVENTS,
            delta_back  = _BACK_30,
            delta_fwd   = _FWD_14,
        )
        return to_relbench_table(df, self.entity_cols, self.entity_tables)


# ---------------------------------------------------------------------------
# Task 3 — Application: next-event time regression
#
# Business meaning : How many days until the next activity for this
#                    application?  Useful for SLA tracking and proactive
#                    follow-up by loan officers.
# Signal           : Applications in early intake (Concept, Submitted) move
#                    quickly; those stuck in validation or pending states
#                    experience longer gaps before the next event.
# ---------------------------------------------------------------------------

class ApplicationNextTime(MEntityTask):
    """Days until the next event for an active application."""

    timedelta    = _DELTA
    task_type    = TaskType.REGRESSION
    object_types = ("Application",)
    metrics      = [mae, rmse, mse, r2]

    def make_table(self, db: Database, timestamps: pd.Series) -> Table:
        df = build_next_time_table(
            db,
            object_type = "Application",
            times       = timestamps,
            delta_back  = _BACK_30,
            delta_fwd   = _FWD_30,
        )
        return to_relbench_table(df, self.entity_cols, self.entity_tables)


# ---------------------------------------------------------------------------
# Task 4 — Case_R: next-event time regression
#
# Business meaning : How many days until the next activity for this credit
#                    case?  Helps case managers prioritise workloads and set
#                    customer expectations on case progression.
# Signal           : Cases that have already received offers tend to advance
#                    faster than those still in initial review or on hold.
# ---------------------------------------------------------------------------

class CaseNextTime(MEntityTask):
    """Days until the next event for an active credit case."""

    timedelta    = _DELTA
    task_type    = TaskType.REGRESSION
    object_types = ("Case_R",)
    metrics      = [mae, rmse, mse, r2]

    def make_table(self, db: Database, timestamps: pd.Series) -> Table:
        df = build_next_time_table(
            db,
            object_type = "Case_R",
            times       = timestamps,
            delta_back  = _BACK_30,
            delta_fwd   = _FWD_30,
        )
        return to_relbench_table(df, self.entity_cols, self.entity_tables)


# ---------------------------------------------------------------------------
# Task 5 — Application: remaining time regression
#
# Business meaning : How many days until the final event (closure) for this
#                    application?  Classic end-to-end cycle-time prediction
#                    for loan processing SLA monitoring.
# Signal           : Applications near acceptance or denial have few remaining
#                    steps; those still in early intake have a long path ahead.
# ---------------------------------------------------------------------------

class ApplicationRemainingTime(MEntityTask):
    """Days until the final event (closure) for an active application."""

    timedelta    = _DELTA
    task_type    = TaskType.REGRESSION
    object_types = ("Application",)
    metrics      = [mae, rmse, mse, r2]

    def make_table(self, db: Database, timestamps: pd.Series) -> Table:
        df = build_remaining_time_table(
            db,
            object_type = "Application",
            times       = timestamps,
            delta_back  = _BACK_30,
        )
        return to_relbench_table(df, self.entity_cols, self.entity_tables)


# ---------------------------------------------------------------------------
# Task 6 — Offer: remaining time regression
#
# Business meaning : How many days until the final event for this offer?
#                    Useful for estimating how long an outstanding offer will
#                    remain active before being accepted, refused, or cancelled.
# Signal           : Offers already sent and awaiting response have a shorter
#                    remaining lifetime than freshly created offers.
# ---------------------------------------------------------------------------

class OfferRemainingTime(MEntityTask):
    """Days until the final event for an active offer."""

    timedelta    = _DELTA
    task_type    = TaskType.REGRESSION
    object_types = ("Offer",)
    metrics      = [mae, rmse, mse, r2]

    def make_table(self, db: Database, timestamps: pd.Series) -> Table:
        df = build_remaining_time_table(
            db,
            object_type = "Offer",
            times       = timestamps,
            delta_back  = _BACK_30,
        )
        return to_relbench_table(df, self.entity_cols, self.entity_tables)


# ---------------------------------------------------------------------------
# Task 7 — Application: will it be denied or cancelled within 14 days?
#
# Business meaning : Is this application at risk of denial or cancellation in
#                    the next two weeks?  Early identification allows officers
#                    to intervene or redirect the applicant proactively.
# Signal           : Applications that have already received an A_Incomplete
#                    or A_Pending event, or that have had multiple workflow
#                    steps without resolution, are at higher risk.
# ---------------------------------------------------------------------------

class ApplicationDeniedWithin14d(MEntityTask):
    """Binary: application denied or cancelled within 14 days of observation."""

    timedelta    = _DELTA
    task_type    = TaskType.BINARY_CLASSIFICATION
    object_types = ("Application",)
    metrics      = [accuracy, f1, roc_auc]

    def make_table(self, db: Database, timestamps: pd.Series) -> Table:
        df = build_event_within_table(
            db,
            object_type        = "Application",
            times              = timestamps,
            target_event_types = _APPLICATION_DENIAL_EVENTS,
            delta_back         = _BACK_30,
            delta_fwd          = _FWD_14,
        )
        return to_relbench_table(df, self.entity_cols, self.entity_tables)


# ---------------------------------------------------------------------------
# Task 8 — Case_R × Offer pair: will they co-appear within 14 days?
#
# Business meaning : Given a case and an offer observed together
#                    recently, will they co-appear in a future offer-lifecycle
#                    event within the next two weeks?  Useful for predicting
#                    which cases will generate renewed offer activity.
# Entity columns   : object_id (Case_R), object_id_partner (Offer)
# Signal           : Cases that have recently received or returned an
#                    offer are more likely to generate further offer activity;
#                    historical co-occurrence is a strong predictor.
# ---------------------------------------------------------------------------

class ApplicationOfferPairInteraction(MEntityTask):
    """Binary: case-offer pair co-appears in a future offer event.

    The registered task name remains ``application_offer_pair`` for backwards
    compatibility with existing experiment scripts.
    """

    timedelta      = _DELTA
    task_type      = TaskType.BINARY_CLASSIFICATION
    entity_cols    = (OBJECT_ID_COL, _PAIR_COL)
    entity_tables  = (OBJECT_TABLE, OBJECT_TABLE)
    object_types   = ("Case_R", "Offer")
    metrics        = [accuracy, f1, roc_auc]

    def make_table(self, db: Database, timestamps: pd.Series) -> Table:
        df = build_pair_interaction_table(
            db,
            src_type                = "Case_R",
            dst_type                = "Offer",
            times                   = timestamps,
            interaction_event_types = _CASE_OFFER_EVENTS,
            delta_back              = _BACK_30,
            delta_fwd               = _FWD_14,
            pair_col                = _PAIR_COL,
            max_negatives_per_positive = 10,
        )
        return to_relbench_table(df, self.entity_cols, self.entity_tables)
