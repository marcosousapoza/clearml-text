"""Order Management task definitions.

Single-entity tasks on orders and products objects.
Multi-entity tasks on customers × products observed pairs.

Dataset stats: ~19 active orders/day, 428-day span, lifecycle median 7 days.
Look-back 14 days. num_eval_timestamps=100 yields ~2k–4k train rows per task.
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

ORDER_EVENT_TYPES = [
    "confirm order",
    "pay order",
    "payment reminder",
    "place order",
    "send invoice",
]

PRODUCT_EVENT_TYPES = [
    "confirm order",
    "pay order",
    "payment reminder",
    "place order",
    "send invoice",
]

CUSTOMER_PRODUCT_PAIR_EVENT_TYPES = [
    "confirm order",
    "pay order",
    "payment reminder",
    "place order",
    "send invoice",
]

_LOOKBACK = pd.Timedelta(days=14)
_FWD_NEXT = pd.Timedelta(days=14)
_N_TIMESTAMPS = 100


# ---------------------------------------------------------------------------
# Single-entity: orders
# ---------------------------------------------------------------------------

class OrderNextEvent(MEntityTask):
    """Next event type for an active order."""

    timedelta           = _FWD_NEXT
    num_eval_timestamps = _N_TIMESTAMPS
    task_type           = TaskType.MULTICLASS_CLASSIFICATION
    object_types        = ("orders",)
    num_classes         = len(ORDER_EVENT_TYPES)
    metrics             = [accuracy, f1, roc_auc]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_generic_next_event_table(
            db=db,
            object_type="orders",
            times=timestamps,
            event_types=ORDER_EVENT_TYPES,
            delta_back=_LOOKBACK,
            delta_fwd=self.timedelta,
        )
        return self._make_table(df)


class OrderNextTime(MEntityTask):
    """Seconds until the next event for an active order."""

    timedelta           = _FWD_NEXT
    num_eval_timestamps = _N_TIMESTAMPS
    task_type           = TaskType.REGRESSION
    object_types        = ("orders",)
    metrics             = [mae, mse, rmse, r2]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_generic_next_time_table(
            db=db,
            object_type="orders",
            times=timestamps,
            delta_back=_LOOKBACK,
        )
        return self._make_table(df)


class OrderRemainingTime(MEntityTask):
    """Days until the final event in an active order's lifecycle."""

    timedelta           = _FWD_NEXT
    num_eval_timestamps = _N_TIMESTAMPS
    task_type           = TaskType.REGRESSION
    object_types        = ("orders",)
    metrics             = [mae, mse, rmse, r2]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_generic_remaining_time_table(
            db=db,
            object_type="orders",
            times=timestamps,
            delta_back=_LOOKBACK,
        )
        return self._make_table(df)


# ---------------------------------------------------------------------------
# Single-entity: products
# ---------------------------------------------------------------------------

class ProductNextEvent(MEntityTask):
    """Next event type for an active product."""

    timedelta           = _FWD_NEXT
    num_eval_timestamps = _N_TIMESTAMPS
    task_type           = TaskType.MULTICLASS_CLASSIFICATION
    object_types        = ("products",)
    num_classes         = len(PRODUCT_EVENT_TYPES)
    metrics             = [accuracy, f1, roc_auc]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_generic_next_event_table(
            db=db,
            object_type="products",
            times=timestamps,
            event_types=PRODUCT_EVENT_TYPES,
            delta_back=_LOOKBACK,
            delta_fwd=self.timedelta,
        )
        return self._make_table(df)


class ProductNextTime(MEntityTask):
    """Seconds until the next event for an active product."""

    timedelta           = _FWD_NEXT
    num_eval_timestamps = _N_TIMESTAMPS
    task_type           = TaskType.REGRESSION
    object_types        = ("products",)
    metrics             = [mae, mse, rmse, r2]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_generic_next_time_table(
            db=db,
            object_type="products",
            times=timestamps,
            delta_back=_LOOKBACK,
        )
        return self._make_table(df)


class ProductRemainingTime(MEntityTask):
    """Days until the final event in an active product's lifecycle."""

    timedelta           = _FWD_NEXT
    num_eval_timestamps = _N_TIMESTAMPS
    task_type           = TaskType.REGRESSION
    object_types        = ("products",)
    metrics             = [mae, mse, rmse, r2]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_generic_remaining_time_table(
            db=db,
            object_type="products",
            times=timestamps,
            delta_back=_LOOKBACK,
        )
        return self._make_table(df)


# ---------------------------------------------------------------------------
# Multi-entity: customers × products observed pairs
# ---------------------------------------------------------------------------

class CustomerProductPairNextEvent(MEntityTask):
    """Next shared event type for an observed customer–product pair."""

    timedelta           = _FWD_NEXT
    num_eval_timestamps = _N_TIMESTAMPS
    task_type           = TaskType.MULTICLASS_CLASSIFICATION
    object_types        = ("customers", "products")
    entity_cols         = (O2O_SRC_COL, O2O_DST_COL)
    entity_tables       = (OBJECT_TABLE, OBJECT_TABLE)
    num_classes         = len(CUSTOMER_PRODUCT_PAIR_EVENT_TYPES)
    metrics             = [accuracy, f1, roc_auc]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_generic_pair_next_event_table(
            db=db,
            src_type="customers",
            dst_type="products",
            times=timestamps,
            event_types=CUSTOMER_PRODUCT_PAIR_EVENT_TYPES,
            delta_back=_LOOKBACK,
            delta_fwd=self.timedelta,
        )
        return self._make_table(df)


class CustomerProductPairNextTime(MEntityTask):
    """Seconds until the next shared event for an observed customer–product pair."""

    timedelta           = _FWD_NEXT
    num_eval_timestamps = _N_TIMESTAMPS
    task_type           = TaskType.REGRESSION
    object_types        = ("customers", "products")
    entity_cols         = (O2O_SRC_COL, O2O_DST_COL)
    entity_tables       = (OBJECT_TABLE, OBJECT_TABLE)
    metrics             = [mae, mse, rmse, r2]

    @check_dbs
    def make_table(self, db: Database, timestamps: Series) -> Table:
        df = build_generic_pair_next_time_table(
            db=db,
            src_type="customers",
            dst_type="products",
            times=timestamps,
            delta_back=_LOOKBACK,
        )
        return self._make_table(df)
