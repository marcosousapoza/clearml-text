from contextlib import contextmanager
import duckdb
import pandas as pd
from relbench.base import Database

from data.const import E2O_TABLE, EVENT_TABLE, OBJECT_TABLE


@contextmanager
def ocel_connection(db: Database, times=None):
    """Opens a DuckDB connection pre-loaded with the three core OCEL tables.

    Registers event, obj, and e2o as views so SQL queries can reference them
    directly. Optionally registers a times_df view from a timestamp array —
    useful for any query that needs to cross-join against observation times.
    Always closes the connection cleanly, even if the query fails.
    """
    con = duckdb.connect()
    try:
        con.register("event", db.table_dict[EVENT_TABLE].df)
        con.register("obj",   db.table_dict[OBJECT_TABLE].df)
        con.register("e2o",   db.table_dict[E2O_TABLE].df)
        if times is not None:
            times_df = pd.DataFrame({"obs_time": pd.to_datetime(times)})
            con.register("times_df", times_df)
        yield con
    finally:
        con.close()
