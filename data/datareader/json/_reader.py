"""
JSON data reader that auto-detects OCEL version and produces EventGraph.

Produces the *full* typed OCEL graph (e_*/o_*), including qualifiers as part of
the edge relation name (e2o_<qualifier>).
"""

import logging

from relbench.base import Database, Table
from relbench.utils import clean_datetime

from data.flat import flatten
from data.datareader.relbench_tables import apply_default_column_dtypes_to_df

from .utils import detect_ocel_version
from ..base import BaseDataReader
from ._ocel1 import _JSONDataReader1
from ._ocel2 import _JSONDataReader2

logger = logging.getLogger("DATA")


class JSONDataReader(BaseDataReader):
    """
    JSON data reader that auto-detects OCEL version and produces EventGraph.

    Supports both OCEL 1.0 and OCEL 2.0 JSON formats.
    """

    def parse_tables(
        self,
        dataset_name: str | None = None,
    ) -> Database:
        """
        Parse JSON file and return standardized tables as pandas DataFrames.
        """
        ocel_version = detect_ocel_version(self.path)
        logger.info("Detected OCEL version=%s for path=%s", ocel_version, self.path)
        if ocel_version.startswith("1"):
            self._internal_reader = _JSONDataReader1(self.path)
        elif ocel_version.startswith("2"):
            self._internal_reader = _JSONDataReader2(self.path)
        else:
            raise ValueError(f"Unsupported OCEL version: {ocel_version}")
        db = self._internal_reader.parse_tables(dataset_name=dataset_name)
        rdb = {}
        for table_name, table in db.table_dict.items():
            if table_name.startswith("object_attr_") and len(table.df.columns) <= 2:
                continue
            if table_name.startswith("event_attr_") and len(table.df.columns) <= 1:
                continue
            rdb[table_name] = Table(
                time_col=table.time_col,
                pkey_col=table.pkey_col,
                fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,
                df=apply_default_column_dtypes_to_df(
                    clean_datetime(table.df, table.time_col)
                    if table.time_col
                    else table.df,
                    table.time_col,
                ),
            )
        db = Database(rdb)
        return db
