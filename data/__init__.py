from relbench.base.table import Table

from .datareader.relbench_tables import apply_default_column_dtypes_to_df


_ORIGINAL_TABLE_LOAD = Table.load.__func__


@classmethod
def _normalized_table_load(cls, path):
    table = _ORIGINAL_TABLE_LOAD(cls, path)
    table.df = apply_default_column_dtypes_to_df(table.df, table.time_col)
    return table


Table.load = _normalized_table_load # type: ignore
