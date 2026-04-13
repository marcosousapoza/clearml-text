import os
from typing import Any

import pandas as pd
from relbench.base.dataset import Dataset
from relbench.base.database import Database
from relbench.base.table import Table
from torch_frame import stype

from ._utils import unzip_file, parse_ocel_to_database
from ..datareader.relbench_tables import apply_default_column_dtypes
from ..const import E2O_TABLE, O2O_TABLE, OBJECT_ID_COL, OBJECT_TABLE, O2O_DST_COL, O2O_SRC_COL
from ..wrapper import check_dbs


@check_dbs
def _drop_attr_columns(db: Database, columns: set[str]) -> Database:
    """
    Drop selected columns from all event/object attribute tables, preserving table metadata.
    """
    if not columns:
        return db

    out: dict[str, Table] = {}
    for name, table in db.table_dict.items():
        if name.startswith("event_attr_") or name.startswith("object_attr_"):
            drop_cols = [c for c in table.df.columns if c in columns]
            df = table.df.drop(columns=drop_cols) if drop_cols else table.df
        else:
            df = table.df
        out[name] = Table(
            df=df,
            time_col=table.time_col,
            pkey_col=table.pkey_col,
            fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,
        )
    return apply_default_column_dtypes(Database(table_dict=out))


def _restore_missing_linked_objects(db: Database, source_db: Database) -> Database:
    """
    Restore object and object-attribute rows for object IDs referenced by link tables.

    Some datasets are filtered by time after OCEL parsing. When the object table is
    time-filtered more aggressively than link tables, surviving `e2o`/`o2o` rows can
    reference objects that no longer exist in the filtered `object` table. We restore
    those object rows from the unfiltered source database so the relational graph
    remains valid without dropping observed event-object links.
    """
    object_df = db.table_dict[OBJECT_TABLE].df
    linked_ids = set(db.table_dict[E2O_TABLE].df[OBJECT_ID_COL].dropna().tolist())
    o2o_table = db.table_dict.get(O2O_TABLE)
    if o2o_table is not None:
        linked_ids.update(o2o_table.df[O2O_SRC_COL].dropna().tolist())
        linked_ids.update(o2o_table.df[O2O_DST_COL].dropna().tolist())

    current_ids = set(object_df[OBJECT_ID_COL].dropna().tolist())
    missing_ids = linked_ids - current_ids
    if not missing_ids:
        return db

    source_object_df = source_db.table_dict[OBJECT_TABLE].df
    restored_object_df = pd.concat(
        [
            object_df,
            source_object_df[source_object_df[OBJECT_ID_COL].isin(missing_ids)],
        ],
        ignore_index=True,
    ).drop_duplicates(subset=[OBJECT_ID_COL], keep="first")

    out: dict[str, Table] = {}
    for name, table in db.table_dict.items():
        df = table.df
        if name == OBJECT_TABLE:
            df = restored_object_df
        elif name.startswith("object_attr_") and OBJECT_ID_COL in df.columns:
            source_attr_df = source_db.table_dict[name].df if name in source_db.table_dict else None
            if source_attr_df is not None:
                df = pd.concat(
                    [
                        df,
                        source_attr_df[source_attr_df[OBJECT_ID_COL].isin(missing_ids)],
                    ],
                    ignore_index=True,
                ).drop_duplicates()
        out[name] = Table(
            df=df,
            time_col=table.time_col,
            pkey_col=table.pkey_col,
            fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,
        )
    return apply_default_column_dtypes(Database(table_dict=out))

class OCELDataset(Dataset):
    """Base dataset that owns its stype post-processing."""

    def set_stype(
        self,
        col_to_stype_dict: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        updated = {
            table_name: dict(col_to_stype)
            for table_name, col_to_stype in col_to_stype_dict.items()
        }

        self._update_table_stypes(
            updated,
            "event",
            {
                "event_id": stype.numerical,
                "type": stype.categorical,
                "time": stype.timestamp,
            },
        )
        self._update_table_stypes(
            updated,
            "object",
            {
                "object_id": stype.numerical,
                "type": stype.categorical,
                "time": stype.timestamp,
            },
        )
        self._update_table_stypes(
            updated,
            "e2o",
            {
                "event_id": stype.numerical,
                "object_id": stype.numerical,
                "qualifier": stype.categorical,
                "time": stype.timestamp,
            },
        )
        self._update_table_stypes(
            updated,
            "o2o",
            {
                "object_id_src": stype.numerical,
                "object_id_dst": stype.numerical,
                "qualifier": stype.categorical,
                "time": stype.timestamp,
            },
        )
        return updated

    @staticmethod
    def _update_table_stypes(
        col_to_stype_dict: dict[str, dict[str, Any]],
        table_name: str,
        overrides: dict[str, Any],
    ) -> None:
        if table_name not in col_to_stype_dict:
            return
        table_map = col_to_stype_dict[table_name]
        for column, value in overrides.items():
            if column in table_map:
                table_map[column] = value


class ContainerLogisticsDataset(OCELDataset):
    """
    ContainerLogisticsDataset

    Source:
        https://zenodo.org/records/18373888/files/container_logistics.json?download=1

    Description:
        See https://www.ocel-standard.org/event-logs/simulations/logistics/ from the official ocel 2.0
        specification website for more information.

    Repository / DOI:
        Zenodo record  8289899 (DOI: 10.5281/zenodo.8289899)

    Reference:
        Knopp, B., & Graves, N. (2023). Container Logistics Object-centric Event Log [Data set].
        Zenodo. https://doi.org/10.5281/zenodo.8428084

    License:
        CC-BY 4.0

    Usage example:
        >>> ds = ContainerLogisticsDataset(cache_dir="ocel/")
        >>> db = ds.get_db()
    """

    val_timestamp = pd.Timestamp("2024-04-06 12:36:51.900000")
    test_timestamp = pd.Timestamp("2024-06-01 00:00:00")

    _uri = (
        "https://zenodo.org/records/18373888/files/container_logistics.json?download=1"
    )
    _file_format = "json"
    _drop_attr_cols = {"Weight"}

    def make_db(self) -> Database:
        db = parse_ocel_to_database(
            uri=self._uri,
            file_format=self._file_format,
            dataset_name=self.__class__.__name__,
        )
        return _drop_attr_columns(db, self._drop_attr_cols)

    def set_stype(
        self,
        col_to_stype_dict: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        updated = super().set_stype(col_to_stype_dict)
        self._update_table_stypes(
            updated,
            "object_attr_Customer Order",
            {
                "object_id": stype.numerical,
                "time": stype.timestamp,
                "AmountofGoods": stype.numerical,
            },
        )
        self._update_table_stypes(
            updated,
            "object_attr_Vehicle",
            {
                "object_id": stype.numerical,
                "time": stype.timestamp,
                "DepartureDate": stype.timestamp,
            },
        )
        self._update_table_stypes(
            updated,
            "object_attr_Container",
            {
                "object_id": stype.numerical,
                "time": stype.timestamp,
                "AmountofHandlingUnits": stype.numerical,
                "Status": stype.categorical,
                "Weight": stype.categorical,
            },
        )
        self._update_table_stypes(
            updated,
            "object_attr_Transport Document",
            {
                "object_id": stype.numerical,
                "time": stype.timestamp,
                "AmountofContainers": stype.numerical,
                "Status": stype.categorical,
            },
        )
        return updated


class OrderManagementDataset(OCELDataset):
    """
    OrderManagementDataset

    Source:
        https://zenodo.org/records/8428112/files/order-management.json?download=1

    Description:
        See https://www.ocel-standard.org/event-logs/simulations/order-management/ from the official ocel 2.0
        specification website for more information.

    Repository / DOI:
        Zenodo record  8428112 (DOI: 10.5281/zenodo.8428112)

    Reference:
        Knopp, B., & Wil M.P. van der Aalst. (2023). Order Management Object-centric Event Log in
        OCEL 2.0 Standard [Data set]. Zenodo. https://doi.org/10.5281/zenodo.8428112

    License:
        CC-BY 4.0

    Usage example:
        >>> ds = OrderManagementDataset(cache_dir="ocel/")
        >>> db = ds.get_db()
    """

    val_timestamp = pd.Timestamp("2023-09-27 09:22:38")
    test_timestamp = pd.Timestamp("2023-12-01 00:00:00")

    _uri = "https://zenodo.org/records/8428112/files/order-management.json?download=1"
    _file_format = "json"

    def make_db(self) -> Database:
        return parse_ocel_to_database(
            uri=self._uri,
            file_format=self._file_format,
            dataset_name=self.__class__.__name__,
        )

    def set_stype(
        self,
        col_to_stype_dict: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        updated = super().set_stype(col_to_stype_dict)
        self._update_table_stypes(
            updated,
            "object_attr_orders",
            {
                "object_id": stype.numerical,
                "time": stype.timestamp,
                "price": stype.numerical,
            },
        )
        self._update_table_stypes(
            updated,
            "object_attr_items",
            {
                "object_id": stype.numerical,
                "time": stype.timestamp,
                "weight": stype.numerical,
                "price": stype.numerical,
            },
        )
        self._update_table_stypes(
            updated,
            "object_attr_products",
            {
                "object_id": stype.numerical,
                "time": stype.timestamp,
                "weight": stype.numerical,
                "price": stype.numerical,
            },
        )
        self._update_table_stypes(
            updated,
            "object_attr_employees",
            {
                "object_id": stype.numerical,
                "time": stype.timestamp,
                "role": stype.categorical,
            },
        )
        self._update_table_stypes(
            updated,
            "object_attr_packages",
            {
                "object_id": stype.numerical,
                "time": stype.timestamp,
                "weight": stype.numerical,
            },
        )
        return updated


class BPI2019(OCELDataset):
    """
    BPI2019

    Source:
        https://data.4tu.nl/file/46a7e15b-10c7-4ab2-988d-ee67d8ea515a/ae11f6ca-2824-407d-98ea-ec8bc456e714

    Description:
        See https://data.4tu.nl/articles/_/12715853/1 for a description of the BPI Challenge 2019. A more detailed
        description can be found at the ICPM competition website:
        https://icpmconference.org/icpm-2019/contests-challenges/bpi-challenge-2019/

    Repository / DOI:
        DOI: 10.4121/UUID:D06AFF4B-79F0-45E6-8EC8-E19730C248F1

    Reference:
        van Dongen, B.F., Dataset BPI Challenge 2019. 4TU.Centre for Research Data.
        https://doi.org/10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1

    License:
        CC-BY 4.0
    """

    val_timestamp = pd.Timestamp("2018-09-25 19:54:24")
    test_timestamp = pd.Timestamp("2018-11-22 04:44:12")

    _uri = "https://data.4tu.nl/file/46a7e15b-10c7-4ab2-988d-ee67d8ea515a/ae11f6ca-2824-407d-98ea-ec8bc456e714"
    _file_format = "json"
    _event_attr_stypes: dict[str, stype] = {
        "cCompany": stype.categorical,
        "cDocType": stype.categorical,
        "cGR": stype.categorical,
        "cGRbasedInvVerif": stype.categorical,
        "cItem": stype.categorical,
        "cItemCat": stype.categorical,
        "cItemType": stype.categorical,
        "cSpendAreaText": stype.categorical,
        "cSpendClassText": stype.categorical,
        "cSubSPendAreaText": stype.categorical,
        "cVendor": stype.categorical,
        "cVendorName": stype.categorical,
        "eCumNetWorth": stype.numerical,
        "resource": stype.categorical,
    }
    _drop_attr_cols = {
        # Identifier-like / near-unique attributes.
        "ID",
        "idx",
        "cID",
        "cPOID",
        # Constant descriptor across BPI2019 attribute tables.
        "cPurDocCat",
    }
    def _pre_parsing(self, file_path: str) -> str:
        # Unzip the file if it is a zip archive
        unzip_file(file_path)
        # The expected file after extraction
        json_cache_dir = os.path.join(os.path.dirname(file_path), "BPIC19.jsonocel")
        if not os.path.isfile(json_cache_dir):
            raise FileNotFoundError(
                f"Expected '{json_cache_dir}' after extraction, but it wasn't found"
            )
        return json_cache_dir

    def make_db(self) -> Database:
        raw_db = parse_ocel_to_database(
            uri=self._uri,
            file_format=self._file_format,
            dataset_name=self.__class__.__name__,
            pre_parse_fn=self._pre_parsing,
        )
        # filter strange periods
        db = raw_db.from_(pd.Timestamp("2018-01-01")).upto(pd.Timestamp("2019-01-30"))
        db = _restore_missing_linked_objects(db, source_db=raw_db)
        return _drop_attr_columns(db, self._drop_attr_cols)

    def set_stype(
        self,
        col_to_stype_dict: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        updated = super().set_stype(col_to_stype_dict)
        for table_name in list(updated):
            if table_name.startswith("event_attr_"):
                self._update_table_stypes(
                    updated,
                    table_name,
                    {
                        "event_id": stype.numerical,
                        "time": stype.timestamp,
                        **self._event_attr_stypes,
                    },
                )
        return updated


class BPI2017(OCELDataset):
    """
    BPI2017

    Source:
        https://data.4tu.nl/file/6889ca3f-97cf-459a-b630-3b0b0d8664b5/5d5b9f89-7fa6-4c92-b6ac-04f854bdf92e

    Description:
        See https://data.4tu.nl/articles/dataset/Event_Graph_of_BPI_Challenge_2017/14169584/1
        for a description of the BPI Challenge 2017.

    Repository / DOI:
        DOI: 10.4121/14169584.v1

    Reference:
        Fahland, Dirk and Esser, Stefan, Dataset BPI Challenge 2017. 4TU.Centre for Research Data.
        https://doi.org/10.4121/14169584.v1

    License:
        CC-BY 4.0
    """

    val_timestamp = pd.Timestamp("2016-10-05 10:29:07.040500")
    test_timestamp = pd.Timestamp("2016-12-04 00:20:05.269750")

    _uri = "https://data.4tu.nl/file/6889ca3f-97cf-459a-b630-3b0b0d8664b5/5d5b9f89-7fa6-4c92-b6ac-04f854bdf92e"
    _file_format = "json"
    _event_attr_stypes: dict[str, stype] = {
        "ApplicationType": stype.categorical,
        "LoanGoal": stype.categorical,
        "RequestedAmount": stype.numerical,
        "Action": stype.categorical,
        "lifecycle": stype.categorical,
        "resource": stype.categorical,
    }
    _drop_attr_cols = {
        # Identifier-like columns and row index.
        "EventID",
        "EventIDraw",
        "case",
        "idx",
        # Constant or near-constant placeholders in this log.
        "Accepted",
        "CreditScore",
        "EventOrigin",
        "FirstWithdrawalAmount",
        "MonthlyCost",
        "NumberOfTerms",
        "OfferID",
        "OfferedAmount",
        "Selected",
    }
    def _pre_parsing(self, file_path: str) -> str:
        # Unzip the file if it is a zip archive
        unzip_file(file_path)
        # The expected file after extraction
        json_cache_dir = os.path.join(os.path.dirname(file_path), "BPIC17.jsonocel")
        if not os.path.isfile(json_cache_dir):
            raise FileNotFoundError(
                f"Expected '{json_cache_dir}' after extraction, but it wasn't found"
            )
        return json_cache_dir

    def make_db(self) -> Database:
        db = parse_ocel_to_database(
            uri=self._uri,
            file_format=self._file_format,
            dataset_name=self.__class__.__name__,
            pre_parse_fn=self._pre_parsing,
        )
        return _drop_attr_columns(db, self._drop_attr_cols)

    def set_stype(
        self,
        col_to_stype_dict: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        updated = super().set_stype(col_to_stype_dict)
        for table_name in list(updated):
            if table_name.startswith("event_attr_"):
                self._update_table_stypes(
                    updated,
                    table_name,
                    {
                        "event_id": stype.numerical,
                        "time": stype.timestamp,
                        **self._event_attr_stypes,
                    },
                )
        return updated
