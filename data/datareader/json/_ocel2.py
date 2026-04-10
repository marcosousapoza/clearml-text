import logging
from typing import Any

import pandas as pd
from tqdm import tqdm

from relbench.base.database import Database

from data.datareader.relbench_tables import tables_to_relbench_database
from data.datareader.utils import normalize_token
from data.const import (
    EVENT_ID_COL,
    OBJECT_ID_COL,
    EVENT_TYPE_COL,
    OBJECT_TYPE_COL,
    E2O_EVENT_ID_COL,
    E2O_OBJECT_ID_COL,
    O2O_SRC_COL,
    O2O_DST_COL,
    QUALIFIER_COL,
    TIME_COL,
)

try:
    import simdjson as json
except ImportError:
    import json

from ..base import BaseDataReader

logger = logging.getLogger("DATA")


class _JSONDataReader2(BaseDataReader):
    """OCEL 2.0 JSON reader that produces a 2-node-type EventGraph."""

    def parse_tables(
        self,
        dataset_name: str | None = None,
        time_format: str | None = None,
    ) -> Database:
        """
        Parse OCEL 2.0 JSON and return raw tables.

        Note: datetime parsing and time inference are centralized in
        `src.data.datareader.relbench_tables`.

        Args:
            dataset_name: Optional dataset name (unused, for interface compatibility).
            time_format: Optional strptime format for parsing timestamps
                (applied in `src.data.datareader.relbench_tables`).
        """
        with open(self.path, "rb") as f:
            data = json.load(f)

        events_raw = data.get("events", []) or []
        objects_raw = data.get("objects", []) or []

        logger.info(
            "OCEL2 JSON loaded: events=%d objects=%d", len(events_raw), len(objects_raw)
        )

        # --- Collect all data into lists of dicts ---
        events = self._collect_events(events_raw)
        objects = self._collect_objects(objects_raw)
        e2o = self._collect_e2o(events_raw)
        o2o = self._collect_o2o(objects_raw)

        logger.info(
            "OCEL2 collected: events=%d objects=%d e2o=%d o2o=%d",
            len(events),
            len(objects),
            len(e2o),
            len(o2o),
        )

        # --- Convert to DataFrames ---
        event_df = pd.DataFrame(events)
        object_df = pd.DataFrame(objects)
        e2o_df = (
            pd.DataFrame(e2o)
            if e2o
            else pd.DataFrame(columns=[E2O_EVENT_ID_COL, E2O_OBJECT_ID_COL])
        )
        o2o_df = pd.DataFrame(o2o) if o2o else pd.DataFrame()

        # --- Build attribute tables ---
        event_attr_tables = self._build_event_attr_tables(events_raw)
        object_type_map = dict(
            zip(object_df[OBJECT_ID_COL], object_df[OBJECT_TYPE_COL])
        )
        object_attr_tables = self._build_object_attr_tables(
            objects_raw, object_type_map
        )

        return tables_to_relbench_database(
            event=event_df,
            object=object_df,
            e2o=e2o_df,
            o2o=o2o_df if not o2o_df.empty else None,
            event_attr_by_type=event_attr_tables or None,
            object_attr_by_type=object_attr_tables or None,
            time_format=time_format,
        )

    def _collect_events(self, events_raw: list[dict]) -> list[dict]:
        """Collect event records as list of dicts."""
        return [
            {
                EVENT_ID_COL: str(evt["id"]),
                EVENT_TYPE_COL: normalize_token(evt["type"], empty_token="__INVALID__"),
                TIME_COL: str(evt["time"]),
            }
            for evt in tqdm(events_raw, desc="Collecting events")
        ]

    def _collect_objects(self, objects_raw: list[dict]) -> list[dict]:
        """Collect object records as list of dicts."""
        return [
            {
                OBJECT_ID_COL: str(obj["id"]),
                OBJECT_TYPE_COL: normalize_token(
                    obj["type"], empty_token="__INVALID__"
                ),
            }
            for obj in tqdm(objects_raw, desc="Collecting objects")
        ]

    def _collect_e2o(self, events_raw: list[dict]) -> list[dict]:
        """Collect event-to-object relations as list of dicts."""
        return [
            {
                E2O_EVENT_ID_COL: str(evt["id"]),
                E2O_OBJECT_ID_COL: str(rel["objectId"]),
                QUALIFIER_COL: normalize_token(rel.get("qualifier"), empty_token="__INVALID__"),
            }
            for evt in events_raw
            for rel in (evt.get("relationships") or [])
            if isinstance(rel, dict) and rel.get("objectId")
        ]

    def _collect_o2o(self, objects_raw: list[dict]) -> list[dict]:
        """Collect object-to-object relations as list of dicts."""
        valid_object_ids = {str(obj["id"]) for obj in objects_raw if obj.get("id") is not None}
        return [
            {
                O2O_SRC_COL: str(obj["id"]),
                O2O_DST_COL: str(rel["objectId"]),
                QUALIFIER_COL: normalize_token(rel.get("qualifier"), empty_token="__INVALID__"),
            }
            for obj in objects_raw
            for rel in (obj.get("relationships") or [])
            if (
                isinstance(rel, dict)
                and rel.get("objectId")
                and str(rel["objectId"]) in valid_object_ids
            )
        ]

    def _parse_attributes(self, attrs: list | None) -> dict[str, Any]:
        """Parse OCEL2 attribute list [{name, value}, ...] into a dict."""
        if not attrs or not isinstance(attrs, list):
            return {}
        return {
            str(attr["name"]): attr.get("value")
            for attr in attrs
            if isinstance(attr, dict) and "name" in attr
        }

    def _build_event_attr_tables(
        self,
        events_raw: list[dict],
    ) -> dict[str, pd.DataFrame]:
        """Build per-type event attribute tables."""
        attrs_by_type: dict[str, list[dict]] = {}

        for evt in events_raw:
            etype = normalize_token(evt["type"], empty_token="__INVALID__")
            attrs = self._parse_attributes(evt.get("attributes"))
            if attrs:
                attrs_by_type.setdefault(etype, []).append(
                    {EVENT_ID_COL: str(evt["id"]), **attrs}
                )

        return {etype: pd.DataFrame(rows) for etype, rows in attrs_by_type.items()}

    def _build_object_attr_tables(
        self,
        objects_raw: list[dict],
        object_type_map: dict[str, str],
    ) -> dict[str, pd.DataFrame]:
        """
        Build per-type object attribute tables.

        OCEL2 object attributes can have explicit times (time-varying).
        Attributes without times are kept with missing timestamps; they are filled later.
        """
        attrs_by_type: dict[str, list[dict]] = {}  # otype -> [rows]

        for obj in objects_raw:
            oid = str(obj["id"])
            otype = object_type_map.get(oid)
            if not otype:
                continue

            raw_attrs = obj.get("attributes") or []
            if not raw_attrs or not isinstance(raw_attrs, list):
                continue

            # Group attributes by explicit time (time_str -> {name: value}) plus a timeless row.
            with_time: dict[str, dict[str, Any]] = {}
            without_time: dict[str, Any] = {}

            for attr in raw_attrs:
                if not isinstance(attr, dict) or "name" not in attr:
                    continue
                name = str(attr["name"])
                value = attr.get("value")
                time_raw = attr.get("time")

                if time_raw is not None:
                    time_str = str(time_raw)
                    with_time.setdefault(time_str, {})[name] = value
                else:
                    without_time[name] = value

            for time_str, name_values in with_time.items():
                attrs_by_type.setdefault(otype, []).append(
                    {OBJECT_ID_COL: oid, TIME_COL: time_str, **name_values}
                )

            if without_time:
                attrs_by_type.setdefault(otype, []).append(
                    {OBJECT_ID_COL: oid, **without_time}
                )

        return {otype: pd.DataFrame(rows) for otype, rows in attrs_by_type.items()}
