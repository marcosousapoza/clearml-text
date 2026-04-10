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
    TIME_COL,
    EVENT_TYPE_COL,
    OBJECT_TYPE_COL,
    E2O_EVENT_ID_COL,
    E2O_OBJECT_ID_COL,
    QUALIFIER_COL,
)

try:
    import simdjson as json
except ImportError:
    import json

from ..base import BaseDataReader

logger = logging.getLogger("DATA")


class _JSONDataReader1(BaseDataReader):
    """OCEL 1.0 JSON reader that produces a 2-node-type EventGraph."""

    def parse_tables(
        self,
        dataset_name: str | None = None,
        time_format: str | None = None,
    ) -> Database:
        """
        Parse OCEL 1.0 JSON and return raw tables.

        Note: datetime parsing and time inference are centralized in
        `src.data.datareader.relbench_tables`.

        Args:
            dataset_name: Optional dataset name (unused, for interface compatibility).
            time_format: Optional strptime format for parsing timestamps
                (applied in `src.data.datareader.relbench_tables`).
        """
        with open(self.path, "rb") as f:
            data = json.load(f)

        events_raw = data.get("ocel:events", {}) or {}
        objects_raw = data.get("ocel:objects", {}) or {}

        logger.info(
            "OCEL1 JSON loaded: events=%d objects=%d", len(events_raw), len(objects_raw)
        )

        # --- Collect all data into lists of dicts ---
        events = self._collect_events(events_raw)
        objects = self._collect_objects(objects_raw)
        e2o = self._collect_e2o(events_raw)

        logger.info(
            "OCEL1 collected: events=%d objects=%d e2o=%d",
            len(events),
            len(objects),
            len(e2o),
        )

        # --- Convert to DataFrames ---
        event_df = pd.DataFrame(events)
        object_df = pd.DataFrame(objects)
        e2o_df = (
            pd.DataFrame(e2o)
            if e2o
            else pd.DataFrame(columns=[E2O_EVENT_ID_COL, E2O_OBJECT_ID_COL])
        )
        o2o_df = pd.DataFrame()  # OCEL1 has no o2o

        # --- Build attribute tables ---
        event_attr_tables = self._build_event_attr_tables(events_raw)
        object_attr_tables = self._build_object_attr_tables(objects_raw)

        return tables_to_relbench_database(
            event=event_df,
            object=object_df,
            e2o=e2o_df,
            o2o=o2o_df if not o2o_df.empty else None,
            event_attr_by_type=event_attr_tables or None,
            object_attr_by_type=object_attr_tables or None,
            time_format=time_format,
        )

    def _collect_events(self, events_raw: dict[str, Any]) -> list[dict]:
        """Collect event records as list of dicts."""
        return [
            {
                EVENT_ID_COL: str(eid),
                EVENT_TYPE_COL: normalize_token(
                    evt["ocel:activity"], empty_token="__INVALID__"
                ),
                TIME_COL: str(evt["ocel:timestamp"]),
            }
            for eid, evt in tqdm(events_raw.items(), desc="Collecting events")
        ]

    def _collect_objects(self, objects_raw: dict[str, Any]) -> list[dict]:
        """Collect object records as list of dicts."""
        return [
            {
                OBJECT_ID_COL: str(oid),
                OBJECT_TYPE_COL: normalize_token(
                    obj["ocel:type"], empty_token="__INVALID__"
                ),
            }
            for oid, obj in tqdm(objects_raw.items(), desc="Collecting objects")
        ]

    def _collect_e2o(self, events_raw: dict[str, Any]) -> list[dict]:
        """Collect event-to-object relations as list of dicts."""
        relations: list[dict] = []
        for eid, evt in events_raw.items():
            typed = evt.get("ocel:typedOmap") or []
            if typed:
                relations.extend(
                    {
                        E2O_EVENT_ID_COL: str(eid),
                        E2O_OBJECT_ID_COL: str(rel["ocel:oid"]),
                        QUALIFIER_COL: normalize_token(rel.get("ocel:qualifier"), empty_token="__INVALID__"),
                    }
                    for rel in typed
                    if isinstance(rel, dict) and rel.get("ocel:oid")
                )
            else:
                relations.extend(
                    {
                        E2O_EVENT_ID_COL: str(eid),
                        E2O_OBJECT_ID_COL: str(obj_id),
                    }
                    for obj_id in (evt.get("ocel:omap") or [])
                )
        return relations

    def _build_event_attr_tables(
        self,
        events_raw: dict[str, Any],
    ) -> dict[str, pd.DataFrame]:
        """Build per-type event attribute tables."""
        # Group attributes by event type
        attrs_by_type: dict[str, list[dict]] = {}
        for eid, evt in events_raw.items():
            etype = normalize_token(evt["ocel:activity"], empty_token="__INVALID__")
            attrs = evt.get("ocel:vmap")
            if attrs and isinstance(attrs, dict):
                attrs_by_type.setdefault(etype, []).append(
                    {EVENT_ID_COL: str(eid), **attrs}
                )

        return {etype: pd.DataFrame(rows) for etype, rows in attrs_by_type.items()}

    def _build_object_attr_tables(
        self,
        objects_raw: dict[str, Any],
    ) -> dict[str, pd.DataFrame]:
        """Build per-type object attribute tables (raw; timestamps inferred later)."""
        # Group attributes by object type
        attrs_by_type: dict[str, list[dict]] = {}
        for oid, obj in objects_raw.items():
            otype = normalize_token(obj["ocel:type"], empty_token="__INVALID__")
            attrs = obj.get("ocel:ovmap")
            if attrs and isinstance(attrs, dict):
                attrs_by_type.setdefault(otype, []).append(
                    {OBJECT_ID_COL: str(oid), **attrs}
                )

        return {otype: pd.DataFrame(rows) for otype, rows in attrs_by_type.items()}
