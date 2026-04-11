from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class OcelObject:
    id: str
    type: str


@dataclass
class OcelEvent:
    id: str
    type: str
    time: datetime
    object_ids: list[str] = field(default_factory=list)


@dataclass
class OcelLog:
    object_types: list[str]
    event_types: list[str]
    objects: list[OcelObject]
    events: list[OcelEvent]


def flatten_to_object_type(log: OcelLog, object_type: str) -> OcelLog:
    """Project an OcelLog onto a single object type (traditional flattening).

    Only objects of *object_type* are retained.  Each event keeps only its
    relationships to objects of that type; events with no such relationships
    are dropped.  This produces a conventional single-case-notion event log
    where the chosen object type acts as the process instance.
    """
    kept_ids = {o.id for o in log.objects if o.type == object_type}
    kept_objects = [o for o in log.objects if o.id in kept_ids]

    kept_events: list[OcelEvent] = []
    for e in log.events:
        new_ids = [oid for oid in e.object_ids if oid in kept_ids]
        if new_ids:
            kept_events.append(OcelEvent(id=e.id, type=e.type, time=e.time, object_ids=new_ids))

    used_evt_types = {e.type for e in kept_events}
    evt_types = [t for t in log.event_types if t in used_evt_types]

    return OcelLog(
        object_types=[object_type],
        event_types=evt_types,
        objects=kept_objects,
        events=kept_events,
    )


def parse(data: dict) -> OcelLog:
    objects = [
        OcelObject(id=o["id"], type=o["type"])
        for o in data.get("objects", [])
    ]

    events = []
    for e in data.get("events", []):
        time = datetime.fromisoformat(e["time"])
        object_ids = [r["objectId"] for r in e.get("relationships", [])]
        events.append(OcelEvent(id=e["id"], type=e["type"], time=time, object_ids=object_ids))

    events.sort(key=lambda e: e.time)

    seen_obj_types: dict[str, None] = {}
    for ot in data.get("objectTypes", []):
        seen_obj_types[ot["name"]] = None
    # also collect any types that appear in objects but not in objectTypes
    for o in objects:
        seen_obj_types.setdefault(o.type, None)

    seen_evt_types: dict[str, None] = {}
    for et in data.get("eventTypes", []):
        seen_evt_types[et["name"]] = None
    for e in events:
        seen_evt_types.setdefault(e.type, None)

    return OcelLog(
        object_types=list(seen_obj_types),
        event_types=list(seen_evt_types),
        objects=objects,
        events=events,
    )
