from .model import OcelEvent, OcelLog, OcelObject


def connected_components(log: OcelLog) -> list[OcelLog]:
    """Split an OcelLog into its connected components.

    Two objects belong to the same component when they co-appear in at least
    one event (directly or transitively).  Events are assigned to the component
    of their participating objects.  Events with no object relationships form
    their own singleton component each.

    Returns a list of OcelLog instances, one per component, ordered by the
    first-seen object id of each component (stable across calls for the same
    input).
    """
    # --- union-find --------------------------------------------------------
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for obj in log.objects:
        parent[obj.id] = obj.id

    for event in log.events:
        ids = event.object_ids
        if len(ids) > 1:
            for oid in ids[1:]:
                if oid in parent and ids[0] in parent:
                    union(ids[0], oid)
                elif oid not in parent:
                    # object referenced by event but not declared — add on the fly
                    parent[oid] = oid
                    union(ids[0], oid)

    # objects referenced in events but not declared in log.objects
    for event in log.events:
        for oid in event.object_ids:
            if oid not in parent:
                parent[oid] = oid

    # --- bucket objects by root --------------------------------------------
    # preserve insertion order of roots as encountered in log.objects
    root_order: list[str] = []
    seen_roots: set[str] = set()
    for obj in log.objects:
        r = find(obj.id)
        if r not in seen_roots:
            root_order.append(r)
            seen_roots.add(r)

    obj_by_root: dict[str, list[OcelObject]] = {r: [] for r in root_order}
    for obj in log.objects:
        obj_by_root[find(obj.id)].append(obj)

    # --- bucket events by root --------------------------------------------
    evt_by_root: dict[str, list[OcelEvent]] = {r: [] for r in root_order}
    orphan_events: list[OcelEvent] = []

    for event in log.events:
        roots = {find(oid) for oid in event.object_ids if oid in parent}
        if not roots:
            orphan_events.append(event)
        else:
            r = find(next(iter(roots)))
            evt_by_root.setdefault(r, []).append(event)

    # --- build sub-logs ----------------------------------------------------
    used_obj_types = {o.type for o in log.objects}
    used_evt_types = {e.type for e in log.events}

    components: list[OcelLog] = []
    for r in root_order:
        objs = obj_by_root[r]
        evts = evt_by_root.get(r, [])
        if not objs and not evts:
            continue
        sub_obj_types = [t for t in log.object_types if t in {o.type for o in objs}]
        sub_evt_types = [t for t in log.event_types if t in {e.type for e in evts}]
        components.append(OcelLog(
            object_types=sub_obj_types,
            event_types=sub_evt_types,
            objects=objs,
            events=evts,
        ))

    # orphan events (no objects) each become their own component
    for event in orphan_events:
        components.append(OcelLog(
            object_types=[],
            event_types=[event.type] if event.type in used_evt_types else [],
            objects=[],
            events=[event],
        ))

    return components


def flatten(log: OcelLog) -> OcelLog:
    """Flatten an OcelLog so every event is associated with exactly one object.

    For each event that involves N objects, N copies of the event are created,
    each carrying a single object id.  The copy ids are derived as
    ``<original_event_id>:<object_id>``.  Events with no object relationships
    are kept as-is (they already satisfy the constraint).

    The returned log preserves the same object_types and event_types lists and
    the same objects.  Events are sorted by time, then by original event id to
    keep the output deterministic.
    """
    flat_events: list[OcelEvent] = []

    for event in log.events:
        if len(event.object_ids) <= 1:
            flat_events.append(event)
        else:
            for oid in event.object_ids:
                flat_events.append(OcelEvent(
                    id=f"{event.id}:{oid}",
                    type=event.type,
                    time=event.time,
                    object_ids=[oid],
                ))

    flat_events.sort(key=lambda e: (e.time, e.id))

    return OcelLog(
        object_types=log.object_types,
        event_types=log.event_types,
        objects=log.objects,
        events=flat_events,
    )
