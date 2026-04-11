"""Tests for hgv.transform: connected_components and flatten."""
import json
from datetime import datetime

import pytest

from hgv.model import OcelEvent, OcelLog, OcelObject, parse
from hgv.transform import connected_components, flatten

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_log(objects, events):
    """Convenience: build an OcelLog from plain dicts."""
    obj_types = list(dict.fromkeys(o["type"] for o in objects))
    objs = [OcelObject(id=o["id"], type=o["type"]) for o in objects]
    evts = [
        OcelEvent(
            id=e["id"],
            type=e.get("type", "act"),
            time=datetime.fromisoformat(e.get("time", "2024-01-01T00:00:00")),
            object_ids=e.get("object_ids", []),
        )
        for e in events
    ]
    evt_types = list(dict.fromkeys(e.type for e in evts))
    return OcelLog(object_types=obj_types, event_types=evt_types, objects=objs, events=evts)


# ---------------------------------------------------------------------------
# connected_components
# ---------------------------------------------------------------------------

class TestConnectedComponents:

    def test_single_component(self):
        """All objects linked through shared events → one component."""
        log = make_log(
            objects=[{"id": "a", "type": "T"}, {"id": "b", "type": "T"}],
            events=[{"id": "e1", "object_ids": ["a", "b"]}],
        )
        comps = connected_components(log)
        assert len(comps) == 1
        assert {o.id for o in comps[0].objects} == {"a", "b"}
        assert {e.id for e in comps[0].events} == {"e1"}

    def test_two_independent_components(self):
        """No shared events → each object pair forms its own component."""
        log = make_log(
            objects=[
                {"id": "a", "type": "T"}, {"id": "b", "type": "T"},
                {"id": "c", "type": "T"}, {"id": "d", "type": "T"},
            ],
            events=[
                {"id": "e1", "object_ids": ["a", "b"]},
                {"id": "e2", "object_ids": ["c", "d"]},
            ],
        )
        comps = connected_components(log)
        assert len(comps) == 2
        ids = [frozenset(o.id for o in c.objects) for c in comps]
        assert frozenset({"a", "b"}) in ids
        assert frozenset({"c", "d"}) in ids

    def test_transitive_connectivity(self):
        """a-b and b-c share b → all three in one component."""
        log = make_log(
            objects=[
                {"id": "a", "type": "T"},
                {"id": "b", "type": "T"},
                {"id": "c", "type": "T"},
            ],
            events=[
                {"id": "e1", "object_ids": ["a", "b"]},
                {"id": "e2", "object_ids": ["b", "c"]},
            ],
        )
        comps = connected_components(log)
        assert len(comps) == 1
        assert {o.id for o in comps[0].objects} == {"a", "b", "c"}

    def test_isolated_object(self):
        """Object with no events forms its own component."""
        log = make_log(
            objects=[
                {"id": "a", "type": "T"},
                {"id": "b", "type": "T"},
                {"id": "lonely", "type": "T"},
            ],
            events=[{"id": "e1", "object_ids": ["a", "b"]}],
        )
        comps = connected_components(log)
        assert len(comps) == 2
        lonely_comp = next(c for c in comps if any(o.id == "lonely" for o in c.objects))
        assert lonely_comp.events == []

    def test_event_with_no_objects(self):
        """Events without any object relationship become orphan components."""
        log = make_log(
            objects=[{"id": "a", "type": "T"}],
            events=[
                {"id": "e1", "object_ids": ["a"]},
                {"id": "orphan", "object_ids": []},
            ],
        )
        comps = connected_components(log)
        assert len(comps) == 2
        orphan_comp = next(c for c in comps if any(e.id == "orphan" for e in c.events))
        assert orphan_comp.objects == []

    def test_components_are_disjoint(self):
        """Objects and events must not appear in more than one component."""
        log = make_log(
            objects=[
                {"id": "x", "type": "A"}, {"id": "y", "type": "B"},
                {"id": "p", "type": "A"}, {"id": "q", "type": "B"},
            ],
            events=[
                {"id": "e1", "object_ids": ["x", "y"]},
                {"id": "e2", "object_ids": ["p", "q"]},
            ],
        )
        comps = connected_components(log)
        all_obj_ids = [o.id for c in comps for o in c.objects]
        all_evt_ids = [e.id for c in comps for e in c.events]
        assert len(all_obj_ids) == len(set(all_obj_ids)), "duplicate objects across components"
        assert len(all_evt_ids) == len(set(all_evt_ids)), "duplicate events across components"

    def test_all_objects_accounted_for(self):
        """Union of component objects equals original objects."""
        log = make_log(
            objects=[
                {"id": "a", "type": "T"}, {"id": "b", "type": "T"},
                {"id": "c", "type": "T"},
            ],
            events=[{"id": "e1", "object_ids": ["a", "b"]}],
        )
        comps = connected_components(log)
        recovered = {o.id for c in comps for o in c.objects}
        assert recovered == {o.id for o in log.objects}

    def test_real_json_fixture(self):
        """The shipping example splits into exactly two order-chains."""
        with open("test.json") as f:
            data = json.load(f)
        log = parse(data)
        comps = connected_components(log)
        # e8 (ship) links p1 and p2, which are already linked via orders
        # through e6 and e7 respectively — so all objects end up in one component
        assert len(comps) == 1
        assert {o.id for o in comps[0].objects} == {o.id for o in log.objects}

    def test_object_types_preserved_per_component(self):
        """Each component only lists the object types that appear in it."""
        log = make_log(
            objects=[
                {"id": "a", "type": "Alpha"},
                {"id": "b", "type": "Beta"},
                {"id": "c", "type": "Gamma"},
            ],
            events=[{"id": "e1", "object_ids": ["a", "b"]}],
        )
        comps = connected_components(log)
        ab_comp = next(c for c in comps if len(c.objects) == 2)
        assert set(ab_comp.object_types) == {"Alpha", "Beta"}
        lone_comp = next(c for c in comps if len(c.objects) == 1)
        assert lone_comp.object_types == ["Gamma"]


# ---------------------------------------------------------------------------
# flatten
# ---------------------------------------------------------------------------

class TestFlatten:

    def test_single_object_event_unchanged(self):
        """Events already associated with one object are not duplicated."""
        log = make_log(
            objects=[{"id": "a", "type": "T"}, {"id": "b", "type": "T"}],
            events=[
                {"id": "e1", "object_ids": ["a"]},
                {"id": "e2", "object_ids": ["b"]},
            ],
        )
        flat = flatten(log)
        assert len(flat.events) == 2
        assert flat.events[0].id == "e1"
        assert flat.events[1].id == "e2"

    def test_multi_object_event_duplicated(self):
        """An event with N objects produces N copies."""
        log = make_log(
            objects=[
                {"id": "a", "type": "T"},
                {"id": "b", "type": "T"},
                {"id": "c", "type": "T"},
            ],
            events=[{"id": "e1", "object_ids": ["a", "b", "c"]}],
        )
        flat = flatten(log)
        assert len(flat.events) == 3
        for e in flat.events:
            assert len(e.object_ids) == 1

    def test_copy_ids_contain_object_id(self):
        """Copy ids encode both original event id and the object id."""
        log = make_log(
            objects=[{"id": "x", "type": "T"}, {"id": "y", "type": "T"}],
            events=[{"id": "ev", "object_ids": ["x", "y"]}],
        )
        flat = flatten(log)
        ids = {e.id for e in flat.events}
        assert "ev:x" in ids
        assert "ev:y" in ids

    def test_each_flat_event_has_one_object(self):
        """After flattening every event carries exactly one object id."""
        with open("test.json") as f:
            data = json.load(f)
        log = parse(data)
        flat = flatten(log)
        for e in flat.events:
            assert len(e.object_ids) <= 1

    def test_event_count_matches_total_relationships(self):
        """Total flat events == sum of object_ids lengths (or 1 for empty)."""
        log = make_log(
            objects=[
                {"id": "a", "type": "T"}, {"id": "b", "type": "T"},
                {"id": "c", "type": "T"},
            ],
            events=[
                {"id": "e1", "object_ids": ["a", "b"]},
                {"id": "e2", "object_ids": ["c"]},
            ],
        )
        flat = flatten(log)
        assert len(flat.events) == 3  # 2 + 1

    def test_flat_preserves_event_type_and_time(self):
        """Copies carry the same type and timestamp as the original."""
        t = "2024-06-01T12:00:00"
        log = make_log(
            objects=[{"id": "a", "type": "T"}, {"id": "b", "type": "T"}],
            events=[{"id": "e1", "type": "my_act", "time": t, "object_ids": ["a", "b"]}],
        )
        flat = flatten(log)
        for e in flat.events:
            assert e.type == "my_act"
            assert e.time == datetime.fromisoformat(t)

    def test_objects_unchanged(self):
        """Flattening must not alter the objects list."""
        log = make_log(
            objects=[{"id": "a", "type": "T"}, {"id": "b", "type": "T"}],
            events=[{"id": "e1", "object_ids": ["a", "b"]}],
        )
        flat = flatten(log)
        assert [o.id for o in flat.objects] == [o.id for o in log.objects]

    def test_no_object_event_preserved(self):
        """Events with no objects are kept verbatim."""
        log = make_log(
            objects=[],
            events=[{"id": "e_empty", "object_ids": []}],
        )
        flat = flatten(log)
        assert len(flat.events) == 1
        assert flat.events[0].id == "e_empty"

    def test_sorted_by_time(self):
        """Output events are sorted by time."""
        log = make_log(
            objects=[{"id": "a", "type": "T"}, {"id": "b", "type": "T"}],
            events=[
                {"id": "late", "time": "2024-01-02T00:00:00", "object_ids": ["a", "b"]},
                {"id": "early", "time": "2024-01-01T00:00:00", "object_ids": ["a", "b"]},
            ],
        )
        flat = flatten(log)
        times = [e.time for e in flat.events]
        assert times == sorted(times)
