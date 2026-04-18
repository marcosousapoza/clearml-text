"""Graph construction utilities for OCEL databases.

Wraps RelBench's ``make_pkey_fkey_graph`` with OCEL-specific metapath
augmentation:

* Object ↔ Event shortcut via the ``e2o`` bridge table.
* Object ↔ Object shortcut via the ``o2o`` bridge table (when present).

If a bridge table carries **no informative attributes** (i.e. ``make_pkey_fkey_graph``
would have given it only a ``__const__`` feature column), it is dropped from
the graph after the metapaths have been added.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.transforms import AddMetaPaths

from relbench.base import Database
from relbench.modeling.graph import make_pkey_fkey_graph

from .const import (
    E2O_TABLE,
    E2O_EVENT_ID_COL,
    E2O_OBJECT_ID_COL,
    EVENT_TABLE,
    OBJECT_TABLE,
    O2O_TABLE,
    O2O_DST_COL,
    O2O_SRC_COL,
)


def _is_uninformative(db: Database, table_name: str) -> bool:
    """Return True when a bridge table has no usable attribute columns.

    A table is considered uninformative when its only non-key, non-time
    columns are either:
    - absent entirely, or
    - absent entirely.
    """
    if table_name not in db.table_dict:
        return False
    table = db.table_dict[table_name]
    df = table.df
    # Columns that are purely structural / relational keys
    skip = set()
    if table.pkey_col is not None:
        skip.add(table.pkey_col)
    skip.update(table.fkey_col_to_pkey_table.keys())
    if table.time_col is not None:
        skip.add(table.time_col)

    informative = [c for c in df.columns if c not in skip]
    return len(informative) == 0


def _drop_node_type(data: HeteroData, node_type: str) -> HeteroData:
    """Remove a node type and all edges that touch it from *data* in-place."""
    if node_type not in data.node_types:
        return data

    # Remove all edge types that reference this node type
    to_remove = [
        et for et in data.edge_types if node_type in (et[0], et[2])
    ]
    for et in to_remove:
        del data[et]

    del data[node_type]
    return data


def make_ocel_graph(
    db: Database,
    col_to_stype_dict: Dict[str, Dict[str, Any]],
    text_embedder_cfg: Optional[TextEmbedderConfig] = None,
    cache_dir: Optional[str] = None,
) -> Tuple[HeteroData, Dict[str, Dict[str, Dict[StatType, Any]]]]:
    """Build a PyG HeteroData graph from an OCEL RelBench database.

    Extends :func:`relbench.modeling.graph.make_pkey_fkey_graph` with:

    1. **Object ↔ Event metapaths** through ``e2o``.
    2. **Object ↔ Object metapaths** through ``o2o`` (when the table exists).
    3. **Bridge-table pruning**: if ``e2o`` (or ``o2o``) carries no informative
       attributes, it is removed from the graph after the shortcut edges have
       been added.

    Args:
        db: RelBench Database built from an OCEL log.
        col_to_stype_dict: Column-to-stype mapping (from
            :func:`data.utils.get_stype_proposal` or a dataset-specific
            override).
        text_embedder_cfg: Optional text-embedding config forwarded to
            ``make_pkey_fkey_graph``.
        cache_dir: Optional cache directory forwarded to
            ``make_pkey_fkey_graph``.

    Returns:
        ``(data, col_stats_dict)`` — same contract as
        ``make_pkey_fkey_graph``.
    """
    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=text_embedder_cfg,
        cache_dir=cache_dir,
    )

    metapaths: list[list[tuple[str, str, str]]] = []

    if E2O_TABLE in data.node_types:
        e2o_to_event = f"f2p_{E2O_EVENT_ID_COL}"
        e2o_to_object = f"f2p_{E2O_OBJECT_ID_COL}"
        object_to_e2o = f"rev_{e2o_to_object}"

        # object → event  (object → e2o → event)
        metapaths.append([
            (OBJECT_TABLE, object_to_e2o, E2O_TABLE),
            (E2O_TABLE, e2o_to_event, EVENT_TABLE),
        ])
        # event → object  (event → e2o → object)
        event_to_e2o = f"rev_{e2o_to_event}"
        metapaths.append([
            (EVENT_TABLE, event_to_e2o, E2O_TABLE),
            (E2O_TABLE, e2o_to_object, OBJECT_TABLE),
        ])

    if O2O_TABLE in data.node_types:
        o2o_to_src = f"f2p_{O2O_SRC_COL}"
        o2o_to_dst = f"f2p_{O2O_DST_COL}"
        src_to_o2o = f"rev_{o2o_to_src}"

        # object_src → object_dst  (object → o2o → object)
        metapaths.append([
            (OBJECT_TABLE, src_to_o2o, O2O_TABLE),
            (O2O_TABLE, o2o_to_dst, OBJECT_TABLE),
        ])

    if metapaths:
        transform = AddMetaPaths(metapaths, drop_unconnected_node_types=False)
        data = transform(data)

    for bridge_table in (E2O_TABLE, O2O_TABLE):
        if _is_uninformative(db, bridge_table):
            data = _drop_node_type(data, bridge_table)
            col_stats_dict.pop(bridge_table, None)

    # Label nodes must be sinks: they receive from the object graph but must
    # not send messages back into it. Removing label→object edges prevents
    # objects from aggregating across label nodes of different timestamps,
    # which would leak future targets back to the seed through the object hub.
    label_node_types = {nt for nt in data.node_types if nt.endswith("_labels")}
    leaky_edges = [et for et in data.edge_types if et[0] in label_node_types and et[2] != et[0]]
    for et in leaky_edges:
        del data[et]

    return data, col_stats_dict
