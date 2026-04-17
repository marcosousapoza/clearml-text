from typing import Any, Dict, List

import torch
import torch_geometric
from torch import Tensor
from torch.nn import Embedding, ModuleDict, ModuleList
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, MLP
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.typing import EdgeType, NodeType

from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder

# HGTConv's HeteroLinear uses a grouped GEMM kernel (segment_matmul) that
# crashes on some GPU/driver combinations. Force the naive per-type matmul path.
torch_geometric.backend.use_segment_matmul = False # type: ignore


def _masked_tf_dict(batch: HeteroData, entity_table: NodeType, col_name: str) -> Dict[NodeType, Any]:
    tf_dict = dict(batch.tf_dict)
    tf = tf_dict[entity_table]
    seed_rows = slice(0, int(batch[entity_table].seed_time.size(0)))
    feat_dict = dict(tf.feat_dict)
    for stype_name, cols in tf.col_names_dict.items():
        if col_name not in cols:
            continue
        feat = feat_dict[stype_name].clone()
        col_idx = cols.index(col_name)
        if feat.dim() == 2:
            feat[seed_rows, col_idx] = torch.nan
        elif feat.dim() == 3:
            feat[seed_rows, col_idx, :] = torch.nan
        else:
            raise ValueError(f"Unsupported feature rank {feat.dim()} for {entity_table}.{col_name}")
        feat_dict[stype_name] = feat
        break
    tf_dict[entity_table] = type(tf)(
        feat_dict=feat_dict,
        col_names_dict=tf.col_names_dict,
        y=tf.y,
        num_rows=tf.num_rows,
    )
    return tf_dict


class HeteroHGT(torch.nn.Module):
    """Multi-layer HGT (Heterogeneous Graph Transformer) backbone.

    Wraps :class:`torch_geometric.nn.HGTConv` with per-layer LayerNorm and
    ReLU, matching the interface of :class:`relbench.modeling.nn.HeteroGraphSAGE`.
    The ``**kwargs`` in ``forward`` absorb ``num_sampled_nodes_dict`` /
    ``num_sampled_edges_dict`` passed by the mini-batch loop (HGTConv does not
    use them).
    """

    def __init__(
        self,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        channels: int,
        heads: int = 4,
        num_layers: int = 4,
    ):
        super().__init__()
        metadata = (node_types, edge_types)
        self.convs = ModuleList([
            HGTConv(channels, channels, metadata, heads=heads)
            for _ in range(num_layers)
        ])
        self.norms = ModuleList([
            ModuleDict({nt: LayerNorm(channels, mode="node") for nt in node_types})
            for _ in range(num_layers)
        ])

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters() # type: ignore
        for norm_dict in self.norms:
            for norm in norm_dict.values(): # type: ignore
                norm.reset_parameters()

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Tensor],
        **kwargs,  # absorbs num_sampled_nodes_dict / num_sampled_edges_dict
    ) -> Dict[NodeType, Tensor]:
        for conv, norm_dict in zip(self.convs, self.norms):
            out = conv(x_dict, edge_index_dict)
            x_dict = {
                nt: norm_dict[nt]( # type: ignore
                    out[nt] if (out.get(nt) is not None) else x_dict[nt]
                ).relu()
                for nt in x_dict
            }
        return x_dict


class Model(torch.nn.Module):

    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        num_layers: int,
        channels: int,
        out_channels: int,
        aggr: str,
        norm: str,
        gnn_type: str = "sage",
        hgt_heads: int = 4,
        # List of node types to add shallow embeddings to input
        shallow_list: List[NodeType] | None = None,
        # ID awareness
        id_awareness: bool = False,
    ):
        super().__init__()
        shallow_list = shallow_list or []

        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[
                node_type for node_type in data.node_types if "time" in data[node_type]
            ],
            channels=channels,
        )

        if gnn_type == "hgt":
            self.gnn = HeteroHGT(
                node_types=data.node_types,
                edge_types=data.edge_types,
                channels=channels,
                heads=hgt_heads,
                num_layers=num_layers,
            )
        else:
            self.gnn = HeteroGraphSAGE(
                node_types=data.node_types,
                edge_types=data.edge_types,
                channels=channels,
                aggr=aggr,
                num_layers=num_layers,
            )

        self.head = MLP(
            channels,
            out_channels=out_channels,
            norm=norm,
            num_layers=1,
        )
        self.embedding_dict = ModuleDict(
            {
                node: Embedding(data.num_nodes_dict[node], channels)
                for node in shallow_list
            }
        )

        self.id_awareness_emb = None
        if id_awareness:
            self.id_awareness_emb = torch.nn.Embedding(1, channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()
        for embedding in self.embedding_dict.values():
            torch.nn.init.normal_(embedding.weight, std=0.1) # type: ignore
        if self.id_awareness_emb is not None:
            self.id_awareness_emb.reset_parameters()

    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType,
    ) -> Tensor:
        seed_time = batch[entity_table].seed_time
        tf_dict = _masked_tf_dict(batch, entity_table, "target")
        x_dict = self.encoder(tf_dict)

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
            num_sampled_nodes_dict=batch.num_sampled_nodes_dict,
            num_sampled_edges_dict=batch.num_sampled_edges_dict,
        )

        return self.head(x_dict[entity_table][: seed_time.size(0)])

    def forward_dst_readout(
        self,
        batch: HeteroData,
        entity_table: NodeType,
        dst_table: NodeType,
    ) -> Tensor:
        if self.id_awareness_emb is None:
            raise RuntimeError(
                "id_awareness must be set True to use forward_dst_readout"
            )
        seed_time = batch[entity_table].seed_time
        tf_dict = _masked_tf_dict(batch, entity_table, "target")
        x_dict = self.encoder(tf_dict)
        # Add ID-awareness to the root node
        x_dict[entity_table][: seed_time.size(0)] += self.id_awareness_emb.weight

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
        )

        return self.head(x_dict[dst_table])
