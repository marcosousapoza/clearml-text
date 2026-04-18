from typing import Any, Dict, List

import torch
import torch_geometric
from torch import Tensor
from torch.nn import Embedding, ModuleDict, ModuleList
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.typing import EdgeType, NodeType

from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder

# HGTConv's HeteroLinear uses a grouped GEMM kernel (segment_matmul) that
# crashes on some GPU/driver combinations. Force the naive per-type matmul path.
torch_geometric.backend.use_segment_matmul = False # type: ignore


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
    """Encoder + temporal encoder + heterogeneous GNN backbone.

    Returns the full local-node embedding tensor for ``entity_table``. A
    downstream head (e.g. ``TupleConcatPredictor``) gathers the seed-node slots
    it cares about using ``batch[entity_table].input_local_nodes_tuple``.
    """

    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        num_layers: int,
        channels: int,
        aggr: str,
        gnn_type: str = "sage",
        hgt_heads: int = 4,
        shallow_list: List[NodeType] | None = None,
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

        self.embedding_dict = ModuleDict(
            {
                node: Embedding(data.num_nodes_dict[node], channels)
                for node in shallow_list
            }
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        for embedding in self.embedding_dict.values():
            torch.nn.init.normal_(embedding.weight, std=0.1) # type: ignore

    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType,
    ) -> Tensor:
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)

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

        return x_dict[entity_table]
