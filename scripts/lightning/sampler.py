from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Sampler
from relbench.base import TaskType

from torch_geometric.data import Data, HeteroData
from torch_geometric.loader.utils import filter_data, filter_hetero_data, get_input_nodes
from torch_geometric.sampler import (
    NeighborSampler,
    NodeSamplerInput,
    SamplerOutput,
    HeteroSamplerOutput,
)
from torch_geometric.sampler.base import SubgraphType
from torch_geometric.typing import EdgeType, InputNodes, OptTensor


class BalancedUnderSampler(Sampler[int]):
    """Train-time undersampler for classification and regression targets.

    Classification targets are balanced per discrete class.
    Regression targets are first binned by quantiles, then balanced per bin.
    """

    def __init__(
        self,
        targets: Tensor,
        task_type: TaskType,
        num_regression_bins: int = 10,
    ) -> None:
        if targets.ndim != 1:
            raise ValueError("BalancedUnderSampler requires 1D targets")

        if task_type in (
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
        ):
            group_ids = self._classification_groups(targets)
        elif task_type == TaskType.REGRESSION:
            group_ids = self._regression_groups(targets, num_regression_bins)
        else:
            raise ValueError(
                "BalancedUnderSampler supports binary classification, "
                "multiclass classification, and regression tasks"
            )

        _, inverse = torch.unique(group_ids, sorted=True, return_inverse=True)
        self.group_indices = [
            torch.nonzero(inverse == group_idx, as_tuple=False).view(-1)
            for group_idx in range(int(inverse.max().item()) + 1)
        ]
        self.group_indices = [indices for indices in self.group_indices if indices.numel() > 0]

        if len(self.group_indices) < 2:
            raise ValueError("BalancedUnderSampler requires at least two populated groups")

        self.samples_per_group = min(indices.numel() for indices in self.group_indices)
        self.num_samples = self.samples_per_group * len(self.group_indices)

    @staticmethod
    def _classification_groups(targets: Tensor) -> Tensor:
        return targets.to(torch.long)

    @staticmethod
    def _regression_groups(targets: Tensor, num_bins: int) -> Tensor:
        targets = targets.to(torch.float32)
        if targets.numel() < 2:
            raise ValueError("BalancedUnderSampler requires at least two regression targets")

        quantiles = torch.linspace(0.0, 1.0, steps=num_bins + 1, device=targets.device)
        boundaries = torch.quantile(targets, quantiles)
        boundaries = torch.unique(boundaries)
        if boundaries.numel() < 2:
            raise ValueError("BalancedUnderSampler could not form regression quantile bins")

        inner_boundaries = boundaries[1:-1]
        return torch.bucketize(targets, inner_boundaries, right=False)

    def __iter__(self):
        sampled_indices = []
        for indices in self.group_indices:
            perm = torch.randperm(indices.numel())[: self.samples_per_group]
            sampled_indices.append(indices[perm])

        epoch_indices = torch.cat(sampled_indices, dim=0)
        epoch_indices = epoch_indices[torch.randperm(epoch_indices.numel())]
        return iter(epoch_indices.tolist())

    def __len__(self) -> int:
        return self.num_samples


class TupleNeighborLoader(DataLoader):
    r"""
    A node-neighbor loader for tuple-level supervision.

    Each training example is a tuple of nodes:
        (entity_0[i], entity_1[i], ..., entity_k[i])

    The loader:
      1. Flattens all tuple members into one seed-node list.
      2. Samples one subgraph around those seed nodes.
      3. Annotates the returned mini-batch with tuple-level bookkeeping.

    Key outputs attached to the returned batch/store:
      - input_nodes_tuple: original global node IDs per tuple slot
      - input_id_tuple: original input IDs per tuple slot
      - input_local_nodes_tuple: local node indices for each tuple slot
      - input_masks_tuple: boolean masks over local nodes for each tuple slot
      - batch_size_tuple: number of seeds contributed by each tuple slot
      - tuple_example_id: flattened-seed -> tuple example index
      - tuple_member_id: flattened-seed -> tuple slot index
      - tuple_y: tuple-level targets, shape [batch_num_examples, ...]
      - num_tuple_examples: number of tuple examples in the batch

    Notes:
      - All tuple entries must refer to the same input node type.
      - All tuple entries must have the same number of examples.
      - Tuple targets belong to examples, not to individual seed nodes.
      - In PyG's node sampling contract, seed nodes are kept at the front of the
        sampled node list in input order; this loader relies on that invariant. :contentReference[oaicite:1]{index=1}
    """

    def __init__(
        self,
        data: Union[Data, HeteroData],
        num_neighbors: Union[List[int], Dict[EdgeType, List[int]]],
        input_nodes_tuple: Sequence[InputNodes],
        targets: Optional[Tensor] = None,
        input_time_tuple: Optional[Sequence[OptTensor]] = None,
        replace: bool = False,
        subgraph_type: Union[SubgraphType, str] = "directional",
        disjoint: bool = False,
        temporal_strategy: str = "uniform",
        time_attr: Optional[str] = None,
        weight_attr: Optional[str] = None,
        transform: Optional[Callable] = None,
        transform_sampler_output: Optional[Callable] = None,
        is_sorted: bool = False,
        neighbor_sampler: Optional[NeighborSampler] = None,
        directed: bool = True,
        **kwargs,
    ):
        if len(input_nodes_tuple) == 0:
            raise ValueError("input_nodes_tuple must not be empty")

        if input_time_tuple is None:
            input_time_tuple = [None] * len(input_nodes_tuple)

        if len(input_time_tuple) != len(input_nodes_tuple):
            raise ValueError("input_time_tuple must match input_nodes_tuple length")

        if time_attr is None and any(t is not None for t in input_time_tuple):
            raise ValueError("input_time_tuple requires time_attr")

        self.data = data
        self.transform = transform
        self.transform_sampler_output = transform_sampler_output

        if neighbor_sampler is None:
            neighbor_sampler = NeighborSampler(
                data,
                num_neighbors=num_neighbors,
                replace=replace,
                subgraph_type=subgraph_type,
                disjoint=disjoint,
                temporal_strategy=temporal_strategy,
                time_attr=time_attr,
                weight_attr=weight_attr,
                is_sorted=is_sorted,
                share_memory=kwargs.get("num_workers", 0) > 0,
                directed=directed,
            )
        self.node_sampler = neighbor_sampler

        # Normalize all inputs through PyG helper:
        self.inputs = []
        for inp, inp_time in zip(input_nodes_tuple, input_time_tuple):
            input_type, nodes, input_id = get_input_nodes(data, inp, input_id=None)
            self.inputs.append((input_type, nodes, input_id, inp_time))

        input_types = [x[0] for x in self.inputs]
        self.input_type = input_types[0]
        if any(t != self.input_type for t in input_types):
            raise ValueError("All tuple entries must have the same input node type")

        lengths = [x[1].size(0) for x in self.inputs]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"All tuple entries must have the same length, got {lengths}"
            )

        self.num_examples = lengths[0]

        if targets is not None and targets.size(0) != self.num_examples:
            raise ValueError(
                f"targets must have first dimension {self.num_examples}, "
                f"got {targets.size(0)}"
            )
        self.targets = targets

        kwargs.pop("dataset", None)
        kwargs.pop("collate_fn", None)

        super().__init__(
            dataset=range(self.num_examples), # type: ignore
            collate_fn=self.collate_fn,
            **kwargs,
        )

    def collate_fn(self, index):
        if not isinstance(index, Tensor):
            index = torch.tensor(index, dtype=torch.long)

        tuple_nodes = []
        tuple_ids = []
        tuple_times = []

        flat_nodes = []
        flat_ids = []
        flat_times = []

        tuple_example_id_parts = []
        tuple_member_id_parts = []

        for member_id, (_, nodes, input_id, inp_time) in enumerate(self.inputs):
            batch_nodes = nodes[index]
            batch_ids = input_id[index] if input_id is not None else index.clone()
            batch_time = inp_time[index] if inp_time is not None else None

            tuple_nodes.append(batch_nodes)
            tuple_ids.append(batch_ids)
            tuple_times.append(batch_time)

            flat_nodes.append(batch_nodes)
            flat_ids.append(batch_ids)
            if batch_time is not None:
                flat_times.append(batch_time)

            # flattened seed -> tuple example index / tuple slot index
            tuple_example_id_parts.append(index.clone())
            tuple_member_id_parts.append(torch.full_like(index, member_id))

        flat_nodes = torch.cat(flat_nodes, dim=0)
        flat_ids = torch.cat(flat_ids, dim=0)
        flat_time = torch.cat(flat_times, dim=0) if len(flat_times) > 0 else None

        tuple_example_id = torch.cat(tuple_example_id_parts, dim=0)
        tuple_member_id = torch.cat(tuple_member_id_parts, dim=0)

        tuple_y = self.targets[index] if self.targets is not None else None

        sampler_input = NodeSamplerInput(
            input_id=flat_ids,
            node=flat_nodes,
            time=flat_time,
            input_type=self.input_type,
        )
        out = self.node_sampler.sample_from_nodes(sampler_input)

        if self.transform_sampler_output is not None:
            out = self.transform_sampler_output(out)

        batch = self._filter(out)
        self._annotate(
            batch=batch,
            tuple_nodes=tuple_nodes,
            tuple_ids=tuple_ids,
            tuple_times=tuple_times,
            tuple_example_id=tuple_example_id,
            tuple_member_id=tuple_member_id,
            tuple_y=tuple_y,
        )

        if self.transform is not None:
            batch = self.transform(batch)

        return batch

    def _filter(self, out):
        if isinstance(out, SamplerOutput):
            batch = filter_data(
                self.data,
                out.node,
                out.row,
                out.col,
                out.edge,
                self.node_sampler.edge_permutation,
            )
            batch.n_id = out.node

            if out.edge is not None:
                edge = out.edge.to(torch.long)
                perm = self.node_sampler.edge_permutation
                batch.e_id = perm[edge] if perm is not None else edge

            batch.batch = out.batch
            batch.num_sampled_nodes = out.num_sampled_nodes
            batch.num_sampled_edges = out.num_sampled_edges
            batch.input_id = out.metadata[0]
            batch.seed_time = out.metadata[1]
            batch.batch_size = out.metadata[0].numel()
            return batch

        if isinstance(out, HeteroSamplerOutput):
            batch = filter_hetero_data(
                self.data,
                out.node,
                out.row,
                out.col,
                out.edge,
                self.node_sampler.edge_permutation,
            )

            for key, node in out.node.items():
                batch[key].n_id = node

            for key, edge in (out.edge or {}).items():
                if edge is not None:
                    edge = edge.to(torch.long)
                    perm = self.node_sampler.edge_permutation
                    edge_perm = perm.get(key) if perm is not None else None
                    batch[key].e_id = (
                        edge_perm[edge] if edge_perm is not None else edge
                    )

            batch.set_value_dict("batch", out.batch)
            batch.set_value_dict("num_sampled_nodes", out.num_sampled_nodes)
            batch.set_value_dict("num_sampled_edges", out.num_sampled_edges)

            store = batch[self.input_type]
            store.input_id = out.metadata[0]
            store.seed_time = out.metadata[1]
            store.batch_size = out.metadata[0].numel()
            return batch

        raise TypeError(f"Unsupported sampler output: {type(out)}")

    def _annotate(
        self,
        batch,
        tuple_nodes,
        tuple_ids,
        tuple_times,
        tuple_example_id,
        tuple_member_id,
        tuple_y=None,
    ):
        sizes = [x.numel() for x in tuple_nodes]

        offsets = []
        cur = 0
        for s in sizes:
            offsets.append(cur)
            cur += s

        if isinstance(batch, Data):
            target = batch
            total_seed_nodes = batch.batch_size
            num_local_nodes = batch.n_id.numel()
        else:
            target = batch[self.input_type]
            total_seed_nodes = target.batch_size
            num_local_nodes = target.n_id.numel()

        expected = sum(sizes)
        if total_seed_nodes != expected:
            raise RuntimeError(
                f"Expected {expected} seed nodes, got {total_seed_nodes}"
            )

        # The first `total_seed_nodes` local nodes correspond to the flattened seeds.
        input_local_nodes_tuple = []
        input_masks_tuple = []

        for off, size in zip(offsets, sizes):
            idx = torch.arange(off, off + size, dtype=torch.long)
            mask = torch.zeros(num_local_nodes, dtype=torch.bool)
            mask[idx] = True
            input_local_nodes_tuple.append(idx)
            input_masks_tuple.append(mask)

        target.input_nodes_tuple = tuple(tuple_nodes)
        target.input_id_tuple = tuple(tuple_ids)
        target.batch_size_tuple = tuple(sizes)
        target.input_local_nodes_tuple = tuple(input_local_nodes_tuple)
        target.input_masks_tuple = tuple(input_masks_tuple)

        # flattened seed metadata
        target.tuple_example_id = tuple_example_id
        target.tuple_member_id = tuple_member_id
        target.num_tuple_examples = sizes[0]
        target.tuple_arity = len(tuple_nodes)

        # example-level supervision
        if tuple_y is not None:
            target.tuple_y = tuple_y

        if any(t is not None for t in tuple_times):
            target.seed_time_tuple = tuple(tuple_times)
