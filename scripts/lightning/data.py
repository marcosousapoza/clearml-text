from dataclasses import dataclass
from pathlib import Path

import lightning as L
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_frame.config.text_embedder import TextEmbedderConfig

from data.dataset  import register_all_datasets
from data.flat import flatten as flatten_db
from task import register_tasks
import torch
from data.cache import configure_cache_environment
from relbench.base import Dataset
from relbench.datasets import get_dataset
from data.graph import make_ocel_graph
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task
from scripts.text_embedder import SentenceTextEmbedding
from task.utils import MEntityTask, add_task_to_database


def _build_num_neighbors(num_neighbors_base: int, num_layers: int) -> list[int]:
    """Build the per-hop neighbor count list for NeighborLoader.

    The first hop (seed label node → entity object) is 1-to-1, so we use -1
    (sample all).  Subsequent hops start at *num_neighbors_base* and decay
    geometrically with ratio ~0.63 (gentler than halving at 0.5).

    Example for num_layers=4, num_neighbors_base=32:
        [-1, 32, 20, 13]
    """
    decay = 0.63
    hops = [-1] + [
        max(1, round(num_neighbors_base * (decay ** i)))
        for i in range(num_layers - 1)
    ]
    return hops


@dataclass
class DataArtifacts:
    dataset: Dataset
    task: MEntityTask
    data: HeteroData
    col_stats_dict: dict
    split_inputs: dict
    cache_root: Path
    task_node_type: str


class RelbenchLightningDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        task_name: str,
        batch_size: int,
        num_layers: int,
        num_neighbors: int,
        temporal_strategy: str,
        num_workers: int,
        cache_dir: str | None = None,
        flatten: bool = False,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.task_name = task_name
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.num_neighbors = num_neighbors
        self.temporal_strategy = temporal_strategy
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.flatten = flatten

        self.artifacts: DataArtifacts | None = None
        self._loader_dict: dict[str, NeighborLoader] = {}

    def setup(self, stage: str | None = None) -> None:

        if self.artifacts is not None:
            return

        cache_root = configure_cache_environment(self.cache_dir)
        register_all_datasets(cache_root)
        register_tasks(cache_root)
        dataset: Dataset = get_dataset(self.dataset_name, download=False)
        task: MEntityTask = get_task(self.dataset_name, self.task_name, download=False)  # type: ignore[assignment]
        db = dataset.get_db()
        if self.flatten:
            db = flatten_db(db, task.object_types)

        col_to_stype_dict = get_stype_proposal(db)
        col_to_stype_dict = {
            table_name: {
                col: col_to_stype[col]
                for col in db.table_dict[table_name].df.columns
                if col in col_to_stype
            }
            for table_name, col_to_stype in col_to_stype_dict.items()
            if table_name in db.table_dict
        }
        if hasattr(dataset, "set_stype"):
            col_to_stype_dict = dataset.set_stype(col_to_stype_dict)  # type: ignore[attr-defined]

        db, split_inputs = add_task_to_database(db, task, self.task_name, col_to_stype_dict)
        if self.flatten:
            db.reindex_pkeys_and_fkeys()
        graph_cache_name = f"{self.dataset_name}_{self.task_name}{'_flat' if self.flatten else ''}"
        data, col_stats_dict = make_ocel_graph(
            db,
            col_to_stype_dict=col_to_stype_dict,
            text_embedder_cfg=TextEmbedderConfig(
                text_embedder=SentenceTextEmbedding(
                    device=torch.device('cpu') # cuda lead to weird warnings
                ), batch_size=256,
            ),
            cache_dir=str(cache_root / graph_cache_name / "materialized"),
        )

        task_node_type = f"{self.task_name}_labels"
        hop_counts = _build_num_neighbors(self.num_neighbors, self.num_layers)
        # For any edge whose destination is a label node (other than the seed
        # itself), set neighbor count to 0. This prevents the object from
        # aggregating over other label nodes of the same entity at different
        # timestamps, which leaks future target information through the object hub.
        # Block all edges that touch a label node except the seed-node
        # self-connection.  Two directions must be blocked:
        #   - src=labels → dst=other: a neighbour would aggregate from the label
        #     node, which carries the attached target y, and pass that signal back
        #     to the seed in the next hop.
        #   - src=other  → dst=labels: a non-seed label node aggregates from
        #     objects, enriching an object hub that the seed can then read.
        def _is_label_leakage(et: tuple[str, str, str]) -> bool:
            src, _, dst = et
            src_is_label = src.endswith("_labels")
            dst_is_label = dst.endswith("_labels")
            # Allow the seed's self-connection (same label type, same node).
            if src == dst and src_is_label:
                return False
            return src_is_label or dst_is_label

        num_neighbors = {
            et: ([0] * self.num_layers if _is_label_leakage(et) else hop_counts)
            for et in data.edge_types
        }
        self._loader_dict = {}
        for split in ["train", "val", "test"]:
            table_input = split_inputs[split]
            self._loader_dict[split] = NeighborLoader(
                data,
                num_neighbors=num_neighbors,
                time_attr="time",
                input_nodes=table_input.nodes,
                input_time=table_input.time,
                transform=table_input.transform,
                batch_size=self.batch_size,
                temporal_strategy=self.temporal_strategy,
                disjoint=True,
                shuffle=split == "train",
                num_workers=self.num_workers,
                persistent_workers=self.num_workers > 0,
            )

        self.artifacts = DataArtifacts(
            dataset=dataset,
            task=task,
            data=data,
            col_stats_dict=col_stats_dict,
            split_inputs=split_inputs,
            cache_root=cache_root,
            task_node_type=task_node_type,
        )

    def train_dataloader(self) -> NeighborLoader:
        return self._loader_dict["train"]

    def val_dataloader(self) -> NeighborLoader:
        return self._loader_dict["val"]

    def test_dataloader(self) -> NeighborLoader:
        return self._loader_dict["test"]
