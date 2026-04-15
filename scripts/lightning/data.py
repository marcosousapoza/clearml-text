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
from scripts.text_embedder import GloveTextEmbedding
from task.utils import MEntityTask, add_task_to_database
from task.utils.transform import TargetTransform


@dataclass
class DataArtifacts:
    dataset: Dataset
    task: MEntityTask
    data: HeteroData
    col_stats_dict: dict
    split_inputs: dict
    cache_root: Path
    task_node_type: str
    target_transform: TargetTransform | None


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

        db, split_inputs, target_transform = add_task_to_database(db, task, self.task_name, col_to_stype_dict)
        if self.flatten:
            db.reindex_pkeys_and_fkeys()
        graph_cache_name = f"{self.dataset_name}_{self.task_name}{'_flat' if self.flatten else ''}"
        data, col_stats_dict = make_ocel_graph(
            db,
            col_to_stype_dict=col_to_stype_dict,
            text_embedder_cfg=TextEmbedderConfig(
                text_embedder=GloveTextEmbedding(
                    device=torch.device('cpu') # cuda lead to weird warnings
                ), batch_size=256,
            ),
            cache_dir=str(cache_root / graph_cache_name / "materialized"),
        )

        task_node_type = f"{self.task_name}_labels"
        num_neighbors = [int(self.num_neighbors / 2**i) for i in range(self.num_layers)]
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
            target_transform=target_transform,
        )

    def train_dataloader(self) -> NeighborLoader:
        return self._loader_dict["train"]

    def val_dataloader(self) -> NeighborLoader:
        return self._loader_dict["val"]

    def test_dataloader(self) -> NeighborLoader:
        return self._loader_dict["test"]
