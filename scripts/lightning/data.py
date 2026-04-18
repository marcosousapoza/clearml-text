from dataclasses import dataclass
from pathlib import Path

import lightning as L
import torch
from torch_geometric.data import HeteroData
from torch_frame.config.text_embedder import TextEmbedderConfig

from data.dataset  import register_all_datasets
from data.flat import flatten as flatten_db
from task import register_tasks
from data.cache import configure_cache_environment
from relbench.base import Dataset
from relbench.datasets import get_dataset
from relbench.modeling.graph import to_unix_time
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task
from data.graph import make_ocel_graph
from scripts.text_embedder import SentenceTextEmbedding
from task.utils import MEntityTask, build_target_tensor

from .sampler import TupleNeighborLoader


def _build_num_neighbors(num_neighbors_base: int, num_layers: int) -> list[int]:
    """Per-hop neighbor counts: start at *num_neighbors_base* and decay ~0.63.

    Example for num_layers=4, num_neighbors_base=32: [32, 20, 13, 8]
    """
    decay = 0.63
    return [
        max(1, round(num_neighbors_base * (decay ** i)))
        for i in range(num_layers)
    ]


@dataclass
class DataArtifacts:
    dataset: Dataset
    task: MEntityTask
    data: HeteroData
    col_stats_dict: dict
    cache_root: Path
    entity_table: str
    tuple_arity: int


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
        self._loader_dict: dict[str, TupleNeighborLoader] = {}

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

        entity_tables = set(task.entity_tables)
        if len(entity_tables) != 1:
            raise ValueError(
                f"TupleNeighborLoader requires a single entity node type, "
                f"got {task.entity_tables}"
            )
        entity_table = next(iter(entity_tables))
        tuple_arity = len(task.entity_cols)

        train_hops = _build_num_neighbors(self.num_neighbors, self.num_layers)
        eval_hops = [-1] * self.num_layers
        train_num_neighbors = {et: train_hops for et in data.edge_types}
        eval_num_neighbors = {et: eval_hops for et in data.edge_types}

        self._loader_dict = {}
        for split in ["train", "val", "test"]:
            split_df = task.get_table(split, mask_input_cols=False).df
            split_df = split_df.sort_values(task.time_col, kind="stable").reset_index(drop=True)

            input_nodes_tuple = [
                (
                    entity_table,
                    torch.from_numpy(split_df[entity_col].to_numpy(dtype="int64", copy=True)),
                )
                for entity_col in task.entity_cols
            ]
            time_tensor = torch.from_numpy(to_unix_time(split_df[task.time_col]))
            input_time_tuple = [time_tensor] * tuple_arity
            targets = build_target_tensor(task, split_df)

            self._loader_dict[split] = TupleNeighborLoader(
                data,
                num_neighbors=train_num_neighbors if split == "train" else eval_num_neighbors,
                input_nodes_tuple=input_nodes_tuple,
                input_time_tuple=input_time_tuple,
                targets=targets,
                time_attr="time",
                temporal_strategy=self.temporal_strategy,
                disjoint=True,
                batch_size=self.batch_size,
                shuffle=split == "train",
                num_workers=self.num_workers,
                persistent_workers=self.num_workers > 0,
            )

        self.artifacts = DataArtifacts(
            dataset=dataset,
            task=task,
            data=data,
            col_stats_dict=col_stats_dict,
            cache_root=cache_root,
            entity_table=entity_table,
            tuple_arity=tuple_arity,
        )

    def train_dataloader(self) -> TupleNeighborLoader:
        return self._loader_dict["train"]

    def val_dataloader(self) -> TupleNeighborLoader:
        return self._loader_dict["val"]

    def test_dataloader(self) -> TupleNeighborLoader:
        return self._loader_dict["test"]
