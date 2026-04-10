import argparse
import copy
import math
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from model import Model
from text_embedder import GloveTextEmbedding
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, L1Loss
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from tqdm import tqdm

from relbench.base import Dataset, Table, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import data.dataset  # noqa: F401
import task as task_registry  # noqa: F401
from data.dataset import DEFAULT_CACHE_ROOT, configure_cache_environment
from task.utils import MEntityTask, add_task_to_database

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-event")
parser.add_argument("--task", type=str, default="user-attendance")
parser.add_argument("--lr", type=float, default=0.005)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--channels", type=int, default=128)
parser.add_argument("--aggr", type=str, default="sum")
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_neighbors", type=int, default=128)
parser.add_argument("--temporal_strategy", type=str, default="last")
parser.add_argument("--max_steps_per_epoch", type=int, default=2000)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--cache_dir",
    type=str,
    default=str(DEFAULT_CACHE_ROOT),
)
args = parser.parse_args()
cache_root = configure_cache_environment(args.cache_dir)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_num_threads(1)
seed_everything(args.seed)

dataset: Dataset = get_dataset(args.dataset, download=False)
task: MEntityTask = get_task(args.dataset, args.task, download=False)  # type: ignore
db = dataset.get_db()

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

# Add one full task table to db BEFORE materialization so row ids remain stable.
db, split_inputs = add_task_to_database(db, task, args.task, col_to_stype_dict)
data, col_stats_dict = make_pkey_fkey_graph(
    db,
    col_to_stype_dict=col_to_stype_dict,
    text_embedder_cfg=TextEmbedderConfig(                                                                    
        text_embedder=GloveTextEmbedding(device=device), batch_size=256                                      
    ), 
    cache_dir=str(cache_root / f"{args.dataset}_{args.task}" / "materialized"),
)

clamp_min, clamp_max = None, None
train_table = task.get_table("train")

if task.task_type == TaskType.BINARY_CLASSIFICATION:
    out_channels = 1
    loss_fn = BCEWithLogitsLoss()
    tune_metric = "roc_auc"
    higher_is_better = True
elif task.task_type == TaskType.REGRESSION:
    out_channels = 1
    loss_fn = L1Loss()
    tune_metric = "mae"
    higher_is_better = False
    # Get the clamp value at inference time
    clamp_min, clamp_max = np.percentile(
        train_table.df[task.target_col].to_numpy(), [2, 98]
    )
elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
    out_channels = task.num_labels # type: ignore
    loss_fn = BCEWithLogitsLoss()
    tune_metric = "multilabel_auprc_macro"
    higher_is_better = True
elif task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
    out_channels = task.num_classes # type: ignore
    train_target = split_inputs["train"].target.long()
    class_counts = torch.bincount(train_target, minlength=out_channels).float()
    class_weights = class_counts.sum() / torch.clamp(class_counts, min=1.0)
    class_weights = class_weights / class_weights.mean()
    loss_fn = CrossEntropyLoss(weight=class_weights.to(device))
    tune_metric = "multiclass_f1"
    higher_is_better = True
else:
    raise ValueError(f"Task type {task.task_type} is unsupported")

loader_dict: Dict[str, NeighborLoader] = {}
task_node_type = f"{args.task}_labels"
for split in ["train", "val", "test"]:
    table_input = split_inputs[split]
    loader_dict[split] = NeighborLoader(
        data,
        num_neighbors=[int(args.num_neighbors / 2**i) for i in range(args.num_layers)],
        time_attr="time",
        input_nodes=table_input.nodes,
        input_time=table_input.time,
        transform=table_input.transform,
        batch_size=args.batch_size,
        temporal_strategy=args.temporal_strategy,
        shuffle=split == "train",
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )


def train() -> float:
    model.train()

    loss_accum = count_accum = 0
    steps = 0
    total_steps = min(len(loader_dict["train"]), args.max_steps_per_epoch)
    for batch in tqdm(loader_dict["train"], total=total_steps):
        batch = batch.to(device)

        optimizer.zero_grad()
        pred = model(
            batch,
            task_node_type,
        )
        pred = pred.view(-1) if pred.size(1) == 1 else pred

        if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            loss = loss_fn(pred, batch[task_node_type].y.long())
        else:
            loss = loss_fn(pred.float(), batch[task_node_type].y.float())
        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().item() * pred.size(0)
        count_accum += pred.size(0)

        steps += 1
        if steps > args.max_steps_per_epoch:
            break

    return loss_accum / count_accum


@torch.no_grad()
def test(loader: NeighborLoader) -> np.ndarray:
    model.eval()

    pred_list = []
    for batch in tqdm(loader):
        batch = batch.to(device)
        pred = model(
            batch,
            task_node_type,
        )
        if task.task_type == TaskType.REGRESSION:
            assert clamp_min is not None
            assert clamp_max is not None
            pred = torch.clamp(pred, clamp_min, clamp_max)

        if task.task_type in [
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTILABEL_CLASSIFICATION,
        ]:
            pred = torch.sigmoid(pred)

        if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            pred = torch.softmax(pred, dim=1)

        pred = pred.view(-1) if pred.size(1) == 1 else pred
        pred_list.append(pred.detach().cpu())
    return torch.cat(pred_list, dim=0).numpy()


model = Model(
    data=data,
    col_stats_dict=col_stats_dict,
    num_layers=args.num_layers,
    channels=args.channels,
    out_channels=out_channels,
    aggr=args.aggr,
    norm="batch_norm",
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
state_dict = None
best_val_metric = -math.inf if higher_is_better else math.inf
for epoch in range(1, args.epochs + 1):
    train_loss = train()
    val_pred = test(loader_dict["val"])
    val_metrics = task.evaluate(val_pred, task.get_table("val"))
    print(f"Epoch: {epoch:02d}, Train loss: {train_loss}, Val metrics: {val_metrics}")

    if (higher_is_better and val_metrics[tune_metric] >= best_val_metric) or (
        not higher_is_better and val_metrics[tune_metric] <= best_val_metric
    ):
        best_val_metric = val_metrics[tune_metric]
        state_dict = copy.deepcopy(model.state_dict())


model.load_state_dict(state_dict)
val_pred = test(loader_dict["val"])
val_metrics = task.evaluate(val_pred, task.get_table("val"))
print(f"Best Val metrics: {val_metrics}")

test_pred = test(loader_dict["test"])
test_metrics = task.evaluate(test_pred)
print(f"Best test metrics: {test_metrics}")
