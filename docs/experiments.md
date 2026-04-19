# Experiments

## Scripts and Setup

All experiments are orchestrated through three script families:

**Baselines.** `scripts/baseline/` evaluates statistical baselines on each registered task. For regression tasks, the baselines are global zero, global mean, global median, per-entity mean, and per-entity median. For multiclass classification, random, majority-class, and per-entity majority predictors are evaluated. These baselines provide a lower-bound reference for GNN performance.

**GNN Training.** `scripts/lightning/` implements a PyTorch Lightning training pipeline. The model stacks a `HeteroEncoder` (for heterogeneous node features via `torch_frame`), a `HeteroTemporalEncoder`, a GNN backbone, and a `TupleConcatPredictor` head. The default backbone is **HeteroGraphSAGE** with sum aggregation. Temporal neighbor sampling uses a geometrically decayed neighborhood: base fan-out of 8, decayed per hop as max(1, round(8 × 0.63^i)), giving approximately [8, 5, 3, 2] neighbors across four hops.

**Flattened Projection Comparison.** The `--flatten` flag in `scripts/lightning/cli.py` applies `flatten_db()` from `data/flat/duckdb.py` before graph construction. Flattening explodes the object-centric representation into a case-based projection: each event is duplicated once per associated object, producing a one-object-per-event table that mirrors a conventional flattened event log. This serves as the primary ablation, allowing a direct comparison between learning on the full object-centric graph and learning on its flattened projection under otherwise identical conditions.

**Multi-seed Evaluation.** `scripts/lightning/train_all.py` runs each task across five random seeds and supports per-GPU job parallelism. Results are aggregated across seeds to report mean and standard deviation.

---

## Hyperparameters

All GNN experiments use the following fixed hyperparameter configuration:

| Parameter | Value |
|---|---|
| Hidden channels | 32 |
| GNN layers | 2 |
| Neighbor fan-out (base) | 8 (decayed per hop) |
| Dropout | 0.4 |
| Batch size | 256 |
| Max epochs | 50 |
| Optimizer | AdamW, lr = 1e-3, weight decay = 1e-4 |
| LR scheduler | ReduceLROnPlateau, patience = 5, min lr = 1e-6 |
| Early stopping patience | 5 |
| Seeds | 1–5 |

This configuration is held constant across all datasets and tasks. The rationale for using a fixed set of hyperparameters is twofold. First, the primary research question concerns the structural advantage of the object-centric graph representation over its flattened projection; exhaustive per-task tuning would conflate representation quality with search effort and risk overfitting to the specific datasets used here. Second, with only three datasets currently available, per-dataset tuning would leave little signal to separate genuine gains from lucky configurations. We therefore treat this as a controlled comparison under a standard budget. Additional performance gains are expected with dedicated hyperparameter optimization — for example via Optuna over learning rate, hidden channels, number of layers, dropout, and neighbor fan-out — and we leave this to future work once broader dataset coverage is established.

---

## HGT

A **Heterogeneous Graph Transformer (HGT)** backbone is implemented in `scripts/model.py` and can be selected via `--gnn-type hgt`. The HGT variant wraps `torch_geometric.nn.HGTConv` with per-layer LayerNorm, ReLU, and dropout, using 4 attention heads by default. However, HGT is not used in the reported experiments. In preliminary trials, HGT underperformed HeteroGraphSAGE on all tasks. We attribute this to current data availability and data quality limitations: HGT's attention mechanism benefits from dense, diverse relational structures, and the datasets used here are relatively small with limited object-type diversity, leaving the transformer-style inductive bias with insufficient signal to leverage. As larger and richer OCEL logs become available, HGT and other attention-based heterogeneous GNNs represent a natural next step.
