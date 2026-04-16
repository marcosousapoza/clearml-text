# Implementation Summary

This document describes the concrete implementation details of the object-centric predictive
process monitoring pipeline built on top of RelBench and PyTorch Geometric.

---

## Model Architecture

The GNN model (`scripts/model.py`) follows a three-stage pipeline for each training instance:

### 1. Heterogeneous Tabular Encoder (`HeteroEncoder`)

A table-specific encoder maps each row-node's raw multi-modal attributes into a shared
`channels`-dimensional embedding. Text fields use GloVe embeddings; categorical fields use
embedding lookups; numerical fields are encoded by small MLPs. This corresponds to the
column-wise encoding described in Equations 3.23–3.25, where each table type has its own
`EncT` function that fuses column-level representations into a single initial node embedding
`h_v^(0)`.

### 2. Temporal Encoder (`HeteroTemporalEncoder`)

Each node's timestamp is encoded relative to the seed time `t_seed` using sinusoidal
positional encoding (Vaswani et al., 2017). The resulting vector is **added** to the
attribute embedding from the previous stage:

```
h_v^(0)' = h_v^(0) + TE(t_v)
```

This keeps time and attribute information in the same embedding space without a separate
branch, as described in Section 4.4.3.

### 3. Heterogeneous GraphSAGE (`HeteroGraphSAGE`)

A heterogeneous variant of GraphSAGE with `K` message-passing layers and hidden dimension
`channels`. Type-specific message and update functions are used for each (node type, edge
type) pair. After `K` layers, only the seed-node embedding is retained and passed to a
1-layer MLP prediction head that maps to the task output.

> **Note:** The codebase implements `HeteroGraphSAGE` only. The GAT variant discussed in
> Section 4.4.4 is a planned alternative not yet present.

---

## Default Hyperparameters

| Hyperparameter | Value |
|---|---|
| Hidden channels | 128 |
| GNN layers (K) | 3 |
| Aggregation | `sum` |
| Neighbors per layer | 128, 64, 32 (halved per hop) |
| Temporal sampling strategy | `uniform` |
| Learning rate | 0.005 |
| Optimizer | Adam |
| LR scheduler | CosineAnnealingLR (T_max = epochs) |
| Epochs | 10 |
| Batch size | 512 |
| Max steps per epoch | 2000 |
| Random seeds | 1, 2, 3, 4, 5 |

---

## Loss Functions and Tuning Metrics

| Task type | Loss function | Tuning metric | Direction |
|---|---|---|---|
| Multiclass classification | CrossEntropyLoss with inverse-frequency class weights | Macro-F1 | ↑ |
| Regression | L1Loss (MAE) | MAE | ↓ |
| Binary classification | BCEWithLogitsLoss | ROC-AUC | ↑ |

For regression tasks, predictions are clamped to the [2nd, 98th] percentile of training
targets. Model checkpointing saves the epoch with the best validation tuning metric.

---

## Registered Tasks

19 tasks are registered across 4 datasets. They are organised into four families that each
correspond to a distinct predictive question. The choice of task families is intentional:
each is process-agnostic, can be expressed as a declarative SQL query over the canonical
OCEL tables, and covers a different predictive horizon and output type. Together they allow
the evaluation to span classification and regression targets, short and long horizons, and
single-entity and multi-entity seeds.

### Task Families

#### Next-event classification (`build_next_event_table`)

**Question asked.** At seed time `t_seed`, which event type will the seed object participate
in next?

**Construction.** For each observed (object, seed time) pair, a DuckDB query joins the
`e2o` and `event` tables, filters to events strictly after `t_seed`, and selects the one
with the smallest timestamp (ties broken by event id). The event type string is then
encoded as an integer class label against a fixed, pre-declared vocabulary. Rows where
the object has no future events — i.e. the object's lifecycle has ended before the
prediction is made — are excluded, since there is no meaningful label to assign.

**Justification.** Next-event prediction is the canonical classification task in predictive
process monitoring. It tests whether the model has learned the sequential transition
structure of the process without flattening to a single linear trace. In the object-centric
setting, the next event is not uniquely determined by a single object's history: it depends
on the joint state of all related objects visible at `t_seed`, exactly the signal that the
relational computation graph is designed to capture.

The vocabulary is fixed at dataset-registration time rather than inferred per-split. This
ensures that the class integer encoding is stable across training, validation, and test
tables, and that inverse-frequency class weights computed from training labels remain
consistent throughout training.

#### Next-time regression (`build_next_time_table`)

**Question asked.** How many days until the seed object's next event?

**Construction.** The same join as next-event is used, but instead of the event type the
query returns:

```sql
EXTRACT(epoch FROM (event_time - t_seed)) / 86400.0
```

The target is therefore elapsed time in **days** (fractional). Rows with no future events
are again excluded.

**Justification.** Next-time regression measures whether the model can estimate short-term
temporal dynamics — how busy a process entity is in the near future. Expressing the target
in days rather than raw seconds keeps values in a numerically stable range across datasets
whose absolute timestamps span months or years. For BPI2017 and BPI2019, where inter-event
gaps can be days long, a 7-day `timedelta` shifts the label window far enough ahead that
the model cannot trivially predict "zero" for entities that already have an imminent event
visible at `t_seed`. For container logistics, where cargo throughput is measured in hours,
a 12-hour shift is used instead.

For `OrderNextTime` and `OrderRemainingTime` a `QuantileTargetTransform` is additionally
applied, normalising targets to the empirical quantile scale before training and inverting
before evaluation. This reduces the effect of long-tailed distributions in the
order-management log without discarding outliers.

#### Remaining-time regression (`build_remaining_time_table`)

**Question asked.** How many days until the seed object's final observed event?

**Construction.** A window function computes `MAX(event.time)` over all future events for
the (object, seed time) pair in a single pass. Only the row with `rn = 1` is selected
(to deduplicate), and the target is:

```sql
EXTRACT(epoch FROM (last_event_time - t_seed)) / 86400.0
```

**Justification.** Remaining time is the classical PPM regression target and is directly
interpretable as an estimate of how far along a process instance is. In a case-centric
model, remaining time is unambiguous because completion is defined relative to a single
trace. In the object-centric setting, the end of a process is not owned by one entity:
it depends on the last event involving any related object. The relational neighbourhood
around the seed at `t_seed` contains evidence from all co-participating objects, so the
model has access to the full relevant context — something a flattened trace-level predictor
would lose.

#### Event-within binary classification (`build_event_within_table`, `build_pair_event_within_table`)

**Question asked (single-entity).** Will the seed object experience a specific target
event within the next `delta` days?

**Question asked (pair-entity).** Will this pair of co-observed objects jointly participate
in a specific target event within the next `delta` days?

**Construction — single entity.** The query first computes `first_seen_time` (the earliest
event for the object) and `first_target_time` (the earliest occurrence of the target event
type for that object). An object is only included as a candidate at seed time `t_seed` if
it has already appeared (`first_seen_time <= t_seed`) and has **not yet** experienced the
target event (`first_target_time IS NULL OR first_target_time > t_seed`). This censoring
is deliberate: once an object has experienced the event it is no longer at risk, and
including post-event observations would inflate the negative class. The binary label is 1
if `first_target_time` falls in the half-open window `(t_seed, t_seed + delta]`, else 0.

**Construction — observed pair.** `build_pair_event_within_table` first identifies all
(src, dst) object pairs that have **co-appeared in at least one event** before `t_seed`.
The pair's candidacy window is identical in structure to the single-entity case: the pair
must have been observed but must not yet have jointly experienced the target event. This
restricts prediction to pairs whose relationship already exists in the relational graph,
keeping the task inductive and leakage-free.

**Construction — complete pair (Order Management only).** `build_complete_pair_event_within_table`
does not require prior co-appearance: it takes the full Cartesian product of `customers ×
products` and labels each pair based on whether a `place order` event connecting them occurs
within 14 days. This turns the task into an open-world link-prediction problem — predicting
which customer will order which product — and is justified by the e-commerce semantics of
the order management log, where any customer can in principle order any product.

**Justification.** Event-within tasks model a different temporal question from next-event
and next-time: they ask about a specific future milestone over a fixed horizon rather than
the immediate next step. This is useful for operational decision-making (e.g. flagging
orders at risk of not clearing invoices in time) and captures process-level risk in a way
that regression targets do not. The censoring mechanism ensures that the task is always
forward-looking: the model is only asked to predict for entities that are genuinely at risk
at the time of prediction. Including post-event rows would introduce a degenerate class
imbalance and would not correspond to a meaningful operational query.

The pair variants are the only tasks in the evaluation that define `Kseed` as a tuple of
two object keys, instantiating the multi-entity target described in Definition 3.8 and
Section 3.3.5. They also directly test whether the model can exploit the O2O (or
co-event) relational structure in the graph: both objects in the pair must be reachable
from each other's computation graphs for the relevant signal to propagate.

---

### Per-dataset task overview

#### BPI2017 — Dutch financial institution loan application process

BPI2017 contains two interacting object types: `Application` (the loan case) and `Case_R`
(the associated risk-assessment record), linked through shared events from a Dutch bank.
The process spans loan application, offer creation, validation, and final decision. Tasks
are defined over `Application` objects for classification and `Case_R` objects for
regression, reflecting that classification targets (next event type, completion) are
semantically richer at the application level while temporal targets are more stable at the
case-review level.

| Task name | Type | Entity | Target | timedelta |
|---|---|---|---|---|
| `next_event_cases` | Multiclass (10 classes) | `Application` | Next event type out of 10 possible activity labels | 7 days |
| `next_time_cases` | Regression | `Case_R` | Days until next event | 7 days |
| `remaining_time_cases` | Regression | `Case_R` | Days until final event in lifecycle | 7 days |
| `application_completed_within_14d` | Binary | `Application` | Whether A_Accepted, A_Denied, or A_Cancelled occurs within 14 days | 14 days |
| `event_within` | Binary | `Offer` | Whether O_Cancelled occurs within 14 days | 14 days |

The `application_completed_within_14d` task uses a multi-event-type target (any of
A_Accepted, A_Denied, A_Cancelled counts as completion), reflecting that the process ends
whenever a final decision of any kind is reached.

#### BPI2019 — Dutch company purchase-to-pay process

BPI2019 is a large procurement log with `POItem` (purchase order line item) and `Vendor`
objects. The process covers purchase requisition, ordering, goods receipt, invoice receipt,
and invoice clearance, spanning over 40 distinct event types. The high cardinality of the
event vocabulary (42 classes) makes next-event classification substantially harder than in
BPI2017 and tests whether the model can distinguish fine-grained procurement sub-processes.

| Task name | Type | Entity | Target | timedelta |
|---|---|---|---|---|
| `next_event_po_items` | Multiclass (42 classes) | `POItem` | Next event type out of 42 activity labels | 14 days |
| `next_time_po_items` | Regression | `POItem` | Days until next event | 14 days |
| `remaining_time_po_items` | Regression | `POItem` | Days until final event in lifecycle | 14 days |
| `po_item_clear_invoice_within_14d` | Binary | `POItem` | Whether Clear Invoice occurs within 14 days | 14 days |
| `event_within` | Binary | (`POItem`, `Vendor`) | Whether the pair jointly reaches Clear Invoice within 14 days | 14 days |

The `event_within` pair task here is an **observed-pair** variant: only (POItem, Vendor)
combinations that have already co-appeared in at least one event are candidates, reflecting
the realistic constraint that a vendor–item relationship must have been established before
a clearance prediction is meaningful.

The 14-day `timedelta` across all BPI2019 tasks is justified by the slower pace of
procurement processes relative to, for example, loan applications or container movements:
inter-event gaps regularly span several days, so a 7-day shift would place the label
window too close to the observation time to be practically useful.

#### Order Management — synthetic e-commerce log

The order management dataset is a synthetic OCEL 2.0 log modelling an e-commerce process
with `orders`, `customers`, `products`, and `items` objects. Because the log is synthetic
it is well-structured and free of the noise found in real-world logs, making it useful for
verifying that the pipeline functions correctly and that the relational structure is
faithfully preserved. It also contains the most semantically constrained next-event
vocabulary (only 3 classes: confirm order, pay order, payment reminder), which provides
a near-upper-bound on classification accuracy for the next-event task.

| Task name | Type | Entity | Target | timedelta |
|---|---|---|---|---|
| `next_event_orders` | Multiclass (3 classes) | `orders` | Next event type: confirm order / pay order / payment reminder | 7 days |
| `next_time_orders` | Regression | `orders` | Days until next event (QuantileTargetTransform applied) | 7 days |
| `remaining_time_orders` | Regression | `orders` | Days until final event (QuantileTargetTransform applied) | 7 days |
| `event_within` | Binary | (`customers`, `products`) | Whether a place order event linking this customer–product pair occurs within 14 days | 14 days |

The `event_within` task here is a **complete-pair** variant using
`build_complete_pair_event_within_table`: every customer × every product combination is
a candidate regardless of prior co-appearance. This makes it a genuine link-prediction
problem — the model must learn which customer–product affinities exist — rather than just
predicting re-occurrence within an already-established pair. This is the only task in the
evaluation that uses a full Cartesian candidate set.

The `QuantileTargetTransform` is applied to both regression tasks here because the
order management log exhibits a notably long-tailed distribution of inter-event times;
normalising to quantiles before training and inverting before evaluation stabilises
gradient updates without discarding outlier instances.

#### Container Logistics — intermodal freight forwarding process

The container logistics dataset is a real-world OCEL 2.0 log from a Dutch freight
forwarding company. It contains `Container`, `Transport Document`, `Vehicle`, `Customer`,
and `Truck` objects interacting through nine distinct container-handling activity types.
The operational tempo is substantially faster than the financial and procurement logs:
events happen within hours rather than days, which is why all task timedeltas are set to
12 hours or 7 days rather than 7 or 14 days.

| Task name | Type | Entity | Target | timedelta |
|---|---|---|---|---|
| `next_event_containers` | Multiclass (9 classes) | `Container` | Next event type out of 9 container-handling activities | 12 hours |
| `next_time_containers` | Regression | `Container` | Days until next event | 12 hours |
| `remaining_time_containers` | Regression | `Container` | Days until final event in lifecycle | 12 hours |
| `container_depart_within_7d` | Binary | `Container` | Whether a Depart event occurs within 7 days | 7 days |
| `event_within` | Binary | (`Transport Document`, `Vehicle`) | Whether the pair jointly participates in a Depart event within 7 days | 7 days |

The 9-class next-event vocabulary covers the full set of distinct container handling steps
(Bring to Loading Bay, Depart, Drive to Terminal, Load Truck, Load to Vehicle, Pick Up
Empty Container, Place in Stock, Reschedule Container, Weigh), so the classification task
directly tests process-step sequencing at a granular operational level.

The `event_within` pair task over (Transport Document, Vehicle) is an **observed-pair**
variant: only document–vehicle combinations that have already co-appeared in an event are
candidates. This reflects the operational reality that a vehicle is only relevant to a
transport document once it has been assigned to it. Together with the BPI2019 pair task,
this provides two real-world instances of the multi-entity target formulation described in
Definition 3.8.

---

## Data Splitting

Splitting follows the temporal scheme from Section 4.3.2. RelBench materialises the target
table at `num_eval_timestamps = 1000` uniformly-spaced seed times over the observed log
timespan. The train/val/test partition is **time-based**, targeting an approximate
70/15/15 split by duration. All three splits are non-overlapping in time; no random
row-level shuffling is applied, ensuring that the evaluation simulates deployment on
future data.

Each training instance's seed time is shifted forward by the task's `timedelta` before
the REG snapshot is filtered. This ensures the label window (the future period from
which the target is computed) does not overlap with the visible evidence at the seed
time.

---

## Neighborhood Sampling

`NeighborLoader` (PyTorch Geometric) performs **temporally-filtered** neighbor sampling.
At layer `i`, the number of sampled neighbors is:

```
num_neighbors[i] = num_neighbors // 2^i
```

which gives 128, 64, 32 for the default 3-layer setting. The loader uses `time_attr="time"`
to enforce the temporal cut at each seed time, corresponding to the `N_≤t_seed`
neighborhood restriction in Algorithm 1. The `temporal_strategy` is `uniform` — past
neighbors are sampled uniformly at random.

---

## Baselines

The baseline runner (`scripts/baseline/core.py`) computes **statistic-only** predictors
fit from the training table alone, with no access to the graph structure. For test
evaluation, baselines are re-fit on the combined train+val table. All baselines use
seed 42.

### Regression baselines

| Name | Description |
|---|---|
| `global_zero` | Predicts 0 for all instances |
| `global_mean` | Predicts the training-set mean |
| `global_median` | Predicts the training-set median |
| `entity_mean`* | Predicts the per-entity mean from training; falls back to global mean for unseen entities |
| `entity_median`* | Predicts the per-entity median from training; falls back to global median for unseen entities |

### Binary classification baselines

| Name | Description |
|---|---|
| `random` | Uniform random score in [0, 1] |
| `majority` | Always predicts the majority class |
| `entity_mean`* | Per-entity positive rate from training |
| `entity_median`* | Per-entity median from training |

### Multiclass classification baselines

| Name | Description |
|---|---|
| `random` | Uniform random class probabilities |
| `majority` | Always predicts the globally most frequent class |
| `entity_majority_multiclass`* | Predicts each entity's most frequent class from training |

\* Marked as leaky (`LEAKY_BASELINES`) in the codebase because they condition on the
entity's own historical labels — these are reported separately from the non-leaky
baselines.
