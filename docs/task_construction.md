# Task Construction in OCEL-OCP

## 1. Background: From Event Log to Relational Database

An OCEL (Object-Centric Event Log) records business process history as a set of **events** (timestamped activities) and **objects** (entities involved in those activities), connected by **event-to-object (e2o)** links. This repo parses OCEL files into a small relational database with four canonical tables:

| Table | What it stores |
|---|---|
| `event` | One row per event: `event_id`, `time`, `type` |
| `object` | One row per object: `object_id`, `type` |
| `e2o` | Many-to-many links: which objects participated in which event |
| `o2o` | Optional direct object-to-object relations |

This is the input that every task builder queries.

---

## 2. What is a Task?

A **task** is a formal prediction problem. It defines:

1. **What to predict** (the target) — a class, a number, a binary flag
2. **For whom** (the entity or entity-pair) — one or two objects
3. **When** (the observation times) — a set of cut-off timestamps
4. **Evaluation splits** — train / val / test, determined by time

The output of a task is a table of labelled rows:

```
(entity_id, [partner_id], observation_time, target)
```

This table is used both to **train** a model and to **evaluate** it.

---

## 3. The Two Entity Modes

### 3a. Single-Entity Tasks

These tasks ask a question about **one object** at a time, e.g.:
*"For container C42, observed at time T, what will happen next?"*

The entity column is `object_id`. There is exactly one foreign key pointing into the `object` table.

**Pseudo-code for construction:**

```
given: object_type, observation_times T, look-back window B, forward window F

for each t in T:
    active_objects = {
        o in objects of type object_type
        | o participated in some event in (t - B, t]
    }
    for each o in active_objects:
        label = compute_target(o, t, future_window=(t, t+F])
        if label is not null:
            emit row (object_id=o, time=t, target=label)
```

The `active_objects` filter is the "has this entity been seen recently?" gate. It prevents predicting on dormant objects that have no context in the look-back window.

---

### 3b. Multi-Entity (Pair) Tasks

These tasks ask a question about a **pair of objects**, e.g.:
*"Will container C42 and truck T09 co-appear in a loading event within 14 days?"*

There are now two entity columns: `object_id` (source) and `object_id_partner` (destination), each a foreign key into the `object` table.

**Pseudo-code for construction:**

```
given: src_type, dst_type, observation_times T, look-back window B, forward window F

for each t in T:
    observed_pairs = {
        (src, dst)
        | src is of type src_type
        AND dst is of type dst_type
        AND src ≠ dst
        AND src and dst co-appeared in the same event in (t - B, t]
    }
    for each (src, dst) in observed_pairs:
        label = check_co_appearance(src, dst, future_window=(t, t+F])
        emit row (object_id=src, object_id_partner=dst, time=t, target=label)
```

The pair enumeration also acts as a "has this relationship been active recently?" filter.

---

## 4. The Six Task Families

| Family | Type | Target definition | Forward window |
|---|---|---|---|
| **Next event** | Multiclass | Type of the very next event for the object | bounded (default 14d) |
| **Next time** | Regression | Days until the next event | bounded (default 30d) |
| **Remaining time** | Regression | Days until the *last* future event (cycle completion) | unbounded |
| **Event within window** | Binary | Does a specific event occur within the window? | bounded (default 14d) |
| **Pair interaction** | Binary (pair) | Will these two objects co-appear in a target event? | bounded (default 14d) |
| **Next co-object** | Multiclass (pair) | Which partner object appears with src in the next event? | bounded (default 14d) |

---

## 5. How the SQL Builders Work

All six builders follow the same three-step SQL pattern, executed in batches of 50 timestamps at a time to keep intermediate join sizes manageable.

### Step 1 — Filter active entities

```sql
obs AS (
  SELECT obs_time AS time, object_id
  FROM times_df CROSS JOIN (SELECT object_id FROM obj WHERE type = '<type>')
  WHERE EXISTS (
    SELECT 1 FROM e2o JOIN event USING (event_id)
    WHERE object_id = object_id
      AND event.time > obs_time - INTERVAL (back_seconds) SECOND
      AND event.time <= obs_time
  )
)
```

This gives the set of (entity, timestamp) pairs that are candidates for labelling.

### Step 2 — Look forward and compute the label

**Next event (multiclass):**
```sql
future AS (
  SELECT object_id, time,
         ROW_NUMBER() OVER (PARTITION BY object_id, time ORDER BY event.time ASC) AS rn,
         event.type AS etype
  FROM obs JOIN e2o JOIN event ...
  WHERE event.time > obs.time AND event.time <= obs.time + INTERVAL (fwd_s) SECOND
)
-- keep only rn = 1, map etype → integer class
```

**Next time (regression):**
```sql
SELECT object_id, time,
       (MIN(event.time) - obs.time) / 86400.0 AS target
FROM obs JOIN e2o JOIN event ...
WHERE event.time > obs.time AND event.time <= obs.time + INTERVAL (fwd_s) SECOND
GROUP BY object_id, time
```

**Remaining time (regression):**
```sql
SELECT object_id, time,
       (MAX(event.time) - obs.time) / 86400.0 AS target
FROM obs JOIN e2o JOIN event ...
WHERE event.time > obs.time   -- NO forward bound
GROUP BY object_id, time
```

### Step 3 — Wrap in a RelBench Table

The resulting DataFrame `(entity_col, time_col, "target")` is wrapped in a `relbench.Table` with explicit foreign-key metadata mapping `entity_col → object` table. For pair tasks, both `object_id` and `object_id_partner` are registered as foreign keys into `object`.

---

## 6. From Question to Sampling to Prediction

Here is the end-to-end flow that connects a business question to a trained model:

```
Business question
      │
      ▼
Task definition
  (object type, event types, window sizes)
      │
      ▼
Observation timestamps
  (1,000 timestamps spread across train / val / test periods)
      │
      ▼
Label construction (SQL builder)
  → For each (entity, timestamp): compute target from future events
  → Filter: only entities active in look-back window
  → Split: rows before val_timestamp → train, before test_timestamp → val, rest → test
      │
      ▼
RelBench Table  (entity_id, time, target)
      │
      ▼
Relational Deep Learning
  The table is joined back into the full relational database.
  A GNN traverses the relational graph: object attributes,
  event attributes, e2o links, o2o links, and per-type tables
  are all available as edge types and node features.
  The model sees everything in the database up to (and not past) obs_time.
      │
      ▼
Prediction at deployment time
  Given a live object at the current timestamp, the same
  feature graph is constructed (using only past data),
  passed through the trained model, and a prediction is returned.
```

The key guarantee is **temporal correctness**: the observation time `t` acts as a hard cut-off. Labels come from strictly after `t`; all features come from at or before `t`. The look-back window `B` further narrows features to the recent past, reflecting realistic operational conditions.

---

## 7. Registered Tasks (all 32)

### 7a. Classification Tasks (ROC-AUC reported)

| Dataset | Task | Type | Classes | Entity |
|---|---|---|---|---|
| BPI2017 | `application_next_event` | Multiclass | 10 | Application |
| BPI2017 | `offer_next_event` | Multiclass | 8 | Offer |
| BPI2017 | `application_denied_within_14d` | Binary | 2 | Application |
| BPI2017 | `application_offer_pair_interaction` | Binary | 2 | Application × Offer |
| BPI2019 | `po_item_next_event` | Multiclass | varies | PO Item |
| BPI2019 | `po_next_srm_event` | Multiclass | varies | Purchase Order |
| BPI2019 | `po_item_blocked_within_14d` | Binary | 2 | PO Item |
| BPI2019 | `po_item_vendor_pair_interaction` | Binary | 2 | PO Item × Vendor |
| Container logistics | `container_next_event` | Multiclass | 9 | Container |
| Container logistics | `transport_doc_next_event` | Multiclass | 5 | Transport Document |
| Container logistics | `container_rescheduled_within_14d` | Binary | 2 | Container |
| Container logistics | `container_truck_pair_interaction` | Binary | 2 | Container × Truck |
| Order management | `order_next_event` | Multiclass | 9 | Order |
| Order management | `package_next_delivery_event` | Multiclass | 3 | Package |
| Order management | `order_stockout_within_14d` | Binary | 2 | Order |
| Order management | `order_employee_pair_interaction` | Binary | 2 | Order × Employee |

**Classification baselines (ROC-AUC on test set):**

| Dataset | Task | Random | Majority |
|---|---|---|---|
| BPI2017 | `application_next_event` | — | — |
| BPI2017 | `offer_next_event` | — | — |
| BPI2017 | `application_denied_within_14d` | — | — |
| BPI2017 | `application_offer_pair_interaction` | — | — |
| BPI2019 | `po_item_next_event` | — | — |
| BPI2019 | `po_next_srm_event` | — | — |
| BPI2019 | `po_item_blocked_within_14d` | — | — |
| BPI2019 | `po_item_vendor_pair_interaction` | — | — |
| Container logistics | `container_next_event` | — | — |
| Container logistics | `transport_doc_next_event` | — | — |
| Container logistics | `container_rescheduled_within_14d` | — | — |
| Container logistics | `container_truck_pair_interaction` | — | — |
| Order management | `order_next_event` | — | — |
| Order management | `package_next_delivery_event` | — | — |
| Order management | `order_stockout_within_14d` | — | — |
| Order management | `order_employee_pair_interaction` | — | — |

### 7b. Regression Tasks (MAE in days reported)

| Dataset | Task | Entity |
|---|---|---|
| BPI2017 | `application_next_time` | Application |
| BPI2017 | `case_next_time` | Case_R |
| BPI2017 | `application_remaining_time` | Application |
| BPI2017 | `offer_remaining_time` | Offer |
| BPI2019 | `po_item_next_time` | PO Item |
| BPI2019 | `po_next_time` | Purchase Order |
| BPI2019 | `po_item_remaining_time` | PO Item |
| BPI2019 | `po_remaining_time` | Purchase Order |
| Container logistics | `container_next_time` | Container |
| Container logistics | `customer_order_next_time` | Customer Order |
| Container logistics | `container_remaining_time` | Container |
| Container logistics | `customer_order_remaining_time` | Customer Order |
| Order management | `order_next_time` | Order |
| Order management | `item_next_time` | Item |
| Order management | `order_remaining_time` | Order |
| Order management | `package_remaining_time` | Package |

**Regression baselines (MAE in days on test set):**

| Dataset | Task | Global zero | Global mean | Global median |
|---|---|---|---|---|
| BPI2017 | `application_next_time` | — | — | — |
| BPI2017 | `case_next_time` | — | — | — |
| BPI2017 | `application_remaining_time` | — | — | — |
| BPI2017 | `offer_remaining_time` | — | — | — |
| BPI2019 | `po_item_next_time` | — | — | — |
| BPI2019 | `po_next_time` | — | — | — |
| BPI2019 | `po_item_remaining_time` | — | — | — |
| BPI2019 | `po_remaining_time` | — | — | — |
| Container logistics | `container_next_time` | — | — | — |
| Container logistics | `customer_order_next_time` | — | — | — |
| Container logistics | `container_remaining_time` | — | — | — |
| Container logistics | `customer_order_remaining_time` | — | — | — |
| Order management | `order_next_time` | — | — | — |
| Order management | `item_next_time` | — | — | — |
| Order management | `order_remaining_time` | — | — | — |
| Order management | `package_remaining_time` | — | — | — |

> **Note:** Baseline values are not yet computed. Run `python -m scripts.baseline.baseline_entity --dataset <name> --all-tasks` to populate these. The entity-mean / entity-median / entity-majority baselines are intentionally excluded: they use per-entity statistics derived from the training split and would leak distributional information about test entities.

---

## 8. Key Design Properties

**Temporal correctness.** The observation time `t` is a strict boundary: labels come from after `t`, all features from at or before `t`. No future information leaks into model inputs.

**Activity filter (look-back window).** Only objects that participated in at least one event within the past 30 days of `t` are included. This mirrors operational reality — you only predict on cases that are currently active.

**Bounded vs unbounded forward window.** Next-event and event-within tasks use a bounded forward window (14 days). Remaining-time tasks use no forward bound — the target is the time until the *last ever* event for that object, which is the standard "case completion" definition from process mining.

**Pair tasks extend RelBench naturally.** Standard RelBench assumes one entity column. Pair tasks add a second foreign key (`object_id_partner`) pointing to the same `object` table. The `MEntityTask` base class handles multiple entity columns uniformly through `entity_cols` and `entity_tables` tuples.
