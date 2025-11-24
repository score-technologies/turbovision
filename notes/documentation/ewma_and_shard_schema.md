# ScoreVision Shard Schema & EWMA Integration

This document describes a clean shard schema for R2-based ingestion and how to integrate EWMA persistence and updates into the window score storage pipeline.

## 1. Shard Schema (Clean Version)

A **shard** represents one miner's scores for one element for one *clip* inside a window. Shards are JSONL entries stored in R2 (local and remote) and later aggregated.

### 1.1 Shard Identity

Each shard is uniquely identified by:

* `window_id: str` — e.g., "2025-01-14"
* `miner_id: str`
* `element_id: str` — e.g., "PlayerDetect_v1@1.0"
* `clip_hash: str` — stable SHA-256 identifying the video fragment
* `source: str` — `"local"` or validator ID, used only for deduplication (not part of uniqueness key)

### 1.2 Shard Payload Schema

```json
{
  "window_id": "2025-01-14",
  "miner_id": "miner_abc",
  "element_id": "PlayerDetect_v1@1.0",
  "clip_hash": "sha256:abcd...",

  "pillar_scores": {                 
      "iou": 0.88,
      "count": 0.92,
      "palette": 0.75
  },

  "pillar_weighted_scores": {
      "iou": 0.44,
      "count": 0.37,
      "palette": 0.19
  },

  "total_raw": 2.55,                 
  "total_weighted": 1.00,            
  "total_weighted_and_gated": 0.73,  

  "ingest_ts": 1736892012,           
  "source": "local"                 
}
```

### 1.3 Validation

* Pillar weights guaranteed to sum to **1.0** (enforced in Manifest.Metrics).
* Element-level shard scores do **not** need to sum to 1; only pillar weights do.
* Window-level scores will later be averaged across clips → these feed EWMA.

---

## 2. Window Score Persistence (window_scores.py)

This module reads all shards for a window from R2 caches (local + remote), filters duplicates, aggregates per-miner/per-element clip means, and writes window-level summaries.

### 2.1 Window Summary Schema (JSONL)

One JSON object per miner × element:

```json
{
  "window_id": "2025-01-14",
  "miner_id": "miner_abc",
  "element_id": "PlayerDetect_v1@1.0",

  "clip_means": {
      "total_raw": 1.23,
      "total_weighted": 0.52,
      "total_weighted_and_gated": 0.38
  },

  "ewma": {
      "previous": 0.41,
      "updated": 0.44,
      "alpha": 0.2063
  }
}
```

### 2.2 Duplicate Handling

Two shards are duplicates if:

```
(window_id, miner_id, element_id, clip_hash) match
```

Use the lexicographically smallest source ("local" < validator IDs) or earliest `ingest_ts` as the tie-breaker.

### 2.3 Completion Detection

A window is considered *complete* if:

* All shards for the window have been received from local ingestion, *or*
* A validator-provided remote manifest indicates completeness (Phase 3), *or*
* A configurable timeout is reached.

Then `save_window_scores()` is triggered.

---

## 3. EWMA Integration

EWMA is applied **after** all clip scores for a miner/element have been averaged for a window.

### 3.1 EWMA Update

For each miner × element:

```
current = clip_mean.total_weighted_and_gated
previous = load_previous_ewma(miner, element)
S_e,t = α * current + (1-α) * previous
```

If no history exists, `previous = None` → `S_e,t = current`.

### 3.2 Where EWMA Lives

The updated EWMA scores are saved in the **window summary** JSONL entry under:

```
"ewma": {
    "previous": ...,   # None allowed
    "updated": ...,    # new S_e,t
    "alpha": 0.2063
}
```

Validators reading current window scores always use the **updated** field.

### 3.3 Alpha Calculation

Configured via settings (defaults to half-life 3 windows).

```
from scorevision.utils.ewma import calculate_ewma_alpha, update_ewma_score
```

---

## 4. Integration Hooks (Pipeline)

1. **Shards Ingested** → local and remote caches populated.
2. **window_scores.aggregate_from_shards(window_id)**

   * dedupe shards
   * group by miner, element
   * compute per-window clip means
3. **EWMA Application**

   * load previous EWMA state from prior window summaries
   * compute updated EWMA
4. **window_scores.save_window_scores()**

   * persist JSONL summaries
   * these become inputs for validator `get_weights()`

---

## 5. Status

Schema, integration points, and EWMA responsibilities defined.
Ready for code implementation in `scorevision/utils/window_scores.py`.
