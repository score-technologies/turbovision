# Score Vision Protocol Implementation Plan

## Moving Toward test.md Methodology

**Goal:** Implement foundational infrastructure aligning current flow with test.md protocol specification

---

## Current State Analysis

### Existing Components ✅

- **Runner**: Executes challenges, builds PGT, calls miners, evaluates, emits shards
- **Validator**: Aggregates shards, performs MAD outlier detection, sets weights on-chain
- **Miner**: Receives challenges via Chutes API, returns predictions
- **Evaluation**: Multi-pillar scoring (IoU, count, team, smoothness, keypoints)
- **Shard Emission**: Basic shard structure with signatures

### Critical Gaps ❌

1. **Manifest System**: No cryptographic manifest per evaluation window
2. **Cryptographic Salting**: No per-validator VRF-based challenge salting
3. **RTF Framework**: Latency measured but not proper RTF calculation with p95 gates
4. **Element Architecture**: Not structured as Elements with formal contracts
5. **Baseline Gates (θ)**: Missing baseline thresholds for reward eligibility
6. **Difficulty Weights (β)**: Missing difficulty multipliers per Element
7. **EWMA**: No temporal smoothing across evaluation windows
8. **Shard Schema v1.3**: Current schema doesn't match specification
9. **Telemetry**: Missing comprehensive telemetry requirements
10. **Window Management**: No explicit evaluation window lifecycle

---

## Implementation Tasks (Logical Order)

### Phase 1: Manifest System Foundation

**Prerequisites:** None  
**Dependencies:** None

- [ ] Create `scorevision/utils/manifest.py` module

  - [ ] Define `Manifest` dataclass with all required fields:
    - `window_id`, `version`, `expiry_block`
    - `elements[]` with Element specifications
    - `metrics` weights per pillar
    - `latency_p95_ms`, `service_rate_fps` per Element
    - `pgt_recipe_hash` (SHA-256)
    - `baseline_theta`, `delta_floor`, `beta` per Element
    - `preproc` parameters (fps, resize, normalization)
  - [ ] Implement Manifest signing/verification using Ed25519
  - [ ] Implement content-addressed storage (SHA-256 hash as identifier)
  - [ ] Implement Manifest distribution via R2/CDN (public access, no auth required)
    - Upload path: `scorevision/manifests/{manifest_hash}.json`
    - Index file: `scorevision/index.json` with current manifests
  - [ ] `fetch_manifest(manifest_hash)` - fetch by hash
  - [ ] `get_current_manifest(block_number)` - get active manifest
  - [ ] `fetch_index()` - get manifest index
  - [ ] `verify_manifest_signature(manifest)` - verify Ed25519 signature

- [ ] Create `scorevision/cli/manifest.py` CLI tool for Score operators
  - [ ] `sv manifest create` - create from template
  - [ ] `sv manifest validate` - validate config before publishing
  - [ ] `sv manifest publish` - sign, hash, upload to R2, update index
  - [ ] `sv manifest list` - list published manifests
  - [ ] `sv manifest current` - get current active manifest
  - [ ] **See `MANIFEST_MANAGEMENT.md` for detailed operator guide**

**Deliverable:** Manifest system with signing/verification and public distribution

---

### Phase 2: Challenge API Integration

**Prerequisites:** Phase 1 (Manifest system)  
**Dependencies:** Manifest fetching must work

- [ ] Create new function `get_next_challenge_v3()` in `scorevision/utils/challenges.py`

  - [ ] New endpoint: `GET /api/challenge/v3` (separate from existing)
  - [ ] Add `manifest_hash` parameter (fetch current if not provided)
  - [ ] Add `X-Manifest-Hash` header to API request
  - [ ] Handle new response schema with `element_id`, `window_id`, `manifest_hash`
  - [ ] Handle new error codes: 404 (no window), 409 (rate limit), 410 (expired)
  - [ ] Extract `element_id` and `window_id` from response
  - [ ] Verify `manifest_hash` matches request
  - [ ] **Keep existing `get_next_challenge()` unchanged** (backward compatibility)

- [ ] Update runner to support both endpoints via config
  - [ ] Add env variable: `SCOREVISION_USE_CHALLENGE_V3` (default: "0")
  - [ ] If v3 enabled: Call `get_current_manifest()` at start of run
  - [ ] If v3 enabled: Use `get_next_challenge_v3()` with `manifest_hash`
  - [ ] If v3 enabled: Verify `window_id` matches Manifest
  - [ ] If legacy: Use existing `get_next_challenge()` (no changes)

**Deliverable:** Challenge API client updated to use Manifest hash  
**See:** `MANIFEST_AND_CHALLENGE_API.md` for detailed API changes

---

### Phase 3: Window Management

**Prerequisites:** Phase 1 (Manifest system)  
**Dependencies:** Manifest must have `window_id` and `expiry_block`

- [ ] Create `scorevision/utils/windows.py` module

  - [ ] `get_current_window_id(block_number, tempo=300)` → window ID string
  - [ ] `is_window_active(window_id, current_block, expiry_block)` → bool
  - [ ] `get_window_start_block(window_id, tempo)` → block number
  - [ ] Window ID format: `"YYYY-MM-DD"` or `"block-{start_block}"`

- [ ] Integrate window management into runner

  - [ ] Derive `window_id` at start of each run (from Manifest or block number)
  - [ ] Check window expiry before processing
  - [ ] Include `window_id` in all shards

- [ ] Update validator to track windows
  - [ ] Group shards by `window_id` during aggregation
  - [ ] Handle window transitions gracefully

**Deliverable:** Evaluation window lifecycle management

---

### Phase 4: Miner Element Discovery & Commitments

**Prerequisites:** Phase 1 (Manifest system), Phase 2 (Challenge API), Phase 3 (Window management)  
**Dependencies:** Manifest index exposes current + upcoming windows

- [ ] Create `scorevision/utils/element_catalog.py`

  - [ ] `list_elements(window_scope="current" | "upcoming")` → aggregates Element metadata from Manifest index
  - [ ] `summarize_window(window_id)` → returns Elements, service rates, theta, β, and available clip counts
  - [ ] Cache manifests locally to avoid redundant CDN hits

- [ ] Build miner-focused CLI: `scorevision/cli/miner.py`

  - [ ] `sv miner elements --window current` → show Elements for active window plus telemetry requirements
  - [ ] `sv miner elements --window upcoming` → show next scheduled window (if index reports one)
  - [ ] `sv miner manifest --hash <hash>` → download + verify manifest locally for offline planning
  - [ ] `sv miner commitments list` → display on-chain commitments per Element

- [ ] Implement on-chain commitment helper in `scorevision/utils/commitments.py`

  - [ ] `commit_to_elements(hotkey, window_id, elements: list[ElementCommitment])` where each commitment records `element_id`, `hf_revision`, `chute_slug`, `chute_id`, and target service cap
  - [ ] `withdraw_commitment(...)` for exiting Elements before window start
  - [ ] Store commitment proof (extrinsic hash + block) for validators to reference when scoring

- [ ] Integrate commitments into runner/miner workflow

  - [ ] Runner filters requested challenges to committed `element_id`s (pass `element_id` to Challenge API v3)
  - [ ] Add health check ensuring miner committed for both current and upcoming window before pulling work
  - [ ] Include commitment reference + declared `chute_slug` in shard telemetry for validator auditing

**Deliverable:** Miners can inspect available Elements for current/upcoming windows and declare participation on-chain before requesting challenges.

---

### Phase 5 (Deferred): Cryptographic Salting

**Prerequisites:** Phase 1 (Manifest system), Phase 2 (Challenge API)  
**Dependencies:** Need `manifest_hash` and `element_id` from challenge  
**Status:** Deferred until after the initial manifest → RTF → EWMA rollout. This work should not block Phases 6-11 and can be scheduled once the rest of the pipeline is stable.

- [ ] Implement VRF/PRF salting in `scorevision/utils/salting.py`

  - [ ] `derive_salt(validator_sk, manifest_hash, element_id, clip_id, challenge_seq)`
  - [ ] Use Ed25519 VRF or BLS (start with Ed25519 for simplicity)
  - [ ] Deterministic salt → frame offset/stride sampling
  - [ ] Salt proof generation for shard inclusion

- [ ] Integrate salting into runner's challenge preparation
  - [ ] Apply validator-specific salt before PGT generation
  - [ ] Store salt_id in shards for verification

**Deliverable:** Cryptographic salting integrated into runner

---

### Phase 6: RTF Framework

**Prerequisites:** Phase 1 (Manifest system)  
**Dependencies:** Need `service_rate_fps` from Manifest

- [ ] Create `scorevision/utils/rtf.py` module

  - [ ] `calculate_rtf(p95_latency_ms, service_rate_fps)` function
    - Formula: `RTF = (t_p95_ms / 1000) × (r_e / 5)`
  - [ ] `check_rtf_gate(rtf_value)` → returns bool (RTF ≤ 1.0)
  - [ ] `get_service_rate(element_id)` → reads from Manifest

- [ ] Update `SVRunOutput` to track p95 latency (currently only mean)

  - [ ] Modify `call_miner_model_on_chutes()` to collect latency distribution
  - [ ] Calculate p50, p95, p99, max from multiple runs or single batch timing

- [ ] Integrate RTF gate into evaluation
  - [ ] Update `post_vlm_ranking()` to check RTF before scoring
  - [ ] Set `score = 0` if RTF gate fails (regardless of accuracy)
  - [ ] Add `latency_pass` field to `SVEvaluation`

**Deliverable:** RTF framework with p95 latency gates

---

### Phase 7: Multi-Pillar Weighted Scoring

**Prerequisites:** Phase 1 (Manifest system), Phase 6 (RTF framework)  
**Dependencies:** Need metric weights from Manifest

- [ ] Update `post_vlm_ranking()` in `scorevision/utils/evaluate.py`
  - [ ] Read metric weights from Manifest (not hardcoded 0.5/0.5)
  - [ ] Apply Element-specific pillar weights:
    - `iou_placement`, `count_accuracy`, `palette_symmetry`, `smoothness`, `role_consistency`
  - [ ] Calculate weighted composite score: `Σ(weight_i × pillar_i)`
  - [ ] Map current metric names to Manifest pillar names
  - [ ] Ensure all pillars are computed (verify completeness)

**Deliverable:** Multi-pillar weighted scoring from Manifest

---

### Phase 8: Baseline Gates & Difficulty Weights

**Prerequisites:** Phase 1 (Manifest system), Phase 7 (Weighted scoring)  
**Dependencies:** Need `baseline_theta` and `beta` from Manifest

- [ ] Create `scorevision/utils/economics.py` module

  - [ ] `apply_baseline_gate(score, baseline_theta)` → `max(score - theta, 0)`
  - [ ] `apply_difficulty_weight(improvement, beta)` → `beta × improvement`
  - [ ] `calculate_improvement(score, baseline_theta, delta_floor)` → `max(score - theta, delta_floor)`

- [ ] Integrate into validator aggregation (`scorevision/cli/validate.py`)
  - [ ] Read `baseline_theta` and `beta` from Manifest (or config for now)
  - [ ] Apply baseline gates before aggregation
  - [ ] Apply difficulty weights in final weight calculation
  - [ ] Handle burn routing for Elements with Q_e = 0

**Deliverable:** Baseline gates and difficulty weights integrated

---

### Phase 9: EWMA Temporal Smoothing

**Prerequisites:** Phase 3 (Window management), Phase 8 (Baseline gates)  
**Dependencies:** Need window_id and score storage, plus the existing R2 shard replication pipeline that lets every validator download every shard (`dataset_sv` + `dataset_sv_multi`).

- [ ] Align smoothing inputs with the current R2-based shard flow

  - [ ] Treat the JSONL shards persisted to each validator's R2 bucket as the source of truth; validators continue to fetch *all* shards (local and remote) before weighting.
  - [ ] Reuse `scorevision/utils/cloudflare_helpers.dataset_sv_multi()` so EWMA operates on the same cross-validator payload set that the weighting loop already consumes.
  - [ ] Document how validators should avoid double-counting shards when both local cache and remote fetch contain the same payload.

- [ ] Create `scorevision/utils/ewma.py` module

  - [ ] `calculate_ewma_alpha(half_life_windows)` → `1 - 2^(-1/h)`
  - [ ] `update_ewma_score(current_score, previous_ewma, alpha)` → new EWMA
  - [ ] Default half-life: `h = 3` windows → `α ≈ 0.2063`

- [ ] Implement window-level score storage that sits on top of the R2-ingested shards

  - [ ] Create `scorevision/utils/window_scores.py`
  - [ ] Store per-miner, per-Element scores by window_id once all shards for that window have been pulled from every validator index
  - [ ] Persist to JSONL or database (start with JSONL in cache dir) reusing the existing cache layout under `scorevision/utils/cloudflare_helpers`
  - [ ] `load_window_scores(miner_hotkey, element_id, window_id)` → score
  - [ ] `save_window_scores(miner_hotkey, element_id, window_id, score)`

- [ ] Integrate EWMA into validator aggregation
  - [ ] Update `get_weights()` in `validate.py`
  - [ ] Load previous EWMA scores for each miner
  - [ ] Calculate clip-level mean for current window after consolidating shards fetched from local + remote R2 indexes
  - [ ] Apply EWMA: `S_e,t = α × ClipMean_e,t + (1 - α) × S_e,t-1`
  - [ ] Save updated EWMA scores for next window

**Deliverable:** EWMA temporal smoothing across windows

---

### Phase 10: Shard Schema v1.3 Migration

**Prerequisites:** Phase 1-9 (All previous phases)  
**Dependencies:** Need all fields from previous phases

- [ ] Update `emit_shard()` in `scorevision/utils/cloudflare_helpers.py`
  - [ ] Migrate to v1.3 schema structure:
    ```json
    {
      "window_id": "...",
      "validator": "ss58:...",
      "element_id": "PlayerDetect_v1@1.0",
      "lane": "public",
      "manifest_hash": "sha256:...",
      "pgt_recipe_hash": "sha256:...",
      "salt_id": "uint64",
      "metrics": {...},
      "composite_score": 0.88,
      "latency_pass": true,
      "p95_latency_ms": 178,
      "telemetry": {...},
      "signature": "nacl:..."
    }
    ```
  - [ ] Add `window_id` from window management
  - [ ] Add `element_id` from challenge response
  - [ ] Add `manifest_hash` from fetched Manifest
  - [ ] Add `pgt_recipe_hash` (hash of PGT generation code/config)
  - [ ] Reserve `salt_id` field (can remain `null`/`0` until deferred Phase 5 lands)
  - [ ] Restructure `metrics` to match pillar names
  - [ ] Add `latency_pass` boolean (RTF ≤ 1.0)
  - [ ] Expand `telemetry` section

**Deliverable:** Shard schema migrated to v1.3 format

---

### Phase 11: Telemetry Expansion

**Prerequisites:** Phase 10 (Shard schema)  
**Dependencies:** Shard schema must support telemetry

- [ ] Expand telemetry collection in `SVRunOutput`

  - [ ] Add `telemetry` dataclass with:
    - `gpu_mem_mb_peak`, `cpu_pct_peak`
    - `jitter_ms` (latency variance)
    - `frames_egress` (boolean, should be false)
    - `encoder_reuse_ratio` (for Agents)
    - `id_stability_rate` (for tracking)
  - [ ] Update miner call sites to collect telemetry

- [ ] Update shard schema to include full telemetry
  - [ ] Ensure `telemetry` section matches v1.3 spec

**Deliverable:** Expanded telemetry collection and shard inclusion

---

## Integration Points

### Critical Dependencies

1. **Manifest → Miner Catalog → Commitments**

   - Manifest/index drive the element catalog exposed to miners
   - Miners inspect current/upcoming Elements via CLI before committing on-chain, mapping each `element_id` to a dedicated HF revision/chute slug
   - Runner requests challenges only for committed `element_id`s and proves commitment + chute slug in shards

2. **Manifest → RTF → Evaluation**

   - Manifest provides Element specs (service rates, metric weights)
   - RTF reads service rates from Manifest
   - Evaluation reads metric weights from Manifest

3. **Window → EWMA → Aggregation**

   - Window management provides window_id
   - EWMA uses window_id for score storage
   - Aggregation groups by window_id and applies EWMA

4. **Shard Schema Consistency**
   - All phases contribute fields to shard
   - Manifest provides: `manifest_hash`, `element_id` (via challenge)
   - Window provides: `window_id`
   - (Deferred) Salting will eventually populate `salt_id`; until then the field is reserved
   - RTF provides: `latency_pass`, `p95_latency_ms`
   - Evaluation provides: `metrics`, `composite_score`
   - Telemetry provides: `telemetry` section

---

## Testing Checklist

- [ ] Manifest can be created, signed, and verified
- [ ] Manifest can be fetched publicly from R2/CDN
- [ ] Miner CLI lists current and upcoming Elements from manifest index
- [ ] On-chain commitment helper records element_id/window_id pairs and runner enforces them
- [ ] Commitment payload stores `{element_id, hf_revision, chute_slug, chute_id}` and runner/shards surface the same slug
- [ ] Challenge API v3 accepts `X-Manifest-Hash` header
- [ ] Challenge API v3 returns `element_id` and `window_id`
- [ ] (Deferred) Salting produces deterministic but validator-specific results
- [ ] RTF calculation matches formula: `(p95_ms/1000) × (r_e/5)`
- [ ] RTF gate correctly zeros scores when RTF > 1.0
- [ ] Multi-pillar weights are read from Manifest and applied
- [ ] Baseline gates prevent rewards below theta
- [ ] Difficulty weights multiply improvement scores
- [ ] EWMA correctly smooths scores across windows
- [ ] Window transitions don't break aggregation
- [ ] Shard schema validates against v1.3 structure
- [ ] Validator can aggregate shards with new schema
- [ ] Legacy endpoint still works (backward compatibility)

---

## File Structure Changes

### New Files

```
scorevision/utils/manifest.py          # Manifest system (fetch, verify, distribute)
scorevision/utils/element_catalog.py   # Miner-facing element catalog from manifests
scorevision/utils/salting.py          # Cryptographic salting (deferred phase)
scorevision/utils/rtf.py               # RTF framework
scorevision/utils/economics.py        # Baseline gates, difficulty weights
scorevision/utils/windows.py           # Window management
scorevision/utils/ewma.py              # EWMA calculations
scorevision/utils/window_scores.py    # Window score persistence
scorevision/utils/commitments.py       # On-chain miner commitments per window
scorevision/cli/manifest.py            # Manifest management CLI
scorevision/cli/miner.py               # Miner CLI for element discovery/commitments
MANIFEST_AND_CHALLENGE_API.md          # Detailed manifest distribution guide
MANIFEST_MANAGEMENT.md                 # Score operator guide for managing manifests
GOLD_TRUTH_INTEGRATION.md              # Guide for integrating Canary Gold annotations
```

### Modified Files

```
scorevision/cli/runner.py              # Integrate manifest, window_id, RTF gates (salting hook later)
scorevision/cli/validate.py           # Integrate EWMA, baseline gates, difficulty weights
scorevision/utils/evaluate.py          # RTF gates, weighted scoring from Manifest
scorevision/utils/cloudflare_helpers.py # Shard schema v1.3 migration
scorevision/utils/challenges.py        # Add get_next_challenge_v3()
scorevision/utils/data_models.py        # Add telemetry, window_id fields
scorevision/utils/predict.py           # Collect p95 latency, telemetry
```

---

## Success Criteria

By completion, the system should:

1. ✅ **Manifest-Driven**: All evaluation parameters come from signed Manifest
2. (Deferred) **Cryptographically Secure**: Per-validator salting prevents pre-computation once Phase 5 ships
3. ✅ **Miner-Ready**: Miners can inspect current/upcoming Elements and commit on-chain before requesting work
4. ✅ **RTF-Compliant**: Only solutions meeting RTF ≤ 1.0 receive rewards
4. ✅ **Properly Weighted**: Multi-pillar metrics use Manifest-defined weights
5. ✅ **Economically Sound**: Baseline gates and difficulty weights applied
6. ✅ **Temporally Smooth**: EWMA provides stability across windows
7. ✅ **Schema Compliant**: Shards match v1.3 specification
8. ✅ **Window-Aware**: Evaluation windows are explicitly managed

---

## Risk Mitigation

### If Manifest System Delays

- **Fallback**: Use environment variables or config files for parameters
- **Impact**: Low - can migrate to Manifest later

### If Salting Implementation Complex

- **Fallback**: Use simpler PRF (hash-based) instead of full VRF
- **Impact**: Medium - security slightly reduced but still functional

### If EWMA Storage Issues

- **Fallback**: Use in-memory dict with periodic persistence
- **Impact**: Low - can add persistence layer later

### If Shard Schema Migration Breaks Validators

- **Fallback**: Support both old and new schema during transition
- **Impact**: Medium - requires careful versioning

---

## Next Steps After Implementation

1. **Gold Truth Integration**: Add Canary Gold system (see `GOLD_TRUTH_INTEGRATION.md`)
2. **Element Architecture**: Refactor to explicit Element contracts
3. **Human Audit**: Integrate disagreement sampling
4. **Agent Support**: Add Vision Agent evaluation track
5. **TEE Integration**: Add Trusted Track with attestation

---

## Notes

- **Priority**: Focus on getting core protocol mechanics working correctly
- **Testing**: Write unit tests for each module as it's implemented
- **Documentation**: Update docstrings and add inline comments
- **Code Review**: Review integration points before merging
- **Backward Compatibility**: Legacy endpoint remains functional during migration
