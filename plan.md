# Polygon Annotation Impact Analysis

```text
┌─────────────────────────────── External Systems ───────────────────────────────┐
│ Challenge Producer / API                                                     │
│ Chutes / Miner Serving Layer                                                 │
│ Hugging Face Repos / Model Artifacts                                         │
│ Database / Persisted Records                                                 │
│ Audit / Spotcheck Storage                                                    │
│ Frontend / Visualization Tools                                               │
└───────────────────────────────────────┬───────────────────────────────────────┘
                                        │
                                        ▼
┌──────────────────────────────────── turbovision ──────────────────────────────┐
│ scorevision/utils/challenges.py                                               │
│ scorevision/vlm_pipeline/utils/response_models.py                             │
│ scorevision/vlm_pipeline/utils/geometry.py                                    │
│ scorevision/utils/evaluate.py                                                 │
│ scorevision/vlm_pipeline/non_vlm_scoring/smoothness.py                       │
│ scorevision/vlm_pipeline/image_annotation/{single,pairwise}.py               │
│ scorevision/validator/central/open_source/runner.py                          │
│ tests/fixtures/* and tests/**/*                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

### Impact Summary

1. `chutes template checker`
   - Impacted because annotation payload expectations change from box-only to geometry-first.

2. `chute miners`
   - Impacted because any miner-facing annotation contract, sample output, or validation logic that assumes `bbox_2d` will need to follow the new geometry model.

3. `validator used databases`
   - Impacted because persisted challenge, audit, and spotcheck records may need to store geometry type plus raw geometry instead of only box-shaped data.

4. `challenges from score api`
   - Impacted because the challenge payload schema returned by the Score API must carry the new annotation model.

5. `manifest`
   - Impacted because the manifest may need to declare geometry capability, schema version, or compatibility expectations for elements that use the new annotation format.

## Summary

The repo is currently box-only at the validation layer. Polygon annotations can be carried through the system only if they are converted to bounding boxes before entering the existing validation and scoring paths.

## What Is Not Supported Yet

1. `scorevision/utils/challenges.py`
   - `_parse_ground_truth_payload()` only accepts `ground_truth["annotations"]` items with a `bbox` field of length 4.
   - Polygon-style fields such as `polygon`, `points`, or masks are ignored and dropped.
   - Result: polygon annotations will not enter `PseudoGroundTruth` unless normalized first.

2. `scorevision/vlm_pipeline/utils/response_models.py`
   - `BoundingBox` only stores `bbox_2d`.
   - `FrameAnnotation` only stores `bboxes`.
   - There is no schema support for polygon geometry in the core annotation models.

3. `scorevision/vlm_pipeline/non_vlm_scoring/smoothness.py`
   - PGT filtering uses box IoU and converts boxes into rectangular masks.
   - Polygon shape is not preserved, and direct polygon IoU is not implemented.

4. `scorevision/vlm_pipeline/image_annotation/pairwise.py`
   - Mask generation is box-based only.
   - No polygon-aware overlap or visualization logic exists here.

5. `scorevision/utils/evaluate.py`
   - Miner predictions are parsed into `BoundingBox` objects only.
   - Scoring uses `bbox_2d` exclusively for AP, precision, recall, and false positive metrics.

6. `scorevision/vlm_pipeline/image_annotation/single.py`
   - Visualization draws rectangles only.
   - Polygon rendering is not implemented.

7. `scorevision/validator/central/open_source/runner.py`
   - Validation gates are based on box counts and box IoU smoothness.
   - Polygon-specific shape does not affect validation unless it changes the box representation used downstream.

## Validation Impact

- If challenge payloads start containing polygons directly, validation will currently drop them during parsing or fail schema assumptions if the payload shape changes too much.
- If the payload keeps the current `bbox` field and adds polygons alongside it, the repo will continue to work, but only the box will be used.
- If the goal is true polygon validation, this repo needs changes in:
  - ground-truth parsing,
  - annotation schema,
  - IoU/smoothness logic,
  - miner prediction parsing,
  - and image annotation/rendering.

## Practical Conclusion

Polygons are only safe today if they are converted to bounding boxes before entering this repo. The current validation pipeline does not understand or preserve polygon geometry.

## Change Plan

### Assumptions About Stored Data

- The source of truth for annotations should be geometry-first, not bbox-first.
- Do not persist `bbox_2d` as the canonical stored shape.
- Store the original geometry and its type for every annotation, for example:
  - `bbox`
  - `polygon`
  - `point`
- Any axis-aligned box used by existing scoring or validation should be derived on demand from the stored geometry.
- If geometry is not persisted, future reprocessing, audit, and visualization will be limited to whatever derived representation was produced at ingestion time.

### Recommended Scope

1. Replace box-only annotation storage with a geometry-first model.
   - Difficulty: High
   - Why: this changes the core annotation contract, not just one parser.
   - What changes:
     - introduce an enum for annotation geometry type,
     - introduce a shared geometry payload structure,
     - remove `bbox_2d` as the stored source of truth.

2. Update challenge ingestion to parse geometry generically.
   - Difficulty: High
   - What changes:
     - accept `bbox`, `polygon`, and `point` annotation types,
     - validate payload shape against the type,
     - store the raw geometry and type without flattening it at ingestion time.

3. Add derived geometry adapters for box-based consumers.
   - Difficulty: High
   - What changes:
     - convert stored geometry to axis-aligned boxes only when needed,
     - keep current validation and scoring paths working during migration,
     - centralize the conversion so box consumers do not each reimplement it.

4. Update validation and quality gates to work off derived boxes for now.
   - Difficulty: Medium
   - What changes:
     - keep existing box-count thresholds,
     - keep current box IoU smoothness unless you intentionally want polygon-aware filtering.
   - This avoids a metric redesign during the initial migration.

5. Add geometry-aware visualization.
   - Difficulty: Medium
   - Why: drawing rectangles, polygons, and points all require different render logic.
   - This should use the stored geometry directly, not a derived bbox.

6. Decide whether audit / spotcheck storage should retain raw geometry.
   - Difficulty: Medium
   - Why: this is a data-contract decision more than a code problem.
   - Recommendation: persist the raw geometry and type in challenge records or metadata so audits can reproduce the original annotation exactly.

7. Add polygon- or point-aware metrics only if you need them.
   - Difficulty: High
   - Why: this requires changing overlap logic, filtering logic, and potentially any scoring that currently expects axis-aligned boxes.
   - Not required if your first rollout only needs geometry storage plus derived-box validation.

### File-Level Impact

1. `scorevision/utils/challenges.py`
   - Difficulty: High
   - Needed for parsing geometry-first annotations and constructing derived boxes only when necessary.

2. `scorevision/vlm_pipeline/utils/response_models.py`
   - Difficulty: High
   - Needed for the new canonical geometry schema.

3. `scorevision/utils/evaluate.py`
   - Difficulty: Medium if derived-box scoring remains
   - Difficulty: High if polygon-aware scoring is required
   - Current box scoring can stay unchanged if it consumes derived boxes from the geometry model.

4. `scorevision/vlm_pipeline/non_vlm_scoring/smoothness.py`
   - Difficulty: Medium if unchanged consumers are fed derived boxes
   - Difficulty: High if polygon-aware filtering is added
   - Current box filtering can remain the default via an adapter.

5. `scorevision/vlm_pipeline/image_annotation/single.py`
   - Difficulty: Medium
   - Needed if you want the UI/debug output to show the actual geometry shape.

6. `scorevision/vlm_pipeline/image_annotation/pairwise.py`
   - Difficulty: Medium to High
   - Only needed if you want geometry-based mask/overlap computations.

7. `scorevision/validator/central/open_source/runner.py`
   - Difficulty: Medium if bbox validation is fed by derived geometry
   - Difficulty: Medium to High if you want geometry-sensitive quality gates

### Suggested Implementation Order

1. Introduce the geometry-first schema and enum.
2. Update ingestion to store raw geometry without flattening.
3. Add a central adapter that derives axis-aligned boxes on demand.
4. Wire existing validation and scoring through the adapter.
5. Add geometry-aware rendering and audits.
6. Only then migrate individual metrics from derived-box logic to native geometry logic where needed.

## External Changes Outside This Repo

This migration will not be fully contained inside `turbovision`. The following external systems or repos will likely need changes too.

### Challenge Producer / API

- Difficulty: High
- Why:
  - the API that creates or serves challenges must emit the new geometry schema,
  - any contract that currently returns `bbox`-only annotations must be updated,
  - any client that assumes a 4-number box will need a compatibility path or a schema bump.
- Typical work:
  - update challenge JSON payloads,
  - update OpenAPI / schema definitions if they exist,
  - version the challenge contract if backward compatibility is required.

### Chutes / Miner Serving Layer

- Difficulty: Medium to High
- Why:
  - if chutes only consume images and produce predictions, they may not need immediate changes,
  - but if they serialize or validate annotation payloads, they must understand the new geometry model.
- Typical work:
  - update request/response models if annotation data is passed through,
  - update any debug or inspection endpoints that assume boxes,
  - update any example miners or templates that emit box-shaped annotations.

### Hugging Face Repos / Model Artifacts

- Difficulty: Medium
- Why:
  - if model cards, sample outputs, or dataset examples contain box-only annotations, they will become outdated,
  - any downstream consumer using those examples as a contract will need to be updated.
- Typical work:
  - update README/examples,
  - update sample annotation JSON,
  - update any repo template or inference example that shows box-only outputs.

### Database / Persisted Records

- Difficulty: High
- Why:
  - any table or document that stores challenge annotations, pseudo-GT, or audit artifacts must be able to store geometry type and payload,
  - if the current schema is box-specific, it will need a migration.
- Typical work:
  - add a geometry type column/field,
  - add a JSON blob or structured geometry payload,
  - backfill or migrate historical records if you want them queryable in the new format.

### Audit / Spotcheck Storage

- Difficulty: Medium to High
- Why:
  - spotcheck and audit pipelines should preserve the original geometry, not just derived boxes,
  - if they persist or replay challenge records, the stored shape contract must change.
- Typical work:
  - update shard payloads and evaluation artifacts,
  - include raw geometry in any public results or replay data,
  - ensure audit tools can render or interpret the new geometry.

### Frontend / Visualization Tools

- Difficulty: Medium
- Why:
  - any UI or notebook that displays annotations will need geometry-aware drawing logic,
  - rectangle-only rendering will no longer reflect the source of truth for polygons or points.

### Backward Compatibility Strategy

- If external systems cannot all be updated at once, introduce a versioned contract.
- Recommended approach:
  - keep a v1 box-only payload for old clients,
  - introduce a v2 geometry-first payload for new clients,
  - make the validator accept v2 and optionally adapt v1 during a transition window.
- If you do not version the contract, every producer and consumer must be updated in lockstep.

### Manual External Changes To Remember

These are the external items that would need to be touched manually alongside the repo changes:

1. Score API challenge payloads and upstream annotation generation.
2. Chutes template checker and any chute-side schema validation.
3. Chute miners, templates, and example payloads that assume `bbox_2d`.
4. Databases or persisted records storing challenge, pseudo-GT, or audit data.
5. Audit / spotcheck storage and replay artifacts.
6. Manifest declarations if element capability or schema version needs to be explicit.
7. Hugging Face repos, examples, and docs.
8. Any frontend or notebook visualization tools that render annotations.

## Proposed Geometry Schema

This is a concrete target shape for the new annotation model.

### Core Types

```python
from enum import StrEnum
from pydantic import BaseModel, Field


class AnnotationGeometryType(StrEnum):
    BBOX = "bbox"
    POLYGON = "polygon"
    POINT = "point"


class Point2D(BaseModel):
    x: float
    y: float


class AnnotationGeometry(BaseModel):
    type: AnnotationGeometryType
    points: list[Point2D] = Field(default_factory=list)

    # Interpretation by type:
    # - bbox: points must contain exactly 2 corners, top-left then bottom-right
    # - polygon: points must contain 3+ vertices in drawing order
    # - point: points must contain exactly 1 coordinate
```

### Annotation Model

```python
class Annotation(BaseModel):
    label: str
    geometry: AnnotationGeometry
    score: float | None = None
    cluster_id: str | None = None
    attributes: dict[str, object] = Field(default_factory=dict)
```

### Frame Model

```python
class FrameAnnotation(BaseModel):
    annotations: list[Annotation]
    category: str
    confidence: int
    reason: str
```

### Derived Box Adapter

For existing validation and scoring, use a helper that derives an axis-aligned box only when needed.

```python
def geometry_to_bbox(geometry: AnnotationGeometry) -> tuple[int, int, int, int]:
    ...
```

### Notes

- The schema keeps the original geometry as the source of truth.
- `bbox` is no longer stored as canonical data.
- `points` is intentionally generic enough to support boxes, polygons, and points without adding separate top-level fields for each shape.
- If you later need rotated boxes, they can be added as another `AnnotationGeometryType` without changing the overall model structure.
- `geometry.type` is the only shape discriminator needed in the simplified contract.

## Current File Mapping

This is how the existing codebase maps onto the proposed geometry-first model.

### Replace Or Refactor

1. `scorevision/vlm_pipeline/utils/response_models.py`
   - Replace `BoundingBox` and `FrameAnnotation` as the canonical annotation schema.
   - New models should carry `geometry` and `type` rather than `bbox_2d`.
   - Existing code that imports `BoundingBox` will need to move to the new generic annotation model or a derived helper.

2. `scorevision/utils/challenges.py`
   - Refactor `_parse_ground_truth_payload()` to parse geometry generically.
   - This becomes the ingestion boundary for the new annotation model.
   - It should not flatten geometry into boxes except via a dedicated adapter for downstream box consumers.

3. `scorevision/utils/evaluate.py`
   - Refactor `parse_miner_prediction()` and the downstream scoring helpers to consume derived geometry adapters.
   - If scoring remains box-based initially, this file should call a geometry-to-box helper rather than assume a stored `bbox_2d`.

4. `scorevision/vlm_pipeline/non_vlm_scoring/smoothness.py`
   - Refactor mask and IoU helpers so they can work from derived geometry.
   - This file is a strong candidate for a shared geometry adapter because it currently encodes box-only assumptions.

5. `scorevision/vlm_pipeline/image_annotation/single.py`
   - Replace rectangle-only rendering with geometry-aware rendering.
   - This should draw polygons and points directly from the stored geometry.

6. `scorevision/vlm_pipeline/image_annotation/pairwise.py`
   - Refactor to support geometry-derived masks.
   - Keep it as a helper for overlap visualization if you still want box/mask comparisons.

7. `scorevision/validator/central/open_source/runner.py`
   - Mostly stays structurally similar.
   - The important change is that it must receive annotations through the new geometry-aware parsing and filtering path.

### Likely New Files

1. `scorevision/vlm_pipeline/utils/geometry.py`
   - Shared geometry models and conversion helpers.
   - Recommended location for `geometry_to_bbox()` and any polygon/point utilities.

2. `scorevision/vlm_pipeline/utils/annotation_adapters.py`
   - Optional dedicated adapter layer if you want to keep conversion logic separate from pure models.
   - Useful if multiple consumers need different derived views of the same annotation.

3. `scorevision/vlm_pipeline/image_annotation/geometry.py`
   - Optional renderer helpers for polygons and points.
   - Could replace or augment the current box-only rendering module.

### Unchanged Or Minimally Changed

1. `scorevision/vlm_pipeline/domain_specific_schemas/challenge_types.py`
   - Probably unchanged unless the challenge type itself needs to expose geometry capability flags.

2. `scorevision/vlm_pipeline/utils/data_models.py`
   - The surrounding score container types can likely remain, but their `annotation` field types may need to be widened.

3. `scorevision/validator/central/open_source/runner.py` control flow
   - The orchestration logic probably stays the same.
   - The data it receives from parsing and adapters changes.

4. Tests and fixtures
   - Expect broad updates across all fixture builders that currently instantiate `BoundingBox` or `FrameAnnotation`.
   - These will need to be rewritten to use the new canonical geometry model.

## Execution Phases

This migration is large enough that it should be split into multiple PRs.

### Phase 1: Schema And Models

- Goal: define the new geometry-first annotation contract.
- Scope:
  - add the geometry enum and geometry model,
  - replace box-only annotation storage with generic annotation storage,
  - add model validation for bbox, polygon, and point payloads.
- Files:
  - `scorevision/vlm_pipeline/utils/response_models.py`
  - `scorevision/vlm_pipeline/utils/geometry.py` if created
  - affected tests and fixtures
- Test expectations:
  - validate each geometry type independently,
  - reject malformed payloads,
  - preserve serialization round-trips.

### Phase 2: Ingestion And Persistence

- Goal: accept geometry-first challenge data without flattening it immediately.
- Scope:
  - update challenge parsing,
  - persist raw geometry and type in the challenge payload path,
  - add any database or record schema changes needed outside this repo.
- Files:
  - `scorevision/utils/challenges.py`
  - any persisted challenge or audit record models
- Test expectations:
  - geometry payloads parse correctly,
  - box, polygon, and point annotations all survive ingestion,
  - derived-box helpers can still be produced for downstream consumers.

### Phase 3: Derived Box Adapter

- Goal: keep existing validation and scoring running while the stored model changes.
- Scope:
  - add a central geometry-to-bbox adapter,
  - route box-based consumers through that adapter,
  - avoid duplicated shape-conversion logic.
- Files:
  - `scorevision/vlm_pipeline/utils/geometry.py` or adapter module
  - `scorevision/utils/evaluate.py`
  - `scorevision/vlm_pipeline/non_vlm_scoring/smoothness.py`
  - `scorevision/vlm_pipeline/image_annotation/pairwise.py`
- Test expectations:
  - derived boxes match prior behavior for bbox input,
  - polygon and point inputs produce deterministic derived boxes where needed,
  - current scoring does not regress.

### Phase 4: Rendering And Inspection

- Goal: make debugging and audit outputs reflect real geometry.
- Scope:
  - add polygon and point rendering,
  - stop relying on rectangle-only visualization.
- Files:
  - `scorevision/vlm_pipeline/image_annotation/single.py`
  - optional geometry rendering helpers
- Test expectations:
  - bbox rendering still works,
  - polygons render with correct vertex order,
  - points render visibly and consistently.

### Phase 5: Validation And Metrics Hardening

- Goal: decide which metrics stay box-derived and which become geometry-native.
- Scope:
  - keep existing box-based gates if that is acceptable,
  - optionally add geometry-native overlap/mask logic later,
  - update smoothness and scoring only where there is clear value.
- Files:
  - `scorevision/validator/central/open_source/runner.py`
  - `scorevision/vlm_pipeline/non_vlm_scoring/smoothness.py`
  - `scorevision/utils/evaluate.py`
- Test expectations:
  - validation still accepts old bbox workflows,
  - geometry-based annotations do not break score calculation,
  - any polygon-native metric is covered by dedicated tests.

### Phase 6: External Contract Updates

- Goal: align outside systems with the new schema.
- Scope:
  - update challenge APIs,
  - update chutes or miner templates if they carry annotation payloads,
  - migrate databases and audit stores,
  - refresh Hugging Face examples and docs.
- Test expectations:
  - producer and consumer payloads agree on the schema version,
  - old clients either still work through a compatibility path or fail clearly.

## PR Breakdown Suggestion

1. PR 1: schema and model types only.
2. PR 2: ingestion and persistence of raw geometry.
3. PR 3: adapter layer for derived boxes plus validation compatibility.
4. PR 4: visualization and audit rendering.
5. PR 5: external contract updates and cleanup.

## Risk Notes

- The highest-risk change is replacing `bbox_2d` as the stored source of truth.
- The biggest integration risk is external systems continuing to emit or expect box-only payloads.
- The safest rollout strategy is to land the geometry-first schema first, then add adapters, then migrate consumers one by one.
