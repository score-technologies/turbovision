# External Changes Needed For Geometry-First Annotations

This note captures the manual changes outside `turbovision` that are likely required to support the geometry-first annotation model.

## Systems To Update

### Upstream Of This Repo

These systems produce or supply the data/contracts that `turbovision` consumes:

1. Score API challenge payloads and upstream annotation generation.
   - Closest in-repo anchor: [scorevision/utils/challenges.py](/Users/mohammed/Code/Score/turbovision/scorevision/utils/challenges.py)
2. Chutes template checker and any chute-side schema validation.
   - Closest in-repo anchor: [scorevision/miner/private_track/MINER.md](/Users/mohammed/Code/Score/turbovision/scorevision/miner/private_track/MINER.md)
3. Manifest declarations if element capability or schema version needs to be explicit.
   - Closest in-repo anchor: [scorevision/utils/manifest.py](/Users/mohammed/Code/Score/turbovision/scorevision/utils/manifest.py)
4. Hugging Face repos, examples, and docs when they define the payload contract used to build or validate data.
   - Closest in-repo anchor: [scorevision/miner/private_track/MINER.md](/Users/mohammed/Code/Score/turbovision/scorevision/miner/private_track/MINER.md)

### Downstream Of This Repo

These systems consume the outputs, records, or artifacts produced by `turbovision`:

1. Databases or persisted records storing challenge, pseudo-GT, or audit data.
   - Closest in-repo anchors: [scorevision/utils/cloudflare_helpers.py](/Users/mohammed/Code/Score/turbovision/scorevision/utils/cloudflare_helpers.py), [scorevision/validator/audit/open_source/storage.py](/Users/mohammed/Code/Score/turbovision/scorevision/validator/audit/open_source/storage.py)
2. Audit / spotcheck storage and replay artifacts.
   - Closest in-repo anchors: [scorevision/validator/audit/open_source/spotcheck.py](/Users/mohammed/Code/Score/turbovision/scorevision/validator/audit/open_source/spotcheck.py), [scorevision/validator/audit/private_track/spotcheck.py](/Users/mohammed/Code/Score/turbovision/scorevision/validator/audit/private_track/spotcheck.py)
3. Chute miners, templates, and example payloads that assume `bbox_2d`.
   - Closest in-repo anchors: [scorevision/miner/private_track/routes.py](/Users/mohammed/Code/Score/turbovision/scorevision/miner/private_track/routes.py), [scorevision/miner/private_track/predictor.py](/Users/mohammed/Code/Score/turbovision/scorevision/miner/private_track/predictor.py), [scorevision/miner/open_source/example_miner/miner.py](/Users/mohammed/Code/Score/turbovision/scorevision/miner/open_source/example_miner/miner.py)
4. Any frontend or notebook visualization tools that render annotations.
   - Closest in-repo anchors: [scorevision/vlm_pipeline/image_annotation/single.py](/Users/mohammed/Code/Score/turbovision/scorevision/vlm_pipeline/image_annotation/single.py), [scorevision/vlm_pipeline/image_annotation/pairwise.py](/Users/mohammed/Code/Score/turbovision/scorevision/vlm_pipeline/image_annotation/pairwise.py)

### Both Sides

These usually need coordinated updates because they define the contract between producer and consumer:

1. Score API challenge payloads and upstream annotation generation.
   - See also: [scorevision/utils/challenges.py](/Users/mohammed/Code/Score/turbovision/scorevision/utils/challenges.py)
2. Chutes template checker and any chute-side schema validation.
   - See also: [scorevision/miner/private_track/MINER.md](/Users/mohammed/Code/Score/turbovision/scorevision/miner/private_track/MINER.md)
3. Chute miners, templates, and example payloads that assume `bbox_2d`.
   - See also: [scorevision/miner/private_track/routes.py](/Users/mohammed/Code/Score/turbovision/scorevision/miner/private_track/routes.py)
4. Databases or persisted records storing challenge, pseudo-GT, or audit data.
   - See also: [scorevision/utils/cloudflare_helpers.py](/Users/mohammed/Code/Score/turbovision/scorevision/utils/cloudflare_helpers.py)
5. Audit / spotcheck storage and replay artifacts.
   - See also: [scorevision/validator/audit/open_source/spotcheck.py](/Users/mohammed/Code/Score/turbovision/scorevision/validator/audit/open_source/spotcheck.py)
6. Manifest declarations if element capability or schema version needs to be explicit.
   - See also: [scorevision/utils/manifest.py](/Users/mohammed/Code/Score/turbovision/scorevision/utils/manifest.py)
7. Hugging Face repos, examples, and docs.
   - See also: [scorevision/miner/private_track/MINER.md](/Users/mohammed/Code/Score/turbovision/scorevision/miner/private_track/MINER.md)
8. Any frontend or notebook visualization tools that render annotations.
   - See also: [scorevision/vlm_pipeline/image_annotation/single.py](/Users/mohammed/Code/Score/turbovision/scorevision/vlm_pipeline/image_annotation/single.py)

## Systems To Update

1. Score API challenge payloads and upstream annotation generation.
2. Chutes template checker and any chute-side schema validation.
3. Chute miners, templates, and example payloads that assume `bbox_2d`.
4. Databases or persisted records storing challenge, pseudo-GT, or audit data.
5. Audit / spotcheck storage and replay artifacts.
6. Manifest declarations if element capability or schema version needs to be explicit.
7. Hugging Face repos, examples, and docs.
8. Any frontend or notebook visualization tools that render annotations.

## Contract Notes

- The Score API should emit the new geometry-first schema.
- If box-only and geometry-first clients must coexist, version the contract instead of overloading one payload shape.
- Persist the raw geometry and type where annotations are stored externally, not only the derived box view.

## Track Split

The upstream changes should not be treated as two unrelated payload shapes.

### Private Track

- Keep the existing bbox/action-based request and response shapes.
- Use a private-track adapter to translate any upstream `type=bbox` geometry into the legacy private-track shape.
- Do not introduce geometry-first annotations into the private-track miner contract unless that track is explicitly being redesigned.
- Private-track documentation and examples should remain bbox-centric.

### Open / VLM Track

- Move to the geometry-first annotation schema.
- This is the path that should carry polygons, points, and bboxes as typed geometry.
- Upstream data producers and validators for this track should be updated separately from private track.

### Unified Upstream Contract

- The recommended upstream model is a single typed-geometry schema.
- `type=bbox` remains a valid geometry variant in that schema.
- Private track consumes the canonical upstream output through an adapter that converts bbox geometry into its legacy contract.
- Open / VLM track consumes the same canonical geometry directly.
