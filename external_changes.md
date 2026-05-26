# External Changes Needed For Geometry-First Annotations

This note captures the manual changes outside `turbovision` that are likely required to support the geometry-first annotation model.

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

