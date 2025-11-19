# Pillar Metrics & Weighting Protocol

## Overview

* Composite scoring for VLM pipelines is now **manifest-driven**.
* Each **element** in a manifest has multiple **pillars**, each with a **weight**.
* Scores are computed per pillar using functions registered in `METRIC_REGISTRY`.
* The **weighted score** for a pillar is `score * weight`.
* The **element-level total** is the sum of weighted pillar scores.
* The **overall evaluation score** is the mean of all element-level totals.

---

## Manifest Example

```yaml
elements:
  - id: PlayerDetect_v1
    metrics:
      pillars:
        IOU: 0.3
        COUNT: 0.1
        SMOOTHNESS: 0.3
        ROLE: 0.3
  - id: PitchCalib_v1
    metrics:
      pillars:
        IOU: 1.0
```

* **PlayerDetect_v1** has four pillars with custom weights summing to 1.0.
* **PitchCalib_v1** has a single pillar with weight 1.0.

---

## Scoring Calculation

1. Compute each pillar score using the registered metric function.
2. Multiply each pillar score by its manifest-specified weight → **weighted score**.
3. Sum the weighted scores across pillars for each element → **total_weighted**.
4. Compute total raw scores similarly (ignoring weights) → **total_raw**.
5. Compute **mean_weighted** as the average of `total_weighted` across all elements.

Example:

| Element            | Pillar     | Score | Weight | Weighted Score |
| ------------------ | ---------- | ----- | ------ | -------------- |
| PlayerDetect_v1    | IOU        | 0.8   | 0.3    | 0.24           |
| PlayerDetect_v1    | COUNT      | 1.0   | 0.1    | 0.10           |
| PlayerDetect_v1    | SMOOTHNESS | 0.7   | 0.3    | 0.21           |
| PlayerDetect_v1    | ROLE       | 0.6   | 0.3    | 0.18           |
| **Total Weighted** |            |       |        | **0.73**       |

* Repeat for all elements.
* Overall **mean_weighted** = average of element total_weighted values.

---

## Developer Notes

* Metrics must be **registered** via `@register_metric(ElementPrefix, PillarName)`.
* `get_element_scores` will raise `NotImplementedError` if a metric is missing.
* Pillars with **weight = 0** are included in raw totals but do **not affect weighted totals**.
* The **metric function signature** supports flexible arguments via `**kwargs`:

```python
def metric_fn(
    pseudo_gt: list[PseudoGroundTruth],
    miner_predictions: dict[int, dict],
    frames: FrameStore,
    image_height: int,
    image_width: int,
    challenge_type: ChallengeType,
    **kwargs
) -> float:
    ...
```

---

## Changelog / Release Notes

* **Feature:** Manifest-driven pillar weights for composite scoring.
* **Change:** `post_vlm_ranking` now uses `get_element_scores`.
* **Impact:** Operators must ensure all pillars in a manifest have a registered metric.
* **Developer Action:** Add new metrics via `@register_metric`.
