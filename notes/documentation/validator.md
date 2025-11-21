# Pillar Metrics, Weighting & Economic Controls

## Overview

* Composite scoring for VLM pipelines is now **manifest-driven**.
* Each **element** in a manifest has multiple **pillars**, each with a **weight**.
* Scores are computed per pillar using functions registered in `METRIC_REGISTRY`.
* The **weighted score** for a pillar is `score * weight`.
* The **element-level total** is the sum of weighted pillar scores.
* **Economic parameters** are applied per element:
  * `baseline_theta (\u03b8\u2090)`: minimum performance threshold
  * `delta_floor (\u03b4\u2090)`: minimum margin above baseline before applying difficulty scaling
  * `beta (\u03b2\u2090)`: difficulty multiplier to scale rewards
* The **overall evaluation score** is the mean of all element-level totals after applying economic adjustments.

---

## Manifest Example with Economic Parameters

```yaml
elements:
  - id: PlayerDetect_v1
    metrics:
      pillars:
        IOU: 0.3
        COUNT: 0.1
        SMOOTHNESS: 0.3
        ROLE: 0.3
    baseline_theta: 0.3
    delta_floor: 0.05
    beta: 1.5

  - id: PitchCalib_v1
    metrics:
      pillars:
        IOU: 1.0
    baseline_theta: 0.2
    delta_floor: 0.0
    beta: 1.0
```

* **PlayerDetect_v1** has four pillars with custom weights summing to 1.0. will only earn rewards for improvements above 0.3, with a minimum contribution of 0.05 scaled by β=1.5.
* **PitchCalib_v1** has a single pillar with weight 1.0. has no delta floor and standard scaling.
---

## Scoring Calculation with Economics

1. Compute each pillar score using the registered metric function.
2. Multiply each pillar score by its manifest-specified weight → **weighted pillar score**.
3. Sum weighted pillar scores across all pillars → **total_weighted**.
4. Apply **baseline gate**: `score - θₑ`, clamp to ≥ 0.
5. Apply **delta floor**: ensure minimum contribution δₑ if score is below θₑ.
6. Apply **difficulty multiplier**: multiply improvement by βₑ → **weighted_and_gated_score**.
7. Repeat for all elements and compute **mean_weighted** across elements.

Example table (simplified):

| Element            | Pillar     | Score | Weight | Weighted Score |
| ------------------ | ---------- | ----- | ------ | -------------- |
| PlayerDetect_v1    | IOU        | 0.8   | 0.3    | 0.24           |
| PlayerDetect_v1    | COUNT      | 0.4   | 0.1    | 0.04           |
| PlayerDetect_v1    | SMOOTHNESS | 0.7   | 0.3    | 0.21           |
| PlayerDetect_v1    | ROLE       | 0.6   | 0.3    | 0.18           |
| **Total Weighted** |            |       |        | **0.67**       |
| **After θ/δ/β**   |            |       |        | **0.99**       |

---

## Developer Notes

* Metrics must be **registered** via `@register_metric(ElementPrefix, PillarName)`.
* `get_element_scores` will raise `NotImplementedError` if a metric is missing.
* Elements with **Qₑ = 0** (after baseline gate and delta floor) can be routed to burn per spec.
* Pillars with **weight = 0** are included in raw totals but do **not affect weighted totals**.
* `Element.weight_score()` computes the final gated and scaled contribution for a given score.
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
* **Feature:** Baseline, delta floor, and β economic controls added per element.
* **Change:** `post_vlm_ranking` now uses `get_element_scores`.
* **Impact:** Operators must ensure all pillars in a manifest have a registered metric.
* **Impact:** Operators can tune θₑ, δₑ, βₑ in the manifest to scale or zero rewards.
* **Developer Action:** Add new metrics via `@register_metric`.
