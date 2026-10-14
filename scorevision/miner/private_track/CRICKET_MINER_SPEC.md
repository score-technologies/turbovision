# Cricket Private Track Miner Spec

This file is miner-facing and describes the current cricket private-track response contract.

## Release Timeline

- Release date: mid May

## Published Ground-Truth Training Data

A large cricket training set is published at:

- [Training index](https://manako.scoredata.me/train/index.json)

The index is a JSON array of challenge paths, for example:

```json
[
  "train/train1.json",
  "train/train2.json"
]
```

Resolve each path against `https://manako.scoredata.me/` to retrieve the
corresponding challenge. Each challenge JSON contains the video URL and its
scored ground truth:

```json
{
  "video_url": "https://scoredata.me/cricket/training/v2/<video>.mp4",
  "groundtruth": {
    "kph": 87.89,
    "bounce_x": 5.317,
    "stump_y": 0.333,
    "stump_z": 0.674,
    "impact_x": null,
    "deviation": 0.449,
    "swing_angle": 2.8,
    "release_y": -0.419,
    "release_z": 2.009,
    "impact_y": null,
    "bounce_y": 0.044,
    "impact_z": null
  }
}
```

For example, the index entry `train/train1.json` is available at
`https://manako.scoredata.me/train/train1.json`. Do not resolve it relative to
the index directory, which would incorrectly produce `/train/train/train1.json`.

The complete dataset can be enumerated with Python:

```python
import requests

origin = "https://manako.scoredata.me/"
index = requests.get(f"{origin}train/index.json", timeout=30).json()

for challenge_path in index:
    challenge = requests.get(f"{origin}{challenge_path}", timeout=30).json()
    video_url = challenge["video_url"]
    scored_ground_truth = challenge["groundtruth"]
    # Download/process video_url and use scored_ground_truth as the label.
```

Some ground-truth values may be `null`; consumers should preserve or explicitly
handle missing values rather than coercing them to zero.

## Response Envelope

For cricket challenges, miners should return exactly one delivery prediction in this envelope:

```json
{
  "challenge_id": "<challenge_id>",
  "prediction": {
    "kph": 126.86,
    "release_y": -0.42,
    "release_z": 2.01,
    "bounce_x": 8.001,
    "bounce_y": 0.21,
    "impact_x": 1.34,
    "impact_y": 0.08,
    "impact_z": 0.74,
    "stump_y": 0.017,
    "stump_z": 1.046,
    "deviation": 1.104,
    "swing_angle": -2.402
  },
  "processing_time": 0.73
}
```

- `challenge_id`: echo the incoming challenge id.
- `prediction`: one canonical delivery row.
- `processing_time`: miner-side processing time in seconds.

## Canonical Field Names

The validator accepts the following canonical cricket fields, aligned to GT columns and excluding `r2_url`:

- `match`
- `matchid`
- `inningsid`
- `overid`
- `ball_in_over`
- `ballid`
- `xlsx_overs`
- `scorecard_overs`
- `kph`
- `release_y`
- `release_z`
- `bounce_x`
- `bounce_y`
- `impact_x`
- `impact_y`
- `impact_z`
- `interception_distance`
- `stump_y`
- `stump_z`
- `swing_angle`
- `deviation`
- `runs`
- `wickets`

Accepted aliases:

- `innings` -> `inningsid`
- `over` -> `overid`
- `ball` -> `ball_in_over`
- `overs` -> `scorecard_overs`
- `rel_y` -> `release_y`
- `rel_z` -> `release_z`
- `inter_d` -> `interception_distance`
- `swing_deg` -> `swing_angle`
- `deviation_deg` -> `deviation`
- `wkts` -> `wickets`

## Field Definitions (Current Coverage)

The definitions below come from the ball-tracking glossary and are limited to fields already documented there.

### Coordinate Axes (for positional fields)

All positional measurements are in meters and use a shared coordinate system:

- origin `(0, 0, 0)`: base of the middle stump at the batter's end
- positive `x`: along the pitch centerline from batter's end toward bowler's end
- positive `y`: horizontal, perpendicular to `x`, to the right from the main camera view
- positive `z`: vertical upward

### Core Ball-Tracking Fields

- `kph`: release speed of the ball as it leaves the bowler's hand.
- `release_y`, `release_z`: release-point width (`y`) and height (`z`) where the ball leaves the bowler's hand.
- `bounce_x`, `bounce_y`: bounce-point coordinates; if intercepted before bouncing, this is the projected bounce point.
- `impact_x`, `impact_y`, `impact_z`: coordinates where the ball trajectory is intercepted by the batter (bat or body).
- `interception_distance`: distance from bounce to impact along `x` (glossary shorthand: `bounce_x - impact_x`).
- `stump_y`, `stump_z`: coordinates where the ball crosses the stumps plane (`x = 0`), or projected crossing if intercepted earlier.
- `swing_angle`: horizontal in-air deviation angle (degrees) from release to bounce.
- `deviation`: horizontal deviation angle (degrees) caused by/after bounce.

## What Miners Should Prioritize

The validator currently supports the full canonical row. Twelve ball-tracking fields contribute to the score, with the following priority tiers:

1. `bounce_x`, `kph`, `stump_y`, `stump_z`, and `impact_x` (10% each)
2. `deviation`, `swing_angle`, `release_y`, `release_z`, and `impact_y` (8% each)
3. `bounce_y` and `impact_z` (5% each)

Together, these fields account for 100% of the score. `interception_distance`, identifiers, and outcome fields remain accepted but carry no scoring weight.

## Recommended Return Shape

### Primary / high-value fields

These fields have the highest individual weights and should be implemented first:

- `bounce_x` (10%)
- `kph` (10%)
- `stump_y` (10%)
- `stump_z` (10%)
- `impact_x` (10%)

### Optional / lower-value metadata fields

These are accepted for challenge correlation. The identifiers listed below have zero scoring weight:

- `match` (0%)
- `matchid` (0%)
- `inningsid` (0%)
- `overid` (0%)
- `ball_in_over` (0%)
- `ballid` (0%)
- `xlsx_overs` (0%)
- `scorecard_overs` (0%)

### Secondary scored fields

These fields complete the trajectory and collectively carry half of the score:

- `deviation` (8%)
- `swing_angle` (8%)
- `release_y` (8%)
- `release_z` (8%)
- `impact_y` (8%)
- `bounce_y` (5%)
- `impact_z` (5%)

### Optional / zero-weight fields

These are accepted for compatibility but have zero scoring weight:

- `runs` (0%)
- `wickets` (0%)
- `interception_distance` (0%)

## Practical Guidance

- Returning identifiers or outcome fields does not increase the score.
- Returning all twelve scored ball-tracking fields is required to maximize the score.
- Missing fields are allowed; they simply score `0`.
- Exact/id-like fields are scored by exact match after light normalization.
- Numeric physical fields are scored with strict tolerance-based decay; prioritize precise ball-tracking estimates over rough approximations.

## Current Scoring Intent

At the moment:

- five highest-priority metrics account for 50% of the score
- seven secondary metrics account for the remaining 50%
- match, delivery identifier, outcome, and `interception_distance` fields have zero weight
- over representations are retained for correlation only and have zero weight

So the intended miner strategy is:

1. get the five 10%-weight ball-tracking outputs working
2. implement the seven secondary scored fields
3. treat zero-weight metadata and outcomes as correlation data only
