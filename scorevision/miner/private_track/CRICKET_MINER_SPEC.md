# Cricket Private Track Miner Spec

This file is miner-facing and describes the current cricket private-track response contract.

## Release Timeline

- Release date: mid May

## Response Envelope

For cricket challenges, miners should return exactly one delivery prediction in this envelope:

```json
{
  "challenge_id": "<challenge_id>",
  "prediction": {
    "kph": 126.86,
    "bounce_x": 8.001,
    "stump_y": 0.017,
    "deviation": 1.104,
    "swing_angle": -2.402,
    "stump_z": 1.046
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

The validator currently supports the full canonical row, but miners should focus first on these six fields:

1. `kph`
2. `bounce_x`
3. `stump_y`
4. `deviation`
5. `swing_angle`
6. `stump_z`

These are the main v1 ball-tracking asks and dominate the score.

## Recommended Return Shape

### Primary / high-value fields

These fields carry most of the reward signal and should be implemented first:

- `kph`
- `bounce_x`
- `stump_y`
- `deviation`
- `swing_angle`
- `stump_z`

### Optional / lower-value metadata fields

These are accepted but currently have much smaller weight:

- `match`
- `matchid`
- `inningsid`
- `overid`
- `ball_in_over`
- `ballid`
- `xlsx_overs`
- `scorecard_overs`

### Optional / secondary geometry fields

These are useful but currently lower priority than the six primary metrics:

- `release_y`
- `release_z`
- `bounce_y`
- `impact_x`
- `impact_y`
- `impact_z`
- `interception_distance`

### Optional / low-value outcome fields

These are accepted but should not be the main focus for miners:

- `runs`
- `wickets`

## Practical Guidance

- Returning only easy metadata is not enough to score well.
- Returning the six primary ball-tracking fields is much more valuable than returning ids alone.
- Missing fields are allowed; they simply score `0`.
- Exact/id-like fields are scored by exact match after light normalization.
- Numeric physical fields are scored with strict tolerance-based decay; prioritize precise ball-tracking estimates over rough approximations.

## Current Scoring Intent

At the moment, roughly:

- primary six metrics account for most of the score
- metadata/id fields are deliberately downweighted
- secondary geometry helps, but less than the primary six
- outcome fields are low weight

So the intended miner strategy is:

1. get the six core ball-tracking outputs working
2. improve secondary geometry
3. fill in metadata and outcomes last
