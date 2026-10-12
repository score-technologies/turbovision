# Snooker Private Track Miner Spec

This is the v0 miner-facing contract for:

```text
manako/DetectSnookerBallState
```

V0 is a narrow ball-state challenge. Miners receive a normalized short clip and explicit target frame ids, then return ball state only for those requested frames.

## Request Envelope

The validator sends `/challenge` requests using `ChallengeRequest`.

```json
{
  "challenge_id": "12345",
  "video_url": "https://assets.example.com/snooker/window-abc.mp4",
  "target_frames": [50, 150, 250, 350, 450]
}
```

For snooker, `video_url` and non-empty `target_frames` are required. Frame ids are relative to the provided 20 second clip, not to the source match.

## Response Envelope

Return one `snooker_ball_state` frame object per requested target frame.

```json
{
  "challenge_id": "12345",
  "prediction": {
    "type": "snooker_ball_state",
    "frames": [
      {
        "frame": 50,
        "balls": [
          {
            "label": "cue",
            "x": 0.52,
            "y": 0.33,
            "state": "on_table",
            "confidence": 0.98
          },
          {
            "label": "red",
            "x": 0.61,
            "y": 0.41,
            "state": "on_table"
          },
          {
            "label": "black",
            "state": "occluded"
          }
        ]
      }
    ]
  },
  "processing_time": 0.82
}
```

Extra non-target frames are ignored by scoring. Missing target frames score zero for that frame. Return at most one object per requested frame; duplicate target-frame objects are merged and penalized.

## Coordinates

`x` and `y` are normalized table coordinates:

- `x=0.0`: baulk end
- `x=1.0`: black-spot end
- `y=0.0`: left cushion in the canonical broadcast-overhead orientation
- `y=1.0`: right cushion in the canonical broadcast-overhead orientation

`x` and `y` are required for balls with `state: "on_table"`. Coordinates may be omitted for `potted`, `occluded`, or `unknown` balls.

On-table coordinate credit decays linearly to zero at a normalized table distance of `0.05`. Predictions farther away than that are treated as unmatched for that ball.

Optional `bbox` fields may be included for audit/debugging, but table-space coordinates are the scoring target.

## Labels

Accepted canonical labels:

- `cue`
- `red`
- `yellow`
- `green`
- `brown`
- `blue`
- `pink`
- `black`

Reds are returned as multiple entries with `label: "red"` and are matched as an unordered set in normalized table coordinates.

## States

Accepted canonical states:

- `on_table`
- `potted`
- `occluded`
- `unknown`

Invalid state strings are scored as invalid predictions. They are not treated as `unknown`.

## Scoring Intent

The scorer rewards:

- coordinate accuracy for matched on-table balls
- identity/class accuracy
- red count and red-set matching quality
- state accuracy
- low false-positive, duplicate, invalid-label, and extra-ball rate

Duplicate unique colours, wrong red counts, missing balls, invalid labels, empty predictions, and `on_table` balls without coordinates are penalized.

V0 deliberately excludes shot boundaries, shot outcomes, cue geometry, table-boundary scoring, and tactical classification.
