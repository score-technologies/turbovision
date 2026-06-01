# Snooker Private Challenge V0

This branch defines the TurboVision-side contract for `manako/DetectSnookerBallState`. It does not implement the external challenge backend or publish the live manifest.

## Challenge Shape

Validators request one short video window and a fixed frame set:

```json
{
  "task_id": "12345",
  "payload": {
    "clip_url": "https://assets.example.com/snooker/window-abc.mp4",
    "target_frames": [50, 150, 250, 350, 450]
  }
}
```

`video_url` may be provided at top level instead of `payload.clip_url`. For snooker, a video URL and non-empty `target_frames` are required. Frame ids are relative to the supplied 20 second clip.

## Ground Truth Shape

`GET /api/tasks/{id}/ground-truth` must return:

```json
{
  "ground_truth": {
    "frames": [
      {
        "frame": 50,
        "balls": [
          {
            "label": "cue",
            "x": 0.52,
            "y": 0.33,
            "state": "on_table",
            "confidence": 1.0,
            "bbox": [120, 90, 135, 105]
          }
        ]
      }
    ]
  }
}
```

Hidden annotation storage should also retain table corners and orientation for QA. V0 miner responses do not require table-corner output.

## Data Target

Prototype dataset target:

- 6 public bootstrap windows
- 20 hidden/private scoring windows
- Candidate source footage may be YouTube prototype footage for dry run
- Hidden clip ids, source timestamps, and ground truth must not be published

Normalize every scoring clip before annotation:

- 20 seconds
- 25 fps
- max height 480p
- target frames `[50, 150, 250, 350, 450]`

## Manifest Snippet

Use this as the private manifest element once the backend and hidden data are ready:

```yaml
id: manako/DetectSnookerBallState
track: private
groundtruth_type: snooker_ball_state
weight: 0.05
window_block: 300
eval_window: 4
metrics:
  pillars:
    snooker_ball_state: 1.0
targets:
  snooker_ball_state: 0.85
baselines:
  snooker_ball_state: 0.0
first_block: <current_publish_block + 100>
```

## V0 Scoring Contract

The validator scores only requested `target_frames`. Missing requested frames score zero for that frame, and extra non-target frames are ignored. Duplicate response objects for a requested target frame are treated as contract violations: their ball lists are merged for scoring and the final frame-set score is penalized.

Labels are canonical: `cue`, `red`, `yellow`, `green`, `brown`, `blue`, `pink`, `black`. Unique colours match by label. Reds match as an unordered set using Hungarian assignment in normalized table coordinates.

States are canonical: `on_table`, `potted`, `occluded`, `unknown`. Invalid state strings are not normalized to `unknown`; they are scored as invalid predictions. `on_table` balls require `x` and `y`; `potted`, `occluded`, and `unknown` may omit coordinates.

On-table coordinate scoring uses normalized table distance with a `0.05` tolerance. A perfect coordinate match receives full coordinate credit; coordinate credit decays linearly to zero at or beyond the tolerance. On-table predictions outside that tolerance are not counted as matched balls. Red-count accuracy only counts valid red predictions, so invalid red entries cannot earn red-count credit.

Breakdown keys remain:

- `coordinate_accuracy`
- `identity_accuracy`
- `red_count_accuracy`
- `state_accuracy`
- `false_positive_score`
- `snooker_ball_state`

## External Dependencies

Backend work still required:

- `/api/challenge/v3` must emit snooker tasks with a clip URL and target frames
- `/api/tasks/{id}/ground-truth` must emit `ground_truth.frames`
- private data hosting must provide short normalized clips
- hidden annotation and QA must be produced outside this repo

Rights/leakage note: public YouTube footage is acceptable only for dry-run examples. Hidden scoring should avoid exposed timestamps and ideally use rights-cleared or non-public assets.
