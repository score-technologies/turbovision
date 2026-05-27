# Snooker Private Challenge V0 Brief

## Decision

Use the existing TurboVision private-track architecture and add a narrow snooker element:

```text
manako/DetectSnookerBallState
```

This is not a redesign. It is the smallest rigorous path to a live snooker CV challenge: validators send a short clip plus explicit target frame ids, miners return ball states for those frames, and scoring stays inside the existing private runner/weights/audit flow.

## What This Branch Provides

- Public request contract: `video_url` plus `target_frames`
- Miner response contract: `prediction.type = "snooker_ball_state"` with `frames[].balls[]`
- Snooker manifest support: `groundtruth_type = "snooker_ball_state"` and `metrics.pillars.snooker_ball_state`
- Parser support for backend `/api/challenge/v3` snooker tasks
- Miner forwarding of `target_frames`
- Runner scoring/upload support for snooker
- Spotcheck rescoring support for snooker
- Score-scaled private weighting for snooker, matching cricket behavior
- Hardened ball-state scorer with target-frame-only evaluation and red Hungarian matching
- Miner-facing spec and backend/data handoff docs
- Full local validator/private test coverage

## V0 Challenge Shape

The backend returns a normalized 20 second clip and fixed target frames:

```json
{
  "task_id": "12345",
  "payload": {
    "clip_url": "https://assets.example.com/snooker/window-abc.mp4",
    "target_frames": [50, 150, 250, 350, 450]
  }
}
```

The miner returns table-normalized ball states:

```json
{
  "challenge_id": "12345",
  "prediction": {
    "type": "snooker_ball_state",
    "frames": [
      {
        "frame": 50,
        "balls": [
          {"label": "cue", "x": 0.52, "y": 0.33, "state": "on_table"},
          {"label": "black", "state": "occluded"}
        ]
      }
    ]
  },
  "processing_time": 0.82
}
```

## Scoring Summary

Only requested target frames are scored. Missing requested frames score zero; extra non-target frames are ignored.

The scorer rewards coordinate accuracy, identity accuracy, red count accuracy, state accuracy, and low false-positive rate. It penalizes duplicate unique colours, wrong red counts, missing balls, extra balls, invalid labels, empty predictions, and `on_table` balls without coordinates.

Canonical labels are `cue`, `red`, `yellow`, `green`, `brown`, `blue`, `pink`, `black`. Reds are matched as an unordered set using Hungarian assignment.

Canonical states are `on_table`, `potted`, `occluded`, `unknown`.

## What Is Still External

The branch deliberately does not implement the external backend or publish the live manifest. The remaining launch work is:

- create 6 public bootstrap windows
- create 20 hidden scoring windows
- normalize clips to 20 seconds, 25 fps, max 480p
- annotate hidden ground truth for frames `[50, 150, 250, 350, 450]`
- host private clips and keep hidden source ids/timestamps unpublished
- implement backend `/api/challenge/v3` and `/api/tasks/{id}/ground-truth` responses using the branch contract
- publish the manifest element after backend/data are ready

## Manifest Snippet

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

## Verification

Current branch verification:

```text
python -m compileall -q scorevision tests
python -m pytest tests/private tests/validator
```

Result: `155 passed`.
