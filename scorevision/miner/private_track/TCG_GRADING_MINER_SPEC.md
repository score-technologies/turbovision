# TCG Grading Private-Track Contract

## Element

- Element ID: `manako/TCGGrading`
- Track: `private`
- Version: `1.0`
- Ground-truth type: `tcg_grading`
- Input: one PNG containing the front and back of the same card side by side

The dataset is primarily made of professionally graded Pokémon cards, with a
smaller number of cards from other trading card games. Grades use the ACE
Grading scale.

## Request

The miner must expose `POST /challenge` on port 8000:

```json
{
  "challenge_id": "76738",
  "image_url": "https://example.com/card-front-back.png"
}
```

## Response

All five numeric values are required and must be between 1 and 10. Miner
values are normalized to integers by rounding down before scoring
(`8.9 -> 8`, `8.5 -> 8`):

```json
{
  "challenge_id": "76738",
  "prediction": {
    "Header": {
      "card_grade": 6
    },
    "Grading_Features": {
      "subgrade_surface": 5,
      "subgrade_centering": 10,
      "subgrade_edges": 9,
      "subgrade_corners": 9
    }
  },
  "processing_time": 2.5
}
```

Field names and capitalization are part of the contract. Card identity,
provenance, and grader metadata present in ground truth are not prediction
targets.

## Scoring

Each field is scored independently with:

```text
field_score = max(0, 1 - abs(prediction - ground_truth) / 2)
```

After integer normalization, this gives `1.0` for an exact grade, `0.5` at
1 grade of error, and `0.0` at 2 or more grades.

The final score is:

```text
0.25 * surface
+ 0.25 * edges
+ 0.25 * corners
+ 0.10 * centering
+ 0.15 * card_grade
```

An invalid response, an omitted field, a value outside `[1, 10]`, or a timed
out request scores zero for the challenge.

## Manifest entry

```yaml
- id: manako/TCGGrading
  track: private
  groundtruth_type: tcg_grading
  challenge_type_version: "1.0"
  metrics:
    pillars:
      tcg_grading: 1.0
```
