from enum import Enum
import json
from pathlib import Path

from numpy import ndarray
from cv2 import imread

class ShirtColor(Enum):
    WHITE = "white"
    BLACK = "black"
    RED = "red"
    BLUE = "blue"
    YELLOW = "yellow"
    GREEN = "green"
    ORANGE = "orange"
    PURPLE = "purple"
    MAROON = "maroon"
    PINK = "pink"
    GREY = "grey"
    BROWN = "brown"
    GOLD = "gold"
    SILVER = "silver"
    TURQUOISE = "turquoise"
    OTHER = "other"


TEAM1_SHIRT_COLOUR = ShirtColor.WHITE
TEAM2_SHIRT_COLOUR = ShirtColor.BLACK

# =======================
# GEOM
# =======================
FOOTBALL_KEYPOINTS: list[tuple[int, int]] = [
    (5, 5),  # 1
    (5, 140),  # 2
    (5, 250),  # 3
    (5, 430),  # 4
    (5, 540),  # 5
    (5, 675),  # 6
    # -------------
    (55, 250),  # 7
    (55, 430),  # 8
    # -------------
    (110, 340),  # 9
    # -------------
    (165, 140),  # 10
    (165, 270),  # 11
    (165, 410),  # 12
    (165, 540),  # 13
    # -------------
    (527, 5),  # 14
    (527, 253),  # 15
    (527, 433),  # 16
    (527, 675),  # 17
    # -------------
    (888, 140),  # 18
    (888, 270),  # 19
    (888, 410),  # 20
    (888, 540),  # 21
    # -------------
    (940, 340),  # 22
    # -------------
    (998, 250),  # 23
    (998, 430),  # 24
    # -------------
    (1045, 5),  # 25
    (1045, 140),  # 26
    (1045, 250),  # 27
    (1045, 430),  # 28
    (1045, 540),  # 29
    (1045, 675),  # 30
    # -------------
    (435, 340),  # 31
    (615, 340),  # 32
]

INDEX_KEYPOINT_CORNER_BOTTOM_LEFT = 5
INDEX_KEYPOINT_CORNER_BOTTOM_RIGHT = 29
INDEX_KEYPOINT_CORNER_TOP_LEFT = 0
INDEX_KEYPOINT_CORNER_TOP_RIGHT = 24


def football_pitch() -> ndarray:
    current_dir = Path(__file__).parent
    return imread(
        str(current_dir/"football_pitch_template.png")
    )


# =======================
# Enums
# =======================
class Person(Enum):
    BALL = "ball"
    GOALIE = "goalkeeper"
    PLAYER = "player"
    REFEREE = "referee"


OBJECT_ID_LOOKUP = {
    0: Person.BALL,
    1: Person.GOALIE,
    2: Person.PLAYER,
    3: Person.REFEREE,
    6: "team 1",
    7: "team 2",
}


class Action(Enum):
    NONE = "No Special Action"
    PENALTY = "Penalty"
    KICK_OFF = "Kick-off"
    GOAL = "Goal"
    SUB = "Substitution"
    OFFSIDE = "Offside"
    SHOT_ON_TARGET = "Shots on target"
    SHOT_OFF_TARGET = "Shots off target"
    CLEARANCE = "Clearance"
    BALL_OOP = "Ball out of play"
    THROW_IN = "Throw-in"
    FOUL = "Foul"
    INDIRECT_FREE_KICK = "Indirect free-kick"
    DIRECT_FREE_KICK = "Direct free-kick"
    CORNER = "Corner"
    YELLOW_CARD = "Yellow card"
    RED_CARD = "Red card"
    YELLOW_RED_CARD = "Yellow->red card"


# =======================
# colors
# =======================

# couleurs autorisées (Enum -> str)
FOOTBALL_COLOR_NAMES = [c.value for c in ShirtColor]


def map_role_color_to_shirtcolor(name: str | None, default: ShirtColor) -> ShirtColor:
    if not name:
        return default
    name = name.lower().strip()
    for c in ShirtColor:
        if c.value == name:
            return c
    return default


# =======================
# Prompts
# =======================

# STEP 1: detection
STEP1_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "persons": {
            "type": "array",
            "items": {
                "type": "array",
                "minItems": 4,
                "maxItems": 4,
                "items": {"type": "integer"},
            },
        },
        "ball": {
            "type": "object",
            "properties": {
                "present": {"type": "boolean"},
                "bbox": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 4,
                    "maxItems": 4,
                },
            },
            "required": ["present"],
        },
    },
    "required": ["persons", "ball"],
}
STEP1_SCHEMA = json.dumps(STEP1_JSON_SCHEMA, ensure_ascii=False)

STEP1_SYSTEM = """You are a meticulous SOCCER on-pitch BOX annotator.
Return ONLY valid compact JSON matching the schema (integers only).
GOAL: Detect ALL people ON THE PITCH (tight boxes) and the BALL (0 or 1).
Sources: #1 RAW (colors allowed), #2 OPTICAL-FLOW (motion hint, ignore its colors).
Quality: cover far/mid/left/right; tight head-torso-legs; deduplicate IoU>0.7; ignore off-pitch staff/crowd."""
STEP1_USER = """Return ONLY:
{"persons":[[x1,y1,x2,y2],...],"ball":{"present":bool,"bbox":[x1,y1,x2,y2]?}}"""


# STEP 2: Context
def build_step2_schema_and_prompts() -> tuple[str, str, str]:
    schema = {
        "type": "object",
        "properties": {
            "roles": [
                {
                    "type": "object",
                    "properties": {
                        "role": {
                            "type": "string",
                            "enum": ["team1", "team2", "referee", "goalkeeper"],
                        },
                        "color": {"type": "string", "enum": FOOTBALL_COLOR_NAMES},
                        "present": {"type": "boolean"},
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                    },
                    "required": ["role", "color"],
                }
            ]
        },
        "required": ["roles"],
    }
    sys_prompt = """You are a soccer COLOR PALETTE extractor.
Return ONLY JSON per schema.

INPUT
- RAW frame ONLY (reference for colors). Ignore crowd/staff.
TASK
- Report the dominant jersey COLOR for each visible ROLE among:
  team1, team2, referee, goalkeeper.
CONSTRAINTS
- ≤1 color for referee; ≤1 color for goalkeeper.
- team1 and team2 MUST be distinct if both present.
- If a role is not visible, set present=false and still keep the role with a best guess or omit it (we'll mark present=false).
- Colors must come from the provided enum; prefer named colors over 'other' when plausible.
OUTPUT
- roles: list of {role, color, present?, confidence?}.
"""
    user_prompt = (
        "Return ONLY a compact JSON like:\n"
        '{"roles":[{"role":"team1","color":"red","present":true,"confidence":0.9},'
        ' {"role":"team2","color":"yellow","present":true,"confidence":0.88},'
        ' {"role":"referee","color":"black","present":true,"confidence":0.95},'
        ' {"role":"goalkeeper","color":"blue","present":false}]}'
    )
    return json.dumps(schema, ensure_ascii=False), sys_prompt, user_prompt


def normalize_palette_roles(raw_res: dict) -> dict:
    roles = {}
    for it in raw_res.get("roles") or []:
        role = it.get("role")
        color = (it.get("color") or "other").lower()
        if role not in {"team1", "team2", "referee", "goalkeeper"}:
            continue
        if color not in FOOTBALL_COLOR_NAMES:
            color = "other"
        present = bool(it.get("present", True))
        conf = (
            float(it.get("confidence", 0.0))
            if isinstance(it.get("confidence"), (int, float))
            else 0.0
        )
        # keep best by confidence per role
        if role not in roles or conf > roles[role]["confidence"]:
            roles[role] = {
                "role": role,
                "color": color,
                "present": present,
                "confidence": conf,
            }
    # ensure distinct team colors if both exist
    if (
        "team1" in roles
        and "team2" in roles
        and roles["team1"]["color"] == roles["team2"]["color"]
    ):
        roles["team2"]["color"] = "other"
    return {"roles": list(roles.values())}


# STEP 3: Class attribution
def build_step3_system_and_user(n_indices: int, palette_json: dict) -> tuple[str, str]:
    sys_prompt = """You are a strict on-pitch ROLE assigner.
Return ONLY JSON per schema.

INPUTS
- A: RAW frame (context; colors taken from RAW ONLY).
- B: An overlay with STEP1 person boxes, each labeled with its index 0..N-1.
- C: A PALETTE JSON that maps roles to colors (team1/team2/referee/goalkeeper) from STEP2.

TASK
- For EACH index, assign exactly one class in {"player","referee","goalkeeper"}.
- If class=="player", also assign team_id in {1,2} based on closest jersey match to team1/team2 colors.
HARD CONSTRAINTS
- Use the EXACT STEP1 indices; DO NOT drop or duplicate any index.
- DO NOT create boxes or indices not in 0..N-1.
- At most 1 goalkeeper in the whole image.
- Total referees 0..3 (same outfit color).
- Maximize inter-team separation according to STEP2 colors; be robust to shading/lighting.
"""
    idxs = ",".join(str(i) for i in range(n_indices))
    user_prompt = (
        f"Indices to classify: [{idxs}].\n"
        f"STEP2 palette JSON:\n{json.dumps(palette_json, ensure_ascii=False)}\n"
        "Return ONLY a compact JSON like:\n"
        '{"assignments":[{"index":0,"class":"player","team_id":1},'
        ' {"index":1,"class":"referee"},'
        ' {"index":2,"class":"goalkeeper"}]}'
    )
    return sys_prompt, user_prompt


# =======================
# Default annotation values
# =======================
FOOTBALL_DEFAULT_CATEGORY = Action.NONE
FOOTBALL_CATEGORY_CONFIDENCE = 100
FOOTBALL_REASON_PREFIX = "Auto pseudo-GT:"
