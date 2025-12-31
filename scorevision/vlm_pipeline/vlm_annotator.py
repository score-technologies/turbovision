from asyncio import Semaphore, gather
from logging import getLogger

import cv2
import numpy as np
from numpy import ndarray

from scorevision.vlm_pipeline.utils.data_models import PseudoGroundTruth
from scorevision.vlm_pipeline.utils.llm_vlm import async_vlm_api, retry_api, VLMProvider
from scorevision.vlm_pipeline.utils.response_models import (
    FrameAnnotation,
    BoundingBox,
    ShirtColor,
    TEAM1_SHIRT_COLOUR,
    TEAM2_SHIRT_COLOUR,
)
from scorevision.vlm_pipeline.domain_specific_schemas.football import (
    STEP1_SCHEMA,
    STEP1_SYSTEM,
    STEP1_USER,
    build_step2_schema_and_prompts,
    build_step3_system_and_user,
    normalize_palette_roles,
    FOOTBALL_DEFAULT_CATEGORY,
    FOOTBALL_CATEGORY_CONFIDENCE,
    FOOTBALL_REASON_PREFIX,
)
from scorevision.vlm_pipeline.domain_specific_schemas.football import (
    Person as ObjectOfInterest,
)
from scorevision.utils.async_clients import get_semaphore

logger = getLogger(__name__)

# --- Models (via provider) ---
QWEN_MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"
INTERNVL_MODEL = QWEN_MODEL
#INTERNVL_MODEL = "OpenGVLab/InternVL3-78B-TEE"


# -------------------- utils: image + boxes --------------------
def _ensure_bgr_contiguous(img: ndarray) -> ndarray:
    """Garantit BGR uint8 C-contiguous pour OpenCV draw ops."""
    if img is None:
        raise ValueError("image is None")
    out = img
    if out.ndim == 2:  # grayscale -> BGR
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    elif out.ndim == 3 and out.shape[2] == 4:  # BGRA -> BGR
        out = cv2.cvtColor(out, cv2.COLOR_BGRA2BGR)
    elif out.ndim != 3 or out.shape[2] != 3:
        raise ValueError(f"Unexpected image shape {out.shape}")
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    if not out.flags["C_CONTIGUOUS"]:
        out = np.ascontiguousarray(out)
    return out


def _flow_to_bgr(flow_rgb_frame: ndarray | None) -> ndarray | None:
    if flow_rgb_frame is None:
        return None
    fr = flow_rgb_frame
    if fr.ndim == 3 and fr.shape[2] == 3:
        fr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
    elif fr.ndim == 2:
        fr = cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR)
    fr = np.clip(fr, 0, 255).astype(np.uint8)
    return _ensure_bgr_contiguous(fr)


def _clamp_box(x1, y1, x2, y2, w, h):
    x1 = int(max(0, min(x1, w - 1)))
    y1 = int(max(0, min(y1, h - 1)))
    x2 = int(max(0, min(x2, w)))
    y2 = int(max(0, min(y2, h)))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _put_label(img, text, x, y):
    x = int(x)
    y = int(y)
    w = 8 + 8 * len(text)
    cv2.rectangle(img, (x, y - 18), (x + w, y), (0, 0, 0), -1)
    cv2.putText(
        img,
        text,
        (x + 4, y - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _draw_boxes_with_idx(img: np.ndarray, boxes: list[list[int]]) -> np.ndarray:
    out = _ensure_bgr_contiguous(img.copy())
    H, W = out.shape[:2]
    for i, b in enumerate(boxes or []):
        if not isinstance(b, (list, tuple)) or len(b) != 4:
            continue
        try:
            x1, y1, x2, y2 = [int(v) for v in b]
        except Exception:
            continue
        clamped = _clamp_box(x1, y1, x2, y2, W, H)
        if clamped is None:
            continue
        x1, y1, x2, y2 = clamped
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        _put_label(out, f"{i}", x1, max(18, y1))
    return out


def _bbox_area(b):
    return max(0, (b[2] - b[0])) * max(0, (b[3] - b[1]))


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / max(1, (area_a + area_b - inter))


def _nms(boxes: list[list[int]], iou_thr=0.7) -> list[list[int]]:
    if not boxes:
        return []
    areas = [_bbox_area(b) for b in boxes]
    order = sorted(range(len(boxes)), key=lambda i: areas[i], reverse=True)
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if _iou(boxes[i], boxes[j]) < iou_thr]
    return [boxes[i] for i in keep]


def _merge_union(
    boxes_a: list[list[int]], boxes_b: list[list[int]], iou_merge=0.6
) -> list[list[int]]:
    out = boxes_a[:]
    for b in boxes_b:
        matched = False
        for i, a in enumerate(out):
            if _iou(a, b) >= iou_merge:
                out[i] = a if _bbox_area(a) <= _bbox_area(b) else b
                matched = True
                break
        if not matched:
            out.append(b)
    return out


# -------------------- VLM wrappers per step --------------------
async def _vlm_call_with_model(
    images, sys_prompt, schema, user_prompt, provider, model, temperature=None
) -> dict:
    sys_full = sys_prompt + "\nJSON SCHEMA:\n" + schema
    return await async_vlm_api(
        images=images,
        system_prompt=sys_full,
        user_prompt=user_prompt,
        provider=provider,
        model_override=model,
        temperature_override=temperature,
    )


async def _step1_detect_persons_and_ball_double_qwen(
    raw_bgr, flow_bgr, provider: VLMProvider
) -> dict:
    raw_bgr = _ensure_bgr_contiguous(raw_bgr)
    imgs = [raw_bgr] + ([flow_bgr] if flow_bgr is not None else [])
    r1 = await _vlm_call_with_model(
        imgs,
        STEP1_SYSTEM,
        STEP1_SCHEMA,
        STEP1_USER,
        provider,
        QWEN_MODEL,
        temperature=0.0,
    )
    r2 = await _vlm_call_with_model(
        imgs,
        STEP1_SYSTEM,
        STEP1_SCHEMA,
        STEP1_USER,
        provider,
        QWEN_MODEL,
        temperature=0.0,
    )

    def _clean(r):
        persons = []
        for b in r.get("persons") or []:
            if not isinstance(b, (list, tuple)) or len(b) != 4:
                continue
            try:
                x1, y1, x2, y2 = [int(v) for v in b]
            except Exception:
                continue
            persons.append([x1, y1, x2, y2])
        ball = r.get("ball") or {}
        present = bool(ball.get("present", False))
        bbox_ball = None
        if (
            present
            and isinstance(ball.get("bbox"), (list, tuple))
            and len(ball["bbox"]) == 4
        ):
            try:
                x1, y1, x2, y2 = [int(v) for v in ball["bbox"]]
                if x2 > x1 and y2 > y1:
                    bbox_ball = [x1, y1, x2, y2]
                else:
                    present = False
            except Exception:
                present = False
        return persons, {"present": present, "bbox": bbox_ball}

    p1, b1 = _clean(r1)
    p2, b2 = _clean(r2)

    merged = _merge_union(p1, p2, iou_merge=0.6)
    merged = _nms(merged, iou_thr=0.7)

    ball = {"present": False, "bbox": None}
    cands = []
    if b1.get("present") and b1.get("bbox"):
        cands.append(b1["bbox"])
    if b2.get("present") and b2.get("bbox"):
        cands.append(b2["bbox"])
    if cands:
        cands.sort(key=_bbox_area)
        ball = {"present": True, "bbox": cands[0]}
    return {"persons": merged, "ball": ball}


async def _step2_palette_internvl(raw_bgr, provider: VLMProvider) -> dict:
    raw_bgr = _ensure_bgr_contiguous(raw_bgr)
    schema_json_str, sys_prompt, user_prompt = build_step2_schema_and_prompts()
    res = await _vlm_call_with_model(
        [raw_bgr],
        sys_prompt,
        schema_json_str,
        user_prompt,
        provider,
        INTERNVL_MODEL,
        temperature=0.0,
    )
    return normalize_palette_roles(res)


async def _step3_assign_classes_from_palette(
    raw_bgr, persons_overlay_bgr, s1_persons, palette_json, provider: VLMProvider
) -> dict:
    raw_bgr = _ensure_bgr_contiguous(raw_bgr)
    persons_overlay_bgr = _ensure_bgr_contiguous(persons_overlay_bgr)
    sys_prompt, user_prompt = build_step3_system_and_user(len(s1_persons), palette_json)
    res = await _vlm_call_with_model(
        [raw_bgr, persons_overlay_bgr],
        sys_prompt,
        "{}",
        user_prompt,
        provider,
        INTERNVL_MODEL,
        temperature=0.0,
    )

    n = len(s1_persons)
    assigned = {}
    for a in res.get("assignments") or []:
        try:
            idx = int(a.get("index"))
        except Exception:
            continue
        if not (0 <= idx < n):
            continue
        cls = a.get("class")
        team_id = a.get("team_id")
        if cls not in {"player", "referee", "goalkeeper"}:
            continue
        if cls == "player" and team_id not in (1, 2):
            continue
        assigned[idx] = {"class": cls, "team_id": team_id}

    for idx in range(n):
        if idx not in assigned:
            assigned[idx] = {"class": "player", "team_id": 1}

    gk_used = False
    objects = []
    for idx in range(n):
        box = s1_persons[idx]
        entry = assigned[idx]
        if entry["class"] == "player":
            objects.append(
                {"class": "player", "bbox": box, "team_id": int(entry["team_id"])}
            )
        elif entry["class"] == "referee":
            objects.append({"class": "referee", "bbox": box})
        elif entry["class"] == "goalkeeper":
            if not gk_used:
                objects.append({"class": "goalkeeper", "bbox": box})
                gk_used = True
            else:
                objects.append({"class": "player", "bbox": box, "team_id": 1})
    return {"objects": objects}


def _to_shirtcolor(name: str | None, default: ShirtColor) -> ShirtColor:
    if not name:
        return default
    name = str(name).strip().lower()
    for c in ShirtColor:
        if c.value == name:
            return c
    return default


def _objects_to_frameannotation(
    objects: list[dict], ball_obj: dict, palette_roles: list[dict]
) -> FrameAnnotation:
    """ """
    role_to_color = {
        r.get("role"): r.get("color") for r in (palette_roles or []) if r.get("role")
    }
    t1_color = _to_shirtcolor(role_to_color.get("team1"), default=TEAM1_SHIRT_COLOUR)
    t2_color = _to_shirtcolor(role_to_color.get("team2"), default=TEAM2_SHIRT_COLOUR)
    ref_color = _to_shirtcolor(role_to_color.get("referee"), default=ShirtColor.BLACK)
    gk_color = _to_shirtcolor(role_to_color.get("goalkeeper"), default=ShirtColor.OTHER)

    bboxes: list[BoundingBox] = []
    for o in objects:
        if not isinstance(o.get("bbox"), (list, tuple)) or len(o["bbox"]) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in o["bbox"]]
        cls = o["class"]
        if cls == "player":
            team_id = int(o.get("team_id") or 1)
            cluster = t1_color if team_id == 1 else t2_color
            bboxes.append(
                BoundingBox(
                    bbox_2d=(x1, y1, x2, y2),
                    label=ObjectOfInterest.PLAYER,
                    cluster_id=cluster,
                )
            )
        elif cls == "referee":
            bboxes.append(
                BoundingBox(
                    bbox_2d=(x1, y1, x2, y2),
                    label=ObjectOfInterest.REFEREE,
                    cluster_id=ref_color,
                )
            )
        elif cls == "goalkeeper":
            bboxes.append(
                BoundingBox(
                    bbox_2d=(x1, y1, x2, y2),
                    label=ObjectOfInterest.GOALIE,
                    cluster_id=gk_color,
                )
            )

    if (
        ball_obj.get("present")
        and isinstance(ball_obj.get("bbox"), (list, tuple))
        and len(ball_obj["bbox"]) == 4
    ):
        x1, y1, x2, y2 = [int(v) for v in ball_obj["bbox"]]
        bboxes.append(
            BoundingBox(
                bbox_2d=(x1, y1, x2, y2),
                label=ObjectOfInterest.BALL,
                cluster_id=ShirtColor.OTHER,
            )
        )

    return FrameAnnotation(
        bboxes=bboxes,
        category=FOOTBALL_DEFAULT_CATEGORY,
        confidence=FOOTBALL_CATEGORY_CONFIDENCE,
        reason=f"{FOOTBALL_REASON_PREFIX} players/referees/goalkeeper via palette + ball if present.",
    )


# -------------------- API publique (appelée par runner) --------------------
@retry_api
async def generate_annotations_for_select_frame(
    video_name: str,
    frame_number: int,
    frame: ndarray,
    flow_frame: ndarray,
    provider: VLMProvider = VLMProvider.PRIMARY,
) -> PseudoGroundTruth | None:

    semaphore = get_semaphore()
    async with semaphore:
        try:
            raw_bgr = _ensure_bgr_contiguous(frame)
            flow_bgr = _flow_to_bgr(flow_frame) if flow_frame is not None else None

            s1 = await _step1_detect_persons_and_ball_double_qwen(
                raw_bgr, flow_bgr, provider=provider
            )
            persons = s1["persons"]
            ball = s1["ball"]
            if not persons and not (ball.get("present") and ball.get("bbox")):
                raise ValueError("No persons/ball detected")

            persons_vis = _draw_boxes_with_idx(raw_bgr, persons)

            s2 = await _step2_palette_internvl(raw_bgr, provider=provider)

            s3 = await _step3_assign_classes_from_palette(
                raw_bgr, persons_vis, persons, s2, provider=provider
            )
            objects = s3["objects"]

            annotation = _objects_to_frameannotation(objects, ball, s2.get("roles", []))

        except Exception as e:
            logger.error(
                f"VLM failed to generate pseudo-GT annotations for frame {frame_number}: {e}"
            )
            return None

        if not any(annotation.bboxes):
            logger.error("No annotations were generated for this frame")
            return None

        return PseudoGroundTruth(
            video_name=video_name,
            frame_number=frame_number,
            spatial_image=frame,
            temporal_image=flow_frame,
            annotation=annotation,
        )


async def generate_annotations_for_select_frames(
    video_name: str,
    frames: list[ndarray],
    flow_frames: list[ndarray],
    frame_numbers: list[int],
) -> list[PseudoGroundTruth]:
    tasks = [
        generate_annotations_for_select_frame(
            video_name=video_name,
            frame_number=frame_number,
            frame=frame,
            flow_frame=flow_frame,
        )
        for frame_number, frame, flow_frame in zip(
            frame_numbers, frames, flow_frames, strict=True
        )
    ]
    results = await gather(*tasks, return_exceptions=True)
    annotations: list[PseudoGroundTruth] = []
    for result in results:
        if isinstance(result, PseudoGroundTruth):
            annotations.append(result)
        else:
            logger.error(result)
    return annotations
