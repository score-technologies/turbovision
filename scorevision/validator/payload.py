from scorevision.validator.models import MinerMeta


def extract_miner_and_score(
    payload: dict, hk_to_uid: dict[str, int]
) -> tuple[int | None, float | None]:
    try:
        telemetry = payload.get("telemetry") or {}
        miner_info = telemetry.get("miner") or {}
        miner_hk = (miner_info.get("hotkey") or "").strip()
        if not miner_hk or miner_hk not in hk_to_uid:
            return None, None
        metrics = payload.get("metrics") or {}
        score = metrics.get("composite_score", payload.get("composite_score", 0.0))
        score = float(score)
        miner_uid = hk_to_uid[miner_hk]
        return miner_uid, score
    except Exception:
        return None, None


def extract_miner_meta(payload: dict) -> MinerMeta | None:
    try:
        telemetry = payload.get("telemetry") or {}
        miner_info = telemetry.get("miner") or {}
        miner_hk = (miner_info.get("hotkey") or "").strip()
        if not miner_hk:
            return None
        return MinerMeta(
            hotkey=miner_hk,
            chute_id=miner_info.get("chute_id"),
            slug=miner_info.get("slug"),
        )
    except Exception:
        return None


def extract_challenge_id(payload: dict) -> str | None:
    meta = payload.get("meta") or {}
    telemetry = payload.get("telemetry") or {}
    cand = (
        meta.get("task_id")
        or payload.get("task_id")
        or telemetry.get("task_id")
        or meta.get("challenge_id")
        or payload.get("challenge_id")
        or telemetry.get("challenge_id")
        or payload.get("job_id")
        or telemetry.get("job_id")
    )
    if cand is None:
        return None
    try:
        s = str(cand).strip()
        return s or None
    except Exception:
        return None


def extract_elements_from_manifest(manifest) -> list[tuple[str, float, int | float | None]]:
    elements = getattr(manifest, "elements", None) or []
    out: list[tuple[str, float, int | float | None]] = []
    for elem in elements:
        eid = None
        weight = None
        eval_window = None
        if hasattr(elem, "element_id"):
            eid = getattr(elem, "element_id")
        elif hasattr(elem, "id"):
            eid = getattr(elem, "id")
        elif isinstance(elem, dict):
            eid = elem.get("element_id") or elem.get("id")
        if hasattr(elem, "weight"):
            weight = getattr(elem, "weight")
        elif isinstance(elem, dict):
            weight = elem.get("weight")
        if hasattr(elem, "eval_window"):
            eval_window = getattr(elem, "eval_window")
        elif isinstance(elem, dict):
            eval_window = elem.get("eval_window")
        if eid is None:
            continue
        try:
            eid_str = str(eid)
        except Exception:
            continue
        try:
            w = float(weight) if weight is not None else 0.0
        except Exception:
            w = 0.0
        ew = None
        if eval_window is not None:
            try:
                ew = float(eval_window)
                if ew.is_integer():
                    ew = int(ew)
            except Exception:
                ew = None
        out.append((eid_str, w, ew))
    return out


def build_winner_meta(
    winner_uid: int | None,
    uid_to_hk: dict[int, str],
    miner_meta_by_hk: dict[str, MinerMeta],
) -> dict[str, str | None] | None:
    if winner_uid is None:
        return None
    winner_hk = uid_to_hk.get(winner_uid)
    if not winner_hk:
        return None
    meta = miner_meta_by_hk.get(winner_hk)
    if meta:
        return meta.to_dict()
    return {"hotkey": winner_hk, "chute_id": None, "slug": None}

