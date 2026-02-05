from collections import defaultdict, deque
from logging import getLogger

from scorevision.utils.bittensor_helpers import (
    _first_commit_block_by_miner,
    get_subtensor,
    get_validator_indexes_from_chain,
)
from scorevision.utils.cloudflare_helpers import dataset_sv, dataset_sv_multi
from scorevision.utils.prometheus import (
    CURRENT_WINNER,
    VALIDATOR_MINERS_CONSIDERED,
    VALIDATOR_MINERS_SKIPPED_TOTAL,
    VALIDATOR_WINNER_SCORE,
)
from scorevision.utils.settings import get_settings
from scorevision.validator.models import MinerMeta
from scorevision.validator.payload import (
    build_winner_meta,
    extract_challenge_id,
    extract_elements_from_manifest,
    extract_miner_and_score,
    extract_miner_meta,
)
from scorevision.validator.scoring import (
    aggregate_challenge_scores_by_miner,
    are_similar_by_challenges,
    days_to_blocks,
    pick_winner_with_tiebreak,
    stake_of,
    weighted_median,
)

logger = getLogger(__name__)


async def get_local_fallback_winner_for_element(
    *,
    element_id: str,
    current_window_id: str,
    tail: int,
    m_min: int,
    hk_to_uid: dict[str, int],
) -> tuple[int | None, dict[int, float], dict[str, str | None] | None]:
    settings = get_settings()
    fallback_uid = settings.VALIDATOR_FALLBACK_UID
    sums: dict[int, float] = {}
    cnt: dict[int, int] = {}
    miner_meta_by_hk: dict[str, MinerMeta] = {}

    async for line in dataset_sv(tail):
        try:
            payload = line.get("payload") or {}
            if payload.get("element_id") != element_id:
                continue
            payload_window = payload.get("window_id") or (
                (payload.get("telemetry") or {}).get("window_id")
            )
            if payload_window != current_window_id:
                continue
            miner_uid, score = extract_miner_and_score(payload, hk_to_uid)
            if miner_uid is None:
                continue
            miner_meta = extract_miner_meta(payload)
            if miner_meta:
                miner_meta_by_hk[miner_meta.hotkey] = miner_meta
        except Exception:
            continue
        sums[miner_uid] = sums.get(miner_uid, 0.0) + score
        cnt[miner_uid] = cnt.get(miner_uid, 0) + 1

    if not cnt:
        logger.warning(
            "[winner] No local data for element_id=%s window_id=%s -> fallback",
            element_id,
            current_window_id,
        )
        VALIDATOR_MINERS_CONSIDERED.set(0)
        uid_to_hk = {u: hk for hk, u in hk_to_uid.items()}
        return (
            fallback_uid,
            {fallback_uid: 0.0},
            build_winner_meta(fallback_uid, uid_to_hk, miner_meta_by_hk),
        )

    elig = [uid for uid, n in cnt.items() if n >= m_min and uid in sums]
    if not elig:
        logger.warning(
            "[winner] No miner reached %d samples for element_id=%s -> fallback",
            m_min,
            element_id,
        )
        VALIDATOR_MINERS_CONSIDERED.set(0)
        uid_to_hk = {u: hk for hk, u in hk_to_uid.items()}
        return (
            fallback_uid,
            {fallback_uid: 0.0},
            build_winner_meta(fallback_uid, uid_to_hk, miner_meta_by_hk),
        )

    avg = {uid: (sums[uid] / cnt[uid]) for uid in elig}
    VALIDATOR_MINERS_CONSIDERED.set(len(elig))
    winner_uid = max(avg, key=avg.get)
    CURRENT_WINNER.set(winner_uid)
    VALIDATOR_WINNER_SCORE.set(avg.get(winner_uid, 0.0))
    uid_to_hk = {u: hk for hk, u in hk_to_uid.items()}
    return winner_uid, avg, build_winner_meta(winner_uid, uid_to_hk, miner_meta_by_hk)


async def collect_recent_challenge_scores_by_validator_miner(
    tail: int,
    validator_indexes: dict[str, str],
    hk_to_uid: dict[str, int],
    *,
    element_id: str,
    current_window_id: str,
    K: int = 25,
) -> dict[tuple[str, int], deque]:
    challenge_scores: dict[tuple[str, int], deque] = defaultdict(lambda: deque(maxlen=K))
    async for line in dataset_sv_multi(tail, validator_indexes, element_id=element_id):
        try:
            payload = line.get("payload") or {}
            if payload.get("element_id") != element_id:
                continue
            payload_window = payload.get("window_id") or (
                (payload.get("telemetry") or {}).get("window_id")
            )
            if payload_window != current_window_id:
                continue
            miner_uid, score = extract_miner_and_score(payload, hk_to_uid)
            if miner_uid is None:
                continue
            validator_hk = (line.get("hotkey") or "").strip()
            if not validator_hk:
                continue
            challenge_id = extract_challenge_id(payload)
            if not challenge_id:
                continue
            challenge_key = f"{validator_hk}:{challenge_id}"
            challenge_scores[(validator_hk, miner_uid)].append((challenge_key, float(score)))
        except Exception:
            continue
    return challenge_scores


async def get_winner_for_element(
    *,
    element_id: str,
    current_window_id: str,
    tail: int,
    m_min: int,
    blacklisted_hotkeys: set[str] | None = None,
    validator_hotkey_ss58: str | None = None,
) -> tuple[int | None, dict[int, float], dict[str, str | None] | None]:
    settings = get_settings()
    subtensor = await get_subtensor()
    netuid = settings.SCOREVISION_NETUID
    mechid = settings.SCOREVISION_MECHID
    fallback_uid = settings.VALIDATOR_FALLBACK_UID

    meta = await subtensor.metagraph(netuid, mechid=mechid)
    if blacklisted_hotkeys is None:
        blacklisted_hotkeys = set()
    hk_to_uid = {hk: i for i, hk in enumerate(meta.hotkeys) if hk not in blacklisted_hotkeys}
    uid_to_hk = {u: hk for hk, u in hk_to_uid.items()}

    stake_tensor = getattr(meta, "S", None) or getattr(meta, "stake", None)
    stake_by_hk: dict[str, float] = {}
    if stake_tensor is not None:
        for i, hk in enumerate(meta.hotkeys):
            try:
                val = stake_tensor[i]
                if hasattr(val, "item"):
                    val = float(val.item())
                else:
                    val = float(val)
            except Exception:
                val = 0.0
            stake_by_hk[hk] = max(0.0, val)
    else:
        for hk in meta.hotkeys:
            stake_by_hk[hk] = 0.0

    validator_indexes = await get_validator_indexes_from_chain(netuid)
    if not validator_indexes:
        logger.warning(
            "[winner] No validator registry found on-chain for element_id=%s",
            element_id,
        )
        return await get_local_fallback_winner_for_element(
            element_id=element_id,
            current_window_id=current_window_id,
            tail=tail,
            m_min=m_min,
            hk_to_uid=hk_to_uid,
        )

    sums_by_validator_miner: dict[tuple[str, int], float] = {}
    cnt_by_validator_miner: dict[tuple[str, int], int] = {}
    miner_meta_by_hk: dict[str, MinerMeta] = {}

    async for line in dataset_sv_multi(tail, validator_indexes, element_id=element_id):
        try:
            payload = line.get("payload") or {}
            if payload.get("element_id") != element_id:
                continue
            payload_window = payload.get("window_id") or (
                (payload.get("telemetry") or {}).get("window_id")
            )
            if payload_window != current_window_id:
                continue
            miner_uid, score = extract_miner_and_score(payload, hk_to_uid)
            if miner_uid is None:
                continue
            miner_meta = extract_miner_meta(payload)
            if miner_meta:
                miner_meta_by_hk[miner_meta.hotkey] = miner_meta
            validator_hk = (line.get("hotkey") or "").strip()
            if not validator_hk:
                continue
        except Exception:
            continue
        key = (validator_hk, miner_uid)
        sums_by_validator_miner[key] = sums_by_validator_miner.get(key, 0.0) + score
        cnt_by_validator_miner[key] = cnt_by_validator_miner.get(key, 0) + 1

    if not cnt_by_validator_miner:
        logger.warning(
            "[winner] No cross-validator data for element_id=%s window_id=%s -> fallback",
            element_id,
            current_window_id,
        )
        VALIDATOR_MINERS_CONSIDERED.set(0)
        return (
            fallback_uid,
            {fallback_uid: 0.0},
            build_winner_meta(fallback_uid, uid_to_hk, miner_meta_by_hk),
        )

    mu_by_validator_miner: dict[tuple[str, int], tuple[float, int]] = {}
    for key, n in cnt_by_validator_miner.items():
        s = sums_by_validator_miner.get(key, 0.0)
        mu_by_validator_miner[key] = (s / max(1, n), n)

    logger.info(
        "[winner] Element=%s Window=%s | Validator->Miner means: %s",
        element_id,
        current_window_id,
        ", ".join(f"{v}->{m}: mu={mu:.4f} (n={n})" for (v, m), (mu, n) in mu_by_validator_miner.items()),
    )

    a_rob, b_rob = 0.5, 0.5
    k = 2.5
    eps = 1e-3
    a_final, b_final = 1.0, 0.5

    miners_seen = set([m for (_v, m) in mu_by_validator_miner.keys()])
    scores_by_miner: dict[int, float] = {}

    for miner_uid in miners_seen:
        mus: list[float] = []
        wtilde: list[float] = []
        triplets: list[tuple[str, float, int]] = []

        for (validator_hk, mm), (mu, n) in mu_by_validator_miner.items():
            if mm != miner_uid:
                continue
            if n < m_min:
                continue
            stake = stake_by_hk.get(validator_hk, 0.0)
            wt = (stake**a_rob) * ((max(1, n)) ** b_rob)
            mus.append(mu)
            wtilde.append(wt)
            triplets.append((validator_hk, mu, n))

        if not mus or sum(max(0.0, w) for w in wtilde) <= 0:
            continue

        med = weighted_median(mus, wtilde)
        abs_dev = [abs(x - med) for x in mus]
        MAD = weighted_median(abs_dev, wtilde)
        if MAD < eps:
            MAD = eps
        thresh = k * (MAD / 0.6745)

        filtered: list[tuple[str, float, int]] = []
        rejected: list[tuple[str, float, int]] = []
        for validator_hk, mu, n in triplets:
            if abs(mu - med) <= thresh:
                filtered.append((validator_hk, mu, n))
            else:
                rejected.append((validator_hk, mu, n))

        if rejected:
            logger.info(
                "Element=%s Miner %d: rejected %d validator means (outliers)",
                element_id,
                miner_uid,
                len(rejected),
            )

        if len(filtered) < 2:
            VALIDATOR_MINERS_SKIPPED_TOTAL.labels(reason="insufficient_filtered").inc()
            continue

        num = 0.0
        den = 0.0
        for validator_hk, mu, n in filtered:
            stake = stake_by_hk.get(validator_hk, 0.0)
            wf = (stake**a_final) * ((max(1, n)) ** b_final)
            num += wf * mu
            den += wf
        if den <= 0:
            continue
        scores_by_miner[miner_uid] = num / den

    validator_uid = None
    if validator_hotkey_ss58:
        validator_uid = hk_to_uid.get(validator_hotkey_ss58)

    if validator_uid is not None and validator_uid in scores_by_miner:
        logger.info(
            "Excluding validator uid=%d from weight candidates for element_id=%s.",
            validator_uid,
            element_id,
        )
        scores_by_miner.pop(validator_uid, None)

    if not scores_by_miner:
        logger.warning(
            "[winner] No miners passed robust filtering for element_id=%s window_id=%s.",
            element_id,
            current_window_id,
        )
        VALIDATOR_MINERS_CONSIDERED.set(0)
        return (
            fallback_uid,
            {fallback_uid: 0.0},
            build_winner_meta(fallback_uid, uid_to_hk, miner_meta_by_hk),
        )

    VALIDATOR_MINERS_CONSIDERED.set(len(scores_by_miner))

    logger.info(
        "Element=%s Window=%s | Final miner means: %s",
        element_id,
        current_window_id,
        ", ".join(f"uid={m}: {s:.4f}" for m, s in sorted(scores_by_miner.items())),
    )

    winner_uid = max(scores_by_miner, key=scores_by_miner.get)
    logger.info(
        "Element=%s Window=%s | Provisional winner uid=%d S=%.4f over last %d blocks",
        element_id,
        current_window_id,
        winner_uid,
        scores_by_miner[winner_uid],
        tail,
    )

    if settings.SCOREVISION_WINDOW_TIEBREAK_ENABLE:
        try:
            challenge_scores_by_validator_miner = await collect_recent_challenge_scores_by_validator_miner(
                tail=tail,
                validator_indexes=validator_indexes,
                hk_to_uid=hk_to_uid,
                element_id=element_id,
                current_window_id=current_window_id,
                K=settings.SCOREVISION_WINDOW_K_PER_VALIDATOR,
            )
            challenge_scores_by_miner = aggregate_challenge_scores_by_miner(challenge_scores_by_validator_miner)
            first_commit_block_by_hk = await _first_commit_block_by_miner(netuid)
            candidate_uids = set(scores_by_miner.keys())
            final_uid = pick_winner_with_tiebreak(
                winner_uid,
                uid_to_hk=uid_to_hk,
                challenge_scores_by_miner=challenge_scores_by_miner,
                candidate_uids=candidate_uids,
                delta_abs=settings.SCOREVISION_WINDOW_DELTA_ABS,
                delta_rel=settings.SCOREVISION_WINDOW_DELTA_REL,
                first_commit_block_by_hk=first_commit_block_by_hk,
                min_common_challenges=5,
            )
            if final_uid != winner_uid:
                logger.info(
                    "[window-tiebreak] Element=%s | Provisional=%d -> Final=%d",
                    element_id,
                    winner_uid,
                    final_uid,
                )
                winner_uid = final_uid
        except Exception as e:
            logger.warning("[window-tiebreak] Element=%s disabled due to error: %s", element_id, e)

    logger.info(
        "Element=%s Window=%s | Winner uid=%d (after tiebreak) over last %d blocks",
        element_id,
        current_window_id,
        winner_uid,
        tail,
    )

    TARGET_UID = 6
    final_score = float(scores_by_miner.get(winner_uid, 0.0) or 0.0)

    if abs(final_score) <= 1e-12:
        logger.info(
            "Final winner uid=%d has score=%.6f -> forcing to TARGET_UID=%d",
            winner_uid,
            final_score,
            TARGET_UID,
        )
        winner_uid = TARGET_UID
        final_score = float(scores_by_miner.get(TARGET_UID, 0.0) or 0.0)

    CURRENT_WINNER.set(winner_uid)
    VALIDATOR_WINNER_SCORE.set(final_score)

    return winner_uid, scores_by_miner, build_winner_meta(winner_uid, uid_to_hk, miner_meta_by_hk)

