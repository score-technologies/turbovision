from time import time
from logging import getLogger
from json import dumps, loads
import asyncio
import os
from pathlib import Path
import aiohttp
import hashlib
from urllib.parse import urljoin, urlparse

from aiobotocore.session import get_session
from botocore.config import Config as BotoConfig

from scorevision.utils.data_models import SVChallenge, SVRunOutput, SVEvaluation
from scorevision.utils.settings import get_settings
from scorevision.utils.signing import _sign_batch
from async_substrate_interface.errors import SubstrateRequestException
from scorevision.utils.bittensor_helpers import get_subtensor, reset_subtensor
from scorevision.utils.prometheus import (
    VALIDATOR_DATASET_LINES_TOTAL,
    VALIDATOR_DATASET_FETCH_ERRORS_TOTAL,
)

logger = getLogger(__name__)


async def _safe_get_current_block(st, rid: str, retries: int = 1):
    attempt = 0
    while True:
        try:
            block = await asyncio.wait_for(st.get_current_block(), timeout=5.0)
            return int(block), st
        except (
            asyncio.TimeoutError,
            SubstrateRequestException,
            ConnectionError,
            KeyError,
        ) as e:
            attempt += 1
            logger.warning(
                "[emit:%s] get_current_block failed (%s); resetting subtensor", rid, e
            )
            reset_subtensor()
            if attempt > retries:
                raise
            st = await get_subtensor()


def _loads(b):
    return loads(b.decode() if isinstance(b, (bytes, bytearray)) else b)


def _dumps(o) -> bytes:
    return dumps(o, separators=(",", ":")).encode()


settings = get_settings()

CACHE_DIR = settings.SCOREVISION_CACHE_DIR
CACHE_DIR.mkdir(parents=True, exist_ok=True)

import nacl.signing, nacl.encoding


from substrateinterface import Keypair
import hashlib
import uuid


def _verify_signature(hk_ss58: str, payload: str, sig_hex: str) -> bool:
    try:
        if not hk_ss58 or not sig_hex:
            return False
        sig_hex = sig_hex[2:] if sig_hex.startswith("0x") else sig_hex
        kp = Keypair(ss58_address=hk_ss58)
        return kp.verify(payload.encode("utf-8"), bytes.fromhex(sig_hex))
    except Exception:
        return False


def _results_prefix(ns: str | None = None) -> str:
    """ """
    ns = (ns or os.getenv("SCOREVISION_RESULTS_PREFIX") or "results").strip().strip("/")
    return f"scorevision/{ns}/"


async def _index_list() -> list[str]:
    """ """
    settings = get_settings()
    index_key = "scorevision/index.json"

    async with get_s3_client() as c:
        try:
            r = await c.get_object(Bucket=settings.R2_BUCKET, Key=index_key)
            body = await r["Body"].read()
            return loads(body)
        except c.exceptions.NoSuchKey:
            return []


async def _put_json_object(key: str, obj) -> None:
    s = get_settings()
    data = dumps(obj, separators=(",", ":")).encode()
    async with get_s3_client() as c:
        await c.put_object(
            Bucket=s.R2_BUCKET, Key=key, Body=data, ContentType="application/json"
        )
    if "/evaluation/" in key:
        await _index_add_if_new(key)


async def _cache_shard(key: str, sem: asyncio.Semaphore) -> Path:
    """ """
    settings = get_settings()
    out = CACHE_DIR / (Path(key).name + ".jsonl")
    mod = out.with_suffix(".modified")

    async with sem, get_s3_client() as c:
        try:
            head = await c.head_object(Bucket=settings.R2_BUCKET, Key=key)
            lm = head["LastModified"].isoformat()
        except c.exceptions.NoSuchKey:
            VALIDATOR_DATASET_FETCH_ERRORS_TOTAL.labels(
                stage="cache_head_missing"
            ).inc()
            return out

        if out.exists() and mod.exists() and mod.read_text().strip() == lm:
            return out

        obj = await c.get_object(Bucket=settings.R2_BUCKET, Key=key)
        body = await obj["Body"].read()
        arr = _loads(body)

    tmp = out.with_suffix(".tmp")
    with tmp.open("wb") as f:
        for line in arr:
            f.write(_dumps(line))
            f.write(b"\n")
    os.replace(tmp, out)
    mod.write_text(lm)
    return out


def build_public_index_url_from_public_base(public_base: str | None) -> str | None:
    """Turn a Public Development URL (e.g. https://pub-xxxx.r2.dev)
    into the scorevision index URL (.../scorevision/index.json)."""
    if not public_base:
        return None
    return public_base.rstrip("/") + "/scorevision/index.json"


def normalize_index_url(url: str | None) -> str | None:
    """Ensure a URL points to the index.json (accepts either a base or a full URL)."""
    if not url:
        return None
    if url.strip().endswith(".json"):
        return url.strip()
    return url.rstrip("/") + "/scorevision/index.json"


def get_s3_client():
    settings = get_settings()
    if not (
        settings.R2_ACCOUNT_ID.get_secret_value()
        and settings.R2_WRITE_ACCESS_KEY_ID.get_secret_value()
        and settings.R2_WRITE_SECRET_ACCESS_KEY.get_secret_value()
    ):
        raise RuntimeError("R2 credentials not set")
    sess = get_session()
    return sess.create_client(
        "s3",
        endpoint_url=f"https://{settings.R2_ACCOUNT_ID.get_secret_value()}.r2.cloudflarestorage.com",
        aws_access_key_id=settings.R2_WRITE_ACCESS_KEY_ID.get_secret_value(),
        aws_secret_access_key=settings.R2_WRITE_SECRET_ACCESS_KEY.get_secret_value(),
        config=BotoConfig(max_pool_connections=settings.R2_CONCURRENCY),
    )


def _r2_enabled() -> bool:
    settings = get_settings()
    return bool(
        settings.R2_ACCOUNT_ID.get_secret_value()
        and settings.R2_WRITE_ACCESS_KEY_ID.get_secret_value()
        and settings.R2_WRITE_SECRET_ACCESS_KEY.get_secret_value()
    )


async def _index_add_if_new(key: str) -> None:
    settings = get_settings()
    local_index = settings.SCOREVISION_LOCAL_ROOT / "index.json"
    index_key = "scorevision/index.json"

    async with get_s3_client() as c:
        try:
            r = await c.get_object(Bucket=settings.R2_BUCKET, Key=index_key)
            items = set(loads(await r["Body"].read()))
        except c.exceptions.NoSuchKey:
            items = set()
        if key not in items:
            items.add(key)
            await c.put_object(
                Bucket=settings.R2_BUCKET,
                Key=index_key,
                Body=dumps(sorted(items)),
                ContentType="application/json",
            )


async def sink_sv(block: int, lines: list[dict]) -> tuple[str, list[dict]]:
    settings = get_settings()

    settings.SCOREVISION_LOCAL_ROOT.mkdir(parents=True, exist_ok=True)

    if not lines:
        return "", []

    payloads = [
        dumps(l.get("payload") or {}, sort_keys=True, separators=(",", ":"))
        for l in lines
    ]
    hk, sigs = await _sign_batch(payloads)
    signed = []
    for base, sig in zip(lines, sigs):
        rec = dict(base)
        rec["signature"] = sig
        rec["hotkey"] = hk
        signed.append(rec)

    key = f"{_results_prefix()}{block:09d}-{hk}.json"

    async with get_s3_client() as c:
        try:
            r = await c.get_object(Bucket=settings.R2_BUCKET, Key=key)
            old = loads(await r["Body"].read())
        except c.exceptions.NoSuchKey:
            old = []
        merged = old + signed
        await c.put_object(
            Bucket=settings.R2_BUCKET,
            Key=key,
            Body=dumps(merged),
            ContentType="application/json",
        )
        await _index_add_if_new(key)

    return hk, signed


async def sink_sv_at(key: str, lines: list[dict]) -> tuple[str, list[dict]]:
    """ """
    if not lines:
        return "", []
    payloads = [
        dumps(l.get("payload") or {}, sort_keys=True, separators=(",", ":"))
        for l in lines
    ]
    hk, sigs = await _sign_batch(payloads)
    signed = []
    for base, sig in zip(lines, sigs):
        rec = dict(base)
        rec["signature"] = sig
        rec["hotkey"] = hk
        signed.append(rec)

    s = get_settings()

    async with get_s3_client() as c:
        await c.put_object(
            Bucket=s.R2_BUCKET,
            Key=key,
            Body=dumps(signed, separators=(",", ":")),
            ContentType="application/json",
        )
        await _index_add_if_new(key)
    return hk, signed


async def emit_shard(
    slug: str,
    challenge: SVChallenge,
    miner_run: SVRunOutput,
    evaluation: SVEvaluation,
    miner_hotkey_ss58: str,
) -> None:

    settings = get_settings()
    rid = f"{challenge.challenge_id[:8]}:{miner_hotkey_ss58[-6:]}"
    st = await get_subtensor()
    current_block, st = await _safe_get_current_block(st, rid)
    timeout_s = float(os.getenv("SV_R2_TIMEOUT_S", "60"))

    ns = None
    prefix = _results_prefix(ns)
    eval_key = f"{prefix}{miner_hotkey_ss58}/evaluation/{current_block:09d}-{challenge.challenge_id}.json"
    resp_key = f"{prefix}{miner_hotkey_ss58}/responses/{current_block:09d}-{challenge.challenge_id}.json"

    video_url = None
    try:
        video_url = getattr(getattr(challenge, "payload", None), "url", None)
    except Exception:
        video_url = None

    if getattr(miner_run, "predictions", None) is not None:
        logger.info(f"[emit:{rid}] uploading responses blob to {resp_key}")
        resp_blob = {
            "video_url": video_url,
            "predictions": miner_run.predictions,
        }
        try:
            await asyncio.wait_for(
                _put_json_object(resp_key, resp_blob), timeout=timeout_s
            )
        except asyncio.TimeoutError:
            logger.error(
                f"[emit:{rid}] storing responses blob timed out after {timeout_s}s"
            )
            raise
        except Exception as e:
            logger.error(f"[emit:{rid}] storing responses blob failed: {e}")
            raise
        else:
            logger.info(f"[emit:{rid}] responses stored: {resp_key}")

    meta_out = (challenge.meta or {}).copy()
    meta_out["block"] = current_block

    shard_payload = {
        "env": "SVEnv",
        "task_id": meta_out.get("task_id"),
        "prompt": challenge.prompt,
        "meta": meta_out,
        "miner": {
            "model": getattr(miner_run, "model", None),
            "slug": slug,
            "hotkey": miner_hotkey_ss58,
        },
        "run": {
            "success": getattr(miner_run, "success", None),
            "latency_ms": getattr(miner_run, "latency_ms", None),
            "error": getattr(miner_run, "error", None),
            "responses_key": resp_key,
        },
        "evaluation": {
            "acc_breakdown": getattr(evaluation, "acc_breakdown", None),
            "acc": getattr(evaluation, "acc", None),
            "score": getattr(evaluation, "score", None),
        },
        "ts": time(),
        "block": current_block,
        "source": "api_v2_video",
    }
    shard_line = {"version": settings.SCOREVISION_VERSION, "payload": shard_payload}

    try:
        logger.info(f"[emit:{rid}] writing evaluation shard to {eval_key}")
        hk, signed_lines = await asyncio.wait_for(
            sink_sv_at(eval_key, [shard_line]), timeout=timeout_s
        )
    except asyncio.TimeoutError:
        logger.error(f"[emit:{rid}] sink_sv_at timed out after {timeout_s}s")
        raise
    except Exception as e:
        logger.error(f"[emit:{rid}] sink_sv_at failed: {e}")
        raise
    else:
        logger.info(f"[emit:{rid}] evaluation shard emitted: {eval_key} (1 line)")

    # --- logs run ---
    logger.info("\n=== SV Runner (R2) ===")
    logger.info(f"challenge_id: {challenge.challenge_id}")
    if getattr(miner_run, "latency_ms", None) is not None:
        logger.info(f"latency_ms  : {miner_run.latency_ms:.1f} ms")
    if getattr(evaluation, "acc", None) is not None:
        logger.info(
            f"acc         : {evaluation.acc:.3f}  breakdown={getattr(evaluation, 'acc_breakdown', None)}"
        )
    if getattr(evaluation, "score", None) is not None:
        logger.info(f"score       : {evaluation.score:.3f}\n")


async def dataset_sv(tail: int, *, max_concurrency: int = None):
    """
    - read index
    - filter shards where 'block' >= max_block - tail
    - concurrent prefetch
    - stream local JSONL and yield verified lines
    """
    sem = asyncio.Semaphore(int(os.getenv("SCOREVISION_DATASET_PREFETCH", "8")))
    index = await _index_list()
    # extract bloc from filename
    pairs: list[tuple[int, str]] = []
    for k in index:
        name = Path(k).name
        try:
            b = int(name.split("-", 1)[0])
            pairs.append((b, k))
        except Exception:
            continue
    if not pairs:
        return
    pairs.sort()
    max_block = pairs[-1][0]
    min_keep = max_block - int(tail)

    keys = [k for (b, k) in pairs if b >= min_keep]
    logger.info(
        f"[dataset] max_block={max_block} tail={tail} -> keeping >= {min_keep} | keys_kept={len(keys)}"
    )
    # prefetch
    tasks = [
        asyncio.create_task(_cache_shard(k, sem))
        for k in keys[: (max_concurrency or 8)]
    ]
    next_i = len(tasks)

    for i, key in enumerate(keys):
        if i < len(tasks):
            p = await tasks[i]
        else:
            p = await _cache_shard(key, sem)
        if next_i < len(keys):
            tasks.append(asyncio.create_task(_cache_shard(keys[next_i], sem)))
            next_i += 1

        # stream jsonl
        if not p.exists():
            continue
        with p.open("rb") as f:
            valid_lines = 0
            for raw in f:
                try:
                    line = _loads(raw.rstrip(b"\n"))
                    line["_key"] = key
                    payload_str = dumps(
                        line.get("payload") or {}, sort_keys=True, separators=(",", ":")
                    )
                    sig = line.get("signature", "")
                    hk = line.get("hotkey", "")
                    if hk and sig and _verify_signature(hk, payload_str, sig):
                        valid_lines += 1
                        VALIDATOR_DATASET_LINES_TOTAL.labels(
                            source="local", result="valid"
                        ).inc()
                        yield line
                    else:
                        VALIDATOR_DATASET_LINES_TOTAL.labels(
                            source="local", result="invalid"
                        ).inc()
                except Exception:
                    VALIDATOR_DATASET_LINES_TOTAL.labels(
                        source="local", result="error"
                    ).inc()
                    continue


# --- HTTP public helpers for cross-validator fetch --------------------------


async def _http_get_json(url: str, timeout_s: int = 20) -> any:
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        async with s.get(url) as r:
            if r.status != 200:
                VALIDATOR_DATASET_FETCH_ERRORS_TOTAL.labels(
                    stage="http_get_non200"
                ).inc()
                raise RuntimeError(f"GET {url} -> {r.status}")
            return await r.json()


async def _http_head_meta(
    url: str, timeout_s: int = 10
) -> tuple[str | None, str | None]:
    """ """
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        async with s.head(url) as r:
            if r.status >= 400:
                VALIDATOR_DATASET_FETCH_ERRORS_TOTAL.labels(
                    stage="http_head_non200"
                ).inc()
                return None, None
            return (r.headers.get("ETag"), r.headers.get("Last-Modified"))


def _cache_path_for_url(url: str) -> Path:
    h = hashlib.sha256(url.encode()).hexdigest()[:16]
    name = Path(urlparse(url).path).name
    if not name:
        name = "index.json"
    return CACHE_DIR / f"{name}.{h}.jsonl"


async def _cache_remote_json_array(url: str, sem: asyncio.Semaphore) -> Path:
    """ """
    out = _cache_path_for_url(url)
    mod = out.with_suffix(".modified")
    async with sem:
        etag, lm = await _http_head_meta(url)
        tag = (etag or lm or "").strip()
        if out.exists() and mod.exists() and mod.read_text().strip() == tag:
            return out
        arr = await _http_get_json(url)
        tmp = out.with_suffix(".tmp")
        with tmp.open("wb") as f:
            for line in arr if isinstance(arr, list) else []:
                f.write(_dumps(line))
                f.write(b"\n")
        os.replace(tmp, out)
        if tag:
            mod.write_text(tag)
    return out


def _bucket_base(index_url: str) -> str:
    u = urlparse(index_url)
    return f"{u.scheme}://{u.netloc}/"


def _join_key_to_base(index_url: str, key_or_url: str) -> str:
    if key_or_url.startswith("http://") or key_or_url.startswith("https://"):
        return key_or_url

    if key_or_url.startswith("scorevision/"):
        return _bucket_base(index_url) + key_or_url

    if key_or_url.startswith("/"):
        return _bucket_base(index_url) + key_or_url.lstrip("/")

    base = index_url.rsplit("/", 1)[0] + "/"
    return urljoin(base, key_or_url)


async def _list_keys_from_remote_index(index_url: str) -> list[str]:
    """ """
    idx = await _http_get_json(index_url)
    keys: list[str] = []
    if isinstance(idx, list):
        keys = [_join_key_to_base(index_url, k) for k in idx if isinstance(k, str)]
    elif isinstance(idx, dict) and isinstance(idx.get("entries"), list):
        for e in idx["entries"]:
            p = e.get("path")
            if isinstance(p, str):
                keys.append(_join_key_to_base(index_url, p))
    return keys


async def dataset_sv_multi(
    tail: int, validator_indexes: dict[str, str], *, prefetch: int = 2
):
    """ """
    if not validator_indexes:
        return
    validator_indexes = {
        hk: normalize_index_url(iurl) for hk, iurl in validator_indexes.items() if iurl
    }
    all_pairs: list[tuple[int, str, str]] = []
    for idx_hk, idx_url in validator_indexes.items():
        try:
            keys = await _list_keys_from_remote_index(idx_url)
        except Exception as e:
            logger.warning(f"[dataset-multi] index fetch failed {idx_url}: {e}")
            VALIDATOR_DATASET_FETCH_ERRORS_TOTAL.labels(stage="index_fetch").inc()
            continue
        for u in keys:
            name = Path(u).name
            b = None
            try:
                b = int(name.split("-", 1)[0])
            except Exception:
                pass
            if b is not None:
                all_pairs.append((b, u, idx_url))

    if not all_pairs:
        return

    all_pairs.sort()
    max_block = all_pairs[-1][0]
    min_keep = max_block - int(tail)
    kept = [(b, u, iurl) for (b, u, iurl) in all_pairs if b >= min_keep]
    logger.info(
        f"[dataset-multi] max_block={max_block} tail={tail} -> kept={len(kept)} shards"
    )

    prefetch = max(1, int(os.getenv("SCOREVISION_DATASET_PREFETCH", str(prefetch))))
    sem = asyncio.Semaphore(prefetch)
    tasks = []
    for i, (_b, url, _iurl) in enumerate(kept[:prefetch]):
        tasks.append(asyncio.create_task(_cache_remote_json_array(url, sem)))
    next_i = len(tasks)

    for i, (b, url, iurl) in enumerate(kept):
        try:
            if i < len(tasks):
                p = await tasks[i]
            else:
                p = await _cache_remote_json_array(url, sem)
            if next_i < len(kept):
                tasks.append(
                    asyncio.create_task(_cache_remote_json_array(kept[next_i][1], sem))
                )
                next_i += 1
        except Exception as e:
            logger.warning(f"[dataset-multi] cache failed {url}: {e}")
            VALIDATOR_DATASET_FETCH_ERRORS_TOTAL.labels(stage="cache_fetch").inc()
            continue

        if not p.exists():
            continue
        valid_lines = 0
        with p.open("rb") as f:
            for raw in f:
                try:
                    line = _loads(raw.rstrip(b"\n"))
                    line["_src_index"] = iurl
                    payload_str = dumps(
                        line.get("payload") or {}, sort_keys=True, separators=(",", ":")
                    )
                    sig = line.get("signature", "")
                    hk = line.get("hotkey", "")
                    if hk and sig and _verify_signature(hk, payload_str, sig):
                        valid_lines += 1
                        VALIDATOR_DATASET_LINES_TOTAL.labels(
                            source="cross", result="valid"
                        ).inc()
                        yield line
                    else:
                        VALIDATOR_DATASET_LINES_TOTAL.labels(
                            source="cross", result="invalid"
                        ).inc()
                except Exception:
                    VALIDATOR_DATASET_LINES_TOTAL.labels(
                        source="cross", result="error"
                    ).inc()
                    continue


def prune_sv(tail: int):
    blocks = []
    for f in CACHE_DIR.glob("*.jsonl"):
        name = f.name.split("-", 1)[0]
        if name.isdigit():
            blocks.append(int(name))
    if not blocks:
        return
    maxb = max(blocks)
    min_keep = maxb - int(tail)
    for f in CACHE_DIR.glob("*.jsonl"):
        name = f.name.split("-", 1)[0]
        if name.isdigit() and int(name) < min_keep:
            try:
                f.unlink()
            except:
                pass
        m = f.with_suffix(".modified")
        if m.exists() and (not f.exists() or int(name) < min_keep):
            try:
                m.unlink()
            except:
                pass


def build_public_index_url() -> str | None:
    """ """
    s = get_settings()
    if not (s.R2_ACCOUNT_ID.get_secret_value() and s.R2_BUCKET):
        return None
    base = f"https://{s.R2_ACCOUNT_ID.get_secret_value()}.r2.cloudflarestorage.com"
    return f"{base}/{s.R2_BUCKET}/scorevision/index.json"


async def ensure_index_exists() -> None:
    """ """
    s = get_settings()
    index_key = "scorevision/index.json"

    async with get_s3_client() as c:
        try:
            await c.head_object(Bucket=s.R2_BUCKET, Key=index_key)
            return
        except c.exceptions.NoSuchKey:
            pass
        await c.put_object(
            Bucket=s.R2_BUCKET,
            Key=index_key,
            Body="[]",
            ContentType="application/json",
        )
