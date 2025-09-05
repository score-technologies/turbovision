from time import time
from logging import getLogger
from json import dumps, loads
import asyncio
import os
from pathlib import Path

from aiobotocore.session import get_session
from botocore.config import Config as BotoConfig

from scorevision.utils.data_models import SVChallenge, SVRunOutput, SVEvaluation
from scorevision.utils.settings import get_settings
from scorevision.utils.signing import _sign_batch

logger = getLogger(__name__)
import os
import asyncio
from pathlib import Path
from json import dumps, loads


def _loads(b):
    return loads(b.decode() if isinstance(b, (bytes, bytearray)) else b)


def _dumps(o) -> bytes:
    return dumps(o, separators=(",", ":")).encode()


settings = get_settings()

CACHE_DIR = settings.SCOREVISION_CACHE_DIR
CACHE_DIR.mkdir(parents=True, exist_ok=True)

import nacl.signing, nacl.encoding


from substrateinterface import Keypair


def _verify_signature(hk_ss58: str, payload: str, sig_hex: str) -> bool:
    try:
        if not hk_ss58 or not sig_hex:
            return False
        sig_hex = sig_hex[2:] if sig_hex.startswith("0x") else sig_hex
        kp = Keypair(ss58_address=hk_ss58)
        return kp.verify(payload.encode("utf-8"), bytes.fromhex(sig_hex))
    except Exception:
        return False


async def _index_list() -> list[str]:
    """ """
    settings = get_settings()
    index_key = "scorevision/index.json"

    if not _r2_enabled():
        local_index = settings.SCOREVISION_LOCAL_ROOT / "index.json"
        if local_index.exists():
            try:
                return loads(local_index.read_text())
            except Exception:
                return []
        return []

    async with get_s3_client() as c:
        try:
            r = await c.get_object(Bucket=settings.R2_BUCKET, Key=index_key)
            body = await r["Body"].read()
            return loads(body)
        except c.exceptions.NoSuchKey:
            return []


async def _cache_shard(key: str, sem: asyncio.Semaphore) -> Path:
    """ """
    settings = get_settings()
    out = CACHE_DIR / (Path(key).name + ".jsonl")
    mod = out.with_suffix(".modified")

    # ---- chemin offline local ----
    if not _r2_enabled():
        src = settings.SCOREVISION_LOCAL_ROOT / key
        if not src.exists():
            return out
        lm = str(int(src.stat().st_mtime))
        if out.exists() and mod.exists() and mod.read_text().strip() == lm:
            return out
        arr = loads(src.read_text())
        tmp = out.with_suffix(".tmp")
        with tmp.open("wb") as f:
            for line in arr:
                f.write(_dumps(line))
                f.write(b"\n")
        os.replace(tmp, out)
        mod.write_text(lm)
        return out

    # ---- chemin R2 ----
    async with sem, get_s3_client() as c:
        try:
            head = await c.head_object(Bucket=settings.R2_BUCKET, Key=key)
            lm = head["LastModified"].isoformat()
        except c.exceptions.NoSuchKey:
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

    if not _r2_enabled():
        items = set()
        if local_index.exists():
            try:
                items = set(loads(local_index.read_text()))
            except Exception:
                items = set()
        if key not in items:
            items.add(key)
            local_index.write_text(dumps(sorted(items)))
        return
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

    key = f"{settings.SCOREVISION_RESULTS_PREFIX}{block:09d}-{hk}.json"

    if not _r2_enabled():
        dst = settings.SCOREVISION_LOCAL_ROOT / key
        dst.parent.mkdir(parents=True, exist_ok=True)
        old = []
        if dst.exists():
            try:
                old = loads(dst.read_text())
            except Exception:
                old = []
        merged = old + signed
        dst.write_text(dumps(merged, separators=(",", ":")))
        await _index_add_if_new(key)
        return hk, signed

    # --- R2 enabled path ---
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


async def emit_shard(
    slug: str,
    challenge: SVChallenge,
    miner_run: SVRunOutput,
    evaluation: SVEvaluation,
    miner_hotkey_ss58: str,
) -> None:
    settings = get_settings()
    meta_out = challenge.meta or {}
    shard_payload = {
        "env": "SVEnv",
        "task_id": meta_out.get("task_id"),
        "prompt": challenge.prompt,
        "meta": meta_out,
        "miner": {
            "model": miner_run.model,
            "slug": slug,
            "hotkey": miner_hotkey_ss58,
        },
        "run": {
            "success": miner_run.success,
            "latency_ms": miner_run.latency_ms,
            "error": miner_run.error,
        },
        "evaluation": {
            "acc_breakdown": evaluation.acc_breakdown,
            "acc": evaluation.acc,
            "score": evaluation.score,
        },
        "ts": time(),
        "source": "api_v2_video",
    }
    shard_line = {"version": settings.SCOREVISION_VERSION, "payload": shard_payload}

    block = int(time())
    try:
        hk, signed_lines = await sink_sv(block, [shard_line])
        logger.info(
            f"Shard emitted: {settings.SCOREVISION_RESULTS_PREFIX}{block:09d}-{hk}.json (1 line)"
        )
    except Exception as e:
        logger.error(f"sink_sv failed: {e}")

    logger.info("\n=== SV Runner (R2) ===")
    logger.info(f"challenge_id: {challenge.challenge_id}")
    logger.info(f"latency_ms  : {miner_run.latency_ms:.1f} ms")
    logger.info(
        f"acc         : {evaluation.acc:.3f}  breakdown={evaluation.acc_breakdown}"
    )
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
            for raw in f:
                try:
                    valid_lines = 0
                    line = _loads(raw.rstrip(b"\n"))
                    line["_key"] = key
                    payload_str = dumps(
                        line.get("payload") or {}, sort_keys=True, separators=(",", ":")
                    )
                    sig = line.get("signature", "")
                    hk = line.get("hotkey", "")
                    if hk and sig and _verify_signature(hk, payload_str, sig):
                        valid_lines += 1
                        yield line
                    logger.info(f"[dataset] {key} -> valid_lines={valid_lines}")
                except Exception:
                    continue


def prune_sv(tail: int):
    # delete files where block < (max_block - tail)
    # calculate max_block from precedent files
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
