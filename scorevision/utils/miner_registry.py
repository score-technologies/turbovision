from __future__ import annotations
import asyncio
import os, json, time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from logging import getLogger
from urllib.parse import urljoin, urlparse

import aiohttp
from huggingface_hub import HfApi
from bittensor import AsyncSubtensor

from scorevision.utils.bittensor_helpers import (
    get_subtensor,
    reset_subtensor,
    get_validator_indexes_from_chain,
)
from scorevision.utils.bittensor_commitments import get_all_revealed_commitments
from scorevision.utils.commit_recovery import (
    is_recovery_commit,
    recover_commitments_from_shards,
)
from scorevision.utils.compliance_failures import (
    ComplianceFailureTuple,
    fetch_compliance_failure_tuples,
    is_compliance_tuple_failed,
)
from scorevision.utils.inactive_miners import (
    InactiveMinerTuple,
    fetch_inactive_miner_tuples,
    is_inactive_miner_tuple,
)
from scorevision.utils.settings import get_settings

logger = getLogger(__name__)

HARDCODED_BLACKLIST_HOTKEYS: set[str] = {
    "5DvY7cxtAvUeA2Goq26LNyzqSfPyjfY9SUsD4bgJa5PMnVNa",
    "5CMaFwgm2rPka66iUcgAa2SpBPskk6KqAGWZeKVx8APLnqTZ",
    "5CfGbGvZz6YUUPT84ntoGHANy1ddk9xGJiaQZVEb9Qi57Foc",
    "5CG3NLAVRzs1uxqFHTjUaSESW4N7PLFYrRosjuh7JRKFoYmi",
    "5CqKodaU2F5atWnLBivA2TeQWoJ9jxfPwADQ2HAWfAQHNsZV",
    "5CmCgAKAW1B3YmBDeVJxxd2MbhMeC6n1esJsVFdb2qUvD55r",
    "5Fnx48i6A6KtN9a8pAxQDZDQo3Z49JeviaLkcJUbLB9yCM4Q",
    "5HBK3g2PURYZ99ufLRCfoqEga6ScT8KzVerxk54BELF7hZvu",
    "5FZiTikQum61QyzWaaGmsiKmH1eX2jLcZ61V2E2WBg1QAAYV",
    "5C81gi3bXLbAWccH6VFY5T7BNhv9usEMdtwxHUVqeFPJ3zy9",
    "5Gj9pWjksQXkuaoVxHRaKN1pmgQYiUddrNweY3SFBWMGo2QD",
    "5FkPaJTKgBr3rXX1YUt952ejVnwui4RZHoafBoR6txGkrSc3",
    "5GYVPC5tTnHEtqZ4qVCHfyRFmQ1SDHHo9nwECphftrN8wbRG",
    "5FkF684znSsYy1bjsaTDYpZDYN983Z5kavsvv6bNYyuNZQHy",
    "5GNvBDx7qBcUUM3EwLTD2a5qVyFcjKs7Nf2zk2qxL1xPsXCw",
    "5DvuNyrQQuDmzBMUDVNsRJSXBMxuBKAuKiJQUiTMAgTJoDoF",
    "5HeKKwL1jyZnKUxSxLjVGxkioHCV2bCci6eEN1Qdd81APg4a",
    "5GE4vXbuK7Q53AgSmwdPGrRrzUmjM4415K9JsbugS2JmcMgG",
    "5F1mgNNNzFBjmGnq1i5SP6svoSxq2n9wp4WhA2DG9ttUKwD8",
    "5DjwjwAhLGnuLYXFMtP5VuCeLiczEQmctiW1FMgBDkHKzhCf",
    "5HddqPvyeF1E5FYZTiyaKQA85mprWfoP6d3GriXvks8vJXUG",
    "5CVBjijH1eey2KobTcQJNBSBAFX5PtvdnqXKLZWswddvphMs",
    "5CcLYznNrKLATjs3aV8USd58wCtD6keQYqmRTrtChBGU8T8C",
    "5DUY4UEpn8EGewfezEKwFo4SfRn11mv6f1N2U8nw1LuMkA9q",
    "5GiwVmxPXh35sWtGJTjE8YKpRy1dc9MvM1VPrgtPMpmmEoxz",
    "5EjVevWZ8RD7ks8BX4YXXpJxSCasbSYitKHrQ1CqhQGp6JZu",
    "5Ec6P4FZZqDHGfrD2iqa3kFAR83rqsPJRHSJtkmrcdwjBYkZ",
    "5DhGGmDn7RrEjexDjSsRT5qWUMurt74esHqWKNyk8sYsaHQb",
    "5CPKYUx11qM4CLJPu6UDMutEvB7VT4QA1BxhoEh5t27DvEBQ",
    "5C4w41JTTSC3W7UPWmeppKDGTJNqAu2k7Nx7z12AZEGBXBrq",
    "5CArLgrxQ4GLPuMzb9Y7gYCfvuGotL69b7iTvXVKdv649kM9",
    "5CG2b2zJdd2sivzb3AEfDyVoEK4aJBtSbuw94ZaLRkMaoY9b",
    "5CJQPab55wqeCttjiDkwg16b5GR7voxADPu3VHPhVNWV8xEe",
    "5CJUrDvci5uSJTrN18T7r2YTavkV3FrzmsGsvnaUsrWRJcsT",
    "5CJs8X2cHSRXSdUKYrCrCcevK5LkmNfiEb55XEgkZNcnFtAD",
    "5CSZqok1zmt2VBpAqgsFrByLoysbzViF4JJir4jSiHaB3qbz",
    "5CUyvfyBFJoKJwvAMhnYWet8LvNdcG6K1VsgVrMQevRKfHUw",
    "5CZhET3dLGyLohLTQsjpMNSVssQ1g88xBoDAEMc9LMukpRDH",
    "5CiVzdvZCyj2PEit6e7RSnhMVXY5xu3tWX5hNQhgRUbQ1A3L",
    "5CkzEqsuUVVNW9KaruNgDSwohu2Z3j1HGmoDARGqCJCN5bDS",
    "5CoXyequMzZrEkkA6MwfMTuxVNXWUr14NGujoa3cdg3EdRub",
    "5CqkPf821oWrUNTiEjhhWLxnUBXiRf9Jr35bmrWDfB6nqSo3",
    "5D7nuaXR8Lrctp11GYYMEqckXEKLXLhqxuaEh7vzi6XJGn4D",
    "5DcbFpyCwLyXm3Xfu5izpp96DxtJefMjQRV5AyRYtDeLffv6",
    "5DeuvQnD3fFiFpXqEYLc3PgbvLBQKtnqRURpvik8usVS6yup",
    "5Dh9t36JU9oHSDyJ44XQP7xsDXGeChVZwLHqXUZdtTNagy8B",
    "5Do6BUVP3WjXe1PKESKXaCQa6afbt1At8gpgJ2THZVZeYxCX",
    "5DoHgTu9SzA6UzXMHnikoiYfXqGBBzTZofwvKgs34GPW5Q5q",
    "5E1jXkTmZw3nXfPEJYFuR9yLfkAeB6iLN7Jpy7pTpoWqkeVG",
    "5E1mJeKatTvGwACF8fy4VaxcjzCSFw3HxSyXYCYaWE7cSDFs",
    "5E51HuZ8a4eU2VYQfZdeMJWDULo55uxfm9UQ87JxTMsUXk1b",
    "5E7ND4HnnMYcYHJCfBgrvtQ1P9HaKMGekAWpXvozgK9o4vQX",
    "5E7vhYAhLgK1HiEJJF161eQHZDxUyu41Rs4uV2vXD678Xj9A",
    "5EADUbTATkEAR5fUcgjXadU9ogmfMMgneTbMNtqgde6i8qDB",
    "5EFAp9FRCYf3yweHT2XURVxP4A12nkhmoQpAXmSYz7dw5XhH",
    "5EKumErdDteNgd8d1tmTYLvAFD7thmXSzLvZTcnR58L1JDqE",
    "5EM1F7mEnqXNpd29NWaGKjcJcBNXrw2V21zW5wHNiZTUGkUJ",
    "5ENxR9bSU6SUtrzoaYJTw2TuBYuae4J2g4FLGE5mrn3SHbti",
    "5EPidbuDHyHZ8VYp78toNrsfUDjnz6W5s6XPgNK42Uh9Y9ta",
    "5EgrCbU1QDDsJndnXW6h3AdxfQX1TZ32j5ZruA5UN2vr4FbR",
    "5EpucAdQFp69BsUYfBfVtEuwiqTU7XawirpqUDZkKjMYAdUU",
    "5Eq9YMcRPjkM3WXXY5ndGKcsfsFfhdgN1cmKDRtRiw2EU9XN",
    "5EyABxgVHqJfuHPJgtdVCfR59ar9VBcmkKihw9aCKkim84w5",
    "5F445iMiCMXUjdKRP1VtEPN59n1fAScYmFgULLquHYXq2z3Q",
    "5F6rXye5FgReC1HpLGmEVcrDHxVLnKPahce93884zday93nw",
    "5FEeLFMj2J6NpPxNoTC7JauPgF62WwM1ZKGxKk3Ku5NWw1oE",
    "5FLoBSkmzy7Ab8m1RXUk3XnZBVGNm513XaX89TthD9yMHKmg",
    "5FWASJ39z6x77RAHeDNWBd8cdY9zBmAVeXrkVDUxadhkb3Hy",
    "5FWXv1FruZZMyvBNN5QS76aLeJyHbuVaAYGhf8QCZbiotSo4",
    "5Fe2qtT9rfXobCTMQdQHQRRtXiHf3eurDXcHgzW1SgZNKCGe",
    "5FeNceWkA6i3NZcPMiosoMm8Xg8JQV2kd8C6qqWsTo4EVX4x",
    "5FxgJ7PgLKpSm5sKDZv9GCw98txhcJF6FXoBuz83KP62xQVZ",
    "5G1e9VQ6LejttBRQrZbdiwMw7WamXoSLAMLVnYmS3KvGPrS9",
    "5G4GaDSzeUTjNmWVgUozJZemz6LwhJZgMro857ZZbG3H1Kc8",
    "5GHaqTTSkxswM2qierqWktdsQWU6neYVn1es2ikPSXLtCaqt",
    "5GNxJ6m12jE6B5EiLYn69Gm9bSniKSJcDHasFMccjpBPPY93",
    "5GTf2GH5fBUacCbhMDXEYcqcYhcKcpxwTY71zVAXVFoty9ta",
    "5GUFy8ExgLdxc19iujsZ77AMD4EnBc6SKEQxqdpHgG1Vyp9h",
    "5GW4y7frkK1xWtEVVHbfSVqX7Kn7BFLkAyE8v3Yrx4t9gWyP",
    "5GbZHx8x1MXfUfYkqFpHESxWvA7ntER8XzxprDW7ZU2Nfmse",
    "5GcE2NMgHc48gGpR4UQiqDyXuNKLaVeMBrVrfuZovwu8JbLQ",
    "5GgyWAHE75Jihopn5CoVJN2Atx8ziE4azHjNg7Pg2vukZPXj",
    "5GhKqFVfu5WtSWGNQM9vShBHQaGGVLsMSo1FUhJgVDCwJNWJ",
    "5GhUm2HABYgSC3PUyBuKexQ47cT3RBnSsgq7ZVPfRV5H6qNy",
    "5GjbnhyzsVqqEoTyrGDeeFMDYMQKmEFi4SiMx4BjpBvUdiJp",
    "5GnMCUJEgu7gjkSGeXh1SjAnfT3hQ8TyC41jDa4KXAij9L7v",
    "5GvTPBc4STfxQXKEfzHZ9Red5rQeGHPe4uRK7HwsYeoseTRJ",
    "5Gxhdw5P7pLsBh3ENS4Dhbq9GT7PjxCDBUfZoqgrf7wPxRfH",
    "5H1m3gqCizC9CQs9YJm7SzQn6annTPFpcVt2BHHUohS7muJe",
    "5HEq9QZi76QLBWEsoeLaWcR68SBHqRenDMX7drhxCMGqmkqd",
    "5HGkB6iXTzBRXhmu26dTbjbTvuw2yHQoKSn15xk5PLMYj2kg",
    "5HGnVLcmPKxAgVq86GRDyKXV4GJnENdiS1auELW8XnP54FcG",
    "5HNLwiZCrqMBT5pTfUz1DtSzVYWeKkQgGuuQzBVLtf27g2wW",
    "5HR5Rc32M7aN3bLRbwetwYkL1R8CXfS7vneHxAbBtC98cEwF",
    "5Hb3gdQBngfuyXj628VECqri2X8XZSdErSFae2bjUB3JYm14",
    "5Hm9t1qUqQY968Avw4K8W9KgZbrJptBNpybEdhvVw9pKE3Bz",
    "5C5GPfX1KjgR1YDPBCADcnR28UQy25CJw1tkcQVVvJ7ZPSzr",
    "5CSkm9JvkqQnTCzUNsEmqWJUiikG5eb9WkztUjcv7LG2rwoh",
    "5CY5auTcUPvymijZxHReVqNKmHs5qFt3X5MwVy1fweywV4Ay",
    "5CfFWzKM1AT47SVjRxLa7ksqdjkdCg3bsoBo221zk2WFmBiT",
    "5CqhdKhL1gDdxYBt3VUCDRJydNKucuuxZ1EHhEBfHo7GQKYh",
    "5Ct54MJJegsTgK1HbqUpZoAjvpDcoguofvtWUVAkKQeZCdL1",
    "5CtqLnDkVwVXYKh57Mmak9JqYcdnFaD23FkkFLnAT7hv6Uzb",
    "5D78MNmvXAgangdayZu3DJtRgJgdN7waddsqWZYf1byzab8R",
    "5DEykXVkAj8s5ZuX5iiBJa6X1ouQ87dAJRSd6zxcsEDz4BrA",
    "5DkJ8h2ns7EuU9KWMy93VT4JvpraeChQ1GPHWrEvYVfVFoZH",
    "5DyGLMtKtroriNA1a6WYKxrhrvKUMY3SbiGBNUtpFQ6U5RER",
    "5E1Waen8TXTK9a57xmijZD1utoTFMLfPNnbfFkvhEYX29QoX",
    "5Ec6UgVEBeMiZikciL7VogZXKBY2NJCMNZ1qY2wrLBnvvwFh",
    "5Ecqhez1dVWUW2i17Wtjdf4X3Bp71etwFXQHPrekCb7D5kpq",
    "5EpvXL4pMGBtQvZYhiYqqDzJzpQDoQazaQ85vEQFLaYL33FQ",
    "5F4kovuFSj68khx3UvGqHBu4wdH67Ua7MeeCasTeMcrNd55K",
    "5FCxLn1oe8rpyovaDMJaDAFnegZ9xf7GHATwfBZiuSZGm8RW",
    "5G1X1vih425XthcEPWY4HivRrWjvfm3nV4RByL1NGwCeuNxD",
    "5GZNy3CmgA5aCuZi361vvJBo2UskDtC8KxWr4WvgReq1DL47",
    "5Gbmes6C8SzSEnN1wuWkwdc7FVFcuaYsyhmH9F1KTXsqnzmr",
    "5GjUo4VKrjoB8AniH6Yb8gTYrLEiVCK4AfRaKVm8MLL5bdtC",
    "5Gmrj9E1xpCRr4LyKBEfRu8Av4z6tYoVnrSMUYrtfD6ZZRgv",
    "5HSsgBxtCZnXxh4gAbbW1dnZHppaJQz5aLMMnhjKardYfEKC",
    "5HmMQq7C1mbF4RBZnyR8ApkYxPzv9N3DmCMMnLhWurnrgfg8",
}

REGISTRY_BYPASS_UIDS = {6}
REGISTRY_BYPASS_HOTKEYS = {"5FsREvyUXSZWYRqVyQLDdpYmZZPnkhZyW6HjooozKP1nQkwu"}


def is_registry_bypass(uid: int | None, hotkey: str | None) -> bool:
    if uid is None or not hotkey:
        return False
    hk = str(hotkey).strip()
    return uid in REGISTRY_BYPASS_UIDS and hk in REGISTRY_BYPASS_HOTKEYS


@dataclass
class Miner:
    uid: int
    hotkey: str
    model: Optional[str]
    revision: Optional[str]
    slug: Optional[str]
    chute_id: Optional[str]
    block: int
    element_id: Optional[str] = None
    registry_skip_reason: Optional[str] = None
    registry_skip_details: Optional[dict] = None


# ------------------------- HF gating & revision checks ------------------------- #
_HF_MODEL_GATING_CACHE: Dict[str, Tuple[bool, float]] = {}
_HF_GATING_TTL = 300  # seconds
_HF_MODEL_SIZE_CACHE: Dict[Tuple[str, str], Tuple[Optional[float], float]] = {}
_HF_MODEL_SIZE_TTL = 300  # seconds
_HF_ONNX_ONLY_CACHE: Dict[Tuple[str, str], Tuple[Optional[bool], float]] = {}
_HF_ONNX_ONLY_TTL = 300  # seconds
_CHUTES_FETCH_RETRIES = max(1, int(os.getenv("SV_REGISTRY_CHUTES_RETRIES", "2")))
_CHUTES_FETCH_BACKOFF_S = float(os.getenv("SV_REGISTRY_CHUTES_RETRY_BACKOFF_S", "0.5"))
_REGISTRY_COMMIT_BACKFILL_ENABLE = str(
    os.getenv("SV_REGISTRY_COMMIT_BACKFILL_ENABLE", "true")
).strip().lower() in ("1", "true", "yes", "on")
_REGISTRY_COMMIT_BACKFILL_ARCHIVE_ENDPOINT = os.getenv(
    "SV_REGISTRY_COMMIT_BACKFILL_ARCHIVE_ENDPOINT",
    "wss://archive.chain.opentensor.ai:443",
).strip()
_REGISTRY_COMMIT_BACKFILL_MAX_HOPS = max(
    1, int(os.getenv("SV_REGISTRY_COMMIT_BACKFILL_MAX_HOPS", "20"))
)
_REGISTRY_COMMIT_BACKFILL_CONCURRENCY = max(
    1, int(os.getenv("SV_REGISTRY_COMMIT_BACKFILL_CONCURRENCY", "1"))
)
_REGISTRY_COMMIT_BACKFILL_FIRST_BLOCK = max(
    0, int(os.getenv("SV_REGISTRY_COMMIT_BACKFILL_FIRST_BLOCK", "0"))
)
_REGISTRY_BACKFILL_INDEX_TIMEOUT_S = float(
    os.getenv("SV_REGISTRY_BACKFILL_INDEX_TIMEOUT_S", "12")
)


async def _hf_is_gated(model_id: str) -> Optional[bool]:
    url = f"https://huggingface.co/api/models/{model_id}"
    try:
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.get(url) as r:
                if r.status == 200:
                    data = await r.json()
                    gated = bool(data.get("gated", False))
                    logger.debug("[HF] model=%s gated=%s", model_id, gated)
                    return gated
                logger.debug("[HF] model=%s status=%s", model_id, r.status)
    except Exception as e:
        logger.debug("[HF] is_gated error for %s: %s", model_id, e)
    return None


def _hf_revision_accessible(model_id: str, revision: Optional[str]) -> bool:
    if not revision:
        return True
    try:
        tok = os.getenv("HF_TOKEN")
        api = HfApi(token=tok) if tok else HfApi()
        api.repo_info(repo_id=model_id, repo_type="model", revision=revision)
        logger.debug("[HF] model=%s revision=%s accessible", model_id, revision)
        return True
    except Exception as e:
        logger.debug(
            "[HF] model=%s revision=%s NOT accessible: %s", model_id, revision, e
        )
        return False


async def _hf_gated_or_inaccessible(
    model_id: Optional[str], revision: Optional[str]
) -> Optional[bool]:
    if not model_id:
        logger.debug("[HF] no model id → treat as not eligible")
        return True
    now = time.time()
    cached = _HF_MODEL_GATING_CACHE.get(model_id)
    if cached and (now - cached[1]) < _HF_GATING_TTL:
        gated = cached[0]
        logger.debug("[HF] cache hit model=%s gated=%s", model_id, gated)
    else:
        gated = await _hf_is_gated(model_id)
        _HF_MODEL_GATING_CACHE[model_id] = (bool(gated) if gated is not None else False, now)
        logger.debug("[HF] cache set model=%s gated=%s", model_id, gated)

    if gated is True:
        logger.info("[HF] model=%s is gated", model_id)
        return True
    if not _hf_revision_accessible(model_id, revision):
        logger.info("[HF] model=%s revision inaccessible", model_id)
        return True
    return False


def _hf_repo_total_size_mb(model_id: str, revision: Optional[str]) -> Optional[float]:
    revision_key = str(revision or "")
    cache_key = (model_id, revision_key)
    now = time.time()
    cached = _HF_MODEL_SIZE_CACHE.get(cache_key)
    if cached and (now - cached[1]) < _HF_MODEL_SIZE_TTL:
        logger.debug(
            "[HF] size cache hit model=%s revision=%s size_mb=%s",
            model_id,
            revision_key,
            cached[0],
        )
        return cached[0]

    try:
        tok = os.getenv("HF_TOKEN")
        api = HfApi(token=tok) if tok else HfApi()
        total_bytes = 0
        for node in api.list_repo_tree(
            repo_id=model_id,
            repo_type="model",
            revision=revision,
            recursive=True,
            expand=True,
        ):
            size = getattr(node, "size", None)
            if isinstance(size, int) and size >= 0:
                total_bytes += size

        size_mb = total_bytes / (1024 * 1024)
        _HF_MODEL_SIZE_CACHE[cache_key] = (size_mb, now)
        logger.debug(
            "[HF] computed size model=%s revision=%s size_mb=%.2f",
            model_id,
            revision_key,
            size_mb,
        )
        return size_mb
    except Exception as e:
        logger.info(
            "[HF] failed to compute repo size model=%s revision=%s: %s",
            model_id,
            revision_key,
            e,
        )
        _HF_MODEL_SIZE_CACHE[cache_key] = (None, now)
        return None


def _hf_repo_has_only_onnx_models(
    model_id: str, revision: Optional[str]
) -> Optional[bool]:
    revision_key = str(revision or "")
    cache_key = (model_id, revision_key)
    now = time.time()
    cached = _HF_ONNX_ONLY_CACHE.get(cache_key)
    if cached and (now - cached[1]) < _HF_ONNX_ONLY_TTL:
        logger.debug(
            "[HF] onnx cache hit model=%s revision=%s onnx_only=%s",
            model_id,
            revision_key,
            cached[0],
        )
        return cached[0]

    try:
        tok = os.getenv("HF_TOKEN")
        api = HfApi(token=tok) if tok else HfApi()
        model_exts = (
            ".onnx",
            ".safetensors",
            ".bin",
            ".pt",
            ".pth",
            ".ckpt",
            ".h5",
            ".keras",
            ".pb",
            ".tflite",
            ".msgpack",
            ".gguf",
        )

        has_onnx = False
        found_non_onnx_model = False

        for node in api.list_repo_tree(
            repo_id=model_id,
            repo_type="model",
            revision=revision,
            recursive=True,
            expand=True,
        ):
            path = str(getattr(node, "path", "") or "").lower()
            if not path:
                continue
            if not path.endswith(model_exts):
                continue
            if path.endswith(".onnx"):
                has_onnx = True
                continue
            found_non_onnx_model = True
            break

        onnx_only = has_onnx and not found_non_onnx_model
        _HF_ONNX_ONLY_CACHE[cache_key] = (onnx_only, now)
        logger.debug(
            "[HF] onnx scan model=%s revision=%s has_onnx=%s non_onnx_model=%s",
            model_id,
            revision_key,
            has_onnx,
            found_non_onnx_model,
        )
        return onnx_only
    except Exception as e:
        logger.info(
            "[HF] failed to verify onnx-only model repo=%s revision=%s: %s",
            model_id,
            revision_key,
            e,
        )
        _HF_ONNX_ONLY_CACHE[cache_key] = (None, now)
        return None


# ------------------------------ Chutes helpers -------------------------------- #
class _ChutesInfo(dict):
    """Chutes metadata with non-sensitive diagnostics for the registry shard."""

    def __init__(self, data: Optional[dict], *, lookup_details: dict):
        super().__init__(data or {})
        self.lookup_details = lookup_details


async def _chutes_get_json(
    url: str, headers: Dict[str, str]
) -> tuple[Optional[dict], dict]:
    timeout = aiohttp.ClientTimeout(total=15)
    started_at = time.monotonic()
    try:
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.get(url, headers=headers) as r:
                if r.status != 200:
                    logger.debug("[Chutes] GET %s -> %s", url, r.status)
                    return None, {
                        "category": f"http_{r.status}",
                        "http_status": r.status,
                        "latency_ms": round((time.monotonic() - started_at) * 1000, 1),
                    }
                try:
                    data = await r.json()
                    if not isinstance(data, dict):
                        logger.debug(
                            "[Chutes] GET %s returned unexpected JSON type %s",
                            url,
                            type(data).__name__,
                        )
                        return None, {
                            "category": "invalid_payload_type",
                            "payload_type": type(data).__name__,
                            "latency_ms": round(
                                (time.monotonic() - started_at) * 1000, 1
                            ),
                        }
                    if not data:
                        logger.debug(
                            "[Chutes] GET %s returned an empty JSON object", url
                        )
                        return None, {
                            "category": "empty_response",
                            "latency_ms": round(
                                (time.monotonic() - started_at) * 1000, 1
                            ),
                        }
                    logger.debug("[Chutes] GET %s -> ok", url)
                    return data, {
                        "category": "success",
                        "http_status": r.status,
                        "latency_ms": round((time.monotonic() - started_at) * 1000, 1),
                    }
                except Exception as e:
                    logger.debug(
                        "[Chutes] JSON decode error for %s: %s: %r",
                        url,
                        type(e).__name__,
                        e,
                    )
                    return None, {
                        "category": "invalid_json",
                        "error_type": type(e).__name__,
                        "latency_ms": round((time.monotonic() - started_at) * 1000, 1),
                    }
    except Exception as e:
        logger.info("[Chutes] GET %s failed: %s: %r", url, type(e).__name__, e)
        if isinstance(e, (asyncio.TimeoutError, aiohttp.ServerTimeoutError)):
            category = "timeout"
        elif isinstance(e, aiohttp.ClientConnectionError):
            category = "network_error"
        elif isinstance(e, aiohttp.ClientError):
            category = "client_error"
        else:
            category = "unexpected_error"
        return None, {
            "category": category,
            "error_type": type(e).__name__,
            "latency_ms": round((time.monotonic() - started_at) * 1000, 1),
        }


async def fetch_chute_info(chute_id: str) -> Optional[dict]:
    started_at = time.monotonic()
    token = os.getenv("CHUTES_API_KEY", "")
    if not token or not chute_id:
        logger.debug("[Chutes] missing token or chute_id")
        category = "missing_api_key" if not token else "missing_chute_id"
        return _ChutesInfo(
            None,
            lookup_details={
                "category": category,
                "attempt_count": 0,
                "attempts": [],
                "total_latency_ms": round(
                    (time.monotonic() - started_at) * 1000, 1
                ),
            },
        )
    url = f"https://api.chutes.ai/chutes/{chute_id}"
    headers = {"Authorization": token}
    attempts: list[dict] = []

    for attempt in range(1, _CHUTES_FETCH_RETRIES + 1):
        data, attempt_details = await _chutes_get_json(url, headers=headers)
        attempts.append({"attempt": attempt, **attempt_details})
        if data:
            return _ChutesInfo(
                data,
                lookup_details={
                    "category": "success",
                    "attempt_count": len(attempts),
                    "attempts": attempts,
                    "total_latency_ms": round(
                        (time.monotonic() - started_at) * 1000, 1
                    ),
                },
            )
        if attempt < _CHUTES_FETCH_RETRIES:
            delay_s = _CHUTES_FETCH_BACKOFF_S * (2 ** (attempt - 1))
            logger.info(
                "[Chutes] retrying chute lookup chute_id=%s attempt=%s/%s in %.2fs",
                chute_id,
                attempt + 1,
                _CHUTES_FETCH_RETRIES,
                delay_s,
            )
            await asyncio.sleep(delay_s)
    return _ChutesInfo(
        None,
        lookup_details={
            "category": attempts[-1]["category"] if attempts else "unknown_error",
            "attempt_count": len(attempts),
            "attempts": attempts,
            "total_latency_ms": round((time.monotonic() - started_at) * 1000, 1),
        },
    )

def _pick_latest_miner_commit_for_element(
    arr,
    wanted_element_id: str | None,
    *,
    include_recovery: bool = True,
):
    best_blk = None
    best_data = None
    best_obj = None

    for blk, data in arr:
        try:
            blk_i = int(blk)
        except Exception:
            continue

        try:
            obj = json.loads(data)
        except Exception:
            continue

        role = obj.get("role")
        if role != "miner" and not (
            include_recovery
            and wanted_element_id is not None
            and is_recovery_commit(obj, wanted_element_id)
        ):
            continue

        committed_eid = obj.get("element_id")
        committed_eid = str(committed_eid).strip() if committed_eid is not None else None

        if wanted_element_id is not None and committed_eid != wanted_element_id:
            continue

        if best_blk is None or blk_i > best_blk:
            best_blk = blk_i
            best_data = data
            best_obj = obj

    return best_blk, best_data, best_obj


async def _find_miner_commit_via_archive_backfill(
    st_archive,
    *,
    netuid: int,
    hotkey: str,
    initial_arr,
    wanted_element_id: str,
    max_hops: int,
    first_block: int,
) -> tuple[int | None, dict | None]:
    if len(initial_arr or []) < 10:
        return None, None

    try:
        oldest_visible = min(int(x[0]) for x in initial_arr)
    except Exception:
        return None, None

    if oldest_visible < first_block:
        return None, None

    cursor = oldest_visible - 1
    prev_oldest = oldest_visible
    hops = 0

    while cursor >= first_block and hops < max_hops:
        try:
            hist = await st_archive.get_revealed_commitment_by_hotkey(
                netuid=netuid,
                hotkey_ss58_address=hotkey,
                block=cursor,
            )
        except Exception as e:
            logger.debug(
                "[Registry] archive backfill error hk=%s block=%s: %s",
                hotkey,
                cursor,
                e,
            )
            return None, None

        hist = list(hist or [])
        if not hist:
            return None, None

        blk, _data, obj = _pick_latest_miner_commit_for_element(
            hist,
            wanted_element_id,
            include_recovery=False,
        )
        if obj is not None:
            return int(blk or 0), obj

        if len(hist) < 10:
            return None, None

        try:
            oldest_hist = min(int(x[0]) for x in hist)
        except Exception:
            return None, None
        if oldest_hist < first_block:
            return None, None
        if oldest_hist >= prev_oldest:
            return None, None

        prev_oldest = oldest_hist
        cursor = oldest_hist - 1
        hops += 1

    return None, None


def _build_miner_candidate(uid: int, hotkey: str, obj: dict, block: int) -> Miner | None:
    model = obj.get("model")
    revision = obj.get("revision")
    slug = obj.get("slug")
    chute_id = obj.get("chute_id")
    committed_eid = obj.get("element_id")
    committed_eid = str(committed_eid).strip() if committed_eid is not None else None
    if not slug:
        return None
    return Miner(
        uid=uid,
        hotkey=hotkey,
        model=model,
        revision=revision,
        slug=slug,
        chute_id=chute_id,
        block=int(block or 0),
        element_id=committed_eid,
    )


def _join_key_to_base(index_url: str, key_or_url: str) -> str:
    key_or_url = str(key_or_url or "").strip()
    if key_or_url.startswith("http://") or key_or_url.startswith("https://"):
        return key_or_url
    base = index_url.rsplit("/", 1)[0] + "/"
    if key_or_url.startswith("/"):
        u = urlparse(index_url)
        return f"{u.scheme}://{u.netloc}{key_or_url}"
    return urljoin(base, key_or_url)


async def _hotkeys_with_prior_scores_for_element(
    *,
    netuid: int,
    element_id: str,
) -> set[str] | None:
    """
    Return hotkeys that already appear in validator index entries for this element.
    Returns None on lookup failure (caller should fail-open).
    """
    try:
        validator_indexes = await get_validator_indexes_from_chain(netuid)
    except Exception as e:
        logger.debug("[Registry] unable to read validator indexes from chain: %s", e)
        return None
    if not validator_indexes:
        return None

    safe_elem = str(element_id or "").strip().replace("/", "_")
    if not safe_elem:
        return None
    elem_seg = f"/manako/{safe_elem}/"
    found: set[str] = set()
    timeout = aiohttp.ClientTimeout(total=_REGISTRY_BACKFILL_INDEX_TIMEOUT_S)

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for _validator_hk, index_url in validator_indexes.items():
                try:
                    async with session.get(index_url) as resp:
                        if resp.status != 200:
                            continue
                        idx = await resp.json()
                except Exception:
                    continue

                keys: list[str] = []
                if isinstance(idx, list):
                    keys = [_join_key_to_base(index_url, k) for k in idx if isinstance(k, str)]
                elif isinstance(idx, dict) and isinstance(idx.get("entries"), list):
                    for entry in idx.get("entries", []):
                        p = entry.get("path")
                        if isinstance(p, str):
                            keys.append(_join_key_to_base(index_url, p))

                for u in keys:
                    try:
                        path = urlparse(u).path
                    except Exception:
                        continue
                    if elem_seg not in path:
                        continue
                    parts = [p for p in path.split("/") if p]
                    try:
                        root_i = parts.index("manako")
                    except ValueError:
                        continue
                    if len(parts) <= root_i + 3:
                        continue
                    hk = parts[root_i + 3]
                    if hk:
                        found.add(hk)
    except Exception as e:
        logger.debug("[Registry] element score index lookup failed: %s", e)
        return None
    return found

# ---------------------------- Miner registry main ----------------------------- #
async def get_miners_from_registry(
    netuid: int,
    *,
    element_id: str | None = None,
    first_block: int | None = None,
    max_model_size_mb: float | None = None,
    onnx_only: bool | None = None,
    blacklisted_hotkeys: set[str] | None = None,
    compliance_failure_tuples: set[ComplianceFailureTuple] | None = None,
    inactive_miner_tuples: set[InactiveMinerTuple] | None = None,
) -> tuple[Dict[int, Miner], Dict[int, Miner]]:
    """
    Reads on-chain commitments, verifies HF gating/revision, optional HF repo size
    cap, optional ONNX-only model artifact policy, and Chutes slug; then returns
    at most one miner per model revision (earliest block wins).
    """
    settings = get_settings()
    mechid = settings.SCOREVISION_MECHID

    if blacklisted_hotkeys is None:
        blacklisted_hotkeys = set()
    if blacklisted_hotkeys:
        logger.info("[Registry] loaded %d blacklisted hotkeys", len(blacklisted_hotkeys))
    if compliance_failure_tuples is None and element_id is not None:
        compliance_failure_tuples = await fetch_compliance_failure_tuples()
    if compliance_failure_tuples:
        logger.info(
            "[Registry] loaded %d compliance failing tuple(s)",
            len(compliance_failure_tuples),
        )
    if inactive_miner_tuples is None and element_id is not None:
        inactive_miner_tuples = await fetch_inactive_miner_tuples()
    if inactive_miner_tuples:
        logger.info("[Registry] loaded %d inactive miner tuple(s)", len(inactive_miner_tuples))

    try:
        st = await get_subtensor()
    except Exception as e:
        logger.warning(
            "[Registry] failed to initialize subtensor (netuid=%s mechid=%s): %s",
            netuid,
            mechid,
            e,
        )
        reset_subtensor()
        return {}, {}

    logger.info(
        "[Registry] extracting candidates (netuid=%s mechid=%s element_id=%s)",
        netuid,
        mechid,
        element_id,
    )

    try:
        meta = await st.metagraph(netuid, mechid=mechid)
        commits = await get_all_revealed_commitments(st, netuid)
    except Exception as e:
        logger.warning("[Registry] error while fetching metagraph/commitments: %s", e)
        reset_subtensor()
        return {}, {}

    # 1) Extract candidates (uid -> Miner)
    candidates: Dict[int, Miner] = {}
    skipped: Dict[int, Miner] = {}
    wanted = str(element_id).strip() if element_id is not None else None
    resolved_first_block = (
        int(first_block)
        if first_block is not None
        else _REGISTRY_COMMIT_BACKFILL_FIRST_BLOCK
    )
    unresolved_for_backfill: list[tuple[int, str, list]] = []
    unresolved_for_recovery: list[tuple[int, str, int]] = []
    for uid, hk in enumerate(meta.hotkeys):
        bypass_registry_checks = is_registry_bypass(uid, hk)
        if hk in blacklisted_hotkeys and not bypass_registry_checks:
            logger.debug("[Registry] skipping blacklisted hotkey=%s", hk)
            continue
        if hk in blacklisted_hotkeys and bypass_registry_checks:
            logger.info("[Registry] uid=%s hotkey=%s bypassed blacklist", uid, hk)
        arr = commits.get(hk)
        if not arr:
            continue
        best_blk, _best_data, obj = _pick_latest_miner_commit_for_element(arr, wanted)
        if obj is None and wanted is not None:
            unresolved_for_backfill.append((uid, hk, list(arr)))
            continue
        if obj is None:
            continue

        if is_recovery_commit(obj, wanted, hotkey=hk):
            if wanted is not None:
                unresolved_for_recovery.append((uid, hk, int(best_blk or 0)))
            continue

        cand = _build_miner_candidate(uid, hk, obj, int(best_blk or 0))
        if cand is not None:
            if is_inactive_miner_tuple(
                inactive_miner_tuples,
                hotkey=cand.hotkey,
                element_id=cand.element_id,
                commit_block=cand.block,
            ):
                logger.info(
                    "[Registry] uid=%s hotkey=%s element_id=%s "
                    "commit_block=%s ignored: inactive miner tuple",
                    uid,
                    cand.hotkey,
                    cand.element_id,
                    cand.block,
                )
                continue
            if is_compliance_tuple_failed(
                compliance_failure_tuples,
                hotkey=cand.hotkey,
                element_id=cand.element_id,
                commit_block=cand.block,
            ):
                cand.registry_skip_reason = "compliance_failed_tuple"
                skipped[uid] = cand
                logger.info(
                    "[Registry] uid=%s hotkey=%s element_id=%s commit_block=%s skipped: compliance failed tuple",
                    uid,
                    cand.hotkey,
                    cand.element_id,
                    cand.block,
                )
                continue
            candidates[uid] = cand

    if wanted is not None and unresolved_for_recovery:
        try:
            validator_indexes = await get_validator_indexes_from_chain(netuid)
            recovered = await recover_commitments_from_shards(
                {(hk, wanted) for _uid, hk, _recovery_block in unresolved_for_recovery},
                validator_indexes,
            )
            for uid, hk, recovery_block in unresolved_for_recovery:
                recovered_commitment = recovered.get((hk, wanted))
                if recovered_commitment is None:
                    logger.warning(
                        "[Registry] recovery unresolved uid=%s hotkey=%s element_id=%s "
                        "recovery_block=%s: no valid matching public shard",
                        uid,
                        hk,
                        wanted,
                        recovery_block,
                    )
                    continue
                cand = _build_miner_candidate(
                    uid,
                    hk,
                    recovered_commitment.as_miner_commitment(hk),
                    recovered_commitment.commit_block,
                )
                if cand is None:
                    continue
                if is_inactive_miner_tuple(
                    inactive_miner_tuples,
                    hotkey=cand.hotkey,
                    element_id=cand.element_id,
                    commit_block=cand.block,
                ):
                    logger.info(
                        "[Registry] recovered uid=%s hotkey=%s element_id=%s "
                        "commit_block=%s ignored: inactive miner tuple",
                        uid,
                        cand.hotkey,
                        cand.element_id,
                        cand.block,
                    )
                    continue
                if is_compliance_tuple_failed(
                    compliance_failure_tuples,
                    hotkey=cand.hotkey,
                    element_id=cand.element_id,
                    commit_block=cand.block,
                ):
                    cand.registry_skip_reason = "compliance_failed_tuple"
                    skipped[uid] = cand
                    continue
                candidates[uid] = cand
                logger.info(
                    "[Registry] recovered uid=%s hotkey=%s element_id=%s "
                    "original_commit_block=%s from shard_block=%s",
                    uid,
                    hk,
                    wanted,
                    cand.block,
                    recovered_commitment.shard_block,
                )
        except Exception as e:
            logger.warning("[Registry] commitment recovery disabled due to error: %s", e)

    if (
        wanted is not None
        and _REGISTRY_COMMIT_BACKFILL_ENABLE
        and unresolved_for_backfill
        and _REGISTRY_COMMIT_BACKFILL_ARCHIVE_ENDPOINT
    ):
        scored_hotkeys = await _hotkeys_with_prior_scores_for_element(
            netuid=netuid,
            element_id=wanted,
        )
        if scored_hotkeys is not None:
            before = len(unresolved_for_backfill)
            unresolved_for_backfill = [
                (uid_i, hk_i, arr_i)
                for (uid_i, hk_i, arr_i) in unresolved_for_backfill
                if hk_i in scored_hotkeys
            ]
            logger.info(
                "[Registry] backfill prefilter by score index element=%s kept=%d/%d unresolved hotkeys",
                wanted,
                len(unresolved_for_backfill),
                before,
            )
        if not unresolved_for_backfill:
            logger.info("[Registry] no unresolved hotkeys left after score-index prefilter")
        else:
            logger.info(
                "[Registry] archive backfill enabled for element=%s unresolved_hotkeys=%d endpoint=%s",
                wanted,
                len(unresolved_for_backfill),
                _REGISTRY_COMMIT_BACKFILL_ARCHIVE_ENDPOINT,
            )
            st_archive = None
            try:
                st_archive = AsyncSubtensor(
                    network=_REGISTRY_COMMIT_BACKFILL_ARCHIVE_ENDPOINT
                )
                await asyncio.wait_for(st_archive.initialize(), timeout=20.0)
                sem = asyncio.Semaphore(_REGISTRY_COMMIT_BACKFILL_CONCURRENCY)

                async def _resolve_one(uid_hk_arr: tuple[int, str, list]):
                    uid_i, hk_i, arr_i = uid_hk_arr
                    async with sem:
                        blk_i, obj_i = await _find_miner_commit_via_archive_backfill(
                            st_archive,
                            netuid=netuid,
                            hotkey=hk_i,
                            initial_arr=arr_i,
                            wanted_element_id=wanted,
                            max_hops=_REGISTRY_COMMIT_BACKFILL_MAX_HOPS,
                            first_block=resolved_first_block,
                        )
                    return uid_i, hk_i, blk_i, obj_i

                results = await asyncio.gather(
                    *[_resolve_one(item) for item in unresolved_for_backfill],
                    return_exceptions=True,
                )

                added = 0
                for item in results:
                    if isinstance(item, Exception):
                        logger.debug("[Registry] archive backfill worker failed: %s", item)
                        continue
                    uid_i, hk_i, blk_i, obj_i = item
                    if obj_i is None or blk_i is None:
                        continue

                    cand = _build_miner_candidate(uid_i, hk_i, obj_i, int(blk_i))
                    if cand is not None:
                        if is_inactive_miner_tuple(
                            inactive_miner_tuples,
                            hotkey=cand.hotkey,
                            element_id=cand.element_id,
                            commit_block=cand.block,
                        ):
                            logger.info(
                                "[Registry] uid=%s hotkey=%s element_id=%s "
                                "commit_block=%s ignored: inactive miner tuple",
                                uid_i,
                                cand.hotkey,
                                cand.element_id,
                                cand.block,
                            )
                            continue
                        if is_compliance_tuple_failed(
                            compliance_failure_tuples,
                            hotkey=cand.hotkey,
                            element_id=cand.element_id,
                            commit_block=cand.block,
                        ):
                            cand.registry_skip_reason = "compliance_failed_tuple"
                            skipped[uid_i] = cand
                            logger.info(
                                "[Registry] uid=%s hotkey=%s element_id=%s commit_block=%s skipped: compliance failed tuple",
                                uid_i,
                                cand.hotkey,
                                cand.element_id,
                                cand.block,
                            )
                            continue
                        candidates[uid_i] = cand
                        added += 1
                logger.info("[Registry] archive backfill added %d candidate(s)", added)
            except Exception as e:
                logger.warning("[Registry] archive backfill disabled due to error: %s", e)
            finally:
                if st_archive is not None and hasattr(st_archive, "close"):
                    try:
                        await st_archive.close()
                    except Exception:
                        pass

    logger.info("[Registry] %d on-chain candidates", len(candidates))
    if not candidates:
        logger.warning("[Registry] No on-chain candidates")
        return {}, skipped

    def _mark_skipped(
        uid: int, miner: Miner, reason: str, details: Optional[dict] = None
    ) -> None:
        miner.registry_skip_reason = reason
        miner.registry_skip_details = details
        skipped[uid] = miner

    # 2) Filter by HF gating/inaccessible + Chutes slug/revision checks
    filtered: Dict[int, Miner] = {}
    for uid, m in candidates.items():
        if is_registry_bypass(uid, m.hotkey):
            logger.info("[Registry] uid=%s hotkey=%s bypassed registry filters", uid, m.hotkey)
            filtered[uid] = m
            continue

        gated = await _hf_gated_or_inaccessible(m.model, m.revision)
        if gated is True:
            logger.info("[Registry] uid=%s slug=%s skipped: HF gated/inaccessible", uid, m.slug)
            _mark_skipped(uid, m, "hf_gated_or_revision_inaccessible")
            continue
        if max_model_size_mb is not None and max_model_size_mb > 0:
            if not m.model:
                logger.info(
                    "[Registry] uid=%s slug=%s skipped: missing HF model id for size check",
                    uid,
                    m.slug,
                )
                _mark_skipped(uid, m, "missing_hf_model_id_for_size_check")
                continue
            model_size_mb = _hf_repo_total_size_mb(m.model, m.revision)
            if model_size_mb is None:
                logger.info(
                    "[Registry] uid=%s slug=%s skipped: unable to resolve HF repo size",
                    uid,
                    m.slug,
                )
                _mark_skipped(uid, m, "hf_repo_size_unresolved")
                continue
            if model_size_mb > max_model_size_mb:
                logger.info(
                    "[Registry] uid=%s slug=%s skipped: HF repo size %.2fMB exceeds max %.2fMB",
                    uid,
                    m.slug,
                    model_size_mb,
                    max_model_size_mb,
                )
                _mark_skipped(
                    uid,
                    m,
                    f"hf_repo_size_exceeds_max:{model_size_mb:.2f}>{max_model_size_mb:.2f}",
                )
                continue

        if onnx_only is True:
            if not m.model:
                logger.info(
                    "[Registry] uid=%s slug=%s skipped: missing HF model id for onnx check",
                    uid,
                    m.slug,
                )
                _mark_skipped(uid, m, "missing_hf_model_id_for_onnx_check")
                continue
            model_is_onnx_only = _hf_repo_has_only_onnx_models(m.model, m.revision)
            if model_is_onnx_only is not True:
                logger.info(
                    "[Registry] uid=%s slug=%s skipped: HF repo is not onnx-only",
                    uid,
                    m.slug,
                )
                _mark_skipped(uid, m, "hf_repo_not_onnx_only")
                continue

        ok = True
        chute_reason = None
        chute_lookup_details = None
        if m.chute_id:
            try:
                info = await fetch_chute_info(m.chute_id)
                chute_lookup_details = getattr(info, "lookup_details", None)
            except Exception as e:
                logger.info("[Registry] uid=%s slug=%s: Chutes lookup error: %s", uid, m.slug, e)
                info = None
                chute_lookup_details = {
                    "category": "unexpected_error",
                    "attempt_count": 0,
                    "attempts": [],
                    "error_type": type(e).__name__,
                }
            if not info:
                lookup_category = (
                    chute_lookup_details.get("category")
                    if isinstance(chute_lookup_details, dict)
                    else "unknown_error"
                )
                logger.info(
                    "[Registry] uid=%s slug=%s: Chutes unfetched (%s)",
                    uid,
                    m.slug,
                    lookup_category,
                )
                ok = False
                chute_reason = "chutes_unfetched"
            else:
                slug_chutes = (info.get("slug") or "").strip()
                if slug_chutes and slug_chutes != (m.slug or ""):
                    ok = False
                    chute_reason = f"chutes_slug_mismatch:{slug_chutes}!={m.slug or ''}"
                    logger.info(
                        "[Registry] uid=%s: slug mismatch (chutes=%s, commit=%s)",
                        uid,
                        slug_chutes,
                        m.slug,
                    )
                ch_rev = info.get("revision")
                if ch_rev and m.revision and str(ch_rev) != str(m.revision):
                    ok = False
                    chute_reason = f"chutes_revision_mismatch:{ch_rev}!={m.revision}"
                    logger.info(
                        "[Registry] uid=%s: revision mismatch (chutes=%s, commit=%s)",
                        uid,
                        ch_rev,
                        m.revision,
                    )

        if ok:
            filtered[uid] = m
        else:
            details = None
            if chute_lookup_details is not None:
                details = {"chutes_lookup": chute_lookup_details}
            _mark_skipped(
                uid,
                m,
                chute_reason or "chutes_validation_failed",
                details,
            )

    logger.info("[Registry] %d miners after filtering", len(filtered))
    if not filtered:
        logger.warning("[Registry] Filter produced no eligible miners")
        return {}, skipped

    # 3) De-duplicate by model revision: keep earliest block per pair (stable)
    best_by_model_revision: Dict[Tuple[str, str], Tuple[int, int]] = {}
    for uid, m in filtered.items():
        if not m.model and is_registry_bypass(uid, m.hotkey):
            dedup_key = (f"__bypass_uid_{uid}", "")
        elif m.model:
            dedup_key = (m.model, m.revision or "")
        else:
            continue
        blk = m.block if isinstance(m.block, int) else (int(m.block) if m.block is not None else (2**63 - 1))
        prev = best_by_model_revision.get(dedup_key)
        if prev is None or blk < prev[0]:
            best_by_model_revision[dedup_key] = (blk, uid)

    keep_uids = {uid for _, uid in best_by_model_revision.values()}
    kept = {uid: filtered[uid] for uid in keep_uids if uid in filtered}
    dedup_skipped = {}
    for uid, miner in filtered.items():
        if uid in keep_uids:
            continue
        winner = best_by_model_revision.get((miner.model or "", miner.revision or ""))
        if winner is not None:
            winner_blk, winner_uid = winner
            miner.registry_skip_reason = (
                f"dedup_by_model_revision_kept_uid:{winner_uid}_block:{winner_blk}"
            )
        else:
            miner.registry_skip_reason = "dedup_by_model_revision"
        dedup_skipped[uid] = miner
    skipped.update(dedup_skipped)
    logger.info("[Registry] %d miners kept after de-dup by model revision", len(kept))
    logger.info("[Registry] %d miners skipped", len(skipped))

    return kept, skipped
