from dataclasses import dataclass
from json import dumps, loads
import asyncio
from aiobotocore.session import get_session
from botocore.config import Config as BotoConfig

from scorevision.utils.r2_public import (
    build_index_url,
    build_public_index_url_from_base,
    bucket_base_from_index,
    extract_base_url,
    extract_block_from_key,
    fetch_head_metadata,
    fetch_index_keys,
    fetch_json_from_url,
    fetch_miner_predictions,
    fetch_responses_data,
    fetch_shard_lines,
    filter_keys_by_tail,
    normalize_index_url,
)
from scorevision.utils.settings import Settings, get_settings


@dataclass(frozen=True)
class R2Config:
    bucket: str
    account_id: str
    access_key_id: str
    secret_access_key: str
    concurrency: int


def central_r2_config(settings: Settings) -> R2Config:
    return R2Config(
        bucket=settings.SCOREVISION_BUCKET,
        account_id=settings.CENTRAL_R2_ACCOUNT_ID.get_secret_value(),
        access_key_id=settings.CENTRAL_R2_WRITE_ACCESS_KEY_ID.get_secret_value(),
        secret_access_key=settings.CENTRAL_R2_WRITE_SECRET_ACCESS_KEY.get_secret_value(),
        concurrency=settings.CENTRAL_R2_CONCURRENCY,
    )


def audit_r2_config(settings: Settings) -> R2Config:
    return R2Config(
        bucket=settings.AUDIT_R2_BUCKET,
        account_id=settings.AUDIT_R2_ACCOUNT_ID.get_secret_value(),
        access_key_id=settings.AUDIT_R2_WRITE_ACCESS_KEY_ID.get_secret_value(),
        secret_access_key=settings.AUDIT_R2_WRITE_SECRET_ACCESS_KEY.get_secret_value(),
        concurrency=settings.AUDIT_R2_CONCURRENCY,
    )


def is_configured(cfg: R2Config, *, require_bucket: bool = True) -> bool:
    if require_bucket and not cfg.bucket:
        return False
    return bool(cfg.account_id and cfg.access_key_id and cfg.secret_access_key)


def create_s3_client(cfg: R2Config, *, error_message: str):
    if not is_configured(cfg, require_bucket=False):
        raise RuntimeError(error_message)
    sess = get_session()
    return sess.create_client(
        "s3",
        endpoint_url=f"https://{cfg.account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=cfg.access_key_id,
        aws_secret_access_key=cfg.secret_access_key,
        config=BotoConfig(max_pool_connections=cfg.concurrency),
    )


def is_not_found_error(exc: Exception) -> bool:
    code = ""
    response = getattr(exc, "response", None) or {}
    error = response.get("Error") if isinstance(response, dict) else None
    if isinstance(error, dict):
        code = str(error.get("Code") or "").strip()
    return code in {"404", "NoSuchKey", "NotFound"} or "Not Found" in str(exc)


async def ensure_index_exists(
    *,
    client_factory,
    bucket: str,
    index_key: str = "scorevision/index.json",
) -> bool:
    async with client_factory() as c:
        try:
            await c.head_object(Bucket=bucket, Key=index_key)
            return True
        except Exception as e:
            if not is_not_found_error(e):
                raise
            await c.put_object(
                Bucket=bucket,
                Key=index_key,
                Body="[]",
                ContentType="application/json",
            )
            return True


async def add_index_key_if_new(
    *,
    client_factory,
    bucket: str,
    key: str,
    index_key: str = "scorevision/index.json",
) -> bool:
    async with client_factory() as c:
        try:
            r = await c.get_object(Bucket=bucket, Key=index_key)
            items = set(loads(await r["Body"].read()))
        except Exception as e:
            if not is_not_found_error(e):
                raise
            items = set()
        if key in items:
            return False
        items.add(key)
        await c.put_object(
            Bucket=bucket,
            Key=index_key,
            Body=dumps(sorted(items)),
            ContentType="application/json",
        )
        return True


def _current_central_cfg() -> R2Config:
    return central_r2_config(get_settings())


async def _r2_get_object_async(bucket: str, key: str) -> tuple[bytes | None, str | None]:
    cfg = _current_central_cfg()
    async with create_s3_client(cfg, error_message="Central R2 credentials not set") as c:
        try:
            obj = await c.get_object(Bucket=bucket, Key=key)
        except Exception as e:
            if is_not_found_error(e):
                return None, None
            raise
        body = await obj["Body"].read()
        etag = obj.get("ETag")
        return body, etag


def r2_get_object(bucket: str, key: str) -> tuple[bytes | None, str | None]:
    return asyncio.run(_r2_get_object_async(bucket, key))


async def _r2_put_json_async(
    bucket: str,
    key: str,
    payload: dict | list,
    if_match: str | None = None,
) -> None:
    cfg = _current_central_cfg()
    async with create_s3_client(cfg, error_message="Central R2 credentials not set") as c:
        if if_match:
            head = await c.head_object(Bucket=bucket, Key=key)
            current_etag = str(head.get("ETag") or "").strip('"')
            expected_etag = str(if_match).strip('"')
            if current_etag != expected_etag:
                raise RuntimeError(
                    f"ETag mismatch for {bucket}/{key}: expected={expected_etag} got={current_etag}"
                )
        await c.put_object(
            Bucket=bucket,
            Key=key,
            Body=dumps(payload, separators=(",", ":")),
            ContentType="application/json",
        )


def r2_put_json(
    bucket: str,
    key: str,
    payload: dict | list,
    if_match: str | None = None,
) -> None:
    asyncio.run(_r2_put_json_async(bucket, key, payload, if_match=if_match))


async def _r2_delete_object_async(bucket: str, key: str) -> None:
    cfg = _current_central_cfg()
    async with create_s3_client(cfg, error_message="Central R2 credentials not set") as c:
        await c.delete_object(Bucket=bucket, Key=key)


def r2_delete_object(bucket: str, key: str) -> None:
    asyncio.run(_r2_delete_object_async(bucket, key))


__all__ = [
    "R2Config",
    "add_index_key_if_new",
    "audit_r2_config",
    "build_index_url",
    "build_public_index_url_from_base",
    "bucket_base_from_index",
    "central_r2_config",
    "create_s3_client",
    "ensure_index_exists",
    "extract_base_url",
    "extract_block_from_key",
    "fetch_head_metadata",
    "fetch_index_keys",
    "fetch_json_from_url",
    "fetch_miner_predictions",
    "fetch_responses_data",
    "fetch_shard_lines",
    "filter_keys_by_tail",
    "is_configured",
    "is_not_found_error",
    "normalize_index_url",
    "r2_delete_object",
    "r2_get_object",
    "r2_put_json",
]
