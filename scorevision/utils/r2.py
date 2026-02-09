from json import dumps

import boto3
from botocore.config import Config as BotoConfig

from scorevision.utils.settings import get_settings


def _r2_client():
    settings = get_settings()
    if not (
        settings.R2_ACCOUNT_ID.get_secret_value()
        and settings.R2_WRITE_ACCESS_KEY_ID.get_secret_value()
        and settings.R2_WRITE_SECRET_ACCESS_KEY.get_secret_value()
    ):
        raise RuntimeError("R2 credentials not set")
    session = boto3.session.Session()
    return session.client(
        "s3",
        endpoint_url=(
            f"https://{settings.R2_ACCOUNT_ID.get_secret_value()}.r2.cloudflarestorage.com"
        ),
        aws_access_key_id=settings.R2_WRITE_ACCESS_KEY_ID.get_secret_value(),
        aws_secret_access_key=settings.R2_WRITE_SECRET_ACCESS_KEY.get_secret_value(),
        config=BotoConfig(max_pool_connections=settings.R2_CONCURRENCY),
    )


def r2_get_object(bucket: str, key: str) -> tuple[bytes | None, str | None]:
    client = _r2_client()
    try:
        resp = client.get_object(Bucket=bucket, Key=key)
    except client.exceptions.NoSuchKey:
        return None, None
    body = resp["Body"].read()
    etag = resp.get("ETag")
    return body, etag


def r2_put_bytes(
    bucket: str,
    key: str,
    body: bytes,
    *,
    content_type: str = "application/octet-stream",
) -> None:
    client = _r2_client()
    kwargs = {
        "Bucket": bucket,
        "Key": key,
        "Body": body,
        "ContentType": content_type,
    }
    client.put_object(**kwargs)


def r2_put_json(bucket: str, key: str, obj) -> None:
    body = dumps(obj, separators=(",", ":")).encode()
    r2_put_bytes(
        bucket,
        key,
        body,
        content_type="application/json",
    )


def r2_delete_object(bucket: str, key: str) -> None:
    client = _r2_client()
    client.delete_object(Bucket=bucket, Key=key)
