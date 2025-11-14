import json
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError

from scorevision.utils.settings import get_settings
from scorevision.utils.retry import retry_network


def get_r2_client():
    settings = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=settings.SCOREVISION_ENDPOINT,
        aws_access_key_id=settings.SCOREVISION_ACCESS_KEY,
        aws_secret_access_key=settings.SCOREVISION_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


@retry_network
def r2_get_object(bucket, key):
    client = get_r2_client()
    try:
        res = client.get_object(Bucket=bucket, Key=key)
        return res["Body"].read(), res.get("ETag")
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            return None, None
        raise


@retry_network
def r2_put_json(bucket, key, data, acl="public-read", if_match=None):
    client = get_r2_client()

    extra = {}
    if if_match:
        extra["IfMatch"] = if_match

    return client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data).encode("utf-8"),
        ContentType="application/json",
        ACL=acl,
        **extra,
    )


@retry_network
def r2_delete_object(bucket, key):
    client = get_r2_client()
    return client.delete_object(Bucket=bucket, Key=key)

