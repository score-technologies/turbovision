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


def r2_get_object():
    return True


def r2_put_json():
    return True


def r2_delete_object():
    return True


__all__ = [
    "build_index_url",
    "build_public_index_url_from_base",
    "bucket_base_from_index",
    "extract_base_url",
    "extract_block_from_key",
    "fetch_head_metadata",
    "fetch_index_keys",
    "fetch_json_from_url",
    "fetch_miner_predictions",
    "fetch_responses_data",
    "fetch_shard_lines",
    "filter_keys_by_tail",
    "normalize_index_url",
    "r2_get_object",
    "r2_put_json",
    "r2_delete_object",
]
