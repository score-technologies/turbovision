from scorevision.utils.cloudflare_helpers import (
    _extract_element_miner_commit_tuple_from_key_or_url,
)


def test_extract_element_miner_commit_tuple_with_commit_folder():
    key = "scorevision/PlayerDetect_v1@1.0/5Fminer/007471716/evaluation/007471800-50414.json"
    assert _extract_element_miner_commit_tuple_from_key_or_url(key) == (
        "PlayerDetect_v1@1.0",
        "5Fminer",
        7471716,
    )


def test_extract_element_miner_commit_tuple_legacy_layout():
    key = "scorevision/PlayerDetect_v1@1.0/5Fminer/evaluation/007471800-50414.json"
    assert _extract_element_miner_commit_tuple_from_key_or_url(key) == (
        "PlayerDetect_v1@1.0",
        "5Fminer",
        -1,
    )


def test_extract_element_miner_commit_tuple_from_absolute_url():
    url = "https://pub.r2.dev/scorevision/PlayerDetect_v1@1.0/5Fminer/007471716/evaluation/007471800-50414.json"
    assert _extract_element_miner_commit_tuple_from_key_or_url(url) == (
        "PlayerDetect_v1@1.0",
        "5Fminer",
        7471716,
    )
