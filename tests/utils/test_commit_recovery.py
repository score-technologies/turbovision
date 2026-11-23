import pytest

from scorevision.utils import commit_recovery


HOTKEY = "5Fminer"
ELEMENT_ID = "PlayerDetect_v1@1.0"
SHARD_KEY = (
    "https://validator.example/manako/PlayerDetect_v1@1.0/5Fminer/"
    "000000123/evaluation/000000999-challenge.json"
)


@pytest.fixture(autouse=True)
def clear_recovery_cache():
    commit_recovery._RECOVERY_CACHE.clear()


def _signed_line(
    *,
    element_id: str = ELEMENT_ID,
    hotkey: str = HOTKEY,
    commit_block: int = 123,
) -> dict:
    return {
        "hotkey": "5Fvalidator",
        "signature": "0xsigned",
        "payload": {
            "element_id": element_id,
            "lane": "public",
            "telemetry": {
                "miner": {
                    "hotkey": hotkey,
                    "model": "org/model",
                    "revision": "rev1",
                    "slug": "slug1",
                    "chute_id": "chute1",
                    "commitment": {
                        "element_id": element_id,
                        "model": "org/model",
                        "revision": "rev1",
                        "chute_slug": "slug1",
                        "chute_id": "chute1",
                        "commit_block": commit_block,
                    },
                }
            },
        },
    }


def test_extract_recovered_commitment_requires_matching_signed_shard(monkeypatch):
    monkeypatch.setattr(
        "scorevision.utils.cloudflare_helpers._verify_signature",
        lambda _hotkey, _payload, _signature: True,
    )

    recovered = commit_recovery.extract_recovered_commitment(
        _signed_line(),
        shard_key=SHARD_KEY,
        hotkey=HOTKEY,
        element_id=ELEMENT_ID,
    )

    assert recovered is not None
    assert recovered.commit_block == 123
    assert recovered.shard_block == 999
    assert recovered.as_miner_commitment(HOTKEY)["slug"] == "slug1"


def test_extract_recovered_commitment_rejects_hotkey_mismatch(monkeypatch):
    monkeypatch.setattr(
        "scorevision.utils.cloudflare_helpers._verify_signature",
        lambda _hotkey, _payload, _signature: True,
    )
    assert (
        commit_recovery.extract_recovered_commitment(
            _signed_line(hotkey="different-hotkey"),
            shard_key=SHARD_KEY,
            hotkey=HOTKEY,
            element_id=ELEMENT_ID,
        )
        is None
    )


def test_recovery_counts_only_indexed_element_triggers():
    assert commit_recovery.is_timely_recovery(
        300, 5000, [300, 600, 900, 1200, 1500, 1800, 2100], activation_block=1
    )
    assert not commit_recovery.is_timely_recovery(
        300, 5000, [300, 600, 900, 1200, 1500, 1800, 2100, 4800], activation_block=1
    )
    assert not commit_recovery.is_timely_recovery(
        300, 300, [300], activation_block=1
    )


def test_recovery_before_activation_is_grandfathered():
    triggers = (300, 600, 900, 1200, 1500, 1800, 2100, 2400)
    assert commit_recovery.is_timely_recovery(
        300, 2900, triggers, activation_block=3000
    )
    assert not commit_recovery.is_timely_recovery(
        300, 3000, triggers, activation_block=3000
    )
    assert commit_recovery.is_timely_recovery(300, 3000, triggers)


@pytest.mark.asyncio
async def test_recovery_only_uses_shards_before_recovery_block(monkeypatch):
    later_key = SHARD_KEY.replace("000000999", "000001200")
    fetched = []

    async def fake_fetch_json(_url):
        return [SHARD_KEY, later_key]

    async def fake_fetch_lines(_public_url, key):
        fetched.append(key)
        return [_signed_line()]

    monkeypatch.setattr(commit_recovery, "fetch_json_from_url", fake_fetch_json)
    monkeypatch.setattr(commit_recovery, "fetch_shard_lines", fake_fetch_lines)
    monkeypatch.setattr(
        "scorevision.utils.cloudflare_helpers._verify_signature",
        lambda _hotkey, _payload, _signature: True,
    )

    recovered = await commit_recovery.recover_commitments_from_shards(
        {(HOTKEY, ELEMENT_ID)},
        {"validator": "https://validator.example/manako/index.json"},
        recovery_blocks={(HOTKEY, ELEMENT_ID): 1100},
    )

    assert recovered[(HOTKEY, ELEMENT_ID)].shard_block == 999
    assert fetched == [SHARD_KEY]


@pytest.mark.asyncio
@pytest.mark.parametrize("missed_triggers,accepted", [(20, True), (21, False)])
async def test_pre_index_recovery_uses_recent_signed_continuity_shard(
    monkeypatch, missed_triggers, accepted
):
    monkeypatch.setattr(commit_recovery, "RECOVERY_ACTIVATION_BLOCK", 500)
    monkeypatch.setattr(commit_recovery, "CONTINUITY_ACTIVATION_BLOCK", 1000)
    latest_key = SHARD_KEY.replace("000000999", "000001000")
    trigger_keys = [
        latest_key.replace("5Fminer", "other-miner").replace(
            "000001000", f"{block:09d}"
        )
        for block in range(1100, 1100 + missed_triggers * 100, 100)
    ]

    async def fake_fetch_json(_url):
        return [latest_key, *trigger_keys]

    async def fake_fetch_lines(_public_url, key):
        assert key == latest_key
        return [_signed_line()]

    monkeypatch.setattr(commit_recovery, "fetch_json_from_url", fake_fetch_json)
    monkeypatch.setattr(commit_recovery, "fetch_shard_lines", fake_fetch_lines)
    monkeypatch.setattr(
        "scorevision.utils.cloudflare_helpers._verify_signature",
        lambda *_args: True,
    )

    recovered = await commit_recovery.recover_commitments_from_shards(
        {(HOTKEY, ELEMENT_ID)},
        {"validator": "https://validator.example/manako/index.json"},
        recovery_blocks={(HOTKEY, ELEMENT_ID): 400},
    )

    proof = recovered[(HOTKEY, ELEMENT_ID)]
    assert proof.proof_mode == "continuity"
    assert commit_recovery.is_timely_recovered_commitment(proof, 400) is accepted


@pytest.mark.asyncio
async def test_recovery_after_index_start_cannot_use_later_shard(monkeypatch):
    monkeypatch.setattr(commit_recovery, "CONTINUITY_ACTIVATION_BLOCK", 1000)
    later_key = SHARD_KEY.replace("000000999", "000001200")
    older_other_element_key = (
        "https://validator.example/manako/OtherElement/other-miner/"
        "000000123/evaluation/000001000-challenge.json"
    )
    fetched = []

    async def fake_fetch_json(_url):
        return [older_other_element_key, later_key]

    async def fake_fetch_lines(_public_url, key):
        fetched.append(key)
        return [_signed_line()]

    monkeypatch.setattr(commit_recovery, "fetch_json_from_url", fake_fetch_json)
    monkeypatch.setattr(commit_recovery, "fetch_shard_lines", fake_fetch_lines)

    recovered = await commit_recovery.recover_commitments_from_shards(
        {(HOTKEY, ELEMENT_ID)},
        {"validator": "https://validator.example/manako/index.json"},
        recovery_blocks={(HOTKEY, ELEMENT_ID): 1100},
    )

    assert recovered == {}
    assert fetched == []


@pytest.mark.asyncio
@pytest.mark.parametrize("missed_triggers,accepted", [(6, True), (7, False)])
async def test_post_activation_pre_index_recovery_keeps_six_trigger_limit(
    monkeypatch, missed_triggers, accepted
):
    monkeypatch.setattr(commit_recovery, "CONTINUITY_ACTIVATION_BLOCK", 1000)
    latest_key = SHARD_KEY.replace("000000999", "000001200")
    trigger_keys = [
        latest_key.replace("5Fminer", "other-miner").replace(
            "000001200", f"{block:09d}"
        )
        for block in range(1300, 1300 + missed_triggers * 100, 100)
    ]

    async def fake_fetch_json(_url):
        return [latest_key, *trigger_keys]

    async def fake_fetch_lines(_public_url, key):
        assert key == latest_key
        return [_signed_line()]

    monkeypatch.setattr(commit_recovery, "fetch_json_from_url", fake_fetch_json)
    monkeypatch.setattr(commit_recovery, "fetch_shard_lines", fake_fetch_lines)
    monkeypatch.setattr(
        "scorevision.utils.cloudflare_helpers._verify_signature",
        lambda *_args: True,
    )

    recovered = await commit_recovery.recover_commitments_from_shards(
        {(HOTKEY, ELEMENT_ID)},
        {"validator": "https://validator.example/manako/index.json"},
        recovery_blocks={(HOTKEY, ELEMENT_ID): 1100},
    )

    proof = recovered[(HOTKEY, ELEMENT_ID)]
    assert proof.proof_mode == "continuity"
    assert commit_recovery.is_timely_recovered_commitment(proof, 1100) is accepted


@pytest.mark.asyncio
async def test_reported_pre_index_recoveries_resolve_from_active_shards(monkeypatch):
    cases = [
        (
            "5CStyAPJzcHkaHxVptqQVEdqLxmXWc2hjGyk4rurHhZSdZii",
            "manak0/Detect-worksite-ppe-compliance-detection-probe-set",
            9410558,
            9523637,
            9562800,
        ),
        (
            "5DyJHRJipprzzVaBmT5mBsXPgePRUVyTBJ7wcw9d6pMqGDF4",
            "manak0/Detect-Person",
            9504665,
            9518697,
            9563100,
        ),
        (
            "5Ff9ovzQsyzqTFzvpgsPgXWUZsW5zMhN1MeSRCrTpydvpMdE",
            "manak0/Detect-beverage-detect",
            9500177,
            9522261,
            9563100,
        ),
    ]
    earliest_key = (
        "https://validator.example/manako/other-element/other-miner/"
        "000000123/evaluation/009527100-challenge.json"
    )
    keys = [earliest_key]
    lines_by_key = {}
    recovery_blocks = {}
    for hotkey, element_id, commit_block, recovery_block, shard_block in cases:
        request = (hotkey, element_id)
        recovery_blocks[request] = recovery_block
        safe_element = element_id.replace("/", "_")
        key = (
            f"https://validator.example/manako/{safe_element}/{hotkey}/"
            f"{commit_block:09d}/evaluation/{shard_block:09d}-challenge.json"
        )
        keys.append(key)
        lines_by_key[key] = [
            _signed_line(
                element_id=element_id,
                hotkey=hotkey,
                commit_block=commit_block,
            )
        ]
        keys.extend(
            f"https://validator.example/manako/{safe_element}/other-miner/"
            f"000000123/evaluation/{block:09d}-challenge.json"
            for block in range(9563400, 9566401, 300)
        )

    async def fake_fetch_json(_url):
        return keys

    async def fake_fetch_lines(_public_url, key):
        return lines_by_key[key]

    monkeypatch.setattr(commit_recovery, "fetch_json_from_url", fake_fetch_json)
    monkeypatch.setattr(commit_recovery, "fetch_shard_lines", fake_fetch_lines)
    monkeypatch.setattr(
        "scorevision.utils.cloudflare_helpers._verify_signature",
        lambda *_args: True,
    )

    recovered = await commit_recovery.recover_commitments_from_shards(
        recovery_blocks,
        {"validator": "https://validator.example/manako/index.json"},
        recovery_blocks=recovery_blocks,
    )

    assert set(recovered) == set(recovery_blocks)
    for request, recovery_block in recovery_blocks.items():
        proof = recovered[request]
        assert proof.proof_mode == "continuity"
        assert proof.commit_block < recovery_block < proof.shard_block
        assert sum(
            trigger > proof.shard_block for trigger in proof.observed_trigger_blocks
        ) == 11
        assert commit_recovery.is_timely_recovered_commitment(proof, recovery_block)


@pytest.mark.asyncio
async def test_continuity_shard_must_name_commit_before_recovery(monkeypatch):
    monkeypatch.setattr(commit_recovery, "CONTINUITY_ACTIVATION_BLOCK", 1000)
    later_commit_key = SHARD_KEY.replace("000000123", "000000500").replace(
        "000000999", "000001000"
    )

    async def fake_fetch_json(_url):
        return [later_commit_key]

    async def fake_fetch_lines(_public_url, key):
        assert key == later_commit_key
        return [_signed_line(commit_block=500)]

    monkeypatch.setattr(commit_recovery, "fetch_json_from_url", fake_fetch_json)
    monkeypatch.setattr(commit_recovery, "fetch_shard_lines", fake_fetch_lines)
    monkeypatch.setattr(
        "scorevision.utils.cloudflare_helpers._verify_signature",
        lambda *_args: True,
    )

    recovered = await commit_recovery.recover_commitments_from_shards(
        {(HOTKEY, ELEMENT_ID)},
        {"validator": "https://validator.example/manako/index.json"},
        recovery_blocks={(HOTKEY, ELEMENT_ID): 400},
    )

    assert recovered == {}


@pytest.mark.asyncio
async def test_recovery_counts_other_miners_once_per_indexed_trigger(monkeypatch):
    other_key_1 = SHARD_KEY.replace("5Fminer", "other-miner").replace(
        "000000999", "000001100"
    )
    other_key_2 = other_key_1.replace("other-miner", "third-miner")

    async def fake_fetch_json(_url):
        return [SHARD_KEY, other_key_1, other_key_2]

    async def fake_fetch_lines(_public_url, key):
        assert key == SHARD_KEY
        return [_signed_line()]

    monkeypatch.setattr(commit_recovery, "fetch_json_from_url", fake_fetch_json)
    monkeypatch.setattr(commit_recovery, "fetch_shard_lines", fake_fetch_lines)
    monkeypatch.setattr(
        "scorevision.utils.cloudflare_helpers._verify_signature",
        lambda _hotkey, _payload, _signature: True,
    )

    recovered = await commit_recovery.recover_commitments_from_shards(
        {(HOTKEY, ELEMENT_ID)},
        {"validator": "https://validator.example/manako/index.json"},
        recovery_blocks={(HOTKEY, ELEMENT_ID): 5000},
    )

    assert recovered[(HOTKEY, ELEMENT_ID)].observed_trigger_blocks == (999, 1100)


@pytest.mark.asyncio
async def test_cached_recovery_refreshes_observed_element_triggers(monkeypatch):
    other_key = SHARD_KEY.replace("5Fminer", "other-miner").replace(
        "000000999", "000001100"
    )
    index_keys = [SHARD_KEY]
    fetched = []

    async def fake_fetch_json(_url):
        return list(index_keys)

    async def fake_fetch_lines(_public_url, key):
        fetched.append(key)
        return [_signed_line()]

    monkeypatch.setattr(commit_recovery, "fetch_json_from_url", fake_fetch_json)
    monkeypatch.setattr(commit_recovery, "fetch_shard_lines", fake_fetch_lines)
    monkeypatch.setattr(
        "scorevision.utils.cloudflare_helpers._verify_signature",
        lambda _hotkey, _payload, _signature: True,
    )

    async def recover():
        return await commit_recovery.recover_commitments_from_shards(
            {(HOTKEY, ELEMENT_ID)},
            {"validator": "https://validator.example/manako/index.json"},
            recovery_blocks={(HOTKEY, ELEMENT_ID): 5000},
        )

    first = await recover()
    index_keys.append(other_key)
    second = await recover()

    assert first[(HOTKEY, ELEMENT_ID)].observed_trigger_blocks == (999,)
    assert second[(HOTKEY, ELEMENT_ID)].observed_trigger_blocks == (999, 1100)
    assert fetched == [SHARD_KEY]


@pytest.mark.asyncio
async def test_recovery_reuses_index_keys_already_loaded_by_weights(monkeypatch):
    index_url = "https://validator.example/manako/index.json"

    async def unexpected_fetch(_url):
        raise AssertionError("index should already be available")

    async def fake_fetch_lines(_public_url, key):
        assert key == SHARD_KEY
        return [_signed_line()]

    monkeypatch.setattr(commit_recovery, "fetch_json_from_url", unexpected_fetch)
    monkeypatch.setattr(commit_recovery, "fetch_shard_lines", fake_fetch_lines)
    monkeypatch.setattr(
        "scorevision.utils.cloudflare_helpers._verify_signature",
        lambda _hotkey, _payload, _signature: True,
    )

    recovered = await commit_recovery.recover_commitments_from_shards(
        {(HOTKEY, ELEMENT_ID)},
        {"validator": index_url},
        recovery_blocks={(HOTKEY, ELEMENT_ID): 5000},
        index_keys_by_url={index_url: [SHARD_KEY]},
    )

    assert recovered[(HOTKEY, ELEMENT_ID)].shard_block == 999


@pytest.mark.asyncio
async def test_recover_commitments_only_searches_active_index(monkeypatch):
    requested_urls = []

    async def fake_fetch_json(url):
        requested_urls.append(url)
        return [SHARD_KEY]

    async def fake_fetch_lines(_public_url, key):
        assert key == SHARD_KEY
        return [_signed_line()]

    monkeypatch.setattr(commit_recovery, "fetch_json_from_url", fake_fetch_json)
    monkeypatch.setattr(commit_recovery, "fetch_shard_lines", fake_fetch_lines)
    monkeypatch.setattr(
        "scorevision.utils.cloudflare_helpers._verify_signature",
        lambda _hotkey, _payload, _signature: True,
    )

    recovered = await commit_recovery.recover_commitments_from_shards(
        {(HOTKEY, ELEMENT_ID)},
        {"validator": "https://validator.example/manako/index.json"},
    )

    assert recovered[(HOTKEY, ELEMENT_ID)].commit_block == 123
    assert requested_urls == ["https://validator.example/manako/index.json"]


@pytest.mark.asyncio
async def test_recover_commitments_keeps_elements_independent(monkeypatch):
    second_element = "ActionDetect_v1@1.0"
    second_key = (
        "https://validator.example/manako/ActionDetect_v1@1.0/5Fminer/"
        "000000456/evaluation/000001000-challenge.json"
    )

    async def fake_fetch_json(_url):
        return [SHARD_KEY, second_key]

    async def fake_fetch_lines(_public_url, key):
        if key == SHARD_KEY:
            return [_signed_line()]
        if key == second_key:
            return [_signed_line(element_id=second_element, commit_block=456)]
        raise AssertionError(key)

    monkeypatch.setattr(commit_recovery, "fetch_json_from_url", fake_fetch_json)
    monkeypatch.setattr(commit_recovery, "fetch_shard_lines", fake_fetch_lines)
    monkeypatch.setattr(
        "scorevision.utils.cloudflare_helpers._verify_signature",
        lambda _hotkey, _payload, _signature: True,
    )

    recovered = await commit_recovery.recover_commitments_from_shards(
        {(HOTKEY, ELEMENT_ID), (HOTKEY, second_element)},
        {"validator": "https://validator.example/manako/index.json"},
    )

    assert recovered[(HOTKEY, ELEMENT_ID)].commit_block == 123
    assert recovered[(HOTKEY, second_element)].commit_block == 456
