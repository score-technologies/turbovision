from click.testing import CliRunner

from scorevision.cli.index_maintenance import index_cli, _split_index_entries


def test_split_index_entries_moves_only_old_blocks():
    entries = [
        "manako/element/hk/evaluation/000010000-a.json",
        "manako/element/hk/evaluation/000020000-b.json",
        "manako/element/hk/evaluation/000030000-c.json",
        "manako/element/hk/evaluation/not-a-block.json",
    ]
    recent, past, unknown = _split_index_entries(entries, cutoff_block=25000)

    assert unknown == 1
    assert past == [
        "manako/element/hk/evaluation/000010000-a.json",
        "manako/element/hk/evaluation/000020000-b.json",
    ]
    assert recent == [
        "manako/element/hk/evaluation/000030000-c.json",
        "manako/element/hk/evaluation/not-a-block.json",
    ]


def test_compact_index_dry_run(monkeypatch):
    runner = CliRunner()
    reads = {
        "manako/index.json": [
            "manako/element/hk/evaluation/000010000-a.json",
            "manako/element/hk/evaluation/000020000-b.json",
            "manako/element/hk/evaluation/000090000-c.json",
        ],
        "manako/indexpast.json": [
            "manako/element/hk/evaluation/000000100-old.json",
        ],
    }
    writes: list[tuple[str, list[str]]] = []

    async def _fake_read(index_key: str) -> list[str]:
        return list(reads.get(index_key, []))

    async def _fake_write(index_key: str, entries: list[str]) -> None:
        writes.append((index_key, entries))

    monkeypatch.setattr("scorevision.cli.index_maintenance._read_index", _fake_read)
    monkeypatch.setattr("scorevision.cli.index_maintenance._write_index", _fake_write)
    monkeypatch.setattr("scorevision.cli.index_maintenance.days_to_blocks", lambda _d: 8_000)

    result = runner.invoke(index_cli, ["compact", "--dry-run", "--days", "8"])
    assert result.exit_code == 0
    assert "Index compact plan" in result.output
    assert "Dry-run enabled" in result.output
    assert writes == []


def test_compact_index_writes_recent_and_past(monkeypatch):
    runner = CliRunner()
    reads = {
        "manako/index.json": [
            "manako/element/hk/evaluation/000010000-a.json",
            "manako/element/hk/evaluation/000090000-c.json",
        ],
        "manako/indexpast.json": [
            "manako/element/hk/evaluation/000000100-old.json",
        ],
    }
    writes: list[tuple[str, list[str]]] = []

    async def _fake_read(index_key: str) -> list[str]:
        return list(reads.get(index_key, []))

    async def _fake_write(index_key: str, entries: list[str]) -> None:
        writes.append((index_key, entries))

    monkeypatch.setattr("scorevision.cli.index_maintenance._read_index", _fake_read)
    monkeypatch.setattr("scorevision.cli.index_maintenance._write_index", _fake_write)
    monkeypatch.setattr("scorevision.cli.index_maintenance.days_to_blocks", lambda _d: 8_000)

    result = runner.invoke(index_cli, ["compact", "--days", "8"])
    assert result.exit_code == 0
    assert "OK: Updated manako/index.json and manako/indexpast.json." in result.output
    assert writes == [
        (
            "manako/index.json",
            ["manako/element/hk/evaluation/000090000-c.json"],
        ),
        (
            "manako/indexpast.json",
            [
                "manako/element/hk/evaluation/000000100-old.json",
                "manako/element/hk/evaluation/000010000-a.json",
            ],
        ),
    ]


def test_compact_index_lane_both(monkeypatch):
    runner = CliRunner()
    reads = {
        "manako/index.json": [
            "manako/element/hk/evaluation/000010000-a.json",
            "manako/element/hk/evaluation/000090000-c.json",
        ],
        "manako/indexpast.json": [],
        "manako/indexprivate.json": [
            "manako/element/hk/evaluation/000020000-p1.json",
            "manako/element/hk/evaluation/000090100-p2.json",
        ],
        "manako/indexprivatepast.json": [],
    }
    writes: list[tuple[str, list[str]]] = []

    async def _fake_read(index_key: str) -> list[str]:
        return list(reads.get(index_key, []))

    async def _fake_write(index_key: str, entries: list[str]) -> None:
        writes.append((index_key, entries))

    monkeypatch.setattr("scorevision.cli.index_maintenance._read_index", _fake_read)
    monkeypatch.setattr("scorevision.cli.index_maintenance._write_index", _fake_write)
    monkeypatch.setattr("scorevision.cli.index_maintenance.days_to_blocks", lambda _d: 8_000)

    result = runner.invoke(index_cli, ["compact", "--days", "8", "--lane", "both"])
    assert result.exit_code == 0
    assert "OK: Updated manako/index.json and manako/indexpast.json." in result.output
    assert "OK: Updated manako/indexprivate.json and manako/indexprivatepast.json." in result.output
    assert writes == [
        (
            "manako/index.json",
            ["manako/element/hk/evaluation/000090000-c.json"],
        ),
        (
            "manako/indexpast.json",
            ["manako/element/hk/evaluation/000010000-a.json"],
        ),
        (
            "manako/indexprivate.json",
            ["manako/element/hk/evaluation/000090100-p2.json"],
        ),
        (
            "manako/indexprivatepast.json",
            ["manako/element/hk/evaluation/000020000-p1.json"],
        ),
    ]
