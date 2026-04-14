import asyncio
from json import dumps, loads
from logging import getLogger

import click

from scorevision.utils.cloudflare_helpers import get_s3_client
from scorevision.utils.r2_public import extract_block_from_key
from scorevision.utils.settings import get_settings
from scorevision.validator.scoring import days_to_blocks

logger = getLogger(__name__)


async def _read_index(index_key: str) -> list[str]:
    settings = get_settings()
    async with get_s3_client() as client:
        try:
            response = await client.get_object(Bucket=settings.SCOREVISION_BUCKET, Key=index_key)
            body = await response["Body"].read()
        except client.exceptions.NoSuchKey:
            return []

    data = loads(body)
    if not isinstance(data, list):
        raise click.ClickException(f"{index_key} must be a JSON array.")
    return [item for item in data if isinstance(item, str)]


async def _write_index(index_key: str, entries: list[str]) -> None:
    settings = get_settings()
    payload = dumps(entries, separators=(",", ":")).encode()
    async with get_s3_client() as client:
        await client.put_object(
            Bucket=settings.SCOREVISION_BUCKET,
            Key=index_key,
            Body=payload,
            ContentType="application/json",
        )


def _split_index_entries(entries: list[str], cutoff_block: int) -> tuple[list[str], list[str], int]:
    recent: list[str] = []
    past: list[str] = []
    unknown_block_count = 0

    for entry in entries:
        block = extract_block_from_key(entry)
        if block is None:
            unknown_block_count += 1
            recent.append(entry)
            continue
        if block < cutoff_block:
            past.append(entry)
        else:
            recent.append(entry)

    return sorted(set(recent)), sorted(set(past)), unknown_block_count


async def _compact_one_index_pair(
    *,
    days: float,
    tail_blocks: int,
    index_key: str,
    past_index_key: str,
    dry_run: bool,
) -> None:
    current_entries = await _read_index(index_key)
    if not current_entries:
        click.echo(f"INFO: {index_key} is empty; nothing to compact.")
        return

    blocks = [extract_block_from_key(item) for item in current_entries]
    blocks = [int(b) for b in blocks if b is not None]
    if not blocks:
        click.echo(f"INFO: No block-prefixed entries in {index_key}; nothing to compact.")
        return

    max_block = max(blocks)
    cutoff_block = max_block - tail_blocks

    recent_entries, past_candidates, unknown_count = _split_index_entries(current_entries, cutoff_block)
    moved_count = len(past_candidates)
    if moved_count == 0:
        click.echo(
            f"INFO: No entries older than {days:g} days "
            f"(tail={tail_blocks} blocks, cutoff={cutoff_block}) for {index_key}."
        )
        return

    existing_past_entries = await _read_index(past_index_key)
    merged_past_entries = sorted(set(existing_past_entries) | set(past_candidates))

    click.echo(
        f"Index compact plan [{index_key}]: move={moved_count}, keep={len(recent_entries)}, "
        f"past_total={len(merged_past_entries)}, unknown_block_entries={unknown_count}, "
        f"max_block={max_block}, cutoff_block={cutoff_block}"
    )

    if dry_run:
        click.echo(f"Dry-run enabled for {index_key}: no changes written.")
        return

    await _write_index(index_key, recent_entries)
    await _write_index(past_index_key, merged_past_entries)
    click.echo(f"OK: Updated {index_key} and {past_index_key}.")


@click.group("index")
def index_cli():
    """Index maintenance commands."""


@index_cli.command("compact")
@click.option(
    "--days",
    type=float,
    default=8.0,
    show_default=True,
    help="Move entries older than this many days to the past index.",
)
@click.option(
    "--index-key",
    default="manako/index.json",
    show_default=True,
    help="Source active index key in R2.",
)
@click.option(
    "--past-index-key",
    default="manako/indexpast.json",
    show_default=True,
    help="Destination archive index key in R2.",
)
@click.option(
    "--lane",
    type=click.Choice(["public", "private", "both"], case_sensitive=False),
    default="public",
    show_default=True,
    help="Which index lane(s) to compact.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would change without writing to R2.",
)
def compact_index_cmd(days: float, index_key: str, past_index_key: str, lane: str, dry_run: bool):
    """Compact active index by moving old evaluation shards into indexpast.json."""
    if days <= 0:
        raise click.ClickException("--days must be > 0")

    tail_blocks = days_to_blocks(days)
    if tail_blocks is None:
        raise click.ClickException("Unable to compute tail blocks from --days.")

    async def _run():
        lane_norm = lane.lower()
        default_index_key = "manako/index.json"
        default_past_index_key = "manako/indexpast.json"

        if lane_norm == "both":
            if index_key != default_index_key or past_index_key != default_past_index_key:
                raise click.ClickException(
                    "--index-key/--past-index-key cannot be customized with --lane both."
                )
            pairs = [
                ("manako/index.json", "manako/indexpast.json"),
                ("manako/indexprivate.json", "manako/indexprivatepast.json"),
            ]
        elif lane_norm == "private":
            if index_key == default_index_key and past_index_key == default_past_index_key:
                pairs = [("manako/indexprivate.json", "manako/indexprivatepast.json")]
            else:
                pairs = [(index_key, past_index_key)]
        else:
            pairs = [(index_key, past_index_key)]

        for active_key, archive_key in pairs:
            await _compact_one_index_pair(
                days=days,
                tail_blocks=tail_blocks,
                index_key=active_key,
                past_index_key=archive_key,
                dry_run=dry_run,
            )

    logger.info(
        "Compacting index (days=%s, lane=%s, index_key=%s, past_index_key=%s, dry_run=%s)",
        days,
        lane,
        index_key,
        past_index_key,
        dry_run,
    )
    asyncio.run(_run())
