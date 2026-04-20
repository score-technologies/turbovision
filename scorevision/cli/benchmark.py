import asyncio
import os
from json import dumps, loads
from pathlib import Path

import bittensor as bt
import click

from scorevision.miner.open_source.chute_template.schemas import TVPredictInput
from scorevision.utils.bittensor_helpers import get_subtensor
from scorevision.utils.cloudflare_helpers import get_s3_client
from scorevision.utils.manifest import get_current_manifest, load_manifest_from_public_index
from scorevision.utils.miner_registry import get_miners_from_registry
from scorevision.utils.predict import predict_sv
from scorevision.utils.settings import get_settings
from scorevision.validator.central.private_track.challenges import Challenge
from scorevision.validator.central.private_track.miners import send_challenge
from scorevision.validator.central.private_track.registry import get_registered_miners


def _winners_index_key() -> str:
    prefix = (os.getenv("SCOREVISION_WINNERS_PREFIX") or "winners").strip().strip("/")
    return f"manako/{prefix}/index.json"


async def _load_manifest():
    settings = get_settings()
    if getattr(settings, "URL_MANIFEST", None):
        cache_dir = getattr(settings, "SCOREVISION_CACHE_DIR", None)
        try:
            return await load_manifest_from_public_index(
                settings.URL_MANIFEST,
                cache_dir=cache_dir,
            )
        except Exception as e:
            click.echo(f"Warning: unable to load manifest from URL_MANIFEST: {e}")
    return get_current_manifest()


def _prompt_element_id(manifest, element_id: str | None) -> str:
    if element_id:
        return element_id

    element_ids = [str(getattr(element, "id", "")).strip() for element in manifest.elements]
    element_ids = list(dict.fromkeys(eid for eid in element_ids if eid))
    if not element_ids:
        raise click.ClickException("No element IDs found in the current manifest.")

    click.echo("Available element IDs from manifest:")
    for idx, eid in enumerate(element_ids, start=1):
        click.echo(f"  {idx}. {eid}")

    choice = click.prompt(
        "Select element ID (number)",
        type=click.IntRange(1, len(element_ids)),
    )
    return element_ids[choice - 1]


def _read_benchmark_urls(benchmark_file: Path) -> list[str]:
    if not benchmark_file.exists():
        raise click.ClickException(f"Benchmark file not found: {benchmark_file}")
    rows = benchmark_file.read_text(encoding="utf-8").splitlines()
    urls = []
    for row in rows:
        val = row.strip()
        if not val or val.startswith("#"):
            continue
        urls.append(val)
    if not urls:
        raise click.ClickException("Benchmark file is empty after filtering blank/comment lines.")
    return urls


async def _fetch_latest_winners_snapshot() -> tuple[str, dict]:
    settings = get_settings()
    index_key = _winners_index_key()

    async with get_s3_client() as c:
        try:
            index_obj = await c.get_object(Bucket=settings.SCOREVISION_BUCKET, Key=index_key)
            keys = loads((await index_obj["Body"].read()).decode())
        except c.exceptions.NoSuchKey as e:
            raise click.ClickException(f"Winners index not found in R2: {index_key}") from e

        if not keys:
            raise click.ClickException("Winners index exists but contains no snapshots.")

        latest_key = sorted(keys)[-1]
        snap_obj = await c.get_object(Bucket=settings.SCOREVISION_BUCKET, Key=latest_key)
        snapshot = loads((await snap_obj["Body"].read()).decode())
        return latest_key, snapshot


def _get_element_track(manifest, element_id: str) -> str:
    element = manifest.get_element(id=element_id)
    if element is None:
        raise click.ClickException(f"Element '{element_id}' not found in manifest.")
    track = str(getattr(element, "track", "") or "").strip()
    return "private" if track == "private" else "public"


async def _resolve_public_target(
    element_id: str,
    winner_entry: dict,
) -> tuple[str, str | None, str]:
    settings = get_settings()
    winner_hotkey = str(winner_entry.get("winner_hotkey") or "").strip()
    slug = str(winner_entry.get("slug") or "").strip()
    chute_id = str(winner_entry.get("chute_id") or "").strip() or None

    if slug:
        return slug, chute_id, winner_hotkey

    miners, _skipped = await get_miners_from_registry(
        settings.SCOREVISION_NETUID,
        element_id=element_id,
        blacklisted_hotkeys=set(),
    )
    if not miners:
        raise click.ClickException("No eligible public miners found in registry.")

    if winner_hotkey:
        for miner in miners.values():
            if miner.hotkey == winner_hotkey and miner.slug:
                return miner.slug, miner.chute_id, miner.hotkey

    raise click.ClickException(
        "Latest winner for this element has no slug/chute metadata and could not be resolved from registry."
    )


async def _resolve_private_target(
    element_id: str,
    winner_entry: dict,
):
    settings = get_settings()
    winner_hotkey = str(winner_entry.get("winner_hotkey") or "").strip()
    if not winner_hotkey:
        raise click.ClickException("Latest private winner has no winner_hotkey.")

    subtensor = await get_subtensor()
    metagraph = await subtensor.metagraph(
        settings.SCOREVISION_NETUID,
        mechid=settings.SCOREVISION_MECHID,
    )
    miners = await get_registered_miners(
        subtensor=subtensor,
        metagraph=metagraph,
        blacklist=set(),
        element_id=element_id,
    )
    for miner in miners:
        if miner.hotkey == winner_hotkey:
            return miner
    raise click.ClickException(
        f"Private winner hotkey not found among active registered private miners: {winner_hotkey}"
    )


async def run_top_performer_benchmark(
    benchmark_file: Path,
    element_id: str | None,
    timeout: float,
    output_file: Path | None,
) -> None:
    settings = get_settings()
    manifest = await _load_manifest()
    selected_element_id = _prompt_element_id(manifest, element_id)
    track = _get_element_track(manifest, selected_element_id)
    urls = _read_benchmark_urls(benchmark_file)

    latest_key, snapshot = await _fetch_latest_winners_snapshot()
    winners = snapshot.get("winners") or {}
    winner_entry = winners.get(selected_element_id)
    if not isinstance(winner_entry, dict):
        raise click.ClickException(
            f"Element '{selected_element_id}' not found in latest winners snapshot: {latest_key}"
        )

    click.echo(f"Using winners snapshot: {latest_key}")
    click.echo(f"Element: {selected_element_id} (track={track})")
    click.echo(f"Benchmark URLs: {len(urls)}")

    results: list[dict] = []
    success_count = 0

    if track == "private":
        target_miner = await _resolve_private_target(selected_element_id, winner_entry)
        wallet = bt.wallet(
            name=settings.BITTENSOR_WALLET_COLD,
            hotkey=settings.BITTENSOR_WALLET_HOT,
        )
        click.echo(
            f"Target private winner: hotkey={target_miner.hotkey} endpoint={target_miner.ip}:{target_miner.port}"
        )

        for idx, url in enumerate(urls, start=1):
            click.echo(f"[{idx}/{len(urls)}] -> {url}")
            challenge = Challenge(
                challenge_id=f"benchmark-{idx}",
                video_url=url,
                ground_truth=[],
            )
            attempt = await send_challenge(
                miner=target_miner,
                challenge=challenge,
                hotkey=wallet.hotkey,
                timeout=timeout,
            )
            ok = attempt.response is not None and not attempt.timed_out
            if ok:
                success_count += 1
            results.append(
                {
                    "index": idx,
                    "url": url,
                    "success": ok,
                    "timed_out": bool(attempt.timed_out),
                    "elapsed_s": round(float(attempt.elapsed_s), 6),
                    "response": attempt.response.model_dump() if attempt.response else None,
                }
            )
    else:
        slug, chute_id, winner_hotkey = await _resolve_public_target(selected_element_id, winner_entry)
        click.echo(
            f"Target public winner: hotkey={winner_hotkey or '<unknown>'} slug={slug} chute_id={chute_id or '<none>'}"
        )

        for idx, url in enumerate(urls, start=1):
            click.echo(f"[{idx}/{len(urls)}] -> {url}")
            payload = TVPredictInput(
                url=url,
                meta={
                    "benchmark": True,
                    "element_id": selected_element_id,
                    "benchmark_index": idx,
                },
            )
            pred = await predict_sv(payload=payload, slug=slug, chute_id=chute_id)
            ok = bool(pred.success)
            if ok:
                success_count += 1
            results.append(
                {
                    "index": idx,
                    "url": url,
                    "success": ok,
                    "latency_s": round(float(pred.latency_seconds), 6),
                    "error": pred.error,
                    "model": pred.model,
                    "predictions": pred.predictions if ok else None,
                }
            )

    summary = {
        "element_id": selected_element_id,
        "track": track,
        "latest_winner_snapshot_key": latest_key,
        "winner": winner_entry,
        "total": len(urls),
        "success": success_count,
        "failed": len(urls) - success_count,
        "results": results,
    }

    click.echo(f"Done: {success_count}/{len(urls)} succeeded")
    if output_file:
        output_file.write_text(dumps(summary, indent=2), encoding="utf-8")
        click.echo(f"Results written to: {output_file}")


@click.group(name="benchmark")
def benchmark_cli():
    pass


@benchmark_cli.command("run-top-performer")
@click.option(
    "--benchmark-file",
    default="benchmark_data",
    type=click.Path(path_type=Path, dir_okay=False),
    show_default=True,
    help="Local file containing one media URL per line.",
)
@click.option(
    "--element-id",
    default=None,
    help="Element ID. If omitted, the CLI prompts from current manifest.",
)
@click.option(
    "--timeout",
    default=30.0,
    type=float,
    show_default=True,
    help="Per-request timeout in seconds (private track only).",
)
@click.option(
    "--output-file",
    default="benchmark_results.json",
    type=click.Path(path_type=Path, dir_okay=False),
    show_default=True,
    help="Where to write benchmark responses.",
)
def run_top_performer_cmd(
    benchmark_file: Path,
    element_id: str | None,
    timeout: float,
    output_file: Path | None,
):
    asyncio.run(
        run_top_performer_benchmark(
            benchmark_file=benchmark_file,
            element_id=element_id,
            timeout=timeout,
            output_file=output_file,
        )
    )
