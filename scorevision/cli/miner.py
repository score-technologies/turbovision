from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from logging import getLogger
from pathlib import Path
from typing import Optional

import click
import bittensor as bt

from scorevision.utils.settings import get_settings
from scorevision.utils.element_catalog import (
    list_elements,
    summarize_window,
)
from scorevision.utils.commitments import (
    list_local_commitments,
    get_commitments_for_hotkey_from_chain,
)

logger = getLogger(__name__)


@click.group(name="miner")
def miner():
    """Miner utilities: elements, manifests, commitments."""
    pass


# --------------------------------------------------------------------------- #
# sv miner elements
# --------------------------------------------------------------------------- #


@miner.command("elements")
@click.option(
    "--window",
    "window_scope",
    type=click.Choice(["current", "upcoming"]),
    default="current",
    show_default=True,
    help="Which evaluation window to inspect.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Output raw JSON instead of a table.",
)
def elements_cmd(window_scope: str, as_json: bool) -> None:
    """
    Show manifest-defined elements for the selected window
    (service rates, theta, beta, telemetry, clip counts).
    """
    try:
        elems = asyncio.run(list_elements(window_scope=window_scope))
    except Exception as e:
        click.echo(f"Error loading elements for window '{window_scope}': {e}")
        return

    if as_json:
        click.echo(json.dumps([asdict(e) for e in elems], indent=2, sort_keys=True))
        return

    if not elems:
        click.echo(f"No elements defined for window '{window_scope}'.")
        return

    # Pretty table (no external deps)
    headers = ["element_id", "window_id", "service_rate", "theta", "beta", "clip_count"]
    rows = []
    for e in elems:
        rows.append(
            [
                e.element_id,
                e.window_id or "",
                f"{e.service_rate:.6f}" if e.service_rate is not None else "",
                f"{e.theta:.6f}" if e.theta is not None else "",
                f"{e.beta:.6f}" if e.beta is not None else "",
                str(e.clip_count) if e.clip_count is not None else "",
            ]
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _fmt_row(cells):
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells))

    click.echo(_fmt_row(headers))
    click.echo("  ".join("-" * w for w in widths))
    for row in rows:
        click.echo(_fmt_row(row))


# --------------------------------------------------------------------------- #
# sv miner manifest --hash ...
# --------------------------------------------------------------------------- #


@miner.command("manifest")
@click.option(
    "--hash",
    "manifest_hash",
    required=True,
    help="Manifest hash to inspect (used for local manifest lookup).",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Print raw JSON content.",
)
def manifest_cmd(manifest_hash: str, as_json: bool) -> None:
    """
    Inspect a manifest stored locally under SCOREVISION_LOCAL_ROOT/manifests/<hash>.json.

    This is intentionally simple: it loads the JSON, prints some summary fields,
    and lets the user optionally see the raw JSON.
    """
    settings = get_settings()
    manifest_dir = settings.SCOREVISION_LOCAL_ROOT / "manifests"
    path = manifest_dir / f"{manifest_hash}.json"

    if not path.exists():
        click.echo(
            f"Manifest file not found at '{path}'. "
            "Make sure the validator/runner has downloaded it first."
        )
        return

    try:
        obj = json.loads(path.read_text())
    except Exception as e:
        click.echo(f"Failed to load manifest JSON from '{path}': {e}")
        return

    if as_json:
        click.echo(json.dumps(obj, indent=2, sort_keys=True))
        return

    window_id = (
        obj.get("window_id")
        or obj.get("id")
        or (obj.get("meta") or {}).get("window_id")
    )
    elements = obj.get("elements") or (obj.get("payload") or {}).get("elements") or []
    n_elements = len(elements)

    click.echo(f"Manifest hash : {manifest_hash}")
    click.echo(f"Window ID     : {window_id or '(unknown)'}")
    click.echo(f"Elements      : {n_elements}")
    click.echo(f"Path          : {path}")
    click.echo("")
    click.echo("Use --json to see the full manifest content.")


# --------------------------------------------------------------------------- #
# sv miner commitments list
# --------------------------------------------------------------------------- #


@miner.group("commitments")
def commitments():
    """Inspect on-chain and local miner commitments."""
    pass


@commitments.command("list")
@click.option(
    "--window",
    "window_opt",
    default=None,
    help="Filter by window: 'current', 'upcoming', or explicit window_id (e.g. block-12300).",
)
@click.option(
    "--source",
    type=click.Choice(["local", "chain", "both"]),
    default="both",
    show_default=True,
    help="Where to read commitments from.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Output raw JSON instead of a human-friendly table.",
)
def commitments_list(window_opt: Optional[str], source: str, as_json: bool) -> None:
    """
    List commitments recorded locally and/or on-chain for this miner hotkey.
    """
    settings = get_settings()
    w = bt.wallet(
        name=settings.BITTENSOR_WALLET_COLD,
        hotkey=settings.BITTENSOR_WALLET_HOT,
    )
    hotkey_ss58 = w.hotkey.ss58_address

    resolved_window_id: Optional[str] = None
    resolved_scope: Optional[str] = None

    if window_opt in (None, "", "all"):
        resolved_window_id = None
    elif window_opt in ("current", "upcoming"):
        try:
            summary = asyncio.run(summarize_window(window_opt))
            resolved_window_id = summary.get("window_id")
            resolved_scope = window_opt
        except Exception as e:
            click.echo(f"Failed to resolve window '{window_opt}': {e}")
            return
    else:
        resolved_window_id = window_opt

    # --- Fetch local / chain commitments ----------------------------------- #
    local_proofs = []
    chain_proofs = []

    if source in ("local", "both"):
        local_proofs = list_local_commitments(hotkey_ss58)
        if resolved_window_id:
            local_proofs = [
                p for p in local_proofs if p.window_id == resolved_window_id
            ]

    if source in ("chain", "both"):
        try:
            chain_proofs = asyncio.run(
                get_commitments_for_hotkey_from_chain(
                    hotkey_ss58,
                    window_id=resolved_window_id,
                )
            )
        except Exception as e:
            click.echo(f"Failed to fetch commitments from chain: {e}")
            return

    # Convert to dicts for JSON / printing
    local_dicts = [asdict(p) for p in local_proofs]
    chain_dicts = [asdict(p) for p in chain_proofs]

    if as_json:
        click.echo(
            json.dumps(
                {
                    "hotkey": hotkey_ss58,
                    "window_scope": resolved_scope or window_opt or "all",
                    "local": local_dicts,
                    "chain": chain_dicts,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if not local_dicts and not chain_dicts:
        click.echo(
            f"No commitments found for hotkey {hotkey_ss58} "
            f"(window={window_opt or 'all'}, source={source})."
        )
        return

    def _print_table(title: str, rows: list[dict]):
        if not rows:
            click.echo(f"{title}: none.")
            return

        headers = [
            "window_id",
            "element_ids",
            "revision",
            "chute_slug",
            "chute_id",
            "service_cap",
            "block",
            "ts",
            "action",
        ]
        table = []
        for r in rows:
            payload = r.get("payload") or {}
            element_ids = r.get("element_ids") or []
            if not isinstance(element_ids, list):
                element_ids = [element_ids]

            table.append(
                [
                    str(r.get("window_id", "")),
                    ", ".join(str(e) for e in element_ids),
                    str(r.get("revision", "")),
                    str(r.get("chute_slug", "")),
                    str(r.get("chute_id", "")),
                    str(r.get("service_cap", "")),
                    str(r.get("block", "")),
                    (
                        f"{r.get('ts', ''):.0f}"
                        if isinstance(r.get("ts", None), (int, float))
                        else str(r.get("ts", ""))
                    ),
                    str(payload.get("action", "")),
                ]
            )

        widths = [len(h) for h in headers]
        for row in table:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))

        def _fmt_row(cells):
            return " ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells))

        click.echo(_fmt_row(headers))
        click.echo(" ".join("-" * w for w in widths))
        for row in table:
            click.echo(_fmt_row(row))
        click.echo("")

    click.echo(f"Hotkey: {hotkey_ss58}")
    click.echo(f"Window filter: {window_opt or 'all'}")
    click.echo(f"Source: {source}")
    click.echo("")

    if source in ("local", "both"):
        _print_table("Local commitment proofs", local_dicts)
    if source in ("chain", "both"):
        _print_table("On-chain commitments", chain_dicts)
