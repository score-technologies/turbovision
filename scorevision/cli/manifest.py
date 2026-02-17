from os import environ
import asyncio
from json import loads
from pathlib import Path
from base64 import b64decode
from datetime import datetime, timezone
from hashlib import sha256

import click
from nacl.signing import SigningKey

from scorevision.utils.manifest import Manifest
from scorevision.utils.settings import get_settings
from scorevision.utils.manifest import yaml
from scorevision.utils.r2 import (
    r2_get_object,
    r2_put_json,
    r2_put_bytes,
    r2_delete_object,
)
from scorevision.utils.bittensor_helpers import get_subtensor

# ============================================================
# TEMPLATES
# ============================================================

TEMPLATES = {
    "default-football": {
        "version": "1.3",
        "tee": {"trusted_share_gamma": 0.2},
        "expiry_block": None,
        "elements": [],
    }
}


# ============================================================
# Manifest CLI
# ============================================================


@click.group(name="manifest")
def manifest_cli():
    """Manage ScoreVision manifests."""
    pass


@manifest_cli.command("create")
@click.option("--template", required=True, type=str, help="Template name.")
@click.option("--window-id", required=True, type=str, help="Window ID (YYYY-MM-DD).")
@click.option("--expiry-block", required=True, type=int, help="Expiry block.")
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(dir_okay=False, writable=True, path_type=Path, exists=False),
    help="output path for generated yaml",
)
@click.option(
    "--tee-key",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    required=False,
    help="TEE Ed25519 private key (optional, can also use TEE_KEY_HEX env)",
)
def create_manifest_cmd(
    template: str, window_id: str, expiry_block: int, output: Path, tee_key: Path | None
):
    """Create a new manifest from a template."""
    if template not in TEMPLATES:
        raise click.ClickException(f"Unknown template: {template}")

    tmpl = TEMPLATES[template]
    tee_data = tmpl.get("tee", {"trusted_share_gamma": 0.2})

    # Use raw hex key if provided via env variable
    key_hex = None
    if tee_key:
        key_hex = tee_key.read_text().strip()
    elif environ.get("TEE_KEY_HEX"):
        key_hex = environ["TEE_KEY_HEX"]

    if key_hex:
        signing_key = SigningKey(bytes.fromhex(key_hex))
        # Example derivation logic (replace with your actual derivation)
        tee_data["trusted_share_gamma"] = tee_data.get("trusted_share_gamma", 0.2)

    manifest_yaml = {
        "window_id": window_id,
        "version": tmpl["version"],
        "expiry_block": expiry_block,
        "tee": tee_data,
        "elements": tmpl.get("elements", []),
    }

    yaml.dump(manifest_yaml, output.open("w"))
    click.echo(f"✅ Manifest scaffold generated at {output}")


@manifest_cli.command("validate")
@click.argument("manifest_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--public-key", type=str, required=False, help="Ed25519 public key (hex or base64)."
)
def validate_manifest_cmd(manifest_path: Path, public_key: str | None):
    """Validate manifest schema and optionally its Ed25519 signature."""
    try:
        # Load the manifest
        manifest = Manifest.load_yaml(manifest_path)

        # Basic schema/hash validation (throws if YAML/JSON is invalid)
        _ = manifest.hash
        click.echo("✅ Schema is valid and canonical hash computed.")

        # If signature is present, verify it
        if manifest.signature:
            if not public_key:
                click.echo(
                    "ℹ Manifest is signed, but no public key was provided to verify it."
                )
            else:
                try:
                    # Determine whether key is a file, hex string, or base64
                    pub_bytes: bytes
                    if Path(public_key).exists():
                        pub_bytes = Path(public_key).read_bytes()
                    else:
                        try:
                            pub_bytes = bytes.fromhex(public_key)
                        except ValueError:
                            pub_bytes = b64decode(public_key)

                    from nacl.signing import VerifyKey

                    verify_key = VerifyKey(pub_bytes)

                    if manifest.verify(verify_key):
                        click.echo("🔐 Signature OK.")
                    else:
                        click.echo("❌ Signature invalid.", err=True)
                        raise click.ClickException(
                            "Manifest signature verification failed."
                        )

                except Exception as e:
                    click.echo(f"❌ Signature verification error: {e}", err=True)
                    raise click.ClickException("Failed to verify manifest signature.")

        else:
            click.echo("ℹ Manifest is unsigned.")

        click.echo("✅ Validation complete.")

    except Exception as e:
        click.echo(f"❌ Validation failed: {e}", err=True)
        raise click.ClickException("Manifest validation failed.")


@manifest_cli.command("publish")
@click.argument("manifest_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--signing-key-path",
    type=click.Path(path_type=Path),
    help="Ed25519 key as raw hex (optional, fallback to TEE_KEY_HEX). If omitted, manifest is uploaded unsigned.",
)
@click.option(
    "--block",
    type=int,
    default=None,
    help="Block number for key naming. If omitted, fetched from subtensor.",
)
def publish_manifest_cdn_cmd(
    manifest_path: Path,
    signing_key_path: Path | None,
    block: int | None,
):
    """
    Sign, upload (with retries), integrity-check, and update index.json.
    """

    settings = get_settings()
    bucket = settings.R2_BUCKET

    # ----------------------------------------------------------
    # Load signing key (from file or TEE_KEY_HEX env)
    # ----------------------------------------------------------
    key_hex = None
    if signing_key_path:
        key_hex = signing_key_path.read_text().strip()
    elif environ.get("TEE_KEY_HEX"):
        key_hex = environ["TEE_KEY_HEX"]

    # ----------------------------------------------------------
    # Load + (optional) sign manifest
    # ----------------------------------------------------------
    manifest = Manifest.load_yaml(manifest_path)
    if key_hex:
        signing_key = SigningKey(bytes.fromhex(key_hex))
        manifest.sign(signing_key)
        manifest.save_yaml(manifest_path)
    manifest_hash = manifest.hash

    # ----------------------------------------------------------
    # Resolve current on-chain block for key naming
    # ----------------------------------------------------------
    if block is None:
        async def _current_block() -> int:
            st = await get_subtensor()
            return int(await st.get_current_block())

        def _run(coro):
            try:
                return asyncio.run(coro)
            except RuntimeError:
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                return loop.run_until_complete(coro)

        try:
            block = _run(_current_block())
        except Exception as e:
            raise click.ClickException(f"Failed to fetch current block: {e}")

    # ----------------------------------------------------------
    # Upload to R2 (YAML, compatible with load_manifest_from_public_index)
    # ----------------------------------------------------------
    manifest_key = f"manifest/{block}-{manifest_hash}.yaml"
    existing, _ = r2_get_object(bucket, manifest_key)

    local_bytes = manifest_path.read_bytes()
    local_sha = sha256(local_bytes).hexdigest()

    if existing is None:
        click.echo(f"⬆ Uploading manifest {manifest_hash}...")
        r2_put_bytes(
            bucket,
            manifest_key,
            local_bytes,
            content_type="application/x-yaml",
        )
    else:
        click.echo("ℹ Manifest already exists in R2 (skipping upload).")

    remote_bytes, _ = r2_get_object(bucket, manifest_key)
    if remote_bytes is None:
        raise click.ClickException("Remote manifest missing after upload.")
    remote_sha = sha256(remote_bytes).hexdigest()
    if remote_sha != local_sha:
        raise click.ClickException(
            f"Integrity mismatch: local={local_sha} remote={remote_sha}"
        )
    click.echo("🧩 Integrity OK.")

    # ----------------------------------------------------------
    # Update manifest/index.json (list of keys)
    # ----------------------------------------------------------
    index_key = "manifest/index.json"
    index_bytes, _ = r2_get_object(bucket, index_key)
    index = loads(index_bytes.decode("utf-8")) if index_bytes else []
    if not isinstance(index, list):
        raise click.ClickException("manifest/index.json must be a JSON array.")

    if manifest_key not in index:
        index.append(manifest_key)
    index = sorted(set(index))

    try:
        click.echo("📝 Updating manifest/index.json...")
        r2_put_json(bucket, index_key, index)
    except Exception as e:
        if existing is None:
            click.echo("⚠ Rolling back manifest upload...")
            r2_delete_object(bucket, manifest_key)
        raise click.ClickException(f"Failed to update manifest/index.json: {e}")

    click.echo("✅ Publish complete.")


@manifest_cli.command("list")
def list_manifests_cmd():
    """List published manifests from CDN/R2 index."""
    settings = get_settings()
    bucket = settings.R2_BUCKET
    index_key = "manifest/index.json"

    index_bytes, _ = r2_get_object(bucket, index_key)
    if not index_bytes:
        click.echo("ℹ No manifests published yet.")
        return

    index = loads(index_bytes.decode("utf-8"))
    if not isinstance(index, list) or not index:
        click.echo("ℹ No manifests found in index.")
        return

    click.echo("📄 Published manifests:")
    def block_from_key(k: str) -> int | None:
        try:
            name = Path(k).name
            return int(name.split("-", 1)[0])
        except Exception:
            return None

    keyed = [(block_from_key(k), k) for k in index]
    keyed.sort(key=lambda x: (x[0] is None, x[0] or 0, x[1]))
    for blk, key in keyed:
        if blk is not None:
            click.echo(f"  - Block: {blk}  Key: {key}")
        else:
            click.echo(f"  - Key: {key}")


@manifest_cli.command("current")
@click.option("--block", type=int, default=None)
def current_manifest_cmd(block: int):
    """Show active manifest for a given block."""
    settings = get_settings()
    bucket = settings.R2_BUCKET
    index_bytes, _ = r2_get_object(bucket, "manifest/index.json")
    if not index_bytes:
        click.echo("❌ No manifests published.")
        return

    index = loads(index_bytes.decode("utf-8"))
    if not isinstance(index, list) or not index:
        click.echo("❌ No manifests in index.")
        return

    def block_from_key(k: str) -> int | None:
        try:
            name = Path(k).name
            return int(name.split("-", 1)[0])
        except Exception:
            return None

    pairs = [(block_from_key(k), k) for k in index]
    pairs = [(b, k) for (b, k) in pairs if b is not None]
    if not pairs:
        click.echo("❌ No block-prefixed manifests in index.")
        return

    pairs.sort(key=lambda x: x[0])
    if block is None:
        blk, key = pairs[-1]
        click.echo(f"ℹ Latest manifest: Block={blk}, Key={key}")
        return

    eligible = [p for p in pairs if p[0] <= block]
    if not eligible:
        click.echo(f"ℹ No active manifest found for block {block}")
        return
    blk, key = eligible[-1]
    click.echo(f"ℹ Active manifest for block {block}: Block={blk}, Key={key}")
