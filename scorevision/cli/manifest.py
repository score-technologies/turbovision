import os
import json
import click
from pathlib import Path
from base64 import b64decode

from ruamel.yaml import YAML
from nacl.signing import SigningKey

from scorevision.utils.manifest import (
    Manifest,
    Element,
    Metrics,
    Pillars,
    Preproc,
    Tee,
    Salt,
    Clip,
)
from scorevision.utils.settings import get_settings


yaml = YAML()
yaml.default_flow_style = False


# ============================================================
# Utility: Load YAML manifest → Manifest object
# ============================================================


def load_manifest_from_yaml(path: Path) -> Manifest:
    data = yaml.load(path.read_text())

    # Normalize element structure (YAML → Python dataclasses)
    elements = []
    for e in data["elements"]:
        pillars = Pillars(**e["metrics"]["pillars"])
        metrics = Metrics(pillars=pillars)

        clips = [Clip(hash=c["hash"], weight=c["weight"]) for c in e["clips"]]

        preproc = Preproc(
            fps=e["preproc"]["fps"],
            resize_long=e["preproc"]["resize_long"],
            norm=e["preproc"]["norm"],
        )

        element = Element(
            id=e["id"],
            clips=clips,
            metrics=metrics,
            preproc=preproc,
            latency_p95_ms=e["latency_p95_ms"],
            service_rate_fps=e["service_rate_fps"],
            pgt_recipe_hash=e["pgt_recipe_hash"],
            baseline_theta=e["baseline_theta"],
            delta_floor=e["delta_floor"],
            beta=e["beta"],
            salt=Salt(
                offsets=e.get("salt", {}).get("offsets", []),
                strides=e.get("salt", {}).get("strides", []),
            ),
        )
        elements.append(element)

    tee = Tee(trusted_share_gamma=data["tee"]["trusted_share_gamma"])

    return Manifest(
        window_id=data["window_id"],
        version=data["version"],
        expiry_block=data["expiry_block"],
        elements=elements,
        tee=tee,
        signature=data.get("signature"),
    )


# ============================================================
# Utility: Dump Manifest → YAML
# ============================================================


def save_manifest_yaml(manifest: Manifest, path: Path):
    raw = json.loads(manifest.to_canonical_json())
    if manifest.signature:
        raw["signature"] = manifest.signature
    yaml.dump(raw, path.open("w"))


# ============================================================
# Manifest CLI
# ============================================================


@click.group(name="manifest")
def manifest_cli():
    """Manage ScoreVision manifests."""
    pass


# ============================================================
# CREATE
# ============================================================

TEMPLATES = {
    "default-football": {
        "version": "1.3",
        "tee": {"trusted_share_gamma": 0.2},
        "expiry_block": None,
        "elements": [],
    }
}


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
    elif os.environ.get("TEE_KEY_HEX"):
        key_hex = os.environ["TEE_KEY_HEX"]

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


# ============================================================
# VALIDATE
# ============================================================


@manifest_cli.command("validate")
@click.argument("manifest_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--public-key", type=str, required=False, help="Ed25519 public key (hex or base64)."
)
def validate_manifest_cmd(manifest_path: Path, public_key: str | None):
    """Validate schema, pillars, baseline, and optionally signature."""
    try:
        manifest = load_manifest_from_yaml(manifest_path)
        _ = manifest.hash  # ensure canonical JSON is valid

        if manifest.signature and public_key:
            # Determine whether key is a file, hex string, or base64
            pub_bytes = None
            if Path(public_key).exists():
                pub_bytes = Path(public_key).read_bytes()
            else:
                try:
                    pub_bytes = bytes.fromhex(public_key)
                except ValueError:
                    pub_bytes = b64decode(public_key)

            verify_key = VerifyKey(pub_bytes)
            try:
                verify_key.verify(
                    manifest.to_canonical_json().encode(), b64decode(manifest.signature)
                )
                click.echo("🔐 Signature OK.")
            except BadSignatureError:
                click.echo("❌ Signature invalid.")
        elif manifest.signature:
            click.echo("ℹ Manifest signed but no public key provided.")

        click.echo("✅ Schema validation successful.")
    except Exception as e:
        click.echo(f"❌ Validation failed: {e}")


# ============================================================
# PUBLISH
# ============================================================


@manifest_cli.command("publish")
@click.argument("manifest_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--signing-key-path",
    type=click.Path(path_type=Path),
    help="Ed25519 key as raw hex (optional, fallback to TEE_KEY_HEX)",
)
def publish_manifest_cdn_cmd(manifest_path: Path, signing_key_path: Path | None):
    """
    Sign, upload (with retries), integrity-check, and update index.json.
    """

    settings = get_settings()
    bucket = settings.SCOREVISION_BUCKET

    # ----------------------------------------------------------
    # Load signing key (from file or TEE_KEY_HEX env)
    # ----------------------------------------------------------
    key_hex = None
    if signing_key_path:
        key_hex = signing_key_path.read_text().strip()
    elif os.environ.get("TEE_KEY_HEX"):
        key_hex = os.environ["TEE_KEY_HEX"]

    if not key_hex:
        raise click.UsageError(
            "Either --signing-key-path or TEE_KEY_HEX env must be provided."
        )

    signing_key = SigningKey(bytes.fromhex(key_hex))

    # ----------------------------------------------------------
    # Load + sign manifest
    # ----------------------------------------------------------
    manifest = load_manifest_from_yaml(manifest_path)
    manifest.sign(signing_key)
    manifest_hash = manifest.hash
    save_manifest_yaml(manifest, manifest_path)

    # ----------------------------------------------------------
    # Upload to R2/CDN
    # ----------------------------------------------------------
    from scorevision.utils.r2 import r2_get_object, r2_put_json, r2_delete_object
    import hashlib
    from datetime import datetime, timezone

    manifest_key = f"manifests/{manifest_hash}.json"
    existing, _ = r2_get_object(bucket, manifest_key)

    if existing is None:
        click.echo(f"⬆ Uploading manifest {manifest_hash}...")
        r2_put_json(bucket, manifest_key, json.loads(manifest.to_canonical_json()))
    else:
        click.echo("ℹ Manifest already exists in CDN (skipping upload).")

    remote_bytes, _ = r2_get_object(bucket, manifest_key)
    if remote_bytes is None:
        raise click.ClickException("Remote manifest missing after upload.")
    remote_hash = hashlib.sha256(remote_bytes).hexdigest()
    if remote_hash != manifest_hash:
        raise click.ClickException(
            f"Integrity mismatch: local={manifest_hash} remote={remote_hash}"
        )
    click.echo("🧩 Integrity OK.")

    # Update index.json
    index_key = "index.json"
    index_bytes, etag = r2_get_object(bucket, index_key)
    index = json.loads(index_bytes.decode("utf-8")) if index_bytes else {"windows": {}}
    win = manifest.window_id
    index.setdefault("windows", {}).setdefault(win, {})
    index["windows"][win].update(
        {
            "current": manifest_hash,
            "version": manifest.version,
            "expiry_block": manifest.expiry_block,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )

    try:
        click.echo("📝 Updating index.json...")
        r2_put_json(bucket, index_key, index, if_match=etag)
    except Exception as e:
        if existing is None:
            click.echo("⚠ Rolling back manifest upload...")
            r2_delete_object(bucket, manifest_key)
        raise click.ClickException(f"Failed to update index.json: {e}")

    click.echo("✅ Publish complete.")


# ============================================================
# LIST (stub)
# ============================================================


@manifest_cli.command("list")
def list_manifests_cmd():
    """List published manifests from CDN/R2 index."""
    click.echo("📄 Listing manifests not yet implemented.")
    # TODO: load index.json from R2 and print entries


# ============================================================
# CURRENT (stub)
# ============================================================


@manifest_cli.command("current")
@click.option("--block", type=int, default=None)
def current_manifest_cmd(block: int):
    """Show active manifest for a given block."""
    settings = get_settings()
    click.echo(
        f"ℹ Would fetch active manifest from CDN index for block={block or 'latest'} "
        f"(network={settings.NETWORK})."
    )


# ============================================================
# ROLLBACK (stub)
# ============================================================


@manifest_cli.command("rollback")
@click.option("--to-hash", required=True, type=str)
def rollback_manifest_cmd(to_hash: str):
    """Rollback CDN manifest pointer to a previous hash."""
    click.echo(f"⚠ Rollback requested to manifest hash {to_hash}")
    # TODO: update CDN index.json pointer
