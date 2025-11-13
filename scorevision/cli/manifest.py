import json
import click
from pathlib import Path
from base64 import b64decode
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from scorevision.utils.manifest import Manifest
from scorevision.utils.settings import get_settings


@click.group(name="manifest")
def manifest_cli():
    """
    Manage ScoreVision manifests.

    See notes/toward-manako/IMPLEMENTATION_PLAN.md for background.
    """
    pass


# ---------------------------------------------------------------------
# CREATE
# ---------------------------------------------------------------------
@manifest_cli.command("create")
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default="manifest_template.json",
    help="Output path for manifest template file.",
)
def create_manifest_cmd(output: Path):
    """Scaffold a manifest template filled with placeholder values."""
    # Create a basic manifest scaffold
    manifest = Manifest.empty()
    canonical = json.loads(manifest.to_canonical_json())
    output.write_text(json.dumps(canonical, indent=2))
    click.echo(f"✅ Manifest template created at: {output}")


# ---------------------------------------------------------------------
# VALIDATE
# ---------------------------------------------------------------------
@manifest_cli.command("validate")
@click.argument("manifest_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--public-key", type=str, required=False, help="Optional Ed25519 public key (base64 or file path).")
def validate_manifest_cmd(manifest_path: Path, public_key: str | None):
    """Validate manifest schema and signature integrity."""
    data = json.loads(manifest_path.read_text())
    try:
        manifest = Manifest(**data)

        # Validate structure
        _ = manifest.hash  # ensure to_canonical_json works without error

        # Optionally verify signature
        if manifest.signature and public_key:
            # load public key (either from file or base64)
            if Path(public_key).exists():
                pubkey_bytes = Path(public_key).read_bytes()
            else:
                pubkey_bytes = b64decode(public_key)
            pub = Ed25519PublicKey.from_public_bytes(pubkey_bytes)
            verified = manifest.verify(pub)
            if verified:
                click.echo("✅ Signature verification successful.")
            else:
                click.echo("❌ Invalid signature.")
        elif manifest.signature:
            click.echo("ℹ️ Manifest has a signature, but no public key was provided for verification.")
        else:
            click.echo("ℹ️ Manifest is unsigned — schema looks valid.")

        click.echo("✅ Manifest validation successful.")

    except Exception as e:
        click.echo(f"❌ Validation failed: {e}")


# ---------------------------------------------------------------------
# PUBLISH
# ---------------------------------------------------------------------
@manifest_cli.command("publish")
@click.argument("manifest_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--private-key", type=str, required=True, help="Path to Ed25519 private key for signing.")
def publish_manifest_cmd(manifest_path: Path, private_key: str):
    """Sign, hash, and (eventually) publish manifest to CDN."""
    data = json.loads(manifest_path.read_text())
    manifest = Manifest(**data)

    try:
        # Load private key
        private_key_bytes = Path(private_key).read_bytes()
        priv = Ed25519PrivateKey.from_private_bytes(private_key_bytes)

        # Sign and compute hash
        manifest.sign(priv)
        manifest_hash = manifest.hash

        # Overwrite manifest file with signature added
        manifest_path.write_text(json.dumps(json.loads(manifest.to_canonical_json()) | {"signature": manifest.signature}, indent=2))

        # Placeholder for future CDN upload logic
        click.echo(f"✅ Manifest signed and saved to {manifest_path}")
        click.echo(f"🔑 Signature: {manifest.signature[:16]}...")
        click.echo(f"🧩 Hash: {manifest_hash}")

        # In the future:
        # upload_to_cdn(manifest, manifest_hash)

    except Exception as e:
        click.echo(f"❌ Publish failed: {e}")


# ---------------------------------------------------------------------
# LIST
# ---------------------------------------------------------------------
@manifest_cli.command("list")
def list_manifests_cmd():
    """List all published manifests from CDN index."""
    try:
        # Placeholder until CDN logic implemented
        click.echo("📄 Manifest index retrieval not yet implemented.")
        click.echo("See notes/toward-manako/IMPLEMENTATION_PLAN.md for future CDN integration.")
    except Exception as e:
        click.echo(f"❌ Could not fetch manifest list: {e}")


# ---------------------------------------------------------------------
# CURRENT
# ---------------------------------------------------------------------
@manifest_cli.command("current")
@click.option("--block", type=int, default=None, help="Optional block number to query.")
def current_manifest_cmd(block: int):
    """Show the currently active manifest (stub for now)."""
    try:
        settings = get_settings()
        click.echo(
            f"ℹ️ Current manifest retrieval not yet implemented.\n"
            f"Would query CDN or index.json based on block={block or 'latest'}.\n"
            f"Network: {settings.NETWORK if hasattr(settings, 'NETWORK') else 'unknown'}"
        )
    except Exception as e:
        click.echo(f"❌ Failed to fetch current manifest: {e}")

