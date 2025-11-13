import json
import click
from pathlib import Path
from scorevision.utils.manifest import Manifest
from scorevision.utils.settings import get_settings


@click.group(name="manifest")
def manifest_cli():
    """
    Manage ScoreVision manifests.

    See notes/toward-manako/IMPLEMENTATION_PLAN.md for background.
    """
    pass


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
    manifest = create_manifest_template()
    output.write_text(json.dumps(manifest, indent=2))
    click.echo(f"✅ Manifest template created at: {output}")


@manifest_cli.command("validate")
@click.argument("manifest_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def validate_manifest_cmd(manifest_path: Path):
    """Validate manifest schema and signature integrity."""
    data = json.loads(manifest_path.read_text())
    try:
        manifest = Manifest(**data)
        validate_manifest(manifest)
        click.echo("✅ Manifest validation successful.")
    except Exception as e:
        click.echo(f"❌ Validation failed: {e}")


@manifest_cli.command("publish")
@click.argument("manifest_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--private-key", type=str, required=True, help="Path to Ed25519 private key for signing.")
def publish_manifest_cmd(manifest_path: Path, private_key: str):
    """Sign, hash, and publish manifest to CDN."""
    data = json.loads(manifest_path.read_text())
    manifest = Manifest(**data)
    try:
        result = publish_manifest(manifest, private_key_path=private_key)
        click.echo(f"✅ Published manifest {result['hash']} to CDN.")
    except Exception as e:
        click.echo(f"❌ Publish failed: {e}")


@manifest_cli.command("list")
def list_manifests_cmd():
    """List all published manifests from CDN index."""
    try:
        index = fetch_index()
        if not index:
            click.echo("No manifests found.")
            return
        for entry in index:
            click.echo(f"- {entry['hash']} (window_id={entry['window_id']}, version={entry['version']})")
    except Exception as e:
        click.echo(f"❌ Could not fetch manifest list: {e}")


@manifest_cli.command("current")
@click.option("--block", type=int, default=None, help="Optional block number to query.")
def current_manifest_cmd(block: int):
    """Show the currently active manifest for a given block."""
    try:
        manifest = get_current_manifest(block_number=block)
        click.echo(json.dumps(manifest.__dict__, indent=2))
    except Exception as e:
        click.echo(f"❌ Failed to fetch current manifest: {e}")

