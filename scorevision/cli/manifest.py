import json
import click
from pathlib import Path
from base64 import b64decode
from ruamel.yaml import YAML

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

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
        pillars = Pillars.from_dict(e["metrics"]["pillars"])
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
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    required=True,
)
def create_manifest_cmd(template: str, window_id: str, expiry_block: int, output: Path):
    """Create a new manifest from a template."""
    if template not in TEMPLATES:
        raise click.ClickException(f"Unknown template: {template}")

    tmpl = TEMPLATES[template]
    manifest_yaml = {
        "window_id": window_id,
        "version": tmpl["version"],
        "expiry_block": expiry_block,
        "tee": tmpl["tee"],
        "elements": tmpl["elements"],   # user will fill manually
    }

    yaml.dump(manifest_yaml, output.open("w"))
    click.echo(f"✅ Manifest scaffold generated at {output}")


# ============================================================
# VALIDATE
# ============================================================

@manifest_cli.command("validate")
@click.argument("manifest_path", type=click.Path(exists=True, path_type=Path))
@click.option("--public-key", type=str, required=False)
def validate_manifest_cmd(manifest_path: Path, public_key: str | None):
    """Validate schema, pillars, baseline, and optionally signature."""
    try:
        manifest = load_manifest_from_yaml(manifest_path)
        _ = manifest.hash  # ensure canonical JSON is valid

        if manifest.signature and public_key:
            if Path(public_key).exists():
                pub_bytes = Path(public_key).read_bytes()
            else:
                pub_bytes = b64decode(public_key)

            pub = Ed25519PublicKey.from_public_bytes(pub_bytes)
            if manifest.verify(pub):
                click.echo("🔐 Signature OK.")
            else:
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
@click.option("--signing-key-path", required=True, type=click.Path(path_type=Path))
def publish_manifest_cmd(manifest_path: Path, signing_key_path: Path):
    """Sign, hash, and upload manifest."""
    try:
        manifest = load_manifest_from_yaml(manifest_path)

        priv = Ed25519PrivateKey.from_private_bytes(signing_key_path.read_bytes())
        manifest.sign(priv)
        h = manifest.hash

        save_manifest_yaml(manifest, manifest_path)

        click.echo(f"🔏 Signed manifest saved to {manifest_path}")
        click.echo(f"🧩 Hash: {h}")

        # TODO: upload_to_r2(manifest, h)

    except Exception as e:
        click.echo(f"❌ Publish failed: {e}")


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

