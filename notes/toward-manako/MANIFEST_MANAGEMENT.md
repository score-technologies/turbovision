# Manifest Management Guide for Score Operators

## Overview

This guide describes the recommended approach for Score operators to create, manage, and publish Manifests. The system is designed to be:

- **Human-friendly**: YAML configs for authoring, JSON for storage
- **Version-controlled**: Configs live in git, can be reviewed/audited
- **Safe**: Validation before publishing, rollback capability
- **Automated**: CLI tools handle signing, hashing, publishing

---

## Architecture

### 1. Config-Based Authoring (Human-Friendly)

**Location:** `manifests/configs/` directory in repo

Manifests are authored as **YAML files** (easier to read/edit than JSON):

```yaml
# manifests/configs/window-2025-10-23.yaml
window_id: "2025-10-23"
version: "1.3"
expiry_block: 123456

elements:
  - id: "PlayerDetect_v1@1.0"
    clips:
      - hash: "sha256:abc123..."
        weight: 1.0
    metrics:
      pillars:
        iou_placement: 0.35
        count_accuracy: 0.20
        palette_symmetry: 0.15
        smoothness: 0.15
        role_consistency: 0.15
    latency_p95_ms: 200
    service_rate_fps: 25
    baseline_theta: 0.78
    delta_floor: 0.01
    beta: 1.0
    preproc:
      fps: 5
      resize_long: 1280
      norm: "rgb-01"

  - id: "BallDetect_v1@1.0"
    # ... similar structure
    beta: 1.4 # Higher difficulty weight

pgt_recipe_hash: "sha256:def456..." # Hash of PGT generation code
tee:
  trusted_share_gamma: 0.2
```

**Benefits:**

- ✅ Human-readable and editable
- ✅ Version-controlled in git
- ✅ Can be reviewed in PRs
- ✅ Supports comments
- ✅ Easy to template/commonize

### 2. Template System

**Location:** `manifests/templates/` directory

Common configurations as templates:

```yaml
# manifests/templates/default-football.yaml
# Base template for football evaluation windows

elements:
  - id: "PlayerDetect_v1@1.0"
    metrics:
      pillars:
        iou_placement: 0.35
        count_accuracy: 0.20
        # ... defaults
    latency_p95_ms: 200
    service_rate_fps: 25
    baseline_theta: 0.78
    beta: 1.0
    preproc:
      fps: 5
      resize_long: 1280
      norm: "rgb-01"
# ... other default Elements
```

**Usage:**

```bash
# Create new manifest from template
sv manifest create --template default-football --window-id 2025-10-24
```

### 3. CLI Tool for Manifest Management

**New Command:** `sv manifest`

```bash
# Create new manifest from template
sv manifest create \
  --template default-football \
  --window-id 2025-10-24 \
  --expiry-block 123456 \
  --output manifests/configs/window-2025-10-24.yaml

# Validate manifest config (before publishing)
sv manifest validate manifests/configs/window-2025-10-24.yaml

# Publish manifest (signs, hashes, uploads to R2, updates index)
sv manifest publish \
  --config manifests/configs/window-2025-10-24.yaml \
  --signing-key-path ~/.score/manifest-signing-key

# List published manifests
sv manifest list

# Get current active manifest
sv manifest current

# Rollback to previous manifest (if needed)
sv manifest rollback --to-hash sha256:abc123...
```

---

## Implementation

### File Structure

```
turbovision/
├── manifests/
│   ├── configs/              # YAML configs (version controlled)
│   │   ├── window-2025-10-23.yaml
│   │   ├── window-2025-10-24.yaml
│   │   └── ...
│   ├── templates/            # Template configs
│   │   ├── default-football.yaml
│   │   ├── default-cricket.yaml
│   │   └── ...
│   └── published/            # Published manifests (gitignored, for reference)
│       └── {manifest_hash}.json
├── scorevision/
│   └── cli/
│       └── manifest.py      # New CLI module
└── ...
```

### CLI Implementation

**New File:** `scorevision/cli/manifest.py`

```python
import click
import yaml
import json
from pathlib import Path
from hashlib import sha256
from datetime import datetime
import nacl.signing
import nacl.encoding
from typing import Optional

from scorevision.utils.settings import get_settings
from scorevision.utils.cloudflare_helpers import get_s3_client, _put_json_object
from scorevision.utils.bittensor_helpers import get_subtensor


@click.group()
def manifest():
    """Manage Score Vision Manifests"""
    pass


@manifest.command("create")
@click.option("--template", required=True, help="Template name")
@click.option("--window-id", required=True, help="Window ID (e.g., 2025-10-24)")
@click.option("--expiry-block", type=int, required=True, help="Expiry block number")
@click.option("--output", required=True, help="Output YAML file path")
def create_manifest(template: str, window_id: str, expiry_block: int, output: str):
    """Create new manifest from template"""
    template_path = Path(f"manifests/templates/{template}.yaml")
    if not template_path.exists():
        click.echo(f"Template not found: {template_path}", err=True)
        return

    with open(template_path) as f:
        template_data = yaml.safe_load(f)

    # Override with provided values
    template_data["window_id"] = window_id
    template_data["expiry_block"] = expiry_block

    # Write to output
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(template_data, f, default_flow_style=False, sort_keys=False)

    click.echo(f"Created manifest config: {output_path}")


@manifest.command("validate")
@click.argument("config_path", type=click.Path(exists=True))
def validate_manifest(config_path: str):
    """Validate manifest config before publishing"""
    config_path = Path(config_path)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    errors = []

    # Required fields
    required = ["window_id", "version", "expiry_block", "elements"]
    for field in required:
        if field not in config:
            errors.append(f"Missing required field: {field}")

    # Validate elements
    if "elements" in config:
        for i, element in enumerate(config["elements"]):
            if "id" not in element:
                errors.append(f"Element {i} missing 'id'")
            if "metrics" not in element:
                errors.append(f"Element {i} missing 'metrics'")
            # ... more validation

    if errors:
        click.echo("Validation errors:", err=True)
        for error in errors:
            click.echo(f"  - {error}", err=True)
        return 1

    click.echo("✓ Manifest config is valid")
    return 0


@manifest.command("publish")
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--signing-key-path", required=True, help="Path to Ed25519 signing key")
@click.option("--dry-run", is_flag=True, help="Validate but don't publish")
def publish_manifest(config_path: str, signing_key_path: str, dry_run: bool):
    """Publish manifest (sign, hash, upload to R2, update index)"""
    config_path = Path(config_path)

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Validate first
    # ... validation logic ...

    # Convert to JSON (canonical format)
    manifest_json = json.dumps(config, sort_keys=True, separators=(",", ":"))

    # Sign manifest
    with open(signing_key_path, "rb") as f:
        signing_key = nacl.signing.SigningKey(f.read())

    signature = signing_key.sign(manifest_json.encode()).signature.hex()

    # Create signed manifest
    signed_manifest = {
        "manifest": config,
        "signature": signature,
        "signed_at": datetime.utcnow().isoformat(),
        "signed_by": signing_key.verify_key.encode(encoder=nacl.encoding.HexEncoder).decode(),
    }

    signed_json = json.dumps(signed_manifest, sort_keys=True, separators=(",", ":"))

    # Compute hash
    manifest_hash = sha256(signed_json.encode()).hexdigest()

    click.echo(f"Manifest hash: sha256:{manifest_hash}")
    click.echo(f"Signature: {signature[:16]}...")

    if dry_run:
        click.echo("Dry run - not publishing")
        return

    # Upload to R2
    settings = get_settings()
    manifest_key = f"scorevision/manifests/{manifest_hash}.json"

    async def _publish():
        async with get_s3_client() as client:
            await client.put_object(
                Bucket=settings.R2_BUCKET,
                Key=manifest_key,
                Body=signed_json.encode(),
                ContentType="application/json",
            )

        # Update index
        await _update_manifest_index(manifest_hash, config)

        click.echo(f"✓ Published manifest to R2: {manifest_key}")
        click.echo(f"  Public URL: {settings.R2_BUCKET_PUBLIC_URL}/{manifest_key}")

    import asyncio
    asyncio.run(_publish())


async def _update_manifest_index(manifest_hash: str, config: dict):
    """Update scorevision/index.json with new manifest"""
    settings = get_settings()
    index_key = "scorevision/index.json"

    async with get_s3_client() as client:
        try:
            resp = await client.get_object(Bucket=settings.R2_BUCKET, Key=index_key)
            index_data = json.loads(await resp["Body"].read())
        except client.exceptions.NoSuchKey:
            index_data = {"version": "1.0", "manifests": []}

        # Add new manifest entry
        manifest_entry = {
            "hash": f"sha256:{manifest_hash}",
            "window_id": config["window_id"],
            "expiry_block": config["expiry_block"],
            "published_at": datetime.utcnow().isoformat(),
            "elements_summary": [
                {
                    "id": element["id"],
                    "beta": element.get("beta", 1.0),
                    "baseline_theta": element.get("baseline_theta"),
                    "service_rate_fps": element.get("service_rate_fps"),
                }
                for element in config.get("elements", [])
            ],
        }

        index_data["manifests"].append(manifest_entry)
        index_data["manifests"].sort(key=lambda x: x["expiry_block"], reverse=True)
        index_data["current_manifest_hash"] = f"sha256:{manifest_hash}"

        await client.put_object(
            Bucket=settings.R2_BUCKET,
            Key=index_key,
            Body=json.dumps(index_data, indent=2).encode(),
            ContentType="application/json",
        )


@manifest.command("list")
def list_manifests():
    """List all published manifests"""
    async def _list():
        settings = get_settings()
        index_key = "scorevision/index.json"

        async with get_s3_client() as client:
            try:
                resp = await client.get_object(Bucket=settings.R2_BUCKET, Key=index_key)
                index_data = json.loads(await resp["Body"].read())
            except client.exceptions.NoSuchKey:
                click.echo("No manifests found")
                return

        st = await get_subtensor()
        current_block = await st.get_current_block()

        click.echo("Published Manifests:")
        click.echo("")
        for m in index_data["manifests"]:
            is_active = m["expiry_block"] > current_block
            status = "ACTIVE" if is_active else "EXPIRED"
            click.echo(f"  {m['hash']}")
            click.echo(f"    Window: {m['window_id']}")
            click.echo(f"    Expiry: Block {m['expiry_block']} ({status})")
            click.echo(f"    Published: {m['published_at']}")
            click.echo("")

    import asyncio
    asyncio.run(_list())


@manifest.command("current")
def get_current_manifest():
    """Get current active manifest"""
    async def _current():
        from scorevision.utils.manifest import get_current_manifest

        manifest = await get_current_manifest()
        if manifest is None:
            click.echo("No active manifest found")
            return

        click.echo(f"Current Manifest:")
        click.echo(f"  Hash: {manifest.hash}")
        click.echo(f"  Window ID: {manifest.window_id}")
        click.echo(f"  Expiry Block: {manifest.expiry_block}")
        click.echo(f"  Elements: {len(manifest.elements)}")

    import asyncio
    asyncio.run(_current())
```

### Integration with Existing CLI

**Update:** `scorevision/__init__.py`

```python
from scorevision.cli.manifest import manifest

cli.add_command(manifest)  # Add manifest command group
```

---

## Miner Discovery & Commitments (Reference)

Operators expose manifests publicly so miners can coordinate via the dedicated CLI shipped alongside the manifest tooling:

- `sv miner elements --window current` pulls `scorevision/index.json`, fetches the active manifest, and renders each Element's metrics (θ, β, service rate, clip counts).
- `sv miner elements --window upcoming` previews the next manifest (highest `expiry_block` beyond current), helping miners stage upgrades before the window flip.
- `sv miner manifest --hash sha256:...` downloads and verifies any manifest locally, relying on the same signature/Ed25519 verification logic.
- `sv miner commit --window-id <id> --elements <elem:revision,...> --service-cap-fps <n>` calls `scorevision/utils/commitments.py` to submit `commit_element_set` extrinsics on Bittensor and stores a `commitment_proof` (`{block}:{extrinsic_hash}`) along with the stated `hf_revision/chute_slug/chute_id` per Element.

Commitment receipts are cached under `~/.score/commitments/{window_id}.json` (gitignored) so the runner and validator CLIs can attach the correct proof to Challenge API requests and shards. Operators should avoid modifying these files manually; miners regenerate them automatically whenever they commit or withdraw.

---

## Workflow Example

### Creating a New Evaluation Window

```bash
# 1. Create manifest config from template
sv manifest create \
  --template default-football \
  --window-id 2025-10-24 \
  --expiry-block 123456 \
  --output manifests/configs/window-2025-10-24.yaml

# 2. Edit config (add specific clips, adjust parameters)
vim manifests/configs/window-2025-10-24.yaml

# 3. Validate before publishing
sv manifest validate manifests/configs/window-2025-10-24.yaml

# 4. Dry run to see what will be published
sv manifest publish \
  --config manifests/configs/window-2025-10-24.yaml \
  --signing-key-path ~/.score/manifest-signing-key \
  --dry-run

# 5. Publish for real
sv manifest publish \
  --config manifests/configs/window-2025-10-24.yaml \
  --signing-key-path ~/.score/manifest-signing-key

# 6. Verify it's published
sv manifest list
sv manifest current
```

### Miner Preparation (Read-Only Example)

Once manifests are published, miners typically:

1. Inspect what is live now: `sv miner elements --window current`.
2. Inspect what is coming next: `sv miner elements --window upcoming`.
3. Pull the manifest locally for offline tuning: `sv miner manifest --hash sha256:<current_manifest_hash>`.
4. Build or update a chute per Element/HF revision, then record the slug/chute_id mapping.
5. Lock in participation (and publish the slug mapping) using the upcoming command format: `sv miner commit --window-id <window> --elements PlayerDetect_v1@1.0:player-v1 --service-cap-fps 30`.

These steps ensure the miner has both visibility into current/upcoming Elements and a recorded `commitment_proof` ready for Challenge API v3 requests and shard emission.

### Updating Challenge API Backend

Once manifest is published, the Challenge API backend needs to:

1. Read manifest from R2 (or cache locally)
2. Validate `X-Manifest-Hash` header matches active manifest
3. Return `element_id` and `window_id` from manifest

---

## Best Practices

### 1. Version Control

- ✅ Keep all configs in git (`manifests/configs/`)
- ✅ Review configs in PRs before publishing
- ✅ Tag releases when publishing major changes

### 2. Signing Key Management

- ✅ Store signing key securely (not in repo)
- ✅ Use separate keys for dev/staging/prod
- ✅ Rotate keys periodically
- ✅ Document key location for team members

### 3. Template Management

- ✅ Create templates for common configurations
- ✅ Document template parameters
- ✅ Version templates (e.g., `default-football-v1.yaml`)

### 4. Validation

- ✅ Always validate before publishing
- ✅ Use dry-run mode first
- ✅ Check expiry_block is reasonable (not too far in future)

### 5. Rollback Plan

- ✅ Keep previous manifest configs
- ✅ Document rollback procedure
- ✅ Test rollback in staging

---

## Security Considerations

1. **Signing Key Protection**

   - Store in secure location (e.g., AWS Secrets Manager, HashiCorp Vault)
   - Use separate keys per environment
   - Rotate keys periodically

2. **Manifest Integrity**

   - Content-addressed (hash prevents tampering)
   - Cryptographic signatures prove authenticity
   - Public verification (anyone can verify signature)

3. **Access Control**
   - R2 write access only for Score operators
   - Public read access (no auth required)
   - Audit log of who published what

---

## Future Enhancements

1. **Automated Publishing**

   - Cron job to publish manifests at window start
   - Integration with CI/CD pipeline

2. **Manifest Diff Tool**

   - Compare two manifests to see changes
   - Highlight parameter differences

3. **Manifest Preview**

   - Web UI to preview manifest before publishing
   - Validate against current network state

4. **Rollback Automation**

   - CLI command to rollback to previous manifest
   - Automatic rollback on errors

5. **Multi-Environment Support**
   - Separate manifests for dev/staging/prod
   - Environment-specific templates
