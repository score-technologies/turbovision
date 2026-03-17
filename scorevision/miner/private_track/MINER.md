# Miner Guide

## Overview

Miners receive video challenges from Score's central validator and return action predictions.

```
Score Validator → POST /challenge → Your Miner → Response → Score Validator
```

## Registration on the Subnet
Make sure your hotkey is registered on subnet **44**:

```bash
btcli subnet register --wallet.name <coldkey_name> --wallet.hotkey <hotkey_name>
```


## Quick Start


```bash
# Setup .venv and activate
python3.10 -m venv .venv
source .venv/bin/activate

# Install (local development)
pip install -e .

# Configure environment
cp env.example .env
# Edit .env with your credentials

# Build, push to GHCR, commit on-chain, and start
sv -v deploy-pt-miner --tag v1.0.0
```

## CLI Reference

### `sv -v deploy-pt-miner`

Builds your Docker image, pushes it to your private GHCR repo, commits on-chain, and starts the service.

```bash
# Full workflow: build → push → commit → start
sv -v deploy-pt-miner --tag v1.0.0

# Build only (for local testing)
sv -v deploy-pt-miner --tag v1.0.0 --no-push --no-start

# Build, push, but don't commit or start
sv -v deploy-pt-miner --tag v1.0.0 --no-commit --no-start
```

**Options:**
| Flag | Description |
|------|-------------|
| `--tag` | **Required.** Docker image tag (e.g., `v1.0.0`, `latest`) |
| `--no-push` | Skip pushing to GHCR |
| `--no-commit` | Skip on-chain commitment |
| `--no-start` | Skip starting the container |

## GHCR Setup

### Why Private Images?

Your Docker image contains your model/solution. To keep it private from other miners while allowing Score to verify it:

1. **You** own a private GHCR package
2. **You** add `DataAndMike` as a read-only collaborator via the GHCR UI
3. **Score** can pull your image for spot checks
4. **Other miners** cannot access your package

### Setup Steps

1. **Create a GitHub Personal Access Token (PAT):**
   - Go to https://github.com/settings/tokens
   - Click "Generate new token (classic)"
   - Select scopes: `write:packages`, `read:packages`, `delete:packages`
   - Copy the token

2. **Configure `.env`:**
   ```bash
   GITHUB_USERNAME=your-github-username
   GITHUB_TOKEN=ghp_xxxxx  # Your PAT with write:packages scope
   GHCR_REPO=pt-solution   # Your repo name (default: pt-solution)
   ```

3. **Deploy:**
   ```bash
   sv -v deploy-pt-miner --tag v1.0.0
   ```

   The CLI will automatically:
   - Build your Docker image
   - Push to `ghcr.io/your-username/pt-solution:v1.0.0`
   - Commit on-chain

   You will see an output like this at the end:
   ```bash
   Starting container on port 8000
   ✓ Miner running: container_id
   ```

   And you can get the live container logs by the following:

   ```bash
   docker logs -f container_id
   ```

   To stop the container (if you want to deploy a different one for example) you can run the following:

   ```bash
   docker stop container_id
   ```

4. **Share with Score (one-time manual step):**

   After your first push, grant Score read access to your private GHCR package:

   1. Go to `https://github.com/users/YOUR_USERNAME/packages/container/pt-solution/settings`
   2. Under **Manage access**, click **Invite teams or people**
   3. Add `DataAndMike` with **Read** access

   Without this step, your miner will fail spot-checks and be blacklisted.


## Environment Variables

Create a `.env` file:

```bash
# ──────────────────────────────────────────────────────────────────────────
# Bittensor Wallet Configuration
# ──────────────────────────────────────────────────────────────────────────
BITTENSOR_WALLET_COLD=default
BITTENSOR_WALLET_HOT=default
# BITTENSOR_WALLET_PATH=~/.bittensor/wallets  # Optional: Override wallet path
SCOREVISION_NETUID=44
BITTENSOR_SUBTENSOR_ENDPOINT=test
BITTENSOR_SUBTENSOR_FALLBACK=wss://test.finney.opentensor.ai:443

# =============================================================================
# PRIVATE TRACK MINER
# =============================================================================

MINER_PORT=8000

# GHCR - miners use their own private package
GITHUB_USERNAME=your-github-username
GITHUB_TOKEN=your-github-pat
GHCR_REPO=pt-solution  # Your private repo name (default: pt-solution)

# Fiber security
SUBTENSOR_NETWORK=finney
MIN_STAKE_THRESHOLD=1000

# Security toggles (default: true)
BLACKLIST_ENABLED=true
VERIFY_ENABLED=true
```

## Workflow

### 1. Implement Your Predictor

Edit `scorevision/miner/private_track/predictor.py`:

```python
from pathlib import Path
from src.schemas import FramePrediction

def predict_actions(video_path: Path) -> list[FramePrediction]:
    # Your model inference here
    predictions = []
    # ... your logic
    predictions.append(FramePrediction(frame=45, action="pass"))
    return predictions
```

### 2. Test Locally (Without Wallet)

For local development without a Bittensor wallet, disable security checks:

```bash
# Build without pushing or committing
sv -v deploy-pt-miner --tag test --no-push --no-start

# Run with security disabled (local testing only!)
docker run -p 8000:8000 \
  -e BLACKLIST_ENABLED=false \
  -e VERIFY_ENABLED=false \
  ghcr.io/your-username/pt-solution:test
```

You should see on startup:
```
INFO: Blacklist: DISABLED
INFO: Verify: DISABLED
WARNING: All security DISABLED - local testing only
```

### 3. Test with GHCR (Without On-Chain)

Test the full GHCR workflow without committing on-chain:

```bash
# Build → Push → Start (but skip on-chain commit)
sv -v deploy-pt-miner --tag test --no-commit
```

### 4. Test Locally (With Wallet)

To test with full security enabled, mount your wallet directory:

```bash
docker run -p 8000:8000 \
  -v ~/.bittensor/wallets:/root/.bittensor/wallets:ro \
  -e COLDKEY=default \
  -e HOTKEY=default \
  -e NETUID=44 \
  -e SUBTENSOR_NETWORK=finney \
  -e MIN_STAKE_THRESHOLD=1000 \
  ghcr.io/your-username/pt-solution:test
```

### 5. Deploy to Production

```bash
# Full deploy: build → push → commit on-chain → start
sv -v deploy-pt-miner --tag v1.0.0
```

The CLI will:
1. Build your Docker image
2. Push to `ghcr.io/your-username/pt-solution:v1.0.0`
3. Commit `{"image_repo": "ghcr.io/your-username/pt-solution", "image_tag": "v1.0.0"}` on-chain
4. Start the container locally

After deployment, **share with Score** (see [GHCR Setup](#ghcr-setup) step 4).

**For production**, always run with wallet mounted and security enabled:

```bash
docker run -d -p 8000:8000 \
  -v ~/.bittensor/wallets:/root/.bittensor/wallets:ro \
  -e COLDKEY=your-coldkey \
  -e HOTKEY=your-hotkey \
  -e NETUID=44 \
  -e SUBTENSOR_NETWORK=finney \
  -e MIN_STAKE_THRESHOLD=1000 \
  ghcr.io/your-username/pt-solution:v1.0.0
```

### 6. Verify On-Chain Commitment

Your commitment is stored on-chain and checked by the central validator before sending challenges.

## Request/Response Format

**Request:**
```json
{
  "challenge_id": "abc123",
  "video_url": "https://scoredata.me/chunks/4af9157146b44f23b006967c44f52f.mp4"
}
```

**Response:**
```json
{
  "challenge_id": "abc123",
  "predictions": [
    {"frame": 45, "action": "pass"},
    {"frame": 120, "action": "tackle"}
  ],
  "processing_time": 2.5
}
```

## Valid Actions

| Action | Weight | Tolerance |
|--------|--------|-----------|
| pass | 1.0 | 1.0s |
| pass_received | 1.4 | 1.0s |
| recovery | 1.5 | 1.5s |
| tackle | 2.5 | 1.5s |
| interception | 2.8 | 2.0s |
| shot | 4.7 | 2.0s |
| goal | 10.9 | 3.0s |

See `src/actions.py` for the full list.

## Security

By default your miner uses fiber's built-in security:

- **`blacklist_low_stake`** (`BLACKLIST_ENABLED=true`): Rejects requests from validators below stake threshold
- **`verify_request`** (`VERIFY_ENABLED=true`): Verifies signed requests from validators

Only Score's central validator (with sufficient stake) can send you challenges.

### Disabling Security (Local Testing Only)

For local development without a wallet:

```bash
BLACKLIST_ENABLED=false
VERIFY_ENABLED=false
```

**Never run with security disabled in production!**

## Spot Checks

Score periodically verifies your Docker image produces the same results as your live endpoint:

1. Score pulls your committed Docker image (using collaborator access)
2. Runs a test challenge against it
3. Compares output to your live response

**If outputs don't match:**
- Score = 0 for that period
- Added to blacklist

**Tips:**
- Ensure Docker image and live server use identical model weights
- Avoid non-deterministic behavior
- Test thoroughly before committing

## Troubleshooting

### "No such file or directory: wallets/default/hotkeys/default"

This means security is enabled but no wallet is mounted. Either:

1. **Mount your wallet directory:**
   ```bash
   docker run -v ~/.bittensor/wallets:/root/.bittensor/wallets:ro ...
   ```

2. **Or disable security for local testing:**
   ```bash
   docker run -e BLACKLIST_ENABLED=false -e VERIFY_ENABLED=false ...
   ```

### Docker build fails
```bash
docker build -f Dockerfile.miner -t test .
```

### Push fails
```bash
echo $GITHUB_TOKEN | docker login ghcr.io -u $GITHUB_USERNAME --password-stdin
```

### On-chain commit fails
- Ensure `COLDKEY` and `HOTKEY` are correct
- Check you're registered on subnet 44

### "Image not accessible"
- Ensure you've added `DataAndMike` as a collaborator with Read access on your GHCR package
- Go to `https://github.com/users/YOUR_USERNAME/packages/container/pt-solution/settings` to check
