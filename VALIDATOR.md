# Turbo Vision Validator Guide

Validators keep Turbo Vision honest by scoring miner outputs and publishing weights on-chain.

## 0. Prerequisites
- Finish the shared setup in `README.md` (wallets, Chutes access, Hugging Face credentials, CLI).
- Prepare a host with Docker and access to `.env`.

## 1. Prepare Cloudflare R2
1. In the [Cloudflare Dashboard](https://dash.cloudflare.com), create/select your R2 bucket.
2. Generate a Read/Write API token.
3. Enable public access and copy the public URL.

## 2. Configure `.env`
Use the current variable names from `env.example`:

```bash
R2_ACCOUNT_ID=<your_r2_account_id>
R2_WRITE_ACCESS_KEY_ID=<your_access_key_id>
R2_WRITE_SECRET_ACCESS_KEY=<your_secret_access_key>
R2_BUCKET=<your_bucket_name>
R2_BUCKET_PUBLIC_URL=<your_public_url>
SCOREVISION_RESULTS_PREFIX=results_soccer
AUDIT_R2_RESULTS_PREFIX=audit_spotcheck
SCOREVISION_NETUID=<target_subnet_id>
```

Also ensure shared values exist: `BITTENSOR_WALLET_COLD`, `BITTENSOR_WALLET_HOT`, `CHUTES_API_KEY`, `HF_USER`, `HF_TOKEN`.

## 3. Launch with Docker (Recommended)
From repo root:

```bash
docker compose --profile validator up --build -d
docker compose ps
docker compose logs -f central-weights
docker compose logs -f central-signer
docker compose logs -f audit-spotcheck
```


## 4. Optional Local CLI Modes
For local debugging without Docker:

```bash
sv -v central-validator start
sv -v central-validator runner
sv -v central-validator weights
sv -v central-validator signer

sv -v audit-validator start
sv -v audit-validator spotcheck --once
```

## 5. Operations Checklist
- Check that R2 keys are written under the configured prefixes.
- Watch `central-weights` / `audit-spotcheck` logs for repeated failures.
- Keep API keys and wallet access healthy and rotated.

When these checks pass, your validator stack is correctly aligned with the current command and config model.
