# Turbo Vision Validator Guide

Validators keep Turbo Vision honest by benchmarking every submitted model against fresh match footage. Follow the steps below to start contributing structured signals back to the network.

## 0. Prerequisites
- Finish the shared setup in `README.md` (Bittensor wallet, Chutes developer access, Hugging Face credentials, ScoreVision CLI).
- Ready access to the validator host’s `.env` file and Docker (recommended deployment path).

## 1. Prepare Cloudflare R2 Storage
1. Log into the [Cloudflare Dashboard](https://dash.cloudflare.com) and open **R2**.
2. Create an R2 bucket (e.g. `scorevision-results`) and write down the **Account ID**.
3. Under **Manage R2 API Tokens**, mint a token with **Read/Write** access. Save the `R2_WRITE_ACCESS_KEY_ID` and `R2_WRITE_SECRET_ACCESS_KEY`.
4. In the bucket’s **Settings → Public Access**, enable public reads and note the **Public URL** your miners will hit for results.

## 2. Configure Environment Variables
Add the following variables to `.env` (or your process manager) on the validator host:

```bash
R2_ACCOUNT_ID=<your_r2_account_id>
R2_WRITE_ACCESS_KEY_ID=<your_access_key_id>
R2_WRITE_SECRET_ACCESS_KEY=<your_secret_access_key>
R2_BUCKET=<bucket_name>
R2_BUCKET_PUBLIC_URL=<public_bucket_url>
SCOREVISION_RESULTS_PREFIX=results_soccer
SCOREVISION_NETUID=<target_subnet_id>  # e.g. 44 for Turbo Vision
```

Double‑check that the shared values from the README (`BITTENSOR_WALLET_COLD`, `BITTENSOR_WALLET_HOT`, `CHUTES_API_KEY`, `HF_USER`, `HF_TOKEN`) are also present.

## 3. Launch the Validator (Docker Recommended)
From the repository root:

```bash
docker compose down && docker compose pull
docker compose up --build -d
docker compose logs -f validator
```

This pulls the latest validator image, rebuilds if necessary, and tails the validator logs so you can confirm it is receiving challenges and publishing scores.

> Turbo Vision validators target subnet mechanism **1**. The logic is baked into the code; only keep `SCOREVISION_NETUID` aligned with the live network.

## 4. Optional: Run the Stack Locally
The CLI lets you run validator components without Docker when debugging:

```bash
sv -vv validate   # end-to-end validation loop
sv -vv runner     # execute scoring jobs
sv -vv signer     # submit results on-chain
```

Use these commands on development machines before redeploying containers.

## 5. Stay Production Ready
- Confirm your R2 bucket is collecting artifacts and that public URLs are reachable.
- Monitor validator logs for failed submissions; miners depend on quick feedback.
- Keep your Chutes API token valid and rotate secrets according to your ops policy.

Once these steps are complete, your validator is live and helping Turbo Vision close the gap between automated models and elite football analysts.
