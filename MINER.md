# Turbo Vision Miner Guide

Miners publish models, deploy them to Chutes, and commit metadata on-chain.

## 0. Prerequisites
- Complete the shared setup in `README.md`.
- Ensure `.env` contains: `BITTENSOR_WALLET_COLD`, `BITTENSOR_WALLET_HOT`, `CHUTES_API_KEY`, `HF_USER`, `HF_TOKEN`, `SCOREVISION_NETUID`.
- Have GPU/cloud capacity for inference.

## 1. Register Your Hotkey on the Target Subnet
Use the same subnet ID as `SCOREVISION_NETUID` in `.env`:

```bash
btcli subnet register --netuid <SCOREVISION_NETUID> --wallet.name <coldkey_name> --wallet.hotkey <hotkey_name>
```

## 2. Enable Chutes Developer Deployments
Turbo Vision deployment requires a funded Chutes account:

```bash
pip install -U chutes
chutes register
```

Confirm `CHUTES_API_KEY` is a developer key.

## 3. Prepare Your Miner Code
- Build your model to handle validator challenge payloads.
- Keep response latency stable enough for live scoring.
- Validate output format against current Element expectations.

For chute structure and local/live testing flow, use `example_miner/README.md`.

## 4. Push, Deploy, Commit
Deploy with the current CLI command:

```bash
sv -v push --model-path <path_to_model_assets> --element-id <element_id>
```

Useful flags:
- `--revision <sha-or-branch>`: force a specific Hugging Face revision.
- `--no-deploy`: upload/update HF only.
- `--no-commit`: skip chain commit and print payload only.

If `--element-id` is omitted (and commit is enabled), `sv push` reads the active manifest and prompts you to choose an element.

## 5. Validate Deployment Health
Use Chutes dashboard and instance logs:

```bash
curl -X GET "https://api.chutes.ai/instances/<CHUTE_INSTANCE_ID>/logs" \
  -H "Authorization: <CHUTES_API_KEY>"
```

You can also inspect available metric pillars from the CLI:

```bash
sv elements list
```

## 6. Iterate Safely
- Track scoring behavior and redeploy frequently.
- Use `--no-commit` for dry-runs before publishing new commitments.
- Keep model artifacts and revisions reproducible.

Once this flow is in place, your miner is aligned with the current Turbo Vision command set.
