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
- Keep response latency within the public-model requirement described below.
- Validate output format against current Element expectations.

For chute structure and local/live testing flow, use `example_miner/README.md`.

### Public-Model Latency Check

An automated compliance loop regularly evaluates every public model in a standardized
2 vCPU environment. Models must achieve a p95 inference latency of **100 ms or less per
frame**. In other words, at least 95% of evaluated frames must be processed within 100 ms.

Benchmark your model under the same 2 vCPU constraint before publishing a new revision;
performance measured on a GPU or a larger CPU instance is not representative of this
compliance check.

### Model-Copy Protection

The highest average score is selected as the provisional winner. When another miner
produces sufficiently similar scores on the same challenges, the validator performs
additional comparisons using both recent results and a separate sample of historical
results. The models are treated as equivalent only when their outputs remain similar
across those comparisons.

If multiple miners are confirmed as equivalent, the tie is resolved using their
on-chain commit blocks for that Element: the miner with the earlier relevant commit
wins. A model copied and committed later therefore cannot displace the earlier model
merely by reproducing the same outputs. The commit block is only a tie-breaker; a model
with meaningfully different and better results can still win through its score.

Commit model revisions promptly and keep the associated Hugging Face revision
reproducible. Publishing the on-chain commitment establishes the ordering used by this
protection.

## 4. Customize the Chute Template
The open-source deploy flow uses the template files inside `scorevision/miner/open_source/chute_template/`:
- `turbovision_chute.py.j2` – main Chute template rendered by the CLI for build/deploy.
- `schemas.py` – shared schema definitions used by the template.

Your model implementation lives in your Hugging Face repo (see `example_miner/README.md`):
- `miner.py` – required miner entrypoint loaded by the Chute template.
- `chute_config.yml` – optional Chutes runtime/image configuration.

## 5. Push, Deploy, Commit
Before deploying, configure two separate Hugging Face tokens:

- `HF_TOKEN`: the local token used to create/update the repository.
- `CHUTES_HF_TOKEN`: a fine-grained read-only token scoped only to the miner repository. The deploy command stores it as the chute's `HF_TOKEN` secret.

The deployment keeps the Hugging Face repository private while Chutes builds and
loads the model. It warms the chute and checks `/health`, submits the on-chain
commit, and only makes the repository public after the commit succeeds. With
`--no-commit`, the repository remains private.

Deploy with the current CLI command:

```bash
sv -v deploy-os-miner --model-path <path_to_model_assets> --element-id <element_id>
```

Useful flags:
- `--revision <sha-or-branch>`: force a specific Hugging Face revision.
- `--no-deploy`: upload/update HF only.
- `--no-commit`: skip chain commit and print payload only.

If `--element-id` is omitted (and commit is enabled), `sv deploy-os-miner` reads the active manifest and prompts you to choose an element.

## 6. Validate Deployment Health
Use Chutes dashboard and instance logs:

```bash
curl -X GET "https://api.chutes.ai/instances/<CHUTE_INSTANCE_ID>/logs" \
  -H "Authorization: <CHUTES_API_KEY>"
```

You can also inspect available metric pillars from the CLI:

```bash
sv elements list
```

## 7. Iterate Safely
- Track scoring behavior and redeploy frequently.
- Use `--no-commit` for dry-runs before publishing new commitments.
- Keep model artifacts and revisions reproducible.

Once this flow is in place, your miner is aligned with the current Turbo Vision command set.
