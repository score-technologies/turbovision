# Turbo Vision Miner Guide

Miners supply Turbo Vision with expert models that understand the flow of the match and answer validator challenges in real time. Use this checklist to go from zero to earning emissions.

## 0. Prerequisites
- Complete the common setup in `README.md` (wallets, Chutes developer access, Hugging Face credentials, ScoreVision CLI).
- Have GPU resources or cloud capacity to train and serve your model.

## 1. Register on the Score Vision Subnet
Make sure your hotkey is registered on subnet **44** (Turbo Vision):

```bash
btcli subnet register --wallet.name <coldkey_name> --wallet.hotkey <hotkey_name>
```

## 2. Unlock Chutes Developer Deployments
Turbo Vision requires a developer-enabled account on [chutes.ai](https://chutes.ai). After support confirms the upgrade:

```bash
pip install -U chutes
chutes register
```

Verify that `CHUTES_API_KEY` in `.env` contains a developer token—standard keys cannot deploy miners.

## 3. Build or Adapt Your Model
Train a model that can process the video frames or features used by the subnet. Keep track of:
- Expected input format coming from validator prompts.
- Latency budget for returning predictions.
- Metrics you will monitor to judge live performance.

## 4. Customize the Chute Template
Edit the files inside `scorevision/miner/open_source/chute_template/` to load and serve your model:
- `setup.py` – install dependencies and fetch artifacts.
- `load.py` – initialize model weights and supporting assets.
- `predict.py` – handle inference requests from the validator runner.

## 5. Ship to Hugging Face and Chutes
Push your model artifacts and deploy them to the live miner:

```bash
sv -vv push --model-path <path_to_model_assets> --element-id <element_id>
```

Optional flags:

- `--revision <sha-or-branch>` to target a specific Hugging Face revision.
- `--no-deploy` to skip Chutes deployment (HF only).
- `--no-commit` to skip on-chain commitment (prints payload only).

Uploads are rate-limited—plan your iterations accordingly.

## 6. Check Your Registration
Inspect element windows and your commitments:

```bash
sv miner elements --window current
sv miner commitments list --source both
```

Investigate failures via the Chutes dashboard. Copy the instance ID from **Statistics** and fetch logs with:

```bash
curl -X GET "https://api.chutes.ai/instances/<CHUTE_INSTANCE_ID>/logs" \
  -H "Authorization: <CHUTES_API_KEY>"
```

## 7. Maintain and Iterate
- Track validator scores and feedback to tune your model.
- Rebuild and redeploy when new datasets or evaluation hints land.

When these steps are complete, your miner is eligible to answer live challenges and earn rewards on Turbo Vision.
