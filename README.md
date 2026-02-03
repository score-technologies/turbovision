# TurboVision

TurboVision is Score's decentralized intelligence layer for live video and imagery. The network pairs expert models with a global community of validators and miners so raw footage becomes structured, decision-ready data in real time, with early deployments focused on professional sports.

## Participate

- Validators keep the network honest by scoring submitted models on live data. Read [VALIDATOR.md](VALIDATOR.md).
- Miners contribute models that solve Elements in the Manifest. Read [MINER.md](MINER.md).

## Common Setup

### Bittensor Wallet

Install the CLI, create a coldkey and hotkey, then copy the hotkey folder and public coldkey (`coldkeypub.txt`) onto every host that will run TurboVision.

```bash
pip install bittensor-cli
btcli wallet new_coldkey --n_words 24 --wallet.name my-wallet
btcli wallet new_hotkey --wallet.name my-wallet --n_words 24 --wallet.hotkey my-hotkey
cp env.example .env
```

Set these in `.env`:

- `BITTENSOR_WALLET_COLD`
- `BITTENSOR_WALLET_HOT`
- `CHUTES_API_KEY`
- `HUGGINGFACE_USERNAME`
- `HUGGINGFACE_API_KEY`

### Chutes Access

Upgrade to a developer-enabled account on [chutes.ai](https://chutes.ai), install the CLI, register, and mint an API key. Store it as `CHUTES_API_KEY`.

```bash
pip install -U chutes
chutes register
```

### Hugging Face Credentials

Create (or reuse) a Hugging Face account, generate a token with write access, and set `HUGGINGFACE_USERNAME` and `HUGGINGFACE_API_KEY`.

## ScoreVision CLI

Install the CLI with `uv`, then sync dependencies and verify the binary:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv && source .venv/bin/activate
uv sync
sv
```

### Runner

The runner executes scoring jobs per Element on a fixed block cadence. It fetches a challenge, builds ground truth (real or pseudo), scores miners, and emits results to R2.

```bash
sv -vv runner
```

Flow (per Element run):

1) Load the active Manifest.
2) Pull a challenge for the current `element_id`.
3) Determine the window ID and start block for timing metadata.
4) Build ground truth:
   - If `elements[].ground_truth=true`, fetch real GT from the API.
   - Otherwise generate pseudo-GT locally (SAM3).
5) Call eligible miners (from on-chain registry) and score outputs.
6) Emit a shard to R2 with evaluation payload + metadata.

Scheduling:

- Runner keeps per-element timers based on `elements[].window_block` (or `tempo`).
- If an Element does not define a cadence, it uses `SV_DEFAULT_ELEMENT_TEMPO_BLOCKS` (default `300`).
- When the manifest changes, runner rebuilds its per-element schedule.

Quality gates:

- Pseudo-GT is retried until enough frames meet bbox thresholds.
- Tune retries with `SV_PGT_MAX_BBOX_RETRIES` and `SV_PGT_MAX_QUALITY_RETRIES`.

### Validator

The validator aggregates recent scores per Element, chooses winners, and submits weights on-chain via the signer service.

```bash
sv -vv validate --manifest-path path/to/manifest.yml
```

Flow (per window):

1) Load the active Manifest for the current block.
2) For each Element:
   - Determine how far back to look using `elements[].eval_window` (days).
   - Pull recent scores for that Element and window.
   - Select the winner and build a per-element weight share.
3) Normalize total weights to 1.0 (add fallback weight if needed).
4) Submit `uids` + `weights` to the signer.
5) Optionally snapshot winners to R2.

Behavior knobs:

- `elements[].eval_window` controls lookback in days (converted to blocks).
- `elements[].weight` controls the Element share of final weights.
- Hotkey blacklist is loaded from `manifest_update/blacklist` (one hotkey per line, `#` for comments).
- Startup commit: `SCOREVISION_COMMIT_VALIDATOR_ON_START=1` publishes the validator index.
- Snapshot cadence: `SCOREVISION_WINNERS_EVERY` (every N successful loops).

## Community & Support

- [Score Website](https://wearescore.com)
- File issues or ideas in this repo.
