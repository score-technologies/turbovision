# 🚀 Example Chute for Turbovision 🪂

This repository demonstrates how to deploy a **Chute** via the **Turbovision CLI**, hosted on **Hugging Face Hub**.
It serves as a minimal example showcasing the required structure and workflow for integrating machine learning models, preprocessing, and orchestration into a reproducible Chute environment.

## Repository Structure 
The following two files **must be present** (in their current locations) for a successful deployment — their content can be modified as needed:

| File | Purpose |
|------|----------|
| `miner.py` | Defines the ML model type(s), orchestration, and all pre/postprocessing logic. |
| `config.yml` | Specifies machine configuration (e.g., GPU type, memory, environment variables). |

Other files — e.g., model weights, utility scripts, or dependencies — are **optional** and can be included as needed for your model. Note: Any required assets must be defined or contained **within this repo**, which is fully open-source, since all network-related operations (downloading challenge data, weights, etc.) are disabled **inside the Chute** 

## Overview

Below is a high-level diagram showing the interaction between Huggingface, Chutes and Turbovision:

![](../../../../images/miner.png)

## Local Testing
After editing the `config.yml` and `miner.py` and saving it into your Huggingface Repo, you will want to test it works locally. 

1. Copy the file `scorevision/miner/open_source/chute_template/turbovision_chute.py.j2` as a python file called `my_chute.py` and fill in the missing variables:
```python
HF_REPO_NAME = "{{ huggingface_repository_name }}"
HF_REPO_REVISION = "{{ huggingface_repository_revision }}"
CHUTES_USERNAME = "{{ chute_username }}"
CHUTE_NAME = "{{ chute_name }}"
```

2. Run the following command to build the chute locally (Caution: there are known issues with the docker location when running this on a mac) 
```bash
chutes build my_chute:chute --local --public
```

3. Run the name of the docker image just built (i.e. `CHUTE_NAME`) and enter it
```bash
docker run -p 8000:8000 -e CHUTES_EXECUTION_CONTEXT=REMOTE -it <image-name> /bin/bash
```

4. Run the file from within the container
```bash
chutes run my_chute:chute --dev --debug
```

5. In another terminal, test the local endpoints to ensure there are no bugs
```bash
curl -X POST http://localhost:8000/health -d '{}'
curl -X POST http://localhost:8000/predict -d '{"url": "https://scoredata.me/2025_03_14/35ae7a/h1_0f2ca0.mp4","meta": {}}'
```

## Live Testing
1. If you have any chute with the same name (ie from a previous deployment), ensure you delete that first (or you will get an error when trying to build).
```bash
chutes chutes list
```
Take note of the chute id that you wish to delete (if any)
```bash
chutes chutes delete <chute-id>
```

You should also delete its associated image 
```bash
chutes images list
```
Take note of the chute image id
```bash
chutes images delete <chute-image-id>
```

2. Create a fine-grained Hugging Face token with read-only access to this model
   repository and expose it to the deploy process as `CHUTES_HF_TOKEN`. Keep
   `HF_TOKEN` as the local token with permission to upload and change repository
   visibility.

3. Use Turbovision's CLI to build, deploy and commit on-chain. The repository
   remains private while Chutes loads it using the scoped secret, and becomes
   public only after a successful on-chain commit. You can skip the on-chain
   commit using `--no-commit`; in that case the repository stays private. You
   can also specify a past Hugging Face revision using `--revision` and/or the
   local files to upload using `--model-path`.
```bash
sv -vv deploy-os-miner --element-id <element_id>
```

4. The deployment command warms and health-checks the chute automatically. To
   warm it again later (if it is cold 🧊), use:
```bash
chutes warmup <chute-id>
```

5. Test the chute's endpoints
```bash
curl -X POST https://<YOUR-CHUTE-SLUG>.chutes.ai/health -d '{}' -H "Authorization: Bearer $CHUTES_API_KEY"
curl -X POST https://<YOUR-CHUTE-SLUG>.chutes.ai/predict -d '{"url": "https://scoredata.me/2025_03_14/35ae7a/h1_0f2ca0.mp4","meta": {}}' -H "Authorization: Bearer $CHUTES_API_KEY"
```

6. Test what your chute would get on a validator (this also applies any validation/integrity checks which may fail if you did not use the Turbovision CLI above to deploy the chute)
```bash
sv -vv run-once
```
