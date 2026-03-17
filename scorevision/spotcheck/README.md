# Spotcheck System

Automated verification of private track miner Docker images for Subnet 44.

## Architecture

The spotcheck system has two components that run as Kubernetes workloads:

### Orchestrator (`orchestrator/`)

A CronJob that:
1. Fetches pending spotchecks from the central API
2. Filters out deregistered miners and scoring version mismatches
3. For each target, launches a miner Job (using the exact Docker image digest from original scoring) and a checker Job
4. Reports pass/fail results back to the blacklist API

### Runner (`runner/`)

The checker container that:
1. Sends a video to the miner's inference endpoint
2. Compares the miner's predictions against ground truth using the shared scoring mechanism
3. Outputs a PASS/FAIL result with the score comparison


## Configuration

All environment variables and their defaults are defined in `orchestrator/config.py` (`SpotcheckConfig`). Sensitive values (`SPOTCHECK_AUTH_TOKEN`, `GHCR_SECRET`) are stored as Kubernetes Secrets and injected via `envFrom` or `valueFrom.secretKeyRef` in the CronJob/Job manifests.

## Docker Images

Build from the turbovision repo root:

```bash
docker build --platform linux/amd64 -f scorevision/spotcheck/orchestrator/Dockerfile -t spotcheck-orchestrator .
docker build --platform linux/amd64 -f scorevision/spotcheck/runner/Dockerfile -t spotcheck-runner .
```
