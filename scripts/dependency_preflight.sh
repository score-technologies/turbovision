#!/usr/bin/env bash
set -euo pipefail

# Preflight build/check for dependency regressions before full compose rebuild.
# Usage examples:
#   scripts/dependency_preflight.sh
#   scripts/dependency_preflight.sh --pull --no-cache

IMAGE_TAG="scorevision:dep-preflight"
DOCKERFILE="Dockerfile"
CONTEXT_DIR="."

BUILD_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --pull|--no-cache)
      BUILD_ARGS+=("$arg")
      ;;
    *)
      echo "Unsupported arg: $arg"
      echo "Allowed args: --pull --no-cache"
      exit 2
      ;;
  esac
done

echo "[preflight] Building test image: ${IMAGE_TAG}"
docker build "${BUILD_ARGS[@]}" -f "${DOCKERFILE}" -t "${IMAGE_TAG}" "${CONTEXT_DIR}"

echo "[preflight] Running pip dependency integrity check"
set +e
PIP_CHECK_OUTPUT="$(docker run --rm --entrypoint python "${IMAGE_TAG}" -m pip check 2>&1)"
PIP_CHECK_EXIT=$?
set -e

if [ "${PIP_CHECK_EXIT}" -ne 0 ]; then
  FILTERED="$(printf '%s\n' "${PIP_CHECK_OUTPUT}" | sed '/^nvidia-cusparselt-cu13 .* is not supported on this platform$/d')"
  if [ -z "${FILTERED}" ]; then
    echo "[preflight] Warning ignored: nvidia-cusparselt-cu13 unsupported on current platform"
  else
    echo "[preflight] pip check FAILED:"
    printf '%s\n' "${FILTERED}"
    exit 1
  fi
else
  printf '%s\n' "${PIP_CHECK_OUTPUT}"
fi

echo "[preflight] Verifying critical pinned package versions"
docker run --rm --entrypoint python "${IMAGE_TAG}" - <<'PY'
import importlib.metadata as md
import sys

expected = {
    "bittensor": "9.12.0",
    "async-substrate-interface": "1.6.4",
    "scalecodec": "1.2.11",
    "bt-decode": "0.8.0",
}

errors = []
for name, wanted in expected.items():
    try:
        got = md.version(name)
    except md.PackageNotFoundError:
        errors.append(f"MISSING: {name} expected {wanted}")
        continue
    if got != wanted:
        errors.append(f"MISMATCH: {name} expected {wanted}, got {got}")

if errors:
    print("[preflight] Critical version check FAILED:")
    for item in errors:
        print(f"  - {item}")
    sys.exit(1)

print("[preflight] Critical versions OK")
PY

echo "[preflight] Snapshot of key package versions"
docker run --rm --entrypoint python "${IMAGE_TAG}" - <<'PY'
import importlib.metadata as md

keys = [
    "bittensor",
    "async-substrate-interface",
    "scalecodec",
    "bt-decode",
    "torch",
    "numpy",
    "opencv-python",
    "pydantic",
    "fastapi",
    "uvicorn",
]

for name in keys:
    try:
        print(f"{name}=={md.version(name)}")
    except md.PackageNotFoundError:
        print(f"{name}==<missing>")
PY

echo "[preflight] SUCCESS: no dependency conflict detected"
