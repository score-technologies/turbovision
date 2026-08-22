from __future__ import annotations

from logging import getLogger
from os import environ

logger = getLogger(__name__)

# Credentials that must never be visible to a process that loads miner code.
SECRET_ENV_PREFIXES = (
    "CHECKER_R2_",
    "CENTRAL_R2_",
    "PRIVATE_RESPONSES_R2_",
    "AUDIT_R2_",
    "R2_",
    "AWS_",
)
SECRET_ENV_NAMES = (
    "SIGNER_SEED",
    "HF_TOKEN",
    "HUGGINGFACE_API_KEY",
    "HUGGINGFACE_HUB_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "CHUTES_API_KEY",
    "CHUTES_HF_TOKEN",
    "OPENROUTER_API_KEY",
)


def is_secret_env_name(name: str) -> bool:
    return name in SECRET_ENV_NAMES or name.startswith(SECRET_ENV_PREFIXES)


def secret_env_names() -> list[str]:
    return sorted(name for name in environ if is_secret_env_name(name))


def scrub_secret_env() -> list[str]:
    """Drop credentials from os.environ.

    Call this once settings are loaded and before any child process is spawned:
    a spawned child inherits os.environ as it is at spawn time, so scrubbing here
    keeps miner code from reading the bucket keys out of its own environment.
    """
    removed = secret_env_names()
    for name in removed:
        environ.pop(name, None)
    if removed:
        logger.info("[secret-env] scrubbed %d credential variables", len(removed))
    return removed


def assert_no_secret_env(context: str) -> None:
    """Fail closed when credentials are still reachable from an untrusted process."""
    leaked = secret_env_names()
    if leaked:
        raise RuntimeError(f"secret_env_visible:{context}:{','.join(leaked)}")
