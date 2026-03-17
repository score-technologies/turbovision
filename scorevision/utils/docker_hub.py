import logging
import httpx

logger = logging.getLogger(__name__)

GHCR_AUTH_URL = "https://ghcr.io/token"
GHCR_REGISTRY_URL = "https://ghcr.io/v2"
BYTES_PER_GB = 1024 ** 3

MANIFEST_ACCEPT = (
    "application/vnd.docker.distribution.manifest.v2+json, "
    "application/vnd.oci.image.manifest.v1+json, "
    "application/vnd.docker.distribution.manifest.list.v2+json, "
    "application/vnd.oci.image.index.v1+json"
)


async def _get_auth_token(image_repo: str, ghcr_pat: str = "") -> str | None:
    params = {
        "service": "ghcr.io",
        "scope": f"repository:{image_repo}:pull",
    }
    auth = ("token", ghcr_pat) if ghcr_pat else None
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(GHCR_AUTH_URL, params=params, auth=auth)
        resp.raise_for_status()
        return resp.json().get("token")


async def check_image_accessible(image_repo: str, reference: str, ghcr_pat: str = "") -> bool:
    try:
        token = await _get_auth_token(image_repo, ghcr_pat=ghcr_pat)
        if not token:
            return False
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": MANIFEST_ACCEPT,
        }
        url = f"{GHCR_REGISTRY_URL}/{image_repo}/manifests/{reference}"
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.head(url, headers=headers)
            return resp.status_code == 200
    except Exception as exc:
        logger.warning("Image access check failed for %s:%s: %s", image_repo, reference, exc)
        return False


async def fetch_image_digest(image_repo: str, image_tag: str) -> str:
    try:
        token = await _get_auth_token(image_repo)
        if not token:
            return ""
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": MANIFEST_ACCEPT,
        }
        url = f"{GHCR_REGISTRY_URL}/{image_repo}/manifests/{image_tag}"
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.head(url, headers=headers)
            resp.raise_for_status()
            return resp.headers.get("docker-content-digest", "")
    except Exception as exc:
        logger.warning("Failed to fetch digest for %s:%s: %s", image_repo, image_tag, exc)
        return ""


async def fetch_image_size_gb(image_repo: str, reference: str, ghcr_pat: str = "") -> float | None:
    try:
        token = await _get_auth_token(image_repo, ghcr_pat=ghcr_pat)
        if not token:
            return None
        return await _get_manifest_size(image_repo, reference, token)
    except Exception as exc:
        logger.warning("Failed to inspect image %s:%s: %s", image_repo, reference, exc)
        return None


def exceeds_size_limit(size_gb: float | None, max_gb: float) -> bool:
    if size_gb is None:
        return False
    return size_gb > max_gb


async def _get_manifest_size(
    image_repo: str, reference: str, token: str,
) -> float | None:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": MANIFEST_ACCEPT,
    }
    url = f"{GHCR_REGISTRY_URL}/{image_repo}/manifests/{reference}"
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        manifest = resp.json()

    if manifest.get("mediaType") in (
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.index.v1+json",
    ):
        return _sum_manifest_list_size(manifest)

    return _sum_layer_sizes(manifest)


def _sum_manifest_list_size(index: dict) -> float:
    amd64_digest = _find_amd64_digest(index)
    if amd64_digest:
        for m in index.get("manifests", []):
            if m.get("digest") == amd64_digest:
                size_bytes = m.get("size", 0)
                return size_bytes / BYTES_PER_GB
    total = sum(m.get("size", 0) for m in index.get("manifests", []))
    return total / BYTES_PER_GB


def _find_amd64_digest(index: dict) -> str | None:
    for m in index.get("manifests", []):
        platform = m.get("platform", {})
        if platform.get("architecture") == "amd64" and platform.get("os") == "linux":
            return m.get("digest")
    return None


def _sum_layer_sizes(manifest: dict) -> float:
    layers = manifest.get("layers", [])
    total_bytes = sum(layer.get("size", 0) for layer in layers)
    config_size = manifest.get("config", {}).get("size", 0)
    return (total_bytes + config_size) / BYTES_PER_GB
