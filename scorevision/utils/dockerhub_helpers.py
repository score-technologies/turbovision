import logging
import httpx

logger = logging.getLogger(__name__)

_DOCKERHUB_API = "https://hub.docker.com/v2"
SCORE_DOCKERHUB_USER = "scorevision"


def get_auth_token(username: str, password: str) -> str | None:
    response = httpx.post(
        f"{_DOCKERHUB_API}/users/login",
        json={"username": username, "password": password},
    )
    if response.status_code != 200:
        logger.error("DockerHub login failed: %s", response.text)
        return None
    return response.json().get("token")


def repo_exists(token: str, namespace: str, repo_name: str) -> bool:
    response = httpx.get(
        f"{_DOCKERHUB_API}/repositories/{namespace}/{repo_name}/",
        headers={"Authorization": f"Bearer {token}"},
    )
    return response.status_code == 200


def create_private_repo(token: str, namespace: str, repo_name: str) -> bool:
    if repo_exists(token, namespace, repo_name):
        logger.info("Repo already exists: %s/%s", namespace, repo_name)
        return True

    response = httpx.post(
        f"{_DOCKERHUB_API}/repositories/",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "namespace": namespace,
            "name": repo_name,
            "is_private": True,
        },
    )
    if response.status_code == 201:
        logger.info("Created private repo: %s/%s", namespace, repo_name)
        return True
    logger.error("Failed to create repo: %s", response.text)
    return False


def add_collaborator(
    token: str,
    namespace: str,
    repo_name: str,
    collaborator: str,
    permission: str = "read",
) -> bool:
    response = httpx.put(
        f"{_DOCKERHUB_API}/repositories/{namespace}/{repo_name}/collaborators/{collaborator}/",
        headers={"Authorization": f"Bearer {token}"},
        json={"permission": permission},
    )
    if response.status_code in (200, 201, 204):
        logger.info("Added %s as %s collaborator to %s/%s", collaborator, permission, namespace, repo_name)
        return True
    logger.error("Failed to add collaborator: %s", response.text)
    return False


def ensure_repo_with_score_access(username: str, password: str, repo_name: str) -> bool:
    token = get_auth_token(username, password)
    if not token:
        return False

    if not create_private_repo(token, username, repo_name):
        return False

    if not add_collaborator(token, username, repo_name, SCORE_DOCKERHUB_USER, "read"):
        return False

    return True
