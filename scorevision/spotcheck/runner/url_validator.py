from urllib.parse import urlparse

ALLOWED_DOMAINS: frozenset[str] = frozenset({"scoredata.me"})


def is_video_url_allowed(url: str, allowed_domains: frozenset[str] = ALLOWED_DOMAINS) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False

    if parsed.scheme not in ("http", "https"):
        return False

    hostname = (parsed.hostname or "").lower()
    return hostname in allowed_domains
