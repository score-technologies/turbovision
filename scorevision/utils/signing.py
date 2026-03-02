import asyncio
from json import loads
from time import time_ns

from aiohttp import ClientError, ClientSession, ClientTimeout
from substrateinterface import Keypair

from scorevision.utils.settings import get_settings


async def _sign_batch(payloads: list[str]) -> tuple[str, list[str]]:
    """ """
    settings = get_settings()
    signer_url = (settings.SIGNER_URL or "").rstrip("/")
    if not signer_url:
        raise ValueError("No Signer URL set")

    timeout = ClientTimeout(connect=2, total=10)
    retries = 3
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            async with ClientSession(timeout=timeout) as sess:
                r = await sess.post(f"{signer_url}/sign", json={"payloads": payloads})
                txt = await r.text()
                if r.status != 200:
                    raise RuntimeError(f"signer status={r.status}")

                data = loads(txt)
                sigs = data.get("signatures") or []
                hk = data.get("hotkey") or ""
                if len(sigs) != len(payloads) or not hk:
                    raise RuntimeError("invalid signer response")
                return hk, sigs
        except (asyncio.TimeoutError, ClientError, RuntimeError, ValueError) as e:
            last_error = e
            if attempt < retries:
                await asyncio.sleep(0.2 * attempt)
                continue

    raise Exception(f"signer unavailable, fallback to local: {last_error}")


def sign_message(keypair: Keypair, message: str | None) -> str | None:
    if message is None:
        return None
    return f"0x{keypair.sign(message).hex()}"


def build_validator_query_params(keypair: Keypair) -> dict[str, str]:
    """
    Create validator authentication query parameters required by the ScoreVision API.
    """
    settings = get_settings()

    nonce = str(time_ns())
    signature = sign_message(keypair, nonce)

    return {
        "validator_hotkey": keypair.ss58_address,
        "signature": signature,
        "nonce": nonce,
        "netuid": str(settings.SCOREVISION_NETUID),
    }
