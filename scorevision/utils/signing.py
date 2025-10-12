from json import loads, dumps
from time import time
import base64

from aiohttp import ClientTimeout, ClientSession
from substrateinterface import Keypair

from scorevision.utils.settings import get_settings


async def _sign_batch(payloads: list[str]) -> tuple[str, list[str]]:
    """ """
    settings = get_settings()
    if settings.SIGNER_URL:
        try:
            timeout = ClientTimeout(connect=2, total=30)
            async with ClientSession(timeout=timeout) as sess:
                r = await sess.post(
                    f"{settings.SIGNER_URL}/sign", json={"payloads": payloads}
                )
                txt = await r.text()
                if r.status == 200:
                    data = loads(txt)
                    sigs = data.get("signatures") or []
                    hk = data.get("hotkey") or ""
                    if len(sigs) == len(payloads) and hk:
                        return hk, sigs
        except Exception as e:
            raise Exception(f"signer unavailable, fallback to local: {e}")
    raise ValueError("No Signer URL set")


def sign_message(keypair: Keypair, message: str | None) -> str | None:
    if message is None:
        return None
    return f"0x{keypair.sign(message).hex()}"


def create_validator_auth_headers(keypair: Keypair) -> dict[str, str]:
    """
    Create authentication headers for ScoreVision API requests.
    
    Args:
        keypair: The validator's keypair for signing
        
    Returns:
        Dictionary containing the Authorization header with base64-encoded JSON auth data
    """
    settings = get_settings()
    
    nonce = str(int(time() * 1e9))
    signature = sign_message(keypair, nonce)
    
    # Create auth data as expected by the API
    auth_data = {
        "validator-hotkey": keypair.ss58_address,
        "signature": signature,
        "nonce": nonce,
        "netuid": str(settings.SCOREVISION_NETUID)
    }
    
    # Encode as JSON and base64 encode for the Bearer token
    auth_token = base64.b64encode(dumps(auth_data).encode()).decode()
    
    return {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }
