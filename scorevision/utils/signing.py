from json import loads

from aiohttp import ClientTimeout, ClientSession
from substrateinterface import Keypair

from scorevision.utils.settings import get_settings


async def _sign_batch(payloads: list[str]) -> tuple[str, list[str]]:
    """
    Retourne (hotkey_hex, signatures_hex[]).
    1) essaie SIGNER_URL /sign
    2) fallback: SIGN_SEED local
    """
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
