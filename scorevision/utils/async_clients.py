from aiohttp import ClientSession, ClientTimeout, TCPConnector
from asyncio import get_running_loop, run, get_event_loop, Semaphore

from scorevision.utils.settings import get_settings

_SESSIONS: dict[int, ClientSession] = {}
_SEMAPHORES: dict[int, Semaphore] = {}


async def _close_all_clients_async():
    for sess in list(_SESSIONS.values()):
        try:
            if sess and not sess.closed:
                await sess.close()
        except Exception:
            pass
    _SESSIONS.clear()


def close_http_clients():
    """ """
    try:
        loop = get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        loop.create_task(_close_all_clients_async())
    else:
        run(_close_all_clients_async())


def _loop_key() -> int:
    try:
        loop = get_running_loop()
    except RuntimeError:
        loop = get_event_loop()
    return id(loop)


async def get_async_client() -> ClientSession:
    settings = get_settings()
    key = _loop_key()
    sess = _SESSIONS.get(key)
    if sess is None or sess.closed:
        sess = ClientSession(
            timeout=ClientTimeout(total=settings.SCOREVISION_API_TIMEOUT_S),
            connector=TCPConnector(
                limit=0,
                limit_per_host=0,
            ),
        )
        _SESSIONS[key] = sess
    return sess


def get_semaphore() -> Semaphore:
    settings = get_settings()
    key = _loop_key()
    sem = _SEMAPHORES.get(key)
    if sem is None:
        cap = max(1, settings.SCOREVISION_MAX_CONCURRENT_API_CALLS)
        sem = Semaphore(cap)
        _SEMAPHORES[key] = sem
    return sem


# @asynccontextmanager
# async def create_async_session():
#     settings = get_settings()
#     connector = TCPConnector(
#         limit=settings.SCOREVISION_MAX_CONCURRENT_API_CALLS * 2,
#         limit_per_host=settings.SCOREVISION_MAX_CONCURRENT_API_CALLS,
#     )
#     session = ClientSession(
#         timeout=ClientTimeout(total=settings.SCOREVISION_API_TIMEOUT_S),
#         connector=connector,
#     )
#     try:
#         yield session
#     finally:
#         await session.close()
