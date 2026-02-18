from contextlib import asynccontextmanager
from fastapi import FastAPI
from scorevision.miner.private_track.logging import log_startup_config
from scorevision.miner.private_track.routes import handle_challenge
from scorevision.miner.private_track.security import get_security_dependencies
from scorevision.utils.schemas import ChallengeRequest, ChallengeResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    log_startup_config()
    yield


app = FastAPI(title="Private Track Turbovision Miner", lifespan=lifespan)


@app.post(
    "/challenge",
    response_model=ChallengeResponse,
    dependencies=get_security_dependencies(),
)
async def challenge_endpoint(request: ChallengeRequest) -> ChallengeResponse:
    return await handle_challenge(request)
