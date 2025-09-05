import click
from asyncio import run
from pathlib import Path
from logging import getLogger, DEBUG, INFO, WARNING, basicConfig
import asyncio

from scorevision.cli.runner import runner_loop
from scorevision.cli.push import push_ml_model
from scorevision.utils.settings import get_settings
from scorevision.utils.bittensor_helpers import test_metagraph
from scorevision.cli.signer_api import run_signer
from scorevision.cli.validate import _validate_main
from scorevision.utils.prometheus import _start_metrics
from scorevision.chute_template.test import (
    deploy_mock_chute,
    test_chute_health_endpoint,
    test_chute_predict_endpoint,
    get_chute_logs,
)
from scorevision.utils.chutes_helpers import (
    render_chute_template,
    get_chute_slug_and_id,
    delete_chute,
)
from scorevision.cli.run_vlm_pipeline import vlm_pipeline

logger = getLogger(__name__)


@click.group(name="sv")
@click.option(
    "-v",
    "--verbosity",
    count=True,
    help="Increase verbosity (-v INFO, -vv DEBUG)",
)
def cli(verbosity: int):
    """Score Vision CLI"""
    settings = get_settings()
    basicConfig(
        level=DEBUG if verbosity == 2 else INFO if verbosity == 1 else WARNING,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.debug(f"Score Vision started (version={settings.SCOREVISION_VERSION})")


@cli.command("runner")
def runner_cmd():
    """Launches runner every TEMPO blocks."""
    asyncio.run(runner_loop())


@cli.command("push")
@click.option(
    "--model-path",
    help="Local path to model artifacts. If none provided, upload skipped",
)
@click.option(
    "--revision",
    default=None,
    help="Explicit revision SHA to commit (otherwise auto-detected).",
)
@click.option(
    "--warmup-url",
    default="https://scoredata.me/chunks/87aa0bba70f444f3a8841f8c214463.mp4",
    help="warmup after deploy.",
)
@click.option("--no-deploy", is_flag=True, help="Skip Chutes deployment (HF only).")
@click.option(
    "--no-commit", is_flag=True, help="Skip on-chain commitment (print payload only)."
)
def push(
    model_path,
    revision,
    warmup_url,
    no_deploy,
    no_commit,
):
    try:
        run(
            push_ml_model(
                ml_model_path=Path(model_path) if model_path else None,
                hf_revision=revision,
                warmup_video_url=warmup_url,
                skip_chutes_deploy=no_deploy,
                skip_bittensor_commit=no_commit,
            )
        )
    except Exception as e:
        click.echo(e)


@cli.command("signer")
def signer_cmd():
    asyncio.run(run_signer())


@cli.command("validate")
@click.option(
    "--tail", type=int, envvar="SCOREVISION_TAIL", default=28800, show_default=True
)
@click.option(
    "--alpha", type=float, envvar="SCOREVISION_ALPHA", default=0.2, show_default=True
)
@click.option(
    "--m-min", type=int, envvar="SCOREVISION_M_MIN", default=25, show_default=True
)
@click.option(
    "--tempo", type=int, envvar="SCOREVISION_TEMPO", default=100, show_default=True
)
def validate_cmd(tail: int, alpha: float, m_min: int, tempo: int):
    """
    ScoreVision validator (mainnet cadence):
      - attend block%tempo==0
      - calcule (uids, weights) winner-takes-all
      - push via signer, fallback local si signer HS
    """
    _start_metrics()
    asyncio.run(_validate_main(tail=tail, alpha=alpha, m_min=m_min, tempo=tempo))


@cli.command("deploy-local-chute")
@click.option("--repo", type=str, default="tmoklc/scorevisionv1", required=True)
@click.option(
    "--revision",
    type=str,
    default="d3fb55db3573c4ff926efa64ba2c3a7479b829d0",
    required=True,
)
def deploy_chute_locally(repo: str, revision: str):
    """Locally deploy your model for testing on localhost:8000: simulating its behaviour when deployed on Chutes"""
    deploy_mock_chute(
        huggingface_repo=repo,
        huggingface_revision=revision,
    )


@cli.command("ping-chute")
@click.option("--local", is_flag=True, help="Use locally deployed mock chute server.")
@click.option(
    "--video-url",
    type=str,
    default="https://scoredata.me/2025_03_14/35ae7a/h1_0f2ca0.mp4",
    required=True,
)
@click.option(
    "--n-frames",
    type=int,
    default=5,
    required=True,
)
@click.option(
    "--revision",
    type=str,
    required=True,
)
def test_chute(revision: str, video_url: str, n_frames: int, local: bool) -> None:
    """Check the response of the model endpoints"""
    if local:
        base_url = "http://localhost:8000"
    else:
        slug, _ = run(get_chute_slug_and_id(revision=revision))
        settings = get_settings()
        base_url = settings.CHUTES_MINER_BASE_URL_TEMPLATE.format(
            slug=slug,
        )
    run(test_chute_health_endpoint(base_url=base_url))
    run(
        test_chute_predict_endpoint(
            base_url=base_url, video_url=video_url, first_n_frames=n_frames
        )
    )


@cli.command("chute-slug")
@click.option(
    "--revision",
    type=str,
    required=True,
)
def query_chute_slug(revision: str) -> None:
    chute_slug, chute_id = run(get_chute_slug_and_id(revision=revision))
    click.echo(f"Slug: {chute_slug}\nID: {chute_id}")


@cli.command("chute-delete")
@click.option(
    "--revision",
    type=str,
    required=True,
)
def delete_model_from_chutes(revision: str) -> None:
    try:
        run(delete_chute(revision=revision))
    except Exception as e:
        click.echo(e)


@cli.command("chute-logs")
@click.option("--instance-id", type=str, required=True)
def chute_logs(instance_id: str) -> None:
    try:
        run(get_chute_logs(instance_id=instance_id))
    except Exception as e:
        click.echo(e)


@cli.command("generate-chute-script")
@click.option(
    "--revision",
    type=str,
    required=True,
)
def generate_chute_file(revision: str) -> None:
    with open("my_chute.py", "w+") as f:
        f.write(
            render_chute_template(
                revision=revision,
            )
        )
        f.flush()


@cli.command("test-chute")
@click.option(
    "--revision",
    type=str,
    required=True,
)
@click.option(
    "--local", is_flag=True, help="Use locally deployed mock chute server for model"
)
def test_vlm_pipeline(revision: str, local: bool) -> None:
    """Run the miner on the VLM-as-Judge pipeline off-chain (results not saved)"""
    try:
        run(vlm_pipeline(hf_revision=revision, local_model=local))
    except Exception as e:
        click.echo(e)


# TODO: remove this later
@cli.command("test-metagraph")
def test_metagraph_cmd():
    run(test_metagraph())


if __name__ == "__main__":
    basicConfig(
        level=INFO,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cli()
