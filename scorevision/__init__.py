import click
from asyncio import run
from pathlib import Path
from logging import getLogger, DEBUG, INFO, WARNING, basicConfig
import asyncio

from scorevision.cli.runner import runner_loop
from scorevision.cli.push import push_ml_model
from scorevision.utils.settings import get_settings
from scorevision.cli.signer_api import run_signer
from scorevision.cli.validate import _validate_main
from scorevision.utils.prometheus import _start_metrics, mark_service_ready
from scorevision.cli.run_vlm_pipeline import run_vlm_pipeline_once_for_single_miner
from scorevision.cli.miner import miner as miner_cli
from scorevision.cli.manifest import manifest_cli
from scorevision.cli.elements import elements_cli

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


cli.add_command(miner_cli)


@cli.command("runner")
def runner_cmd():
    """Launches runner every TEMPO blocks."""
    _start_metrics()
    mark_service_ready("runner")
    root_dir = Path(__file__).parent.parent
    path_manifest = root_dir / "tests/test_data/manifests/example_manifest.yml"
    asyncio.run(runner_loop(path_manifest=path_manifest))


@cli.command("push")
@click.option(
    "--model-path",
    default=None,
    help="Local path to model artifacts. If none provided, upload skipped",
)
@click.option(
    "--revision",
    default=None,
    help="Explicit revision SHA to commit (otherwise auto-detected).",
)
@click.option("--no-deploy", is_flag=True, help="Skip Chutes deployment (HF only).")
@click.option(
    "--no-commit", is_flag=True, help="Skip on-chain commitment (print payload only)."
)
def push(
    model_path,
    revision,
    no_deploy,
    no_commit,
):
    """Push the miner's ML model stored on Huggingface onto Chutes and commit information on-chain"""
    try:
        run(
            push_ml_model(
                ml_model_path=Path(model_path) if model_path else None,
                hf_revision=revision,
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
    mark_service_ready("validator")
    asyncio.run(_validate_main(tail=tail, alpha=alpha, m_min=m_min, tempo=tempo))


@cli.command("run-once")
@click.option("--revision", type=str, default=None)
@click.option("--path-manifest", type=Path, default=None)
def test_vlm_pipeline(revision: str, path_manifest: Path) -> None:
    """Run the miner on the VLM-as-Judge pipeline off-chain (results not saved)"""
    try:
        if path_manifest is None:
            root_dir = Path(__file__).parent.parent
            path_manifest = root_dir / "tests/test_data/manifests/example_manifest.yml"
            click.echo(f"No manifest path specified. using default: {path_manifest}")
        result = run(
            run_vlm_pipeline_once_for_single_miner(
                path_manifest=path_manifest, hf_revision=revision
            )
        )
        click.echo(result)
    except Exception as e:
        click.echo(e)
        click.echo(e)


cli.add_command(manifest_cli)
cli.add_command(elements_cli)
