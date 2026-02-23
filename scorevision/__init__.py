import asyncio
from asyncio import run
from logging import getLogger
from pathlib import Path
import click
from scorevision.cli.audit_validator import audit_validator
from scorevision.cli.central_validator import central_validator
from scorevision.cli.elements import elements_cli
from scorevision.cli.manifest import manifest_cli
from scorevision.utils.logging import setup_logging
from scorevision.utils.settings import get_settings

logger = getLogger(__name__)


@click.group(name="sv")
@click.option(
    "-v",
    "--verbosity",
    count=True,
    help="Increase verbosity (-v INFO, -vv DEBUG)",
)
def app(verbosity: int):
    setup_logging(verbosity)
    logger.debug("Score Vision started (version=%s)", get_settings().SCOREVISION_VERSION)


@app.command("runner")
def runner_cmd():
    from scorevision.validator.central import runner_loop
    from scorevision.utils.prometheus import _start_metrics, mark_service_ready
    setup_logging()

    _start_metrics()
    mark_service_ready("runner")
    asyncio.run(runner_loop(path_manifest=None))


@app.command("push")
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
@click.option(
    "--element-id",
    required=False,
    default=None,
    help="Element ID to commit. If omitted, sv push reads the manifest and prompts you to choose.",
)
def push(
    model_path,
    revision,
    no_deploy,
    no_commit,
    element_id,
):
    from scorevision.cli.push import push_ml_model
    setup_logging()

    try:
        run(
            push_ml_model(
                ml_model_path=Path(model_path) if model_path else None,
                hf_revision=revision,
                skip_chutes_deploy=no_deploy,
                skip_bittensor_commit=no_commit,
                element_id=element_id,
            )
        )
    except Exception as e:
        click.echo(e)


@app.command("signer")
def signer_cmd():
    from scorevision.validator.core import run_signer
    setup_logging()

    asyncio.run(run_signer())


@app.command("validate")
@click.option(
    "--tail", type=int, envvar="SCOREVISION_TAIL", default=28800, show_default=True
)
@click.option(
    "--m-min", type=int, envvar="SCOREVISION_M_MIN", default=25, show_default=True
)
@click.option(
    "--tempo", type=int, envvar="SCOREVISION_TEMPO", default=100, show_default=True
)
@click.option(
    "--manifest-path", type=click.Path(exists=True, dir_okay=False), default=None
)
def validate_cmd(tail: int, m_min: int, tempo: int, manifest_path):
    from scorevision.validator.core import weights_loop
    from scorevision.utils.prometheus import _start_metrics, mark_service_ready
    setup_logging()

    _start_metrics()
    mark_service_ready("validator")
    path_manifest = Path(manifest_path) if manifest_path else None
    asyncio.run(
        weights_loop(
            tail=tail,
            m_min=m_min,
            tempo=tempo,
            path_manifest=path_manifest,
            commit_on_start=False,
        )
    )


app.add_command(audit_validator)
app.add_command(central_validator)
app.add_command(manifest_cli)
app.add_command(elements_cli)
