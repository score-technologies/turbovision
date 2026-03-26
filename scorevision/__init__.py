import asyncio
from asyncio import run
from dataclasses import dataclass
from importlib import import_module
from logging import getLogger
from pathlib import Path

import click

from scorevision.utils.logging import setup_logging
from scorevision.utils.settings import get_settings

logger = getLogger(__name__)


@dataclass(frozen=True)
class LazyCommandSpec:
    module_path: str
    attribute: str
    short_help: str


LAZY_COMMANDS = {
    "central-validator": LazyCommandSpec(
        module_path="scorevision.cli.central_validator",
        attribute="central_validator",
        short_help="Run central validator commands.",
    ),
    "audit-validator": LazyCommandSpec(
        module_path="scorevision.cli.audit_validator",
        attribute="audit_validator",
        short_help="Run audit validator commands.",
    ),
    "manifest": LazyCommandSpec(
        module_path="scorevision.cli.manifest",
        attribute="manifest_cli",
        short_help="Manage manifests.",
    ),
    "elements": LazyCommandSpec(
        module_path="scorevision.cli.elements",
        attribute="elements_cli",
        short_help="Inspect registered elements.",
    ),
}

ROOT_COMMAND_ORDER = [
    "runner",
    "push",
    "signer",
    "validate",
    "central-validator",
    "audit-validator",
    "manifest",
    "elements",
]


def _load_click_command(spec: LazyCommandSpec) -> click.Command:
    return getattr(import_module(spec.module_path), spec.attribute)


class LazyRootGroup(click.Group):
    """Root Click group that avoids importing every command module for `sv --help`."""

    def list_commands(self, ctx: click.Context) -> list[str]:
        available = set(self.commands) | set(LAZY_COMMANDS)
        return [name for name in ROOT_COMMAND_ORDER if name in available]

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        command = super().get_command(ctx, cmd_name)
        if command is not None:
            return command

        spec = LAZY_COMMANDS.get(cmd_name)
        if spec is None:
            return None

        return _load_click_command(spec)

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        rows: list[tuple[str, str]] = []
        for subcommand in self.list_commands(ctx):
            if subcommand in self.commands:
                cmd = self.commands[subcommand]
                help_text = cmd.get_short_help_str(formatter.width)
            else:
                help_text = LAZY_COMMANDS[subcommand].short_help
            rows.append((subcommand, help_text))

        if rows:
            with formatter.section("Commands"):
                formatter.write_dl(rows)


@click.group(
    name="sv",
    cls=LazyRootGroup,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="ScoreVision CLI for validators, manifests, and element tooling.",
)
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
    """Run the central runner loop."""
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
    """Upload model artifacts and optionally deploy or commit them."""
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
    """Run the signer service."""
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
    "--tempo", type=int, envvar="SCOREVISION_TEMPO", default=150, show_default=True
)
@click.option(
    "--manifest-path", type=click.Path(exists=True, dir_okay=False), default=None
)
def validate_cmd(tail: int, m_min: int, tempo: int, manifest_path):
    """Run the validator weights loop."""
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
