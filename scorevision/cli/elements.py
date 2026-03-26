import click
from json import dumps


@click.group(name="elements")
def elements_cli():
    """Inspect supported ScoreVision elements."""
    pass


@elements_cli.command("list")
def list_available_elements():
    """List supported elements and their registered pillar coverage."""
    from scorevision.utils.pillar_metric_registry import (
        element_pillar_registry_availability,
    )

    click.secho(dumps(element_pillar_registry_availability(), indent=2), fg="green")
