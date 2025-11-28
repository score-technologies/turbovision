import click
from json import dumps

from scorevision.utils.pillar_metric_registry import (
    element_pillar_registry_availability,
)


@click.group(name="elements")
def elements_cli():
    pass


@elements_cli.command("list")
def list_available_elements():
    """Create a new manifest from a template."""
    click.secho(dumps(element_pillar_registry_availability(), indent=2), fg="green")
