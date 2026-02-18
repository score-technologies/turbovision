from rich.console import Console

console = Console()


def info(message: str) -> None:
    console.print(f"[bold blue]{message}[/bold blue]")


def success(message: str) -> None:
    console.print(f"[green]✓[/green] {message}")


def error(message: str) -> None:
    console.print(f"[red]✗[/red] {message}")


def warn(message: str) -> None:
    console.print(f"[yellow]⊘[/yellow] {message}")


def done() -> None:
    console.print("[bold green]Done![/bold green]")


