import asyncio
import logging
import multiprocessing
import signal
import click
from scorevision.utils.logging import setup_logging

logger = logging.getLogger(__name__)

shutdown_event = asyncio.Event()


def setup_signal_handlers():
    def handler(signum, frame):
        logger.warning("Received shutdown signal, stopping central validator...")
        shutdown_event.set()

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def run_runner_process(path_manifest: str | None):
    from pathlib import Path
    from scorevision.validator.central import runner_loop
    setup_logging()

    manifest_path = Path(path_manifest) if path_manifest else None
    asyncio.run(runner_loop(path_manifest=manifest_path))


def run_weights_process(tail: int, m_min: int, tempo: int, path_manifest: str | None):
    from pathlib import Path
    from scorevision.validator.core import weights_loop
    setup_logging()

    manifest_path = Path(path_manifest) if path_manifest else None
    asyncio.run(
        weights_loop(
            tail=tail,
            m_min=m_min,
            tempo=tempo,
            path_manifest=manifest_path,
            commit_on_start=False,
        )
    )


def run_signer_process():
    from scorevision.validator.core import run_signer
    setup_logging()

    asyncio.run(run_signer())


@click.group("central-validator")
def central_validator():
    pass


@central_validator.command("start")
@click.option("--tail", default=28800, help="Tail blocks for data fetching")
@click.option("--m-min", default=25, help="Minimum samples per miner")
@click.option("--tempo", default=150, help="Weights loop tempo in blocks")
@click.option("--manifest", default=None, type=click.Path(exists=True), help="Path to manifest file")
def start_cmd(tail: int, m_min: int, tempo: int, manifest: str | None):
    setup_signal_handlers()

    logger.info("Starting central validator with three processes")
    logger.info("  Runner loop: manifest=%s", manifest or "auto")
    logger.info("  Weights loop: tempo=%d blocks, tail=%d, m_min=%d", tempo, tail, m_min)
    logger.info("  Signer service: starting...")

    signer_proc = multiprocessing.Process(
        target=run_signer_process,
        name="central-signer",
    )
    runner_proc = multiprocessing.Process(
        target=run_runner_process,
        args=(manifest,),
        name="central-runner",
    )
    weights_proc = multiprocessing.Process(
        target=run_weights_process,
        args=(tail, m_min, tempo, manifest),
        name="central-weights",
    )

    signer_proc.start()
    runner_proc.start()
    weights_proc.start()

    logger.info(
        "All processes started (signer pid=%d, runner pid=%d, weights pid=%d)",
        signer_proc.pid,
        runner_proc.pid,
        weights_proc.pid,
    )

    try:
        signer_proc.join()
        runner_proc.join()
        weights_proc.join()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, terminating processes...")
        signer_proc.terminate()
        runner_proc.terminate()
        weights_proc.terminate()
        signer_proc.join(timeout=5)
        runner_proc.join(timeout=5)
        weights_proc.join(timeout=5)

    logger.info("Central validator shutdown complete")


@central_validator.command("runner")
@click.option("--manifest", default=None, type=click.Path(exists=True), help="Path to manifest file")
def runner_cmd(manifest: str | None):
    from pathlib import Path
    from scorevision.validator.central import runner_loop
    setup_logging()

    manifest_path = Path(manifest) if manifest else None
    logger.info("Starting runner-only mode (manifest=%s)", manifest or "auto")
    asyncio.run(runner_loop(path_manifest=manifest_path))


@central_validator.command("weights")
@click.option("--tail", default=28800, help="Tail blocks for data fetching")
@click.option("--m-min", default=25, help="Minimum samples per miner")
@click.option("--tempo", default=150, help="Weights loop tempo in blocks")
@click.option("--manifest", default=None, type=click.Path(exists=True), help="Path to manifest file")
def weights_cmd(tail: int, m_min: int, tempo: int, manifest: str | None):
    from pathlib import Path
    from scorevision.validator.core import weights_loop
    setup_logging()

    manifest_path = Path(manifest) if manifest else None
    logger.info("Starting weights-only mode (tempo=%d blocks)", tempo)
    asyncio.run(
        weights_loop(
            tail=tail,
            m_min=m_min,
            tempo=tempo,
            path_manifest=manifest_path,
            commit_on_start=False,
        )
    )


@central_validator.command("signer")
def signer_cmd():
    from scorevision.validator.core import run_signer
    setup_logging()

    logger.info("Starting signer-only mode")
    asyncio.run(run_signer())
