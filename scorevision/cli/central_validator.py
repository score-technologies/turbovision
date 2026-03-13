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


def start_and_manage_processes(procs: list[multiprocessing.Process]):
    for proc in procs:
        proc.start()

    pid_summary = ", ".join(f"{proc.name} pid={proc.pid}" for proc in procs)
    logger.info("All processes started (%s)", pid_summary)

    try:
        for proc in procs:
            proc.join()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, terminating processes...")
        for proc in procs:
            proc.terminate()
        for proc in procs:
            proc.join(timeout=5)

    logger.info("Central validator shutdown complete")


def run_signer_process():
    from scorevision.validator.core import run_signer
    setup_logging()

    asyncio.run(run_signer())


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


def run_os_runner_process(path_manifest: str | None):
    from pathlib import Path
    from scorevision.validator.central import runner_loop
    setup_logging()

    manifest_path = Path(path_manifest) if path_manifest else None
    asyncio.run(runner_loop(path_manifest=manifest_path))


def run_pt_runner_process():
    from scorevision.validator.central.private_track.runner import run_challenge_process
    setup_logging()

    run_challenge_process()


def run_pt_spotcheck_process():
    from scorevision.validator.central.private_track.runner import run_spotcheck_process
    setup_logging()

    run_spotcheck_process()


@click.group("central-validator")
def central_validator():
    pass


@central_validator.group("open-source")
def open_source():
    pass


@open_source.command("start")
@click.option("--tail", default=28800, help="Tail blocks for data fetching")
@click.option("--m-min", default=25, help="Minimum samples per miner")
@click.option("--tempo", default=150, help="Weights loop tempo in blocks")
@click.option("--manifest", default=None, type=click.Path(exists=True), help="Path to manifest file")
def os_start_cmd(tail: int, m_min: int, tempo: int, manifest: str | None):
    setup_signal_handlers()

    logger.info("Starting open-source central validator")
    logger.info("  Signer service: starting...")
    logger.info("  Runner loop: manifest=%s", manifest or "auto")
    logger.info("  Weights loop: tempo=%d blocks, tail=%d, m_min=%d", tempo, tail, m_min)

    start_and_manage_processes([
        multiprocessing.Process(target=run_signer_process, name="os-signer"),
        multiprocessing.Process(target=run_os_runner_process, args=(manifest,), name="os-runner"),
        multiprocessing.Process(target=run_weights_process, args=(tail, m_min, tempo, manifest), name="os-weights"),
    ])


@open_source.command("runner")
@click.option("--manifest", default=None, type=click.Path(exists=True), help="Path to manifest file")
def os_runner_cmd(manifest: str | None):
    setup_logging()
    run_os_runner_process(manifest)


@open_source.command("weights")
@click.option("--tail", default=28800, help="Tail blocks for data fetching")
@click.option("--m-min", default=25, help="Minimum samples per miner")
@click.option("--tempo", default=150, help="Weights loop tempo in blocks")
@click.option("--manifest", default=None, type=click.Path(exists=True), help="Path to manifest file")
def os_weights_cmd(tail: int, m_min: int, tempo: int, manifest: str | None):
    setup_logging()
    run_weights_process(tail, m_min, tempo, manifest)


@open_source.command("signer")
def os_signer_cmd():
    setup_logging()
    run_signer_process()


@central_validator.group("private-track")
def private_track():
    pass


@private_track.command("runner")
def pt_runner_cmd():
    setup_logging()
    run_pt_runner_process()


@private_track.command("spotcheck")
def pt_spotcheck_cmd():
    setup_logging()
    run_pt_spotcheck_process()


@private_track.command("weights")
@click.option("--tail", default=28800, help="Tail blocks for data fetching")
@click.option("--m-min", default=25, help="Minimum samples per miner")
@click.option("--tempo", default=100, help="Weights loop tempo in blocks")
@click.option("--manifest", default=None, type=click.Path(exists=True), help="Path to manifest file")
def pt_weights_cmd(tail: int, m_min: int, tempo: int, manifest: str | None):
    setup_logging()
    run_weights_process(tail, m_min, tempo, manifest)


@private_track.command("signer")
def pt_signer_cmd():
    setup_logging()
    run_signer_process()


@central_validator.command("start")
@click.option("--tail", default=28800, help="Tail blocks for data fetching")
@click.option("--m-min", default=25, help="Minimum samples per miner")
@click.option("--tempo", default=150, help="Weights loop tempo in blocks")
@click.option("--manifest", default=None, type=click.Path(exists=True), help="Path to manifest file")
def start_all_cmd(tail: int, m_min: int, tempo: int, manifest: str | None):
    setup_signal_handlers()

    logger.info("Starting unified central validator (open-source + private-track)")
    logger.info("  Signer service: starting...")
    logger.info("  Runner loop (open-source): manifest=%s", manifest or "auto")
    logger.info("  Weights loop (both tracks): tempo=%d blocks, tail=%d, m_min=%d", tempo, tail, m_min)
    logger.info("  Private-track challenge runner: starting...")
    logger.info("  Private-track spotcheck loop: starting...")

    start_and_manage_processes([
        multiprocessing.Process(target=run_signer_process, name="signer"),
        multiprocessing.Process(target=run_os_runner_process, args=(manifest,), name="os-runner"),
        multiprocessing.Process(target=run_weights_process, args=(tail, m_min, tempo, manifest), name="weights"),
        multiprocessing.Process(target=run_pt_runner_process, name="pt-runner"),
        multiprocessing.Process(target=run_pt_spotcheck_process, name="pt-spotcheck"),
    ])
