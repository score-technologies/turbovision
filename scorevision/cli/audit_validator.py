import asyncio
import logging
import multiprocessing
import signal
import click
from scorevision.utils.logging import setup_logging
from scorevision.utils.settings import get_settings

logger = logging.getLogger(__name__)

shutdown_event = asyncio.Event()


def setup_signal_handlers():
    def handler(signum, frame):
        logger.warning("Received shutdown signal, stopping audit validator...")
        shutdown_event.set()

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


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


def run_spotcheck_process(
    min_interval: int,
    max_interval: int,
    tail_blocks: int,
    threshold: float,
    element_id: str | None,
    commit_on_start: bool,
):
    from scorevision.validator.audit import spotcheck_loop
    setup_logging()

    asyncio.run(spotcheck_loop(
        min_interval_seconds=min_interval,
        max_interval_seconds=max_interval,
        tail_blocks=tail_blocks,
        threshold=threshold,
        element_id=element_id,
        commit_on_start=commit_on_start,
    ))


def run_signer_process():
    from scorevision.validator.core import run_signer
    setup_logging()

    asyncio.run(run_signer())


@click.group("audit-validator")
def audit_validator():
    pass


@audit_validator.command("start")
@click.option("--tail", default=28800, help="Tail blocks for data fetching")
@click.option("--m-min", default=25, help="Minimum samples per miner")
@click.option("--tempo", default=100, help="Weights loop tempo in blocks")
@click.option("--manifest", default=None, type=click.Path(exists=True), help="Path to manifest file")
@click.option("--spotcheck-min", default=None, type=int, help="Min spotcheck interval in seconds")
@click.option("--spotcheck-max", default=None, type=int, help="Max spotcheck interval in seconds")
@click.option("--threshold", default=None, type=float, help="Spotcheck match threshold (0.0-1.0)")
@click.option("--element-id", default=None, help="Filter spotcheck to specific element")
@click.option("--no-commit", is_flag=True, help="Skip on-chain audit index commitment on startup")
def start_cmd(
    tail: int,
    m_min: int,
    tempo: int,
    manifest: str | None,
    spotcheck_min: int | None,
    spotcheck_max: int | None,
    threshold: float | None,
    element_id: str | None,
    no_commit: bool,
):
    settings = get_settings()

    if spotcheck_min is None:
        spotcheck_min = settings.AUDIT_SPOTCHECK_MIN_INTERVAL_S
    if spotcheck_max is None:
        spotcheck_max = settings.AUDIT_SPOTCHECK_MAX_INTERVAL_S
    if threshold is None:
        threshold = settings.AUDIT_SPOTCHECK_THRESHOLD

    setup_signal_handlers()

    logger.info("Starting audit validator with three processes")
    logger.info("  Signer service: starting...")
    logger.info("  Weights loop: tempo=%d blocks, tail=%d, m_min=%d", tempo, tail, m_min)
    logger.info("  Spotcheck loop: interval=%d-%d seconds, threshold=%.0f%%", spotcheck_min, spotcheck_max, threshold * 100)
    if no_commit:
        logger.info("  Spotcheck startup commitment: disabled (--no-commit)")

    signer_proc = multiprocessing.Process(
        target=run_signer_process,
        name="audit-signer",
    )
    weights_proc = multiprocessing.Process(
        target=run_weights_process,
        args=(tail, m_min, tempo, manifest),
        name="audit-weights",
    )
    spotcheck_proc = multiprocessing.Process(
        target=run_spotcheck_process,
        args=(spotcheck_min, spotcheck_max, tail, threshold, element_id, not no_commit),
        name="audit-spotcheck",
    )

    signer_proc.start()
    weights_proc.start()
    spotcheck_proc.start()

    logger.info(
        "All processes started (signer pid=%d, weights pid=%d, spotcheck pid=%d)",
        signer_proc.pid,
        weights_proc.pid,
        spotcheck_proc.pid,
    )

    try:
        signer_proc.join()
        weights_proc.join()
        spotcheck_proc.join()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, terminating processes...")
        signer_proc.terminate()
        weights_proc.terminate()
        spotcheck_proc.terminate()
        signer_proc.join(timeout=5)
        weights_proc.join(timeout=5)
        spotcheck_proc.join(timeout=5)

    logger.info("Audit validator shutdown complete")


@audit_validator.command("weights")
@click.option("--tail", default=28800, help="Tail blocks for data fetching")
@click.option("--m-min", default=25, help="Minimum samples per miner")
@click.option("--tempo", default=100, help="Weights loop tempo in blocks")
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


@audit_validator.command("spotcheck")
@click.option("--min-interval", default=None, type=int, help="Min interval between spotchecks (seconds)")
@click.option("--max-interval", default=None, type=int, help="Max interval between spotchecks (seconds)")
@click.option("--tail", default=28800, help="Tail blocks for data fetching")
@click.option("--threshold", default=None, type=float, help="Match threshold (0.0-1.0)")
@click.option("--element-id", default=None, help="Filter to specific element")
@click.option("--once", is_flag=True, help="Run single spotcheck and exit")
@click.option("--mock-data-dir", default=None, type=click.Path(exists=True), help="Load mock data from local directory (testing only)")
@click.option("--no-commit", is_flag=True, help="Skip on-chain audit index commitment on startup")
def spotcheck_cmd(
    min_interval: int | None,
    max_interval: int | None,
    tail: int,
    threshold: float | None,
    element_id: str | None,
    once: bool,
    mock_data_dir: str | None,
    no_commit: bool,
):
    from pathlib import Path
    from scorevision.validator.audit import spotcheck_loop, run_single_spotcheck
    setup_logging()

    settings = get_settings()
    mock_path = Path(mock_data_dir) if mock_data_dir else None

    if mock_path is None:
        public_url = settings.SCOREVISION_PUBLIC_RESULTS_URL
        if not public_url:
            logger.error("SCOREVISION_PUBLIC_RESULTS_URL is not set")
            return

    if min_interval is None:
        min_interval = settings.AUDIT_SPOTCHECK_MIN_INTERVAL_S
    if max_interval is None:
        max_interval = settings.AUDIT_SPOTCHECK_MAX_INTERVAL_S
    if threshold is None:
        threshold = settings.AUDIT_SPOTCHECK_THRESHOLD

    if once or mock_path is not None:
        if mock_path:
            logger.info("Running spotcheck with mock data from: %s", mock_path)
        else:
            logger.info("Running single spotcheck (tail=%d, element=%s, threshold=%.0f%%)", tail, element_id or "any", threshold * 100)
        asyncio.run(run_single_spotcheck(
            tail_blocks=tail,
            element_id=element_id,
            threshold=threshold,
            mock_data_dir=mock_path,
            commit_on_start=not no_commit,
        ))
    else:
        logger.info("Starting spotcheck loop (interval=%d-%d seconds)", min_interval, max_interval)
        asyncio.run(spotcheck_loop(
            min_interval_seconds=min_interval,
            max_interval_seconds=max_interval,
            tail_blocks=tail,
            threshold=threshold,
            element_id=element_id,
            commit_on_start=not no_commit,
        ))


@audit_validator.command("signer")
def signer_cmd():
    from scorevision.validator.core import run_signer
    setup_logging()

    logger.info("Starting signer-only mode")
    asyncio.run(run_signer())
