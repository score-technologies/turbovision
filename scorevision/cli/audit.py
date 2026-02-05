import asyncio
import logging
import multiprocessing
import signal
import click
from scorevision.utils.settings import get_settings

logger = logging.getLogger("scorevision.audit")

shutdown_event = asyncio.Event()


def setup_signal_handlers():
    def handler(signum, frame):
        logger.warning("Received shutdown signal, stopping audit validator...")
        shutdown_event.set()

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def run_weights_process(tail: int, m_min: int, tempo: int):
    from scorevision.validator.weights import weights_loop
    asyncio.run(weights_loop(tail=tail, m_min=m_min, tempo=tempo))


def run_spotcheck_process(
    min_interval: int,
    max_interval: int,
    tail_blocks: int,
    threshold: float,
):
    from scorevision.validator.spotcheck import spotcheck_loop
    asyncio.run(spotcheck_loop(
        min_interval_seconds=min_interval,
        max_interval_seconds=max_interval,
        tail_blocks=tail_blocks,
        threshold=threshold,
    ))


@click.group()
def audit():
    pass


@audit.command("start")
@click.option("--tail", default=28800, help="Tail blocks for data fetching")
@click.option("--m-min", default=25, help="Minimum samples per miner")
@click.option("--tempo", default=100, help="Weights loop tempo in blocks")
@click.option("--spotcheck-min", default=None, type=int, help="Min spotcheck interval in seconds")
@click.option("--spotcheck-max", default=None, type=int, help="Max spotcheck interval in seconds")
@click.option("--threshold", default=None, type=float, help="Spotcheck match threshold (0.0-1.0)")
def start_cmd(
    tail: int,
    m_min: int,
    tempo: int,
    spotcheck_min: int | None,
    spotcheck_max: int | None,
    threshold: float | None,
):
    settings = get_settings()

    if spotcheck_min is None:
        spotcheck_min = settings.AUDIT_SPOTCHECK_MIN_INTERVAL_S
    if spotcheck_max is None:
        spotcheck_max = settings.AUDIT_SPOTCHECK_MAX_INTERVAL_S
    if threshold is None:
        threshold = settings.AUDIT_SPOTCHECK_THRESHOLD

    setup_signal_handlers()

    logger.info("Starting audit validator with two processes")
    logger.info("  Weights loop: tempo=%d blocks, tail=%d, m_min=%d", tempo, tail, m_min)
    logger.info("  Spotcheck loop: interval=%d-%d seconds, threshold=%.0f%%", spotcheck_min, spotcheck_max, threshold * 100)

    weights_proc = multiprocessing.Process(
        target=run_weights_process,
        args=(tail, m_min, tempo),
        name="audit-weights",
    )
    spotcheck_proc = multiprocessing.Process(
        target=run_spotcheck_process,
        args=(spotcheck_min, spotcheck_max, tail, threshold),
        name="audit-spotcheck",
    )

    weights_proc.start()
    spotcheck_proc.start()

    logger.info("Both processes started (weights pid=%d, spotcheck pid=%d)", weights_proc.pid, spotcheck_proc.pid)

    try:
        weights_proc.join()
        spotcheck_proc.join()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, terminating processes...")
        weights_proc.terminate()
        spotcheck_proc.terminate()
        weights_proc.join(timeout=5)
        spotcheck_proc.join(timeout=5)

    logger.info("Audit validator shutdown complete")


@audit.command("weights")
@click.option("--tail", default=28800, help="Tail blocks for data fetching")
@click.option("--m-min", default=25, help="Minimum samples per miner")
@click.option("--tempo", default=100, help="Weights loop tempo in blocks")
def weights_cmd(tail: int, m_min: int, tempo: int):
    from scorevision.validator.weights import weights_loop

    logger.info("Starting weights-only mode (tempo=%d blocks)", tempo)
    asyncio.run(weights_loop(tail=tail, m_min=m_min, tempo=tempo))


@audit.command("spotcheck")
@click.option("--min-interval", default=None, type=int)
@click.option("--max-interval", default=None, type=int)
@click.option("--tail", default=28800)
@click.option("--threshold", default=None, type=float)
@click.option("--element-id", default=None)
@click.option("--once", is_flag=True)
@click.option("--mock-data-dir", default=None, type=click.Path(exists=True), help="Load mock data from local directory (testing only)")
def spotcheck_cmd(
    min_interval: int | None,
    max_interval: int | None,
    tail: int,
    threshold: float | None,
    element_id: str | None,
    once: bool,
    mock_data_dir: str | None,
):
    from pathlib import Path
    from scorevision.validator.spotcheck import spotcheck_loop, run_single_spotcheck

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logging.getLogger("scorevision.spotcheck").setLevel(logging.INFO)
    logging.getLogger("scorevision.audit").setLevel(logging.INFO)

    settings = get_settings()

    mock_path = Path(mock_data_dir) if mock_data_dir else None

    if mock_path is None:
        public_url = settings.R2_BUCKET_PUBLIC_URL
        if not public_url:
            logger.error("R2_BUCKET_PUBLIC_URL is not set")
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
        ))
    else:
        logger.info("Starting spotcheck loop (interval=%d-%d seconds)", min_interval, max_interval)
        asyncio.run(spotcheck_loop(
            min_interval_seconds=min_interval,
            max_interval_seconds=max_interval,
            tail_blocks=tail,
            threshold=threshold,
            element_id=element_id,
        ))
