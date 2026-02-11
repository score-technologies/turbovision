import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
_DATEFMT = "%H:%M:%S"

_RESET = "\033[0m"
_BLUE = "\033[34m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_GRAY = "\033[90m"

_LEVEL_COLORS = {
    "DEBUG": _GRAY,
    "INFO": _GREEN,
    "WARNING": _YELLOW,
    "ERROR": _RED,
    "CRITICAL": _RED,
}

_VERBOSITY_MAP = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG,
}

_configured_level: int = logging.INFO


class ColorFormatter(logging.Formatter):
    def format(self, record):
        color = _LEVEL_COLORS.get(record.levelname, _RESET)
        record.levelname = f"{color}{record.levelname:7}{_RESET}"
        record.msg = f"{_BLUE}{record.msg}{_RESET}"
        return super().format(record)


def setup_logging(verbosity: int | None = None) -> None:
    global _configured_level
    if verbosity is not None:
        _configured_level = _VERBOSITY_MAP.get(verbosity, logging.INFO)

    sv_logger = logging.getLogger("scorevision")
    sv_logger.setLevel(_configured_level)
    sv_logger.propagate = False

    if not sv_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ColorFormatter(_FORMAT, datefmt=_DATEFMT))
        sv_logger.addHandler(handler)

    sv_logger.handlers[0].setLevel(_configured_level)

    logging.getLogger("bittensor").setLevel(logging.WARNING)
