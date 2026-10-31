"""Logging configuration shared by the CLI and the API server."""

import logging

from rich.logging import RichHandler

_NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "huggingface_hub", "fastembed")


def configure_logging(level: str = "INFO", *, rich: bool = True) -> None:
    """Configure the root logger once; ``rich`` gives colored output for terminals."""
    handler: logging.Handler
    if rich:
        handler = RichHandler(rich_tracebacks=True, show_path=False)
        fmt = "%(message)s"
    else:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=level.upper(), format=fmt, handlers=[handler], force=True)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
