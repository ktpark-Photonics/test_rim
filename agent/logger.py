"""Logging utilities for the RimWorld agent stack."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logging(log_level: str = "INFO", log_dir: Optional[Path] = None) -> None:
    """Configure loguru logging sinks.

    Parameters
    ----------
    log_level:
        Initial log level. Accepted values follow loguru's configuration.
    log_dir:
        Optional directory for rotating file logs. When provided a daily
        rotating log file is created and console logging is preserved.
    """

    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level=log_level)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_dir / "agent.log",
            level=log_level,
            rotation="1 day",
            retention="7 days",
            enqueue=True,
        )


__all__ = ["logger", "setup_logging"]
