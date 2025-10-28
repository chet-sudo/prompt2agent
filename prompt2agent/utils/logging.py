"""Logging utilities for the Prompt2Agent project."""
from __future__ import annotations

import logging
from typing import Optional

_LOGGER: Optional[logging.Logger] = None


def get_logger(name: str = "prompt2agent") -> logging.Logger:
    """Return a configured logger instance."""
    global _LOGGER
    if _LOGGER is None:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        )
        _LOGGER = logging.getLogger(name)
    return logging.getLogger(name)
