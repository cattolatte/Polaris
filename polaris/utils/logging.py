"""Standard-library logging for Polaris.

A thin helper so components (starting with the training engine) log through a
single, consistently-formatted logger instead of ``print``.
"""

from __future__ import annotations

import logging

__all__ = ["get_logger"]


def get_logger(name: str = "polaris") -> logging.Logger:
    """Return a configured logger.

    The logger is set up once (a stream handler at ``INFO``); repeated calls with
    the same name return the same logger without adding duplicate handlers.

    Parameters
    ----------
    name : str, default "polaris"
        The logger name.

    Returns
    -------
    logging.Logger
        The configured logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
