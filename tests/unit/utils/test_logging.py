"""Unit tests for :func:`polaris.utils.logging.get_logger`."""

from __future__ import annotations

import logging

from polaris.utils.logging import get_logger


def test_returns_a_logger_with_the_given_name() -> None:
    """``get_logger`` returns a standard-library logger with the requested name."""
    logger = get_logger("polaris.test.name")

    assert isinstance(logger, logging.Logger)
    assert logger.name == "polaris.test.name"


def test_is_idempotent_and_adds_no_duplicate_handlers() -> None:
    """Repeated calls return the same logger without stacking handlers."""
    first = get_logger("polaris.test.idempotent")
    handler_count = len(first.handlers)
    second = get_logger("polaris.test.idempotent")

    assert first is second
    assert len(second.handlers) == handler_count
    assert handler_count >= 1
