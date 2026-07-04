"""Unit tests for :func:`polaris.experiments.environment.capture_environment`."""

from __future__ import annotations

import polaris
from polaris.experiments.environment import capture_environment


def test_capture_environment_has_expected_keys() -> None:
    """The environment record includes the version and platform keys."""
    environment = capture_environment()

    for key in ("polaris", "python", "platform", "torch"):
        assert key in environment


def test_polaris_version_matches() -> None:
    """The recorded Polaris version matches the installed one."""
    environment = capture_environment()

    assert environment["polaris"] == polaris.__version__
