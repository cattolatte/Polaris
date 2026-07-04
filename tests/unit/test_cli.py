"""Unit tests for the Polaris CLI (:mod:`polaris.cli`)."""

from __future__ import annotations

from typer.testing import CliRunner

from polaris.cli import app
from polaris.version import __version__

runner = CliRunner()


def test_info_reports_version() -> None:
    """`polaris info` succeeds and prints the version."""
    result = runner.invoke(app, ["info"])

    assert result.exit_code == 0
    assert __version__ in result.output


def test_no_arguments_shows_help() -> None:
    """Invoking with no command lists the available commands."""
    result = runner.invoke(app, [])

    assert "info" in result.output
