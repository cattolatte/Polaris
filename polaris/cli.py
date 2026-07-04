"""Polaris command-line interface.

Currently minimal — a single ``info`` command. The CLI grows one command per
phase as real functionality (training, evaluation, prediction) is exposed, rather
than arriving all at once.
"""

from __future__ import annotations

import typer

from polaris.version import __version__

app = typer.Typer(
    help="Polaris — an educational, from-scratch NLP engineering platform.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def main() -> None:
    """Polaris — an educational, from-scratch NLP engineering platform."""
    # A root callback makes commands (e.g. `info`) named subcommands rather than
    # a single implicit command; it also gives the CLI room to grow per phase.


@app.command()
def info() -> None:
    """Print basic information about this Polaris installation."""
    typer.echo(f"Polaris {__version__}")
    typer.echo("An educational, from-scratch NLP engineering platform.")
    typer.echo("https://github.com/cattolatte/Polaris")


if __name__ == "__main__":  # pragma: no cover
    app()
