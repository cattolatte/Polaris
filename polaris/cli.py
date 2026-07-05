"""Polaris command-line interface.

Grows one command per phase as real functionality is exposed, rather than arriving
all at once: ``info`` (v0.4) and ``predict`` (v0.12, run a saved model on text).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

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


@app.command()
def predict(
    text: Annotated[str, typer.Argument(help="The text to classify.")],
    model: Annotated[
        Path,
        typer.Option(
            "--model",
            "-m",
            exists=True,
            dir_okay=False,
            help="Path to a saved model bundle.",
        ),
    ],
    probs: Annotated[
        bool, typer.Option("--probs", help="Also print per-class probabilities.")
    ] = False,
) -> None:
    """Classify TEXT with a saved model bundle."""
    # Imported lazily so `polaris info` works without the optional torch extra.
    from polaris.inference import load_bundle

    predictor = load_bundle(model)
    prediction = predictor.predict(text)
    typer.echo(prediction.label)
    if probs:
        for name, probability in prediction.probabilities.items():
            typer.echo(f"  {name}: {probability:.4f}")


@app.command()
def serve(
    model: Annotated[
        Path,
        typer.Option(
            "--model",
            "-m",
            exists=True,
            dir_okay=False,
            help="Path to a saved model bundle.",
        ),
    ],
    host: Annotated[str, typer.Option(help="Host to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to bind.")] = 8000,
) -> None:
    """Serve a saved model bundle over HTTP (POST /predict)."""
    # Imported lazily so the rest of the CLI works without the `serving` extra.
    from polaris.deployment import create_app

    app_instance = create_app(model)
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        from polaris.errors import MissingDependencyError

        raise MissingDependencyError(
            "The 'serving' extra (uvicorn) is required to serve a model.\n\n"
            "Install it with:\n\n    uv sync --extra serving"
        ) from exc

    uvicorn.run(app_instance, host=host, port=port)  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    app()
