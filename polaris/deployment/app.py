"""A minimal HTTP serving layer for a trained Polaris model.

Wraps a :class:`~polaris.inference.predictor.Predictor` (loaded from a saved
bundle) in a small FastAPI application exposing ``POST /predict`` and a
``GET /health`` probe. It is a thin adapter — all the real work lives in the
inference layer; this module only maps HTTP to a `Predictor` call.

Design Principles
-----------------
- **Own the interface.** FastAPI and uvicorn are optional (`serving` extra),
  imported lazily, and their absence raises a `MissingDependencyError` with an
  install hint — never a bare `ImportError`. The request/response schemas are
  Polaris-native (Pydantic, already a dependency).
- Reuse, don't rebuild: the app holds one `Predictor` and delegates to it. There
  is no model logic here.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from polaris.errors import MissingDependencyError
from polaris.inference import load_bundle

if TYPE_CHECKING:
    from fastapi import FastAPI

__all__ = ["PredictRequest", "PredictResponse", "create_app"]


class PredictRequest(BaseModel):
    """A prediction request body."""

    text: str = Field(..., min_length=1, description="The text to classify.")


class PredictResponse(BaseModel):
    """A prediction response body."""

    label: str
    label_id: int
    probabilities: dict[str, float]


def create_app(bundle_path: str | Path) -> FastAPI:
    """Build a FastAPI app that serves predictions from a saved bundle.

    Parameters
    ----------
    bundle_path : str or Path
        Path to a bundle written by
        :func:`~polaris.inference.bundle.save_bundle`. Loaded once at startup.

    Returns
    -------
    fastapi.FastAPI
        An app exposing ``GET /health`` and ``POST /predict``.

    Raises
    ------
    MissingDependencyError
        If the ``serving`` extra (FastAPI) is not installed.
    """
    try:
        from fastapi import FastAPI
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise MissingDependencyError(
            "The 'serving' extra (FastAPI) is required to serve a model.\n\n"
            "Install it with:\n\n    uv sync --extra serving"
        ) from exc

    predictor = load_bundle(bundle_path)
    app = FastAPI(
        title="Polaris",
        description="Serve a trained Polaris model over HTTP.",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness probe."""
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> Any:
        """Classify a single text."""
        prediction = predictor.predict(request.text)
        return PredictResponse(
            label=prediction.label,
            label_id=prediction.label_id,
            probabilities=dict(prediction.probabilities),
        )

    return app
