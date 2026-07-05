"""
Deployment subsystem for Polaris.

A thin HTTP serving layer over a trained model: load a saved bundle into a
:class:`~polaris.inference.predictor.Predictor` and expose it as a FastAPI app
(``POST /predict``). FastAPI/uvicorn are an optional (`serving`) extra, imported
lazily — the model and inference layers never depend on them.
"""

from __future__ import annotations

from polaris.deployment.app import PredictRequest, PredictResponse, create_app

__all__ = ["PredictRequest", "PredictResponse", "create_app"]
