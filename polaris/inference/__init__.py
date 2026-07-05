"""
Inference subsystem for Polaris.

Turns a trained model into a usable artifact: save a self-describing bundle
(model type + config + weights + vocabulary + labels), reload it anywhere, and
predict class labels on raw text — with no training code in sight.
"""

from __future__ import annotations

from polaris.inference.bundle import load_bundle, save_bundle
from polaris.inference.factory import build_model
from polaris.inference.predictor import Prediction, Predictor

__all__ = [
    "Prediction",
    "Predictor",
    "build_model",
    "load_bundle",
    "save_bundle",
]
