"""Reconstruct a model from a saved bundle's type + config.

A trained model is persisted as a ``model_type`` string plus the exact keyword
arguments it was built with (see :mod:`polaris.inference.bundle`). This module
turns those back into a live ``nn.Module``.

Design Principles
-----------------
- A concrete ``match`` over the two model types — **not** the dormant registry
  (ADR-0005). Two implementations and a single consumer do not justify a plugin
  lookup; a match statement is more readable and has no indirection. The registry
  becomes justified only when model types proliferate *and* several subsystems
  need name-based lookup.
- The factory only *constructs*; it never loads weights (the bundle loader does
  that) — keeping this a pure, obvious mapping.
"""

from __future__ import annotations

from typing import Any

from torch import nn

from polaris.errors import PolarisError
from polaris.models import MeanPoolingClassifier, TransformerEncoderClassifier

__all__ = ["MODEL_TYPES", "build_model"]

# The model types a bundle may name, mapped to their concrete classes.
MODEL_TYPES: tuple[str, ...] = ("pooling", "transformer")


class UnknownModelTypeError(PolarisError):
    """Raised when a bundle names a model type the factory cannot build."""


def build_model(model_type: str, model_config: dict[str, Any]) -> nn.Module:
    """Construct a model from its type and saved constructor arguments.

    Parameters
    ----------
    model_type : str
        One of :data:`MODEL_TYPES` (``"pooling"`` or ``"transformer"``).
    model_config : dict
        The exact keyword arguments the model was originally built with.

    Returns
    -------
    nn.Module
        A freshly constructed model (with randomly initialized weights; the
        caller loads the trained ``state_dict``).

    Raises
    ------
    UnknownModelTypeError
        If ``model_type`` is not a known model type.

    Examples
    --------
    >>> model = build_model(
    ...     "pooling", {"vocab_size": 10, "num_classes": 2, "embedding_dim": 8}
    ... )
    >>> type(model).__name__
    'MeanPoolingClassifier'
    """
    match model_type:
        case "pooling":
            return MeanPoolingClassifier(**model_config)
        case "transformer":
            return TransformerEncoderClassifier(**model_config)
        case _:
            msg = (
                f"unknown model_type {model_type!r}; " f"expected one of {MODEL_TYPES}"
            )
            raise UnknownModelTypeError(msg)
