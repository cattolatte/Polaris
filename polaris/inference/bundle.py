"""Save and load a self-describing model bundle.

Unlike a training checkpoint (which stores weights only, and needs the exact model
object to load them back), a *bundle* carries everything required to rebuild the
model **and** tokenize new text: the model type and constructor arguments, the
trained weights, the vocabulary, and the class names. A caller who has never seen
the training code can :func:`load_bundle` and predict.

Design Principles
-----------------
- The bundle is a single ``torch.save`` payload of Polaris-native, JSON-able parts
  (plus the weight tensors). The format is owned by Polaris, not by any framework.
- ``load_bundle`` returns a ready :class:`~polaris.inference.predictor.Predictor`,
  not loose parts — the boundary object is the usable thing.
- Whitespace tokenizer only, for now: BPE bundles need merge serialization, which
  we add when there is a consumer (ADR-0004).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from polaris.errors import PolarisError
from polaris.inference.factory import build_model
from polaris.inference.predictor import Predictor
from polaris.tokenizers import Tokenizer, Vocabulary, WhitespaceTokenizer
from polaris.version import __version__

__all__ = ["save_bundle", "load_bundle"]


def save_bundle(
    path: str | Path,
    *,
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    model_type: str,
    model_config: dict[str, Any],
    label_names: Sequence[str],
    max_length: int | None = None,
) -> None:
    """Write a self-describing model bundle to ``path``.

    Parameters
    ----------
    path : str or Path
        Destination file. Parent directories are created if needed.
    model : nn.Module
        The trained model whose ``state_dict`` is saved.
    tokenizer : Tokenizer
        The tokenizer the model was trained with. Must be a
        :class:`~polaris.tokenizers.WhitespaceTokenizer` (see module notes).
    model_type : str
        The bundle's model type (``"pooling"`` or ``"transformer"``), used by
        :func:`~polaris.inference.factory.build_model` to reconstruct it.
    model_config : dict
        The exact keyword arguments the model was built with (no weight tensors).
    label_names : Sequence[str]
        Class names, indexed by class id.
    max_length : int, optional
        Truncation length to apply at inference (should match training).

    Raises
    ------
    PolarisError
        If ``tokenizer`` is not a whitespace tokenizer.
    """
    if not isinstance(tokenizer, WhitespaceTokenizer):
        msg = (
            "save_bundle currently supports only WhitespaceTokenizer; "
            f"got {type(tokenizer).__name__}"
        )
        raise PolarisError(msg)

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "polaris_version": __version__,
        "model_type": model_type,
        "model_config": dict(model_config),
        "model_state": model.state_dict(),
        "tokenizer_type": "whitespace",
        "vocabulary": tokenizer.vocabulary.to_dict(),
        "label_names": list(label_names),
        "max_length": max_length,
    }
    torch.save(payload, destination)


def load_bundle(path: str | Path) -> Predictor:
    """Load a bundle and return a ready-to-use predictor.

    Parameters
    ----------
    path : str or Path
        The bundle file written by :func:`save_bundle`.

    Returns
    -------
    Predictor
        A predictor wrapping the reconstructed model and tokenizer.

    Raises
    ------
    PolarisError
        If the bundle names an unsupported tokenizer type.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)

    tokenizer_type = payload.get("tokenizer_type", "whitespace")
    if tokenizer_type != "whitespace":
        msg = f"unsupported tokenizer_type {tokenizer_type!r} in bundle"
        raise PolarisError(msg)

    vocabulary = Vocabulary.from_dict(payload["vocabulary"])
    tokenizer = WhitespaceTokenizer(vocabulary=vocabulary)

    model = build_model(payload["model_type"], payload["model_config"])
    model.load_state_dict(payload["model_state"])

    return Predictor(
        model,
        tokenizer,
        label_names=payload["label_names"],
        max_length=payload.get("max_length"),
    )
