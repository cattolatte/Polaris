"""Unit tests for the inference model factory."""

from __future__ import annotations

import pytest

from polaris.inference.factory import UnknownModelTypeError, build_model
from polaris.models import (
    MeanPoolingClassifier,
    TextEmbedder,
    TransformerEncoderClassifier,
)

# --- construction ---


def test_builds_pooling_model() -> None:
    """A ``"pooling"`` type builds a MeanPoolingClassifier from its config."""
    model = build_model(
        "pooling", {"vocab_size": 10, "num_classes": 2, "embedding_dim": 8}
    )

    assert isinstance(model, MeanPoolingClassifier)


def test_builds_transformer_model() -> None:
    """A ``"transformer"`` type builds a TransformerEncoderClassifier."""
    model = build_model(
        "transformer",
        {
            "vocab_size": 10,
            "num_classes": 2,
            "embed_dim": 8,
            "num_heads": 2,
            "num_layers": 1,
        },
    )

    assert isinstance(model, TransformerEncoderClassifier)


def test_builds_embedder_model() -> None:
    """An ``"embedder"`` type builds a TextEmbedder from its config."""
    model = build_model(
        "embedder",
        {"vocab_size": 10, "embed_dim": 8, "num_heads": 2, "num_layers": 1},
    )

    assert isinstance(model, TextEmbedder)


# --- errors ---


def test_unknown_model_type_raises() -> None:
    """An unrecognized model type raises a typed error."""
    with pytest.raises(UnknownModelTypeError, match="unknown model_type"):
        build_model("recurrent", {"vocab_size": 10, "num_classes": 2})
