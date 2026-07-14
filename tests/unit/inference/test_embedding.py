"""Unit tests for the encode_texts corpus-embedding helper."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from polaris.collation import collate
from polaris.errors import PolarisError
from polaris.inference import encode_texts
from polaris.models import TextEmbedder
from polaris.tokenizers import WhitespaceTokenizer, build_vocabulary


def _setup() -> tuple[TextEmbedder, WhitespaceTokenizer]:
    torch.manual_seed(0)
    vocab = build_vocabulary(
        [["good", "movie"], ["bad", "film"]], unk_token="<unk>", pad_token="<pad>"
    )
    tokenizer = WhitespaceTokenizer(vocabulary=vocab)
    embedder = TextEmbedder(
        vocab_size=len(vocab), embed_dim=16, num_heads=2, num_layers=1, max_len=16
    ).eval()
    return embedder, tokenizer


# --- shape and content ---


def test_returns_one_row_per_text() -> None:
    """The result is a ``(len(texts), embedding_dim)`` float array."""
    embedder, tokenizer = _setup()

    array = encode_texts(embedder, tokenizer, ["good movie", "bad film", "good"])

    assert array.shape == (3, embedder.embedding_dim)
    assert array.dtype == np.float32


def test_matches_a_direct_embedder_call() -> None:
    """A row equals embedding that text directly through the embedder."""
    embedder, tokenizer = _setup()

    array = encode_texts(embedder, tokenizer, ["good movie"], batch_size=2)
    direct = (
        embedder(collate([(tokenizer.encode("good movie"), 0)], pad_id=0))
        .detach()
        .numpy()
    )

    assert np.allclose(array[0], direct[0], atol=1e-5)


def test_batching_does_not_change_the_result() -> None:
    """Chunking by ``batch_size`` yields the same array as one big batch."""
    embedder, tokenizer = _setup()
    texts = ["good movie", "bad film", "good", "film bad movie"]

    small = encode_texts(embedder, tokenizer, texts, batch_size=1)
    large = encode_texts(embedder, tokenizer, texts, batch_size=32)

    assert np.allclose(small, large, atol=1e-5)


# --- edge cases ---


def test_empty_input_returns_empty_matrix() -> None:
    """No texts yields a ``(0, embedding_dim)`` array (no model call)."""
    embedder, tokenizer = _setup()

    array = encode_texts(embedder, tokenizer, [])

    assert array.shape == (0, embedder.embedding_dim)


def test_requires_a_padding_token() -> None:
    """A vocabulary without a pad id cannot back corpus encoding."""
    vocab = build_vocabulary([["good", "bad"]], unk_token="<unk>")
    tokenizer = WhitespaceTokenizer(vocabulary=vocab)
    embedder = TextEmbedder(vocab_size=len(vocab), embed_dim=8, num_heads=2)

    with pytest.raises(PolarisError, match="padding token"):
        encode_texts(embedder, tokenizer, ["good"])
