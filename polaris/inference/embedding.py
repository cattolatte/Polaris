"""Batch-embed a corpus of texts with a :class:`TextEmbedder`.

The convenience wrapper for the common inference task behind a dense retriever:
turn many raw strings into a matrix of embeddings. It tokenizes, collates into
padded batches, runs the embedder in ``eval`` mode under ``torch.no_grad``, and
returns a single NumPy array — ready to hand to an index.

Design Principles
-----------------
- A thin helper over the existing tokenizer + collation + embedder; no model math.
- Returns a Polaris-external ``numpy.ndarray`` at the boundary (the shape an ANN
  index expects), not a tensor.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from polaris.collation import collate
from polaris.errors import PolarisError
from polaris.models.embedder import TextEmbedder
from polaris.tokenizers import Tokenizer

__all__ = ["encode_texts"]


def encode_texts(
    embedder: TextEmbedder,
    tokenizer: Tokenizer,
    texts: Sequence[str],
    *,
    batch_size: int = 32,
    max_length: int | None = None,
) -> np.ndarray:
    """Embed a sequence of texts into a ``(len(texts), embedding_dim)`` array.

    Parameters
    ----------
    embedder : TextEmbedder
        The trained embedder (placed in evaluation mode for the call).
    tokenizer : Tokenizer
        The tokenizer whose vocabulary the embedder was trained on. Its vocabulary
        must define a padding id.
    texts : Sequence[str]
        The raw texts to embed.
    batch_size : int, default 32
        How many texts to embed per forward pass.
    max_length : int, optional
        Truncation length applied during collation.

    Returns
    -------
    numpy.ndarray
        A ``(len(texts), embedding_dim)`` float array. Empty input yields a
        ``(0, embedding_dim)`` array.

    Raises
    ------
    PolarisError
        If the tokenizer's vocabulary has no padding id (required for collation).
    """
    pad_id = tokenizer.vocabulary.pad_id
    if pad_id is None:
        msg = "the tokenizer's vocabulary must define a padding token for encoding"
        raise PolarisError(msg)

    if not texts:
        return np.empty((0, embedder.embedding_dim), dtype=np.float32)

    embedder.eval()
    device = next(embedder.parameters()).device

    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            window = texts[start : start + batch_size]
            batch = collate(
                [(tokenizer.encode(text), 0) for text in window],
                pad_id=pad_id,
                max_length=max_length,
            ).to(device)
            embeddings = embedder(batch)
            chunks.append(embeddings.cpu().numpy())

    return np.concatenate(chunks, axis=0)
