"""
Embeddings subsystem for Polaris.

Load pretrained word vectors (GloVe) and align them to a vocabulary so a model's
embedding layer can start from real word meaning instead of random noise.
"""

from __future__ import annotations

from polaris.embeddings.glove import build_embedding_matrix, load_glove

__all__ = [
    "build_embedding_matrix",
    "load_glove",
]
