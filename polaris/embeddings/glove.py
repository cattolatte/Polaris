"""Load pretrained GloVe word vectors and align them to a vocabulary.

GloVe vectors carry word meaning learned from a huge corpus. Initializing a
model's embedding layer with them — rather than random noise — lets the model
start out already knowing that "great" is positive and "terrible" is negative,
which is the single biggest lever on a small model's accuracy.

We *consume* pretrained vectors; we do not train them. The vector file is large
and is not bundled — download it (e.g. ``glove.6B.100d.txt``) and pass its path.
"""

from __future__ import annotations

from pathlib import Path

import torch

from polaris.tokenizers.vocabulary import Vocabulary

__all__ = ["load_glove", "build_embedding_matrix"]


def load_glove(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a GloVe text file into a mapping from word to vector.

    Each line is ``word v1 v2 … vd``.

    Parameters
    ----------
    path : str or Path
        Path to the GloVe ``.txt`` file.

    Returns
    -------
    dict[str, torch.Tensor]
        A mapping from each word to its vector.
    """
    vectors: dict[str, torch.Tensor] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip().split(" ")
            if len(parts) < 2:
                continue
            word = parts[0]
            vectors[word] = torch.tensor(
                [float(value) for value in parts[1:]], dtype=torch.float32
            )
    return vectors


def build_embedding_matrix(
    vocabulary: Vocabulary,
    vectors: dict[str, torch.Tensor],
    *,
    embedding_dim: int,
    seed: int = 0,
) -> torch.Tensor:
    """Build an embedding matrix aligned to ``vocabulary``.

    Each token gets its GloVe vector if available (falling back to the
    lower-cased form, since GloVe is lower-cased); out-of-vocabulary tokens get a
    seeded random vector so they remain learnable; the padding token gets zeros.

    Parameters
    ----------
    vocabulary : Vocabulary
        The vocabulary whose tokens to embed.
    vectors : dict[str, torch.Tensor]
        Word vectors, e.g. from :func:`load_glove`.
    embedding_dim : int
        The dimensionality of the vectors.
    seed : int, default 0
        Seed for the random vectors given to out-of-vocabulary tokens.

    Returns
    -------
    torch.Tensor
        A ``(len(vocabulary), embedding_dim)`` float tensor.

    Raises
    ------
    ValueError
        If a matched GloVe vector does not have length ``embedding_dim``.
    """
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(len(vocabulary), embedding_dim, generator=generator)

    for token, token_id in vocabulary.token_to_id.items():
        vector = vectors.get(token)
        if vector is None:
            vector = vectors.get(token.lower())
        if vector is not None:
            if vector.shape[0] != embedding_dim:
                msg = (
                    f"vector for {token!r} has dim {vector.shape[0]}, "
                    f"expected {embedding_dim}"
                )
                raise ValueError(msg)
            matrix[token_id] = vector

    if vocabulary.pad_id is not None:
        matrix[vocabulary.pad_id] = 0.0
    return matrix
