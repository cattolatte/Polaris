"""Unit tests for :mod:`polaris.embeddings.glove`.

A tiny synthetic GloVe file is written to a temp directory — offline, no network.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from polaris.embeddings.glove import build_embedding_matrix, load_glove
from polaris.tokenizers.vocabulary import Vocabulary


def _write_glove(path: Path) -> None:
    path.write_text("the 0.1 0.2 0.3\ncat 0.4 0.5 0.6\ndog -0.1 -0.2 -0.3\n")


# ---------------------------------------------------------------------------
# load_glove
# ---------------------------------------------------------------------------


def test_load_glove_reads_words_and_vectors(tmp_path: Path) -> None:
    """Each line becomes a word mapped to its vector."""
    path = tmp_path / "glove.txt"
    _write_glove(path)

    vectors = load_glove(path)

    assert set(vectors) == {"the", "cat", "dog"}
    assert vectors["cat"].shape == (3,)
    assert torch.allclose(vectors["cat"], torch.tensor([0.4, 0.5, 0.6]))


# ---------------------------------------------------------------------------
# build_embedding_matrix
# ---------------------------------------------------------------------------


def test_matrix_aligns_known_oov_and_padding(tmp_path: Path) -> None:
    """Known words get their vector; padding is zero; OOV is non-zero random."""
    path = tmp_path / "glove.txt"
    _write_glove(path)
    vectors = load_glove(path)
    vocabulary = Vocabulary(
        {"<pad>": 0, "<unk>": 1, "the": 2, "cat": 3, "zzz": 4},
        pad_token="<pad>",
        unk_token="<unk>",
    )

    matrix = build_embedding_matrix(vocabulary, vectors, embedding_dim=3)

    assert matrix.shape == (5, 3)
    assert torch.allclose(matrix[2], torch.tensor([0.1, 0.2, 0.3]))  # 'the'
    assert torch.allclose(matrix[3], torch.tensor([0.4, 0.5, 0.6]))  # 'cat'
    assert torch.allclose(matrix[0], torch.zeros(3))  # padding
    assert not torch.allclose(matrix[4], torch.zeros(3))  # OOV 'zzz' is random


def test_lowercase_fallback(tmp_path: Path) -> None:
    """A capitalized token matches GloVe's lower-cased entry."""
    path = tmp_path / "glove.txt"
    _write_glove(path)
    vectors = load_glove(path)
    vocabulary = Vocabulary({"<pad>": 0, "The": 1}, pad_token="<pad>")

    matrix = build_embedding_matrix(vocabulary, vectors, embedding_dim=3)

    assert torch.allclose(matrix[1], torch.tensor([0.1, 0.2, 0.3]))


def test_dimension_mismatch_raises(tmp_path: Path) -> None:
    """A GloVe vector whose length differs from embedding_dim is rejected."""
    path = tmp_path / "glove.txt"
    _write_glove(path)
    vectors = load_glove(path)
    vocabulary = Vocabulary({"cat": 0})

    with pytest.raises(ValueError, match="dim"):
        build_embedding_matrix(vocabulary, vectors, embedding_dim=5)


def test_is_seeded_and_deterministic(tmp_path: Path) -> None:
    """The random OOV vectors are reproducible for a fixed seed."""
    path = tmp_path / "glove.txt"
    _write_glove(path)
    vectors = load_glove(path)
    vocabulary = Vocabulary({"<pad>": 0, "zzz": 1}, pad_token="<pad>")

    first = build_embedding_matrix(vocabulary, vectors, embedding_dim=3, seed=7)
    second = build_embedding_matrix(vocabulary, vectors, embedding_dim=3, seed=7)

    assert torch.equal(first, second)
