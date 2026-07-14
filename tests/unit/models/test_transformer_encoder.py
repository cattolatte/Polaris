"""Unit tests for optional segment (token-type) support on TransformerEncoder."""

from __future__ import annotations

import torch

from polaris.models.transformer_encoder import TransformerEncoder

ARCH = {
    "vocab_size": 20,
    "embed_dim": 16,
    "num_heads": 2,
    "num_layers": 1,
    "max_len": 8,
}


def _encoder() -> TransformerEncoder:
    torch.manual_seed(0)
    return TransformerEncoder(**ARCH).eval()  # type: ignore[arg-type]


# --- backward compatibility ---


def test_no_token_type_ids_is_unchanged() -> None:
    """Omitting token_type_ids matches passing all-zeros only if untrained.

    The segment embedding is zero-initialized, so a fresh encoder produces the
    same output with no token types or with all-zero token types.
    """
    encoder = _encoder()
    ids = torch.randint(2, ARCH["vocab_size"], (2, 5))
    mask = torch.ones(2, 5, dtype=torch.long)

    without = encoder(ids, mask)
    with_zeros = encoder(ids, mask, torch.zeros(2, 5, dtype=torch.long))

    assert torch.equal(without, with_zeros)


# --- segment embeddings take effect once learned ---


def test_token_type_ids_change_output_after_the_embedding_is_set() -> None:
    """Segment-B positions shift once the token-type embedding is non-zero."""
    encoder = _encoder()
    with torch.no_grad():
        # A non-uniform shift (a constant across features would be removed by the
        # pre-norm LayerNorm) on segment B.
        encoder.token_type_embedding.weight[1, 0] = 5.0
    ids = torch.randint(2, ARCH["vocab_size"], (1, 6))
    mask = torch.ones(1, 6, dtype=torch.long)
    token_types = torch.tensor([[0, 0, 0, 1, 1, 1]])

    baseline = encoder(ids, mask)
    segmented = encoder(ids, mask, token_types)

    assert not torch.allclose(baseline, segmented)
