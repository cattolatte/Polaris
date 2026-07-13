"""Unit tests for the TextEmbedder bi-encoder tower."""

from __future__ import annotations

import torch

from polaris.collation import collate
from polaris.models import TextEmbedder
from polaris.pretraining import MaskedLanguageModel
from polaris.tokenizers.encoding import Encoding

ARCH = {
    "vocab_size": 30,
    "embed_dim": 32,
    "num_heads": 4,
    "num_layers": 2,
    "ff_dim": 64,
    "max_len": 16,
    "pad_id": 0,
}


def _batch() -> object:
    return collate(
        [
            (Encoding(ids=(3, 4, 5, 6), tokens=("a", "b", "c", "d")), 0),
            (Encoding(ids=(7, 8), tokens=("e", "f")), 0),
        ],
        pad_id=0,
        max_length=16,
    )


# --- output shape and normalization ---


def test_forward_emits_one_vector_per_text() -> None:
    """The embedder returns ``(batch_size, embed_dim)`` — a vector per input."""
    embedder = TextEmbedder(**ARCH)  # type: ignore[arg-type]

    out = embedder(_batch())  # type: ignore[arg-type]

    assert out.shape == (2, ARCH["embed_dim"])
    assert embedder.embedding_dim == ARCH["embed_dim"]


def test_normalized_embeddings_are_unit_norm() -> None:
    """With ``normalize=True`` every row has L2 norm 1."""
    embedder = TextEmbedder(**ARCH, normalize=True).eval()  # type: ignore[arg-type]

    norms = embedder(_batch()).norm(dim=-1)  # type: ignore[arg-type]

    assert torch.allclose(norms, torch.ones(2), atol=1e-5)


def test_unnormalized_embeddings_are_not_forced_to_unit_norm() -> None:
    """With ``normalize=False`` the raw pooled vector is returned."""
    embedder = TextEmbedder(**ARCH, normalize=False).eval()  # type: ignore[arg-type]

    norms = embedder(_batch()).norm(dim=-1)  # type: ignore[arg-type]

    assert not torch.allclose(norms, torch.ones(2), atol=1e-3)


def test_projection_changes_output_dimension() -> None:
    """A projection dim reshapes the embedding and updates ``embedding_dim``."""
    embedder = TextEmbedder(**ARCH, projection_dim=16)  # type: ignore[arg-type]

    out = embedder(_batch())  # type: ignore[arg-type]

    assert out.shape == (2, 16)
    assert embedder.embedding_dim == 16


# --- determinism and the encode/forward equivalence ---


def test_forward_matches_encode_in_eval_mode() -> None:
    """``forward(batch)`` equals ``encode(ids, mask)`` (deterministic in eval)."""
    embedder = TextEmbedder(**ARCH).eval()  # type: ignore[arg-type]
    batch = _batch()

    via_forward = embedder(batch)  # type: ignore[arg-type]
    via_encode = embedder.encode(batch.input_ids, batch.attention_mask)  # type: ignore[attr-defined]

    assert torch.equal(via_forward, via_encode)


def test_same_seed_reproduces_the_embedder() -> None:
    """Constructing under the same seed yields identical embeddings."""
    torch.manual_seed(0)
    a = TextEmbedder(**ARCH).eval()  # type: ignore[arg-type]
    torch.manual_seed(0)
    b = TextEmbedder(**ARCH).eval()  # type: ignore[arg-type]

    batch = _batch()
    assert torch.equal(a(batch), b(batch))  # type: ignore[arg-type]


# --- transfer from a pretrained MLM trunk ---


def test_transfer_from_mlm_reproduces_trunk_weights() -> None:
    """A masked-LM trunk transfers into the embedder's encoder exactly."""
    mlm = MaskedLanguageModel(**ARCH)  # type: ignore[arg-type]
    embedder = TextEmbedder(**ARCH)  # type: ignore[arg-type]

    # Independently initialized trunks differ (random embedding weights).
    assert not torch.equal(
        mlm.encoder.embedding.weight, embedder.encoder.embedding.weight
    )

    mlm.transfer_encoder_to(embedder)

    for pretrained, received in zip(
        mlm.encoder.state_dict().values(),
        embedder.encoder.state_dict().values(),
        strict=True,
    ):
        assert torch.equal(pretrained, received)
