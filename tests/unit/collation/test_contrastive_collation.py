"""Unit tests for contrastive collation."""

from __future__ import annotations

import pytest
import torch

from polaris.collation import ContrastiveBatch, collate_contrastive
from polaris.tokenizers.encoding import Encoding


def _enc(*ids: int) -> Encoding:
    return Encoding(ids=tuple(ids), tokens=tuple(f"t{i}" for i in ids))


# --- pairs (in-batch negatives only) ---


def test_pairs_produce_aligned_anchor_and_positive() -> None:
    """Two-tuples collate into padded anchor and positive batches, no negatives."""
    batch = collate_contrastive(
        [(_enc(3, 4, 5), _enc(6, 7)), (_enc(8, 9), _enc(10, 11, 12))], pad_id=0
    )

    assert batch.anchor.input_ids.shape == (2, 3)
    assert batch.positive.input_ids.shape == (2, 3)
    assert batch.negatives is None
    assert batch.num_negatives == 0
    assert len(batch) == 2


# --- triples (explicit hard negatives) ---


def test_hard_negatives_are_flattened_with_count() -> None:
    """Hard negatives flatten to ``(B * N, T)`` and record ``num_negatives``."""
    batch = collate_contrastive(
        [
            (_enc(3, 4), _enc(5, 6), [_enc(7), _enc(8)]),
            (_enc(9, 10), _enc(11), [_enc(12), _enc(13)]),
        ],
        pad_id=0,
    )

    assert batch.negatives is not None
    assert batch.negatives.input_ids.shape[0] == 4  # B (2) * N (2)
    assert batch.num_negatives == 2


def test_hard_negative_row_order_is_anchor_major() -> None:
    """Anchor i's negatives occupy rows [i*N, (i+1)*N)."""
    batch = collate_contrastive(
        [
            (_enc(3), _enc(4), [_enc(5), _enc(6)]),
            (_enc(7), _enc(8), [_enc(9), _enc(10)]),
        ],
        pad_id=0,
    )

    assert batch.negatives is not None
    assert batch.negatives.input_ids[:, 0].tolist() == [5, 6, 9, 10]


# --- validation ---


def test_empty_samples_raise() -> None:
    """Collating nothing is a usage error."""
    with pytest.raises(ValueError, match="empty"):
        collate_contrastive([], pad_id=0)


def test_mixed_arity_raises() -> None:
    """Mixing pairs and triples in one batch is rejected."""
    with pytest.raises(ValueError, match="same arity"):
        collate_contrastive(
            [(_enc(3), _enc(4)), (_enc(5), _enc(6), [_enc(7)])],  # type: ignore[list-item]
            pad_id=0,
        )


def test_empty_hard_negative_lists_raise() -> None:
    """A triple must supply at least one hard negative."""
    with pytest.raises(ValueError, match="at least one negative"):
        collate_contrastive([(_enc(3), _enc(4), [])], pad_id=0)


def test_ragged_hard_negative_counts_raise() -> None:
    """Every sample must supply the same number of hard negatives."""
    with pytest.raises(ValueError, match="same number of hard negatives"):
        collate_contrastive(
            [
                (_enc(3), _enc(4), [_enc(5), _enc(6)]),
                (_enc(7), _enc(8), [_enc(9)]),
            ],
            pad_id=0,
        )


# --- device move ---


def test_to_moves_all_sub_batches() -> None:
    """``to`` returns a batch with every sub-batch on the target device."""
    batch = collate_contrastive(
        [(_enc(3, 4), _enc(5), [_enc(6)]), (_enc(7), _enc(8), [_enc(9)])], pad_id=0
    )

    moved = batch.to(torch.device("cpu"))

    assert isinstance(moved, ContrastiveBatch)
    assert moved.negatives is not None
    assert moved.num_negatives == batch.num_negatives
