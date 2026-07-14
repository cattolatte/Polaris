"""Unit tests for sentence-pair collation."""

from __future__ import annotations

import pytest
import torch

from polaris.collation import PairBatch, collate_pairs
from polaris.tokenizers.encoding import Encoding

CLS, SEP, PAD = 1, 2, 0


def _enc(*ids: int) -> Encoding:
    return Encoding(ids=tuple(ids), tokens=tuple(f"t{i}" for i in ids))


# --- layout ---


def test_builds_cls_a_sep_b_sep() -> None:
    """A pair packs to ``[CLS] a [SEP] b [SEP]`` with matching segment ids."""
    batch = collate_pairs(
        [(_enc(3, 4), _enc(5), 1)], pad_id=PAD, cls_id=CLS, sep_id=SEP
    )

    assert batch.input_ids[0].tolist() == [CLS, 3, 4, SEP, 5, SEP]
    assert batch.token_type_ids[0].tolist() == [0, 0, 0, 0, 1, 1]
    assert batch.attention_mask[0].tolist() == [1, 1, 1, 1, 1, 1]
    assert batch.labels.tolist() == [1]


def test_ragged_pairs_are_padded_and_aligned() -> None:
    """Shorter pairs are right-padded; mask and segment ids pad with 0."""
    batch = collate_pairs(
        [(_enc(3, 4), _enc(5), 1), (_enc(6), _enc(7), 0)],
        pad_id=PAD,
        cls_id=CLS,
        sep_id=SEP,
    )

    assert batch.input_ids.shape == (2, 6)
    assert batch.input_ids[1].tolist() == [CLS, 6, SEP, 7, SEP, PAD]
    assert batch.attention_mask[1].tolist() == [1, 1, 1, 1, 1, 0]
    assert batch.token_type_ids[1].tolist() == [0, 0, 0, 1, 1, 0]


# --- truncation ---


def test_max_length_truncates_longest_first_keeping_specials() -> None:
    """Truncation fits ``max_length`` while keeping all three special tokens."""
    batch = collate_pairs(
        [(_enc(3, 4, 5, 6), _enc(7, 8, 9), 0)],
        pad_id=PAD,
        cls_id=CLS,
        sep_id=SEP,
        max_length=5,
    )

    ids = batch.input_ids[0].tolist()
    assert len(ids) == 5
    assert ids[0] == CLS and ids[-1] == SEP  # specials survive
    assert ids.count(SEP) == 2


# --- validation ---


def test_empty_samples_raise() -> None:
    """Collating nothing is a usage error."""
    with pytest.raises(ValueError, match="empty"):
        collate_pairs([], pad_id=PAD, cls_id=CLS, sep_id=SEP)


def test_max_length_below_three_raises() -> None:
    """There must be room for the three special tokens."""
    with pytest.raises(ValueError, match="at least 3"):
        collate_pairs(
            [(_enc(3), _enc(4), 0)], pad_id=PAD, cls_id=CLS, sep_id=SEP, max_length=2
        )


def test_to_moves_all_tensors() -> None:
    """``to`` returns an aligned batch on the target device."""
    batch = collate_pairs([(_enc(3), _enc(4), 0)], pad_id=PAD, cls_id=CLS, sep_id=SEP)

    moved = batch.to(torch.device("cpu"))

    assert isinstance(moved, PairBatch)
    assert moved.token_type_ids.shape == batch.token_type_ids.shape
