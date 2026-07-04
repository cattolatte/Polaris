"""Unit tests for :func:`polaris.collation.collator.collate`.

All tests run fully offline on tiny encodings; no model or dataset is involved.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch

from polaris.collation.batch import Batch
from polaris.collation.collator import collate
from polaris.tokenizers.encoding import Encoding


def _encoding(ids: Sequence[int]) -> Encoding:
    return Encoding(ids=tuple(ids), tokens=tuple(str(i) for i in ids))


# ---------------------------------------------------------------------------
# Basic collation
# ---------------------------------------------------------------------------


def test_returns_batch() -> None:
    """Collation produces a ``Batch``."""
    batch = collate([(_encoding([1, 2]), 0)], pad_id=0)

    assert isinstance(batch, Batch)


def test_pads_to_longest_sequence() -> None:
    """Shorter sequences are right-padded to the longest in the group."""
    batch = collate([(_encoding([4, 5, 6]), 1), (_encoding([7]), 0)], pad_id=0)

    assert batch.input_ids.tolist() == [[4, 5, 6], [7, 0, 0]]


def test_attention_mask_marks_real_tokens() -> None:
    """The attention mask is 1 for real tokens and 0 for padding."""
    batch = collate([(_encoding([4, 5, 6]), 1), (_encoding([7]), 0)], pad_id=0)

    assert batch.attention_mask.tolist() == [[1, 1, 1], [1, 0, 0]]


def test_uses_pad_id_for_padding() -> None:
    """Padding positions are filled with the supplied pad id."""
    batch = collate([(_encoding([4, 5, 6]), 1), (_encoding([7]), 0)], pad_id=99)

    assert batch.input_ids.tolist() == [[4, 5, 6], [7, 99, 99]]


def test_labels_are_collected_in_order() -> None:
    """Labels are stacked in the order of the samples."""
    batch = collate(
        [(_encoding([1]), 1), (_encoding([2]), 0), (_encoding([3]), 1)],
        pad_id=0,
    )

    assert batch.labels.tolist() == [1, 0, 1]


def test_tensors_are_long_dtype() -> None:
    """All three tensors use the long dtype expected by embeddings/loss."""
    batch = collate([(_encoding([1]), 0)], pad_id=0)

    assert batch.input_ids.dtype == torch.long
    assert batch.attention_mask.dtype == torch.long
    assert batch.labels.dtype == torch.long


def test_equal_length_sequences_need_no_padding() -> None:
    """When all sequences are equal length the mask is all ones."""
    batch = collate([(_encoding([1, 2]), 0), (_encoding([3, 4]), 1)], pad_id=0)

    assert batch.attention_mask.tolist() == [[1, 1], [1, 1]]


def test_single_sample() -> None:
    """A single-sample batch collates correctly."""
    batch = collate([(_encoding([1, 2, 3]), 1)], pad_id=0)

    assert len(batch) == 1
    assert batch.input_ids.tolist() == [[1, 2, 3]]


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


def test_truncates_to_max_length() -> None:
    """Sequences longer than ``max_length`` are truncated."""
    batch = collate([(_encoding([1, 2, 3, 4, 5]), 0)], pad_id=0, max_length=3)

    assert batch.input_ids.tolist() == [[1, 2, 3]]
    assert batch.attention_mask.tolist() == [[1, 1, 1]]


def test_truncation_and_padding_combine() -> None:
    """After truncation, shorter sequences still pad to the longest kept length."""
    batch = collate(
        [(_encoding([1, 2, 3, 4]), 0), (_encoding([5, 6]), 1)],
        pad_id=0,
        max_length=3,
    )

    assert batch.input_ids.tolist() == [[1, 2, 3], [5, 6, 0]]
    assert batch.attention_mask.tolist() == [[1, 1, 1], [1, 1, 0]]


# ---------------------------------------------------------------------------
# Edge cases and errors
# ---------------------------------------------------------------------------


def test_empty_batch_raises() -> None:
    """Collating no samples is an error."""
    with pytest.raises(ValueError, match="empty batch"):
        collate([], pad_id=0)


def test_max_length_below_one_raises() -> None:
    """A ``max_length`` below one is rejected."""
    with pytest.raises(ValueError, match="max_length"):
        collate([(_encoding([1]), 0)], pad_id=0, max_length=0)


def test_all_empty_encodings_produce_zero_width_batch() -> None:
    """A batch of empty encodings yields a zero-width, correctly-labelled batch."""
    batch = collate([(_encoding([]), 0), (_encoding([]), 1)], pad_id=0)

    assert batch.input_ids.shape == (2, 0)
    assert batch.attention_mask.shape == (2, 0)
    assert batch.labels.tolist() == [0, 1]
