"""Unit tests for :class:`polaris.collation.batch.Batch`.

All tests run fully offline on tiny tensors; no model or dataset is involved.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from polaris.collation.batch import Batch


def _batch(rows: int = 2, cols: int = 3) -> Batch:
    return Batch(
        input_ids=torch.zeros((rows, cols), dtype=torch.long),
        attention_mask=torch.ones((rows, cols), dtype=torch.long),
        labels=torch.zeros((rows,), dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_len_returns_batch_size() -> None:
    """``len`` returns the number of rows."""
    assert len(_batch(rows=4)) == 4


def test_fields_are_accessible() -> None:
    """The three tensors are exposed unchanged."""
    input_ids = torch.tensor([[1, 2], [3, 4]])
    attention_mask = torch.tensor([[1, 1], [1, 0]])
    labels = torch.tensor([0, 1])

    batch = Batch(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    assert torch.equal(batch.input_ids, input_ids)
    assert torch.equal(batch.attention_mask, attention_mask)
    assert torch.equal(batch.labels, labels)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_mismatched_input_and_mask_shape_raises() -> None:
    """A shape mismatch between ids and mask is rejected."""
    with pytest.raises(ValueError, match="same shape"):
        Batch(
            input_ids=torch.zeros((2, 3), dtype=torch.long),
            attention_mask=torch.zeros((2, 4), dtype=torch.long),
            labels=torch.zeros((2,), dtype=torch.long),
        )


def test_label_count_mismatch_raises() -> None:
    """A label count that does not match the row count is rejected."""
    with pytest.raises(ValueError, match="one entry per row"):
        Batch(
            input_ids=torch.zeros((2, 3), dtype=torch.long),
            attention_mask=torch.zeros((2, 3), dtype=torch.long),
            labels=torch.zeros((3,), dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


def test_fields_cannot_be_reassigned() -> None:
    """``Batch`` is frozen; fields cannot be rebound."""
    batch = _batch()

    with pytest.raises(dataclasses.FrozenInstanceError):
        batch.input_ids = torch.zeros((1, 1), dtype=torch.long)  # type: ignore[misc]
