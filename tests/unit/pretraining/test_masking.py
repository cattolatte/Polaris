"""Unit tests for masked-language-model masking."""

from __future__ import annotations

import pytest
import torch

from polaris.pretraining.masking import IGNORE_INDEX, MaskedLMBatch, mask_tokens

# --- MaskedLMBatch validation ---


def test_masked_batch_accepts_aligned_tensors() -> None:
    """A batch with three same-shaped tensors constructs successfully."""
    ids = torch.zeros(2, 3, dtype=torch.long)
    batch = MaskedLMBatch(input_ids=ids, attention_mask=ids, labels=ids)

    assert len(batch) == 2


def test_masked_batch_rejects_shape_mismatch() -> None:
    """Tensors of differing shapes are rejected at construction."""
    ids = torch.zeros(2, 3, dtype=torch.long)
    wrong = torch.zeros(2, 4, dtype=torch.long)

    with pytest.raises(ValueError, match="share a shape"):
        MaskedLMBatch(input_ids=ids, attention_mask=ids, labels=wrong)


# --- mask_tokens: shapes and label semantics ---


def test_mask_tokens_preserves_shape_and_mask() -> None:
    """Masking returns the input shape and leaves the attention mask untouched."""
    ids = torch.arange(3, 3 + 20).reshape(2, 10)
    attention = torch.ones(2, 10, dtype=torch.long)
    generator = torch.Generator().manual_seed(0)

    batch = mask_tokens(ids, attention, mask_id=1, vocab_size=40, generator=generator)

    assert batch.input_ids.shape == ids.shape
    assert torch.equal(batch.attention_mask, attention)
    assert batch.labels.shape == ids.shape


def test_supervised_positions_carry_original_ids() -> None:
    """Where a label is set, it equals the original (pre-corruption) token id."""
    ids = torch.arange(3, 3 + 40).reshape(2, 20)
    attention = torch.ones(2, 20, dtype=torch.long)
    generator = torch.Generator().manual_seed(0)

    batch = mask_tokens(ids, attention, mask_id=1, vocab_size=60, generator=generator)

    supervised = batch.labels != IGNORE_INDEX
    assert supervised.any()
    assert torch.equal(batch.labels[supervised], ids[supervised])


def test_unsupervised_positions_are_ignored() -> None:
    """Positions not selected for masking carry the ignore index."""
    ids = torch.arange(3, 3 + 40).reshape(2, 20)
    attention = torch.ones(2, 20, dtype=torch.long)
    generator = torch.Generator().manual_seed(0)

    batch = mask_tokens(ids, attention, mask_id=1, vocab_size=60, generator=generator)

    unsupervised = batch.labels == IGNORE_INDEX
    assert unsupervised.any()  # not everything is masked at 15%


# --- mask_tokens: exclusions ---


def test_padding_is_never_supervised() -> None:
    """Padding positions (attention 0) are never selected for masking."""
    ids = torch.arange(3, 3 + 20).reshape(2, 10)
    attention = torch.ones(2, 10, dtype=torch.long)
    attention[:, 5:] = 0  # second half is padding
    generator = torch.Generator().manual_seed(0)

    batch = mask_tokens(ids, attention, mask_id=1, vocab_size=40, generator=generator)

    assert torch.all(batch.labels[:, 5:] == IGNORE_INDEX)


def test_special_tokens_are_never_masked() -> None:
    """Ids listed as special are neither supervised nor corrupted."""
    special = 7
    ids = torch.full((4, 12), special, dtype=torch.long)
    attention = torch.ones(4, 12, dtype=torch.long)
    generator = torch.Generator().manual_seed(0)

    batch = mask_tokens(
        ids,
        attention,
        mask_id=1,
        vocab_size=40,
        special_token_ids=(special,),
        generator=generator,
    )

    assert torch.all(batch.labels == IGNORE_INDEX)
    assert torch.equal(batch.input_ids, ids)  # nothing corrupted


# --- mask_tokens: reproducibility and distribution ---


def test_same_seed_reproduces_masking() -> None:
    """The same generator seed yields identical corruption and labels."""
    ids = torch.arange(3, 3 + 40).reshape(2, 20)
    attention = torch.ones(2, 20, dtype=torch.long)

    a = mask_tokens(
        ids,
        attention,
        mask_id=1,
        vocab_size=60,
        generator=torch.Generator().manual_seed(3),
    )
    b = mask_tokens(
        ids,
        attention,
        mask_id=1,
        vocab_size=60,
        generator=torch.Generator().manual_seed(3),
    )

    assert torch.equal(a.input_ids, b.input_ids)
    assert torch.equal(a.labels, b.labels)


def test_masking_follows_the_80_10_10_scheme() -> None:
    """Over many tokens, ~15% are supervised and ~80% of those become <mask>."""
    mask_id = 1
    ids = torch.randint(3, 100, (200, 200))
    attention = torch.ones(200, 200, dtype=torch.long)
    generator = torch.Generator().manual_seed(0)

    batch = mask_tokens(
        ids, attention, mask_id=mask_id, vocab_size=100, generator=generator
    )

    total = ids.numel()
    supervised = batch.labels != IGNORE_INDEX
    supervised_fraction = int(supervised.sum()) / total
    assert 0.13 < supervised_fraction < 0.17  # ~15%

    became_mask = (batch.input_ids == mask_id) & supervised
    mask_fraction = int(became_mask.sum()) / int(supervised.sum())
    assert 0.75 < mask_fraction < 0.85  # ~80% of supervised
