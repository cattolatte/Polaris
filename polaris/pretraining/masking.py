"""Masked-language-model masking: hide tokens and record what to predict.

The self-supervision behind BERT-style pretraining. Given a padded batch of token
ids, we randomly choose ~15% of the real (non-special) positions to *supervise*,
and for each chosen position apply the standard 80/10/10 corruption: 80% become
the ``<mask>`` token, 10% become a random token, and 10% are left unchanged. The
model is then trained to recover the **original** token at every chosen position.

The 10%-random and 10%-unchanged fractions matter: if masked positions were always
the literal ``<mask>`` token, the model would only ever see ``<mask>`` at
prediction time and could ignore real tokens. Mixing in random and unchanged
tokens forces it to build a contextual representation of *every* position.

Design Principles
-----------------
- Produces a :class:`MaskedLMBatch` — the same shape of value object as
  :class:`~polaris.collation.batch.Batch`, but whose ``labels`` are per-token
  targets (original ids at supervised positions, :data:`IGNORE_INDEX` elsewhere)
  rather than one class per row.
- Pure and seedable: pass a ``torch.Generator`` for reproducible masking. No
  global RNG is touched.
- Special tokens (padding, unknown, the mask token itself) are never selected for
  masking; pass their ids via ``special_token_ids``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch

__all__ = ["IGNORE_INDEX", "MaskedLMBatch", "mask_tokens"]

# Cross-entropy ignores targets equal to this value, so unsupervised positions
# contribute no loss. Matches PyTorch's ``nn.CrossEntropyLoss`` default.
IGNORE_INDEX = -100


@dataclass(frozen=True, slots=True, eq=False)
class MaskedLMBatch:
    """A batch of masked inputs with per-token prediction targets.

    Parameters
    ----------
    input_ids : torch.Tensor
        Long tensor of shape ``(batch_size, seq_len)`` — the corrupted token ids
        fed to the model (some replaced by ``<mask>`` or random tokens).
    attention_mask : torch.Tensor
        Long tensor of shape ``(batch_size, seq_len)`` with ``1`` at real token
        positions and ``0`` at padding positions (unchanged by masking).
    labels : torch.Tensor
        Long tensor of shape ``(batch_size, seq_len)`` holding the **original**
        token id at supervised positions and :data:`IGNORE_INDEX` everywhere else.

    Raises
    ------
    ValueError
        If the three tensors do not all share the same shape.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor

    def __post_init__(self) -> None:
        """Validate that the three tensors are mutually aligned."""
        if not (self.input_ids.shape == self.attention_mask.shape == self.labels.shape):
            msg = (
                "input_ids, attention_mask and labels must share a shape, got "
                f"{tuple(self.input_ids.shape)}, "
                f"{tuple(self.attention_mask.shape)} and "
                f"{tuple(self.labels.shape)}"
            )
            raise ValueError(msg)

    def __len__(self) -> int:
        """Return the number of examples (rows) in the batch."""
        return int(self.input_ids.shape[0])

    def to(self, device: torch.device | str) -> MaskedLMBatch:
        """Return a copy of this batch with all tensors moved to ``device``."""
        return MaskedLMBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            labels=self.labels.to(device),
        )


def mask_tokens(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    mask_id: int,
    vocab_size: int,
    special_token_ids: Iterable[int] = (),
    probability: float = 0.15,
    generator: torch.Generator | None = None,
) -> MaskedLMBatch:
    """Apply BERT-style masking to a padded batch of token ids.

    Parameters
    ----------
    input_ids : torch.Tensor
        Long tensor of shape ``(batch_size, seq_len)`` of padded token ids.
    attention_mask : torch.Tensor
        Long tensor of the same shape, ``1`` at real tokens and ``0`` at padding.
    mask_id : int
        The id of the ``<mask>`` token substituted at (most) chosen positions.
    vocab_size : int
        Size of the vocabulary; random replacements are drawn from
        ``[0, vocab_size)``.
    special_token_ids : Iterable[int], optional
        Ids never selected for masking (e.g. padding, unknown, mask). Padding is
        already excluded via ``attention_mask``; list the others here.
    probability : float, default 0.15
        Fraction of maskable positions selected for supervision.
    generator : torch.Generator, optional
        Seeded RNG for reproducible masking. When ``None``, the global RNG is
        used.

    Returns
    -------
    MaskedLMBatch
        The corrupted inputs, the unchanged attention mask, and per-token targets.

    Examples
    --------
    >>> ids = torch.arange(2, 12).reshape(2, 5)
    >>> mask = torch.ones(2, 5, dtype=torch.long)
    >>> g = torch.Generator().manual_seed(0)
    >>> batch = mask_tokens(ids, mask, mask_id=1, vocab_size=12, generator=g)
    >>> batch.labels.shape
    torch.Size([2, 5])
    """
    # Positions eligible for masking: real tokens that are not special tokens.
    maskable = attention_mask.bool()
    for token_id in special_token_ids:
        maskable &= input_ids != token_id

    selection_prob = torch.full(input_ids.shape, probability) * maskable
    selected = torch.bernoulli(selection_prob, generator=generator).bool()

    # Targets: the original id where supervised, IGNORE_INDEX elsewhere.
    labels = input_ids.clone()
    labels[~selected] = IGNORE_INDEX

    corrupted = input_ids.clone()

    # 80% of selected positions -> <mask>.
    replace_with_mask = (
        torch.bernoulli(torch.full(input_ids.shape, 0.8), generator=generator).bool()
        & selected
    )
    corrupted[replace_with_mask] = mask_id

    # Of the remainder, half (10% overall) -> a random token; the rest unchanged.
    replace_with_random = (
        torch.bernoulli(torch.full(input_ids.shape, 0.5), generator=generator).bool()
        & selected
        & ~replace_with_mask
    )
    random_tokens = torch.randint(
        vocab_size, input_ids.shape, generator=generator, dtype=input_ids.dtype
    )
    corrupted[replace_with_random] = random_tokens[replace_with_random]

    return MaskedLMBatch(
        input_ids=corrupted, attention_mask=attention_mask, labels=labels
    )
