"""Batched, model-ready tensors produced by collation.

Design Principles
------------------
- ``Batch`` is a thin container over three aligned PyTorch tensors. It carries
  no collation logic (that lives in
  :func:`~polaris.collation.collator.collate`) and no model logic.
- It is the boundary object between the tokenization/collation layers and the
  model layer: a model consumes a ``Batch`` and nothing more.
- Unlike Polaris' other value objects it is **not** a hashable, value-equal
  dataclass, because PyTorch tensors do not support value equality
  (``tensor == tensor`` returns a tensor, not a bool). It is therefore a
  frozen, slotted dataclass with ``eq=False`` — immutable in its fields, but
  compared by identity.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["Batch"]


@dataclass(frozen=True, slots=True, eq=False)
class Batch:
    """A batch of encoded examples as aligned PyTorch tensors.

    Parameters
    ----------
    input_ids : torch.Tensor
        Long tensor of shape ``(batch_size, seq_len)`` holding token ids, with
        padding positions filled by the vocabulary's pad id.
    attention_mask : torch.Tensor
        Long tensor of shape ``(batch_size, seq_len)`` with ``1`` at real token
        positions and ``0`` at padding positions.
    labels : torch.Tensor
        Long tensor of shape ``(batch_size,)`` holding the integer label of each
        example.

    Raises
    ------
    ValueError
        If ``input_ids`` and ``attention_mask`` do not have the same shape, or
        if ``labels`` does not have one entry per row.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor

    def __post_init__(self) -> None:
        """Validate that the three tensors are mutually aligned."""
        if self.input_ids.shape != self.attention_mask.shape:
            msg = (
                "input_ids and attention_mask must have the same shape, got "
                f"{tuple(self.input_ids.shape)} and "
                f"{tuple(self.attention_mask.shape)}"
            )
            raise ValueError(msg)

        if self.input_ids.shape[0] != self.labels.shape[0]:
            msg = (
                "labels must have one entry per row, got "
                f"{self.labels.shape[0]} labels for "
                f"{self.input_ids.shape[0]} rows"
            )
            raise ValueError(msg)

    def __len__(self) -> int:
        """Return the number of examples in the batch.

        Returns
        -------
        int
            The batch size (number of rows).
        """
        return int(self.input_ids.shape[0])
