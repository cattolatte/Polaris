"""Collate sentence pairs into ``[CLS] a [SEP] b [SEP]`` batches.

A cross-encoder reads *both* texts of a pair jointly, so the two segments must be
packed into one sequence with a separator and per-token segment ids. This module
builds that layout — the input a sentence-pair classifier consumes.

Design Principles
-----------------
- Owns its own padding: it must pad three aligned tensors (ids, mask, and segment
  ids), so it cannot reuse the single-label
  :func:`~polaris.collation.collator.collate`.
- Truncation is longest-first over the two segments, so both segments and all three
  special tokens always survive.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from polaris.tokenizers.encoding import Encoding

__all__ = ["PairBatch", "collate_pairs"]

# Number of special tokens in [CLS] a [SEP] b [SEP].
_NUM_SPECIALS = 3


@dataclass(frozen=True, slots=True, eq=False)
class PairBatch:
    """A batch of sentence pairs as aligned tensors.

    Parameters
    ----------
    input_ids : torch.Tensor
        Long tensor ``(B, T)`` of the packed ``[CLS] a [SEP] b [SEP]`` ids, padded.
    attention_mask : torch.Tensor
        Long tensor ``(B, T)``, ``1`` at real tokens and ``0`` at padding.
    token_type_ids : torch.Tensor
        Long tensor ``(B, T)``, ``0`` for segment A (``[CLS] a [SEP]``) and ``1``
        for segment B (``b [SEP]``); ``0`` at padding.
    labels : torch.Tensor
        Long tensor ``(B,)`` of per-pair labels.

    Raises
    ------
    ValueError
        If the three ``(B, T)`` tensors do not share a shape, or ``labels`` does
        not have one entry per row.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor
    labels: torch.Tensor

    def __post_init__(self) -> None:
        """Validate that the tensors are mutually aligned."""
        if not (
            self.input_ids.shape
            == self.attention_mask.shape
            == self.token_type_ids.shape
        ):
            msg = (
                "input_ids, attention_mask and token_type_ids must share a shape, "
                f"got {tuple(self.input_ids.shape)}, "
                f"{tuple(self.attention_mask.shape)} and "
                f"{tuple(self.token_type_ids.shape)}"
            )
            raise ValueError(msg)
        if self.input_ids.shape[0] != self.labels.shape[0]:
            msg = (
                f"labels must have one entry per row, got {self.labels.shape[0]} "
                f"labels for {self.input_ids.shape[0]} rows"
            )
            raise ValueError(msg)

    def __len__(self) -> int:
        """Return the number of pairs (rows) in the batch."""
        return int(self.input_ids.shape[0])

    def to(self, device: torch.device | str) -> PairBatch:
        """Return a copy of this batch with all tensors moved to ``device``."""
        return PairBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            token_type_ids=self.token_type_ids.to(device),
            labels=self.labels.to(device),
        )


def _truncate_pair(a: list[int], b: list[int], budget: int) -> None:
    """Trim ``a`` and ``b`` in place, longest-first, until they fit ``budget``."""
    while len(a) + len(b) > budget:
        if len(a) >= len(b):
            a.pop()
        else:
            b.pop()


def collate_pairs(
    samples: Sequence[tuple[Encoding, Encoding, int]],
    *,
    pad_id: int,
    cls_id: int,
    sep_id: int,
    max_length: int | None = None,
) -> PairBatch:
    """Collate ``(text_a, text_b, label)`` triples into ``[CLS] a [SEP] b [SEP]``.

    Parameters
    ----------
    samples : Sequence[tuple[Encoding, Encoding, int]]
        The pairs to collate, each two encodings and an integer label.
    pad_id : int
        The id used to fill padding positions.
    cls_id : int
        The ``[CLS]`` token id (prepended).
    sep_id : int
        The ``[SEP]`` token id (after each segment).
    max_length : int, optional
        If given, each packed sequence is truncated (longest segment first) to at
        most this many tokens, including the three special tokens. Must be at least
        ``3``.

    Returns
    -------
    PairBatch
        The padded, aligned pair tensors.

    Raises
    ------
    ValueError
        If ``samples`` is empty, or ``max_length`` is less than ``3``.
    """
    if not samples:
        msg = "cannot collate an empty batch"
        raise ValueError(msg)
    if max_length is not None and max_length < _NUM_SPECIALS:
        msg = f"max_length must be at least {_NUM_SPECIALS}, got {max_length}"
        raise ValueError(msg)

    rows_ids: list[list[int]] = []
    rows_types: list[list[int]] = []
    for encoding_a, encoding_b, _label in samples:
        a = list(encoding_a.ids)
        b = list(encoding_b.ids)
        if max_length is not None:
            _truncate_pair(a, b, max_length - _NUM_SPECIALS)
        ids = [cls_id, *a, sep_id, *b, sep_id]
        # Segment A is [CLS] a [SEP]; segment B is b [SEP].
        types = [0] * (len(a) + 2) + [1] * (len(b) + 1)
        rows_ids.append(ids)
        rows_types.append(types)

    longest = max(len(ids) for ids in rows_ids)
    batch_size = len(samples)

    input_ids = torch.full((batch_size, longest), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, longest), dtype=torch.long)
    token_type_ids = torch.zeros((batch_size, longest), dtype=torch.long)
    for row, (ids, types) in enumerate(zip(rows_ids, rows_types, strict=True)):
        length = len(ids)
        input_ids[row, :length] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, :length] = 1
        token_type_ids[row, :length] = torch.tensor(types, dtype=torch.long)

    labels = torch.tensor([label for _a, _b, label in samples], dtype=torch.long)
    return PairBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        labels=labels,
    )
