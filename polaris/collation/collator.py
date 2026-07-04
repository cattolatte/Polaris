"""Collate encoded examples into a padded tensor :class:`Batch`.

This is the seam between the tokenizer and the model. Tokenizers produce
variable-length :class:`~polaris.tokenizers.encoding.Encoding` objects; models
consume rectangular tensors. ``collate`` bridges the two by padding a group of
encodings to a common length and stacking them into aligned tensors.

Design Principles
------------------
- Pure function: no state, no model or tokenizer knowledge beyond the pad id it
  is given.
- Padding uses the vocabulary's pad id; the attention mask records which
  positions are real. Padding never changes which positions a model should
  attend to.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from polaris.collation.batch import Batch
from polaris.tokenizers.encoding import Encoding

__all__ = ["collate"]


def collate(
    samples: Sequence[tuple[Encoding, int]],
    *,
    pad_id: int,
    max_length: int | None = None,
) -> Batch:
    """Pad a group of ``(Encoding, label)`` pairs into a :class:`Batch`.

    Every sequence is padded on the right to the longest sequence in the group
    (after optional truncation to ``max_length``). The attention mask marks the
    real tokens with ``1`` and the padding with ``0``.

    Parameters
    ----------
    samples : Sequence[tuple[Encoding, int]]
        The examples to collate, each an encoding paired with its integer label.
    pad_id : int
        The vocabulary id used to fill padding positions.
    max_length : int, optional
        If given, each sequence is truncated to at most this many tokens before
        padding. Must be at least ``1``.

    Returns
    -------
    Batch
        The padded, aligned tensors for the group.

    Raises
    ------
    ValueError
        If ``samples`` is empty, or if ``max_length`` is less than ``1``.

    Examples
    --------
    >>> from polaris.tokenizers.encoding import Encoding
    >>> batch = collate(
    ...     [
    ...         (Encoding(ids=(4, 5, 6), tokens=("a", "b", "c")), 1),
    ...         (Encoding(ids=(7,), tokens=("d",)), 0),
    ...     ],
    ...     pad_id=0,
    ... )
    >>> batch.input_ids.tolist()
    [[4, 5, 6], [7, 0, 0]]
    >>> batch.attention_mask.tolist()
    [[1, 1, 1], [1, 0, 0]]
    >>> batch.labels.tolist()
    [1, 0]
    """
    if not samples:
        msg = "cannot collate an empty batch"
        raise ValueError(msg)

    if max_length is not None and max_length < 1:
        msg = f"max_length must be at least 1, got {max_length}"
        raise ValueError(msg)

    id_sequences: list[list[int]] = []
    for encoding, _label in samples:
        ids = list(encoding.ids)
        if max_length is not None:
            ids = ids[:max_length]
        id_sequences.append(ids)

    longest = max(len(ids) for ids in id_sequences)
    batch_size = len(samples)

    input_ids = torch.full((batch_size, longest), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, longest), dtype=torch.long)

    for row, ids in enumerate(id_sequences):
        if ids:
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row, : len(ids)] = 1

    labels = torch.tensor([label for _encoding, label in samples], dtype=torch.long)

    return Batch(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
