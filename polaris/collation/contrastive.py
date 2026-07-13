"""Collate anchor/positive (and optional hard-negative) pairs for contrastive
training.

A bi-encoder embedder is trained by pulling an *anchor* (e.g. a query) and its
*positive* (a relevant passage) together while pushing everything else apart. This
module packs a group of such pairs into aligned :class:`~polaris.collation.batch.Batch`
tensors, optionally with explicit **hard negatives** per anchor.

Design Principles
-----------------
- Reuse, don't reinvent: each side (anchors, positives, flattened negatives) is
  padded by the existing :func:`~polaris.collation.collator.collate`; contrastive
  training carries no labels, so a dummy label is supplied and ignored.
- Hard negatives are stored flattened as one ``(B * num_negatives, T)`` batch plus
  the ``num_negatives`` count, so the training driver can reshape them to
  ``(B, num_negatives, D)`` after embedding.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from polaris.collation.batch import Batch
from polaris.collation.collator import collate
from polaris.tokenizers.encoding import Encoding

__all__ = ["ContrastiveBatch", "collate_contrastive"]

# A pair, optionally with a list of hard negatives for the anchor.
type ContrastiveSample = (
    tuple[Encoding, Encoding] | tuple[Encoding, Encoding, Sequence[Encoding]]
)


@dataclass(frozen=True, slots=True, eq=False)
class ContrastiveBatch:
    """Aligned batches for contrastive training.

    Parameters
    ----------
    anchor : Batch
        The anchors (e.g. queries), shape ``(B, T_a)``.
    positive : Batch
        Each anchor's positive (e.g. relevant passage), shape ``(B, T_p)``.
    negatives : Batch, optional
        Explicit hard negatives, flattened to ``(B * num_negatives, T_n)`` — anchor
        ``i``'s negatives occupy rows ``[i * num_negatives, (i + 1) * num_negatives)``.
        ``None`` when only in-batch negatives are used.
    num_negatives : int, default 0
        Hard negatives per anchor (``0`` when ``negatives`` is ``None``).
    """

    anchor: Batch
    positive: Batch
    negatives: Batch | None = None
    num_negatives: int = 0

    def __len__(self) -> int:
        """Return the number of anchor/positive pairs (the batch size)."""
        return len(self.anchor)

    def to(self, device: torch.device | str) -> ContrastiveBatch:
        """Return a copy with every batch moved to ``device``."""
        return ContrastiveBatch(
            anchor=self.anchor.to(device),
            positive=self.positive.to(device),
            negatives=None if self.negatives is None else self.negatives.to(device),
            num_negatives=self.num_negatives,
        )


def collate_contrastive(
    samples: Sequence[ContrastiveSample],
    *,
    pad_id: int,
    max_length: int | None = None,
) -> ContrastiveBatch:
    """Collate ``(anchor, positive)`` or ``(anchor, positive, hard_negatives)`` pairs.

    Parameters
    ----------
    samples : Sequence[ContrastiveSample]
        The pairs to collate. Either all are ``(anchor, positive)`` two-tuples, or
        all are ``(anchor, positive, hard_negatives)`` three-tuples with the **same
        number** of hard negatives each.
    pad_id : int
        The vocabulary id used to fill padding positions.
    max_length : int, optional
        If given, each sequence is truncated to at most this many tokens.

    Returns
    -------
    ContrastiveBatch
        The aligned anchor / positive (/ negatives) batches.

    Raises
    ------
    ValueError
        If ``samples`` is empty, if some samples carry hard negatives and others
        do not, or if the hard-negative counts are not uniform.
    """
    if not samples:
        msg = "cannot collate an empty batch"
        raise ValueError(msg)

    arities = {len(sample) for sample in samples}
    if len(arities) != 1:
        msg = "every sample must have the same arity (all pairs, or all triples)"
        raise ValueError(msg)

    anchors: list[Encoding] = [sample[0] for sample in samples]
    positives: list[Encoding] = [sample[1] for sample in samples]

    anchor_batch = collate(
        [(encoding, 0) for encoding in anchors], pad_id=pad_id, max_length=max_length
    )
    positive_batch = collate(
        [(encoding, 0) for encoding in positives], pad_id=pad_id, max_length=max_length
    )

    if arities == {2}:
        return ContrastiveBatch(anchor=anchor_batch, positive=positive_batch)

    # Triples: collect and validate the per-anchor hard negatives.
    negatives_per_sample = [sample[2] for sample in samples if len(sample) == 3]
    num_negatives = len(negatives_per_sample[0])
    if num_negatives == 0:
        msg = "hard-negative triples must supply at least one negative each"
        raise ValueError(msg)
    if any(len(negs) != num_negatives for negs in negatives_per_sample):
        msg = "every sample must supply the same number of hard negatives"
        raise ValueError(msg)

    flat_negatives = [
        neg for sample_negs in negatives_per_sample for neg in sample_negs
    ]
    negatives_batch = collate(
        [(encoding, 0) for encoding in flat_negatives],
        pad_id=pad_id,
        max_length=max_length,
    )
    return ContrastiveBatch(
        anchor=anchor_batch,
        positive=positive_batch,
        negatives=negatives_batch,
        num_negatives=num_negatives,
    )
