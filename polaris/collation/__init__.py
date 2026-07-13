"""
Collation subsystem for Polaris.

Turns variable-length tokenizer output into padded, model-ready tensor batches.
"""

from __future__ import annotations

from polaris.collation.batch import Batch
from polaris.collation.collator import collate
from polaris.collation.contrastive import ContrastiveBatch, collate_contrastive

__all__ = [
    "Batch",
    "ContrastiveBatch",
    "collate",
    "collate_contrastive",
]
