"""
Collation subsystem for Polaris.

Turns variable-length tokenizer output into padded, model-ready tensor batches.
"""

from __future__ import annotations

from polaris.collation.batch import Batch
from polaris.collation.collator import collate

__all__ = [
    "Batch",
    "collate",
]
