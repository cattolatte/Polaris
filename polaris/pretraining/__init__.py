"""
Self-supervised pretraining subsystem for Polaris.

Masked-language-model pretraining (the BERT recipe), implemented from scratch:
mask a fraction of the tokens in unlabeled text and train the shared transformer
trunk to recover them, then transfer that pretrained trunk into a classifier for
fine-tuning. Nothing is downloaded — "pretraining" here is a training method, not
a borrowed model.
"""

from __future__ import annotations

from polaris.pretraining.loop import PretrainEpoch, pretrain
from polaris.pretraining.masking import IGNORE_INDEX, MaskedLMBatch, mask_tokens
from polaris.pretraining.model import MaskedLanguageModel

__all__ = [
    "IGNORE_INDEX",
    "MaskedLMBatch",
    "MaskedLanguageModel",
    "PretrainEpoch",
    "mask_tokens",
    "pretrain",
]
