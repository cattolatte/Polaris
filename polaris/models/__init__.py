"""
Models subsystem for Polaris.
"""

from __future__ import annotations

from polaris.models.embedder import TextEmbedder
from polaris.models.pair_classifier import SentencePairClassifier
from polaris.models.pooling import mean_pool
from polaris.models.pooling_classifier import MeanPoolingClassifier
from polaris.models.transformer_classifier import TransformerEncoderClassifier
from polaris.models.transformer_encoder import HasEncoder, TransformerEncoder

__all__ = [
    "HasEncoder",
    "MeanPoolingClassifier",
    "SentencePairClassifier",
    "TextEmbedder",
    "TransformerEncoder",
    "TransformerEncoderClassifier",
    "mean_pool",
]
