"""A transformer-encoder text classifier, from Polaris' from-scratch blocks.

The v0.5 model: the shared
:class:`~polaris.models.transformer_encoder.TransformerEncoder` trunk (token
embeddings + positional encoding + encoder blocks + final norm) followed by
mask-aware mean pooling and a linear head. It consumes a
:class:`~polaris.collation.batch.Batch` and returns logits — a drop-in
replacement for the v0.4 baseline in the same training loop.

Since v0.11 the trunk is a reusable module (:class:`TransformerEncoder`) shared
with the masked-language model, so a pretrained trunk transfers into this
classifier by copying ``encoder`` weights.
"""

from __future__ import annotations

import torch
from torch import nn

from polaris.collation.batch import Batch
from polaris.models.transformer_encoder import TransformerEncoder

__all__ = ["TransformerEncoderClassifier"]


class TransformerEncoderClassifier(nn.Module):
    """A transformer encoder for text classification.

    Parameters
    ----------
    vocab_size : int
        Number of tokens in the vocabulary.
    num_classes : int
        Number of output classes.
    embed_dim : int, default 128
        The model dimension. Must be divisible by ``num_heads``.
    num_heads : int, default 4
        Attention heads per block.
    num_layers : int, default 2
        Number of stacked encoder blocks.
    ff_dim : int, default 256
        Hidden size of each block's feed-forward network.
    max_len : int, default 512
        Maximum supported sequence length.
    dropout : float, default 0.1
        Dropout probability.
    pad_id : int, default 0
        The padding id; its embedding is fixed to zeros and excluded from the
        pooled mean via the attention mask.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        max_len: int = 512,
        dropout: float = 0.1,
        pad_id: int = 0,
        pretrained_embeddings: torch.Tensor | None = None,
        freeze_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            max_len=max_len,
            dropout=dropout,
            pad_id=pad_id,
            pretrained_embeddings=pretrained_embeddings,
            freeze_embeddings=freeze_embeddings,
        )
        self.classifier = nn.Linear(self.encoder.embed_dim, num_classes)

    def forward(self, batch: Batch) -> torch.Tensor:
        """Compute class logits for a batch.

        Parameters
        ----------
        batch : Batch
            The padded input batch.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch_size, num_classes)``.
        """
        hidden = self.encoder(batch.input_ids, batch.attention_mask)  # (B, S, E)

        # Mask-aware mean pool over the real tokens.
        mask_f = batch.attention_mask.unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * mask_f).sum(dim=1)
        token_counts = mask_f.sum(dim=1).clamp(min=1.0)
        pooled = summed / token_counts

        logits: torch.Tensor = self.classifier(pooled)
        return logits
