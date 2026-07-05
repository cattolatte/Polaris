"""A transformer-encoder text classifier, from Polaris' from-scratch blocks.

The v0.5 model: token embeddings + positional encoding, a stack of transformer
encoder blocks, mask-aware mean pooling, and a linear head. It consumes a
:class:`~polaris.collation.batch.Batch` and returns logits — a drop-in
replacement for the v0.4 baseline in the same training loop.
"""

from __future__ import annotations

import torch
from torch import nn

from polaris.collation.batch import Batch
from polaris.models.transformer import (
    LayerNorm,
    SinusoidalPositionalEncoding,
    TransformerEncoderBlock,
)

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
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(  # type: ignore[no-untyped-call]  # torch stub is untyped
                pretrained_embeddings, freeze=freeze_embeddings, padding_idx=pad_id
            )
            embed_dim = pretrained_embeddings.shape[1]
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.positional = SinusoidalPositionalEncoding(
            embed_dim=embed_dim, max_len=max_len
        )
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

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
        hidden: torch.Tensor = self.embedding(batch.input_ids)
        hidden = self.positional(hidden)
        hidden = self.dropout(hidden)

        mask = batch.attention_mask
        for block in self.blocks:
            hidden = block(hidden, key_padding_mask=mask)
        hidden = self.norm(hidden)

        # Mask-aware mean pool over the real tokens.
        mask_f = mask.unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * mask_f).sum(dim=1)
        token_counts = mask_f.sum(dim=1).clamp(min=1.0)
        pooled = summed / token_counts

        logits: torch.Tensor = self.classifier(pooled)
        return logits
