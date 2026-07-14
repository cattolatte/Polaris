"""The shared transformer encoder trunk, from Polaris' from-scratch blocks.

This is the reusable body of a transformer: token embeddings + positional
encoding, a stack of encoder blocks, and a final layer norm, producing one
contextual hidden vector per input position. It is deliberately *headless* — it
returns per-token hidden states and applies no pooling or task head.

Two concrete consumers justify extracting it (ADR-0004): the
:class:`~polaris.models.transformer_classifier.TransformerEncoderClassifier`
pools these hidden states for classification, and the masked-language model
(v0.11) projects them back to the vocabulary for pretraining. Both embed the
*same* module, so a pretrained trunk transfers into a classifier by a plain
``state_dict`` copy.

Design Principles
-----------------
- Headless by design: the trunk owns representation, not any task. Pooling and
  output projections live in the head modules that wrap it.
- Built on Polaris' own from-scratch blocks (``LayerNorm``, positional encoding,
  ``TransformerEncoderBlock``); the framework supplies only autograd and the
  embedding/linear parameter containers (ADR-0003).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import nn

from polaris.models.transformer import (
    LayerNorm,
    SinusoidalPositionalEncoding,
    TransformerEncoderBlock,
)

__all__ = ["HasEncoder", "TransformerEncoder"]


@runtime_checkable
class HasEncoder(Protocol):
    """A model that wraps a :class:`TransformerEncoder` trunk as ``encoder``.

    The structural type shared by every head that composes the trunk — the
    classifier, the masked-language model, and the :class:`TextEmbedder`. It
    exists so a pretrained trunk can be transferred between any two of them by a
    plain ``state_dict`` copy (see
    :meth:`~polaris.pretraining.model.MaskedLanguageModel.transfer_encoder_to`),
    without those heads needing a common base class.
    """

    encoder: TransformerEncoder


class TransformerEncoder(nn.Module):
    """Embed, contextualize, and normalize — returning per-token hidden states.

    Parameters
    ----------
    vocab_size : int
        Number of tokens in the vocabulary.
    embed_dim : int, default 128
        The model dimension. Must be divisible by ``num_heads``. Ignored (and
        derived from the matrix) when ``pretrained_embeddings`` is given.
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
        The padding id; its embedding row is fixed to zeros and its positions are
        excluded from attention via the batch's attention mask.
    pretrained_embeddings : torch.Tensor, optional
        A ``(vocab_size, embed_dim)`` matrix to initialize the embedding from.
    freeze_embeddings : bool, default False
        If ``True`` and ``pretrained_embeddings`` is given, the embedding is not
        updated during training.

    Attributes
    ----------
    embed_dim : int
        The resolved model dimension (derived from ``pretrained_embeddings`` when
        one is supplied).
    """

    def __init__(
        self,
        *,
        vocab_size: int,
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
        self.embed_dim = embed_dim
        # Optional segment (token-type) embeddings for sentence-pair inputs. Two
        # segments (A/B). Zero-initialized so a trunk that never sees token type
        # ids (or that was transferred from one) starts perfectly neutral, and so
        # the addition is a no-op until fine-tuning learns it.
        self.token_type_embedding = nn.Embedding(2, embed_dim)
        nn.init.zeros_(self.token_type_embedding.weight)
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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Contextualize token ids into per-token hidden states.

        Parameters
        ----------
        input_ids : torch.Tensor
            Long tensor of shape ``(batch_size, seq_len)`` of padded token ids.
        attention_mask : torch.Tensor
            Long tensor of the same shape, ``1`` at real tokens and ``0`` at
            padding. Padding positions are excluded from attention.
        token_type_ids : torch.Tensor, optional
            Long tensor of the same shape with ``0`` for segment A and ``1`` for
            segment B (sentence-pair inputs). When ``None`` (the default) no
            segment embedding is added, so single-segment call sites are unchanged.

        Returns
        -------
        torch.Tensor
            Hidden states of shape ``(batch_size, seq_len, embed_dim)``.
        """
        hidden: torch.Tensor = self.embedding(input_ids)
        if token_type_ids is not None:
            hidden = hidden + self.token_type_embedding(token_type_ids)
        hidden = self.positional(hidden)
        hidden = self.dropout(hidden)

        for block in self.blocks:
            hidden = block(hidden, key_padding_mask=attention_mask)
        normalized: torch.Tensor = self.norm(hidden)
        return normalized
