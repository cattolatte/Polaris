"""A text embedding model — the bi-encoder tower (v1.2.0).

Where :class:`~polaris.models.transformer_classifier.TransformerEncoderClassifier`
pools the trunk's hidden states and predicts a *class*, ``TextEmbedder`` pools them
and emits the **pooled representation itself** — a single vector per text. This is
the tower of a bi-encoder: a query and a passage are encoded *independently* to
vectors and compared by cosine similarity (see
:func:`~polaris.training.losses.info_nce_loss`).

It shares the same :class:`~polaris.models.transformer_encoder.TransformerEncoder`
trunk (stored as ``encoder``) as the classifier and the masked-language model, so a
pretrained trunk transfers in by a plain ``state_dict`` copy
(:meth:`~polaris.pretraining.model.MaskedLanguageModel.transfer_encoder_to`).

Design Principles
-----------------
- No task head, no loss: the model only *represents*. Contrastive objectives live
  in :mod:`polaris.training.losses`, mirroring how classifiers leave loss to the
  trainer.
- Optional linear projection (to shrink/reshape the embedding) and optional L2
  normalization (so cosine similarity is a plain dot product) — both off the trunk,
  both pure PyTorch primitives.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from polaris.collation.batch import Batch
from polaris.models.pooling import mean_pool
from polaris.models.transformer_encoder import TransformerEncoder

__all__ = ["TextEmbedder"]


class TextEmbedder(nn.Module):
    """Encode text to a single (optionally L2-normalized) embedding.

    Parameters
    ----------
    vocab_size : int
        Number of tokens in the vocabulary.
    embed_dim : int, default 128
        The trunk's model dimension. Must be divisible by ``num_heads``.
    num_heads : int, default 4
        Attention heads per block.
    num_layers : int, default 2
        Number of stacked encoder blocks.
    ff_dim : int, default 256
        Hidden size of each block's feed-forward network.
    max_len : int, default 512
        Maximum supported sequence length.
    dropout : float, default 0.1
        Dropout probability inside the trunk.
    pad_id : int, default 0
        The padding id; its positions are excluded from the pooled mean.
    projection_dim : int, optional
        If given, a linear layer projects the pooled ``embed_dim`` vector to this
        dimension. When ``None`` (default) the embedding dimension is ``embed_dim``.
    normalize : bool, default True
        If ``True``, L2-normalize the output so cosine similarity is a dot product.
    pretrained_embeddings : torch.Tensor, optional
        A ``(vocab_size, embed_dim)`` matrix to initialize the trunk's embedding.
    freeze_embeddings : bool, default False
        If ``True`` and ``pretrained_embeddings`` is given, the embedding is frozen.

    Attributes
    ----------
    embedding_dim : int
        The dimensionality of the emitted embedding (``projection_dim`` if set,
        else the trunk's ``embed_dim``).
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
        projection_dim: int | None = None,
        normalize: bool = True,
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
        self.normalize = normalize
        if projection_dim is not None:
            self.projection: nn.Linear | None = nn.Linear(
                self.encoder.embed_dim, projection_dim
            )
            self.embedding_dim = projection_dim
        else:
            self.projection = None
            self.embedding_dim = self.encoder.embed_dim

    def encode(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode padded token ids into pooled embeddings.

        Parameters
        ----------
        input_ids : torch.Tensor
            Long tensor of shape ``(batch_size, seq_len)`` of padded token ids.
        attention_mask : torch.Tensor
            Mask of the same shape, ``1`` at real tokens and ``0`` at padding.

        Returns
        -------
        torch.Tensor
            Embeddings of shape ``(batch_size, embedding_dim)``, L2-normalized when
            ``normalize`` is ``True``.
        """
        hidden = self.encoder(input_ids, attention_mask)  # (B, S, E)
        pooled = mean_pool(hidden, attention_mask)  # (B, E)
        if self.projection is not None:
            pooled = self.projection(pooled)  # (B, projection_dim)
        if self.normalize:
            pooled = F.normalize(pooled, p=2.0, dim=-1)
        return pooled

    def forward(self, batch: Batch) -> torch.Tensor:
        """Encode a :class:`Batch` into pooled embeddings.

        Parameters
        ----------
        batch : Batch
            The padded input batch (its ``labels`` are ignored).

        Returns
        -------
        torch.Tensor
            Embeddings of shape ``(batch_size, embedding_dim)``.
        """
        return self.encode(batch.input_ids, batch.attention_mask)
