"""A sentence-pair cross-encoder (v1.3.0).

Where the bi-encoder :class:`~polaris.models.embedder.TextEmbedder` encodes two
texts *independently*, a cross-encoder encodes them **jointly** — full
cross-attention over ``[CLS] a [SEP] b [SEP]`` — then pools and scores with a
linear head. Reading both texts together is far more accurate for top-of-ranking
precision and entailment, at the cost of not being precomputable.

One head serves three tasks by its ``num_classes``: a relevance score
(``num_classes=1``), a binary gate (``2``), or NLI (``3``). It shares the same
:class:`~polaris.models.transformer_encoder.TransformerEncoder` trunk as every
other head, so a pretrained trunk transfers in via ``transfer_encoder_to``.

Design Principles
-----------------
- The joint encoding uses the encoder's optional segment (token-type) embeddings,
  so the model can tell segment A from segment B.
- Pooling is ``"cls"`` (the first position) or ``"mean"`` (mask-aware mean); the
  loss stays with the trainer, as with the other classifiers.
"""

from __future__ import annotations

import torch
from torch import nn

from polaris.collation.pairs import PairBatch
from polaris.models.pooling import mean_pool
from polaris.models.transformer_encoder import TransformerEncoder

__all__ = ["SentencePairClassifier"]


class SentencePairClassifier(nn.Module):
    """Cross-encoder: joint-encode a sentence pair, pool, and score.

    Parameters
    ----------
    vocab_size : int
        Number of tokens in the vocabulary.
    num_classes : int
        Number of output logits. ``1`` for a relevance/ranking score, ``2`` for a
        binary gate, ``3`` for NLI — any positive integer.
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
        The padding id.
    pooling : {"cls", "mean"}, default "cls"
        How to reduce the token hidden states to one vector: the ``[CLS]`` position
        or a mask-aware mean over real tokens.
    pretrained_embeddings : torch.Tensor, optional
        A ``(vocab_size, embed_dim)`` matrix to initialize the trunk's embedding.
    freeze_embeddings : bool, default False
        If ``True`` and ``pretrained_embeddings`` is given, the embedding is frozen.

    Raises
    ------
    ValueError
        If ``pooling`` is not ``"cls"`` or ``"mean"``.
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
        pooling: str = "cls",
        pretrained_embeddings: torch.Tensor | None = None,
        freeze_embeddings: bool = False,
    ) -> None:
        super().__init__()
        if pooling not in ("cls", "mean"):
            msg = f"pooling must be 'cls' or 'mean', got {pooling!r}"
            raise ValueError(msg)
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
        self.pooling = pooling
        self.classifier = nn.Linear(self.encoder.embed_dim, num_classes)

    def forward(self, batch: PairBatch) -> torch.Tensor:
        """Score a batch of sentence pairs.

        Parameters
        ----------
        batch : PairBatch
            The packed ``[CLS] a [SEP] b [SEP]`` batch with segment ids.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch_size, num_classes)``.
        """
        hidden = self.encoder(
            batch.input_ids, batch.attention_mask, batch.token_type_ids
        )  # (B, T, E)
        if self.pooling == "cls":
            pooled = hidden[:, 0]  # the [CLS] position
        else:
            pooled = mean_pool(hidden, batch.attention_mask)
        logits: torch.Tensor = self.classifier(pooled)
        return logits
