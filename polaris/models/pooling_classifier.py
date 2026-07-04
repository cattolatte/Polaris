"""A minimal mean-pooling text classifier.

This is the first Polaris model: the simplest architecture that proves the
end-to-end pipeline works. It embeds token ids, averages the embeddings over the
real (non-padding) positions of each sequence, and applies a linear
classification head.

Design Principles
-----------------
- Built on PyTorch primitives (``nn.Embedding``, ``nn.Linear``) used as
  substrate. The only hand-written, model-defining logic is the mask-aware mean
  pooling — there is deliberately no attention or transformer here (that arrives
  in v0.5).
- The model consumes a :class:`~polaris.collation.batch.Batch` and returns raw
  logits. Loss is applied by the training loop, not the model.
"""

from __future__ import annotations

import torch
from torch import nn

from polaris.collation.batch import Batch

__all__ = ["MeanPoolingClassifier"]


class MeanPoolingClassifier(nn.Module):
    """Embed, mean-pool over real tokens, then classify.

    Parameters
    ----------
    vocab_size : int
        Number of tokens in the vocabulary (embedding rows).
    num_classes : int
        Number of output classes.
    embedding_dim : int, default 64
        Dimensionality of the token embeddings.
    pad_id : int, default 0
        The padding id. Its embedding row is fixed to zeros and excluded from the
        mean by the attention mask.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int = 64,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.classifier = nn.Linear(embedding_dim, num_classes)

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
        embedded: torch.Tensor = self.embedding(batch.input_ids)  # (B, S, E)

        # Mask out padding, then average over the real tokens only.
        mask = batch.attention_mask.unsqueeze(-1).to(embedded.dtype)  # (B, S, 1)
        summed = (embedded * mask).sum(dim=1)  # (B, E)
        token_counts = mask.sum(dim=1).clamp(min=1.0)  # (B, 1), avoid div-by-zero
        pooled = summed / token_counts  # (B, E)

        logits: torch.Tensor = self.classifier(pooled)  # (B, num_classes)
        return logits
