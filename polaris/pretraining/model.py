"""The masked-language model: the shared trunk plus a vocabulary head.

The pretraining counterpart of
:class:`~polaris.models.transformer_classifier.TransformerEncoderClassifier`.
Both wrap the *same*
:class:`~polaris.models.transformer_encoder.TransformerEncoder` trunk; where the
classifier pools the trunk's hidden states and predicts a class, this model
projects each hidden state back to the vocabulary and predicts the original token.

Because the two models share the ``encoder`` submodule, a trunk pretrained here
transfers into a classifier by copying ``encoder`` weights — the whole point of
pretraining (see :meth:`transfer_encoder_to`).

Design Principles
-----------------
- The head is a single linear projection to vocabulary logits. Weight tying with
  the embedding is a known refinement we deliberately skip for readability.
- ``forward`` returns raw per-token logits; the cross-entropy loss (over masked
  positions only) is applied by the pretraining loop, not the model — mirroring
  how the classifier leaves loss to the trainer.
"""

from __future__ import annotations

import torch
from torch import nn

from polaris.models.transformer_classifier import TransformerEncoderClassifier
from polaris.models.transformer_encoder import TransformerEncoder
from polaris.pretraining.masking import MaskedLMBatch

__all__ = ["MaskedLanguageModel"]


class MaskedLanguageModel(nn.Module):
    """A transformer trunk with a masked-language-model head.

    Parameters
    ----------
    vocab_size : int
        Number of tokens in the vocabulary (also the size of the output head).
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
        The padding id; its embedding row is fixed to zeros.
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
        )
        self.head = nn.Linear(self.encoder.embed_dim, vocab_size)

    def forward(self, batch: MaskedLMBatch) -> torch.Tensor:
        """Predict vocabulary logits for every position.

        Parameters
        ----------
        batch : MaskedLMBatch
            The masked input batch.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch_size, seq_len, vocab_size)``.
        """
        hidden = self.encoder(batch.input_ids, batch.attention_mask)  # (B, S, E)
        logits: torch.Tensor = self.head(hidden)  # (B, S, V)
        return logits

    def transfer_encoder_to(self, classifier: TransformerEncoderClassifier) -> None:
        """Copy this model's pretrained trunk into a classifier, in place.

        The classifier's own head is left untouched; only the shared ``encoder``
        weights are overwritten, so subsequent fine-tuning starts from the
        pretrained representation.

        Parameters
        ----------
        classifier : TransformerEncoderClassifier
            The classifier whose ``encoder`` receives the pretrained weights. Its
            trunk must have the same architecture as this model's.
        """
        classifier.encoder.load_state_dict(self.encoder.state_dict())
