"""Run a trained model on raw text.

``Predictor`` is the inference counterpart of the training loop: it owns the whole
raw-text path — ``tokenize → encode → collate → forward → softmax → label`` — so a
caller who has never seen the training code can turn a string into a prediction.

Design Principles
-----------------
- Stateless per call and side-effect free: the model is held in ``eval`` mode and
  every prediction runs under ``torch.no_grad``.
- Returns a Polaris-native :class:`Prediction` value object, never raw tensors, at
  the module boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch import nn
from torch.nn import functional as F

from polaris.collation import collate
from polaris.errors import PolarisError
from polaris.tokenizers import Tokenizer

__all__ = ["Prediction", "Predictor"]


@dataclass(frozen=True, slots=True)
class Prediction:
    """The predicted class for one input, with per-class probabilities.

    Parameters
    ----------
    label : str
        The predicted class name.
    label_id : int
        The predicted class id (index into the model's outputs).
    probabilities : Mapping[str, float]
        Softmax probability for every class, keyed by class name (read-only).
    """

    label: str
    label_id: int
    probabilities: Mapping[str, float]

    def __post_init__(self) -> None:
        """Freeze the probabilities mapping against mutation."""
        object.__setattr__(
            self, "probabilities", MappingProxyType(dict(self.probabilities))
        )


class Predictor:
    """Predict class labels for raw text with a trained model and tokenizer.

    Parameters
    ----------
    model : nn.Module
        The trained model (consumes a :class:`~polaris.collation.batch.Batch`,
        returns logits). Placed in evaluation mode on construction.
    tokenizer : Tokenizer
        The tokenizer whose vocabulary the model was trained on.
    label_names : Sequence[str]
        Class names, indexed by class id. Its length must match the model's
        number of output classes.
    max_length : int, optional
        Truncation length applied during collation (should match training).

    Raises
    ------
    PolarisError
        If the tokenizer's vocabulary has no padding id (required for collation).
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Tokenizer,
        *,
        label_names: Sequence[str],
        max_length: int | None = None,
    ) -> None:
        pad_id = tokenizer.vocabulary.pad_id
        if pad_id is None:
            msg = "the tokenizer's vocabulary must define a padding token for inference"
            raise PolarisError(msg)
        self._model = model.eval()
        self._tokenizer = tokenizer
        self._label_names = tuple(label_names)
        self._pad_id = pad_id
        self._max_length = max_length

    def predict(self, text: str) -> Prediction:
        """Predict the class of a single text.

        Parameters
        ----------
        text : str
            The raw input text.

        Returns
        -------
        Prediction
            The predicted label and per-class probabilities.
        """
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: Sequence[str]) -> list[Prediction]:
        """Predict classes for a batch of texts.

        Parameters
        ----------
        texts : Sequence[str]
            The raw input texts.

        Returns
        -------
        list[Prediction]
            One prediction per input, in order.
        """
        if not texts:
            return []

        # A dummy label per row satisfies collation; the model ignores labels.
        samples = [(self._tokenizer.encode(text), 0) for text in texts]
        batch = collate(samples, pad_id=self._pad_id, max_length=self._max_length)
        device = next(self._model.parameters()).device
        batch = batch.to(device)

        with torch.no_grad():
            logits = self._model(batch)
            probabilities = F.softmax(logits, dim=-1)

        predictions: list[Prediction] = []
        for row in probabilities:
            label_id = int(torch.argmax(row))
            predictions.append(
                Prediction(
                    label=self._label_names[label_id],
                    label_id=label_id,
                    probabilities={
                        name: float(prob)
                        for name, prob in zip(self._label_names, row, strict=True)
                    },
                )
            )
        return predictions
