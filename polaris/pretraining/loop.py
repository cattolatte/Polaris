"""The masked-language-model pretraining loop.

Trains a :class:`~polaris.pretraining.model.MaskedLanguageModel` on unlabeled text
by predicting masked tokens. It reuses the from-scratch warmup schedule (v0.6) but
is deliberately a *separate* loop from
:class:`~polaris.training.trainer.Trainer`: the classifier trains on a per-row
class label with accuracy as the signal, whereas pretraining optimizes a
per-token objective over masked positions with masked-token accuracy as the
signal. Forcing both into one abstraction would be speculative generality
(ADR-0004); two small concrete loops read more clearly.

Masking is applied **dynamically** — freshly each epoch, from a seeded generator —
so the model sees different corruptions of the same text across epochs (the
RoBERTa refinement over BERT's static masking), which is both better and simpler
to express here.

Design Principles
-----------------
- Loss is cross-entropy over masked positions only (``IGNORE_INDEX`` elsewhere).
- Seeded and reproducible: the same ``seed`` reproduces the same masking and thus
  the same run.
- Returns a per-epoch history of loss and masked-token accuracy; it does not
  print, checkpoint, or early-stop (kept minimal until a real need appears).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from polaris.collation.batch import Batch
from polaris.pretraining.masking import IGNORE_INDEX, mask_tokens
from polaris.pretraining.model import MaskedLanguageModel
from polaris.training.scheduler import WarmupSchedule
from polaris.utils.device import module_device
from polaris.utils.logging import get_logger

__all__ = ["PretrainEpoch", "pretrain"]


@dataclass(frozen=True, slots=True)
class PretrainEpoch:
    """Metrics recorded for a single pretraining epoch.

    Parameters
    ----------
    epoch : int
        The 1-based epoch number.
    loss : float
        Mean masked-token cross-entropy over the epoch.
    masked_accuracy : float
        Fraction of masked positions whose original token was predicted correctly.
    """

    epoch: int
    loss: float
    masked_accuracy: float


def pretrain(
    model: MaskedLanguageModel,
    optimizer: torch.optim.Optimizer,
    batches: Sequence[Batch],
    *,
    mask_id: int,
    vocab_size: int,
    special_token_ids: Iterable[int] = (),
    epochs: int = 1,
    base_lr: float | None = None,
    warmup_ratio: float = 0.1,
    mask_probability: float = 0.15,
    seed: int = 0,
    logger: logging.Logger | None = None,
) -> tuple[PretrainEpoch, ...]:
    """Pretrain a masked-language model on unlabeled batches.

    Parameters
    ----------
    model : MaskedLanguageModel
        The model to pretrain.
    optimizer : torch.optim.Optimizer
        The optimizer stepping the model's parameters.
    batches : Sequence[Batch]
        Padded batches of unlabeled text (their classification ``labels`` are
        ignored; only ``input_ids`` and ``attention_mask`` are used).
    mask_id : int
        The vocabulary's ``<mask>`` id.
    vocab_size : int
        Vocabulary size (for random-token replacement and the loss).
    special_token_ids : Iterable[int], optional
        Ids never masked (e.g. padding, unknown, mask).
    epochs : int, default 1
        Number of passes over ``batches``.
    base_lr : float, optional
        Peak learning rate after warmup. Defaults to the optimizer's current rate.
    warmup_ratio : float, default 0.1
        Fraction of total steps spent warming the learning rate up.
    mask_probability : float, default 0.15
        Fraction of maskable positions supervised each step.
    seed : int, default 0
        Seed for the masking generator (and thus the whole run).
    logger : logging.Logger, optional
        Logger for per-epoch lines; defaults to the Polaris logger.

    Returns
    -------
    tuple[PretrainEpoch, ...]
        One record per epoch (loss and masked-token accuracy).

    Raises
    ------
    ValueError
        If ``batches`` is empty or ``epochs`` is less than ``1``.
    """
    if not batches:
        raise ValueError("batches must be non-empty")
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")

    log = logger if logger is not None else get_logger()
    device = module_device(model)
    special_token_ids = tuple(special_token_ids)

    if base_lr is None:
        base_lr = float(optimizer.param_groups[0]["lr"])
    total_steps = epochs * len(batches)
    scheduler = WarmupSchedule(
        optimizer,
        base_lr=base_lr,
        warmup_steps=int(total_steps * warmup_ratio),
        total_steps=total_steps,
    )

    # A CPU generator drives masking reproducibly, independent of the model device.
    generator = torch.Generator().manual_seed(seed)

    history: list[PretrainEpoch] = []
    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        stepped = 0
        correct = 0
        supervised = 0
        for batch in batches:
            masked = mask_tokens(
                batch.input_ids,
                batch.attention_mask,
                mask_id=mask_id,
                vocab_size=vocab_size,
                special_token_ids=special_token_ids,
                probability=mask_probability,
                generator=generator,
            ).to(device)

            is_masked = masked.labels != IGNORE_INDEX
            # If nothing was masked (possible on tiny batches), the loss would be
            # NaN — mean over zero supervised positions. Skip the step entirely.
            if not bool(is_masked.any()):
                continue

            logits = model(masked)  # (B, S, V)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                masked.labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )

            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            scheduler.step()
            optimizer.step()

            running_loss += float(loss.item())
            stepped += 1
            with torch.no_grad():
                predictions = logits.argmax(dim=-1)
                correct += int(((predictions == masked.labels) & is_masked).sum())
                supervised += int(is_masked.sum())

        mean_loss = running_loss / stepped if stepped else 0.0
        accuracy = correct / supervised if supervised else 0.0
        history.append(PretrainEpoch(epoch, mean_loss, accuracy))
        log.info(
            "pretrain epoch %d | loss %.4f | masked_acc %.4f",
            epoch,
            mean_loss,
            accuracy,
        )

    return tuple(history)
