"""Train a sentiment classifier on IMDB, end to end.

This is the runnable proof of Polaris' first end-to-end slice (v0.4). It loads
IMDB, builds a vocabulary from the training split, tokenizes and collates the
reviews into batches, trains a ``MeanPoolingClassifier``, and reports accuracy on
the test split.

The model is a deliberately simple baseline (mean-pooled embeddings + a linear
head), so it is fast on CPU and easy to read — not state of the art. The defaults
below are small on purpose; increase them for a better score.

Run it with the optional extras installed::

    uv run --extra datasets --extra torch python examples/train_imdb_sentiment.py
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import torch

from polaris.collation import Batch, collate
from polaris.data import TextSample
from polaris.data.datasets import IMDBDataset
from polaris.evaluation import (
    confusion_matrix,
    evaluate,
    precision_recall_f1,
    predict,
)
from polaris.models import MeanPoolingClassifier
from polaris.tokenizers import WhitespaceTokenizer, build_vocabulary
from polaris.training import train
from polaris.utils import resolve_device, set_seed

# --- Small, CPU-friendly defaults. Turn these up for a better model. ---
SEED = 0
# None => auto-select (Apple Silicon MPS, else CUDA, else CPU). Set to "cpu" to force.
DEVICE: str | None = None
TRAIN_SAMPLES = 5000
TEST_SAMPLES = 5000
MAX_VOCAB = 20_000
MAX_LENGTH = 256
BATCH_SIZE = 32
EMBEDDING_DIM = 64
EPOCHS = 5
LEARNING_RATE = 1e-2

# Regularization to curb overfitting (train loss collapses far below test loss).
MIN_FREQUENCY = 2  # drop tokens seen only once — they tend to be memorized
WEIGHT_DECAY = 1e-4  # L2 penalty via the optimizer

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
NUM_CLASSES = 2
CLASS_NAMES = ("neg", "pos")


def load_samples(split: str, limit: int) -> list[TextSample]:
    """Load a shuffled subset of an IMDB split.

    IMDB is stored sorted by label, so we shuffle before subsetting to get a mix
    of positive and negative reviews. The shuffle is seeded for reproducibility.
    """
    dataset = IMDBDataset(split=split)  # type: ignore[arg-type]
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    return [dataset[i] for i in indices[:limit]]


def make_batches(
    samples: Sequence[TextSample],
    tokenizer: WhitespaceTokenizer,
    *,
    pad_id: int,
) -> list[Batch]:
    """Encode samples and group them into padded batches."""
    encoded = [(tokenizer.encode(sample.text), int(sample.label)) for sample in samples]
    batches: list[Batch] = []
    for start in range(0, len(encoded), BATCH_SIZE):
        chunk = encoded[start : start + BATCH_SIZE]
        batches.append(collate(chunk, pad_id=pad_id, max_length=MAX_LENGTH))
    return batches


def main() -> None:
    set_seed(SEED)

    print("Loading IMDB (this downloads the dataset on first run)...")
    train_samples = load_samples("train", TRAIN_SAMPLES)
    test_samples = load_samples("test", TEST_SAMPLES)
    print(f"  train samples: {len(train_samples)}, test samples: {len(test_samples)}")

    print("Building vocabulary...")
    tokenized_corpus = [sample.text.split() for sample in train_samples]
    vocabulary = build_vocabulary(
        tokenized_corpus,
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN,
        min_frequency=MIN_FREQUENCY,
        max_size=MAX_VOCAB,
    )
    pad_id = vocabulary.pad_id
    assert pad_id is not None  # guaranteed: we passed pad_token
    print(f"  vocabulary size: {len(vocabulary)}")

    tokenizer = WhitespaceTokenizer(vocabulary=vocabulary)
    train_batches = make_batches(train_samples, tokenizer, pad_id=pad_id)
    test_batches = make_batches(test_samples, tokenizer, pad_id=pad_id)

    device = resolve_device(DEVICE)
    print(f"Using device: {device}")

    model = MeanPoolingClassifier(
        vocab_size=len(vocabulary),
        num_classes=NUM_CLASSES,
        embedding_dim=EMBEDDING_DIM,
        pad_id=pad_id,
    )
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    print("Training...")
    epoch_losses = train(model, train_batches, optimizer=optimizer, epochs=EPOCHS)
    for epoch, loss in enumerate(epoch_losses, start=1):
        print(f"  epoch {epoch}: train loss {loss:.4f}")

    # --- Report ---
    test_loss, test_accuracy = evaluate(model, test_batches)
    logits, labels = predict(model, test_batches)
    precisions, recalls, f1s = precision_recall_f1(
        logits, labels, num_classes=NUM_CLASSES
    )
    matrix = confusion_matrix(logits, labels, num_classes=NUM_CLASSES)

    print(f"\nTest loss:     {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    print("\nPer-class metrics:")
    print(f"  {'class':<5} {'precision':>9} {'recall':>7} {'f1':>6}")
    for index, name in enumerate(CLASS_NAMES):
        print(
            f"  {name:<5} {precisions[index]:>9.4f} "
            f"{recalls[index]:>7.4f} {f1s[index]:>6.4f}"
        )

    print("\nConfusion matrix (rows = true, cols = predicted):")
    print(f"  {'':<5} {'pred neg':>9} {'pred pos':>9}")
    for index, name in enumerate(CLASS_NAMES):
        row = matrix[index].tolist()
        print(f"  {name:<5} {row[0]:>9} {row[1]:>9}")


if __name__ == "__main__":
    main()
