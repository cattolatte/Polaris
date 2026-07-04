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
from polaris.evaluation import evaluate
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

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
NUM_CLASSES = 2


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
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Training...")
    epoch_losses = train(model, train_batches, optimizer=optimizer, epochs=EPOCHS)
    for epoch, loss in enumerate(epoch_losses, start=1):
        print(f"  epoch {epoch}: train loss {loss:.4f}")

    test_loss, test_accuracy = evaluate(model, test_batches)
    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
