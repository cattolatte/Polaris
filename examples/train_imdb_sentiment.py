"""Train a sentiment classifier on IMDB, end to end.

Loads IMDB, builds a vocabulary from the training split, tokenizes and collates
the reviews into batches, trains a model, and reports metrics on the test split.

Set ``MODEL`` below to choose the model — the same data, collation, training,
and evaluation code is reused either way (that reuse is the point):

- ``"transformer"`` (default): the from-scratch transformer encoder (v0.5).
- ``"pooling"``: the mean-pooled-embeddings baseline (v0.4).

The defaults are small so it runs quickly; increase them for a better score.

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
from polaris.embeddings import build_embedding_matrix, load_glove
from polaris.evaluation import evaluate, evaluate_model
from polaris.experiments import capture_environment, record_run
from polaris.models import MeanPoolingClassifier, TransformerEncoderClassifier
from polaris.tokenizers import (
    Tokenizer,
    WhitespaceTokenizer,
    build_vocabulary,
    train_bpe,
)
from polaris.training import Trainer, TrainingConfig
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

# Which model to train: "transformer" (from scratch, v0.5) or "pooling" (v0.4).
MODEL = "transformer"

# Tokenizer: "whitespace" (v0.3) or "bpe" (subword, v0.9).
# BPE splits rare/unseen words into subwords instead of <unk>. Its training is a
# readable reference (not optimized), so on full data it takes a few minutes;
# lower BPE_VOCAB_SIZE or raise MIN_FREQUENCY if it is too slow.
TOKENIZER = "whitespace"
BPE_VOCAB_SIZE = 10_000

EMBEDDING_DIM = 64  # mean-pooling baseline

# Transformer hyperparameters
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2
FF_DIM = 256
DROPOUT = 0.1

# Pretrained GloVe embeddings (v0.10): path to a GloVe .txt file, or None.
# Requires TOKENIZER = "whitespace" (GloVe is word-level). Download e.g.
# glove.6B.100d.txt and set GLOVE_DIM to match. For the transformer, NUM_HEADS
# must divide GLOVE_DIM (e.g. 100 / 4 = 25).
GLOVE_PATH: str | None = None
GLOVE_DIM = 100
FREEZE_EMBEDDINGS = False

EPOCHS = 10
# Transformers prefer a smaller learning rate than the pooling baseline.
LEARNING_RATE = 1e-3 if MODEL == "transformer" else 1e-2

# Training-engine settings (v0.6): validation split, warmup, early stopping.
VAL_RATIO = 0.1  # fraction of the training set held out for validation
WARMUP_RATIO = 0.1  # fraction of steps spent warming the learning rate up
EARLY_STOPPING_PATIENCE = 3  # stop after N epochs without validation improvement

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


def build_tokenizer(train_samples: Sequence[TextSample]) -> Tokenizer:
    """Build the tokenizer selected by ``TOKENIZER`` from the training corpus."""
    corpus = [sample.text.split() for sample in train_samples]
    if TOKENIZER == "bpe":
        return train_bpe(
            corpus,
            vocab_size=BPE_VOCAB_SIZE,
            unk_token=UNK_TOKEN,
            pad_token=PAD_TOKEN,
            min_frequency=MIN_FREQUENCY,
        )
    if TOKENIZER == "whitespace":
        vocabulary = build_vocabulary(
            corpus,
            unk_token=UNK_TOKEN,
            pad_token=PAD_TOKEN,
            min_frequency=MIN_FREQUENCY,
            max_size=MAX_VOCAB,
        )
        return WhitespaceTokenizer(vocabulary=vocabulary)
    msg = f"unknown TOKENIZER {TOKENIZER!r}; expected 'bpe' or 'whitespace'"
    raise ValueError(msg)


def make_batches(
    samples: Sequence[TextSample],
    tokenizer: Tokenizer,
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


def build_model(
    *, vocab_size: int, pad_id: int, pretrained: torch.Tensor | None = None
) -> torch.nn.Module:
    """Construct the model selected by ``MODEL`` (optionally GloVe-initialized)."""
    if MODEL == "transformer":
        return TransformerEncoderClassifier(
            vocab_size=vocab_size,
            num_classes=NUM_CLASSES,
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            ff_dim=FF_DIM,
            max_len=MAX_LENGTH,
            dropout=DROPOUT,
            pad_id=pad_id,
            pretrained_embeddings=pretrained,
            freeze_embeddings=FREEZE_EMBEDDINGS,
        )
    if MODEL == "pooling":
        return MeanPoolingClassifier(
            vocab_size=vocab_size,
            num_classes=NUM_CLASSES,
            embedding_dim=EMBEDDING_DIM,
            pad_id=pad_id,
            pretrained_embeddings=pretrained,
            freeze_embeddings=FREEZE_EMBEDDINGS,
        )
    msg = f"unknown MODEL {MODEL!r}; expected 'transformer' or 'pooling'"
    raise ValueError(msg)


def main() -> None:
    set_seed(SEED)

    print("Loading IMDB (this downloads the dataset on first run)...")
    train_samples = load_samples("train", TRAIN_SAMPLES)
    test_samples = load_samples("test", TEST_SAMPLES)
    print(f"  train samples: {len(train_samples)}, test samples: {len(test_samples)}")

    print(f"Building tokenizer ({TOKENIZER})...")
    tokenizer = build_tokenizer(train_samples)
    pad_id = tokenizer.vocabulary.pad_id
    assert pad_id is not None  # guaranteed: we passed pad_token
    print(f"  vocabulary size: {len(tokenizer.vocabulary)}")

    # Hold out part of the (already shuffled) training set for validation.
    split = int(len(train_samples) * (1 - VAL_RATIO))
    train_batches = make_batches(train_samples[:split], tokenizer, pad_id=pad_id)
    val_batches = make_batches(train_samples[split:], tokenizer, pad_id=pad_id)
    test_batches = make_batches(test_samples, tokenizer, pad_id=pad_id)

    device = resolve_device(DEVICE)
    print(f"Using device: {device}")

    pretrained = None
    if GLOVE_PATH is not None:
        if TOKENIZER != "whitespace":
            raise SystemExit("GLOVE_PATH requires TOKENIZER = 'whitespace'.")
        print("Loading GloVe embeddings...")
        vectors = load_glove(GLOVE_PATH)
        pretrained = build_embedding_matrix(
            tokenizer.vocabulary, vectors, embedding_dim=GLOVE_DIM, seed=SEED
        )
        print(f"  loaded {len(vectors)} vectors ({GLOVE_DIM}d)")

    model = build_model(
        vocab_size=len(tokenizer.vocabulary), pad_id=pad_id, pretrained=pretrained
    )
    model.to(device)
    print(f"Model: {MODEL}" + (" + GloVe" if GLOVE_PATH else ""))

    config = TrainingConfig(
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    print("Training (warmup scheduling, validation, early stopping)...")
    result = Trainer(model, optimizer, config).fit(train_batches, val_batches)
    print(f"Best validation accuracy: {result.best_val_accuracy:.4f}")

    # --- Report ---
    test_loss, _ = evaluate(model, test_batches)
    report = evaluate_model(
        model, test_batches, num_classes=NUM_CLASSES, class_names=CLASS_NAMES
    )

    print(f"\nTest loss: {test_loss:.4f}\n")
    print(report.to_text())

    # Record the run for reproducibility (config + metrics + report + environment).
    run_dir = record_run(
        f"runs/imdb_{MODEL}_{TOKENIZER}{'_glove' if GLOVE_PATH else ''}",
        config=config,
        history=result.history,
        report=report,
        environment=capture_environment(),
        seed=SEED,
    )
    print(f"\nRun recorded to: {run_dir}")


if __name__ == "__main__":
    main()
