"""Pretrain a transformer on unlabeled IMDB, then fine-tune it for sentiment.

The v0.11 self-supervised slice, end to end and entirely from scratch:

1. **Pretrain** the shared :class:`TransformerEncoder` trunk on the 50,000
   *unlabeled* IMDB reviews with a masked-language-model objective — the model
   learns language structure by predicting hidden words, no labels involved.
2. **Transfer** the pretrained trunk into a classifier and **fine-tune** it on
   the 25,000 labeled reviews.

Nothing is downloaded: our own model, our own objective, our own weights. This is
the recipe (BERT) that actually breaks the ~86% ceiling the v0.8-v0.10 benchmarks
kept hitting — because the model arrives at fine-tuning already knowing the
language, instead of learning it from 25k labels alone.

The defaults are the full-scale run. Turn them down (e.g. ``PRETRAIN_SAMPLES`` and
the epochs) for a quick smoke test. Run it with the optional extras installed::

    uv run --extra datasets --extra torch python examples/pretrain_finetune_imdb.py
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import torch

from polaris.collation import Batch, collate
from polaris.data import TextSample
from polaris.data.datasets import IMDBDataset
from polaris.evaluation import evaluate, evaluate_model
from polaris.experiments import capture_environment, record_run
from polaris.inference import save_bundle
from polaris.models import TransformerEncoderClassifier
from polaris.pretraining import MaskedLanguageModel, pretrain
from polaris.tokenizers import WhitespaceTokenizer, build_vocabulary
from polaris.training import Trainer, TrainingConfig
from polaris.utils import resolve_device, set_seed

# --- Data / vocabulary ---
SEED = 0
# None => auto-select (Apple Silicon MPS, else CUDA, else CPU). Set to "cpu" to force.
DEVICE: str | None = None
PRETRAIN_SAMPLES = 50_000  # the unlabeled split (max 50,000)
TRAIN_SAMPLES = 25_000
TEST_SAMPLES = 25_000
MAX_VOCAB = 20_000
MAX_LENGTH = 256
BATCH_SIZE = 32
MIN_FREQUENCY = 2

# --- Transformer trunk (shared by the MLM and the classifier) ---
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4
FF_DIM = 256
DROPOUT = 0.1

# --- Pretraining (masked language modeling) ---
# Set to False for the controlled ablation: same vocab + architecture, fine-tuned
# from a random-init trunk. The accuracy gap vs True is what pretraining buys.
PRETRAIN = False
PRETRAIN_EPOCHS = 30
PRETRAIN_LR = 5e-4
MASK_PROBABILITY = 0.15

# --- Fine-tuning (sentiment classification) ---
FINETUNE_EPOCHS = 15
LEARNING_RATE = 1e-3
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 1e-4
VAL_RATIO = 0.1
EARLY_STOPPING_PATIENCE = 3

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
MASK_TOKEN = "<mask>"
NUM_CLASSES = 2
CLASS_NAMES = ("neg", "pos")


def load_samples(split: str, limit: int) -> list[TextSample]:
    """Load a shuffled subset of an IMDB split (seeded for reproducibility)."""
    dataset = IMDBDataset(split=split)  # type: ignore[arg-type]
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    return [dataset[i] for i in indices[:limit]]


def make_batches(
    samples: Sequence[TextSample], tokenizer: WhitespaceTokenizer, *, pad_id: int
) -> list[Batch]:
    """Encode samples and group them into padded batches."""
    encoded = [(tokenizer.encode(s.text), int(s.label)) for s in samples]
    batches: list[Batch] = []
    for start in range(0, len(encoded), BATCH_SIZE):
        chunk = encoded[start : start + BATCH_SIZE]
        batches.append(collate(chunk, pad_id=pad_id, max_length=MAX_LENGTH))
    return batches


def build_encoder_kwargs(*, vocab_size: int, pad_id: int) -> dict[str, object]:
    """The trunk architecture, shared identically by the MLM and the classifier."""
    return {
        "vocab_size": vocab_size,
        "embed_dim": EMBED_DIM,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "ff_dim": FF_DIM,
        "max_len": MAX_LENGTH,
        "dropout": DROPOUT,
        "pad_id": pad_id,
    }


def main() -> None:
    set_seed(SEED)
    device = resolve_device(DEVICE)
    print(f"Using device: {device}")

    print("Loading IMDB (downloads on first run: train, test, unsupervised)...")
    unlabeled = load_samples("unsupervised", PRETRAIN_SAMPLES)
    train_samples = load_samples("train", TRAIN_SAMPLES)
    test_samples = load_samples("test", TEST_SAMPLES)
    print(
        f"  unlabeled: {len(unlabeled)}, "
        f"train: {len(train_samples)}, test: {len(test_samples)}"
    )

    # Build the vocabulary from all available text (labeled + unlabeled), reserving
    # the <mask> token the pretraining objective needs.
    print("Building vocabulary (with <mask>)...")
    corpus = [s.text.split() for s in (*unlabeled, *train_samples)]
    vocabulary = build_vocabulary(
        corpus,
        pad_token=PAD_TOKEN,
        unk_token=UNK_TOKEN,
        mask_token=MASK_TOKEN,
        min_frequency=MIN_FREQUENCY,
        max_size=MAX_VOCAB,
    )
    tokenizer = WhitespaceTokenizer(vocabulary=vocabulary)
    pad_id = vocabulary.pad_id
    mask_id = vocabulary.mask_id
    assert pad_id is not None and mask_id is not None
    vocab_size = len(vocabulary)
    special_ids = tuple(
        i
        for i in (vocabulary.pad_id, vocabulary.unk_id, vocabulary.mask_id)
        if i is not None
    )
    print(f"  vocabulary size: {vocab_size}")

    encoder_kwargs = build_encoder_kwargs(vocab_size=vocab_size, pad_id=pad_id)
    classifier = TransformerEncoderClassifier(
        num_classes=NUM_CLASSES, **encoder_kwargs  # type: ignore[arg-type]
    )

    # --- Phase 1: self-supervised pretraining on unlabeled text ---
    # Set PRETRAIN = False to run the controlled ablation: the *same* vocabulary
    # and architecture, fine-tuned from a random-init trunk (no pretraining). The
    # gap between the two runs is exactly what pretraining buys.
    if PRETRAIN:
        print(f"\nPretraining (MLM, {PRETRAIN_EPOCHS} epochs on unlabeled reviews)...")
        pretrain_batches = make_batches(unlabeled, tokenizer, pad_id=pad_id)
        mlm = MaskedLanguageModel(**encoder_kwargs)  # type: ignore[arg-type]
        mlm.to(device)
        mlm_optimizer = torch.optim.Adam(mlm.parameters(), lr=PRETRAIN_LR)
        pretrain(
            mlm,
            mlm_optimizer,
            pretrain_batches,
            mask_id=mask_id,
            vocab_size=vocab_size,
            special_token_ids=special_ids,
            epochs=PRETRAIN_EPOCHS,
            warmup_ratio=WARMUP_RATIO,
            mask_probability=MASK_PROBABILITY,
            seed=SEED,
        )
        print("\nTransferring pretrained trunk into a classifier and fine-tuning...")
        mlm.transfer_encoder_to(classifier)
    else:
        print("\nSkipping pretraining (control: random-init trunk). Fine-tuning...")

    # --- Phase 2: fine-tune the classifier on the labeled task ---
    classifier.to(device)

    split = int(len(train_samples) * (1 - VAL_RATIO))
    train_batches = make_batches(train_samples[:split], tokenizer, pad_id=pad_id)
    val_batches = make_batches(train_samples[split:], tokenizer, pad_id=pad_id)
    test_batches = make_batches(test_samples, tokenizer, pad_id=pad_id)

    config = TrainingConfig(
        epochs=FINETUNE_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
    )
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    result = Trainer(classifier, optimizer, config).fit(train_batches, val_batches)
    print(f"Best validation accuracy: {result.best_val_accuracy:.4f}")

    # --- Report ---
    test_loss, _ = evaluate(classifier, test_batches)
    report = evaluate_model(
        classifier, test_batches, num_classes=NUM_CLASSES, class_names=CLASS_NAMES
    )
    print(f"\nTest loss: {test_loss:.4f}\n")
    print(report.to_text())

    run_name = "pretrained" if PRETRAIN else "scratch"
    run_dir = record_run(
        f"runs/imdb_{run_name}_transformer",
        config=config,
        history=result.history,
        report=report,
        environment=capture_environment(),
        seed=SEED,
    )
    print(f"\nRun recorded to: {run_dir}")

    # Save a self-describing bundle so the model can be served / predicted with:
    #   polaris predict "a wonderful film" --model <bundle> --probs
    bundle_path = f"{run_dir}/model.pt"
    save_bundle(
        bundle_path,
        model=classifier,
        tokenizer=tokenizer,
        model_type="transformer",
        model_config={"num_classes": NUM_CLASSES, **encoder_kwargs},
        label_names=CLASS_NAMES,
        max_length=MAX_LENGTH,
    )
    print(f"Model bundle saved to: {bundle_path}")


if __name__ == "__main__":
    main()
