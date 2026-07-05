# Pretraining Module

Self-supervised **masked-language-model (MLM) pretraining** — the BERT recipe,
implemented from scratch. The model learns language from *unlabeled* text by
predicting hidden tokens, and that pretrained knowledge transfers into a
classifier for fine-tuning. Nothing is downloaded: "pretraining" here is a
training *method*, not a borrowed model, so Polaris stays fully from scratch.

This is the accuracy lever the v0.8-v0.10 benchmarks isolated: subword
tokenization and pretrained *word* embeddings both bounced off the ~86% ceiling on
IMDB, because the ceiling is the model class learning from labels alone. Contextual
pretraining changes what the model *is* before it ever sees a label.

## Public surface

- `mask_tokens` (`masking.py`) — apply BERT-style masking to a padded batch:
  ~15% of real (non-special) positions are supervised, and of those 80% become
  `<mask>`, 10% a random token, and 10% are left unchanged. Seedable via a
  `torch.Generator`.
- `MaskedLMBatch` (`masking.py`) — the value object masking produces: corrupted
  `input_ids`, the unchanged `attention_mask`, and per-token `labels` (the
  original id at supervised positions, `IGNORE_INDEX` elsewhere).
- `IGNORE_INDEX` (`masking.py`) — the target value cross-entropy ignores, so only
  masked positions contribute to the loss.
- `MaskedLanguageModel` (`model.py`) — the shared
  `TransformerEncoder` trunk plus a linear head projecting each hidden state to
  vocabulary logits. `transfer_encoder_to` copies the pretrained trunk into a
  `TransformerEncoderClassifier` (both embed the same trunk, so it is a plain
  `state_dict` copy).
- `pretrain` (`loop.py`) — the pretraining loop: dynamic per-epoch masking,
  cross-entropy over masked positions, the from-scratch warmup schedule, and a
  per-epoch history of loss and masked-token accuracy.

## The shape of a run

```python
from polaris.pretraining import MaskedLanguageModel, pretrain
from polaris.models import TransformerEncoderClassifier

# 1. Pretrain the trunk on unlabeled text (batches carry no real labels).
mlm = MaskedLanguageModel(vocab_size=V, embed_dim=128, num_heads=4, num_layers=2)
optimizer = torch.optim.Adam(mlm.parameters(), lr=5e-4)
pretrain(
    mlm, optimizer, unlabeled_batches,
    mask_id=vocab.mask_id, vocab_size=V,
    special_token_ids=(vocab.pad_id, vocab.unk_id, vocab.mask_id),
    epochs=3,
)

# 2. Transfer the pretrained trunk into a classifier and fine-tune as usual.
classifier = TransformerEncoderClassifier(vocab_size=V, num_classes=2,
                                          embed_dim=128, num_heads=4, num_layers=2)
mlm.transfer_encoder_to(classifier)
# ... fine-tune `classifier` with the standard Trainer on labeled data ...
```

See `examples/pretrain_finetune_imdb.py` for the full IMDB thread.

## Design notes

- **Two concrete loops, not a generalized `Trainer`.** Classification and MLM
  differ (per-row class vs. per-token targets, accuracy vs. masked-token accuracy);
  unifying them now would be speculative generality (ADR-0004).
- **Dynamic masking.** Masking is applied fresh each epoch from a seeded generator
  (the RoBERTa refinement over BERT's static masking) — better, and simpler to
  express than caching masked copies.
- **Deliberately omitted (until there is evidence to add them):** embedding/head
  weight tying, next-sentence prediction, and whole-word masking.
