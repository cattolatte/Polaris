# Collation Module

Turns variable-length tokenizer output into padded, model-ready tensor batches —
the bridge between the tokenization layer and the model layer.

## Public surface

- `Batch` (`batch.py`) — the boundary object between tokenization and models: three
  aligned PyTorch tensors (`input_ids`, `attention_mask`, `labels`). A frozen,
  slotted dataclass (compared by identity, since tensors have no value equality),
  with a `to(device)` helper. A model consumes a `Batch` and nothing more.
- `collate` (`collator.py`) — group encoded samples into a padded `Batch`: pad each
  sequence to the batch's longest (or to `max_length`) with the vocabulary's pad id,
  and build the matching attention mask.

## Example

```python
from polaris.collation import collate

batch = collate(
    [(tokenizer.encode("good film"), 1), (tokenizer.encode("bad"), 0)],
    pad_id=vocab.pad_id,
    max_length=256,
)
batch.input_ids.shape  # (2, seq_len)
```
