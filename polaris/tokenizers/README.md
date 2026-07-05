# Tokenizers Module

The tokenizers module is responsible for turning raw text into numerical
representations and back, while remaining independent from any specific model
architecture.

This package defines the core tokenization abstractions:

- `Tokenizer` (`tokenizer.py`) — the behavioral contract every tokenizer
  implements: `tokenize`, `encode`, `decode`, and access to its `vocabulary`.
- `Vocabulary` (`vocabulary.py`) — an immutable, validated bidirectional
  mapping between tokens and integer ids. Optionally designates an unknown
  token (`unk_token`), a padding token (`pad_token`), and a mask token
  (`mask_token`, for masked-language-model pretraining), exposing their ids as
  `unk_id` / `pad_id` / `mask_id`. `get_id` resolves unknown tokens to the unk id
  when one is configured; `lookup_id` stays strict.
- `Encoding` (`encoding.py`) — the immutable output of a tokenizer, pairing
  token ids with their string tokens.

Vocabularies are built from a corpus with a dedicated function (kept out of the
`Vocabulary` value object so it stays a pure mapping):

- `build_vocabulary` (`vocabulary_builder.py`) — counts token frequencies over
  already-tokenized text, reserves special tokens first, orders the rest by
  descending frequency (ties broken alphabetically for determinism), and
  supports `min_frequency` and `max_size`.

Concrete tokenizer implementations live alongside these abstractions:

- `WhitespaceTokenizer` (`whitespace.py`) — the first reference implementation,
  splitting text on whitespace via `str.split`.

Additional strategies (Character, BPE, WordPiece, SentencePiece, Unigram) will
be added incrementally, each satisfying the same `Tokenizer` contract.

## Example

```python
from polaris.tokenizers import build_vocabulary, WhitespaceTokenizer

# Build a vocabulary from a tokenized corpus, with special tokens.
corpus = [text.split() for text in ["good movie", "good film", "bad movie"]]
vocabulary = build_vocabulary(corpus, unk_token="<unk>", pad_token="<pad>")

tokenizer = WhitespaceTokenizer(vocabulary=vocabulary)

encoding = tokenizer.encode("good unseen movie")
encoding.ids     # e.g. (2, 1, 3) — "unseen" maps to the <unk> id (1)
encoding.tokens  # ('good', 'unseen', 'movie')

tokenizer.decode(encoding.ids)  # 'good <unk> movie'
```
