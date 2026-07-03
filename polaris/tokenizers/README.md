# Tokenizers Module

The tokenizers module is responsible for turning raw text into numerical
representations and back, while remaining independent from any specific model
architecture.

This package defines the core tokenization abstractions:

- `Tokenizer` (`tokenizer.py`) — the behavioral contract every tokenizer
  implements: `tokenize`, `encode`, `decode`, and access to its `vocabulary`.
- `Vocabulary` (`vocabulary.py`) — an immutable, validated bidirectional
  mapping between tokens and integer ids.
- `Encoding` (`encoding.py`) — the immutable output of a tokenizer, pairing
  token ids with their string tokens.

Concrete implementations live alongside these abstractions:

- `WhitespaceTokenizer` (`whitespace.py`) — the first reference implementation,
  splitting text on whitespace via `str.split`.

Additional strategies (Character, BPE, WordPiece, SentencePiece, Unigram) will
be added incrementally, each satisfying the same `Tokenizer` contract.

## Example

```python
from polaris.tokenizers import Vocabulary, WhitespaceTokenizer

vocabulary = Vocabulary({"hello": 0, "world": 1})
tokenizer = WhitespaceTokenizer(vocabulary=vocabulary)

encoding = tokenizer.encode("hello world")
encoding.ids     # (0, 1)
encoding.tokens  # ('hello', 'world')

tokenizer.decode(encoding.ids)  # 'hello world'
```
