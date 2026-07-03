# Phase 03 — Tokenization

**Status:** 🚧 In Progress

---

# Overview

Phase 3 introduces the first text processing subsystem in Polaris.

Until now, Polaris has been able to load datasets and represent text through
`TextSample`. The next step is transforming raw text into a machine-readable
representation.

Rather than depending immediately on Hugging Face or SentencePiece, Polaris
first defines its own tokenizer abstraction. Concrete wrappers around external
libraries will be added later.

This follows one of Polaris' core design principles:

> Own the interface, not the implementation.

---

# Goals

At the end of Phase 3 Polaris should be able to:

- tokenize raw text
- build a vocabulary
- convert tokens to integer IDs
- convert IDs back to tokens
- represent encoded text in a structured object
- support multiple tokenizer implementations through a common interface

---

# Non-Goals

Phase 3 intentionally does NOT include:

- Hugging Face tokenizers
- SentencePiece
- BPE
- WordPiece
- Unigram
- Byte-level tokenization
- Fast Rust tokenizers
- Training tokenizers
- Parallel tokenization
- GPU acceleration

Those belong in later phases.

---

# Directory Structure

```
polaris/
│
├── tokenizers/
│   ├── __init__.py
│   ├── tokenizer.py
│   ├── encoding.py
│   ├── vocabulary.py
│   └── whitespace.py
│
tests/
└── unit/
    └── tokenizers/
        ├── test_tokenizer.py
        ├── test_encoding.py
        ├── test_vocabulary.py
        └── test_whitespace.py
```

---

# Components

## 1. Tokenizer Protocol

File

```
polaris/tokenizers/tokenizer.py
```

Purpose

Defines the common interface implemented by every tokenizer.

Responsibilities

- tokenize text
- encode text
- decode ids
- expose vocabulary
- report vocabulary size

No implementation details belong here.

---

## 2. Vocabulary

File

```
polaris/tokenizers/vocabulary.py
```

Purpose

Represents the mapping between tokens and integer IDs.

Responsibilities

- token → id
- id → token
- unknown token handling
- special tokens
- vocabulary size

Vocabulary is independent from any tokenizer implementation.

---

## 3. Encoding

File

```
polaris/tokenizers/encoding.py
```

Purpose

Represents the output of a tokenizer.

Instead of returning a raw list of integers, Polaris returns a structured object.

Example

```python
Encoding(
    ids=[12, 54, 98],
    tokens=["hello", "world", "!"],
)
```

Future phases may extend this with

- attention mask
- offsets
- token type ids
- special token mask

without changing the tokenizer API.

---

## 4. Whitespace Tokenizer

File

```
polaris/tokenizers/whitespace.py
```

Purpose

Reference implementation of the tokenizer interface.

Behavior

Input

```
Hello world!
```

Output tokens

```
["Hello", "world!"]
```

This tokenizer simply splits on whitespace.

It exists to validate the tokenizer abstraction and provide a simple,
dependency-free implementation.

---

# Public API

```python
from polaris.tokenizers import Vocabulary, WhitespaceTokenizer

vocabulary = Vocabulary({"Hello": 0, "world!": 1})
tokenizer = WhitespaceTokenizer(vocabulary=vocabulary)

encoding = tokenizer.encode("Hello world!")

encoding.tokens
encoding.ids

text = tokenizer.decode(encoding.ids)
```

---

# Design Principles

## Simple

The tokenizer interface should be understandable in minutes.

---

## Backend Independent

Polaris should never expose Hugging Face tokenizer objects directly.

---

## Strong Typing

All public methods must have complete type annotations.

---

## Immutable Outputs

Encoding objects should be immutable.

---

## Composition

Vocabulary and Encoding are independent reusable components.

---

## Extensibility

Future tokenizer implementations should require no changes to user code.

---

# Testing Strategy

Every tokenizer implementation must satisfy the same behavioral contract.

Tests include

- construction
- tokenization
- encoding
- decoding
- unknown token handling
- vocabulary lookup
- round-trip encoding
- empty input
- whitespace handling

All unit tests must

- run offline
- avoid external dependencies
- pass Black
- pass Ruff
- pass strict MyPy
- pass Pytest

---

# Future Extensions

Later phases may introduce

- Hugging Face tokenizer adapter
- SentencePiece tokenizer
- WordPiece tokenizer
- Byte Pair Encoding
- Byte-level tokenization
- tokenizer training
- serialization
- tokenizer registry
- multilingual tokenizers

The public Tokenizer interface should remain stable.

---

# Deliverables

At the completion of Phase 3 the repository will contain

- Tokenizer protocol
- Vocabulary
- Encoding
- Whitespace tokenizer
- Complete unit tests
- Documentation
- 100% passing CI

Phase 3 intentionally prioritizes a clean abstraction over tokenizer
performance or feature completeness.