# Phase 02 — Data Layer

**Status:** 🚧 In Progress

---

# Overview

Phase 02 marks the beginning of Polaris' first real NLP workflow.

Until Phase 01, the project focused on establishing a professional engineering
foundation:

- Repository architecture
- Documentation
- CI/CD
- Testing
- Core abstractions
- Registry system

Those components provide the infrastructure required to build Polaris, but they
do not yet solve an NLP problem.

Phase 02 begins the transition from framework infrastructure to actual machine
learning functionality.

Rather than continuing to build generic infrastructure, Polaris now follows an
evidence-driven development approach where abstractions are introduced only after
real use cases justify them.

---

# Vision

The objective of Phase 02 is to teach Polaris how to understand datasets.

Every NLP pipeline begins with data.

Before tokenization, training, or inference can exist, the framework must first
define how textual information is represented internally.

This phase establishes those foundational concepts.

---

# Guiding Philosophy

The Data Layer should remain completely independent from every future subsystem.

Specifically, it must NOT depend on:

- Tokenizers
- Models
- Training
- Evaluation
- Inference

Instead, every later subsystem will depend on the Data Layer.

Dependency direction:

Core
↓

Registry
↓

Data
↓

Tokenization
↓

Models
↓

Training
↓

Evaluation
↓

Inference

Maintaining this dependency direction is essential for long-term
maintainability.

---

# Design Goals

The Data Layer should be:

- Strongly typed
- Immutable where appropriate
- Framework-agnostic
- Model-agnostic
- Tokenizer-agnostic
- Easy to extend
- Easy to test
- Predictable
- Minimal

Every abstraction introduced during this phase must solve a real problem.

No speculative abstractions should be added.

---

# Responsibilities

The Data Layer is responsible for:

- Representing textual samples
- Representing labels
- Defining dataset interfaces
- Loading datasets
- Dataset iteration
- Train/Test split support
- Metadata representation
- Dataset validation

The Data Layer is NOT responsible for:

- Tokenization
- Vocabulary generation
- Numerical encoding
- Batching
- Neural networks
- Training loops
- Metrics
- Inference

---

# Deliverables

This phase will introduce:

## 1. Text Sample Representation

A strongly typed representation of a single NLP example.

Example:

- Raw text
- Label
- Optional metadata

This becomes the common language spoken throughout Polaris.

Every future tokenizer and model will consume these objects.

---

## 2. Dataset Protocol

A generic interface implemented by every dataset.

Examples include:

- IMDB
- AG News
- Yelp Reviews
- Custom user datasets

The protocol defines expected behaviour without imposing implementation details.

---

## 3. Built-in Dataset

The first built-in dataset supported by Polaris.

Current target:

**IMDB Movie Reviews**

Reason:

- Well-known benchmark
- Binary classification
- Easy to understand
- Suitable for demonstrating the first complete NLP pipeline

---

## 4. Dataset Utilities

Utilities required for working with datasets.

Examples may include:

- Dataset information
- Validation
- Statistics
- Train/Test access

Only utilities that are required by real implementations should be added.

---

## 5. Comprehensive Unit Tests

Every public API introduced during this phase must include unit tests.

Testing focuses on:

- Object creation
- Validation
- Iteration
- Edge cases
- Invalid inputs
- Dataset behaviour

The Data Layer should maintain high test coverage.

---

# Package Structure

At the end of Phase 02 the package is expected to resemble:

```text
polaris/

data/
│
├── __init__.py
├── base.py
├── sample.py
├── README.md
│
└── datasets/
    ├── __init__.py
    ├── imdb.py
    └── README.md
```

Additional files may be introduced only when justified.

---

# Development Order

To avoid premature abstraction, work will proceed in the following order.

## Step 1

Design the data representation.

Deliverable:

TextSample

---

## Step 2

Design dataset contracts.

Deliverable:

Dataset Protocol

---

## Step 3

Implement IMDB dataset.

Deliverable:

Dataset loading and access.

---

## Step 4

Write comprehensive tests.

---

## Step 5

Review the public API.

No additional abstractions should be introduced unless repeated patterns are
observed.

---

# Success Criteria

Phase 02 is considered complete when Polaris can:

- Represent text samples
- Load IMDB
- Iterate through dataset samples
- Validate datasets
- Expose a clean public API
- Pass all unit tests
- Pass Black
- Pass Ruff
- Pass MyPy
- Pass GitHub Actions

---

# Out of Scope

The following features intentionally do NOT belong in Phase 02.

- Tokenizers
- Vocabulary
- Encoders
- Neural Networks
- Transformers
- Attention
- Training
- Optimizers
- Metrics
- Inference
- Configuration
- CLI
- Plugin System

Those responsibilities belong to later phases.

---

# Lessons Applied

Phase 02 incorporates several architectural decisions made after reviewing the
overall Polaris roadmap.

Most importantly:

- Infrastructure should emerge from real requirements.
- Abstractions should be earned rather than predicted.
- Every new component must directly contribute toward a working NLP pipeline.

This philosophy helps prevent unnecessary complexity while improving long-term
maintainability.

---

# Completion Checklist

- [ ] Package structure created
- [ ] TextSample implemented
- [ ] Dataset Protocol implemented
- [ ] IMDB dataset implemented
- [ ] Unit tests written
- [ ] Documentation completed
- [ ] Black passes
- [ ] Ruff passes
- [ ] MyPy passes
- [ ] Pytest passes
- [ ] GitHub Actions pass
- [ ] Changes committed
- [ ] Changes pushed

---

# Next Phase

Upon completion of the Data Layer, Polaris will proceed to:

**Phase 03 — Tokenization**

The Tokenization Layer will transform raw textual samples into numerical
representations suitable for machine learning models while remaining independent
from any specific model architecture.

At that point, Polaris will officially begin processing natural language rather
than simply representing it.