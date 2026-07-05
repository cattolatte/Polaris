<div align="center">

# Polaris

### A Production-Inspired NLP Engineering Platform

*Build. Train. Evaluate. Deploy.*

[![Status](https://img.shields.io/badge/status-under%20development-orange)](https://github.com/cattolatte/Polaris)
[![Version](https://img.shields.io/badge/version-v0.9.0-blue)](CHANGELOG.md)
[![Python](https://img.shields.io/badge/python-3.12+-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

**Polaris is currently under active development and is not yet ready for production use.**

This repository represents the beginning of a long-term effort to build a production-inspired, **educational** engineering platform for modern Natural Language Processing (NLP) systems, emphasizing clean architecture, reproducibility, modularity, and software engineering best practices.

</div>

---

## Vision

Modern NLP projects often focus on isolated parts of the lifecycle, such as model training or inference. This fragmentation leads to integration challenges and technical debt.

Polaris aims to provide a unified engineering platform that covers the **complete NLP lifecycle**—from raw datasets and tokenization to model development, experimentation, evaluation, optimization, and deployment.

Polaris prioritizes engineering transparency, allowing every stage of the NLP lifecycle to be understood, extended, benchmarked, and reproduced.

---

## Why Polaris?

Existing libraries solve specific problems. Polaris aims to connect the complete NLP engineering lifecycle into a single cohesive platform while exposing every component for learning, experimentation, and extension.

Rather than hiding complexity, Polaris embraces engineering transparency so developers can understand how modern NLP systems are designed, trained, evaluated, and deployed.

Polaris is a **reference implementation first**. Rather than wrapping existing frameworks, the NLP stack—tokenizers, models, and training loops—is built from scratch on PyTorch tensors, so every component can be read and understood end to end. The goal is not to compete with libraries like Hugging Face on features, but to be the clearest, best-engineered from-scratch NLP system you can learn from. See [ADR-0001](docs/adr/0001-project-identity.md).

---

## Project Goals

Polaris is being designed around the following principles:

- **Production-Grade Engineering**: Build robust, maintainable, and scalable systems.
- **Modular & Extensible**: Easily add new models, datasets, and components.
- **Reproducible Workflows**: Ensure that experiments and results can be consistently reproduced.
- **Configuration-Driven**: Define and manage complex workflows through simple configuration files.
- **Clean Architecture**: Adhere to SOLID principles and clear separation of concerns.
- **Strong Testing & Documentation**: Maintain high standards for code quality and usability.
- **Educational**: Provide clear, well-documented implementations of modern NLP concepts.
- **End-to-End**: Cover the entire NLP engineering workflow within a single, cohesive platform.

---

## Core Modules

The platform is organized into dedicated modules with clearly defined responsibilities. Each module is developed independently while maintaining clear architectural boundaries.

| Module | Description | Status |
| :--- | :--- | :--- |
| `core/` | Provides the foundational building blocks, interfaces, and architectural patterns for the entire platform. | ✅ Implemented |
| `registry/` | A mechanism for registering and accessing components by name. Implemented and tested, but currently **dormant** — no consumers yet (see [ADR-0005](docs/adr/0005-registry-dormant.md)). | 🧊 Dormant |
| `data/` | Handles data loading and management, exposing datasets through a Polaris-native interface. | ✅ Implemented |
| `tokenizers/` | A laboratory for building and using various tokenization strategies. | ✅ Implemented |
| `collation/` | Turns tokenizer output into padded, model-ready tensor batches. | ✅ Implemented |
| `models/` | From-scratch implementations of NLP architectures on PyTorch primitives. | ✅ Implemented |
| `training/` | The training engine — training loops now; checkpointing and schedulers later. | ✅ Implemented |
| `experiments/` | Manages experiment tracking, logging, and versioning of results. | 🏗️ Planned |
| `evaluation/` | Tools and metrics for model evaluation. | ✅ Implemented |
| `inference/` | Runtime for running predictions with trained models. | 🏗️ Planned |
| `deployment/` | Contains utilities for packaging, optimizing, and deploying models for inference. | 🏗️ Planned |
| `visualization/` | Tools for visualizing data, model architectures, and experiment results. | 🏗️ Planned |
| `plugins/` | An extensible plugin system for integrating third-party tools and custom functionality. | 🏗️ Planned |
| `utils/` | Common utilities (e.g. reproducible seeding) used across the framework. | ✅ Implemented |

---

## Roadmap

Polaris is being built incrementally, with a focus on quality and architectural integrity at each step. For a detailed development plan and upcoming milestones, please see the official project roadmap.

➡️ **[View the Project Roadmap](ROADMAP.md)** ⬅️

---

## Benchmarks

Polaris trains end to end on IMDB sentiment classification. Every model *and*
tokenizer below is implemented **from scratch** and trained through the same
pipeline (data → tokenization → collation → model → training engine →
evaluation). Runs are recorded and reproducible.

| Model | Whitespace (20k) | BPE (10k) | GloVe (100d) |
| :--- | :---: | :---: | :---: |
| Mean-pooling baseline | 0.856 | 0.839 | **0.857** |
| Transformer encoder (from scratch) | 0.855 | 0.838 | 0.849 |

<sub>Test accuracy on IMDB · 25,000 train / 25,000 test · seed 0 · Apple Silicon
(MPS). Reproduce with `examples/train_imdb_sentiment.py` (`TRAIN_SAMPLES =
TEST_SAMPLES = 25000`, plus the `MODEL` / `TOKENIZER` / `GLOVE_PATH` switches).</sub>

**What this shows** — three honest findings, each more interesting than a single
number. We pulled the three cheap levers, and **every one bounces off ~86%**:

1. **The transformer does not beat the mean-pooling baseline** (they tie at
   ~85.5%). A small from-scratch transformer has no edge over a bag-of-embeddings
   on a task whose signal is a handful of strong words — and it overfits more, at
   ~14× the cost.
2. **Subword tokenization (BPE) slightly *hurts* here.** IMDB sentiment lives in
   common whole words ("great", "terrible"); BPE splits them into subwords,
   diluting the signal and lengthening sequences (so more of each review is lost
   to truncation). BPE pays off when the problem is out-of-vocabulary or
   morphology — not this one.
3. **Pretrained *word* embeddings (GloVe) do not move it either** (+0.001 for the
   pooling model; the transformer overfits *harder* and slips to 0.849).
   Pretrained word vectors help most when labeled data is **scarce** — with
   25,000 labeled reviews the model learns good embeddings from scratch anyway,
   so GloVe's head start is redundant.

### Self-supervised pretraining (the fourth lever)

The lever that *actually* transformed NLP is **self-supervised pretraining**:
learn language from unlabeled text with a masked-language-model objective, then
fine-tune. Polaris implements this from scratch (v0.11) — pretrain our own
transformer trunk on the 50,000 **unlabeled** IMDB reviews, then transfer it into
the classifier. A controlled ablation (identical vocabulary and 4-layer
architecture; the only difference is the pretraining):

| 4-layer transformer | Test accuracy | Best val | Epoch-1 val |
| :--- | :---: | :---: | :---: |
| Random-init trunk (no pretraining) | 0.852 | 0.851 | 0.736 |
| **MLM-pretrained trunk** | **0.853** | **0.864** | **0.810** |

Pretraining **works, in exactly the expected direction** — a large head start
(epoch-1 validation 0.810 vs 0.736), faster convergence, and a higher best
validation — but it converges to the **same ~86% test ceiling**. Two reasons, and
they are the whole lesson:

- **Labels aren't scarce.** 25,000 labeled reviews already suffice for the model
  to learn good features directly, so a warm start reaches the same destination,
  just faster.
- **The pretraining corpus is small and in-domain.** Real BERT pretrains on
  *billions* of words of diverse text; we pretrained on ~11M words of IMDB (the
  same domain as the task), which injects little knowledge the labels can't teach.
  Masked accuracy plateaued at ~0.24.

So **four from-scratch levers — a transformer, subwords, pretrained word vectors,
and self-supervised pretraining — all land at ~85–86%.** That is the honest
result: the ceiling is the **task and the data/compute regime**, not any single
component. Breaking it would take large-external-corpus pretraining at a scale
beyond a laptop. Measuring and *explaining* all of this — with reproducible,
controlled experiments — is the entire point of building the stack from scratch.

> **Full metrics** — per-class precision/recall/F1 and confusion matrices for all
> eight runs, plus the exact setup and reproduction commands, are in
> [`BENCHMARKS.md`](BENCHMARKS.md).

---

## Using a trained model

A trained model is saved as a self-describing **bundle** (weights + architecture +
vocabulary + labels) and reloaded with no training code — from the shell, in
Python, or over HTTP.

**Predict from the command line:**

```bash
polaris predict "a wonderful, moving film" --model model.pt --probs
# pos
#   neg: 0.0180
#   pos: 0.9820
```

**Serve it over HTTP** (requires the `serving` extra — `uv sync --extra serving`):

```bash
polaris serve --model model.pt --port 8000
curl -s localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"text": "a wonderful, moving film"}'
# {"label":"pos","label_id":1,"probabilities":{"neg":0.02,"pos":0.98}}
```

**Or in a container:**

```bash
docker build -t polaris-serve .
docker run --rm -p 8000:8000 -v "$PWD/runs:/models:ro" \
  -e POLARIS_MODEL=/models/imdb_pretrained_transformer/model.pt polaris-serve
```

**Or in Python:**

```python
from polaris.inference import load_bundle

predictor = load_bundle("model.pt")
predictor.predict("a wonderful, moving film")
# Prediction(label="pos", label_id=1, probabilities={"neg": 0.02, "pos": 0.98})
```

Running `examples/pretrain_finetune_imdb.py` writes a ready-to-serve bundle to
`runs/`.

---

## Engineering Philosophy

Polaris is built with long-term maintainability as the primary objective.

The project follows modern software engineering practices including:

- **Clean Architecture**: Enforcing separation of concerns and dependency rules.
- **SOLID Principles**: Creating understandable, flexible, and maintainable designs.
- **Vertical Slices**: Every release leaves the system runnable, not scaffolding for a future layer.
- **Evidence-Driven Abstraction**: Interfaces are extracted from working code, never designed in advance.
- **Configuration-Driven Design**: Enabling flexible and reproducible experiments.
- **Modular Components**: Promoting reusability and independent development.
- **Type Safety**: Using modern Python type hints to improve code quality.
- **Comprehensive Testing**: Ensuring reliability through a robust test suite.
- **Continuous Documentation**: Keeping documentation in sync with the codebase.

Every major design decision is intended to prioritize readability, extensibility, and reproducibility.

---

## Project Documentation

Comprehensive project documentation is a core goal and is available throughout the repository.

- **Architecture Docs**: `docs/architecture/`
- **Decision Records (ADRs)**: `docs/adr/`
- **Design Docs**: `docs/design/`
- **API Reference**: `docs/api/`
- **Tutorials**: `docs/tutorials/`

---

## Repository Structure

The repository is organized to separate concerns, making it easy to navigate and contribute.

```
Polaris/
│
├── polaris/
├── docs/
├── configs/
├── tests/
├── examples/
├── benchmarks/
├── docker/
├── notebooks/
└── scripts/
```

---

## Contributing

Community contributions are welcome once the project reaches its first public development milestone. As the project matures, contribution guidelines, issue templates, and development documentation will be expanded to support community contributions.

---

## License

This project is licensed under the MIT License.

See the `LICENSE` file for details.

---

<div align="center">

**Polaris is being built one engineering milestone at a time.**

Every phase prioritizes maintainability, reproducibility, and clean software architecture over rapid feature development.

</div>
