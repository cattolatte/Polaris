# ADR 0003 — PyTorch as the tensor framework; model internals from scratch

**Status:** Accepted (2026-07-03)

## Context

Polaris needs a numerical/tensor/autograd substrate before it can build models
(roadmap v0.4+). This is the single most consequential technical decision in the
project: it shapes collation, models, training, checkpointing, and deployment, and it
is expensive to reverse once model code exists. The decision was previously absent
from the roadmap, which was a latent risk.

Two sub-questions:
1. Which framework — PyTorch, JAX, or a from-scratch NumPy tensor/autograd?
2. How much of the model do we implement ourselves vs. use the framework's high-level
   modules?

The project identity (ADR-0001) is educational reference implementation.

## Decision

1. **Use PyTorch** as the tensor + autograd + optimizer substrate. Introduce it as an
   optional/required dependency in the phase that first needs it (the first model
   slice), not before.
2. **Implement the NLP-defining internals from scratch** on tensor primitives:
   attention, multi-head attention, positional encoding, transformer blocks, layer
   normalization, embeddings, loss where instructive. Use PyTorch only for the
   *undifferentiated* machinery — autograd, `optim`, tensor ops, and data plumbing.
   **Avoid** high-level black boxes such as `nn.Transformer` /
   `nn.MultiheadAttention` / `nn.TransformerEncoder` for the parts we are teaching.

## Consequences

**Good**

- PyTorch is the most readable framework for teaching, the most widely known, and the
  highest-value skill to demonstrate.
- Implementing internals from scratch is where nearly all the educational value lives:
  a reader learns how attention actually works, not how to wire `nn.Transformer`.
- We still get autograd and optimizers for free, so scope stays realistic — we are not
  reimplementing calculus.

**Tradeoffs / costs**

- From-scratch layers are more code to write, test, and maintain than calling
  `nn.*`, and are easier to get subtly wrong (must be covered by shape/behavior
  tests).
- A hard dependency on PyTorch (large wheel) enters the project; it must be an extra
  and imported lazily so the data/tokenizer layers stay lightweight and offline.
- Ties Polaris to PyTorch's programming model; a future JAX port would be a rewrite.

**Guardrails**

- Keep from-scratch implementations to what is *instructive*. Do not reimplement
  autograd, optimizers, or CUDA kernels — that is scope with little teaching payoff
  (and is why "pure from-scratch incl. autograd" was rejected).
- Every from-scratch layer needs tests asserting output shapes and known invariants
  (e.g. attention rows sum to 1, causal mask zeroes the upper triangle).

## Alternatives considered

- **JAX:** elegant and great for pure-function training, but a smaller audience,
  steeper learning curve, and more friction as a teaching read. Rejected for identity
  fit.
- **NumPy with hand-written autograd:** maximal educational depth but enormous scope;
  realistically cannot reach working transformers before v1.0. Rejected as
  impractical.
- **PyTorch with high-level `nn` modules:** fast and low-maintenance, but reads as
  "wiring," not "understanding" — directly at odds with ADR-0001. Rejected for the
  parts we teach (still fine for undifferentiated plumbing).
