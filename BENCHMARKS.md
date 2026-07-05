# Polaris Benchmarks

Full, reproducible evaluation of every model Polaris implements, on **IMDB movie
review sentiment** (binary classification). Every model, tokenizer, and metric
below is built **from scratch** (see [ADR-0003](docs/adr/0003-tensor-framework.md))
and trained through the same pipeline: data → tokenization → collation → model →
training engine → evaluation. Each run is recorded (config + metrics + environment
+ report) for reproducibility.

The headline is not a single number — it is a **controlled, four-lever
investigation** into what does and does not move accuracy on this task, and *why*.

## Setup

| | |
| :--- | :--- |
| Task | IMDB sentiment (binary: `neg` / `pos`) |
| Data | 25,000 train / 25,000 test (balanced 12.5k / 12.5k); 50,000 unlabeled for pretraining |
| Seed | 0 (deterministic) |
| Device | Apple Silicon (MPS) |
| Environment | Python 3.12, PyTorch 2.12, `polaris` 0.11 |
| Max sequence length | 256 · Batch size 32 |

Reproduce: `uv run --extra datasets --extra torch python examples/train_imdb_sentiment.py`
(baselines, via `MODEL` / `TOKENIZER` / `GLOVE_PATH` switches) and
`examples/pretrain_finetune_imdb.py` (pretraining, via the `PRETRAIN` switch).

## Summary — test accuracy

| Model | Whitespace | BPE (subword) | GloVe (100d) |
| :--- | :---: | :---: | :---: |
| Mean-pooling baseline | 0.8564 | 0.8394 | **0.8566** |
| Transformer encoder (from scratch) | 0.8550 | 0.8378 | 0.8494 |

Self-supervised pretraining (4-layer transformer, controlled ablation):

| 4-layer transformer | Test acc | Macro-F1 | Best val | Epoch-1 val |
| :--- | :---: | :---: | :---: | :---: |
| Random-init trunk (no pretraining) | 0.8518 | 0.8516 | 0.8508 | 0.7356 |
| **MLM-pretrained trunk** | **0.8534** | **0.8534** | **0.8636** | **0.8104** |

**Every lever lands at ~85–86%.** Because IMDB classes are balanced, macro-F1
tracks accuracy closely throughout — the per-class detail below is included for
completeness and to exercise the evaluation framework, not because it reveals class
imbalance.

## The four findings

1. **A from-scratch transformer does not beat a mean-pooling baseline** (0.8550 vs
   0.8564). On a task whose signal is a handful of strong words, attention adds no
   edge — and overfits more, at ~14× the compute.
2. **Subword tokenization (BPE) slightly hurts** (0.838 vs 0.855). IMDB sentiment
   lives in whole words ("great", "terrible"); BPE splits them, diluting signal and
   lengthening sequences (more of each review lost to truncation).
3. **Pretrained word embeddings (GloVe) do not move it** (+0.0002 pooling; the
   transformer overfits harder to 0.849). Word vectors help when labels are scarce;
   with 25k labeled reviews the model learns good embeddings itself.
4. **Self-supervised (MLM) pretraining works in the expected direction but hits the
   same ceiling.** It buys a large warm start (epoch-1 val 0.810 vs 0.736), faster
   convergence, and a higher best validation (0.864 vs 0.851) — yet the same ~0.853
   test accuracy. 25k labels already suffice, and the small in-domain pretraining
   corpus (~11M words, masked-accuracy plateaued ~0.24) injects little new
   knowledge. Real BERT pretrains on *billions* of words of diverse text.

**Conclusion.** The ~86% ceiling is a property of the **task and the data/compute
regime**, not of any single component. Demonstrating that with reproducible,
*controlled* experiments — including a proper pretraining ablation that changes
only one variable — is the point of building the whole stack from scratch.

## Full classification reports

Precision / recall / F1 per class, plus confusion matrices (rows = true, columns =
predicted), as emitted by Polaris' own evaluation framework.

### Mean-pooling baseline — whitespace (acc 0.8564)

```
class        precision    recall        f1   support
neg             0.8573    0.8552    0.8563     12500
pos             0.8556    0.8577    0.8566     12500
accuracy                            0.8564     25000
macro avg       0.8564    0.8564    0.8564     25000

confusion:   neg  10690  1810
             pos   1779 10721
```

### Transformer encoder — whitespace (acc 0.8550)

```
class        precision    recall        f1   support
neg             0.8546    0.8557    0.8551     12500
pos             0.8555    0.8544    0.8549     12500
accuracy                            0.8550     25000
macro avg       0.8550    0.8550    0.8550     25000

confusion:   neg  10696  1804
             pos   1820 10680
```

### Mean-pooling baseline — BPE (acc 0.8394)

```
class        precision    recall        f1   support
neg             0.8361    0.8442    0.8401     12500
pos             0.8426    0.8346    0.8386     12500
accuracy                            0.8394     25000
macro avg       0.8394    0.8394    0.8394     25000

confusion:   neg  10552  1948
             pos   2068 10432
```

### Transformer encoder — BPE (acc 0.8378)

```
class        precision    recall        f1   support
neg             0.8265    0.8552    0.8406     12500
pos             0.8500    0.8205    0.8350     12500
accuracy                            0.8378     25000
macro avg       0.8382    0.8378    0.8378     25000

confusion:   neg  10690  1810
             pos   2244 10256
```

### Mean-pooling baseline — GloVe 100d (acc 0.8566)

```
class        precision    recall        f1   support
neg             0.8569    0.8561    0.8565     12500
pos             0.8562    0.8570    0.8566     12500
accuracy                            0.8566     25000
macro avg       0.8566    0.8566    0.8566     25000

confusion:   neg  10701  1799
             pos   1787 10713
```

### Transformer encoder — GloVe 100d (acc 0.8494)

```
class        precision    recall        f1   support
neg             0.8662    0.8263    0.8458     12500
pos             0.8340    0.8724    0.8528     12500
accuracy                            0.8494     25000
macro avg       0.8501    0.8494    0.8493     25000

confusion:   neg  10329  2171
             pos   1595 10905
```

### 4-layer transformer — random init, no pretraining (acc 0.8518)

```
class        precision    recall        f1   support
neg             0.8775    0.8177    0.8465     12500
pos             0.8293    0.8858    0.8566     12500
accuracy                            0.8518     25000
macro avg       0.8534    0.8518    0.8516     25000

confusion:   neg  10221  2279
             pos   1427 11073
```

### 4-layer transformer — MLM-pretrained (acc 0.8534)

```
class        precision    recall        f1   support
neg             0.8525    0.8546    0.8536     12500
pos             0.8543    0.8522    0.8532     12500
accuracy                            0.8534     25000
macro avg       0.8534    0.8534    0.8534     25000

confusion:   neg  10683  1817
             pos   1848 10652
```

<sub>All runs recorded under `runs/` (config, metrics, environment, report). Numbers
above are copied verbatim from those records.</sub>
