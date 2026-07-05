# Evaluation Module

Measure a trained model honestly: core metrics and a human-readable classification
report, all computed from scratch (no scikit-learn). This is the layer the
benchmarks and the `polaris` tooling report through.

## Public surface

- Metrics (`metrics.py`):
  - `evaluate(model, batches)` — mean loss and accuracy over batches.
  - `predict(model, batches)` — predicted and true labels.
  - `accuracy`, `confusion_matrix`, `precision_recall_f1` — the underlying
    computations, per class and averaged.
- Report (`report.py`):
  - `ClassificationReport` — an immutable value object holding per-class
    precision/recall/F1, support, accuracy, and the confusion matrix, with a
    `to_text()` table renderer.
  - `classification_report(...)` / `evaluate_model(model, batches, ...)` — build a
    report from labels, or straight from a model and batches.

## Example

```python
from polaris.evaluation import evaluate_model

report = evaluate_model(model, test_batches, num_classes=2, class_names=("neg", "pos"))
print(report.to_text())
```
