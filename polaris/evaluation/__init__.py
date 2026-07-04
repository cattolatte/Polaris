"""
Evaluation subsystem for Polaris.
"""

from __future__ import annotations

from polaris.evaluation.metrics import (
    accuracy,
    confusion_matrix,
    evaluate,
    precision_recall_f1,
    predict,
)
from polaris.evaluation.report import (
    ClassificationReport,
    classification_report,
    evaluate_model,
)

__all__ = [
    "ClassificationReport",
    "accuracy",
    "classification_report",
    "confusion_matrix",
    "evaluate",
    "evaluate_model",
    "precision_recall_f1",
    "predict",
]
