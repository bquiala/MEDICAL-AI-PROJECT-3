"""Metric computation and confusion matrix utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_classification_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    """Compute standard classification metrics."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def make_classification_report(y_true: list[int], y_pred: list[int], labels: list[str]) -> dict:
    """Build detailed per-class report dictionary."""
    return classification_report(
        y_true,
        y_pred,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )


def save_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    labels: list[str],
    output_path: str,
) -> None:
    """Generate and save confusion matrix heatmap."""
    matrix = confusion_matrix(y_true, y_pred)
    df = pd.DataFrame(matrix, index=labels, columns=labels)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
