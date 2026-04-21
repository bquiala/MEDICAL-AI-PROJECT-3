"""CLI to compute evaluation artifacts from prediction files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from medical_ai_project.evaluation.metrics import (
    compute_classification_metrics,
    make_classification_report,
    save_confusion_matrix,
)
from medical_ai_project.utils.io_utils import save_json


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate model predictions")
    parser.add_argument("--predictions", type=str, required=True, help="CSV with true_label and pred_label")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for evaluation outputs")
    return parser.parse_args()


def main() -> None:
    """Run standalone metrics computation for prediction CSVs."""
    args = parse_args()
    df = pd.read_csv(args.predictions)

    required_columns = {"true_label", "pred_label"}
    if not required_columns.issubset(df.columns):
        raise ValueError("Prediction CSV must include true_label and pred_label columns")

    labels = sorted(set(df["true_label"].tolist()) | set(df["pred_label"].tolist()))
    label_to_id = {label: idx for idx, label in enumerate(labels)}

    y_true = [label_to_id[val] for val in df["true_label"].tolist()]
    y_pred = [label_to_id[val] for val in df["pred_label"].tolist()]

    metrics = compute_classification_metrics(y_true, y_pred)
    report = make_classification_report(y_true, y_pred, labels)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_json(metrics, output_dir / "test_metrics.json")
    save_json(report, output_dir / "classification_report.json")
    save_confusion_matrix(y_true, y_pred, labels, str(output_dir / "confusion_matrix.png"))


if __name__ == "__main__":
    main()
