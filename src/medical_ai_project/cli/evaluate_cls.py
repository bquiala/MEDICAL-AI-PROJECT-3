"""CLI to compute sentence-classification evaluation artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from medical_ai_project.evaluation.metrics import (
    bootstrap_metric_ci,
    compute_classification_metrics,
    make_confusion_matrix_table,
)
from medical_ai_project.utils.io_utils import save_json


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate sentence classification predictions")
    parser.add_argument("--predictions", type=str, required=True, help="CSV with true/pred label columns")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for evaluation outputs")
    parser.add_argument(
        "--true-column",
        type=str,
        default="true_label_id",
        help="Column name for true label ids",
    )
    parser.add_argument(
        "--pred-column",
        type=str,
        default="pred_label_id",
        help="Column name for predicted label ids",
    )
    return parser.parse_args()


def main() -> None:
    """Run standalone metrics computation for classification prediction CSVs."""
    args = parse_args()
    df = pd.read_csv(args.predictions)

    required_columns = {args.true_column, args.pred_column}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Prediction CSV must include columns: {sorted(required_columns)}"
        )

    y_true = df[args.true_column].astype(int).tolist()
    y_pred = df[args.pred_column].astype(int).tolist()

    ids = sorted(set(y_true) | set(y_pred))
    if "true_label" in df.columns and "true_label_id" in df.columns:
        id_to_name = {}
        for row in df[["true_label_id", "true_label"]].dropna().itertuples(index=False):
            id_to_name[int(row.true_label_id)] = str(row.true_label)
        label_names = [id_to_name.get(idx, str(idx)) for idx in ids]
    else:
        label_names = [str(idx) for idx in ids]

    metrics = compute_classification_metrics(y_true, y_pred)
    metrics["accuracy_ci95"] = bootstrap_metric_ci(y_true, y_pred, metric_name="accuracy")
    metrics["f1_macro_ci95"] = bootstrap_metric_ci(y_true, y_pred, metric_name="f1_macro")

    cm_rows = make_confusion_matrix_table(y_true, y_pred, label_names)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_json(metrics, output_dir / "test_metrics.json")
    pd.DataFrame(cm_rows).to_csv(output_dir / "confusion_matrix.csv", index=False)


if __name__ == "__main__":
    main()
