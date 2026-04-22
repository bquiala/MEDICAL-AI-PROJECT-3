"""CLI to compute evaluation artifacts from prediction files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from medical_ai_project.evaluation.metrics import (
    compute_ner_metrics,
    make_span_level_report,
)
from medical_ai_project.utils.io_utils import save_json


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate model predictions")
    parser.add_argument("--predictions", type=str, required=True, help="CSV with true_tags and pred_tags")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for evaluation outputs")
    return parser.parse_args()


def main() -> None:
    """Run standalone metrics computation for prediction CSVs."""
    args = parse_args()
    df = pd.read_csv(args.predictions)

    required_columns = {"true_tags", "pred_tags"}
    if not required_columns.issubset(df.columns):
        raise ValueError("Prediction CSV must include true_tags and pred_tags columns")

    true_sequences = [str(value).split() for value in df["true_tags"].fillna("").tolist()]
    pred_sequences = [str(value).split() for value in df["pred_tags"].fillna("").tolist()]

    metrics = compute_ner_metrics(true_sequences, pred_sequences)
    report = make_span_level_report(true_sequences, pred_sequences)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_json(metrics, output_dir / "test_metrics.json")
    save_json(report, output_dir / "entity_report.json")


if __name__ == "__main__":
    main()
