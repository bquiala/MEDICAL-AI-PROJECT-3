"""CLI for qualitative error analysis summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from medical_ai_project.evaluation.analysis import (
    extract_representative_examples,
    summarize_error_modes,
)
from medical_ai_project.utils.io_utils import save_json


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate qualitative error analysis")
    parser.add_argument("--predictions", type=str, required=True, help="CSV with text, true_label, pred_label")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for error analysis outputs")
    return parser.parse_args()


def main() -> None:
    """Run error analysis on prediction CSV."""
    args = parse_args()
    df = pd.read_csv(args.predictions)

    required_columns = {"true_label", "pred_label"}
    if not required_columns.issubset(df.columns):
        raise ValueError("Prediction CSV must include true_label and pred_label columns")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_error_modes(df)
    examples = extract_representative_examples(df)

    save_json(summary, output_dir / "error_summary.json")
    save_json(examples, output_dir / "example_slices.json")


if __name__ == "__main__":
    main()
