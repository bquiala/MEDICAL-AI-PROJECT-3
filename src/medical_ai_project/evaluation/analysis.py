"""Qualitative error analysis helpers."""

from __future__ import annotations

import pandas as pd


def summarize_error_modes(predictions_df: pd.DataFrame, top_k: int = 20) -> dict:
    """Create a compact summary of common model errors.

    Expected columns: text, true_label, pred_label.
    """
    errors = predictions_df[predictions_df["true_label"] != predictions_df["pred_label"]].copy()
    if errors.empty:
        return {"total_errors": 0, "top_confusions": []}

    confusions = (
        errors.groupby(["true_label", "pred_label"])  # type: ignore[arg-type]
        .size()
        .sort_values(ascending=False)
        .head(top_k)
        .reset_index(name="count")
    )

    return {
        "total_errors": int(len(errors)),
        "top_confusions": confusions.to_dict(orient="records"),
    }


def extract_representative_examples(
    predictions_df: pd.DataFrame,
    max_correct: int = 10,
    max_errors: int = 10,
) -> dict:
    """Collect representative correct and incorrect predictions."""
    correct = predictions_df[predictions_df["true_label"] == predictions_df["pred_label"]].head(max_correct)
    errors = predictions_df[predictions_df["true_label"] != predictions_df["pred_label"]].head(max_errors)

    return {
        "correct_examples": correct.to_dict(orient="records"),
        "error_examples": errors.to_dict(orient="records"),
    }
