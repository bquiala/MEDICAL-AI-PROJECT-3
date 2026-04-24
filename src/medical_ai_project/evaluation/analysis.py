"""Qualitative error analysis helpers."""

from __future__ import annotations

import pandas as pd

from medical_ai_project.evaluation.metrics import bio_tags_to_spans


def _parse_tags(value: str) -> list[str]:
    """Split a space-delimited tag string into a list.

    Args:
        value: Tag string (e.g., "O B-DISEASE I-DISEASE").

    Returns:
        List of tag tokens. Returns an empty list for None.
    """
    return str(value).split() if value is not None else []


def summarize_error_modes(predictions_df: pd.DataFrame, top_k: int = 20) -> dict:
    """Create a compact summary of missed and spurious NER entities."""
    confusion_counter: dict[tuple[str, str], int] = {}
    total_errors = 0

    for row in predictions_df.itertuples(index=False):
        true_tags = _parse_tags(getattr(row, "true_tags", ""))
        pred_tags = _parse_tags(getattr(row, "pred_tags", ""))

        true_spans = set((span["entity"], span["start"], span["end"]) for span in bio_tags_to_spans(true_tags))
        pred_spans = set((span["entity"], span["start"], span["end"]) for span in bio_tags_to_spans(pred_tags))

        missed = true_spans - pred_spans
        spurious = pred_spans - true_spans
        total_errors += len(missed) + len(spurious)

        for entity, _start, _end in missed:
            key = (entity, "MISSED")
            confusion_counter[key] = confusion_counter.get(key, 0) + 1
        for entity, _start, _end in spurious:
            key = ("SPURIOUS", entity)
            confusion_counter[key] = confusion_counter.get(key, 0) + 1

    top_confusions = [
        {"expected_entity": true_label, "observed_entity": pred_label, "count": count}
        for (true_label, pred_label), count in sorted(
            confusion_counter.items(), key=lambda item: item[1], reverse=True
        )[:top_k]
    ]

    return {"total_errors": int(total_errors), "top_confusions": top_confusions}


def extract_representative_examples(
    predictions_df: pd.DataFrame,
    max_correct: int = 10,
    max_errors: int = 10,
) -> dict:
    """Collect representative exact-match and error NER predictions."""
    exact_match_mask = predictions_df["true_tags"].fillna("") == predictions_df["pred_tags"].fillna("")
    correct = predictions_df[exact_match_mask].head(max_correct)
    errors = predictions_df[~exact_match_mask].head(max_errors)

    return {
        "correct_examples": correct.to_dict(orient="records"),
        "error_examples": errors.to_dict(orient="records"),
    }
