"""Metric computation utilities for BIO NER."""

from __future__ import annotations

def bio_tags_to_spans(tags: list[str]) -> list[dict]:
    """Convert BIO tags into entity spans with start/end token offsets."""
    spans: list[dict] = []
    active_type = None
    start_idx = None

    def close_span(end_idx: int) -> None:
        nonlocal active_type, start_idx
        if active_type is not None and start_idx is not None:
            spans.append(
                {
                    "entity": active_type,
                    "start": int(start_idx),
                    "end": int(end_idx),
                }
            )
        active_type = None
        start_idx = None

    for index, tag in enumerate(tags):
        if tag == "O":
            close_span(index - 1)
            continue

        prefix, _, entity = tag.partition("-")
        if prefix == "B":
            close_span(index - 1)
            active_type = entity
            start_idx = index
        elif prefix == "I":
            if active_type != entity:
                close_span(index - 1)
                active_type = entity
                start_idx = index
        else:
            close_span(index - 1)

    close_span(len(tags) - 1)
    return spans


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def compute_ner_metrics(y_true_tags: list[list[str]], y_pred_tags: list[list[str]]) -> dict:
    """Compute entity-level precision/recall/F1 and token accuracy."""
    true_spans = []
    pred_spans = []
    total_tokens = 0
    correct_tokens = 0

    for true_tags, pred_tags in zip(y_true_tags, y_pred_tags):
        true_spans.append(set((span["entity"], span["start"], span["end"]) for span in bio_tags_to_spans(true_tags)))
        pred_spans.append(set((span["entity"], span["start"], span["end"]) for span in bio_tags_to_spans(pred_tags)))

        for true_tag, pred_tag in zip(true_tags, pred_tags):
            total_tokens += 1
            if true_tag == pred_tag:
                correct_tokens += 1

    tp = sum(len(true_set & pred_set) for true_set, pred_set in zip(true_spans, pred_spans))
    fp = sum(len(pred_set - true_set) for true_set, pred_set in zip(true_spans, pred_spans))
    fn = sum(len(true_set - pred_set) for true_set, pred_set in zip(true_spans, pred_spans))

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    return {
        "token_accuracy": _safe_divide(correct_tokens, total_tokens),
        "entity_precision": precision,
        "entity_recall": recall,
        "entity_f1": f1,
        "true_entities": int(tp + fn),
        "predicted_entities": int(tp + fp),
        "true_positives": int(tp),
    }


def make_span_level_report(y_true_tags: list[list[str]], y_pred_tags: list[list[str]]) -> dict:
    """Build a per-entity-type report from BIO sequences."""
    by_type = {}

    for true_tags, pred_tags in zip(y_true_tags, y_pred_tags):
        true_spans = bio_tags_to_spans(true_tags)
        pred_spans = bio_tags_to_spans(pred_tags)

        true_sets = {}
        pred_sets = {}
        for span in true_spans:
            true_sets.setdefault(span["entity"], set()).add((span["start"], span["end"]))
        for span in pred_spans:
            pred_sets.setdefault(span["entity"], set()).add((span["start"], span["end"]))

        entity_types = set(true_sets.keys()) | set(pred_sets.keys())
        for entity_type in entity_types:
            stats = by_type.setdefault(entity_type, {"tp": 0, "fp": 0, "fn": 0})
            true_set = true_sets.get(entity_type, set())
            pred_set = pred_sets.get(entity_type, set())
            stats["tp"] += len(true_set & pred_set)
            stats["fp"] += len(pred_set - true_set)
            stats["fn"] += len(true_set - pred_set)

    report = {}
    for entity_type, stats in sorted(by_type.items()):
        precision = _safe_divide(stats["tp"], stats["tp"] + stats["fp"])
        recall = _safe_divide(stats["tp"], stats["tp"] + stats["fn"])
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        report[entity_type] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(stats["tp"] + stats["fn"]),
        }

    return report
