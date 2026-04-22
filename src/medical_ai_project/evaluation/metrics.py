"""Metric computation utilities for BIO NER."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

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


def compute_classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    label_names: list[str] | None = None,
) -> dict:
    """Compute sentence classification metrics and optional per-class report."""
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    report = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
    }

    if label_names is not None:
        precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(range(len(label_names))),
            average=None,
            zero_division=0,
        )
        report["per_class"] = {
            label_names[idx]: {
                "precision": float(precision_per[idx]),
                "recall": float(recall_per[idx]),
                "f1": float(f1_per[idx]),
                "support": int(support_per[idx]),
            }
            for idx in range(len(label_names))
        }

    return report


def bootstrap_metric_ci(
    y_true: list[int],
    y_pred: list[int],
    metric_name: str,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Estimate bootstrap confidence interval for accuracy or macro-F1."""
    if metric_name not in {"accuracy", "f1_macro"}:
        raise ValueError("metric_name must be one of {'accuracy', 'f1_macro'}")

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")

    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n_bootstrap": int(n_bootstrap)}

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    def _score(indices: np.ndarray) -> float:
        yt = y_true_arr[indices]
        yp = y_pred_arr[indices]
        if metric_name == "accuracy":
            return float(accuracy_score(yt, yp))
        return float(f1_score(yt, yp, average="macro", zero_division=0))

    scores = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        scores.append(_score(indices))

    scores_arr = np.asarray(scores, dtype=float)
    low_q = float(np.quantile(scores_arr, alpha / 2.0))
    high_q = float(np.quantile(scores_arr, 1.0 - alpha / 2.0))
    return {
        "mean": float(scores_arr.mean()),
        "ci_low": low_q,
        "ci_high": high_q,
        "n_bootstrap": int(n_bootstrap),
    }


def make_confusion_matrix_table(y_true: list[int], y_pred: list[int], label_names: list[str]) -> list[dict]:
    """Return confusion matrix rows as dictionaries for CSV/JSON export."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    rows: list[dict] = []
    for i, true_name in enumerate(label_names):
        row = {"true_label": true_name}
        for j, pred_name in enumerate(label_names):
            row[pred_name] = int(cm[i, j])
        rows.append(row)
    return rows
