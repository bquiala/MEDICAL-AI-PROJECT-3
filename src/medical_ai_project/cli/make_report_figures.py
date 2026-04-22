"""Generate report figures and summary tables from experiment artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build report figures from saved metrics")
    parser.add_argument("--artifacts-root", type=str, default="artifacts", help="Artifacts root path")
    parser.add_argument("--output-dir", type=str, default="reports/figures", help="Figure output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts_root = Path(args.artifacts_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    ner_lstm_metrics = _load_json(artifacts_root / "lstm" / "metrics" / "test_metrics.json")
    ner_trf_metrics = _load_json(artifacts_root / "transformer" / "metrics" / "test_metrics.json")
    cls_lstm_metrics = _load_json(artifacts_root / "classification_lstm" / "metrics" / "test_metrics.json")
    cls_trf_metrics = _load_json(artifacts_root / "classification_transformer" / "metrics" / "test_metrics.json")

    summary = pd.DataFrame(
        [
            {"task": "NER", "model": "BiLSTM", "metric": "Entity F1", "value": ner_lstm_metrics["entity_f1"]},
            {"task": "NER", "model": "Transformer", "metric": "Entity F1", "value": ner_trf_metrics["entity_f1"]},
            {"task": "Classification", "model": "BiLSTM", "metric": "Macro F1", "value": cls_lstm_metrics["f1_macro"]},
            {"task": "Classification", "model": "Transformer", "metric": "Macro F1", "value": cls_trf_metrics["f1_macro"]},
        ]
    )
    summary.to_csv(output_dir / "model_summary_metrics.csv", index=False)

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=summary, x="task", y="value", hue="model")
    ax.set_title("Model Comparison Across Tasks")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(output_dir / "figure_1_model_comparison.png", dpi=200)
    plt.close()

    ner_report_lstm = _load_json(artifacts_root / "lstm" / "metrics" / "entity_report.json")
    ner_report_trf = _load_json(artifacts_root / "transformer" / "metrics" / "entity_report.json")

    entity_rows = []
    for entity_type, stats in ner_report_lstm.items():
        entity_rows.append({"entity": entity_type, "model": "BiLSTM", "f1": stats.get("f1", 0.0)})
    for entity_type, stats in ner_report_trf.items():
        entity_rows.append({"entity": entity_type, "model": "Transformer", "f1": stats.get("f1", 0.0)})
    entity_df = pd.DataFrame(entity_rows)
    entity_df.to_csv(output_dir / "ner_entity_f1_by_model.csv", index=False)

    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=entity_df, x="entity", y="f1", hue="model")
    ax.set_title("NER Per-Entity F1 by Model")
    ax.set_ylabel("F1 Score")
    ax.set_xlabel("Entity Type")
    ax.set_ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(output_dir / "figure_2_ner_entity_f1.png", dpi=200)
    plt.close()

    cm = pd.read_csv(artifacts_root / "classification_transformer" / "metrics" / "confusion_matrix.csv")
    labels = cm["true_label"].tolist()
    matrix = cm.drop(columns=["true_label"]).to_numpy()

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    ax.set_title("Transformer Classification Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_dir / "figure_3_transformer_confusion_matrix.png", dpi=200)
    plt.close()

    pred_df = pd.read_csv(artifacts_root / "classification_transformer" / "predictions" / "test_predictions.csv")
    pred_df["is_correct"] = pred_df["true_label"] == pred_df["pred_label"]
    confidence_summary = (
        pred_df.groupby("is_correct", as_index=False)["confidence"]
        .mean()
        .assign(outcome=lambda df: df["is_correct"].map({True: "Correct", False: "Incorrect"}))
    )

    plt.figure(figsize=(6, 5))
    ax = sns.barplot(data=confidence_summary, x="outcome", y="confidence")
    ax.set_title("Transformer Confidence by Prediction Outcome")
    ax.set_ylabel("Mean Confidence")
    ax.set_ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(output_dir / "figure_4_confidence_vs_correctness.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
