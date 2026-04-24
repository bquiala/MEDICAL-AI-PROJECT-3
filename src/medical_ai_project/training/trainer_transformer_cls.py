"""Trainer for transformer sentence classification fine-tuning."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from medical_ai_project.data.pubmed_rct20k import make_sentence_label_maps
from medical_ai_project.evaluation.metrics import (
    bootstrap_metric_ci,
    compute_classification_metrics,
    make_confusion_matrix_table,
)
from medical_ai_project.utils.io_utils import save_json


def train_transformer_classification(config: dict, dataset, output_dir: str) -> dict:
    """Fine-tune transformer sequence classifier and save outputs."""
    ds_cfg = config["dataset"]
    tr_cfg = config["classification"]["transformer"]
    text_column = ds_cfg["text_column"]
    label_column = ds_cfg.get("label_column", "label")

    label2id, id2label = make_sentence_label_maps(dataset, label_column)
    label_names = [id2label[idx] for idx in sorted(id2label)]

    tokenizer = AutoTokenizer.from_pretrained(tr_cfg["model_name"])

    def preprocess(examples: dict) -> dict:
        """Tokenize text examples and map labels to integer ids."""
        tokenized = tokenizer(
            examples[text_column],
            truncation=True,
            max_length=tr_cfg["max_seq_len"],
        )

        labels = []
        for value in examples[label_column]:
            if isinstance(value, str):
                labels.append(label2id[value])
            else:
                labels.append(int(value))
        tokenized["labels"] = labels
        return tokenized

    tokenized = dataset.map(preprocess, batched=True)
    keep_columns = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in tokenized["train"].column_names:
        keep_columns.append("token_type_ids")
    tokenized = tokenized.remove_columns([c for c in tokenized["train"].column_names if c not in keep_columns])
    tokenized.set_format(type="torch", columns=keep_columns)

    model = AutoModelForSequenceClassification.from_pretrained(
        tr_cfg["model_name"],
        num_labels=len(id2label),
        id2label={idx: label for idx, label in id2label.items()},
        label2id=label2id,
    )

    model_out = Path(output_dir) / "checkpoints"
    metrics_dir = Path(output_dir) / "metrics"
    predictions_dir = Path(output_dir) / "predictions"
    model_out.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(model_out),
        learning_rate=tr_cfg["learning_rate"],
        per_device_train_batch_size=tr_cfg["batch_size"],
        per_device_eval_batch_size=tr_cfg["batch_size"],
        num_train_epochs=tr_cfg["num_epochs"],
        weight_decay=tr_cfg["weight_decay"],
        warmup_ratio=tr_cfg["warmup_ratio"],
        logging_steps=100,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        report_to="none",
        remove_unused_columns=False,
    )

    def compute_metrics(eval_pred):
        """Compute classification metrics from logits and labels."""
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return compute_classification_metrics(labels.tolist(), preds.tolist(), label_names=label_names)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    test_predictions = trainer.predict(tokenized["test"])
    test_pred_ids = np.argmax(test_predictions.predictions, axis=1).tolist()
    test_true_ids = test_predictions.label_ids.tolist()

    metrics = compute_classification_metrics(test_true_ids, test_pred_ids, label_names=label_names)
    metrics["accuracy_ci95"] = bootstrap_metric_ci(test_true_ids, test_pred_ids, metric_name="accuracy")
    metrics["f1_macro_ci95"] = bootstrap_metric_ci(test_true_ids, test_pred_ids, metric_name="f1_macro")

    cm_rows = make_confusion_matrix_table(test_true_ids, test_pred_ids, label_names)
    save_json(metrics, metrics_dir / "test_metrics.json")
    pd.DataFrame(cm_rows).to_csv(metrics_dir / "confusion_matrix.csv", index=False)

    probs = np.exp(test_predictions.predictions)
    probs = probs / probs.sum(axis=1, keepdims=True)
    confidence = probs.max(axis=1)

    rows = []
    for truth, pred, conf in zip(test_true_ids, test_pred_ids, confidence.tolist()):
        rows.append(
            {
                "true_label_id": int(truth),
                "pred_label_id": int(pred),
                "true_label": id2label[int(truth)],
                "pred_label": id2label[int(pred)],
                "confidence": float(conf),
            }
        )
    pd.DataFrame(rows).to_csv(predictions_dir / "test_predictions.csv", index=False)

    return metrics
