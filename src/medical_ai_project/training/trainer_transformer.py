"""Trainer for transformer fine-tuning."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from medical_ai_project.evaluation.metrics import (
    compute_classification_metrics,
    make_classification_report,
    save_confusion_matrix,
)
from medical_ai_project.utils.io_utils import save_json


def train_transformer(
    config: dict,
    dataset,
    id2label: dict[int, str],
    output_dir: str,
) -> dict:
    """Fine-tune a transformer model and save metrics/predictions."""
    ds_cfg = config["dataset"]
    tr_cfg = config["transformer"]
    text_col = ds_cfg["text_column"]
    label_col = ds_cfg["label_column"]

    tokenizer = AutoTokenizer.from_pretrained(tr_cfg["model_name"])

    def tokenize_fn(examples: dict) -> dict:
        return tokenizer(
            examples[text_col],
            truncation=True,
            max_length=tr_cfg["max_seq_len"],
        )

    tokenized = dataset.map(tokenize_fn, batched=True)
    keep_columns = ["input_ids", "attention_mask", label_col]
    if "token_type_ids" in tokenized["train"].column_names:
        keep_columns.append("token_type_ids")

    tokenized = tokenized.remove_columns(
        [c for c in tokenized["train"].column_names if c not in keep_columns]
    )
    tokenized = tokenized.rename_column(label_col, "labels")
    tokenized.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        tr_cfg["model_name"],
        num_labels=len(id2label),
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
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return compute_classification_metrics(labels.tolist(), preds.tolist())

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    test_predictions = trainer.predict(tokenized["test"])

    y_true = test_predictions.label_ids.tolist()
    y_pred = np.argmax(test_predictions.predictions, axis=1).tolist()
    label_names = [id2label[i] for i in range(len(id2label))]

    metrics = compute_classification_metrics(y_true, y_pred)
    report = make_classification_report(y_true, y_pred, label_names)
    save_json(metrics, metrics_dir / "test_metrics.json")
    save_json(report, metrics_dir / "classification_report.json")

    save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=label_names,
        output_path=str(metrics_dir / "confusion_matrix.png"),
    )

    predictions_df = pd.DataFrame(
        {
            "text": dataset["test"][text_col],
            "true_label": [id2label[idx] for idx in y_true],
            "pred_label": [id2label[idx] for idx in y_pred],
        }
    )
    predictions_df.to_csv(predictions_dir / "test_predictions.csv", index=False)

    return metrics
