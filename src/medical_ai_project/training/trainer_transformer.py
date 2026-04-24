"""Trainer for transformer token-classification NER fine-tuning."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from medical_ai_project.data.pubmed_rct20k import add_weak_ner_annotations, make_ner_labels
from medical_ai_project.evaluation.metrics import (
    bio_tags_to_spans,
    compute_ner_metrics,
    make_span_level_report,
)
from medical_ai_project.utils.io_utils import save_json


def train_transformer(
    config: dict,
    dataset,
    output_dir: str,
    ignore_index: int = -100,
) -> dict:
    """Fine-tune a transformer token classifier and save NER outputs."""
    ds_cfg = config["dataset"]
    tr_cfg = config["transformer"]
    label2id, id2label = make_ner_labels(ds_cfg.get("entity_types", ["DISEASE", "DRUG", "PROCEDURE", "ANATOMY"]))

    annotated = add_weak_ner_annotations(config, dataset)

    tokenizer = AutoTokenizer.from_pretrained(tr_cfg["model_name"])

    def tokenize_fn(examples: dict) -> dict:
        """Tokenize pre-split tokens and align word-level tags to subword tokens."""
        tokenized = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=tr_cfg["max_seq_len"],
        )

        labels = []
        for batch_index, word_ids in enumerate(tokenized.word_ids(batch_index=i) for i in range(len(examples["tokens"]))):
            source_labels = examples["ner_tag_ids"][batch_index]
            previous_word_id = None
            aligned = []
            for word_id in word_ids:
                if word_id is None:
                    aligned.append(ignore_index)
                elif word_id != previous_word_id:
                    aligned.append(source_labels[word_id])
                else:
                    aligned.append(ignore_index)
                previous_word_id = word_id
            labels.append(aligned)

        tokenized["labels"] = labels
        return tokenized

    tokenized = annotated.map(tokenize_fn, batched=True)
    keep_columns = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in tokenized["train"].column_names:
        keep_columns.append("token_type_ids")

    tokenized = tokenized.remove_columns(
        [c for c in tokenized["train"].column_names if c not in keep_columns]
    )
    tokenized.set_format(type="torch", columns=keep_columns)

    model = AutoModelForTokenClassification.from_pretrained(
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
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
    )

    def compute_metrics(eval_pred):
        """Compute NER metrics from transformer logits and aligned labels."""
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=2)

        true_sequences = []
        pred_sequences = []
        for pred_seq, label_seq in zip(preds, labels):
            true_tags = []
            pred_tags = []
            for pred_id, label_id in zip(pred_seq, label_seq):
                if label_id == ignore_index:
                    continue
                true_tags.append(id2label[int(label_id)])
                pred_tags.append(id2label[int(pred_id)])
            true_sequences.append(true_tags)
            pred_sequences.append(pred_tags)

        return compute_ner_metrics(true_sequences, pred_sequences)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    test_predictions = trainer.predict(tokenized["test"])

    label_ids = test_predictions.label_ids
    pred_ids = np.argmax(test_predictions.predictions, axis=2)

    true_tag_sequences = []
    pred_tag_sequences = []
    prediction_rows = []

    for tokens, true_weak_tags, pred_seq, label_seq in zip(
        annotated["test"]["tokens"],
        annotated["test"]["ner_tags"],
        pred_ids,
        label_ids,
    ):
        true_tags = []
        pred_tags = []

        aligned_tokens = []
        token_cursor = 0
        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id == ignore_index:
                continue
            if token_cursor < len(tokens):
                aligned_tokens.append(tokens[token_cursor])
                token_cursor += 1
            true_tags.append(id2label[int(label_id)])
            pred_tags.append(id2label[int(pred_id)])

        # Fallback to weak tags when alignment truncates unexpectedly.
        if not true_tags and true_weak_tags:
            true_tags = list(true_weak_tags)
            pred_tags = ["O"] * len(true_tags)
            aligned_tokens = list(tokens[: len(true_tags)])

        true_tag_sequences.append(true_tags)
        pred_tag_sequences.append(pred_tags)

        prediction_rows.append(
            {
                "tokens": " ".join(aligned_tokens),
                "true_tags": " ".join(true_tags),
                "pred_tags": " ".join(pred_tags),
                "true_spans": str(bio_tags_to_spans(true_tags)),
                "pred_spans": str(bio_tags_to_spans(pred_tags)),
            }
        )

    metrics = compute_ner_metrics(true_tag_sequences, pred_tag_sequences)
    report = make_span_level_report(true_tag_sequences, pred_tag_sequences)
    save_json(metrics, metrics_dir / "test_metrics.json")
    save_json(report, metrics_dir / "entity_report.json")

    predictions_df = pd.DataFrame(prediction_rows)
    predictions_df.to_csv(predictions_dir / "test_predictions.csv", index=False)

    return metrics
