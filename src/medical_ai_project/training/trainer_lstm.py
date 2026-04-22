"""Training loop for BiLSTM NER baseline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from medical_ai_project.data.pubmed_rct20k import class_weights_from_labels
from medical_ai_project.evaluation.metrics import (
    bio_tags_to_spans,
    compute_ner_metrics,
    make_span_level_report,
)
from medical_ai_project.models.lstm_classifier import LSTMClassifier
from medical_ai_project.utils.io_utils import save_json
from medical_ai_project.utils.logging_utils import setup_logger


def _epoch_step(
    model: nn.Module,
    dataloader,
    loss_fn,
    optimizer,
    device: torch.device,
    train: bool,
    id2label: dict[int, str],
    ignore_index: int,
) -> tuple[float, float]:
    """Run one train/validation epoch and return token-level loss and F1."""
    model.train(train)

    running_loss = 0.0
    all_pred_sequences: list[list[str]] = []
    all_true_sequences: list[list[str]] = []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch_inputs, batch_labels, _batch_attention in tqdm(dataloader, leave=False):
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(batch_inputs)
            loss = loss_fn(logits.view(-1, logits.shape[-1]), batch_labels.view(-1))

            if train:
                loss.backward()
                optimizer.step()

            running_loss += float(loss.item())
            preds = logits.argmax(dim=2)

            preds_cpu = preds.detach().cpu().tolist()
            labels_cpu = batch_labels.detach().cpu().tolist()
            for pred_seq, true_seq in zip(preds_cpu, labels_cpu):
                aligned_pred = []
                aligned_true = []
                for pred_id, true_id in zip(pred_seq, true_seq):
                    if true_id == ignore_index:
                        continue
                    aligned_pred.append(id2label[pred_id])
                    aligned_true.append(id2label[true_id])
                all_pred_sequences.append(aligned_pred)
                all_true_sequences.append(aligned_true)

    mean_loss = running_loss / max(len(dataloader), 1)
    metrics = compute_ner_metrics(all_true_sequences, all_pred_sequences)
    return mean_loss, float(metrics["entity_f1"])


def train_lstm(
    config: dict,
    dataloaders: dict,
    vocab_size: int,
    pad_id: int,
    num_classes: int,
    id2label: dict[int, str],
    vocab_itos: list[str],
    output_dir: str,
    train_labels: list[int],
    ignore_index: int = -100,
) -> dict:
    """Train BiLSTM NER model and persist span-aware outputs."""
    logger = setup_logger("lstm_trainer")
    lstm_cfg = config["lstm"]

    device = torch.device(
        "cuda"
        if config["runtime"]["device"] == "auto" and torch.cuda.is_available()
        else "cpu"
    )

    model = LSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=lstm_cfg["embedding_dim"],
        hidden_dim=lstm_cfg["hidden_dim"],
        num_layers=lstm_cfg["num_layers"],
        dropout=lstm_cfg["dropout"],
        num_classes=num_classes,
        pad_id=pad_id,
    ).to(device)

    if lstm_cfg.get("class_weighting", False):
        class_weights = class_weights_from_labels(
            train_labels,
            num_classes=num_classes,
            ignore_index=ignore_index,
        ).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
    else:
        loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    optimizer = Adam(model.parameters(), lr=lstm_cfg["learning_rate"])

    history = {"train_loss": [], "val_loss": [], "train_entity_f1": [], "val_entity_f1": []}
    best_val_loss = float("inf")
    model_dir = Path(output_dir) / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = model_dir / "best_lstm.pt"

    for epoch in range(lstm_cfg["num_epochs"]):
        train_loss, train_f1 = _epoch_step(
            model,
            dataloaders["train"],
            loss_fn,
            optimizer,
            device,
            train=True,
            id2label=id2label,
            ignore_index=ignore_index,
        )
        val_loss, val_f1 = _epoch_step(
            model,
            dataloaders["validation"],
            loss_fn,
            optimizer,
            device,
            train=False,
            id2label=id2label,
            ignore_index=ignore_index,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_entity_f1"].append(train_f1)
        history["val_entity_f1"].append(val_f1)

        logger.info(
            "Epoch %d/%d | train_loss=%.4f val_loss=%.4f train_entity_f1=%.4f val_entity_f1=%.4f",
            epoch + 1,
            lstm_cfg["num_epochs"],
            train_loss,
            val_loss,
            train_f1,
            val_f1,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    true_tag_sequences: list[list[str]] = []
    pred_tag_sequences: list[list[str]] = []
    prediction_rows = []

    with torch.no_grad():
        for batch_inputs, batch_labels, _batch_attention in dataloaders["test"]:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_inputs)
            preds = logits.argmax(dim=2)

            preds_cpu = preds.cpu().tolist()
            labels_cpu = batch_labels.cpu().tolist()
            inputs_cpu = batch_inputs.cpu().tolist()

            for input_ids, pred_seq, true_seq in zip(inputs_cpu, preds_cpu, labels_cpu):
                tokens = []
                true_tags = []
                pred_tags = []

                for token_id, pred_id, true_id in zip(input_ids, pred_seq, true_seq):
                    if true_id == ignore_index:
                        continue
                    if 0 <= token_id < len(vocab_itos):
                        tokens.append(vocab_itos[token_id])
                    else:
                        tokens.append("<unk>")
                    true_tags.append(id2label[true_id])
                    pred_tags.append(id2label[pred_id])

                true_tag_sequences.append(true_tags)
                pred_tag_sequences.append(pred_tags)

                prediction_rows.append(
                    {
                        "tokens": " ".join(tokens),
                        "true_tags": " ".join(true_tags),
                        "pred_tags": " ".join(pred_tags),
                        "true_spans": str(bio_tags_to_spans(true_tags)),
                        "pred_spans": str(bio_tags_to_spans(pred_tags)),
                    }
                )

    metrics = compute_ner_metrics(true_tag_sequences, pred_tag_sequences)
    report = make_span_level_report(true_tag_sequences, pred_tag_sequences)

    metrics_dir = Path(output_dir) / "metrics"
    predictions_dir = Path(output_dir) / "predictions"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    save_json(metrics, metrics_dir / "test_metrics.json")
    save_json(report, metrics_dir / "entity_report.json")
    save_json(history, metrics_dir / "training_history.json")

    predictions_df = pd.DataFrame(prediction_rows)
    predictions_df.to_csv(predictions_dir / "test_predictions.csv", index=False)

    return metrics
