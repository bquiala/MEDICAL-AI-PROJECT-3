"""Training loop for BiLSTM sentence classification baseline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from medical_ai_project.data.pubmed_rct20k import class_weights_from_labels
from medical_ai_project.evaluation.metrics import (
    bootstrap_metric_ci,
    compute_classification_metrics,
    make_confusion_matrix_table,
)
from medical_ai_project.models.lstm_classifier import LSTMSentenceClassifier
from medical_ai_project.utils.io_utils import save_json
from medical_ai_project.utils.logging_utils import setup_logger


def _epoch_step(model: nn.Module, dataloader, loss_fn, optimizer, device: torch.device, train: bool) -> tuple[float, list[int], list[int]]:
    """Run one train/validation epoch."""
    model.train(train)
    running_loss = 0.0
    all_true: list[int] = []
    all_pred: list[int] = []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch_inputs, batch_labels, batch_attention in tqdm(dataloader, leave=False):
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            batch_attention = batch_attention.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(batch_inputs, batch_attention)
            loss = loss_fn(logits, batch_labels)

            if train:
                loss.backward()
                optimizer.step()

            running_loss += float(loss.item())
            preds = logits.argmax(dim=1)
            all_true.extend(batch_labels.detach().cpu().tolist())
            all_pred.extend(preds.detach().cpu().tolist())

    mean_loss = running_loss / max(len(dataloader), 1)
    return mean_loss, all_true, all_pred


def train_lstm_classification(
    config: dict,
    dataloaders: dict,
    vocab_size: int,
    pad_id: int,
    num_classes: int,
    id2label: dict[int, str],
    output_dir: str,
    train_labels: list[int],
) -> dict:
    """Train sentence classifier and persist metrics/predictions."""
    logger = setup_logger("lstm_cls_trainer")
    cfg = config["classification"]["lstm"]

    device = torch.device(
        "cuda"
        if config["runtime"]["device"] == "auto" and torch.cuda.is_available()
        else "cpu"
    )

    model = LSTMSentenceClassifier(
        vocab_size=vocab_size,
        embedding_dim=cfg["embedding_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        num_classes=num_classes,
        pad_id=pad_id,
    ).to(device)

    if cfg.get("class_weighting", True):
        class_weights = class_weights_from_labels(train_labels, num_classes=num_classes).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fn = nn.CrossEntropyLoss()

    optimizer = Adam(model.parameters(), lr=cfg["learning_rate"])

    history = {"train_loss": [], "val_loss": [], "train_f1_macro": [], "val_f1_macro": []}
    best_val_loss = float("inf")

    model_dir = Path(output_dir) / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = model_dir / "best_lstm_cls.pt"

    label_names = [id2label[idx] for idx in sorted(id2label)]

    for epoch in range(cfg["num_epochs"]):
        train_loss, train_true, train_pred = _epoch_step(
            model,
            dataloaders["train"],
            loss_fn,
            optimizer,
            device,
            train=True,
        )
        val_loss, val_true, val_pred = _epoch_step(
            model,
            dataloaders["validation"],
            loss_fn,
            optimizer,
            device,
            train=False,
        )

        train_metrics = compute_classification_metrics(train_true, train_pred, label_names=label_names)
        val_metrics = compute_classification_metrics(val_true, val_pred, label_names=label_names)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1_macro"].append(train_metrics["f1_macro"])
        history["val_f1_macro"].append(val_metrics["f1_macro"])

        logger.info(
            "Epoch %d/%d | train_loss=%.4f val_loss=%.4f train_f1_macro=%.4f val_f1_macro=%.4f",
            epoch + 1,
            cfg["num_epochs"],
            train_loss,
            val_loss,
            train_metrics["f1_macro"],
            val_metrics["f1_macro"],
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    rows = []
    true_labels: list[int] = []
    pred_labels: list[int] = []

    with torch.no_grad():
        for batch_inputs, batch_labels, batch_attention in dataloaders["test"]:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            batch_attention = batch_attention.to(device)

            logits = model(batch_inputs, batch_attention)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            true_list = batch_labels.cpu().tolist()
            pred_list = preds.cpu().tolist()
            conf_list = probs.max(dim=1).values.cpu().tolist()

            true_labels.extend(true_list)
            pred_labels.extend(pred_list)

            for truth, pred, confidence in zip(true_list, pred_list, conf_list):
                rows.append(
                    {
                        "true_label_id": int(truth),
                        "pred_label_id": int(pred),
                        "true_label": id2label[int(truth)],
                        "pred_label": id2label[int(pred)],
                        "confidence": float(confidence),
                    }
                )

    metrics = compute_classification_metrics(true_labels, pred_labels, label_names=label_names)
    metrics["accuracy_ci95"] = bootstrap_metric_ci(true_labels, pred_labels, metric_name="accuracy")
    metrics["f1_macro_ci95"] = bootstrap_metric_ci(true_labels, pred_labels, metric_name="f1_macro")

    cm_rows = make_confusion_matrix_table(true_labels, pred_labels, label_names)

    metrics_dir = Path(output_dir) / "metrics"
    predictions_dir = Path(output_dir) / "predictions"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    save_json(metrics, metrics_dir / "test_metrics.json")
    save_json(history, metrics_dir / "training_history.json")
    pd.DataFrame(cm_rows).to_csv(metrics_dir / "confusion_matrix.csv", index=False)
    pd.DataFrame(rows).to_csv(predictions_dir / "test_predictions.csv", index=False)

    return metrics
