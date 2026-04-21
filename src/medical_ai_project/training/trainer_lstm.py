"""Training loop for LSTM baseline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from medical_ai_project.data.pubmed_rct20k import class_weights_from_labels
from medical_ai_project.evaluation.metrics import (
    compute_classification_metrics,
    make_classification_report,
    save_confusion_matrix,
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
) -> tuple[float, float]:
    """Run one train or validation epoch and return loss and accuracy."""
    model.train(train)

    running_loss = 0.0
    all_preds = []
    all_labels = []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch_inputs, batch_labels in tqdm(dataloader, leave=False):
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(batch_inputs)
            loss = loss_fn(logits, batch_labels)

            if train:
                loss.backward()
                optimizer.step()

            running_loss += float(loss.item())
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(batch_labels.detach().cpu().tolist())

    mean_loss = running_loss / max(len(dataloader), 1)
    accuracy = float(accuracy_score(all_labels, all_preds)) if all_labels else 0.0
    return mean_loss, accuracy


def train_lstm(
    config: dict,
    dataloaders: dict,
    vocab_size: int,
    pad_id: int,
    num_classes: int,
    id2label: dict[int, str],
    output_dir: str,
    train_labels: list[int],
) -> dict:
    """Train LSTM model and persist predictions plus metrics."""
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
        class_weights = class_weights_from_labels(train_labels, num_classes=num_classes).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fn = nn.CrossEntropyLoss()

    optimizer = Adam(model.parameters(), lr=lstm_cfg["learning_rate"])

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    model_dir = Path(output_dir) / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = model_dir / "best_lstm.pt"

    for epoch in range(lstm_cfg["num_epochs"]):
        train_loss, train_acc = _epoch_step(
            model,
            dataloaders["train"],
            loss_fn,
            optimizer,
            device,
            train=True,
        )
        val_loss, val_acc = _epoch_step(
            model,
            dataloaders["validation"],
            loss_fn,
            optimizer,
            device,
            train=False,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        logger.info(
            "Epoch %d/%d | train_loss=%.4f val_loss=%.4f train_acc=%.4f val_acc=%.4f",
            epoch + 1,
            lstm_cfg["num_epochs"],
            train_loss,
            val_loss,
            train_acc,
            val_acc,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch_inputs, batch_labels in dataloaders["test"]:
            batch_inputs = batch_inputs.to(device)
            logits = model(batch_inputs)
            preds = logits.argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(batch_labels.tolist())

    label_names = [id2label[i] for i in range(num_classes)]
    metrics = compute_classification_metrics(y_true, y_pred)
    report = make_classification_report(y_true, y_pred, label_names)

    metrics_dir = Path(output_dir) / "metrics"
    predictions_dir = Path(output_dir) / "predictions"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    save_json(metrics, metrics_dir / "test_metrics.json")
    save_json(report, metrics_dir / "classification_report.json")
    save_json(history, metrics_dir / "training_history.json")
    save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=label_names,
        output_path=str(metrics_dir / "confusion_matrix.png"),
    )

    predictions_df = pd.DataFrame(
        {
            "true_label": [id2label[idx] for idx in y_true],
            "pred_label": [id2label[idx] for idx in y_pred],
        }
    )
    predictions_df.to_csv(predictions_dir / "test_predictions.csv", index=False)

    return metrics
