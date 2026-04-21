"""CLI to train and evaluate the LSTM baseline."""

from __future__ import annotations

import argparse

from medical_ai_project.data.pubmed_rct20k import create_lstm_dataloaders, load_pubmed_rct20k
from medical_ai_project.training.trainer_lstm import train_lstm
from medical_ai_project.utils.config import load_config
from medical_ai_project.utils.logging_utils import setup_logger
from medical_ai_project.utils.reproducibility import set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train LSTM baseline on PubMed RCT 20k")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    return parser.parse_args()


def main() -> None:
    """Run LSTM training flow."""
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])
    logger = setup_logger("train_lstm")

    dataset = load_pubmed_rct20k(config)
    label_feature = dataset["train"].features[config["dataset"]["label_column"]]
    id2label = {i: name for i, name in enumerate(label_feature.names)}

    dataloaders, vocab = create_lstm_dataloaders(config, dataset)

    output_root = config["paths"]["artifacts_root"]
    metrics = train_lstm(
        config=config,
        dataloaders=dataloaders,
        vocab_size=len(vocab.itos),
        pad_id=vocab.pad_id,
        num_classes=len(id2label),
        id2label=id2label,
        output_dir=f"{output_root}/lstm",
        train_labels=list(dataset["train"][config["dataset"]["label_column"]]),
    )
    logger.info("LSTM test metrics: %s", metrics)


if __name__ == "__main__":
    main()
