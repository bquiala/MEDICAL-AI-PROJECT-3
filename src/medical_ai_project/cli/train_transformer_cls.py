"""CLI to fine-tune and evaluate transformer sentence classifier."""

from __future__ import annotations

import argparse

from medical_ai_project.data.pubmed_rct20k import load_pubmed_rct20k
from medical_ai_project.training.trainer_transformer_cls import train_transformer_classification
from medical_ai_project.utils.config import load_config
from medical_ai_project.utils.logging_utils import setup_logger
from medical_ai_project.utils.reproducibility import set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train transformer sentence classifier on PubMed RCT 20k")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    return parser.parse_args()


def main() -> None:
    """Run transformer sentence classification flow."""
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])
    logger = setup_logger("train_transformer_cls")

    dataset = load_pubmed_rct20k(config)

    output_root = config["paths"]["artifacts_root"]
    metrics = train_transformer_classification(
        config=config,
        dataset=dataset,
        output_dir=f"{output_root}/classification_transformer",
    )
    logger.info("Transformer classification test metrics: %s", metrics)


if __name__ == "__main__":
    main()
