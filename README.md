# MEDICAL-AI-PROJECT-3

Biomedical Named Entity Recognition (NER) using PubMed RCT 20k text with a BiLSTM baseline and a pretrained transformer token classifier.

## Clinical Context

This project is intended for biomedical NLP learners and clinical research teams who need structured extraction of clinically relevant entities from medical text. The task identifies entities such as diseases, drugs, procedures, and anatomy mentions in token sequences, supporting evidence mining and downstream clinical NLP pipelines.

## Task and Dataset

- Task: Named Entity Recognition (BIO token tagging)
- Dataset: PubMed RCT 20k
- Source: https://huggingface.co/datasets/armanc/pubmed-rct20k
- Splits: Predefined train, validation, and test
- Note: This repository uses weak supervision to create biomedical NER tags from domain lexicons.

## Quick Start

### 1. Environment

- Python version: 3.10+ (tested with 3.11)
- Create and activate virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run quick pipeline (smoke test)

```bash
bash scripts/run_quick_pipeline.sh
```

Expected outcome:
- Creates run artifacts under `artifacts/`
- Trains both models on a subset
- Generates token/entity metrics and span predictions

### 4. Run full pipeline

```bash
bash scripts/run_full_pipeline.sh
```

Expected runtime and compute:
- CPU only quick run: 10-30 minutes
- Full transformer fine-tuning on GPU: typically 30-120 minutes depending on hardware
- Suggested minimum RAM: 8GB (16GB recommended)

## Usage Guide

### Train LSTM baseline

```bash
python -m medical_ai_project.cli.train_lstm --config configs/default_config.json
```

Output:
- `artifacts/lstm/checkpoints/`
- `artifacts/lstm/metrics/`
- `artifacts/lstm/predictions/`

### Train transformer model

```bash
python -m medical_ai_project.cli.train_transformer --config configs/default_config.json
```

Output:
- `artifacts/transformer/checkpoints/`
- `artifacts/transformer/metrics/`
- `artifacts/transformer/predictions/`

### Evaluate model predictions

```bash
python -m medical_ai_project.cli.evaluate \
	--predictions artifacts/lstm/predictions/test_predictions.csv \
	--output-dir artifacts/lstm/metrics
```

Output:
- Token accuracy
- Entity-level precision, recall, and F1
- Entity-level report by type

### Run qualitative error analysis

```bash
python -m medical_ai_project.cli.error_analysis \
	--predictions artifacts/transformer/predictions/test_predictions.csv \
	--output-dir artifacts/transformer/analysis
```

Output:
- Hard example slices
- Frequent failure modes summary

## Data Description

- Source: Hugging Face Datasets (armanc/pubmed-rct20k)
- Structure: sentence-level text converted into token-level weak BIO tags
- Fields used: text column transformed into tokens and BIO labels
- License and citation: follow the dataset card and original paper references on Hugging Face
- Data access: automatically downloaded via `datasets` package at runtime

If your environment is offline, pre-download and cache Hugging Face datasets before running.

## Results Summary

Fill this section after experiments.

| Model | Token Accuracy | Entity Precision | Entity Recall | Entity F1 |
|---|---:|---:|---:|---:|
| LSTM baseline | TBD | TBD | TBD | TBD |
| Transformer | TBD | TBD | TBD | TBD |

### Notes on model behavior

- Overfitting checks: compare train vs validation loss and entity F1.
- Label quality checks: weak supervision quality depends on lexicon coverage.
- Error patterns: inspect missed and spurious entities by type.

## Project Structure

```text
.
├── configs/
│   └── default_config.json
├── scripts/
│   ├── run_full_pipeline.sh
│   └── run_quick_pipeline.sh
├── src/
│   └── medical_ai_project/
│       ├── cli/                # Command-line entrypoints
│       ├── data/               # Data loading, preprocessing, tokenization
│       ├── models/             # LSTM and model components
│       ├── training/           # Training loops for each approach
│       ├── evaluation/         # Metrics and error analysis helpers
│       └── utils/              # Config, logging, reproducibility, I/O
├── artifacts/                  # Generated outputs (kept out of git)
├── reports/                    # Optional report assets
├── requirements.txt
└── README.md
```

## Authors and Contributions


| Team Member | Role | Tasks |
| Bryan Quiala Llera| Project lead | Coordination, Abstract, Introduction, & Literature Review |
| Jai Raccioppi | Baseline engineer | Establish framework, Methods, Results, & Discussion |

## Dependencies

All pinned dependencies are listed in `requirements.txt`.

## Reproducibility and Code Quality Checklist

- Random seed setup across Python, NumPy, and PyTorch
- Config-driven runs (no hard-coded data paths)
- Logging and error handling in CLI workflows
- PEP 8 oriented naming and module organization
- Docstrings and targeted inline comments for complex logic

## Repository Setup Notes

- Repository visibility must be set to Public in GitHub settings.
- Add the repository link on the report title page.
- Use meaningful commit messages for each logical stage.
