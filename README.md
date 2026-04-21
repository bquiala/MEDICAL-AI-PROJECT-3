# MEDICAL-AI-PROJECT-3

Clinical Trial Abstract Sentence Classification using PubMed RCT 20k with a required LSTM baseline and a pretrained transformer for fair comparison.

## Clinical Context

This project is intended for biomedical NLP learners and clinical research teams who need structured understanding of clinical trial abstracts. The task predicts rhetorical roles for each sentence (Background, Objective, Methods, Results, Conclusions), which can support evidence summarization, literature triage, and downstream decision-support pipelines.

## Task and Dataset

- Task: Multi-class sentence classification (5 labels)
- Dataset: PubMed RCT 20k
- Source: https://huggingface.co/datasets/armanc/pubmed-rct20k
- Splits: Predefined train, validation, and test
- Approximate size:
	- Train: ~180k sentences
	- Validation: ~30k sentences
	- Test: ~30k sentences

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
- Generates evaluation files and a confusion matrix

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
- Accuracy, Precision, Recall, F1
- Classification report
- Confusion matrix image

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
- Structure: sentence-level text with one rhetorical label per sentence
- Fields expected: text and label columns
- License and citation: follow the dataset card and original paper references on Hugging Face
- Data access: automatically downloaded via `datasets` package at runtime

If your environment is offline, pre-download and cache Hugging Face datasets before running.

## Results Summary

Fill this section after experiments.

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---:|---:|---:|---:|
| LSTM baseline | TBD | TBD | TBD | TBD |
| Transformer | TBD | TBD | TBD | TBD |

### Notes on model behavior

- Overfitting checks: compare train vs validation loss curves.
- Class imbalance checks: inspect per-class support and macro vs weighted metrics.
- Error patterns: analyze ambiguous and long sentences, negation, and rare terms.

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

Replace placeholders before submission.

| Team Member | Role |
|---|---|
| Name 1 | Project lead, experiment design, report writing |
| Name 2 | LSTM baseline implementation and tuning |
| Name 3 | Transformer fine-tuning and evaluation |
| Name 4 | Error analysis, clinical interpretation, presentation |

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
