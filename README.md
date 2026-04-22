# MEDICAL-AI-PROJECT-3

Biomedical NLP project on PubMed RCT 20k with two tracks:
1) weakly supervised Named Entity Recognition (NER), and
2) sentence-level rhetorical role classification.

## Clinical Context

This project is intended for biomedical NLP learners and clinical research teams who need both sentence-level evidence structuring and token-level entity extraction from biomedical text.

## Tasks and Dataset

- Dataset: PubMed RCT 20k
- Source: https://huggingface.co/datasets/armanc/pubmed-rct20k
- Splits: Predefined train, validation, and test
- Task A (NER): weakly supervised BIO tagging for DISEASE, DRUG, PROCEDURE, ANATOMY
- Task B (Classification): sentence role classification (Background/Objective/Methods/Results/Conclusions)

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
- Trains all quick-run models on a subset
- Generates NER metrics, classification metrics, predictions, and report figures

### 4. Run full pipeline

```bash
bash scripts/run_full_pipeline.sh
```

Expected runtime and compute:
- CPU only quick run: 10-30 minutes
- Full transformer fine-tuning on GPU: typically 30-120 minutes depending on hardware
- Suggested minimum RAM: 8GB (16GB recommended)

## Usage Guide

### NER: Train LSTM baseline

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

### Classification: Train LSTM baseline

```bash
python -m medical_ai_project.cli.train_lstm_cls --config configs/default_config.json
```

Output:
- `artifacts/classification_lstm/checkpoints/`
- `artifacts/classification_lstm/metrics/`
- `artifacts/classification_lstm/predictions/`

### Classification: Train transformer model

```bash
python -m medical_ai_project.cli.train_transformer_cls --config configs/default_config.json
```

Output:
- `artifacts/classification_transformer/checkpoints/`
- `artifacts/classification_transformer/metrics/`
- `artifacts/classification_transformer/predictions/`

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

### Generate report figures from artifacts

```bash
python -m medical_ai_project.cli.make_report_figures \
	--artifacts-root artifacts \
	--output-dir reports/figures
```

Output:
- `reports/figures/figure_1_model_comparison.png`
- `reports/figures/figure_2_ner_entity_f1.png`
- `reports/figures/figure_3_transformer_confusion_matrix.png`
- `reports/figures/figure_4_confidence_vs_correctness.png`
- summary CSVs used in the report

## Data Description

- Source: Hugging Face Datasets (armanc/pubmed-rct20k)
- Text field: `text`
- Classification label field: `label`
- NER labels: generated weak BIO tags from lexicon matching
- License and citation: follow the dataset card and original paper references on Hugging Face
- Data access: automatically downloaded via `datasets` package at runtime

If your environment is offline, pre-download and cache Hugging Face datasets before running.

## Results Summary

Fill this section after experiments.

### NER (weak supervision)

| Model | Token Accuracy | Entity Precision | Entity Recall | Entity F1 |
|---|---:|---:|---:|---:|
| LSTM baseline | TBD | TBD | TBD | TBD |
| Transformer | TBD | TBD | TBD | TBD |

### Sentence Classification (gold labels)

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| LSTM baseline | TBD | TBD | TBD |
| Transformer | TBD | TBD | TBD |

### Notes on model behavior

- Overfitting checks: compare train vs validation metrics in both tasks.
- Label quality checks: weak supervision quality depends on lexicon coverage.
- Error patterns: inspect NER missed/spurious entities and classification confusion matrix.

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
