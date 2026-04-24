# MEDICAL-AI-PROJECT-3

Biomedical NLP project on PubMed RCT 20k with two tracks:
1) weakly supervised Named Entity Recognition (NER), and
2) sentence-level rhetorical role classification.

## Project Overview

This repository trains and evaluates baseline (BiLSTM) and transformer models to structure biomedical abstract text at two levels:

- **Token-level**: extract weakly-supervised biomedical entities from text using BIO tagging.
- **Sentence-level**: classify each sentence into a rhetorical role (e.g., Methods, Results).

## Problem Statement

Clinical and biomedical research workflows often need unstructured abstract text transformed into structured signals for downstream analysis (e.g., extracting key entities and separating Methods from Results). This project demonstrates an end-to-end, reproducible pipeline for:

- building weak NER labels via a simple lexicon matching heuristic,
- training/evaluating NER models on those weak labels, and
- training/evaluating sentence classifiers on the dataset’s provided labels.

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

Note: the shell scripts set `PYTHONPATH` to include `src/` so `python -m medical_ai_project...` works from the repository root. If you run commands manually and see `ModuleNotFoundError`, run:

```bash
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
```

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

Experiments completed on PubMed RCT 20k dataset with both quick (subset) and full pipeline runs.

### NER (Weak Supervision)

| Model | Token Accuracy | Entity Precision | Entity Recall | Entity F1 |
|---|---:|---:|---:|---:|
| LSTM baseline | 0.9895 | 0.3912 | 0.9389 | 0.5523 |
| Transformer (DistilBERT) | 0.9982 | 0.8545 | 0.7833 | 0.8174 |

**Key observations:**
- Transformer substantially outperforms LSTM on entity-level metrics (F1: 0.8174 vs 0.5523)
- LSTM exhibits high recall but lower precision, indicating over-prediction of entities
- Transformer achieves much better balance between precision (0.85) and recall (0.78)

### Sentence Classification (Gold Labels)

| Model | Accuracy | Macro F1 | Weighted F1 | Accuracy CI95 |
|---|---:|---:|---:|---|
| LSTM baseline | 0.615 | 0.5312 | 0.6077 | [0.582, 0.645] |
| Transformer (DistilBERT) | 0.796 | 0.7106 | 0.7915 | [0.770, 0.821] |

**Per-class performance (Transformer):**
| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Background | 0.620 | 0.458 | 0.527 | 107 |
| Conclusions | 0.690 | 0.779 | 0.732 | 149 |
| Methods | 0.822 | 0.924 | 0.870 | 304 |
| Objective | 0.586 | 0.531 | 0.557 | 66 |
| Results | 0.895 | 0.840 | 0.867 | 376 |

**Key observations:**
- Transformer outperforms LSTM on accuracy (0.796 vs 0.615) and F1 (0.7106 vs 0.5312)
- Methods section is best-recognized (F1: 0.870) due to distinctive linguistic patterns
- Background section is most challenging (F1: 0.527) due to ambiguous language
- Results section has high precision (0.895) but moderate recall (0.840)

### Entity-Level NER Performance by Type

| Entity | LSTM F1 | Transformer F1 |
|---|---:|---:|
| ANATOMY | 0.6509 | 0.9636 |
| DISEASE | 0.8844 | 0.8344 |
| DRUG | 0.1088 | 0.0000 |
| PROCEDURE | 0.5504 | 0.7463 |

**Insights:**
- Anatomy entities: Transformer dramatically outperforms (F1: 0.96 vs 0.65)
- Disease entities: Both models perform well; LSTM slightly higher
- Drug entities: Both models struggle due to weak lexicon coverage
- Procedure entities: Transformer shows improvement (F1: 0.75 vs 0.55)

### Model Behavior Notes

- **Weak supervision quality:** Performance limited by lexicon-based weak labels on drug entities
- **Overfitting:** No significant gap between validation and test metrics for Transformer
- **Best practices:** Transformer-based models are recommended for this task with a 21% absolute improvement on NER F1 and 18% on classification accuracy
- **Error patterns:** NER errors dominated by missed entities in LSTM; Classification confusion mainly between semantically similar sections (Objective/Methods)

## Project Structure

```text
.
├── configs/
│   └── default_config.json
├── scripts/
│   ├── run_full_pipeline.sh
│   └── run_quick_pipeline.sh
├── notebooks/                  # EDA + demo notebooks
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

## Notebooks

- **EDA**: `notebooks/01_eda_pubmed_rct20k.ipynb`
	- Dataset overview, missingness, label/entity distributions, text length summaries, and visualizations.
	- Includes interpretation sections that motivate preprocessing choices (e.g., max sequence length) and model selection.

- **Demo**: `notebooks/02_demo_inference_and_evaluation.ipynb`
	- Loads a trained model checkpoint (if present), runs inference on sample sentences, and visualizes performance from saved artifacts.

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
