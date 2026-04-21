#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

CONFIG_PATH="configs/default_config.json"

python -m medical_ai_project.cli.train_lstm --config "$CONFIG_PATH"
python -m medical_ai_project.cli.train_transformer --config "$CONFIG_PATH"
python -m medical_ai_project.cli.error_analysis \
  --predictions artifacts/lstm/predictions/test_predictions.csv \
  --output-dir artifacts/lstm/analysis
python -m medical_ai_project.cli.error_analysis \
  --predictions artifacts/transformer/predictions/test_predictions.csv \
  --output-dir artifacts/transformer/analysis
