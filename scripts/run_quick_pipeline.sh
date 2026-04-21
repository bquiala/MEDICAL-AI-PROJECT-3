#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

CONFIG_PATH="configs/default_config.json"
TMP_CONFIG="configs/.quick_config.json"

python - <<'PY'
import json
from pathlib import Path

cfg_path = Path("configs/default_config.json")
quick_path = Path("configs/.quick_config.json")
cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

cfg["dataset"]["max_train_samples"] = 5000
cfg["dataset"]["max_validation_samples"] = 1000
cfg["dataset"]["max_test_samples"] = 1000
cfg["lstm"]["num_epochs"] = 1
cfg["transformer"]["num_epochs"] = 1

quick_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
PY

python -m medical_ai_project.cli.train_lstm --config "$TMP_CONFIG"
python -m medical_ai_project.cli.train_transformer --config "$TMP_CONFIG"
python -m medical_ai_project.cli.error_analysis \
  --predictions artifacts/transformer/predictions/test_predictions.csv \
  --output-dir artifacts/transformer/analysis

rm -f "$TMP_CONFIG"
