#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT}"

# Load .env if present
if [ -f "$PROJECT_ROOT/.env" ]; then
  set -a; source "$PROJECT_ROOT/.env"; set +a
fi

notify() {
  python "$PROJECT_ROOT/training/notify.py" "$1" 2>/dev/null || true
}

# --- checks ---
echo "[launch] Checking GPU..."
python -c "import torch; assert torch.cuda.is_available(), 'No CUDA GPU found'; print('GPU:', torch.cuda.get_device_name(0))"
GPU=$(python -c "import torch; print(torch.cuda.get_device_name(0))")

echo "[launch] Checking data paths..."
[ -d "$DATA_ROOT/data/processed/interactions" ] || { echo "ERROR: missing $DATA_ROOT/data/processed/interactions"; exit 1; }
[ -f "$DATA_ROOT/data/raw/labels/train_labels.pkl" ] || { echo "ERROR: missing train_labels.pkl"; exit 1; }

# --- run ---
notify "ATLAS training started on $GPU"
echo "[launch] Starting training with args: $*"

cd "$PROJECT_ROOT/training"
if python train.py "$@"; then
  notify "ATLAS training completed successfully"
else
  notify "ATLAS training FAILED — check logs"
  exit 1
fi
