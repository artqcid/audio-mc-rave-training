#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/mnt/c/Users/marku/Documents/GitHub/artqcid/ai-projects/audio-mc-rave-training"
cd "$PROJECT_DIR"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "WSL environment ready. Install torch separately with the correct CUDA wheel."
