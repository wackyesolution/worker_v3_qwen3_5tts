#!/usr/bin/env bash
set -euo pipefail

cd /workspace/Chatterblez_FINITIO

source .venv-azzurra/bin/activate

if [[ -f .azzurra-env ]]; then
  source .azzurra-env
fi

python3 backend/main.py "$@"
