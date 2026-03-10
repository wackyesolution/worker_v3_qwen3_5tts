#!/usr/bin/env bash
set -euo pipefail

cd /workspace/Chatterblez_FINITIO

source .venv-qwen/bin/activate

# Keep imports stable even if the orchestrator forgets to pass PYTHONPATH.
export PYTHONPATH="${PYTHONPATH:-/workspace/Chatterblez_FINITIO}"

python3 backend/main.py "$@"
