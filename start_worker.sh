#!/usr/bin/env bash
set -euo pipefail

cd /workspace/Chatterblez_FINITIO

source .venv-qwen/bin/activate

python3 backend/main.py "$@"
