#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_PATH="${VENV_PATH:-$SCRIPT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -f "$VENV_PATH/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_PATH/bin/activate"
else
  echo "Virtualenv not found at $VENV_PATH. Using system Python ($PYTHON_BIN)." >&2
fi

"$PYTHON_BIN" cli.py \
  --file DD_book/TZone.pdf \
  --output DD_Output \
  --speed 0.88 \
  --repetition-penalty 1.05 \
  --min-p 0.02 \
  --top-p 0.92 \
  --exaggeration 0.72 \
  --cfg-weight 0.32 \
  --temperature 0.92 \
  --sentence-gap-ms 350 \
  --question-gap-ms 1000 \
  --use-multilingual \
  --language-id it \
  --disable-alignment-guard
