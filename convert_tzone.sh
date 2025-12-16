#!/usr/bin/env bash
set -euo pipefail

python cli.py \
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
