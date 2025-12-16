#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_PATH="${VENV_PATH:-$SCRIPT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BOOK_DIR="$SCRIPT_DIR/DD_book"
OUTPUT_BASE="$SCRIPT_DIR/DD_Output"

if [[ -f "$VENV_PATH/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_PATH/bin/activate"
else
  echo "Virtualenv not found at $VENV_PATH. Using system Python ($PYTHON_BIN)." >&2
fi

# Collect available books and prompt the user to pick one for conversion.
if [[ ! -d "$BOOK_DIR" ]]; then
  echo "Book directory not found: $BOOK_DIR" >&2
  exit 1
fi

books=()
while IFS= read -r -d '' file; do
  books+=("$file")
done < <(find "$BOOK_DIR" -type f -mindepth 1 -maxdepth 1 -print0)

if ((${#books[@]} == 0)); then
  echo "No files found in $BOOK_DIR" >&2
  exit 1
fi

echo "Libri disponibili:"
for i in "${!books[@]}"; do
  printf '  %d) %s\n' $((i + 1)) "$(basename "${books[i]}")"
done

selection=""
while true; do
  read -rp "Seleziona il numero del libro da convertire (1-${#books[@]}): " selection
  if [[ "$selection" =~ ^[0-9]+$ ]] && ((selection >= 1 && selection <= ${#books[@]})); then
    break
  fi
  echo "Scelta non valida, riprova."
done

BOOK_FILE="${books[selection-1]}"
BOOK_NAME="$(basename "$BOOK_FILE")"
BOOK_BASENAME="${BOOK_NAME%.*}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$OUTPUT_BASE/${BOOK_BASENAME}_${TIMESTAMP}"

mkdir -p "$OUTPUT_DIR"
echo "Conversione di '$BOOK_NAME' in $OUTPUT_DIR"

"$PYTHON_BIN" cli.py \
  --file "$BOOK_FILE" \
  --output "$OUTPUT_DIR" \
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
