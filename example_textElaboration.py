#!/usr/bin/env python3
from __future__ import annotations

import sys

from core import batch_sentences_intelligently, clean_line, get_nlp

SAMPLE_TEXT = (
    "Attivita\u0300 e attivit\u00e0: confronto. Prezzo 18% e 10\u20ac o 10$.\n"
    "Domanda? Risposta! Nota: valori 3-4; esempio \"test\".\n"
    "Altro punto. Fine."
)


def main() -> int:
    if len(sys.argv) > 1:
        raw_text = " ".join(sys.argv[1:])
    else:
        raw_text = SAMPLE_TEXT

    lines = raw_text.splitlines()
    cleaned_text = "\n".join(
        cleaned_line
        for line in lines
        if (cleaned_line := clean_line(line)).strip()
    )

    nlp = get_nlp()
    doc = nlp(cleaned_text)
    sentences = list(doc.sents)
    batches = batch_sentences_intelligently(sentences, min_chars=150, max_chars=800)

    print("RAW:")
    print(raw_text)
    print("\nCLEANED (input to spaCy):")
    print(cleaned_text)
    print("\nBATCHES (input to TTS):")
    for i, batch in enumerate(batches, start=1):
        print(f"{i}. {batch}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
