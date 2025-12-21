#!/usr/bin/env python3

import argparse
import re
import sys
from pathlib import Path

import spacy
from PyPDF2 import PdfReader


def normalize_quotes(text):
    return (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )


NON_ALLOWED_RE = re.compile(r"[^0-9A-Za-z\u00C0-\u017F\s.,'\"?!:;%\u20AC$-]+")
SPACE_RE = re.compile(r"\s+")
SPACE_BEFORE_PERIOD_RE = re.compile(r"\s+\.")
MULTIPLE_PERIODS_RE = re.compile(r"\.{2,}")


def clean_line(line):
    line = normalize_quotes(line)
    line = NON_ALLOWED_RE.sub(" ", line)
    line = SPACE_BEFORE_PERIOD_RE.sub(".", line)
    line = MULTIPLE_PERIODS_RE.sub(".", line)
    line = SPACE_RE.sub(" ", line)
    return line.strip()


def merge_hyphenated_lines(lines):
    """Merge words split by end-of-line hyphens (PDF/EPUB line wraps)."""
    merged = []
    i = 0
    while i < len(lines):
        line = lines[i]
        while line.endswith("-") and i + 1 < len(lines):
            next_line = lines[i + 1].lstrip()
            if next_line and re.match(r"[0-9A-Za-z\u00C0-\u017F]", next_line):
                line = line[:-1] + next_line
                i += 1
            else:
                break
        merged.append(line)
        i += 1
    return merged


def get_nlp():
    nlp = spacy.blank("xx")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


def batch_sentences_intelligently(sentences, min_chars=150, max_chars=800):
    batches = []
    current_batch = []
    current_length = 0
    for sent in sentences:
        sent_text = sent.text.strip()
        sent_length = len(sent_text)
        if not sent_text or sent_length < 2:
            continue
        if sent_length > max_chars:
            if current_batch:
                batches.append(" ".join(current_batch))
                current_batch = []
                current_length = 0
            batches.append(sent_text)
            continue
        if current_length > 0 and (current_length + sent_length + 1) > max_chars:
            batches.append(" ".join(current_batch))
            current_batch = [sent_text]
            current_length = sent_length
        else:
            current_batch.append(sent_text)
            current_length += sent_length + (1 if current_batch else 0)
        if current_length >= min_chars and sent_text.endswith((".", "!", "?", '"', "'")):
            batches.append(" ".join(current_batch))
            current_batch = []
            current_length = 0
    if current_batch:
        batches.append(" ".join(current_batch))
    return batches


def extract_page_text(pdf_path, page_number):
    reader = PdfReader(str(pdf_path))
    if page_number < 1 or page_number > len(reader.pages):
        raise ValueError(
            "Page {} out of range (1-{}).".format(page_number, len(reader.pages))
        )
    page = reader.pages[page_number - 1]
    try:
        page_text = page.extract_text() or ""
    except Exception as exc:  # pragma: no cover - diagnostic helper
        raise RuntimeError("Failed to extract page text: {}".format(exc))
    return page_text.strip()


def main():
    if sys.version_info < (3, 8):
        print("This script requires Python 3.8+. Run with python3.")
        return 1
    parser = argparse.ArgumentParser(
        description="Inspect PDF page text as it would be fed into TTS."
    )
    parser.add_argument(
        "pdf",
        nargs="?",
        default="PUT_PDF_PATH_HERE",
        help="Path to the PDF to inspect.",
    )
    parser.add_argument(
        "--page",
        type=int,
        default=2,
        help="1-based page number to extract (default: 2).",
    )
    args = parser.parse_args()
    pdf_path = Path(args.pdf).expanduser()
    if not pdf_path.exists():
        print("PDF not found: {}".format(pdf_path))
        print("Pass a valid path or edit the default in this script.")
        return 1

    raw_text = extract_page_text(pdf_path, args.page)
    lines = raw_text.splitlines()
    cleaned_lines = []
    for line in lines:
        cleaned_line = clean_line(line)
        if cleaned_line.strip() and re.search(r"\w", cleaned_line):
            cleaned_lines.append(cleaned_line)
    cleaned_lines = merge_hyphenated_lines(cleaned_lines)
    cleaned_text = " ".join(cleaned_lines)

    nlp = get_nlp()
    doc = nlp(cleaned_text)
    batches = batch_sentences_intelligently(list(doc.sents))

    print("RAW PAGE TEXT:")
    print(raw_text)
    print("\nCLEANED TEXT (input to spaCy):")
    print(cleaned_text)
    print("\nBATCHES (input to TTS):")
    for idx, batch in enumerate(batches, start=1):
        print("{}. {}".format(idx, batch))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
