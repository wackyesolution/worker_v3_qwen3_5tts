#!/usr/bin/env python3
"""Minimal CLI to synthesize a short sentence using the configured TTS engine."""

from __future__ import annotations

import argparse
from pathlib import Path

from core import (
    clean_line,
    gen_audio_segments,
    get_nlp,
    load_tts_resources,
    write_audio_stream,
)


def build_text(raw: str) -> str:
    parts = []
    for line in raw.splitlines():
        cleaned = clean_line(line)
        if cleaned:
            parts.append(cleaned)
    text = " ".join(parts).strip()
    return text or "Ciao!"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a quick audio sample via the configured TTS engine.")
    parser.add_argument("--text", required=True, help="Short sentence to synthesize.")
    parser.add_argument("--output", default="sample.wav", help="Path of the WAV file to create.")
    parser.add_argument("--temperature", type=float, default=0.85, help="Temperature for generation (default: 0.85).")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p for generation (default: 0.95).")
    parser.add_argument("--top-k", type=int, help="Top-k for generation (Azzurra only).")
    args = parser.parse_args()

    text = build_text(args.text)
    output_path = Path(args.output).expanduser().resolve()

    tts_resources = load_tts_resources(use_multilingual=False, cache=False)
    nlp = get_nlp()
    chunks = gen_audio_segments(
        tts_resources,
        nlp,
        text,
        speed=1.0,
        stats=None,
        max_sentences=None,
        post_event=None,
        should_stop=None,
        repetition_penalty=1.1,
        min_p=0.02,
        top_p=args.top_p,
        top_k=args.top_k,
        exaggeration=0.4,
        cfg_weight=0.8,
        temperature=args.temperature,
        use_multilingual=False,
        language_id="it",
        audio_prompt_wav=None,
        sentence_gap_ms=0,
        question_gap_ms=0,
    )
    frames = write_audio_stream(output_path, chunks)
    if frames <= 0:
        raise RuntimeError("Sintesi non riuscita: nessun frame audio.")
    print(f"Audio salvato in {output_path}")


if __name__ == "__main__":
    main()
