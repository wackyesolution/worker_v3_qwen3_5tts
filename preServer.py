#!/usr/bin/env python3
"""
Pre-processing server script that mirrors convert_tzone.sh settings,
collects finished audiobooks, and runs Whisper to create SRT/VTT subtitles.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
BOOK_DIR = SCRIPT_DIR / "DD_book"
OUTPUT_BASE = SCRIPT_DIR / "DD_Output"
COLLECTION_DIR = SCRIPT_DIR / "audioBook"

# Settings mirrored from convert_tzone.sh
CLI_ARGUMENTS = (
    "--speed", "0.88",
    "--repetition-penalty", "1.05",
    "--min-p", "0.02",
    "--top-p", "0.92",
    "--exaggeration", "0.72",
    "--cfg-weight", "0.32",
    "--temperature", "0.92",
    "--sentence-gap-ms", "350",
    "--question-gap-ms", "1000",
    "--use-multilingual",
    "--language-id", "it",
    "--disable-alignment-guard",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a single e-book and transcribe the final WAV with Whisper."
    )
    parser.add_argument(
        "--book",
        required=True,
        help=(
            "Path to the EPUB/PDF to convert. "
            "Can be an absolute path or relative to the DD_book directory."
        ),
    )
    parser.add_argument(
        "--output-base",
        default=str(OUTPUT_BASE),
        help="Base folder for the intermediate Chatterblez outputs.",
    )
    parser.add_argument(
        "--collect-dir",
        default=str(COLLECTION_DIR),
        help="Destination folder where the final WAV/SRT/VTT are stored.",
    )
    parser.add_argument(
        "--whisper-model",
        default="large-v3",
        help="Name of the Whisper model to load (default: large-v3).",
    )
    parser.add_argument(
        "--whisper-language",
        default="it",
        help="Language hint passed to Whisper for faster decoding.",
    )
    parser.add_argument(
        "--skip-transcription",
        action="store_true",
        help="Only convert the audiobook, skip Whisper transcription.",
    )
    parser.add_argument(
        "--run-id",
        help="Optional identifier (timestamp) to reuse for output folders/files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def resolve_book_path(book_argument: str) -> Path:
    candidate = Path(book_argument).expanduser()
    if candidate.is_file():
        return candidate
    fallback = (BOOK_DIR / book_argument).resolve()
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(
        f"Book not found at '{book_argument}'. "
        f"Tried absolute path and '{BOOK_DIR / book_argument}'."
    )


def ensure_directories(*directories: Path) -> None:
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def build_cli_command(book_path: Path, output_dir: Path) -> List[str]:
    return [
        sys.executable,
        str(SCRIPT_DIR / "cli.py"),
        "--file",
        str(book_path),
        "--output",
        str(output_dir),
        *CLI_ARGUMENTS,
    ]


def run_cli(command: Sequence[str]) -> None:
    logging.info("Running CLI command: %s", " ".join(command))
    subprocess.run(command, check=True)


def find_generated_file(folder: Path, pattern: str, description: str) -> Path:
    matches = sorted(folder.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Unable to find {description} in {folder} (pattern: {pattern}).")
    if len(matches) > 1:
        logging.warning("Multiple %s found. Using %s", description, matches[0])
    return matches[0]


def convert_to_wav(source_file: Path, destination_file: Path) -> None:
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-i",
        str(source_file),
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        "24000",
        str(destination_file),
    ]
    logging.info("Converting %s to WAV via ffmpeg.", source_file.name)
    subprocess.run(ffmpeg_cmd, check=True)


def load_whisper_model(model_name: str):
    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("Torch is required to run Whisper. Please install torch first.") from exc

    try:
        import whisper  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "The openai-whisper package is required. Install it with 'pip install openai-whisper'."
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("Loading Whisper model '%s' on %s.", model_name, device)
    return whisper.load_model(model_name, device=device)


def format_timestamp(seconds: float, delimiter: str) -> str:
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(milliseconds, 3600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}{delimiter}{millis:03}"


def write_srt(segments, destination: Path) -> None:
    blocks = []
    for idx, segment in enumerate(segments, start=1):
        text = segment.get("text", "").strip().replace("\n", " ")
        start = format_timestamp(segment.get("start", 0.0), ",")
        end = format_timestamp(segment.get("end", 0.0), ",")
        blocks.append(f"{idx}\n{start} --> {end}\n{text}")
    destination.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def write_vtt(segments, destination: Path) -> None:
    lines = ["WEBVTT", ""]
    for segment in segments:
        text = segment.get("text", "").strip().replace("\n", " ")
        start = format_timestamp(segment.get("start", 0.0), ".")
        end = format_timestamp(segment.get("end", 0.0), ".")
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    destination.write_text("\n".join(lines), encoding="utf-8")


def run_whisper(audio_path: Path, model_name: str, language: str, srt_path: Path, vtt_path: Path) -> None:
    model = load_whisper_model(model_name)
    logging.info("Transcribing %s with Whisper...", audio_path.name)
    language_hint = language or None
    result = model.transcribe(str(audio_path), language=language_hint, verbose=False)
    segments = result.get("segments", [])
    if not segments:
        raise RuntimeError("Whisper did not return any segments.")
    write_srt(segments, srt_path)
    write_vtt(segments, vtt_path)
    logging.info("Transcription complete. Files saved to %s and %s.", srt_path.name, vtt_path.name)


def process_book(
    book: str | Path,
    output_base: Path,
    collect_dir: Path,
    whisper_model: str,
    whisper_language: str,
    skip_transcription: bool,
    run_id: str | None = None,
) -> dict[str, Path | str | None]:
    book_path = resolve_book_path(str(book))
    book_basename = book_path.stem
    run_token = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    ensure_directories(output_base, collect_dir)

    output_dir = output_base / f"{book_basename}_{run_token}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cli_command = build_cli_command(book_path, output_dir)
    run_cli(cli_command)

    generated_m4b = find_generated_file(output_dir, "*.m4b", "final M4B file")

    final_folder = collect_dir / book_basename
    final_folder.mkdir(parents=True, exist_ok=True)
    wav_target = final_folder / f"{book_basename}_{run_token}.wav"
    convert_to_wav(generated_m4b, wav_target)

    srt_path = None
    vtt_path = None
    if not skip_transcription:
        srt_path = final_folder / f"{book_basename}_{run_token}.srt"
        vtt_path = final_folder / f"{book_basename}_{run_token}.vtt"
        run_whisper(wav_target, whisper_model, whisper_language, srt_path, vtt_path)
    else:
        logging.info("Transcription skipped per flag.")

    logging.info("All assets saved under %s", final_folder)
    return {
        "folder": final_folder,
        "wav": wav_target,
        "srt": srt_path,
        "vtt": vtt_path,
        "run_id": run_token,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    outputs = process_book(
        book=args.book,
        output_base=Path(args.output_base),
        collect_dir=Path(args.collect_dir),
        whisper_model=args.whisper_model,
        whisper_language=args.whisper_language,
        skip_transcription=args.skip_transcription,
        run_id=args.run_id,
    )
    logging.info("Run %s finished. Assets: %s", outputs["run_id"], outputs["folder"])


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI guard
        logging.error("preServer.py failed: %s", exc)
        raise
