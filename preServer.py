#!/usr/bin/env python3
"""
Pre-processing server script that mirrors convert_tzone.sh settings,
collects finished audiobooks, and runs Whisper to create SRT/VTT subtitles.
"""

from __future__ import annotations

import argparse
import logging
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Union

SCRIPT_DIR = Path(__file__).resolve().parent
BOOK_DIR = SCRIPT_DIR / "DD_book"
OUTPUT_BASE = SCRIPT_DIR / "DD_Output"
COLLECTION_DIR = SCRIPT_DIR / "audioBook"
CHAPTER_MANIFEST = "chapter_exports.json"

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
        default="small",
        help="Name of the Whisper model to load (default: small).",
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
        "--filterlist",
        help="Comma-separated chapter names to ignore during conversion.",
    )
    parser.add_argument(
        "--chapter-indices",
        help="Comma-separated indexes of chapters to include during synthesis.",
    )
    parser.add_argument(
        "--per-chapter-export",
        action="store_true",
        help="Generate standalone files per chapter instead of a single audiobook.",
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


def build_cli_command(
    book_path: Path,
    output_dir: Path,
    filterlist: str | None = None,
    chapter_indices: str | None = None,
    per_chapter_export: bool = False,
) -> List[str]:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "cli.py"),
        "--file",
        str(book_path),
        "--output",
        str(output_dir),
        *CLI_ARGUMENTS,
    ]
    if filterlist:
        cmd += ["--filterlist", filterlist]
    if chapter_indices:
        cmd += ["--chapter-indices", chapter_indices]
    if per_chapter_export:
        cmd.append("--per-chapter-export")
    return cmd


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


def convert_to_m4a(source_file: Path, destination_file: Path, bitrate: str = "96k") -> None:
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-i",
        str(source_file),
        "-c:a",
        "aac",
        "-b:a",
        bitrate,
        str(destination_file),
    ]
    logging.info("Converting %s to M4A (%s).", source_file.name, destination_file.name)
    subprocess.run(ffmpeg_cmd, check=True)


def concat_audio_files(audio_paths: List[Path], destination_file: Path) -> None:
    if not audio_paths:
        raise RuntimeError("Nessun file audio per la concatenazione.")
    temp_list = destination_file.with_suffix(destination_file.suffix + ".txt")
    with temp_list.open("w", encoding="utf-8") as handle:
        for path in audio_paths:
            safe_path = str(path).replace("'", "'\\''")
            handle.write(f"file '{safe_path}'\n")
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(temp_list),
        "-c:a",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        "24000",
        str(destination_file),
    ]
    logging.info("Concatenating %d capitoli in %s", len(audio_paths), destination_file.name)
    try:
        subprocess.run(ffmpeg_cmd, check=True)
    finally:
        temp_list.unlink(missing_ok=True)


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


def transcribe_segments(model, audio_path: Path, language: str):
    language_hint = language or None
    result = model.transcribe(str(audio_path), language=language_hint, verbose=False)
    segments = result.get("segments", [])
    if not segments:
        raise RuntimeError(f"Whisper non ha prodotto segmenti per {audio_path}.")
    simplified = []
    for segment in segments:
        simplified.append(
            {
                "text": segment.get("text", "").strip(),
                "start": float(segment.get("start", 0.0)),
                "end": float(segment.get("end", 0.0)),
            }
        )
    return simplified


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
    segments = transcribe_segments(model, audio_path, language)
    write_srt(segments, srt_path)
    write_vtt(segments, vtt_path)
    logging.info("Transcription complete. Files saved to %s and %s.", srt_path.name, vtt_path.name)


def probe_duration_seconds(audio_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
        return float(output)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Impossibile ottenere la durata di {audio_path}: {exc.output}") from exc
    except ValueError as exc:
        raise RuntimeError(f"Durata non valida per {audio_path}: {output}") from exc


def process_book(
    book: str | Path,
    output_base: Path,
    collect_dir: Path,
    whisper_model: str,
    whisper_language: str,
    skip_transcription: bool,
    run_id: str | None = None,
    filterlist: str | None = None,
    chapter_indices: str | None = None,
    per_chapter_export: bool = False,
) -> Dict[str, Union[Path, str, None]]:
    book_path = resolve_book_path(str(book))
    book_basename = book_path.stem
    run_token = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    ensure_directories(output_base, collect_dir)

    output_dir = output_base / f"{book_basename}_{run_token}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cli_command = build_cli_command(
        book_path,
        output_dir,
        filterlist,
        chapter_indices,
        per_chapter_export=per_chapter_export,
    )
    run_cli(cli_command)

    final_folder = collect_dir / book_basename
    final_folder.mkdir(parents=True, exist_ok=True)
    if per_chapter_export:
        manifest_path = output_dir / CHAPTER_MANIFEST
        if not manifest_path.exists():
            raise FileNotFoundError(f"Chapter manifest not found at {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest.get("chapters") or []
        if not entries:
            raise RuntimeError("Chapter manifest is empty; no per-chapter exports detected.")
        saved_chapters: List[Dict[str, Union[Path, int, str, None]]] = []
        chapter_audio_paths: List[Path] = []
        combined_segments: List[Dict[str, Union[str, float]]] = []
        offset_seconds = 0.0
        whisper_model_instance = None if skip_transcription else load_whisper_model(whisper_model)
        for entry in sorted(entries, key=lambda item: item.get("sequence", 0)):
            source = Path(entry.get("audio_path", "")).expanduser()
            if not source.exists():
                raise FileNotFoundError(f"Per-chapter audio not found: {source}")
            sequence = int(entry.get("sequence") or (len(saved_chapters) + 1))
            dest_name = f"{book_basename}_{run_token}_chapter{sequence:03d}{source.suffix}"
            dest_path = final_folder / dest_name
            shutil.copy2(source, dest_path)
            chapter_audio_paths.append(dest_path)
            if whisper_model_instance:
                segments = transcribe_segments(whisper_model_instance, dest_path, whisper_language)
                for segment in segments:
                    combined_segments.append(
                        {
                            "text": segment.get("text", ""),
                            "start": segment.get("start", 0.0) + offset_seconds,
                            "end": segment.get("end", 0.0) + offset_seconds,
                        }
                    )
            offset_seconds += probe_duration_seconds(dest_path)
            saved_chapters.append(
                {
                    "sequence": sequence,
                    "chapter_index": entry.get("chapter_index"),
                    "chapter_name": entry.get("chapter_name"),
                    "audio": dest_path,
                    "srt": None,
                    "vtt": None,
                }
            )
        merged_wav = final_folder / f"{book_basename}_{run_token}.wav"
        concat_audio_files(chapter_audio_paths, merged_wav)
        merged_m4a = final_folder / f"{book_basename}_{run_token}.m4a"
        convert_to_m4a(merged_wav, merged_m4a)
        srt_path = None
        vtt_path = None
        if not skip_transcription:
            if not combined_segments:
                raise RuntimeError("Nessun segmento Whisper raccolto nonostante la trascrizione sia attiva.")
            srt_path = final_folder / f"{book_basename}_{run_token}.srt"
            vtt_path = final_folder / f"{book_basename}_{run_token}.vtt"
            write_srt(combined_segments, srt_path)
            write_vtt(combined_segments, vtt_path)
        else:
            logging.info("Transcription skipped per flag.")
        logging.info(
            "Exported %d per-chapter audio files into %s e generato audio unico %s",
            len(saved_chapters),
            final_folder,
            merged_m4a.name,
        )
        return {
            "folder": final_folder,
            "chapters": saved_chapters,
            "wav": merged_wav,
            "m4a": merged_m4a,
            "srt": srt_path,
            "vtt": vtt_path,
            "run_id": run_token,
        }

    generated_m4b = find_generated_file(output_dir, "*.m4b", "final M4B file")

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
        filterlist=args.filterlist,
        chapter_indices=args.chapter_indices,
        per_chapter_export=args.per_chapter_export,
    )
    logging.info("Run %s finished. Assets: %s", outputs["run_id"], outputs["folder"])


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI guard
        logging.error("preServer.py failed: %s", exc)
        raise
