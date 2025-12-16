#!/usr/bin/env python3
"""Single-user Gradio UI with lightweight API endpoints for the audiobook pipeline."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import gradio as gr
from fastapi import HTTPException
from fastapi.responses import FileResponse, JSONResponse

SCRIPT_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = SCRIPT_DIR / "remote_uploads"
BOOK_COLLECTION = SCRIPT_DIR / "audioBook"
PRE_SERVER = SCRIPT_DIR / "preServer.py"
SUPPORTED_EXTS = {".epub", ".pdf"}
RUN_ID_PATTERN = re.compile(r"\d{8}_\d{6}$")
ARTIFACT_SUFFIXES = {
    "wav": [".wav"],
    "srt": [".srt"],
    "vtt": [".vtt"],
    "ebook": [".epub", ".pdf"],
}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

processing_lock = threading.Lock()


def slugify(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")
    return cleaned or "book"


def _blank_state() -> dict:
    return {"wav": None, "srt": None, "vtt": None, "ebook": None}


def _blank_download():
    return gr.update(value=None, visible=False)


def _prepare_download(path: str | Path | None):
    if path and Path(path).exists():
        return gr.update(value=str(path), visible=True)
    return _blank_download()


def _empty_response(message: str):
    blanks = (_blank_download(),) * 4
    return (message, {}, *blanks, _blank_state())


def handle_conversion(file_path: str | None):
    if not file_path:
        return _empty_response("Carica prima un file EPUB/PDF.")

    if not processing_lock.acquire(blocking=False):
        return _empty_response("È già in corso un'altra conversione. Riprova tra poco.")

    try:
        source = Path(file_path)
        if not source.exists():
            return _empty_response("File temporaneo non trovato. Ricarica il documento.")

        ext = source.suffix.lower()
        if ext not in SUPPORTED_EXTS:
            return _empty_response(f"Formato {ext} non supportato. Carica un EPUB o PDF.")

        slug = slugify(source.stem)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        upload_target = UPLOAD_DIR / f"{slug}{ext}"
        shutil.copy(source, upload_target)

        cmd = [
            sys.executable,
            str(PRE_SERVER),
            "--book",
            str(upload_target),
            "--run-id",
            run_id,
        ]

        process = subprocess.run(cmd, capture_output=True, text=True)
        if process.returncode != 0:
            stderr = process.stderr.strip() or process.stdout.strip()
            upload_target.unlink(missing_ok=True)
            return _empty_response(f"Errore durante la conversione: {stderr}")

        final_folder = BOOK_COLLECTION / slug
        wav_path = final_folder / f"{slug}_{run_id}.wav"
        srt_path = final_folder / f"{slug}_{run_id}.srt"
        vtt_path = final_folder / f"{slug}_{run_id}.vtt"
        ebook_target = final_folder / f"{slug}_{run_id}{ext}"
        try:
            shutil.copy(upload_target, ebook_target)
        finally:
            upload_target.unlink(missing_ok=True)

        info = {
            "run_id": run_id,
            "wav": str(wav_path) if wav_path.exists() else "non trovato",
            "srt": str(srt_path) if srt_path.exists() else "non trovato",
            "vtt": str(vtt_path) if vtt_path.exists() else "non trovato",
            "folder": str(final_folder),
            "ebook": str(ebook_target) if ebook_target.exists() else "non trovato",
        }

        logs = process.stdout.strip().splitlines()
        if logs:
            info["log_tail"] = logs[-10:]

        paths_state = {
            "wav": str(wav_path) if wav_path.exists() else None,
            "srt": str(srt_path) if srt_path.exists() else None,
            "vtt": str(vtt_path) if vtt_path.exists() else None,
            "ebook": str(ebook_target) if ebook_target.exists() else None,
        }

        return (
            f"Conversione completata: {slug}",
            info,
            _prepare_download(paths_state["wav"]),
            _prepare_download(paths_state["srt"]),
            _prepare_download(paths_state["vtt"]),
            _prepare_download(paths_state["ebook"]),
            paths_state,
        )

    finally:
        processing_lock.release()


def delete_artifact(kind: str, paths: dict | None):
    label = {"wav": "WAV", "srt": "SRT", "vtt": "VTT", "ebook": "ebook"}.get(kind, kind)
    paths = dict(paths or _blank_state())
    target = paths.get(kind)
    if not target:
        return f"Nessun file {label} disponibile.", _blank_download(), paths
    file_path = Path(target)
    if not file_path.exists():
        paths[kind] = None
        return f"File {label} già rimosso.", _blank_download(), paths
    try:
        file_path.unlink()
    except Exception as exc:
        return f"Errore rimuovendo {label}: {exc}", _prepare_download(file_path), paths
    paths[kind] = None
    return f"File {label} eliminato.", _blank_download(), paths


def _list_library() -> List[Dict]:
    library: List[Dict] = []
    if not BOOK_COLLECTION.exists():
        return library
    for book_dir in sorted(BOOK_COLLECTION.iterdir()):
        if not book_dir.is_dir():
            continue
        runs: Dict[str, Dict[str, str]] = {}
        for artifact in sorted(book_dir.iterdir()):
            if not artifact.is_file():
                continue
            match = RUN_ID_PATTERN.search(artifact.stem)
            if not match:
                continue
            run_id = match.group(0)
            entry = runs.setdefault(run_id, {"run_id": run_id, "book": book_dir.name})
            suffix = artifact.suffix.lower()
            if suffix == ".wav":
                entry["wav"] = str(artifact)
            elif suffix == ".srt":
                entry["srt"] = str(artifact)
            elif suffix == ".vtt":
                entry["vtt"] = str(artifact)
            elif suffix in SUPPORTED_EXTS:
                entry["ebook"] = str(artifact)
        if runs:
            run_list = sorted(runs.values(), key=lambda r: r["run_id"], reverse=True)
            library.append({"book": book_dir.name, "runs": run_list})
    return library


def _resolve_artifact(book: str, run_id: str, kind: str) -> Path:
    kind = kind.lower()
    suffixes = ARTIFACT_SUFFIXES.get(kind)
    if not suffixes:
        raise HTTPException(status_code=400, detail="Tipo di file non supportato.")
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise HTTPException(status_code=400, detail="run_id non valido.")
    folder = (BOOK_COLLECTION / book).resolve()
    base = BOOK_COLLECTION.resolve()
    if not str(folder).startswith(str(base)) or not folder.is_dir():
        raise HTTPException(status_code=404, detail="Libro non trovato.")
    for suffix in suffixes:
        candidate = folder / f"{book}_{run_id}{suffix}"
        if candidate.exists():
            return candidate
    for suffix in suffixes:
        matches = sorted(folder.glob(f"*_{run_id}{suffix}"))
        if matches:
            return matches[0]
    raise HTTPException(status_code=404, detail="File non trovato.")


def register_api_routes(demo: gr.Blocks) -> None:
    app = demo.app

    @app.get("/api/status")
    def api_status():
        return {"active": processing_lock.locked()}

    @app.get("/api/library")
    def api_library():
        return JSONResponse(_list_library())

    @app.get("/api/download")
    def api_download(book: str, run_id: str, kind: str):
        artifact = _resolve_artifact(book, run_id, kind)
        return FileResponse(artifact, filename=artifact.name)

    @app.delete("/api/library")
    def api_delete(book: str, run_id: str, kind: str):
        artifact = _resolve_artifact(book, run_id, kind)
        artifact.unlink()
        return {"status": "deleted", "book": book, "run_id": run_id, "kind": kind}


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Chatterblez PreServer UI") as demo:
        gr.Markdown("## Upload EPUB/PDF e genera audiolibro + sottotitoli")
        with gr.Row():
            file_input = gr.File(
                label="Carica file",
                file_types=list(SUPPORTED_EXTS),
                type="filepath",
            )
        status = gr.Textbox(
            label="Stato",
            interactive=False,
            placeholder="In attesa...",
        )
        info_output = gr.JSON(label="Dettagli output")
        start_button = gr.Button("Avvia conversione", variant="primary")
        with gr.Row():
            wav_download = gr.File(label="Scarica WAV", interactive=False, visible=False)
            srt_download = gr.File(label="Scarica SRT", interactive=False, visible=False)
            vtt_download = gr.File(label="Scarica VTT", interactive=False, visible=False)
            ebook_download = gr.File(label="Scarica eBook", interactive=False, visible=False)
        with gr.Row():
            wav_delete = gr.Button("Elimina WAV")
            srt_delete = gr.Button("Elimina SRT")
            vtt_delete = gr.Button("Elimina VTT")
            ebook_delete = gr.Button("Elimina eBook")

        paths_state = gr.State(_blank_state())

        start_button.click(
            handle_conversion,
            inputs=file_input,
            outputs=[status, info_output, wav_download, srt_download, vtt_download, ebook_download, paths_state],
        )

        wav_delete.click(
            lambda paths: delete_artifact("wav", paths),
            inputs=paths_state,
            outputs=[status, wav_download, paths_state],
        )
        srt_delete.click(
            lambda paths: delete_artifact("srt", paths),
            inputs=paths_state,
            outputs=[status, srt_download, paths_state],
        )
        vtt_delete.click(
            lambda paths: delete_artifact("vtt", paths),
            inputs=paths_state,
            outputs=[status, vtt_download, paths_state],
        )
        ebook_delete.click(
            lambda paths: delete_artifact("ebook", paths),
            inputs=paths_state,
            outputs=[status, ebook_download, paths_state],
        )

    demo.queue(concurrency_count=1)
    return demo


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    ui = build_interface()
    register_api_routes(ui)
    ui.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True,
    )
