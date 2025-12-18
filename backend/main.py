#!/usr/bin/env python3
"""Minimal FastAPI backend that mirrors the Gradio pipeline in API form."""

from __future__ import annotations

import os
import json
import re
import secrets
import shutil
import sqlite3
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from ebooklib import epub
import torch
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

from core import (
    find_document_chapters_and_extract_texts,
    find_good_chapters,
    extract_pdf_chapters,
    chapter_beginning_one_liner,
    get_nlp,
    gen_audio_segments,
    write_audio_stream,
    sample_rate,
    clean_line,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
DB_PATH = BACKEND_ROOT / "backend.db"
USERS_ROOT = BACKEND_ROOT / "users"
LOGS_DIR = BACKEND_ROOT / "logs"
PRE_SERVER = PROJECT_ROOT / "preServer.py"
COLLECTION_DIR = PROJECT_ROOT / "audioBook"
AUDIO_PROVE_DIR = PROJECT_ROOT / "audioProve"
MAX_LOG_LINES = 10000

SUPPORTED_EXTS = {".pdf", ".epub"}
RUN_ID_PATTERN = re.compile(r"\d{8}_\d{6}$")
ARTIFACT_KINDS = {
    ".wav": "wav",
    ".m4a": "wav",
    ".srt": "srt",
    ".vtt": "vtt",
    ".pdf": "ebook",
    ".epub": "ebook",
    ".vst": "vst",
    ".vst3": "vst",
    ".zip": "vst",
}

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin"

app = FastAPI(title="Chatterblez Backend", version="1.0.0")
security = HTTPBasic()
service_lock = threading.Lock()
preview_lock = threading.Lock()
tts_cache: Dict[str, Any] = {"mono": None, "multi": None}
CURRENT_JOB: Optional[Dict[str, Any]] = None
CURRENT_JOB_PROCESS: Optional[subprocess.Popen] = None


class ProcessOptions(BaseModel):
    filterlist: Optional[str] = None
    selected_chapters: Optional[List[int]] = None


class VoiceTestRequest(BaseModel):
    text: str
    repetition_penalty: float = 1.1
    min_p: float = 0.02
    top_p: float = 0.95
    exaggeration: float = 0.4
    cfg_weight: float = 0.8
    temperature: float = 0.85
    speed: float = 1.0
    use_multilingual: bool = False
    language_id: str = "en"


def ensure_backend_layout() -> None:
    USERS_ROOT.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_PROVE_DIR.mkdir(parents=True, exist_ok=True)


def init_db() -> None:
    ensure_backend_layout()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                is_admin INTEGER NOT NULL DEFAULT 0,
                in_use INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS books (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                import_path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                processed_runs INTEGER NOT NULL DEFAULT 0,
                last_run_id TEXT,
                last_processed_at TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )
        cur = conn.execute("SELECT id FROM users WHERE username=?", (ADMIN_USERNAME,))
        if not cur.fetchone():
            conn.execute(
                "INSERT INTO users (username, password, is_admin, in_use) VALUES (?, ?, 1, 0)",
                (ADMIN_USERNAME, ADMIN_PASSWORD),
            )
        conn.commit()


init_db()


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")
    return cleaned or "book"


def timestamp() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def require_book_owner(book_id: int, user_id: int) -> sqlite3.Row:
    with get_connection() as conn:
        cur = conn.execute(
            "SELECT * FROM books WHERE id=? AND user_id=?",
            (book_id, user_id),
        )
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Libro non trovato per l'utente.")
    return row


def user_folders(username: str) -> Dict[str, Path]:
    folder_name = f"ID{username.upper()}"
    base = USERS_ROOT / folder_name
    import_dir = base / "BookImport"
    export_dir = base / "BookExport"
    for directory in (base, import_dir, export_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return {"base": base, "import": import_dir, "export": export_dir}


def get_current_user(credentials: HTTPBasicCredentials = Depends(security)) -> Dict:
    with get_connection() as conn:
        cur = conn.execute(
            "SELECT * FROM users WHERE username=?",
            (credentials.username,),
        )
        row = cur.fetchone()
    if not row or not secrets.compare_digest(row["password"], credentials.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenziali non valide.",
            headers={"WWW-Authenticate": "Basic"},
        )
    user = dict(row)
    user["folders"] = user_folders(user["username"])
    return user


def set_user_in_use(user_id: int, flag: bool) -> None:
    with get_connection() as conn:
        conn.execute("UPDATE users SET in_use=? WHERE id=?", (1 if flag else 0, user_id))
        conn.commit()


def reset_all_users_in_use() -> None:
    with get_connection() as conn:
        conn.execute("UPDATE users SET in_use=0")
        conn.commit()


def list_exports(book_id: int, export_dir: Path) -> List[Dict[str, str]]:
    exports: List[Dict[str, str]] = []
    prefix = f"book_{book_id}_"
    if not export_dir.exists():
        return exports
    for artifact in sorted(export_dir.glob(f"{prefix}*")):
        suffix = artifact.suffix.lower()
        kind = ARTIFACT_KINDS.get(suffix)
        if not kind:
            continue
        match = RUN_ID_PATTERN.search(artifact.stem)
        run_id = match.group(0) if match else ""
        exports.append(
            {
                "kind": kind,
                "path": str(artifact),
                "run_id": run_id if run_id else "",
                "name": artifact.name,
            }
        )
    return exports


def get_tts_model(use_multilingual: bool):
    key = "multi" if use_multilingual else "mono"
    model = tts_cache.get(key)
    if model is not None:
        return model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if use_multilingual:
        model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    else:
        model = ChatterboxTTS.from_pretrained(device=device)
    tts_cache[key] = model
    return model


def synthesize_voice_preview(payload: VoiceTestRequest, user: Dict) -> Path:
    text_lines = []
    for line in (payload.text or "").splitlines():
        cleaned = clean_line(line)
        if cleaned:
            text_lines.append(cleaned)
    text = " ".join(text_lines).strip()
    if not text:
        raise HTTPException(status_code=400, detail="Inserisci almeno qualche parola da leggere.")
    if not text.endswith((".", "!", "?")):
        text += "."
    slug = slugify(user["username"])
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = AUDIO_PROVE_DIR / f"{slug}_{run_id}.wav"
    model = get_tts_model(payload.use_multilingual)
    nlp = get_nlp()

    chunks = gen_audio_segments(
        model,
        nlp,
        text,
        payload.speed,
        stats=None,
        max_sentences=None,
        post_event=None,
        should_stop=lambda: False,
        repetition_penalty=payload.repetition_penalty,
        min_p=payload.min_p,
        top_p=payload.top_p,
        exaggeration=payload.exaggeration,
        cfg_weight=payload.cfg_weight,
        temperature=payload.temperature,
        use_multilingual=payload.use_multilingual,
        language_id=payload.language_id,
        audio_prompt_wav=None,
        sentence_gap_ms=0,
        question_gap_ms=0,
    )
    frames = write_audio_stream(output_path, chunks)
    if frames <= 0:
        raise HTTPException(status_code=500, detail="Sintesi non riuscita, nessun audio generato.")
    return output_path


def describe_book(row: sqlite3.Row, export_dir: Path) -> Dict:
    exports = list_exports(row["id"], export_dir)
    return {
        "id": row["id"],
        "title": row["title"],
        "user_id": row["user_id"],
        "original_filename": row["original_filename"],
        "import_path": row["import_path"],
        "created_at": row["created_at"],
        "processed_runs": row["processed_runs"],
        "last_run_id": row["last_run_id"],
        "last_processed_at": row["last_processed_at"],
        "exports": exports,
    }


def list_book_chapters(row: sqlite3.Row) -> List[Dict[str, Any]]:
    source = Path(row["import_path"])
    if not source.exists():
        raise HTTPException(status_code=404, detail="File del libro non trovato.")
    suffix = source.suffix.lower()
    if suffix == ".epub":
        book = epub.read_epub(str(source))
        chapters = find_document_chapters_and_extract_texts(book)
        default_selected = {c.chapter_index for c in find_good_chapters(chapters)}
    elif suffix == ".pdf":
        chapters = extract_pdf_chapters(str(source), row["title"])
        default_selected = {c.chapter_index for c in chapters}
    else:
        raise HTTPException(status_code=400, detail="Formato non supportato per la lettura dei capitoli.")
    payload = []
    for chapter in chapters:
        extracted = getattr(chapter, "extracted_text", "") or ""
        payload.append(
            {
                "index": chapter.chapter_index,
                "name": chapter.get_name(),
                "preview": chapter_beginning_one_liner(chapter, 120),
                "length": len(extracted),
                "default_selected": chapter.chapter_index in default_selected,
            }
        )
    return payload


def tail_lines(path: Path, limit: int = 30) -> List[str]:
    """Return the last `limit` lines of a potentially large log file efficiently."""
    if limit <= 0:
        return []
    chunk_size = 8192
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            position = handle.tell()
            chunks: List[bytes] = []
            newline_count = 0
            while position > 0 and newline_count <= limit:
                read_size = min(chunk_size, position)
                position -= read_size
                handle.seek(position)
                chunk = handle.read(read_size)
                chunks.append(chunk)
                newline_count += chunk.count(b"\n")
            data = b"".join(reversed(chunks))
    except FileNotFoundError:
        return []
    if not data:
        return []
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    return lines[-limit:]


def trim_log_file(path: Path, retain_lines: int = MAX_LOG_LINES) -> None:
    """Truncate a log file to at most `retain_lines` lines, keeping the newest ones."""
    if retain_lines <= 0:
        path.unlink(missing_ok=True)
        return
    lines = tail_lines(path, retain_lines)
    if not lines:
        return
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temp_path, path)


@app.get("/status")
def api_status():
    with get_connection() as conn:
        cur = conn.execute("SELECT COUNT(*) FROM users WHERE in_use=1")
        busy_users = cur.fetchone()[0]
    job_overview = None
    if CURRENT_JOB:
        job_overview = {
            "book_id": CURRENT_JOB["book_id"],
            "book_title": CURRENT_JOB["book_title"],
            "run_id": CURRENT_JOB["run_id"],
            "started_at": CURRENT_JOB["started_at"],
            "username": CURRENT_JOB["username"],
        }
    return {
        "status": "ok",
        "busy": busy_users > 0,
        "busy_users": busy_users,
        "current_job": job_overview,
    }


@app.get("/jobs/current")
def get_current_job_status(
    lines: int = Query(30, ge=1, le=500),
    user: Dict = Depends(get_current_user),
):
    job = CURRENT_JOB
    if not job or job["user_id"] != user["id"]:
        return {"running": False}
    log_tail = tail_lines(Path(job["log_path"]), lines)
    return {
        "running": True,
        "book_id": job["book_id"],
        "book_title": job["book_title"],
        "run_id": job["run_id"],
        "started_at": job["started_at"],
        "log_tail": log_tail,
        "lines": len(log_tail),
    }


@app.post("/jobs/cancel")
def cancel_current_job(user: Dict = Depends(get_current_user)):
    job = CURRENT_JOB
    if not job:
        raise HTTPException(status_code=404, detail="Nessun job in esecuzione.")
    if job["user_id"] != user["id"] and not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Non puoi annullare questo job.")
    process = CURRENT_JOB_PROCESS
    if not process:
        return {"status": "idle"}
    if process.poll() is not None:
        return {"status": "completed"}
    try:
        process.terminate()
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Errore nell'annullamento: {exc}") from exc
    return {"status": "cancelling"}


@app.post("/books")
async def upload_book(
    title: str = Form(...),
    replace_existing: bool = Form(False),
    file: UploadFile = File(...),
    user: Dict = Depends(get_current_user),
):
    if not title.strip():
        raise HTTPException(status_code=400, detail="Il titolo non può essere vuoto.")
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise HTTPException(status_code=400, detail="Formato file non supportato.")

    with get_connection() as conn:
        if replace_existing:
            conn.execute(
                "DELETE FROM books WHERE user_id=? AND title=?",
                (user["id"], title.strip()),
            )
        cur = conn.execute(
            """
            INSERT INTO books (user_id, title, original_filename, import_path, created_at)
            VALUES (?, ?, ?, '', ?)
            """,
            (user["id"], title.strip(), file.filename or f"{title}{ext}", timestamp()),
        )
        book_id = cur.lastrowid
        conn.commit()

    folders = user["folders"]
    import_dir = folders["import"]
    destination = import_dir / f"book_{book_id}{ext}"
    with destination.open("wb") as handle:
        while True:
            chunk = await file.read(2 * 1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    await file.close()

    with get_connection() as conn:
        conn.execute(
            "UPDATE books SET import_path=? WHERE id=?",
            (str(destination), book_id),
        )
        conn.commit()

    return {
        "book_id": book_id,
        "title": title.strip(),
        "import_path": str(destination),
    }


@app.get("/books")
def list_books(user: Dict = Depends(get_current_user)):
    with get_connection() as conn:
        cur = conn.execute(
            "SELECT * FROM books WHERE user_id=? ORDER BY created_at DESC",
            (user["id"],),
        )
        rows = cur.fetchall()
    export_dir = user["folders"]["export"]
    described = [describe_book(row, export_dir) for row in rows]
    processed = sum(1 for row in described if row["processed_runs"] > 0)
    return {
        "total_books": len(described),
        "processed_books": processed,
        "items": described,
    }


@app.get("/books/{book_id}/chapters")
def get_book_chapters(book_id: int, user: Dict = Depends(get_current_user)):
    book = require_book_owner(book_id, user["id"])
    try:
        chapters = list_book_chapters(book)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Impossibile leggere i capitoli: {exc}") from exc
    return {
        "book_id": book["id"],
        "title": book["title"],
        "chapters": chapters,
    }


def run_pipeline(
    book: sqlite3.Row,
    user: Dict,
    filterlist: Optional[str] = None,
    selected_chapter_indices: Optional[List[int]] = None,
) -> Dict:
    global CURRENT_JOB, CURRENT_JOB_PROCESS
    import_path = Path(book["import_path"])
    if not import_path.exists():
        raise HTTPException(status_code=404, detail="File del libro non trovato.")
    slug = slugify(import_path.stem)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"book_{book['id']}_{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(PRE_SERVER),
        "--book",
        str(import_path),
        "--run-id",
        run_id,
    ]
    if filterlist:
        cmd += ["--filterlist", filterlist]
    if selected_chapter_indices:
        indices = []
        for idx in selected_chapter_indices:
            try:
                indices.append(str(int(idx)))
            except (TypeError, ValueError):
                continue
        if indices:
            cmd += ["--chapter-indices", ",".join(indices)]
    cmd.append("--per-chapter-export")
    job_info = {
        "book_id": book["id"],
        "book_title": book["title"],
        "run_id": run_id,
        "user_id": user["id"],
        "username": user["username"],
        "started_at": timestamp(),
        "log_path": str(log_path),
    }
    retcode = -1
    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                cmd,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            CURRENT_JOB = job_info
            CURRENT_JOB_PROCESS = process
            retcode = process.wait()
    finally:
        CURRENT_JOB = None
        CURRENT_JOB_PROCESS = None
    log_excerpt = tail_lines(log_path, 30)
    trim_log_file(log_path, MAX_LOG_LINES)
    if retcode != 0:
        raise HTTPException(
            status_code=500,
            detail="Errore durante il job:\n" + "\n".join(log_excerpt),
        )

    final_folder = COLLECTION_DIR / slug
    ebook_copy = final_folder / f"{slug}_{run_id}{import_path.suffix.lower()}"

    export_dir = user["folders"]["export"]
    artifacts: List[Dict[str, str]] = []
    chapter_audio = sorted(final_folder.glob(f"{slug}_{run_id}_chapter*.m4a"))
    if not chapter_audio:
        raise HTTPException(
            status_code=500,
            detail="Nessun file per capitolo trovato dopo la conversione.",
        )
    chapter_count = len(chapter_audio)
    for idx, audio_file in enumerate(chapter_audio, start=1):
        chapter_suffix = f"_chapter{idx:03d}"
        dest_audio = export_dir / f"book_{book['id']}_{run_id}{chapter_suffix}{audio_file.suffix.lower()}"
        shutil.copy2(audio_file, dest_audio)
        artifacts.append({"kind": "wav", "path": str(dest_audio), "chapter": idx})
        srt_file = audio_file.with_suffix(".srt")
        if srt_file.exists():
            dest_srt = export_dir / f"book_{book['id']}_{run_id}{chapter_suffix}.srt"
            shutil.copy2(srt_file, dest_srt)
            artifacts.append({"kind": "srt", "path": str(dest_srt), "chapter": idx})
        vtt_file = audio_file.with_suffix(".vtt")
        if vtt_file.exists():
            dest_vtt = export_dir / f"book_{book['id']}_{run_id}{chapter_suffix}.vtt"
            shutil.copy2(vtt_file, dest_vtt)
            artifacts.append({"kind": "vtt", "path": str(dest_vtt), "chapter": idx})

    final_m4a = final_folder / f"{slug}_{run_id}.m4a"
    final_wav = final_folder / f"{slug}_{run_id}.wav"
    final_srt = final_folder / f"{slug}_{run_id}.srt"
    final_vtt = final_folder / f"{slug}_{run_id}.vtt"
    for final_asset in (final_m4a, final_wav, final_srt, final_vtt):
        if final_asset.exists():
            suffix = final_asset.suffix.lower()
            kind = ARTIFACT_KINDS.get(suffix)
            if not kind:
                continue
            target = export_dir / f"book_{book['id']}_{run_id}{suffix}"
            shutil.copy2(final_asset, target)
            artifacts.append({"kind": kind, "path": str(target)})

    if ebook_copy.exists():
        ebook_dest = export_dir / f"book_{book['id']}_{run_id}{ebook_copy.suffix.lower()}"
        shutil.copy2(ebook_copy, ebook_dest)
        artifacts.append({"kind": "ebook", "path": str(ebook_dest)})

    with get_connection() as conn:
        conn.execute(
            """
            UPDATE books
            SET processed_runs = processed_runs + 1,
                last_run_id = ?,
                last_processed_at = ?
            WHERE id=?
            """,
            (run_id, timestamp(), book["id"]),
        )
        conn.commit()

    return {
        "run_id": run_id,
        "mode": "per_chapter",
        "chapter_count": chapter_count,
        "artifacts": artifacts,
        "stdout": log_excerpt,
        "log_path": str(log_path),
    }


@app.post("/books/{book_id}/process")
def process_book(
    book_id: int,
    options: Optional[ProcessOptions] = None,
    user: Dict = Depends(get_current_user),
):
    job_state: Optional[Dict] = None
    with get_connection() as conn:
        cur = conn.execute("SELECT in_use FROM users WHERE id=?", (user["id"],))
        in_use = cur.fetchone()[0]
    if in_use:
        raise HTTPException(status_code=409, detail="Utente già in elaborazione.")
    set_user_in_use(user["id"], True)
    acquired = service_lock.acquire(blocking=False)
    if not acquired:
        set_user_in_use(user["id"], False)
        raise HTTPException(status_code=409, detail="Il servizio sta processando un altro libro.")
    try:
        book = require_book_owner(book_id, user["id"])
        filterlist = (options.filterlist.strip() if options and options.filterlist else None)
        selected_indices = None
        if options and options.selected_chapters:
            cleaned: List[int] = []
            for value in options.selected_chapters:
                try:
                    cleaned.append(int(value))
                except (TypeError, ValueError):
                    continue
            if cleaned:
                selected_indices = cleaned
        job_state = run_pipeline(
            book,
            user,
            filterlist=filterlist,
            selected_chapter_indices=selected_indices,
        )
    finally:
        set_user_in_use(user["id"], False)
        if acquired:
            service_lock.release()
    return {"status": "ok", "details": job_state}


@app.get("/books/{book_id}/exports")
def get_book_exports(book_id: int, user: Dict = Depends(get_current_user)):
    book = require_book_owner(book_id, user["id"])
    export_dir = user["folders"]["export"]
    return {
        "book_id": book["id"],
        "title": book["title"],
        "exports": list_exports(book["id"], export_dir),
    }


def resolve_export_path(book_id: int, run_id: Optional[str], kind: str, export_dir: Path) -> Path:
    if run_id and not RUN_ID_PATTERN.fullmatch(run_id):
        raise HTTPException(status_code=400, detail="run_id non valido.")
    suffixes = [suffix for suffix, mapped in ARTIFACT_KINDS.items() if mapped == kind]
    if not suffixes:
        raise HTTPException(status_code=400, detail="Tipo di file non supportato.")
    if run_id:
        for suffix in suffixes:
            candidate = export_dir / f"book_{book_id}_{run_id}{suffix}"
            if candidate.exists():
                return candidate
    matches = sorted(
        export_dir.glob(f"book_{book_id}_*"),
        reverse=True,
    )
    for match in matches:
        if match.suffix.lower() in suffixes:
            return match
    raise HTTPException(status_code=404, detail="File richiesto non trovato.")


@app.get("/books/{book_id}/download")
def download_artifact(
    book_id: int,
    kind: str = Query(..., description="wav/srt/vtt/ebook/vst"),
    run_id: Optional[str] = Query(None),
    filename: Optional[str] = Query(None),
    user: Dict = Depends(get_current_user),
):
    kind = kind.lower()
    require_book_owner(book_id, user["id"])
    export_dir = user["folders"]["export"]
    if filename:
        sanitized = Path(filename).name
        if not sanitized.startswith(f"book_{book_id}_"):
            raise HTTPException(status_code=400, detail="Filename non valido per questo libro.")
        path = (export_dir / sanitized).resolve()
        base = export_dir.resolve()
        if not str(path).startswith(str(base)) or not path.exists():
            raise HTTPException(status_code=404, detail="File richiesto non trovato.")
    else:
        path = resolve_export_path(book_id, run_id, kind, export_dir)
    return FileResponse(path, filename=path.name)


def delete_exports(book_id: int, export_dir: Path, run_id: Optional[str] = None) -> List[str]:
    removed: List[str] = []
    pattern = f"book_{book_id}_*"
    if run_id:
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise HTTPException(status_code=400, detail="run_id non valido.")
        pattern = f"book_{book_id}_{run_id}*"
    for artifact in export_dir.glob(pattern):
        artifact.unlink(missing_ok=True)
        removed.append(artifact.name)
    return removed


@app.delete("/books/{book_id}")
def delete_book(
    book_id: int,
    delete_exports_flag: bool = Query(False, alias="delete_exports"),
    user: Dict = Depends(get_current_user),
):
    book = require_book_owner(book_id, user["id"])
    import_path = Path(book["import_path"])
    if import_path.exists():
        import_path.unlink()
    if delete_exports_flag:
        delete_exports(book_id, user["folders"]["export"])
    with get_connection() as conn:
        conn.execute("DELETE FROM books WHERE id=?", (book_id,))
        conn.commit()
    return {"status": "deleted", "book_id": book_id}


@app.delete("/books/{book_id}/exports")
def delete_book_exports(
    book_id: int,
    run_id: Optional[str] = Query(None),
    user: Dict = Depends(get_current_user),
):
    require_book_owner(book_id, user["id"])
    removed = delete_exports(book_id, user["folders"]["export"], run_id=run_id)
    return {"book_id": book_id, "removed": removed}


@app.post("/voice-test")
def voice_test(
    payload: VoiceTestRequest,
    user: Dict = Depends(get_current_user),
):
    acquired = preview_lock.acquire(blocking=False)
    if not acquired:
        raise HTTPException(status_code=409, detail="È già in corso una generazione di prova.")
    try:
        audio_path = synthesize_voice_preview(payload, user)
    finally:
        if acquired:
            preview_lock.release()
    return FileResponse(audio_path, filename=audio_path.name)


@app.get("/example/useTest")
def example_use_test():
    sample_payload = {
        "text": "Ciao! Questa è una prova veloce della tua voce virtuale.",
        "repetition_penalty": 1.05,
        "min_p": 0.02,
        "top_p": 0.92,
        "exaggeration": 0.4,
        "cfg_weight": 0.32,
        "temperature": 0.9,
        "speed": 0.95,
        "use_multilingual": True,
        "language_id": "it",
    }
    curl_example = (
        "curl -u admin:admin "
        "-H 'Content-Type: application/json' "
        "-X POST http://localhost:8000/voice-test "
        f"-d '{json.dumps(sample_payload)}' --output prova.wav"
    )
    return {
        "endpoint": "/voice-test",
        "method": "POST",
        "authentication": "HTTP Basic",
        "content_type": "application/json",
        "sample_payload": sample_payload,
        "ios_hint": "Invia il JSON via fetch/axios e salva il body della risposta (audio/wav) sul dispositivo.",
        "curl_example": curl_example,
        "response_description": "Restituisce un file WAV con la frase sintetizzata usando i parametri indicati.",
    }


def kill_external_processes() -> Dict[str, bool]:
    results: Dict[str, bool] = {}
    pkill_exists = shutil.which("pkill")
    if not pkill_exists:
        results["pkill_available"] = False
        return results
    for target in ("ffmpeg", "preServer.py"):
        try:
            proc = subprocess.run(["pkill", "-f", target], check=False)
            results[target] = proc.returncode == 0
        except Exception:
            results[target] = False
    return results


@app.post("/jobs/abort-all")
def abort_all_jobs(user: Dict = Depends(get_current_user)):
    global CURRENT_JOB, CURRENT_JOB_PROCESS
    terminated = False
    if CURRENT_JOB_PROCESS and CURRENT_JOB_PROCESS.poll() is None:
        try:
            CURRENT_JOB_PROCESS.terminate()
            CURRENT_JOB_PROCESS.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                CURRENT_JOB_PROCESS.kill()
            except Exception:
                pass
        except Exception:
            pass
        terminated = True
    CURRENT_JOB_PROCESS = None
    CURRENT_JOB = None
    cleanup = kill_external_processes()
    reset_all_users_in_use()
    try:
        service_lock.release()
    except RuntimeError:
        pass
    return {
        "status": "aborted",
        "terminated_current": terminated,
        "external_cleanup": cleanup,
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=False)
