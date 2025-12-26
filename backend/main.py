#!/usr/bin/env python3
"""Minimal FastAPI backend that mirrors the Gradio pipeline in API form."""

from __future__ import annotations

import argparse
import math
import os
import json
import logging
import re
import secrets
import shutil
import sqlite3
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
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
    load_tts_resources,
    get_tts_engine_name,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
# Local sqlite is only to satisfy the legacy pipeline; Central is the source of truth.
DB_PATH = BACKEND_ROOT / "backend.db"
USERS_ROOT = BACKEND_ROOT / "users"
LOGS_DIR = BACKEND_ROOT / "logs"
PRE_SERVER = PROJECT_ROOT / "preServer.py"
AUDIO_PROVE_DIR = PROJECT_ROOT / "audioProve"
MAX_LOG_LINES = 10000

SUPPORTED_EXTS = {".pdf", ".epub"}
RUN_ID_PATTERN = re.compile(r"\d{8}_\d{6}$")
ARTIFACT_KINDS = {
    ".m4a": "wav",
    ".wav": "wav",
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

# These are injected by the Central when the worker is spawned (no client auth here).
CENTRAL_BASE_URL = os.getenv("CENTRAL_BASE_URL", "").rstrip("/")
WORKER_SHARED_TOKEN = os.getenv("WORKER_SHARED_TOKEN", "")
WORKER_JOB_ID = os.getenv("WORKER_JOB_ID", "")
WORKER_HEARTBEAT_SECONDS = int(os.getenv("WORKER_HEARTBEAT_SECONDS", "300"))
WORKER_REQUEST_TIMEOUT = int(os.getenv("WORKER_REQUEST_TIMEOUT", "30"))
WORKER_UPLOAD_CHUNK_MB = int(os.getenv("WORKER_UPLOAD_CHUNK_MB", "25"))

# FastAPI only matters in standalone mode; worker jobs exit after processing.
app = FastAPI(title="Chatterblez Backend", version="1.0.0")
security = HTTPBasic()
service_lock = threading.Lock()
preview_lock = threading.Lock()
tts_cache: Dict[Tuple[str, bool], Dict[str, Any]] = {}
CURRENT_JOB: Optional[Dict[str, Any]] = None
CURRENT_JOB_PROCESS: Optional[subprocess.Popen] = None


class ProcessOptions(BaseModel):
    filterlist: Optional[str] = None
    selected_chapters: Optional[List[int]] = None
    repetition_penalty: Optional[float] = None
    min_p: Optional[float] = None
    top_p: Optional[float] = None
    exaggeration: Optional[float] = None
    cfg_weight: Optional[float] = None
    temperature: Optional[float] = None
    speed: Optional[float] = None
    use_multilingual: Optional[bool] = None
    language_id: Optional[str] = None
    top_k: Optional[int] = None


class VoiceTestRequest(BaseModel):
    text: str
    repetition_penalty: float = 1.1
    min_p: float = 0.02
    top_p: float = 0.95
    exaggeration: float = 0.4
    cfg_weight: float = 0.8
    temperature: float = 0.85
    top_k: Optional[int] = None
    speed: float = 1.0
    use_multilingual: bool = False
    language_id: str = "en"


# Central HTTP client for internal endpoints (job info, heartbeat, artifacts).
class CentralClient:
    def __init__(self, base_url: str, token: str, timeout: int = WORKER_REQUEST_TIMEOUT):
        if not base_url:
            raise ValueError("Central base URL non configurata.")
        if not token:
            raise ValueError("Worker token non configurato.")
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        return {"X-Worker-Token": self.token}

    def request(self, method: str, path: str, **kwargs) -> requests.Response:
        url = f"{self.base_url}{path}"
        headers = kwargs.pop("headers", {}) or {}
        timeout = kwargs.pop("timeout", self.timeout)
        headers.update(self._headers())
        response = requests.request(
            method,
            url,
            headers=headers,
            timeout=timeout,
            **kwargs,
        )
        if not response.ok:
            raise RuntimeError(f"Central {method} {path} failed: {response.status_code} {response.text}")
        return response

    def get_job_info(self, job_id: int) -> Dict[str, Any]:
        return self.request("GET", f"/internal/jobs/{job_id}").json()

    def heartbeat(self, job_id: int, message: Optional[str] = None) -> None:
        payload: Dict[str, Any] = {}
        if message:
            payload["message"] = message
        self.request("POST", f"/internal/jobs/{job_id}/heartbeat", json=payload)

    def complete(
        self,
        job_id: int,
        status_value: str,
        error: Optional[str] = None,
        artifacts: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        payload: Dict[str, Any] = {"status": status_value}
        if error:
            payload["error"] = error
        if artifacts is not None:
            payload["artifacts"] = artifacts
        self.request("POST", f"/internal/jobs/{job_id}/complete", json=payload)

    def download_input(self, job_id: int, destination: Path) -> None:
        response = self.request(
            "GET",
            f"/internal/jobs/{job_id}/input",
            stream=True,
            timeout=(10, 600),
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)

    def upload_artifact(self, job_id: int, path: Path) -> None:
        if not path.exists():
            raise RuntimeError(f"Artifact not found: {path}")
        chunk_bytes = max(1, WORKER_UPLOAD_CHUNK_MB) * 1024 * 1024
        size = path.stat().st_size
        if size <= chunk_bytes:
            with path.open("rb") as handle:
                files = {"file": (path.name, handle, "application/octet-stream")}
                data = {"filename": path.name}
                self.request(
                    "POST",
                    f"/internal/jobs/{job_id}/artifacts",
                    files=files,
                    data=data,
                    timeout=(10, 600),
                )
            return

        total_chunks = max(1, math.ceil(size / chunk_bytes))
        with path.open("rb") as handle:
            for index in range(total_chunks):
                chunk = handle.read(chunk_bytes)
                if not chunk:
                    break
                files = {"file": (path.name, chunk, "application/octet-stream")}
                data = {
                    "filename": path.name,
                    "chunk_index": str(index),
                    "total_chunks": str(total_chunks),
                }
                self.request(
                    "POST",
                    f"/internal/jobs/{job_id}/artifacts",
                    files=files,
                    data=data,
                    timeout=(10, 600),
                )

    def upload_log(self, job_id: int, path: Path) -> None:
        if not path.exists():
            return
        with path.open("rb") as handle:
            files = {"file": (path.name, handle, "text/plain")}
            self.request(
                "POST",
                f"/internal/jobs/{job_id}/log",
                files=files,
                timeout=(10, 600),
            )

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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS job_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                book_id INTEGER NOT NULL,
                run_id TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                status TEXT NOT NULL DEFAULT 'running',
                error TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (book_id) REFERENCES books(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_job_history_user ON job_history(user_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_job_history_book ON job_history(book_id)"
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


# These helpers create local stub rows for the worker pipeline (no user auth here).
def ensure_worker_user(username: str) -> sqlite3.Row:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        if row:
            return row
        password = secrets.token_urlsafe(16)
        conn.execute(
            "INSERT INTO users (username, password, is_admin, in_use) VALUES (?, ?, 0, 0)",
            (username, password),
        )
        conn.commit()
        return conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()


def ensure_worker_book(
    user_id: int,
    book_id: int,
    title: str,
    original_filename: str,
    import_path: str,
) -> sqlite3.Row:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM books WHERE id=?", (book_id,)).fetchone()
        if row:
            conn.execute(
                """
                UPDATE books
                SET user_id = ?, title = ?, original_filename = ?, import_path = ?
                WHERE id = ?
                """,
                (user_id, title, original_filename, import_path, book_id),
            )
        else:
            conn.execute(
                """
                INSERT INTO books (id, user_id, title, original_filename, import_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (book_id, user_id, title, original_filename, import_path, timestamp()),
            )
        conn.commit()
        return conn.execute("SELECT * FROM books WHERE id=?", (book_id,)).fetchone()


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")
    return cleaned or "book"


def timestamp() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def create_job_history_entry(user_id: int, book_id: int, run_id: str) -> Optional[int]:
    try:
        with get_connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO job_history (user_id, book_id, run_id, started_at, status)
                VALUES (?, ?, ?, ?, 'running')
                """,
                (user_id, book_id, run_id, timestamp()),
            )
            conn.commit()
            return cur.lastrowid
    except Exception as exc:  # pragma: no cover - non-blocking log
        logging.warning("Failed to log job start: %s", exc)
        return None


def finish_job_history_entry(job_id: Optional[int], status: str, error: Optional[str] = None) -> None:
    if job_id is None:
        return
    try:
        with get_connection() as conn:
            conn.execute(
                """
                UPDATE job_history
                SET finished_at = ?, status = ?, error = ?
                WHERE id = ?
                """,
                (timestamp(), status, error, job_id),
            )
            conn.commit()
    except Exception as exc:  # pragma: no cover - non-blocking log
        logging.warning("Failed to log job finish: %s", exc)


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
    # Per-user library/work folders keep outputs out of global paths.
    export_dir = base / "Books"
    work_dir = base / "BookWork"
    for directory in (base, import_dir, export_dir, work_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return {"base": base, "import": import_dir, "export": export_dir, "work": work_dir}


def book_export_dir(export_root: Path, book_id: int) -> Path:
    # Keep a stable per-book folder under each user, so exports are not global.
    return export_root / f"book_{book_id}"


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
    book_dir = book_export_dir(export_dir, book_id)
    prefix = f"book_{book_id}_"
    if not book_dir.exists():
        return exports
    for artifact in sorted(book_dir.glob(f"{prefix}*")):
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


def get_tts_resources_cached(use_multilingual: bool):
    engine = get_tts_engine_name()
    key = (engine, use_multilingual)
    cached = tts_cache.get(key)
    if cached is not None:
        return cached
    resources = load_tts_resources(use_multilingual=use_multilingual, cache=False)
    tts_cache[key] = resources
    return resources


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
    tts_resources = get_tts_resources_cached(payload.use_multilingual)
    nlp = get_nlp()

    chunks = gen_audio_segments(
        tts_resources,
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
        top_k=payload.top_k,
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
    repetition_penalty: Optional[float] = None,
    min_p: Optional[float] = None,
    top_p: Optional[float] = None,
    exaggeration: Optional[float] = None,
    cfg_weight: Optional[float] = None,
    temperature: Optional[float] = None,
    speed: Optional[float] = None,
    use_multilingual: Optional[bool] = None,
    language_id: Optional[str] = None,
    top_k: Optional[int] = None,
    run_id_override: Optional[str] = None,
) -> Dict:
    global CURRENT_JOB, CURRENT_JOB_PROCESS
    # This calls PRE_SERVER which handles ffmpeg/whisper/azzurra/csm end-to-end.
    job_history_id: Optional[int] = None
    job_status = "failed"
    job_error: Optional[str] = None
    try:
        import_path = Path(book["import_path"])
        if not import_path.exists():
            raise HTTPException(status_code=404, detail="File del libro non trovato.")
        slug = slugify(import_path.stem)
        if run_id_override:
            if not RUN_ID_PATTERN.fullmatch(run_id_override):
                raise HTTPException(status_code=400, detail="run_id non valido.")
            run_id = run_id_override
        else:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_history_id = create_job_history_entry(user["id"], book["id"], run_id)
        log_path = LOGS_DIR / f"book_{book['id']}_{run_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        work_root = user["folders"]["work"]
        output_base = work_root / "output"
        # Use per-user temp dirs so chapter splits never land in global folders.
        collect_dir = work_root / f"collect_{run_id}"
        # Launch the core pipeline in a separate process (logs go to file).
        cmd = [
            sys.executable,
            str(PRE_SERVER),
            "--book",
            str(import_path),
            "--run-id",
            run_id,
            "--output-base",
            str(output_base),
            "--collect-dir",
            str(collect_dir),
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
        if repetition_penalty is not None:
            cmd += ["--repetition-penalty", f"{float(repetition_penalty):.6f}"]
        if min_p is not None:
            cmd += ["--min-p", f"{float(min_p):.6f}"]
        if top_p is not None:
            cmd += ["--top-p", f"{float(top_p):.6f}"]
        if exaggeration is not None:
            cmd += ["--exaggeration", f"{float(exaggeration):.6f}"]
        if cfg_weight is not None:
            cmd += ["--cfg-weight", f"{float(cfg_weight):.6f}"]
        if temperature is not None:
            cmd += ["--temperature", f"{float(temperature):.6f}"]
        if speed is not None:
            cmd += ["--speed", f"{float(speed):.6f}"]
        if use_multilingual is not None:
            cmd.append("--use-multilingual" if use_multilingual else "--no-use-multilingual")
        if language_id:
            cmd += ["--language-id", language_id]
        if top_k is not None:
            cmd += ["--top-k", str(int(top_k))]
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
            job_error = "Errore durante il job:\n" + "\n".join(log_excerpt)
            raise HTTPException(
                status_code=500,
                detail="Errore durante il job:\n" + "\n".join(log_excerpt),
            )

        export_root = user["folders"]["export"]
        # Final deliverables live under backend/users/ID<USER>/Books/book_<id>/.
        book_dir = book_export_dir(export_root, book["id"])
        book_dir.mkdir(parents=True, exist_ok=True)
        final_folder = collect_dir / slug

        artifacts: List[Dict[str, str]] = []
        chapter_audio = sorted(final_folder.glob(f"{slug}_{run_id}_chapter*.m4a"))
        if not chapter_audio:
            raise HTTPException(
                status_code=500,
                detail="Nessun file per capitolo trovato dopo la conversione.",
            )
        chapter_count = len(chapter_audio)
        final_m4a = final_folder / f"{slug}_{run_id}.m4a"
        final_srt = final_folder / f"{slug}_{run_id}.srt"
        if not final_m4a.exists():
            raise HTTPException(
                status_code=500,
                detail="File finale M4A non trovato dopo la conversione.",
            )
        dest_m4a = book_dir / f"book_{book['id']}_{run_id}{final_m4a.suffix.lower()}"
        shutil.copy2(final_m4a, dest_m4a)
        artifacts.append({"kind": "wav", "path": str(dest_m4a)})
        if final_srt.exists():
            dest_srt = book_dir / f"book_{book['id']}_{run_id}{final_srt.suffix.lower()}"
            shutil.copy2(final_srt, dest_srt)
            artifacts.append({"kind": "srt", "path": str(dest_srt)})
        ebook_dest = book_dir / f"book_{book['id']}_{run_id}{import_path.suffix.lower()}"
        shutil.copy2(import_path, ebook_dest)
        artifacts.append({"kind": "ebook", "path": str(ebook_dest)})

        # Cleanup: drop per-chapter/temp outputs once final files are copied.
        output_dir = output_base / f"{slug}_{run_id}"
        for temp_dir in (output_dir, collect_dir):
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

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

        job_status = "success"
        return {
            "run_id": run_id,
            "mode": "per_chapter",
            "chapter_count": chapter_count,
            "artifacts": artifacts,
            "stdout": log_excerpt,
            "log_path": str(log_path),
        }
    except HTTPException as exc:
        if job_error is None:
            job_error = str(exc.detail)
        raise
    except Exception as exc:  # pragma: no cover - bubble up
        job_error = str(exc)
        raise
    finally:
        finish_job_history_entry(job_history_id, job_status, job_error)


def build_pipeline_kwargs(options: Optional[ProcessOptions]) -> Dict[str, Any]:
    if not options:
        return {}
    kwargs: Dict[str, Any] = {}
    if options.filterlist:
        kwargs["filterlist"] = options.filterlist.strip()
    if options.selected_chapters:
        cleaned: List[int] = []
        for value in options.selected_chapters:
            try:
                cleaned.append(int(value))
            except (TypeError, ValueError):
                continue
        if cleaned:
            kwargs["selected_chapter_indices"] = cleaned
    if options.repetition_penalty is not None:
        kwargs["repetition_penalty"] = options.repetition_penalty
    if options.min_p is not None:
        kwargs["min_p"] = options.min_p
    if options.top_p is not None:
        kwargs["top_p"] = options.top_p
    if options.exaggeration is not None:
        kwargs["exaggeration"] = options.exaggeration
    if options.cfg_weight is not None:
        kwargs["cfg_weight"] = options.cfg_weight
    if options.temperature is not None:
        kwargs["temperature"] = options.temperature
    if options.speed is not None:
        kwargs["speed"] = options.speed
    if options.use_multilingual is not None:
        kwargs["use_multilingual"] = options.use_multilingual
    if options.language_id:
        kwargs["language_id"] = options.language_id
    if options.top_k is not None:
        kwargs["top_k"] = options.top_k
    return kwargs


def heartbeat_loop(
    client: CentralClient,
    job_id: int,
    stop_event: threading.Event,
    interval_seconds: int,
) -> None:
    while not stop_event.wait(timeout=max(1, interval_seconds)):
        try:
            client.heartbeat(job_id)
        except Exception as exc:  # pragma: no cover - best effort
            logging.warning("Heartbeat failed for job %s: %s", job_id, exc)


def run_worker_job(
    job_id: int,
    central_url: str,
    worker_token: str,
    heartbeat_interval: int,
) -> int:
    # Fetch job from Central, run pipeline, upload artifacts/log, then notify completion.
    client = CentralClient(central_url, worker_token)
    job_info = client.get_job_info(job_id)
    run_id = job_info["run_id"]
    username = job_info["username"]
    book_id = int(job_info["book_id"])
    book_title = job_info["book_title"]
    payload = job_info.get("payload") or {}
    input_filename = job_info.get("input_filename") or f"book_{book_id}.pdf"
    ext = Path(input_filename).suffix or ".pdf"

    client.heartbeat(job_id, "worker-started")

    folders = user_folders(username)
    import_path = folders["import"] / f"book_{book_id}{ext}"
    client.download_input(job_id, import_path)

    user_row = ensure_worker_user(username)
    book_row = ensure_worker_book(
        user_id=user_row["id"],
        book_id=book_id,
        title=book_title,
        original_filename=input_filename,
        import_path=str(import_path),
    )
    user = dict(user_row)
    user["folders"] = folders

    options = ProcessOptions(**payload) if payload else None
    pipeline_kwargs = build_pipeline_kwargs(options)

    stop_event = threading.Event()
    heartbeat_thread = threading.Thread(
        target=heartbeat_loop,
        args=(client, job_id, stop_event, heartbeat_interval),
        daemon=True,
    )
    heartbeat_thread.start()

    status_value = "failed"
    error_message: Optional[str] = None
    artifacts: Optional[List[Dict[str, str]]] = None
    try:
        result = run_pipeline(book_row, user, run_id_override=run_id, **pipeline_kwargs)
        artifacts = result.get("artifacts") if isinstance(result, dict) else None
        if artifacts:
            for artifact in artifacts:
                artifact_path = Path(artifact["path"])
                client.upload_artifact(job_id, artifact_path)
        status_value = "success"
    except HTTPException as exc:
        error_message = str(exc.detail)
    except Exception as exc:
        error_message = str(exc)
    finally:
        stop_event.set()
        heartbeat_thread.join(timeout=5)
        log_path = LOGS_DIR / f"book_{book_id}_{run_id}.log"
        try:
            client.upload_log(job_id, log_path)
        except Exception as exc:  # pragma: no cover - best effort
            logging.warning("Failed to upload log for job %s: %s", job_id, exc)
        try:
            client.complete(job_id, status_value, error_message, artifacts)
        except Exception as exc:  # pragma: no cover - best effort
            logging.error("Failed to notify completion for job %s: %s", job_id, exc)
            return 1
    return 0 if status_value == "success" else 1


def parse_worker_args(argv: List[str]) -> Dict[str, Optional[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("job_id", nargs="?", help="Job ID assegnato dal server central.")
    parser.add_argument("--job-id", dest="job_id_opt")
    parser.add_argument("--central-url", dest="central_url")
    parser.add_argument("--worker-token", dest="worker_token")
    parser.add_argument("--heartbeat", dest="heartbeat", type=int)
    args, _ = parser.parse_known_args(argv)
    job_id_value = args.job_id_opt or args.job_id
    return {
        "job_id": job_id_value,
        "central_url": args.central_url,
        "worker_token": args.worker_token,
        "heartbeat": str(args.heartbeat) if args.heartbeat is not None else None,
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
        repetition_penalty = options.repetition_penalty if options else None
        min_p = options.min_p if options else None
        top_p = options.top_p if options else None
        exaggeration = options.exaggeration if options else None
        cfg_weight = options.cfg_weight if options else None
        temperature = options.temperature if options else None
        speed = options.speed if options else None
        use_multilingual = options.use_multilingual if options else None
        language_id = options.language_id if options else None
        top_k = options.top_k if options else None
        job_state = run_pipeline(
            book,
            user,
            filterlist=filterlist,
            selected_chapter_indices=selected_indices,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            speed=speed,
            use_multilingual=use_multilingual,
            language_id=language_id,
            top_k=top_k,
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
    book_dir = book_export_dir(export_dir, book_id)
    if not book_dir.exists():
        raise HTTPException(status_code=404, detail="File richiesto non trovato.")
    if run_id:
        for suffix in suffixes:
            candidate = book_dir / f"book_{book_id}_{run_id}{suffix}"
            if candidate.exists():
                return candidate
    matches = sorted(
        book_dir.glob(f"book_{book_id}_*"),
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
    book_dir = book_export_dir(export_dir, book_id)
    if filename:
        sanitized = Path(filename).name
        if not sanitized.startswith(f"book_{book_id}_"):
            raise HTTPException(status_code=400, detail="Filename non valido per questo libro.")
        # Resolve inside the per-book folder to keep exports scoped per user/book.
        path = (book_dir / sanitized).resolve()
        base = book_dir.resolve()
        if not str(path).startswith(str(base)) or not path.exists():
            raise HTTPException(status_code=404, detail="File richiesto non trovato.")
    else:
        path = resolve_export_path(book_id, run_id, kind, export_dir)
    return FileResponse(path, filename=path.name)


def delete_exports(book_id: int, export_dir: Path, run_id: Optional[str] = None) -> List[str]:
    removed: List[str] = []
    book_dir = book_export_dir(export_dir, book_id)
    pattern = f"book_{book_id}_*"
    if run_id:
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise HTTPException(status_code=400, detail="run_id non valido.")
        pattern = f"book_{book_id}_{run_id}*"
    for artifact in book_dir.glob(pattern):
        artifact.unlink(missing_ok=True)
        removed.append(artifact.name)
    if book_dir.exists() and not any(book_dir.iterdir()):
        book_dir.rmdir()
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

    args = parse_worker_args(sys.argv[1:])
    job_id_raw = args["job_id"] or WORKER_JOB_ID
    central_url = args["central_url"] or CENTRAL_BASE_URL
    worker_token = args["worker_token"] or WORKER_SHARED_TOKEN
    heartbeat_raw = args["heartbeat"]
    heartbeat_interval = (
        int(heartbeat_raw) if heartbeat_raw is not None else WORKER_HEARTBEAT_SECONDS
    )

    # If a job id is provided, run once and exit; otherwise start the API server.
    if job_id_raw:
        try:
            job_id = int(job_id_raw)
        except ValueError:
            raise SystemExit(f"Job ID non valido: {job_id_raw}")
        try:
            exit_code = run_worker_job(job_id, central_url, worker_token, heartbeat_interval)
        except Exception as exc:
            logging.error("Worker job failed: %s", exc)
            exit_code = 1
        raise SystemExit(exit_code)

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=False)
