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
import time
from datetime import datetime
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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

# Make root-level modules (e.g. core.py) importable regardless of PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import (
    find_document_chapters_and_extract_texts,
    find_good_chapters,
    extract_pdf_chapters,
    extract_txt_chapters,
    chapter_beginning_one_liner,
    get_nlp,
    gen_audio_segments,
    write_audio_stream,
    clean_line,
    load_tts_resources,
    get_tts_engine_name,
)


BACKEND_ROOT = PROJECT_ROOT / "backend"
# Local sqlite is only to satisfy the legacy pipeline; Central is the source of truth.
DB_PATH = BACKEND_ROOT / "backend.db"
USERS_ROOT = BACKEND_ROOT / "users"
LOGS_DIR = BACKEND_ROOT / "logs"
PRE_SERVER = PROJECT_ROOT / "preServer.py"
AUDIO_PROVE_DIR = PROJECT_ROOT / "audioProve"
MAX_LOG_LINES = 10000

SUPPORTED_EXTS = {".pdf", ".epub", ".txt"}
RUN_ID_PATTERN = re.compile(r"\d{8}_\d{6}(?:_preview)?$")
CHUNK_PROGRESS_RE = re.compile(r"CHUNK_PROGRESS current=(\d+) total=(\d+) remaining=(\d+)")
CHAPTER_PROGRESS_RE = re.compile(r"CHAPTER_PROGRESS current=(\d+) total=(\d+) remaining=(\d+)")
CHAPTER_CHUNK_PLAN_RE = re.compile(r"CHAPTER_CHUNK_PLAN chapter=(\d+) total=(\d+)")
GLOBAL_CHUNK_PLAN_RE = re.compile(r"GLOBAL_CHUNK_PLAN total=(\d+) chapters=(\d+)")
SPLIT_BATCH_RE = re.compile(r"Split\s+\d+\s+sentences\s+into\s+(\d+)\s+batches", re.IGNORECASE)
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
TRIAL_MODE = (os.getenv("TRIAL_MODE", "0") or "0").strip().lower() in {"1", "true", "yes", "y", "on"}
WHISPER_MODEL = os.getenv("WHISPER_MODEL", os.getenv("WORKER_WHISPER_MODEL", "small")).strip() or "small"
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "it").strip() or "it"
DEFAULT_PREVIEW_MIN_WORDS = max(1, int(os.getenv("DEFAULT_PREVIEW_MIN_WORDS", "2000")))

# FastAPI only matters in standalone mode; worker jobs exit after processing.
app = FastAPI(title="Chatterblez Backend", version="1.0.0")
security = HTTPBasic()
service_lock = threading.Lock()
preview_lock = threading.Lock()
trial_lock = threading.Lock()
trial_lock_started_at = 0.0
tts_cache: Dict[Tuple[str, bool], Dict[str, Any]] = {}
CURRENT_JOB: Optional[Dict[str, Any]] = None
CURRENT_JOB_PROCESS: Optional[subprocess.Popen] = None

TRIAL_SYNTH_TIMEOUT_SECONDS = int(os.getenv("TRIAL_SYNTH_TIMEOUT_SECONDS", "180"))
LIVE_LOG_LINE_MAX_CHARS = int(os.getenv("WORKER_LIVE_LOG_LINE_MAX_CHARS", "1200"))
LIVE_LOG_BATCH_MAX_LINES = int(os.getenv("WORKER_LIVE_LOG_BATCH_MAX_LINES", "120"))
LIVE_LOG_READ_CHUNK_BYTES = int(os.getenv("WORKER_LIVE_LOG_READ_CHUNK_BYTES", str(256 * 1024)))


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
    preview: Optional[bool] = None
    max_preview_words: Optional[int] = None


class VoiceTestRequest(BaseModel):
    text: str
    repetition_penalty: Optional[float] = None
    min_p: Optional[float] = None
    top_p: Optional[float] = None
    exaggeration: Optional[float] = None
    cfg_weight: Optional[float] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    speed: float = 1.0
    use_multilingual: bool = False
    language_id: Optional[str] = "it"
    sentence_gap_ms: Optional[int] = None
    question_gap_ms: Optional[int] = None
    disable_cleaning: bool = False
    qwen_instruct: Optional[str] = None
    qwen_speaker: Optional[str] = None


class TrialRequest(VoiceTestRequest):
    pass


class HeartbeatState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: Dict[str, Any] = {}
        self._log_lines: List[str] = []

    def update(self, **kwargs: Any) -> None:
        with self._lock:
            for key, value in kwargs.items():
                if value is not None:
                    self._state[key] = value

    def append_log_lines(self, lines: List[str]) -> None:
        if not lines:
            return
        cleaned: List[str] = []
        for line in lines:
            text = str(line or "").rstrip("\r")
            if not text:
                continue
            if len(text) > LIVE_LOG_LINE_MAX_CHARS:
                text = text[: LIVE_LOG_LINE_MAX_CHARS - 1] + "…"
            cleaned.append(text)
        if not cleaned:
            return
        with self._lock:
            self._log_lines.extend(cleaned)
            overflow = len(self._log_lines) - (LIVE_LOG_BATCH_MAX_LINES * 4)
            if overflow > 0:
                self._log_lines = self._log_lines[overflow:]

    def snapshot(
        self,
        *,
        drain_log_lines: bool = False,
        max_log_lines: int = LIVE_LOG_BATCH_MAX_LINES,
    ) -> Dict[str, Any]:
        with self._lock:
            payload = dict(self._state)
            if max_log_lines < 0:
                max_log_lines = 0
            if self._log_lines:
                chunk = self._log_lines[:max_log_lines] if max_log_lines else []
                if drain_log_lines and chunk:
                    self._log_lines = self._log_lines[len(chunk):]
                if chunk:
                    payload["log_lines"] = chunk
            return payload


class ProgressTelemetryTracker:
    """Parse CLI log lines and expose normalized progress telemetry."""

    def __init__(self) -> None:
        self.started_monotonic = time.monotonic()
        self.chapter_current: Optional[int] = None
        self.chapter_total: Optional[int] = None
        self.chapter_chunk_totals: Dict[int, int] = {}
        self.global_chunk_total: Optional[int] = None
        # Local chunk info for the current chapter.
        self.chunk_current: Optional[int] = None
        self.chunk_total: Optional[int] = None
        self.chunks_remaining: Optional[int] = None

    def _sum_known_chunk_totals(self) -> int:
        return int(sum(max(0, int(total or 0)) for total in self.chapter_chunk_totals.values()))

    def _chunks_completed_before_current_chapter(self) -> int:
        if self.chapter_current is None:
            return 0
        current = int(self.chapter_current)
        completed = 0
        for chapter_index, total in self.chapter_chunk_totals.items():
            if int(chapter_index) < current:
                completed += max(0, int(total or 0))
        return int(completed)

    def _resolved_global_chunk_total(self) -> Optional[int]:
        if self.global_chunk_total is not None and int(self.global_chunk_total) > 0:
            return int(self.global_chunk_total)
        known_total = self._sum_known_chunk_totals()
        if known_total > 0:
            return int(known_total)
        if self.chunk_total is not None and int(self.chunk_total) > 0:
            return int(self.chunk_total)
        return None

    def update_from_line(self, line: str) -> Optional[Dict[str, Any]]:
        text = str(line or "")
        changed = False

        global_chunk_match = GLOBAL_CHUNK_PLAN_RE.search(text)
        if global_chunk_match:
            global_chunk_total = max(0, int(global_chunk_match.group(1)))
            if self.global_chunk_total != global_chunk_total:
                changed = True
            self.global_chunk_total = global_chunk_total

        chapter_chunk_plan_match = CHAPTER_CHUNK_PLAN_RE.search(text)
        if chapter_chunk_plan_match:
            chapter_index = max(1, int(chapter_chunk_plan_match.group(1)))
            chapter_chunk_total = max(0, int(chapter_chunk_plan_match.group(2)))
            if self.chapter_chunk_totals.get(chapter_index) != chapter_chunk_total:
                changed = True
            self.chapter_chunk_totals[chapter_index] = chapter_chunk_total

        chapter_match = CHAPTER_PROGRESS_RE.search(text)
        if chapter_match:
            chapter_current = int(chapter_match.group(1))
            chapter_total = max(1, int(chapter_match.group(2)))
            if self.chapter_current != chapter_current or self.chapter_total != chapter_total:
                changed = True
            self.chapter_current = chapter_current
            self.chapter_total = chapter_total
            # New chapter starts from zero progress until first CHUNK_PROGRESS.
            if self.chunk_current != 0:
                changed = True
            self.chunk_current = 0
            planned_for_chapter = self.chapter_chunk_totals.get(chapter_current)
            if planned_for_chapter is not None:
                if self.chunk_total != planned_for_chapter:
                    changed = True
                self.chunk_total = int(planned_for_chapter)
                chapter_remaining = max(0, int(planned_for_chapter))
                if self.chunks_remaining != chapter_remaining:
                    changed = True
                self.chunks_remaining = chapter_remaining
            else:
                if self.chunks_remaining is not None:
                    changed = True
                self.chunks_remaining = None

        split_match = SPLIT_BATCH_RE.search(text)
        if split_match:
            chunk_total = max(1, int(split_match.group(1)))
            if self.chapter_current is not None:
                chapter_idx = int(self.chapter_current)
                if self.chapter_chunk_totals.get(chapter_idx) != chunk_total:
                    changed = True
                self.chapter_chunk_totals[chapter_idx] = chunk_total
            if self.chunk_total != chunk_total:
                changed = True
            self.chunk_total = chunk_total
            if self.chunk_current is None:
                self.chunk_current = 0
            self.chunks_remaining = max(0, chunk_total - int(self.chunk_current or 0))

        chunk_match = CHUNK_PROGRESS_RE.search(text)
        if chunk_match:
            chunk_current = int(chunk_match.group(1))
            chunk_total = max(1, int(chunk_match.group(2)))
            chunks_remaining = max(0, int(chunk_match.group(3)))
            if (
                self.chunk_current != chunk_current
                or self.chunk_total != chunk_total
                or self.chunks_remaining != chunks_remaining
            ):
                changed = True
            self.chunk_current = chunk_current
            self.chunk_total = chunk_total
            self.chunks_remaining = chunks_remaining

        if not changed:
            return None
        return self.snapshot()

    def snapshot(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if self.chapter_current is not None:
            payload["chapter_current"] = int(self.chapter_current)
        if self.chapter_total is not None:
            payload["chapter_total"] = int(self.chapter_total)

        global_chunk_current: Optional[int] = None
        if self.chapter_current is not None:
            global_chunk_current = self._chunks_completed_before_current_chapter()
            if self.chunk_current is not None:
                global_chunk_current += max(0, int(self.chunk_current))
        elif self.chunk_current is not None:
            global_chunk_current = max(0, int(self.chunk_current))
        global_chunk_total = self._resolved_global_chunk_total()

        if global_chunk_current is not None:
            payload["chunk_current"] = int(global_chunk_current)
        if global_chunk_total is not None:
            payload["chunk_total"] = int(global_chunk_total)
        if global_chunk_current is not None and global_chunk_total is not None:
            payload["chunks_remaining"] = max(0, int(global_chunk_total - global_chunk_current))
        elif self.chunks_remaining is not None:
            payload["chunks_remaining"] = int(self.chunks_remaining)

        progress_percent: Optional[float] = None
        if global_chunk_current is not None and global_chunk_total and global_chunk_total > 0:
            progress_percent = (float(global_chunk_current) / float(global_chunk_total)) * 100.0
        elif self.chapter_current is not None and self.chapter_total:
            progress_percent = (float(self.chapter_current - 1) / float(self.chapter_total)) * 100.0

        if progress_percent is not None:
            bounded = max(0.0, min(100.0, progress_percent))
            payload["progress"] = bounded
            payload["progress_percent"] = bounded
            elapsed = max(0.0, time.monotonic() - self.started_monotonic)
            if bounded >= 100.0:
                payload["eta_seconds"] = 0
            elif bounded > 0:
                remaining_seconds = elapsed * ((100.0 - bounded) / bounded)
                payload["eta_seconds"] = int(max(0, round(remaining_seconds)))

        if self.chapter_current is not None and self.chapter_total:
            if global_chunk_current is not None and global_chunk_total:
                payload["message"] = (
                    f"chapter {self.chapter_current}/{self.chapter_total} · "
                    f"chunk {global_chunk_current}/{global_chunk_total}"
                )
            else:
                payload["message"] = f"chapter {self.chapter_current}/{self.chapter_total}"
        elif global_chunk_current is not None and global_chunk_total:
            payload["message"] = f"chunk {global_chunk_current}/{global_chunk_total}"
        return payload


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

    def heartbeat(
        self,
        job_id: int,
        message: Optional[str] = None,
        progress: Optional[float] = None,
        progress_percent: Optional[float] = None,
        chunk_current: Optional[int] = None,
        chunk_total: Optional[int] = None,
        chunks_remaining: Optional[int] = None,
        chapter_current: Optional[int] = None,
        chapter_total: Optional[int] = None,
        eta_seconds: Optional[int] = None,
        log_lines: Optional[List[str]] = None,
    ) -> None:
        payload: Dict[str, Any] = {}
        if message:
            payload["message"] = message
        if progress is not None:
            payload["progress"] = float(progress)
        if progress_percent is not None:
            payload["progress_percent"] = float(progress_percent)
        if chunk_current is not None:
            payload["chunk_current"] = int(chunk_current)
        if chunk_total is not None:
            payload["chunk_total"] = int(chunk_total)
        if chunks_remaining is not None:
            payload["chunks_remaining"] = int(chunks_remaining)
        if chapter_current is not None:
            payload["chapter_current"] = int(chapter_current)
        if chapter_total is not None:
            payload["chapter_total"] = int(chapter_total)
        if eta_seconds is not None:
            payload["eta_seconds"] = int(eta_seconds)
        if log_lines:
            payload["log_lines"] = [str(line) for line in log_lines if str(line or "").strip()]
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


def _append_trial_log(log_path: Optional[Path], message: str) -> None:
    if not log_path:
        return
    timestamp = datetime.utcnow().isoformat(timespec="seconds")
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] {message}\n")
    except Exception:  # noqa: BLE001
        # best-effort: avoid breaking the flow for logging issues
        pass


def synthesize_voice_preview(payload: VoiceTestRequest, user: Dict, log_path: Optional[Path] = None) -> Path:
    text_lines = []
    last_blank = False
    for line in (payload.text or "").splitlines():
        cleaned = line.strip() if payload.disable_cleaning else clean_line(line)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()  # normalizza spazi multipli
        if not cleaned:
            if last_blank:
                continue
            last_blank = True
            continue
        last_blank = False
        text_lines.append(cleaned)
    text = " ".join(text_lines).strip()
    if not text:
        raise HTTPException(status_code=400, detail="Inserisci almeno qualche parola da leggere.")
    if not text.endswith((".", "!", "?")):
        text += "."
    _append_trial_log(
        log_path,
        f"Preview text len={len(text)} disable_cleaning={payload.disable_cleaning} "
        f"use_multilingual={payload.use_multilingual} language_id={payload.language_id or 'it'}",
    )
    qwen_instruct = str(payload.qwen_instruct or "").strip() or None
    qwen_speaker = str(payload.qwen_speaker or "").strip() or None
    slug = slugify(user["username"])
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = AUDIO_PROVE_DIR / f"{slug}_{run_id}.wav"
    _append_trial_log(
        log_path,
        f"Loading TTS resources (multilingual={payload.use_multilingual})...",
    )
    tts_resources = dict(get_tts_resources_cached(payload.use_multilingual))
    if qwen_instruct:
        tts_resources["instruct"] = qwen_instruct
    if qwen_speaker:
        tts_resources["speaker"] = qwen_speaker
    _append_trial_log(
        log_path,
        f"TTS ready (engine={tts_resources.get('engine')})",
    )
    _append_trial_log(log_path, "Loading NLP pipeline...")
    nlp = get_nlp()
    _append_trial_log(log_path, "NLP ready, starting synthesis...")

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
        sentence_gap_ms=payload.sentence_gap_ms if payload.sentence_gap_ms is not None else 0,
        question_gap_ms=payload.question_gap_ms if payload.question_gap_ms is not None else 0,
    )
    _append_trial_log(log_path, "Audio chunks generated, writing stream...")
    frames = write_audio_stream(output_path, chunks)
    if frames <= 0:
        _append_trial_log(log_path, "Nessun frame scritto: sintesi fallita.")
        raise HTTPException(status_code=500, detail="Sintesi non riuscita, nessun audio generato.")
    _append_trial_log(log_path, f"Audio scritto: frames={frames}, path={output_path}")
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
    elif suffix == ".txt":
        chapters = extract_txt_chapters(str(source), row["title"])
        default_selected = {0}
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


def read_new_log_lines(path: Path, cursor: Dict[str, Any]) -> List[str]:
    """
    Read appended log lines from `path`, keeping cursor state in-memory.
    Returns only newly appended complete lines.
    """
    if not path.exists():
        return []
    offset = int(cursor.get("offset") or 0)
    remainder = str(cursor.get("remainder") or "")
    file_size = path.stat().st_size
    if offset < 0 or offset > file_size:
        offset = 0
        remainder = ""
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read(max(1024, LIVE_LOG_READ_CHUNK_BYTES))
    if not data:
        cursor["offset"] = offset
        cursor["remainder"] = remainder
        return []
    new_offset = offset + len(data)
    text = remainder + data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if text and not text.endswith("\n"):
        remainder = lines.pop() if lines else text
    else:
        remainder = ""
    cursor["offset"] = new_offset
    cursor["remainder"] = remainder
    return lines


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
    preview_mode: bool = False,
    max_preview_words: Optional[int] = None,
    on_chunk_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_log_lines: Optional[Callable[[List[str]], None]] = None,
) -> Dict:
    global CURRENT_JOB, CURRENT_JOB_PROCESS
    # This calls PRE_SERVER which handles ffmpeg/whisper/azzurra/csm end-to-end.
    job_history_id: Optional[int] = None
    job_status = "failed"
    job_error: Optional[str] = None
    try:
        source_import_path = Path(book["import_path"])
        import_path = source_import_path
        if not import_path.exists():
            raise HTTPException(status_code=404, detail="File del libro non trovato.")
        output_book_basename = import_path.stem
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
        preview_path: Optional[Path] = None
        if preview_mode:
            try:
                preview_text = ""
                suffix = import_path.suffix.lower()
                chapters = None
                if suffix == ".epub":
                    epub_book = epub.read_epub(str(import_path))
                    chapters = find_document_chapters_and_extract_texts(epub_book)
                elif suffix == ".pdf":
                    chapters = extract_pdf_chapters(str(import_path), book["title"])
                elif suffix == ".txt":
                    chapters = extract_txt_chapters(str(import_path), book["title"])
                if chapters:
                    max_words = max_preview_words or 2200
                    min_words = min(DEFAULT_PREVIEW_MIN_WORDS, max_words)
                    collected_parts: List[str] = []
                    collected_words = 0
                    collected_indices: List[int] = []
                    for chapter_position, chapter in enumerate(chapters):
                        chapter_text = (getattr(chapter, "extracted_text", "") or "").strip()
                        if not chapter_text:
                            continue
                        chapter_words = chapter_text.split()
                        if not chapter_words:
                            continue
                        chapter_index = getattr(chapter, "chapter_index", chapter_position)
                        collected_parts.append(chapter_text)
                        collected_words += len(chapter_words)
                        try:
                            collected_indices.append(int(chapter_index))
                        except (TypeError, ValueError):
                            collected_indices.append(chapter_position)
                        if collected_words >= min_words:
                            break
                    preview_text = "\n\n".join(collected_parts).strip()
                    selected_chapter_indices = collected_indices or selected_chapter_indices
                    logging.info(
                        "Preview mode: using %s chapter(s) %s for %s words (min=%s max=%s)",
                        len(collected_indices),
                        collected_indices,
                        collected_words,
                        min_words,
                        max_words,
                    )
                if preview_text:
                    words = preview_text.split()
                    if max_preview_words and len(words) > max_preview_words:
                        words = words[:max_preview_words]
                        preview_text = " ".join(words)
                        logging.info("Preview mode: truncated to %s words", len(words))
                    preview_path = work_root / f"preview_{run_id}.txt"
                    preview_path.write_text(preview_text, encoding="utf-8")
                    import_path = preview_path
                    output_book_basename = import_path.stem
                    logging.info("Preview mode: preview file created at %s", preview_path)
            except Exception as exc:  # pragma: no cover - best effort
                logging.warning("Preview mode setup failed, proceeding with full book: %s", exc)
                preview_path = None

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
        if WHISPER_MODEL:
            cmd += ["--whisper-model", WHISPER_MODEL]
        if WHISPER_LANGUAGE:
            cmd += ["--whisper-language", WHISPER_LANGUAGE]
        logging.info("Whisper settings -> model=%s, language=%s", WHISPER_MODEL, WHISPER_LANGUAGE)
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
        last_progress_signature: Optional[Tuple[int, int, int, int, int]] = None
        telemetry_tracker = ProgressTelemetryTracker()
        log_cursor: Dict[str, Any] = {"offset": 0, "remainder": ""}
        try:
            with log_path.open("w", encoding="utf-8", buffering=1) as log_handle:
                process = subprocess.Popen(
                    cmd,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                CURRENT_JOB = job_info
                CURRENT_JOB_PROCESS = process
                while True:
                    polled = process.poll()
                    new_lines = read_new_log_lines(log_path, log_cursor)
                    if new_lines and on_log_lines:
                        on_log_lines(new_lines)
                    for line in new_lines:
                        progress_payload = telemetry_tracker.update_from_line(line)
                        if not progress_payload:
                            continue
                        signature = (
                            int(progress_payload.get("chapter_current") or 0),
                            int(progress_payload.get("chapter_total") or 0),
                            int(progress_payload.get("chunk_current") or 0),
                            int(progress_payload.get("chunk_total") or 0),
                            int(round(float(progress_payload.get("progress_percent") or 0.0) * 100)),
                        )
                        if signature == last_progress_signature:
                            continue
                        last_progress_signature = signature
                        if on_chunk_progress:
                            on_chunk_progress(progress_payload)
                    if polled is not None:
                        retcode = int(polled)
                        break
                    time.sleep(0.7)
                trailing_lines = read_new_log_lines(log_path, log_cursor)
                if trailing_lines and on_log_lines:
                    on_log_lines(trailing_lines)
                for line in trailing_lines:
                    progress_payload = telemetry_tracker.update_from_line(line)
                    if progress_payload and on_chunk_progress:
                        on_chunk_progress(progress_payload)
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
        final_folder = collect_dir / output_book_basename

        artifacts: List[Dict[str, str]] = []
        chapter_audio = sorted(final_folder.glob(f"{output_book_basename}_{run_id}_chapter*.m4a"))
        if not chapter_audio:
            raise HTTPException(
                status_code=500,
                detail="Nessun file per capitolo trovato dopo la conversione.",
            )
        chapter_count = len(chapter_audio)
        final_m4a = final_folder / f"{output_book_basename}_{run_id}.m4a"
        final_srt = final_folder / f"{output_book_basename}_{run_id}.srt"
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
        ebook_dest = book_dir / f"book_{book['id']}_{run_id}{source_import_path.suffix.lower()}"
        shutil.copy2(source_import_path, ebook_dest)
        artifacts.append({"kind": "ebook", "path": str(ebook_dest)})

        # Cleanup: drop per-chapter/temp outputs once final files are copied.
        output_dir = output_base / f"{output_book_basename}_{run_id}"
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
            "mode": "per_chapter" if not preview_mode else "preview",
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
        if preview_mode and preview_path:
            try:
                preview_path.unlink(missing_ok=True)
            except Exception:
                pass


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
    state: HeartbeatState,
) -> None:
    pulse_seconds = max(1, min(max(1, interval_seconds), 20))
    while not stop_event.wait(timeout=pulse_seconds):
        try:
            snapshot = state.snapshot(drain_log_lines=True)
            client.heartbeat(
                job_id,
                message=snapshot.get("message"),
                progress=snapshot.get("progress"),
                progress_percent=snapshot.get("progress_percent"),
                chunk_current=snapshot.get("chunk_current"),
                chunk_total=snapshot.get("chunk_total"),
                chunks_remaining=snapshot.get("chunks_remaining"),
                chapter_current=snapshot.get("chapter_current"),
                chapter_total=snapshot.get("chapter_total"),
                eta_seconds=snapshot.get("eta_seconds"),
                log_lines=snapshot.get("log_lines"),
            )
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

    heartbeat_state = HeartbeatState()
    client.heartbeat(job_id, "worker-started")
    heartbeat_state.update(message="worker-running")

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
    preview_mode = bool(options.preview) if options else False
    max_preview_words = options.max_preview_words if options else None
    pipeline_kwargs = build_pipeline_kwargs(options)

    stop_event = threading.Event()
    heartbeat_thread = threading.Thread(
        target=heartbeat_loop,
        args=(client, job_id, stop_event, heartbeat_interval, heartbeat_state),
        daemon=True,
    )
    heartbeat_thread.start()

    status_value = "failed"
    error_message: Optional[str] = None
    artifacts: Optional[List[Dict[str, str]]] = None
    last_progress_push_at = 0.0

    def on_chunk_progress(progress_payload: Dict[str, Any]) -> None:
        nonlocal last_progress_push_at
        heartbeat_state.update(**progress_payload)
        now = time.monotonic()
        if now - last_progress_push_at < 5.0:
            return
        snapshot = heartbeat_state.snapshot(drain_log_lines=True)
        try:
            client.heartbeat(
                job_id,
                message=snapshot.get("message"),
                progress=snapshot.get("progress"),
                progress_percent=snapshot.get("progress_percent"),
                chunk_current=snapshot.get("chunk_current"),
                chunk_total=snapshot.get("chunk_total"),
                chunks_remaining=snapshot.get("chunks_remaining"),
                chapter_current=snapshot.get("chapter_current"),
                chapter_total=snapshot.get("chapter_total"),
                eta_seconds=snapshot.get("eta_seconds"),
                log_lines=snapshot.get("log_lines"),
            )
            last_progress_push_at = now
        except Exception as exc:  # pragma: no cover - best effort
            logging.warning("Chunk progress heartbeat failed for job %s: %s", job_id, exc)

    def on_log_lines(lines: List[str]) -> None:
        heartbeat_state.append_log_lines(lines)

    try:
        result = run_pipeline(
            book_row,
            user,
            run_id_override=run_id,
            preview_mode=preview_mode,
            max_preview_words=max_preview_words or 2200,
            on_chunk_progress=on_chunk_progress,
            on_log_lines=on_log_lines,
            **pipeline_kwargs,
        )
        artifacts = result.get("artifacts") if isinstance(result, dict) else None
        if artifacts:
            for artifact in artifacts:
                artifact_path = Path(artifact["path"])
                client.upload_artifact(job_id, artifact_path)
        status_value = "success"
    except HTTPException as exc:
        logging.error("Worker job %s failed with HTTPException: %s", job_id, exc.detail)
        error_message = str(exc.detail)
    except Exception as exc:
        logging.exception("Worker job %s raised an unhandled exception", job_id)
        error_message = str(exc)
    finally:
        final_snapshot = heartbeat_state.snapshot(drain_log_lines=True)
        if final_snapshot.get("log_lines"):
            try:
                client.heartbeat(
                    job_id,
                    message=final_snapshot.get("message"),
                    progress=final_snapshot.get("progress"),
                    progress_percent=final_snapshot.get("progress_percent"),
                    chunk_current=final_snapshot.get("chunk_current"),
                    chunk_total=final_snapshot.get("chunk_total"),
                    chunks_remaining=final_snapshot.get("chunks_remaining"),
                    chapter_current=final_snapshot.get("chapter_current"),
                    chapter_total=final_snapshot.get("chapter_total"),
                    eta_seconds=final_snapshot.get("eta_seconds"),
                    log_lines=final_snapshot.get("log_lines"),
                )
            except Exception as exc:  # pragma: no cover - best effort
                logging.warning("Final log heartbeat failed for job %s: %s", job_id, exc)
        stop_event.set()
        heartbeat_thread.join(timeout=5)
        log_path = LOGS_DIR / f"book_{book_id}_{run_id}.log"
        if not log_path.exists():
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(
                    f"Job {job_id} failed before log streaming was ready.\nerror={error_message or 'unknown'}\n",
                    encoding="utf-8",
                )
            except Exception as exc:  # pragma: no cover - best effort
                logging.warning("Impossibile creare log minimo per job %s: %s", job_id, exc)
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
    preview_mode = bool(options.preview) if options else False
    max_preview_words = options.max_preview_words if options else None
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
            preview_mode=preview_mode,
            max_preview_words=max_preview_words or 2200,
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
        raise HTTPException(status_code=409, detail="è già in corso una generazione di prova.")
    try:
        audio_path = synthesize_voice_preview(payload, user)
    finally:
        if acquired:
            preview_lock.release()
    return FileResponse(audio_path, filename=audio_path.name)


@app.post("/trial/text")
def trial_text(payload: TrialRequest):
    if not TRIAL_MODE:
        raise HTTPException(status_code=404, detail="Trial mode non attivo su questo worker.")
    global trial_lock_started_at, trial_lock
    acquired = False
    log_path = LOGS_DIR / f"trial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    text_preview = (payload.text or "").replace("\n", " ")[:400]
    _append_trial_log(
        log_path,
        f"Request received (len={len(payload.text or '')}, disable_cleaning={payload.disable_cleaning}) preview=\"{text_preview}\"",
    )
    # Se il lock è bloccato da troppo tempo, resettiamo per evitare stalli.
    max_lock_seconds = 300
    if trial_lock.locked() and trial_lock_started_at:
        elapsed = time.time() - trial_lock_started_at
        if elapsed > max_lock_seconds:
            logger.warning("Trial lock stallo dopo %.1fs, forzo reset del lock.", elapsed)
            trial_lock = threading.Lock()
            trial_lock_started_at = 0.0
    acquired = trial_lock.acquire(blocking=False)
    _append_trial_log(log_path, f"Lock acquired={acquired}")
    if not acquired:
        raise HTTPException(status_code=409, detail="è già in corso una prova.")
    trial_lock_started_at = time.time()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(synthesize_voice_preview, payload, {"username": "trial"}, log_path)
            try:
                audio_path = future.result(timeout=TRIAL_SYNTH_TIMEOUT_SECONDS)
            except TimeoutError:
                _append_trial_log(
                    log_path,
                    f"Timeout dopo {TRIAL_SYNTH_TIMEOUT_SECONDS}s: sintesi ancora in corso, lock rilasciato.",
                )
                raise HTTPException(
                    status_code=504,
                    detail=f"Sintesi di prova timeout dopo {TRIAL_SYNTH_TIMEOUT_SECONDS}s (log {log_path.name}).",
                )
            except Exception as exc:  # noqa: BLE001
                _append_trial_log(log_path, "Errore trial:\n" + traceback.format_exc())
                raise HTTPException(
                    status_code=500,
                    detail=f"Errore trial worker (vedi log {log_path.name}).",
                ) from exc
            _append_trial_log(
                log_path,
                f"Audio generato: {audio_path} (bytes={audio_path.stat().st_size if audio_path.exists() else 0})",
            )
    finally:
        if acquired:
            trial_lock.release()
            trial_lock_started_at = 0.0
    return FileResponse(audio_path, filename=audio_path.name, headers={"X-Trial-Log": str(log_path)})


@app.get("/example/useTest")
def example_use_test():
    sample_payload = {
        "text": "Ciao! Questa è una prova veloce della tua voce virtuale.",
        "speed": 1.0,
        "use_multilingual": False,
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
        except Exception:
            logging.exception("Worker job failed")
            exit_code = 1
        raise SystemExit(exit_code)

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=False)
