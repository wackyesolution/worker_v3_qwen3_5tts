#!/usr/bin/env python3
"""Minimal FastAPI backend that mirrors the Gradio pipeline in API form."""

from __future__ import annotations

import os
import re
import secrets
import shutil
import sqlite3
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from collections import deque
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
DB_PATH = BACKEND_ROOT / "backend.db"
USERS_ROOT = BACKEND_ROOT / "users"
LOGS_DIR = BACKEND_ROOT / "logs"
PRE_SERVER = PROJECT_ROOT / "preServer.py"
COLLECTION_DIR = PROJECT_ROOT / "audioBook"

SUPPORTED_EXTS = {".pdf", ".epub"}
RUN_ID_PATTERN = re.compile(r"\d{8}_\d{6}$")
ARTIFACT_KINDS = {
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

app = FastAPI(title="Chatterblez Backend", version="1.0.0")
security = HTTPBasic()
service_lock = threading.Lock()
CURRENT_JOB: Optional[Dict[str, Any]] = None
CURRENT_JOB_PROCESS: Optional[subprocess.Popen] = None


def ensure_backend_layout() -> None:
    USERS_ROOT.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


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


def tail_lines(path: Path, limit: int = 30) -> List[str]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            last_lines = deque(handle, maxlen=limit)
            return [line.rstrip("\n") for line in last_lines]
    except FileNotFoundError:
        return []


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


def run_pipeline(book: sqlite3.Row, user: Dict) -> Dict:
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
    if retcode != 0:
        log_excerpt = tail_lines(log_path, 30)
        raise HTTPException(
            status_code=500,
            detail="Errore durante il job:\n" + "\n".join(log_excerpt),
        )

    final_folder = COLLECTION_DIR / slug
    wav = final_folder / f"{slug}_{run_id}.wav"
    srt = final_folder / f"{slug}_{run_id}.srt"
    vtt = final_folder / f"{slug}_{run_id}.vtt"
    ebook_copy = final_folder / f"{slug}_{run_id}{import_path.suffix.lower()}"

    missing = [
        path.name
        for path in (wav, srt, vtt)
        if not path.exists()
    ]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"File mancanti dopo la conversione: {', '.join(missing)}",
        )

    export_dir = user["folders"]["export"]
    artifacts = []
    for source in (wav, srt, vtt, ebook_copy if ebook_copy.exists() else None):
        if not source:
            continue
        suffix = source.suffix.lower()
        kind = ARTIFACT_KINDS.get(suffix)
        if not kind:
            continue
        destination = export_dir / f"book_{book['id']}_{run_id}{suffix}"
        shutil.copy2(source, destination)
        artifacts.append({"kind": kind, "path": str(destination)})

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

    log_excerpt = tail_lines(log_path, 30)
    return {
        "run_id": run_id,
        "artifacts": artifacts,
        "stdout": log_excerpt,
        "log_path": str(log_path),
    }


@app.post("/books/{book_id}/process")
def process_book(book_id: int, user: Dict = Depends(get_current_user)):
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
        job_state = run_pipeline(book, user)
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
    user: Dict = Depends(get_current_user),
):
    kind = kind.lower()
    require_book_owner(book_id, user["id"])
    export_dir = user["folders"]["export"]
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


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=False)
