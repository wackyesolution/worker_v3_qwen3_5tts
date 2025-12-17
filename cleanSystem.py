#!/usr/bin/env python3
"""Utility script to wipe backend storage (DB, logs, user folders, artifacts)."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from backend.main import (
    COLLECTION_DIR,
    DB_PATH,
    LOGS_DIR,
    USERS_ROOT,
    ensure_backend_layout,
    init_db,
)

PROJECT_ROOT = Path(__file__).resolve().parent
REMOTE_UPLOADS = PROJECT_ROOT / "remote_uploads"


def remove_path(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file() or path.is_symlink():
        path.unlink(missing_ok=True)
    else:
        shutil.rmtree(path, ignore_errors=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset backend state (database, logs, users, artifacts).")
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompt.",
    )
    args = parser.parse_args()

    targets = [
        ("database", DB_PATH),
        ("user exports/imports", USERS_ROOT),
        ("backend logs", LOGS_DIR),
        ("audiobook collection", COLLECTION_DIR),
        ("remote uploads", REMOTE_UPLOADS),
    ]

    print("The following paths will be deleted and recreated:")
    for label, path in targets:
        print(f"- {label}: {path}")

    if not args.yes:
        confirm = input("Proceed with cleanup? [y/N]: ").strip().lower()
        if confirm != "y":
            print("Cleanup aborted.")
            return

    for label, path in targets:
        if remove_path(Path(path)):
            print(f"Removed {label} ({path})")
        else:
            print(f"{label} ({path}) already clean.")

    ensure_backend_layout()
    COLLECTION_DIR.mkdir(parents=True, exist_ok=True)
    REMOTE_UPLOADS.mkdir(parents=True, exist_ok=True)
    DB_PATH.unlink(missing_ok=True)
    init_db()

    print("System cleaned. Fresh database and folders are ready.")


if __name__ == "__main__":
    main()
