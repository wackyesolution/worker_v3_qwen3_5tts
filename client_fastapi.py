#!/usr/bin/env python3
"""Simple terminal client for the FastAPI backend."""

from __future__ import annotations

import argparse
import getpass
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import requests
from requests.auth import HTTPBasicAuth


class BackendClient:
    def __init__(self, base_url: str, username: str, password: str):
        self.base_url = base_url.rstrip("/")
        self.auth = HTTPBasicAuth(username, password)

    def _request(self, method: str, path: str, **kwargs):
        url = f"{self.base_url}{path}"
        resp = requests.request(method, url, auth=self.auth, timeout=300, **kwargs)
        if resp.status_code >= 400:
            detail = resp.text
            try:
                detail = resp.json()
            except Exception:
                pass
            raise RuntimeError(f"Errore API ({resp.status_code}): {detail}")
        if resp.headers.get("content-type", "").startswith("application/json"):
            return resp.json()
        return resp

    def check_status(self) -> Dict:
        return self._request("GET", "/status")

    def list_books(self) -> Dict:
        return self._request("GET", "/books")

    def upload_book(self, title: str, file_path: Path) -> Dict:
        files = {"file": (file_path.name, file_path.open("rb"), "application/octet-stream")}
        data = {"title": title, "replace_existing": "false"}
        try:
            return self._request("POST", "/books", data=data, files=files)
        finally:
            files["file"][1].close()

    def process_book(self, book_id: int) -> Dict:
        return self._request("POST", f"/books/{book_id}/process")

    def delete_book(self, book_id: int, delete_exports: bool = False) -> Dict:
        params = {"delete_exports": str(delete_exports).lower()}
        return self._request("DELETE", f"/books/{book_id}", params=params)

    def delete_exports(self, book_id: int, run_id: Optional[str] = None) -> Dict:
        params = {}
        if run_id:
            params["run_id"] = run_id
        return self._request("DELETE", f"/books/{book_id}/exports", params=params)

    def list_exports(self, book_id: int) -> Dict:
        return self._request("GET", f"/books/{book_id}/exports")

    def download_artifact(
        self,
        book_id: int,
        kind: str,
        destination: Path,
        run_id: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> Path:
        params = {"kind": kind}
        if run_id:
            params["run_id"] = run_id
        if filename:
            params["filename"] = filename
        resp = self._request("GET", f"/books/{book_id}/download", params=params, stream=True)
        filename = resp.headers.get("content-disposition")
        if filename and "filename=" in filename:
            suggestion = filename.split("filename=")[-1].strip().strip('"')
            destination = destination / suggestion if destination.is_dir() else destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    handle.write(chunk)
        return destination

    def current_job(self, lines: int = 30) -> Dict:
        params = {"lines": lines}
        return self._request("GET", "/jobs/current", params=params)


def prompt(prompt_text: str) -> str:
    try:
        return input(prompt_text)
    except EOFError:
        print()
        sys.exit(0)


def choose_from_list(items: List[Dict], description: str) -> Optional[Dict]:
    if not items:
        print("Nessun elemento disponibile.")
        return None
    for idx, item in enumerate(items, start=1):
        print(f"[{idx}] {item['title']} (ID {item['id']}) - run: {item.get('processed_runs', 0)}")
    choice = prompt(f"Seleziona {description} (0 per tornare indietro): ")
    if not choice.isdigit() or not (0 <= int(choice) <= len(items)):
        print("Selezione non valida.")
        return None
    idx = int(choice)
    if idx == 0:
        return None
    return items[idx - 1]


def handle_list_books(client: BackendClient) -> Optional[Dict]:
    payload = client.list_books()
    items = payload.get("items", [])
    print("\n=== Libri caricati ===")
    print(f"Totale: {payload.get('total_books', 0)} | Elaborati: {payload.get('processed_books', 0)}")
    selection = choose_from_list(items, "un libro")
    return selection


def handle_upload(client: BackendClient) -> Optional[Dict]:
    raw_path = prompt("Trascina qui il file EPUB/PDF e premi invio: ").strip().strip('"')
    file_path = Path(raw_path).expanduser()
    if not file_path.exists():
        print("Percorso non valido.")
        return None
    default_title = file_path.stem
    title = prompt(f"Titolo del libro [{default_title}]: ").strip() or default_title
    print("Caricamento in corso...")
    result = client.upload_book(title, file_path)
    print(f"Libro caricato: {result['title']} (ID {result['book_id']})")
    return {"id": result["book_id"], "title": result["title"], "processed_runs": 0}


def handle_delete(client: BackendClient, book: Dict) -> bool:
    print("1) Elimina libro sorgente")
    print("2) Elimina file elaborati")
    choice = prompt("Scegli: ").strip()
    if choice == "1":
        confirm = prompt(f"Confermi eliminazione del libro '{book['title']}'? [y/N]: ").lower()
        if confirm == "y":
            client.delete_book(book["id"])
            print("Libro eliminato.")
            return True
    elif choice == "2":
        exports = client.list_exports(book["id"])["exports"]
        if not exports:
            print("Nessun file elaborato disponibile.")
            return False
        unique_runs = sorted({entry.get("run_id") for entry in exports if entry.get("run_id")})
        if unique_runs:
            print("Run disponibili:")
            for idx, run_id in enumerate(unique_runs, start=1):
                print(f"[{idx}] {run_id}")
            print("[0] Tutti")
            selection = prompt("Seleziona run da eliminare: ").strip()
            if selection.isdigit() and int(selection) == 0:
                run_id = None
            elif selection.isdigit() and 1 <= int(selection) <= len(unique_runs):
                run_id = unique_runs[int(selection) - 1]
            else:
                print("Selezione non valida.")
                return False
        else:
            run_id = None
        removed = client.delete_exports(book["id"], run_id=run_id)
        print(f"File rimossi: {removed.get('removed', [])}")
    else:
        print("Scelta non valida.")
    return False


def handle_process(client: BackendClient, book: Dict) -> None:
    confirm = prompt(f"Avviare elaborazione per '{book['title']}'? [y/N]: ").lower()
    if confirm != "y":
        return
    print("Elaborazione in corso (attendi completamento Whisper/Chatterblez)...")
    result = client.process_book(book["id"])
    details = result.get("details", {})
    print(f"Run completato: {details.get('run_id')}")
    if details.get("artifacts"):
        print("File creati:")
        print(json.dumps(details["artifacts"], indent=2, ensure_ascii=False))


def handle_download(client: BackendClient, book: Dict, download_dir: Path) -> None:
    exports_payload = client.list_exports(book["id"])
    exports = exports_payload.get("exports", [])
    if not exports:
        print("Nessun file da scaricare.")
        return
    print("File disponibili:")
    for idx, item in enumerate(exports, start=1):
        label = f"{item['kind']} ({item['name']})"
        run_id = item.get("run_id") or "-"
        print(f"[{idx}] {label} - run {run_id}")
    selection = prompt("Seleziona file da scaricare (0 per annullare): ").strip()
    if not selection.isdigit() or not (0 <= int(selection) <= len(exports)):
        print("Selezione non valida.")
        return
    idx = int(selection)
    if idx == 0:
        return
    entry = exports[idx - 1]
    run_id = entry.get("run_id") or None
    destination = download_dir / entry["name"]
    path = client.download_artifact(
        book["id"],
        entry["kind"],
        destination,
        run_id=run_id,
        filename=entry["name"],
    )
    print(f"Scaricato in: {path}")


def show_job_status(client: BackendClient, target_book_id: Optional[int] = None) -> None:
    try:
        status = client.current_job()
    except Exception as exc:
        print(f"Errore nel recupero dello stato: {exc}")
        return
    if not status.get("running"):
        print("Nessun job in esecuzione.")
        return
    if target_book_id and status.get("book_id") != target_book_id:
        print("Questo libro non è in elaborazione al momento.")
        return
    print(
        f"Job attivo: libro #{status.get('book_id')} '{status.get('book_title')}' "
        f"(run {status.get('run_id')})"
    )
    print("Ultime righe di log:")
    for line in status.get("log_tail", []):
        print(line)


def book_menu(client: BackendClient, book: Dict, download_dir: Path) -> None:
    while True:
        print(f"\n=== Libro selezionato: {book['title']} (ID {book['id']}) ===")
        print("1) Elimina")
        print("2) Elabora audio")
        print("3) Scarica file")
        print("4) Indietro")
        print("5) Mostra log job (se attivo)")
        choice = prompt("Scegli un'opzione: ").strip()
        if choice == "1":
            removed = handle_delete(client, book)
            if removed:
                break
        elif choice == "2":
            handle_process(client, book)
        elif choice == "3":
            handle_download(client, book, download_dir)
        elif choice == "4":
            break
        elif choice == "5":
            show_job_status(client, target_book_id=book["id"])
        else:
            print("Opzione non valida.")


def main():
    parser = argparse.ArgumentParser(description="Client CLI per il backend FastAPI.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="URL del backend FastAPI.")
    parser.add_argument("--username", default="admin")
    parser.add_argument("--password")
    parser.add_argument("--download-dir", default=str(Path.cwd()), help="Cartella dove salvare i file scaricati.")
    args = parser.parse_args()

    password = args.password or getpass.getpass(f"Password per {args.username}: ")
    client = BackendClient(args.base_url, args.username, password)

    try:
        status_payload = client.check_status()
        print(f"Connessione OK. Stato servizio: {status_payload}")
    except Exception as exc:
        print(f"Impossibile connettersi al backend: {exc}")
        sys.exit(1)

    download_dir = Path(args.download_dir).expanduser()

    while True:
        print("\n=== Menu principale ===")
        print("1) Visualizza libri caricati")
        print("2) Carica libro")
        print("3) Stato job corrente")
        print("4) Esci")
        choice = prompt("Scegli un'opzione: ").strip()
        if choice == "1":
            selected = handle_list_books(client)
            if selected:
                book_menu(client, selected, download_dir)
        elif choice == "2":
            uploaded = handle_upload(client)
            if uploaded:
                book_menu(client, uploaded, download_dir)
        elif choice == "3":
            show_job_status(client)
        elif choice == "4":
            print("Alla prossima!")
            break
        else:
            print("Opzione non valida.")


if __name__ == "__main__":
    main()
