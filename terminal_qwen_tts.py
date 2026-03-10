#!/usr/bin/env python3
"""Interactive terminal TTS tester for the Qwen worker (no server required)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from core import clean_line, gen_audio_segments, get_nlp, load_tts_resources, write_audio_stream


@dataclass
class SessionOptions:
    speaker: str
    instruct: Optional[str]
    language_id: Optional[str]
    speed: float
    temperature: Optional[float]
    top_p: Optional[float]
    repetition_penalty: Optional[float]
    top_k: Optional[int]
    play_audio: bool


def parse_opt_float(raw: str, name: str) -> Optional[float]:
    text = (raw or "").strip()
    if not text or text.lower() in {"none", "default", "auto"}:
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"{name} deve essere un numero o 'none'.") from exc


def parse_opt_int(raw: str, name: str) -> Optional[int]:
    text = (raw or "").strip()
    if not text or text.lower() in {"none", "default", "auto"}:
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"{name} deve essere un intero o 'none'.") from exc


def choose_player() -> Optional[list[str]]:
    candidates = [
        ["afplay"],  # macOS
        ["ffplay", "-nodisp", "-autoexit"],  # ffmpeg
        ["aplay"],  # ALSA
        ["paplay"],  # PulseAudio
    ]
    for candidate in candidates:
        if shutil.which(candidate[0]):
            return candidate
    return None


def play_file(path: Path, player_cmd: Optional[list[str]]) -> None:
    if not player_cmd:
        return
    cmd = [*player_cmd, str(path)]
    subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def normalize_text(raw: str) -> str:
    parts = []
    for line in (raw or "").splitlines():
        cleaned = clean_line(line)
        if cleaned:
            parts.append(cleaned)
    text = " ".join(parts).strip()
    return text


def print_help() -> None:
    print(
        "\nComandi:\n"
        "  /help                          Mostra questo aiuto\n"
        "  /show                          Mostra impostazioni correnti\n"
        "  /speaker <nome>                Cambia speaker (es: Serena, Ryan)\n"
        "  /instruct <testo>              Imposta instruct\n"
        "  /instruct none                 Rimuove instruct\n"
        "  /lang <codice|none>            Imposta language_id (es: it, en, none)\n"
        "  /speed <numero>                Imposta speed\n"
        "  /temperature <num|none>        Override temperature\n"
        "  /top_p <num|none>              Override top_p\n"
        "  /repetition <num|none>         Override repetition_penalty\n"
        "  /top_k <int|none>              Override top_k\n"
        "  /play on|off                   Attiva/disattiva riproduzione audio\n"
        "  /quit                          Esce\n"
    )


def print_state(state: SessionOptions, player_cmd: Optional[list[str]], out_dir: Path) -> None:
    print(
        "\nStato:\n"
        f"  speaker={state.speaker}\n"
        f"  instruct={state.instruct or '(none)'}\n"
        f"  language_id={state.language_id or '(none->Auto)'}\n"
        f"  speed={state.speed}\n"
        f"  temperature={state.temperature}\n"
        f"  top_p={state.top_p}\n"
        f"  repetition_penalty={state.repetition_penalty}\n"
        f"  top_k={state.top_k}\n"
        f"  play_audio={state.play_audio}\n"
        f"  output_dir={out_dir}\n"
        f"  player={' '.join(player_cmd) if player_cmd else '(nessuno)'}\n"
    )


def apply_command(command: str, state: SessionOptions) -> bool:
    # Returns True to continue, False to quit.
    text = (command or "").strip()
    if not text:
        return True
    if text == "/quit":
        return False
    if text == "/help":
        print_help()
        return True
    if text == "/show":
        return True
    if text.startswith("/speaker "):
        value = text[len("/speaker ") :].strip()
        if value:
            state.speaker = value
        return True
    if text.startswith("/instruct "):
        value = text[len("/instruct ") :].strip()
        state.instruct = None if value.lower() in {"none", "default", ""} else value
        return True
    if text.startswith("/lang "):
        value = text[len("/lang ") :].strip()
        state.language_id = None if value.lower() in {"none", "default", "auto", ""} else value
        return True
    if text.startswith("/speed "):
        value = text[len("/speed ") :].strip()
        state.speed = float(value)
        return True
    if text.startswith("/temperature "):
        value = text[len("/temperature ") :].strip()
        state.temperature = parse_opt_float(value, "temperature")
        return True
    if text.startswith("/top_p "):
        value = text[len("/top_p ") :].strip()
        state.top_p = parse_opt_float(value, "top_p")
        return True
    if text.startswith("/repetition "):
        value = text[len("/repetition ") :].strip()
        state.repetition_penalty = parse_opt_float(value, "repetition_penalty")
        return True
    if text.startswith("/top_k "):
        value = text[len("/top_k ") :].strip()
        state.top_k = parse_opt_int(value, "top_k")
        return True
    if text.startswith("/play "):
        value = text[len("/play ") :].strip().lower()
        if value in {"on", "1", "true", "yes"}:
            state.play_audio = True
        elif value in {"off", "0", "false", "no"}:
            state.play_audio = False
        else:
            raise ValueError("/play accetta solo on|off.")
        return True
    raise ValueError("Comando non riconosciuto. Usa /help.")


def synthesize_once(
    text: str,
    output_path: Path,
    state: SessionOptions,
    tts_resources: dict,
    nlp,
) -> None:
    run_resources = dict(tts_resources)
    run_resources["speaker"] = state.speaker
    run_resources["instruct"] = state.instruct or None
    chunks = gen_audio_segments(
        run_resources,
        nlp,
        text,
        speed=state.speed,
        stats=None,
        max_sentences=None,
        post_event=None,
        should_stop=lambda: False,
        repetition_penalty=state.repetition_penalty,
        min_p=None,
        top_p=state.top_p,
        top_k=state.top_k,
        exaggeration=None,
        cfg_weight=None,
        temperature=state.temperature,
        use_multilingual=False,
        language_id=state.language_id,
        audio_prompt_wav=None,
        sentence_gap_ms=0,
        question_gap_ms=0,
    )
    frames = write_audio_stream(output_path, chunks)
    if frames <= 0:
        raise RuntimeError("Sintesi fallita: nessun frame audio.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Terminale TTS interattivo (Qwen worker local).")
    parser.add_argument("--speaker", default="Serena", help="Speaker iniziale.")
    parser.add_argument("--instruct", default="", help="Instruct iniziale (opzionale).")
    parser.add_argument("--language-id", default="it", help="Language id iniziale (es. it, en, auto).")
    parser.add_argument("--speed", type=float, default=1.0, help="Velocita iniziale.")
    parser.add_argument("--temperature", default="none", help="Override iniziale temperature o 'none'.")
    parser.add_argument("--top-p", default="none", help="Override iniziale top_p o 'none'.")
    parser.add_argument(
        "--repetition-penalty",
        default="none",
        help="Override iniziale repetition_penalty o 'none'.",
    )
    parser.add_argument("--top-k", default="none", help="Override iniziale top_k o 'none'.")
    parser.add_argument("--output-dir", default="audioProve/terminal", help="Cartella output wav.")
    parser.add_argument("--no-play", action="store_true", help="Non riprodurre automaticamente l'audio.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    player_cmd = choose_player()
    if not player_cmd and not args.no_play:
        print("Nessun player trovato (afplay/ffplay/aplay/paplay). Continuero senza play automatico.")

    state = SessionOptions(
        speaker=(args.speaker or "Serena").strip() or "Serena",
        instruct=(args.instruct or "").strip() or None,
        language_id=None
        if str(args.language_id).strip().lower() in {"", "none", "auto", "default"}
        else str(args.language_id).strip(),
        speed=float(args.speed),
        temperature=parse_opt_float(str(args.temperature), "temperature"),
        top_p=parse_opt_float(str(args.top_p), "top_p"),
        repetition_penalty=parse_opt_float(str(args.repetition_penalty), "repetition_penalty"),
        top_k=parse_opt_int(str(args.top_k), "top_k"),
        play_audio=(not args.no_play),
    )

    print("Carico il modello TTS (prima volta puo richiedere un po)...")
    tts_resources = load_tts_resources(use_multilingual=False, cache=False)
    nlp = get_nlp()
    print("Pronto. Scrivi una frase e premi invio.")
    print("Comandi rapidi: /help, /show, /instruct ..., /speaker ..., /quit")
    print_state(state, player_cmd, out_dir)

    while True:
        try:
            raw = input("tts> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nUscita.")
            break
        if not raw:
            continue
        if raw.startswith("/"):
            try:
                keep_going = apply_command(raw, state)
            except Exception as exc:
                print(f"Errore comando: {exc}")
                continue
            if raw == "/show":
                print_state(state, player_cmd, out_dir)
            if not keep_going:
                print("Uscita.")
                break
            continue

        text = normalize_text(raw)
        if not text:
            print("Testo vuoto dopo pulizia.")
            continue
        if not text.endswith((".", "!", "?")):
            text += "."
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = out_dir / f"terminal_{stamp}.wav"
        try:
            synthesize_once(text, output_path, state, tts_resources, nlp)
        except Exception as exc:
            print(f"Errore sintesi: {exc}")
            continue
        print(f"OK -> {output_path}")
        if state.play_audio:
            play_file(output_path, player_cmd)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrotto.")
        sys.exit(130)
