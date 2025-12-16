#!/usr/bin/env python3
"""
Interactive helper to launch Chatterblez conversions or quick voice tests.

It assumes the following project layout (already used in your workspace):
* DD_book/   -> contains PDF/EPUB inputs
* DD_timbro/ -> contains WAV conditioning prompts
* DD_Output/ -> receives generated audio
"""
from pathlib import Path
import sys
import time

from core import main as core_main, sample_rate
from pick import pick
import soundfile as sf
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

BOOK_DIR = Path("DD_book")
TIMBRE_DIR = Path("DD_timbro")
OUTPUT_DIR = Path("DD_Output")

# Shared presets built from existing CLI parameters
VOICE_PROFILES = {
    "Rilassato / Emotivo (consigliato)": {
        "speed": 0.88,
        "params": dict(
            repetition_penalty=1.05,
            min_p=0.02,
            top_p=0.92,
            exaggeration=0.72,
            cfg_weight=0.32,
            temperature=0.92,
        ),
    },
    "Bilanciato (default Chatterbox)": {
        "speed": 1.0,
        "params": dict(
            repetition_penalty=1.1,
            min_p=0.02,
            top_p=0.95,
            exaggeration=0.5,
            cfg_weight=0.5,
            temperature=0.85,
        ),
    },
    "Energetico / Rapido": {
        "speed": 1.08,
        "params": dict(
            repetition_penalty=1.15,
            min_p=0.03,
            top_p=0.97,
            exaggeration=0.55,
            cfg_weight=0.65,
            temperature=0.8,
        ),
    },
}
DEFAULT_LANGUAGE = "it"


def ensure_directories() -> None:
    """Make sure the expected directories exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not BOOK_DIR.exists():
        raise SystemExit(f"Cartella libri non trovata: {BOOK_DIR.resolve()}")
    if not TIMBRE_DIR.exists():
        raise SystemExit(f"Cartella timbri non trovata: {TIMBRE_DIR.resolve()}")


def list_files(directory: Path, allowed_exts) -> list[Path]:
    return sorted(
        [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in allowed_exts]
    )


def prompt_path(paths: list[Path], title: str, allow_none: bool = False) -> Path | None:
    """Use pick() to select a file path."""
    if not paths and not allow_none:
        raise SystemExit(f"Nessun elemento disponibile per {title}")
    options = [p.name for p in paths]
    mapping = list(range(len(options)))
    if allow_none:
        options = ["[Voce predefinita – nessun timbro]"] + options
        mapping = [None] + mapping
    option, idx = pick(options, title)
    mapped = mapping[idx]
    if mapped is None:
        return None
    return paths[mapped]


def choose_profile():
    labels = list(VOICE_PROFILES.keys())
    selection, idx = pick(labels, "Scegli il profilo di voce")
    profile = VOICE_PROFILES[selection]
    return profile["speed"], profile["params"], selection


def convert_book():
    """Interactive flow to convert a book using core.main."""
    books = list_files(BOOK_DIR, (".pdf", ".epub"))
    book_path = prompt_path(books, "Seleziona il libro da convertire")

    timbres = list_files(TIMBRE_DIR, (".wav",))
    timbre_path = prompt_path(timbres, "Seleziona il timbro vocale (WAV)", allow_none=True)
    speed, params, profile_name = choose_profile()

    print(f"\n➡️  Libro: {book_path}")
    if timbre_path:
        print(f"🎙️  Timbro: {timbre_path}")
    else:
        print("🎙️  Timbro: voce predefinita del modello")
    print(f"🎚️  Profilo voce: {profile_name} (speed={speed})")
    print(f"💾 Output: {OUTPUT_DIR.resolve()}\n")

    core_main(
        file_path=str(book_path),
        pick_manually=False,
        speed=speed,
        output_folder=str(OUTPUT_DIR),
        audio_prompt_wav=str(timbre_path) if timbre_path else None,
        ignore_list=None,
        use_multilingual=True,
        language_id=DEFAULT_LANGUAGE,
        **params,
    )


def run_voice_test():
    """Generate ~30s preview audio before launching a long conversion."""
    timbres = list_files(TIMBRE_DIR, (".wav",))
    timbre_path = prompt_path(timbres, "Scegli il timbro da testare", allow_none=True)
    speed, params, profile_name = choose_profile()

    # 30-second Italian sample text (roughly)
    sample_paragraph = (
        "Ciao, sono Daniel, e questa è una prova di trenta secondi per verificare il timbro. "
        "Immagina che io stia leggendo il tuo libro preferito con calma, ritmo naturale e dizione chiara. "
        "Ascolta bene come vengono pronunciate le vocali, come varia l'intonazione e come il tono rimane costante. "
        "Se questo suono ti convince possiamo iniziare subito la generazione completa dell'audiolibro. "
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = ChatterboxMultilingualTTS.from_pretrained(device=device)
    if timbre_path:
        try:
            tts.prepare_conditionals(wav_fpath=str(timbre_path))
        except AttributeError:
            pass

    print("🎧 Genero il test audio, attendi qualche secondo...")
    wav = tts.generate(
        sample_paragraph,
        language_id=DEFAULT_LANGUAGE,
        audio_prompt_path=str(timbre_path) if timbre_path else None,
        **params,
    )
    audio = wav.numpy().flatten()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    timbre_tag = timbre_path.stem if timbre_path else "default"
    output_path = OUTPUT_DIR / f"voice_test_{timbre_tag}_{profile_name.replace(' ', '_')}_{timestamp}.wav"
    sf.write(output_path, audio, sample_rate)

    print(f"✅ Test salvato: {output_path.resolve()}")
    print("Riascolta il file per decidere se procedere con il libro completo.\n")


def main():
    ensure_directories()

    menu_options = [
        ("Converti libro da DD_book", convert_book),
        ("Test timbro (circa 30 secondi)", run_voice_test),
        ("Esci", lambda: sys.exit(0)),
    ]

    while True:
        labels = [label for label, _ in menu_options]
        selection, idx = pick(labels, "Ciao! Cosa vuoi fare?")
        _, action = menu_options[idx]
        action()


if __name__ == "__main__":
    main()
