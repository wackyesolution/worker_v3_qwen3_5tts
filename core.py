#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# chatterblez - A program to convert e-books into audiobooks using
# qwen3-tts
# by Zachary Erskine
# by Claudio Santini 2025 - https://claudio.uk
import logging
import json
import os
import sys
import importlib.util
from glob import glob

import torch
import spacy
import ebooklib
import soundfile
import numpy as np
import time
import shutil
import subprocess
import platform
import re
from io import StringIO
from types import SimpleNamespace
from tabulate import tabulate
from pathlib import Path
from string import Formatter
from bs4 import BeautifulSoup
from ebooklib import epub
from pick import pick
import threading
import queue  # Import queue for concurrent reading
from pydub import AudioSegment
from pydub.silence import split_on_silence
from PyPDF2 import PdfReader
try:
    from qwen_tts import Qwen3TTSModel
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
except ImportError:  # optional until runtime image installs qwen-tts
    Qwen3TTSModel = None
    Qwen3TTSConfig = None

from functools import lru_cache
from typing import Any, Dict, Tuple, List, Callable

_ALIGNMENT_GUARD_DISABLED = True


def disable_alignment_guard_checks():
    logging.info("Alignment guard non applicabile: worker v3 usa solo Qwen3-TTS.")

sample_rate = 24000
CHAPTER_MANIFEST_FILENAME = "chapter_exports.json"
TTS_ENGINE = os.getenv("CHATTERBLEZ_TTS_ENGINE", "qwen3_5").strip().lower()
_TTS_RESOURCE_CACHE: Dict[Tuple[str, bool], Dict[str, Any]] = {}
QWEN_MODEL_ID = os.getenv("CHATTERBLEZ_QWEN_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice").strip()
QWEN_SPEAKER_DEFAULT = os.getenv("CHATTERBLEZ_QWEN_SPEAKER", "aiden").strip() or "aiden"
QWEN_INSTRUCT_DEFAULT = os.getenv("CHATTERBLEZ_QWEN_INSTRUCT", "").strip()
QWEN_ATTN_IMPL = os.getenv("CHATTERBLEZ_QWEN_ATTN_IMPL", "flash_attention_2").strip()
QWEN_DTYPE = os.getenv("CHATTERBLEZ_QWEN_DTYPE", "auto").strip().lower()
_QWEN_CACHE: Dict[str, Any] = {"model": None}
# Default chunk sizes for TTS batching.
# Keep chunks moderate to limit EOS/truncation and preserve pacing consistency.
BATCH_MIN_CHARS = max(80, int(os.getenv("CHATTERBLEZ_BATCH_MIN_CHARS", "300")))
BATCH_MAX_CHARS = max(BATCH_MIN_CHARS, int(os.getenv("CHATTERBLEZ_BATCH_MAX_CHARS", "650")))
# Number of text batches sent in one Qwen generate call.
QWEN_MICROBATCH_SIZE = max(1, int(os.getenv("CHATTERBLEZ_QWEN_MICROBATCH_SIZE", "9")))
# Disabled dynamically if qwen-tts raises runtime errors for batched generation.
_QWEN_MICROBATCH_DISABLED = False
QWEN_SDPA_ALIAS_PATCH_ENABLED = str(os.getenv("CHATTERBLEZ_QWEN_SDPA_ALIAS_PATCH", "1")).strip().lower() not in {
    "0",
    "false",
    "no",
}
_QWEN_SDPA_ALIAS_PATCH_APPLIED = False


def remove_silence_from_audio(input_file, output_file, silence_thresh=-50, min_silence_len=1000, keep_silence=200):
    """
    Remove silences from an audio file using pydub with high quality.

    Args:
        input_file: Path to input audio file.
        output_file: Path to output audio file.
        silence_thresh: Silence threshold in dBFS (try -50 to -60).
        min_silence_len: Minimum silence length in milliseconds to remove.
        keep_silence: Amount of silence to keep in ms.
    """
    # Load audio file
    audio = AudioSegment.from_file(input_file)

    # Analyze audio loudness for debugging
    logging.info(f"Audio dBFS: {audio.dBFS:.2f}")
    logging.info(f"Max dBFS: {audio.max_dBFS:.2f}")
    logging.info(f"Sample rate: {audio.frame_rate}Hz")
    logging.info(f"Channels: {audio.channels}")
    logging.info(f"Sample width: {audio.sample_width} bytes")

    # Split audio on silence
    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_silence
    )

    # Check if any chunks were found
    logging.info(f"Found {len(chunks)} audio chunks")

    if len(chunks) == 0:
        logging.warning("WARNING: No audio chunks found! Adjust silence_thresh or min_silence_len")
        logging.warning(f"Try setting silence_thresh to {audio.dBFS - 10:.1f} dBFS")
        return

    # Combine chunks
    combined = AudioSegment.empty()
    for chunk in chunks:
        combined += chunk

    # Export based on file extension
    output_format = Path(output_file).suffix[1:].lower()

    if output_format == 'wav':
        # Export as WAV with PCM (uncompressed)
        combined.export(
            output_file,
            format='wav',
            parameters=[
                '-ar', str(audio.frame_rate),  # Preserve sample rate
                '-ac', str(audio.channels)  # Preserve channels
            ]
        )
    elif output_format in ['m4a', 'm4b']:
        # Export as M4A/M4B with AAC
        export_params = {
            'format': 'ipod',
            'codec': 'aac',
            'bitrate': '128k',
            'parameters': [
                '-ar', str(audio.frame_rate),
                '-ac', str(audio.channels),
                '-q:a', '2'
            ]
        }

        if output_format == 'm4b':
            temp_output = Path(output_file).with_suffix('.m4a')
            combined.export(temp_output, **export_params)
            if os.path.exists(output_file):
                os.remove(output_file)
            os.rename(temp_output, output_file)
        else:
            combined.export(output_file, **export_params)
    else:
        # For other formats, let pydub handle it
        combined.export(output_file, format=output_format)

    logging.info(f"Processed audio saved to: {output_file}")
    original_duration = len(audio) / 1000
    new_duration = len(combined) / 1000
    removed_time = original_duration - new_duration

    logging.info(f"\nOriginal duration: {original_duration:.2f}s")
    logging.info(f"New duration: {new_duration:.2f}s")
    logging.info(f"Removed silence: {removed_time:.2f}s ({removed_time / original_duration * 100:.1f}%)")


def _resolve_tts_engine() -> str:
    env_value = os.getenv("CHATTERBLEZ_TTS_ENGINE")
    if env_value:
        return env_value.strip().lower()
    return TTS_ENGINE


def get_tts_engine_name() -> str:
    return _resolve_tts_engine()


def map_language_id_to_qwen(language_id: str | None) -> str:
    if not language_id:
        return "Auto"
    lang = str(language_id).strip().lower()
    mapping = {
        "auto": "Auto",
        "it": "Italian",
        "ita": "Italian",
        "italian": "Italian",
        "en": "English",
        "eng": "English",
        "english": "English",
        "zh": "Chinese",
        "zh-cn": "Chinese",
        "chinese": "Chinese",
        "es": "Spanish",
        "spanish": "Spanish",
        "fr": "French",
        "french": "French",
        "de": "German",
        "german": "German",
        "pt": "Portuguese",
        "portuguese": "Portuguese",
        "ru": "Russian",
        "russian": "Russian",
        "ja": "Japanese",
        "japanese": "Japanese",
        "ko": "Korean",
        "korean": "Korean",
    }
    return mapping.get(lang, "Auto")


def resample_audio_linear(wav: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate <= 0 or src_rate == dst_rate or wav.size == 0:
        return wav.astype(np.float32, copy=False)
    src = np.asarray(wav, dtype=np.float32).flatten()
    duration = len(src) / float(src_rate)
    target_len = max(1, int(round(duration * float(dst_rate))))
    x_old = np.linspace(0.0, 1.0, num=len(src), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
    return np.interp(x_new, x_old, src).astype(np.float32)


def _resolve_qwen_cuda_dtype() -> Tuple[Any, str]:
    requested = QWEN_DTYPE or "auto"
    if requested in {"auto"}:
        bf16_supported = bool(
            torch.cuda.is_available()
            and hasattr(torch.cuda, "is_bf16_supported")
            and torch.cuda.is_bf16_supported()
        )
        if bf16_supported:
            return torch.bfloat16, "bfloat16(auto)"
        return torch.float16, "float16(auto-fallback)"
    if requested in {"bf16", "bfloat16"}:
        bf16_supported = bool(
            torch.cuda.is_available()
            and hasattr(torch.cuda, "is_bf16_supported")
            and torch.cuda.is_bf16_supported()
        )
        if not bf16_supported:
            logging.warning(
                "CHATTERBLEZ_QWEN_DTYPE=%s ma GPU senza supporto bf16; fallback a float16.",
                requested,
            )
            return torch.float16, "float16(bf16-unsupported-fallback)"
        return torch.bfloat16, "bfloat16"
    if requested in {"fp16", "float16", "half"}:
        return torch.float16, "float16"
    if requested in {"fp32", "float32", "full"}:
        return torch.float32, "float32"
    logging.warning("CHATTERBLEZ_QWEN_DTYPE sconosciuto '%s'; uso auto.", requested)
    bf16_supported = bool(
        torch.cuda.is_available()
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )
    if bf16_supported:
        return torch.bfloat16, "bfloat16(auto-fallback)"
    return torch.float16, "float16(auto-fallback)"


def _apply_dtype_to_nested_qwen_configs(config_obj: Any, dtype: Any, seen: set[int] | None = None) -> None:
    """
    Propagate dtype across nested qwen config objects (e.g. talker/code_predictor).
    FA2 checks inspect each sub-config independently during model construction.
    """
    if config_obj is None:
        return
    if seen is None:
        seen = set()
    obj_id = id(config_obj)
    if obj_id in seen:
        return
    seen.add(obj_id)
    try:
        config_obj.dtype = dtype
    except Exception:
        pass
    for name in dir(config_obj):
        if not name.endswith("_config") or name.startswith("__"):
            continue
        try:
            child = getattr(config_obj, name)
        except Exception:
            continue
        if child is None or callable(child):
            continue
        if hasattr(child, "__dict__"):
            _apply_dtype_to_nested_qwen_configs(child, dtype, seen)


def _is_qwen_sampling_instability(exc: BaseException) -> bool:
    text = str(exc or "").lower()
    return (
        "probability tensor contains either" in text
        or ("nan" in text and "probability" in text)
        or ("inf" in text and "probability" in text)
    )


def _flash_attn_installed() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


def _maybe_patch_transformers_sdpa_aliasing() -> None:
    """
    Work around torch<2.5 + transformers sdpa mask aliasing:
    the stock implementation performs an in-place `|=` on an expanded tensor.
    With some batch shapes this raises:
    "unsupported operation ... single memory location ... clone()".
    """
    global _QWEN_SDPA_ALIAS_PATCH_APPLIED
    if _QWEN_SDPA_ALIAS_PATCH_APPLIED or not QWEN_SDPA_ALIAS_PATCH_ENABLED:
        return
    try:
        import transformers  # type: ignore[import]
        import transformers.masking_utils as masking_utils  # type: ignore[import]
    except Exception as exc:
        logging.debug("SDPA alias patch skipped (transformers import failed): %s", exc)
        return

    if getattr(masking_utils, "_is_torch_greater_or_equal_than_2_5", False):
        _QWEN_SDPA_ALIAS_PATCH_APPLIED = True
        return

    original = getattr(masking_utils, "sdpa_mask_older_torch", None)
    if not callable(original):
        return
    if getattr(original, "__name__", "") == "_chatterblez_sdpa_mask_older_torch_safe":
        _QWEN_SDPA_ALIAS_PATCH_APPLIED = True
        return

    def _chatterblez_sdpa_mask_older_torch_safe(
        batch_size: int,
        cache_position: torch.Tensor,
        kv_length: int,
        kv_offset: int = 0,
        mask_function: Callable = masking_utils.causal_mask_function,
        attention_mask: torch.Tensor | None = None,
        local_size: int | None = None,
        allow_is_causal_skip: bool = True,
        allow_torch_fix: bool = True,
        **kwargs,
    ) -> torch.Tensor | None:
        q_length = cache_position.shape[0]
        padding_mask = masking_utils.prepare_padding_mask(attention_mask, kv_length, kv_offset)

        if allow_is_causal_skip and masking_utils._ignore_causal_mask_sdpa(
            padding_mask, q_length, kv_length, kv_offset, local_size
        ):
            return None

        kv_arange = torch.arange(kv_length, device=cache_position.device)
        kv_arange += kv_offset

        causal_mask = masking_utils._vmap_for_bhqkv(mask_function, bh_indices=False)(
            None, None, cache_position, kv_arange
        )
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, -1, -1, -1)

        if padding_mask is not None:
            causal_mask = causal_mask * padding_mask[:, None, None, :]

        # Out-of-place OR avoids writes on expanded views with shared storage.
        if not masking_utils._is_torch_greater_or_equal_than_2_5 and allow_torch_fix:
            causal_mask = causal_mask | torch.all(~causal_mask, dim=-1, keepdim=True)
        return causal_mask

    masking_utils.sdpa_mask_older_torch = _chatterblez_sdpa_mask_older_torch_safe
    try:
        masking_utils.ALL_MASK_ATTENTION_FUNCTIONS["sdpa"] = _chatterblez_sdpa_mask_older_torch_safe
    except Exception:
        pass

    _QWEN_SDPA_ALIAS_PATCH_APPLIED = True
    logging.info(
        "Applied transformers sdpa alias patch (transformers=%s, torch=%s).",
        getattr(transformers, "__version__", "unknown"),
        torch.__version__,
    )


def _log_qwen_runtime_diagnostics(model_wrapper: Any, dtype_name: str) -> None:
    model = getattr(model_wrapper, "model", None)
    model_device = str(getattr(model_wrapper, "device", "unknown"))
    model_param_dtype = "unknown"
    resolved_attn = None
    try:
        if model is not None:
            first_param = next(model.parameters(), None)
            if first_param is not None:
                model_param_dtype = str(first_param.dtype)
                model_device = str(first_param.device)
            cfg = getattr(model, "config", None)
            if cfg is not None:
                resolved_attn = getattr(cfg, "_attn_implementation", None) or getattr(
                    cfg, "attn_implementation", None
                )
    except Exception:
        pass

    logging.info(
        "QWEN_RUNTIME torch=%s cuda_available=%s torch_cuda=%s flash_attn_installed=%s "
        "requested_attn=%s resolved_attn=%s requested_dtype=%s model_param_dtype=%s model_device=%s",
        torch.__version__,
        torch.cuda.is_available(),
        torch.version.cuda,
        _flash_attn_installed(),
        QWEN_ATTN_IMPL or "default",
        resolved_attn or "default",
        dtype_name,
        model_param_dtype,
        model_device,
    )


def load_qwen_resources() -> Dict[str, Any]:
    if Qwen3TTSModel is None:
        raise RuntimeError("qwen-tts non installato. Installa il pacchetto per usare il worker qwen3_5.")
    _maybe_patch_transformers_sdpa_aliasing()
    if _QWEN_CACHE["model"] is None:
        from_pretrained_kwargs: Dict[str, Any] = {}
        dtype_name = "default(cpu)"
        if torch.cuda.is_available():
            cuda_dtype, dtype_name = _resolve_qwen_cuda_dtype()
            from_pretrained_kwargs["device_map"] = "cuda:0"
            if Qwen3TTSConfig is not None:
                try:
                    config = Qwen3TTSConfig.from_pretrained(QWEN_MODEL_ID, dtype=cuda_dtype)
                    # Ensure dtype is already present on every nested config during
                    # model init. FA2 checks inspect sub-configs independently.
                    _apply_dtype_to_nested_qwen_configs(config, cuda_dtype)
                    from_pretrained_kwargs["config"] = config
                except Exception as exc:
                    logging.warning(
                        "Qwen config preload con dtype fallito; continuo senza config esplicita: %s",
                        exc,
                    )
            # Transformers >=4.57 expects `dtype` (not `torch_dtype`).
            # Passing `torch_dtype` emits a deprecation warning and may cause
            # FA2 checks to think dtype is unset.
            from_pretrained_kwargs["dtype"] = cuda_dtype
            if QWEN_ATTN_IMPL:
                from_pretrained_kwargs["attn_implementation"] = QWEN_ATTN_IMPL
            logging.info(
                "Loading Qwen3-TTS model=%s device=cuda:0 dtype=%s attn=%s",
                QWEN_MODEL_ID,
                dtype_name,
                QWEN_ATTN_IMPL or "default",
            )
        else:
            from_pretrained_kwargs["device_map"] = "cpu"
            logging.info(
                "Loading Qwen3-TTS model=%s device=cpu dtype=default",
                QWEN_MODEL_ID,
            )
        try:
            _QWEN_CACHE["model"] = Qwen3TTSModel.from_pretrained(
                QWEN_MODEL_ID,
                **from_pretrained_kwargs,
            )
        except Exception as exc:
            if from_pretrained_kwargs.pop("attn_implementation", None):
                logging.warning("Qwen3-TTS fallback senza flash attention: %s", exc)
                _QWEN_CACHE["model"] = Qwen3TTSModel.from_pretrained(
                    QWEN_MODEL_ID,
                    **from_pretrained_kwargs,
                )
            else:
                raise
        _log_qwen_runtime_diagnostics(_QWEN_CACHE["model"], dtype_name)
    return {
        "engine": "qwen",
        "model": _QWEN_CACHE["model"],
        "speaker": QWEN_SPEAKER_DEFAULT,
        "instruct": QWEN_INSTRUCT_DEFAULT or None,
    }


def load_tts_resources(use_multilingual: bool, cache: bool = False) -> Dict[str, Any]:
    engine = get_tts_engine_name()
    key = (engine, use_multilingual)
    if cache and key in _TTS_RESOURCE_CACHE:
        return _TTS_RESOURCE_CACHE[key]
    if engine not in {"qwen3_5", "qwen", "qwen3.5"}:
        raise RuntimeError(
            f"Worker qwen3_5 supporta solo CHATTERBLEZ_TTS_ENGINE=qwen3_5 (ricevuto: '{engine}')."
        )
    resources = load_qwen_resources()
    if cache:
        _TTS_RESOURCE_CACHE[key] = resources
    return resources


def _resolve_qwen_speaker(model: Any, requested_speaker: str | None) -> str:
    selected_speaker = (requested_speaker or QWEN_SPEAKER_DEFAULT).strip() or QWEN_SPEAKER_DEFAULT
    try:
        supported_speakers = model.get_supported_speakers()
    except Exception:
        supported_speakers = []
    if supported_speakers:
        supported_map = {str(item).lower(): str(item) for item in supported_speakers}
        resolved = supported_map.get(selected_speaker.lower())
        if resolved:
            selected_speaker = resolved
        elif selected_speaker not in set(supported_speakers):
            logging.warning(
                "Speaker '%s' non supportato dal modello. Uso '%s'.",
                selected_speaker,
                supported_speakers[0],
            )
            selected_speaker = supported_speakers[0]
    return selected_speaker


def _build_qwen_generation_kwargs(
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
    top_k: int | None = None,
) -> Dict[str, Any]:
    generation_kwargs: Dict[str, Any] = {}
    if temperature is not None:
        generation_kwargs["temperature"] = float(temperature)
    if top_p is not None:
        generation_kwargs["top_p"] = float(top_p)
    if repetition_penalty is not None:
        generation_kwargs["repetition_penalty"] = float(repetition_penalty)
    if top_k is not None:
        generation_kwargs["top_k"] = int(top_k)
    return generation_kwargs


def _normalize_qwen_wavs(wavs: List[Any], sr: int) -> List[np.ndarray]:
    if not wavs:
        raise RuntimeError("Qwen3-TTS non ha prodotto alcun output audio.")
    out: List[np.ndarray] = []
    for wav in wavs:
        arr = np.asarray(wav, dtype=np.float32).flatten()
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, -1.0, 1.0)
        if int(sr) != sample_rate:
            logging.warning("Sample rate inatteso da Qwen3-TTS (%s). Resample a %s.", sr, sample_rate)
            arr = resample_audio_linear(arr, int(sr), sample_rate)
        out.append(arr)
    return out


def synthesize_many_with_qwen(
    tts_resources: Dict[str, Any],
    texts: List[str],
    *,
    language_id: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
    top_k: int | None = None,
    speaker: str | None = None,
) -> List[np.ndarray]:
    model = tts_resources["model"]
    cleaned_texts = [str(item or "").strip() for item in texts if str(item or "").strip()]
    if not cleaned_texts:
        return []
    selected_speaker = _resolve_qwen_speaker(
        model,
        speaker or tts_resources.get("speaker") or QWEN_SPEAKER_DEFAULT,
    )
    generation_kwargs = _build_qwen_generation_kwargs(
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        top_k=top_k,
    )
    language = map_language_id_to_qwen(language_id)
    instruct = tts_resources.get("instruct")
    try:
        wavs, sr = model.generate_custom_voice(
            text=cleaned_texts,
            language=language,
            speaker=selected_speaker,
            instruct=instruct,
            **generation_kwargs,
        )
    except RuntimeError as exc:
        if generation_kwargs and _is_qwen_sampling_instability(exc):
            logging.warning(
                "Qwen sampling instabile con kwargs=%s; retry con default del checkpoint.",
                generation_kwargs,
            )
            wavs, sr = model.generate_custom_voice(
                text=cleaned_texts,
                language=language,
                speaker=selected_speaker,
                instruct=instruct,
            )
        else:
            raise
    if len(wavs or []) != len(cleaned_texts):
        raise RuntimeError(
            f"Qwen3-TTS output batch size mismatch: input={len(cleaned_texts)} output={len(wavs or [])}."
        )
    return _normalize_qwen_wavs(wavs, int(sr))


def synthesize_with_qwen(
    tts_resources: Dict[str, Any],
    text: str,
    *,
    language_id: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
    top_k: int | None = None,
    speaker: str | None = None,
) -> np.ndarray:
    generated = synthesize_many_with_qwen(
        tts_resources,
        [text],
        language_id=language_id,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        top_k=top_k,
        speaker=speaker,
    )
    if not generated:
        raise RuntimeError("Qwen3-TTS non ha prodotto alcun output audio.")
    return generated[0]


import string

# Set of all punctuation characters to preserve (from `string.punctuation`)
PUNCTUATION = set(string.punctuation)

# Precompiled regex: sequences of 2 or more non-alphanumeric characters
non_alnum_seq_re = re.compile(r'[^a-zA-Z0-9]{2,}')

# Substitution function
def replace_non_alnum_sequence(match):
    first = match.group(0)[0]
    return first if first in PUNCTUATION else ''

allowed_chars_re = re.compile(r"[^’\"a-zA-Z0-9\s.,;:'\"-]")

@lru_cache(maxsize=1)
def get_nlp():
    """
    Lightweight, cached spacy pipeline used only for sentence segmentation.
    Falls back to full model if `spacy.blank` is not available for the
    requested language.
    """
    try:
        nlp = spacy.blank("xx")  # very small, language-agnostic
    except Exception:  # Fallback – should basically never happen
        load_spacy()
        nlp = spacy.load("en_core_web_trf")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


# ---------------------------------------------------------------------------
# Helper for progress / ETA
# ---------------------------------------------------------------------------
def update_stats(stats, added_chars):
    """
    Update statistics (chars processed, speed, ETA) using an exponential
    moving average to smooth the instantaneous chars/sec measurement. This
    greatly improves the accuracy of the ETA that is reported to the user.
    """
    stats.processed_chars += added_chars
    elapsed = time.perf_counter() - stats.start_time
    if elapsed <= 0:
        return
    current_rate = stats.processed_chars / elapsed
    alpha = 0.3  # smoothing factor
    stats.chars_per_sec = alpha * current_rate + (1 - alpha) * stats.chars_per_sec
    remaining_chars = max(stats.total_chars - stats.processed_chars, 0)
    stats.eta = strfdelta(remaining_chars / stats.chars_per_sec) if stats.chars_per_sec else "?:??"
    stats.progress = stats.processed_chars * 100 // stats.total_chars


def load_spacy():
    if not spacy.util.is_package("en_core_web_trf"):
        logging.info("Downloading Spacy model en_core_web_trf...")
        spacy.cli.download("en_core_web_trf")


import ctypes

# Constants from WinBase.h
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002


# Set execution state to prevent sleep
def prevent_sleep():
    if platform.system() == "Windows":
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
        )


# Reset execution state to allow sleep
def allow_sleep():
    if platform.system() == "Windows":
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


def set_espeak_library():
    """Find the espeak library path"""
    try:
        if os.environ.get('ESPEAK_LIBRARY'):
            library = os.environ['ESPEAK_LIBRARY']
        elif platform.system() == 'Darwin':
            from subprocess import check_output
            try:
                cellar = Path(check_output(["brew", "--cellar"], text=True).strip())
                pattern = cellar / "espeak-ng" / "*" / "lib" / "*.dylib"
                if not (library := next(iter(glob(str(pattern))), None)):
                    raise RuntimeError("No espeak-ng library found; please set the path manually")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                raise RuntimeError("Cannot locate Homebrew Cellar. Is 'brew' installed and in PATH?") from e
        elif platform.system() == 'Linux':
            library = glob('/usr/lib/*/libespeak-ng*')[0]
        elif platform.system() == 'Windows':
            paths = glob('C:\\Program Files\\eSpeak NG\\libespeak-ng.dll') + \
                    glob('C:\\Program Files (x86)\\eSpeak NG\\libespeak-ng.dll')
            if paths:
                library = paths[0]
            else:
                raise RuntimeError(
                    "eSpeak NG library not found in default paths. Please set ESPEAK_LIBRARY environment variable.")
        else:
            logging.warning('Unsupported OS, please set the espeak library path manually')
            return
        logging.info('Using espeak library: %s', library)
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        EspeakWrapper.set_library(library)
    except Exception:
        logging.exception("Error finding espeak-ng library")
        logging.warning("Probably you haven't installed espeak-ng.")
        logging.warning("On Mac: brew install espeak-ng")
        logging.warning("On Linux: sudo apt install espeak-ng")
        logging.warning("On Windows: Download from https://github.com/espeak-ng/espeak-ng/releases")


def match_case(word, replacement):
    if word.isupper():
        return replacement.upper()
    elif word.islower():
        return replacement.lower()
    elif word[0].isupper():
        return replacement.capitalize()
    else:
        return replacement  # fallback (e.g., mixed case)


def replace_preserve_case(text, old, new):
    if len(old) != len(new):
        raise ValueError("Replacement arrays must be the same length.")

    for o, n in zip(old, new):
        pattern = re.compile(rf'\b{re.escape(o)}\b', re.IGNORECASE)

        def repl(match):
            return match_case(match.group(), n)

        text = pattern.sub(repl, text)

    return text

# Define only "speakable" punctuation - ones that affect how text is read aloud
SPEAKABLE_PUNCT = '.,-\'"'
ESCAPED_SPEAKABLE = re.escape(SPEAKABLE_PUNCT)

# All punctuation for removal purposes
ALL_PUNCT = re.escape(string.punctuation)

# Compiled regex patterns
remove_unwanted = re.compile(rf'[^\w\s{ALL_PUNCT}]+')
remove_unspeakable = re.compile(rf'[{re.escape("".join(set(string.punctuation) - set(SPEAKABLE_PUNCT)))}]+')
normalize_quotes = re.compile(r'[""''`]')  # Smart quotes and backticks to normalize
replace_em_dash = re.compile(r'—')  # Em dash to replace with space
collapse_punct = re.compile(rf'[{ESCAPED_SPEAKABLE}][\s{ESCAPED_SPEAKABLE}]*(?=[{ESCAPED_SPEAKABLE}])')

def clean_string(text):
    """
    Remove non-alphanumeric chars, keep only speakable punctuation,
    normalize quotes, replace em dashes with spaces, and collapse multiple punctuation to keep only the last one.
    """
    # Replace em dashes with spaces FIRST before any other processing
    step1 = replace_em_dash.sub(' ', text)
    
    # Remove all characters that aren't alphanumeric, whitespace, or punctuation
    step2 = remove_unwanted.sub('', step1)
    
    # Normalize smart quotes and backticks to standard quotes
    step3 = normalize_quotes.sub(lambda m: '"' if m.group() in '""' else "'", step2)
    
    # Remove unspeakable punctuation (symbols like @#$%^&*()[]{}|\ etc.)
    step4 = remove_unspeakable.sub('', step3)
    
    # Then collapse sequences of speakable punctuation (with optional whitespace) to keep only the last one
    step5 = collapse_punct.sub('', step4)
    
    # Clean up any remaining multiple whitespace
    result = re.sub(r'\s+', ' ', step5).strip()
    
    return result


# Step 1: Normalize curly quotes
def normalize_quotes(text: str) -> str:
    return (
        text.replace("“", '"')
            .replace("”", '"')
            .replace("‘", "'")
            .replace("’", "'")
    )


def soften_double_quotes(text: str) -> str:
    """
    Replace double-quoted spans with guillemets to cue a slight pause without
    the TTS pronouncing the quote as a word (es. elimina il \"met\").
    """
    def repl(match: re.Match[str]) -> str:
        content = match.group(1).strip()
        return f" «{content}» "

    text = double_quote_span_re.sub(repl, text)
    # Remove any stray double quotes that were not part of a pair
    return text.replace('"', '')

# Step 2: Replace disallowed characters (keep basic Latin + common accents + speech punctuation/currency)
# Allow basic punctuation plus parentheses; block everything else.
# Added «» so we can preserve converted virgolette without being spelled out.
non_allowed_re = re.compile(r"[^0-9A-Za-z\u00C0-\u017F\s.,'\"?!:;%€$-(){}«»]+")

# Step 3: Collapse multiple spaces
space_re = re.compile(r'\s+')

# Step 4: Remove space(s) before a period
space_before_period_re = re.compile(r'\s+\.')

# Step 5: Collapse consecutive periods
multiple_periods_re = re.compile(r'\.{2,}')

# Step 6: Remove a trailing period immediately after a closing parenthesis
paren_dot_re = re.compile(r'\)\s*\.')

# Quote handling: turn "testo" into «testo» to avoid the TTS pronouncing the quote marker.
double_quote_span_re = re.compile(r'"([^"]+)"')

# Helpers: conversione numeri -> testo (cardinali/ordinali) e numeri romani
_cardinal_0_19 = [
    "zero",
    "uno",
    "due",
    "tre",
    "quattro",
    "cinque",
    "sei",
    "sette",
    "otto",
    "nove",
    "dieci",
    "undici",
    "dodici",
    "tredici",
    "quattordici",
    "quindici",
    "sedici",
    "diciassette",
    "diciotto",
    "diciannove",
]

_tens_map = {
    2: "venti",
    3: "trenta",
    4: "quaranta",
    5: "cinquanta",
    6: "sessanta",
    7: "settanta",
    8: "ottanta",
    9: "novanta",
}

_thousands_map = {
    1: "mille",
    2: "duemila",
    3: "tremila",
    4: "quattromila",
    5: "cinquemila",
    6: "seimila",
    7: "settemila",
    8: "ottomila",
    9: "novemila",
}

_ordinal_map_explicit = {
    1: "primo",
    2: "secondo",
    3: "terzo",
    4: "quarto",
    5: "quinto",
    6: "sesto",
    7: "settimo",
    8: "ottavo",
    9: "nono",
    10: "decimo",
    11: "undicesimo",
    12: "dodicesimo",
    13: "tredicesimo",
    14: "quattordicesimo",
    15: "quindicesimo",
    16: "sedicesimo",
    17: "diciassettesimo",
    18: "diciottesimo",
    19: "diciannovesimo",
    20: "ventesimo",
    21: "ventunesimo",
    22: "ventiduesimo",
    23: "ventitreesimo",
    24: "ventiquattresimo",
    25: "venticinquesimo",
    26: "ventiseiesimo",
    27: "ventisettesimo",
    28: "ventottesimo",
    29: "ventinovesimo",
    30: "trentesimo",
}

_roman_values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def roman_to_int(value: str) -> int | None:
    total = 0
    prev = 0
    for ch in reversed(value.upper()):
        if ch not in _roman_values:
            return None
        val = _roman_values[ch]
        if val < prev:
            total -= val
        else:
            total += val
            prev = val
    return total


def _under_hundred(n: int) -> str:
    if n < 20:
        return _cardinal_0_19[n]
    tens, unit = divmod(n, 10)
    tens_word = _tens_map.get(tens, "")
    # Elide the final vowel of the decade with 1 or 8 (ventuno, ventotto, ottantuno, ecc.)
    if unit in (1, 8) and tens_word:
        tens_word = tens_word[:-1]
    if unit == 0:
        return tens_word
    return f"{tens_word}{_cardinal_0_19[unit]}"


def _under_thousand(n: int) -> str:
    if n < 100:
        return _under_hundred(n)
    hundreds, rem = divmod(n, 100)
    if hundreds == 1:
        prefix = "cento"
    else:
        prefix = f"{_cardinal_0_19[hundreds]}cento"
    # Elide the 'o' in cento when followed by ottanta/otto (centottanta, duecentottanta, ecc.)
    if rem and (rem == 8 or 80 <= rem < 90):
        if prefix.endswith("cento"):
            prefix = prefix[:-1]
    return prefix if rem == 0 else f"{prefix}{_under_hundred(rem)}"


def int_to_italian_cardinal(n: int) -> str:
    if n < 0 or n > 9999:
        return str(n)
    if n < 20:
        return _cardinal_0_19[n]
    if n < 100:
        return _under_hundred(n)
    if n < 1000:
        return _under_thousand(n)
    thousands, rem = divmod(n, 1000)
    if thousands in _thousands_map:
        prefix = _thousands_map[thousands]
    else:
        # defensive fallback, though with n <= 9999 we never hit this
        prefix = f"{int_to_italian_cardinal(thousands)}mila"
    return prefix if rem == 0 else f"{prefix}{_under_thousand(rem)}"


def int_to_italian_ordinal(n: int) -> str:
    if n in _ordinal_map_explicit:
        return _ordinal_map_explicit[n]
    if n < 1:
        return str(n)
    # Generic fallback: cardinal without last vowel + "esimo"
    base = int_to_italian_cardinal(n)
    if base and base[-1] in "aeiou":
        base = base[:-1]
    return f"{base}esimo"


def convert_numbers_in_text(text: str) -> str:
    def parse_token(raw: str) -> int | None:
        if raw.isdigit():
            try:
                return int(raw)
            except ValueError:
                return None
        maybe = roman_to_int(raw)
        return maybe

    # 1) Centuries: "XX secolo" / "20 secolo" -> "ventesimo secolo"
    def repl_century(match: re.Match[str]) -> str:
        raw = match.group(1)
        number = parse_token(raw)
        if number is None or number <= 0 or number > 9999:
            return match.group(0)
        return f"{int_to_italian_ordinal(number)} secolo"

    century_re = re.compile(r"\b([IVXLCDM]+|\d{1,4})\s*°?\s+secolo\b", flags=re.IGNORECASE)
    text = century_re.sub(repl_century, text)

    # 2) Roman numerals (length >= 2 to avoid accidental "I" pronouns)
    def repl_roman(match: re.Match[str]) -> str:
        raw = match.group(0)
        number = roman_to_int(raw)
        if number is None or number <= 0 or number > 9999:
            return raw
        return int_to_italian_cardinal(number)

    text = re.sub(r"\b[IVXLCDM]{2,}\b", repl_roman, text)

    # 3) 1-2 letters + digits (es. "C17" -> "C diciassette", "RX100" -> "RX cento") keeping the prefix.
    def repl_letter_num(match: re.Match[str]) -> str:
        letter = match.group(1)
        try:
            number = int(match.group(2))
        except ValueError:
            return match.group(0)
        if number > 9999:
            return match.group(0)
        return f"{letter} {int_to_italian_cardinal(number)}"

    text = re.sub(r"\b([A-Za-z]{1,2})(\d{1,4})\b", repl_letter_num, text)

    # 4) Arabic numbers up to 4 digits
    def repl_number(match: re.Match[str]) -> str:
        try:
            number = int(match.group(0))
        except ValueError:
            return match.group(0)
        if number > 9999:
            return match.group(0)
        return int_to_italian_cardinal(number)

    text = re.sub(r"\b\d{1,4}\b", repl_number, text)
    return text


def _convert_enne_age(match: re.Match[str]) -> str:
    """Convert forms like '13enne' -> 'tredicenne'."""
    raw = match.group(1)
    try:
        number = int(raw)
    except ValueError:
        return match.group(0)
    if number < 0 or number > 9999:
        return match.group(0)
    base = int_to_italian_cardinal(number)
    if base and base[-1] in "aeiou":
        base = base[:-1]
    return f"{base}enne"

def clean_line(line: str) -> str:
    line = normalize_quotes(line)
    line = line.replace("È", "è")                             # Normalize uppercase accented E to lowercase
    line = re.sub(r"\bcazzo\b", "catzo", line, flags=re.IGNORECASE)  # Soften explicit term
    line = re.sub(r"\b(\d{1,4})enne\b", _convert_enne_age, line, flags=re.IGNORECASE)  # 13enne -> tredicenne
    line = soften_double_quotes(line)                         # Trasforma "..." in «...» per evitare pronuncia dei segni
    line = non_allowed_re.sub(' ', line)                      # Remove unwanted chars
    line = space_before_period_re.sub('.', line)              # Remove space before .
    line = re.sub(r'([?!])\.', r'\1', line)                    # Drop trailing '.' if it follows ? or !
    line = paren_dot_re.sub(')', line)                         # Drop '.' that immediately follows a closing parenthesis
    line = multiple_periods_re.sub('.', line)                 # Remove repeated .
    line = convert_numbers_in_text(line)                      # Convert numeric tokens (romani e arabi) in parole
    line = space_re.sub(' ', line)                            # Collapse spaces
    return line.strip()


def merge_hyphenated_lines(lines: list[str]) -> list[str]:
    """Merge words split by end-of-line hyphens (PDF/EPUB line wraps)."""
    merged: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        while line.endswith("-") and i + 1 < len(lines):
            next_line = lines[i + 1].lstrip()
            if next_line and re.match(r"[0-9A-Za-z\u00C0-\u017F]", next_line):
                line = line[:-1] + next_line
                i += 1
            else:
                break
        merged.append(line)
        i += 1
    return merged


def split_sentence_by_length(text: str, max_chars: int) -> List[str]:
    """Split a single (very long) sentence into <= max_chars chunks on word boundaries."""
    words = text.split()
    if not words:
        return []

    parts: List[str] = []
    current: List[str] = []
    current_len = 0

    for word in words:
        word_len = len(word)
        # +1 accounts for the space that will be inserted when joining
        projected_len = current_len + (1 if current else 0) + word_len
        if current and projected_len > max_chars:
            parts.append(' '.join(current))
            current = [word]
            current_len = word_len
        else:
            if current:
                current_len += 1  # space
            current.append(word)
            current_len += word_len

    if current:
        parts.append(' '.join(current))

    return parts


def plan_batches_for_text(nlp, text: str, max_sentences: int | None = None) -> List[str]:
    """Build the exact batch plan used for TTS so global chunk progress can be deterministic."""
    doc = nlp(text)
    sentences = list(doc.sents)
    batches = batch_sentences_intelligently(
        sentences,
        min_chars=BATCH_MIN_CHARS,
        max_chars=BATCH_MAX_CHARS,
    )
    batches = coalesce_short_batches(
        batches,
        min_chars=BATCH_MIN_CHARS,
        max_chars=BATCH_MAX_CHARS,
    )
    if max_sentences and int(max_sentences) > 0:
        batches = batches[: int(max_sentences)]
    return batches


def main(file_path, pick_manually, speed, book_year='', output_folder='.',
         max_chapters=None, max_sentences=None, selected_chapters=None, selected_chapter_indices=None, post_event=None, audio_prompt_wav=None, batch_files=None, ignore_list=None, should_stop=None,
         repetition_penalty=1.1, min_p=0.02, top_p=0.95, top_k=None, exaggeration=0.4, cfg_weight=0.8, temperature=0.85,
         enable_silence_trimming=False, silence_thresh=-50, min_silence_len=500, keep_silence=100,
         use_multilingual=False, language_id='en', sentence_gap_ms=0, question_gap_ms=0, disable_alignment_guard=False,
         force_sentence_gaps=True, per_chapter_export=False):
    """
    Main entry point for audiobook synthesis.
    - ignore_list: list of chapter names to ignore (case-insensitive substring match)
    - batch_files: if provided, a list of file paths to process sequentially
    - should_stop: optional callback, returns True if synthesis should be interrupted
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s',
        handlers=[
            logging.FileHandler("logs/app.log", encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.getLogger('qwen_tts').setLevel(logging.WARNING)
    params = {
        "repetition_penalty":repetition_penalty,
        "min_p":min_p,
        "top_p":top_p,
        "top_k":top_k,
        "exaggeration":exaggeration,
        "cfg_weight":cfg_weight,
        "temperature":temperature,
        "enable_silence_trimming":enable_silence_trimming,
        "silence_thresh":silence_thresh,
        "min_silence_len":min_silence_len,
        "keep_silence":keep_silence,
        "use_multilingual":use_multilingual,
        "language_id":language_id,
        "sentence_gap_ms":sentence_gap_ms,
        "question_gap_ms":question_gap_ms,
        "disable_alignment_guard":disable_alignment_guard,
        "per_chapter_export":per_chapter_export,
    }

    # Log all parameters
    for key, value in params.items():
        logging.info(f"{key} = {value}")
    if should_stop is None:
        should_stop = lambda: False

    if batch_files is not None:
        # Sequentially process each file in batch_files
        for batch_file in batch_files:
            # Call main for each file, passing ignore_list and other params
            main(
                file_path=batch_file,
                pick_manually=pick_manually,
                speed=speed,
                book_year=book_year,
                output_folder=output_folder,
                max_chapters=max_chapters,
                max_sentences=max_sentences,
                selected_chapters=None,
                post_event=post_event,
                audio_prompt_wav=audio_prompt_wav,
                batch_files=None,  # Prevent infinite recursion
                ignore_list=ignore_list,
                should_stop=should_stop,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
                top_k=top_k,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                enable_silence_trimming=enable_silence_trimming,
                silence_thresh=silence_thresh,
                min_silence_len=min_silence_len,
                keep_silence=keep_silence,
                use_multilingual=use_multilingual,
                language_id=language_id,
                sentence_gap_ms=sentence_gap_ms,
                question_gap_ms=question_gap_ms,
                disable_alignment_guard=disable_alignment_guard,
                per_chapter_export=per_chapter_export,
            )
            if post_event:
                post_event('CORE_FILE_FINISHED', file_path=batch_file)
            if should_stop():
                break
        return

    if post_event: post_event('CORE_STARTED')
    IS_WINDOWS = sys.platform.startswith("win")

    prevent_sleep()

    load_spacy()
    if output_folder != '.':
        Path(output_folder).mkdir(parents=True, exist_ok=True)

    filename = Path(file_path).name
    filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
    extension = os.path.splitext(file_path)[1].lower()
    logging.info(f"extension {extension}")
    if extension == '.pdf':
        title = os.path.splitext(os.path.basename(file_path))[0]
        creator = "Unknown"
        cover_image = b""
        document_chapters = extract_pdf_chapters(file_path, title)
        selected_chapters = document_chapters
    elif extension == '.txt':
        title = os.path.splitext(os.path.basename(file_path))[0]
        creator = "Unknown"
        cover_image = b""
        document_chapters = extract_txt_chapters(file_path, title)
        selected_chapters = document_chapters
    elif extension == '.epub':
        book = epub.read_epub(file_path)
        meta_title = book.get_metadata('DC', 'title')
        title = meta_title[0][0] if meta_title else ''
        meta_creator = book.get_metadata('DC', 'creator')
        creator = meta_creator[0][0] if meta_creator else ''
        cover_maybe = find_cover(book)
        cover_image = cover_maybe.get_content() if cover_maybe else b""
        if cover_maybe:
            logging.info(f'Found cover image {cover_maybe.file_name} in {cover_maybe.media_type} format')
            if False:
                # Save cover image as "<book name>.<image extension>"
                media_type = cover_maybe.media_type  # e.g., "image/jpeg"
                ext_map = {
                    "image/jpeg": ".jpg",
                    "image/jpg": ".jpg",
                    "image/png": ".png",
                    "image/gif": ".gif",
                    "image/bmp": ".bmp",
                    "image/webp": ".webp"
                }
                ext = ext_map.get(media_type, ".img")
                # Clean title for filename
                safe_title = re.sub(r'[\\/:*?"<>|]', '_', title).strip() or "cover"
                cover_filename = f"{safe_title}{ext}"
                cover_path = Path(output_folder) / cover_filename
                with open(cover_path, "wb") as f:
                    f.write(cover_image)
                logging.info(f"Cover image saved as {cover_path}")
        document_chapters = find_document_chapters_and_extract_texts(book)

        if not selected_chapters:
            if pick_manually is True:
                selected_chapters = pick_chapters(document_chapters)
            else:
                selected_chapters = find_good_chapters(document_chapters)
    else:
        logging.error("Unsupported file extension: %s", extension)
        if post_event:
            post_event('CORE_ERROR', message=f"Unsupported file type: {extension}")
        allow_sleep()
        return
    if selected_chapters is None:
        selected_chapters = document_chapters

    if selected_chapter_indices:
        chapter_by_index = {c.chapter_index: c for c in document_chapters}
        filtered = []
        for idx in selected_chapter_indices:
            chapter = chapter_by_index.get(idx)
            if chapter:
                filtered.append(chapter)
        if filtered:
            selected_chapters = filtered

    # Filter chapters based on ignore_list
    if ignore_list:
        def should_include(chapter):
            name = chapter.get_name().lower()
            for ignore in ignore_list:
                if ignore.lower() in name:
                    return False
            return True
        selected_chapters = [c for c in selected_chapters if should_include(c)]

    print_selected_chapters(document_chapters, selected_chapters)
    total_chars = sum(len(getattr(c, "extracted_text", "")) for c in selected_chapters)
    total_words = sum(len(getattr(c, "extracted_text", "").split()) for c in selected_chapters)

    has_ffmpeg = shutil.which('ffmpeg') is not None
    if not has_ffmpeg:
        logging.error('ffmpeg not found. Please install ffmpeg to create mp3 and m4b audiobook files.')
        if post_event:
            post_event('CORE_ERROR', message="FFmpeg not found. Please install it to create audiobooks.")
        allow_sleep()
        return

    stats = SimpleNamespace(
        total_chars=total_chars,
        processed_chars=0,
        chars_per_sec=500 if torch.cuda.is_available() else 50,  # initial guess
        start_time=time.perf_counter(),
        eta='–',
        progress=0
    )
    logging.info('Started at: %s', time.strftime('%H:%M:%S'))
    logging.info(f'Total characters: {stats.total_chars:,}')
    logging.info('Total words: %d', total_words)
    eta = strfdelta((stats.total_chars - stats.processed_chars) / stats.chars_per_sec)
    logging.info(f'Estimated time remaining (assuming {stats.chars_per_sec} chars/sec): {eta}')
    chapter_wav_files = []
    chapter_exports = []

    tts_resources = load_tts_resources(use_multilingual=use_multilingual, cache=False)
    engine_name = tts_resources["engine"]
    logging.info(f'Using TTS engine: {engine_name}')

    if disable_alignment_guard and use_multilingual:
        disable_alignment_guard_checks()
    if audio_prompt_wav:
        logging.warning("Audio prompt non supportato con il motore %s; verrà ignorato.", engine_name)

    nlp = get_nlp()
    total_selected_chapters = len(selected_chapters)
    prepared_chapter_texts: Dict[int, str] = {}
    planned_chunk_totals: Dict[int, int] = {}
    global_chunk_total = 0
    for idx, chapter in enumerate(selected_chapters, start=1):
        lines = chapter.extracted_text.splitlines()
        cleaned_lines = []
        for line in lines:
            cleaned_line = clean_line(line)
            if cleaned_line.strip() and re.search(r"\w", cleaned_line):
                cleaned_lines.append(cleaned_line)
        cleaned_lines = merge_hyphenated_lines(cleaned_lines)
        text = " ".join(cleaned_lines)
        prepared_chapter_texts[idx] = text
        if len(text.strip()) < 10:
            planned_batches = []
        else:
            planned_batches = plan_batches_for_text(nlp, text, max_sentences=max_sentences)
        chapter_total_chunks = len(planned_batches)
        planned_chunk_totals[idx] = chapter_total_chunks
        global_chunk_total += chapter_total_chunks
        logging.info(
            "CHAPTER_CHUNK_PLAN chapter=%s total=%s",
            idx,
            chapter_total_chunks,
        )
    logging.info(
        "GLOBAL_CHUNK_PLAN total=%s chapters=%s",
        global_chunk_total,
        total_selected_chapters,
    )

    for i, chapter in enumerate(selected_chapters, start=1):
        if should_stop():
            logging.info("Synthesis interrupted by user (chapter loop).")
            break
        if max_chapters and i > max_chapters: break
        logging.info(
            "CHAPTER_PROGRESS current=%s total=%s remaining=%s",
            i,
            total_selected_chapters,
            max(0, total_selected_chapters - i),
        )
        text = prepared_chapter_texts.get(i, "")

        # Sanitize the chapter name to remove all non-alphanumeric characters for the filename
        xhtml_file_name = re.sub(r'[^a-zA-Z0-9-]', '', chapter.get_name()).replace('xhtml', '').replace('html', '')
        chapter_wav_path = Path(output_folder) / filename.replace(extension, f'_chapter_{xhtml_file_name}.wav')
        include_in_concat = not per_chapter_export
        if include_in_concat:
            chapter_wav_files.append(chapter_wav_path)
        if Path(chapter_wav_path).exists():
            logging.info(f'File for chapter {i} already exists. Skipping')
            stats.processed_chars += len(text)
            planned_for_chapter = planned_chunk_totals.get(i, 0)
            if planned_for_chapter > 0:
                logging.info(
                    "CHUNK_PROGRESS current=%s total=%s remaining=0",
                    planned_for_chapter,
                    planned_for_chapter,
                )
            if post_event and hasattr(chapter, "chapter_index"):
                post_event('CORE_CHAPTER_FINISHED', chapter_index=chapter.chapter_index)
            continue
        if len(text.strip()) < 10:
            logging.info(f'Skipping empty chapter {i}')
            if include_in_concat and chapter_wav_path in chapter_wav_files:
                chapter_wav_files.remove(chapter_wav_path)
            continue

        preview = re.sub(r"\s+", " ", text).strip()
        if len(preview) > 140:
            preview = preview[:140] + "..."
        logging.info(
            "CHAPTER_TEXT chapter=%s chars=%s words=%s preview=%s",
            i,
            len(text),
            len(text.split()),
            preview,
        )
        start_time = time.time()
        if post_event and hasattr(chapter, "chapter_index"):
            post_event('CORE_CHAPTER_STARTED', chapter_index=chapter.chapter_index)
        audio_chunks = gen_audio_segments(
            tts_resources,
            nlp,
            text,
            speed,
            stats,
            post_event=post_event,
            max_sentences=max_sentences,
            should_stop=should_stop,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            use_multilingual=use_multilingual,
            language_id=language_id,
            audio_prompt_wav=audio_prompt_wav,
            sentence_gap_ms=sentence_gap_ms,
            question_gap_ms=question_gap_ms,
            force_sentence_gaps=force_sentence_gaps,
        )
        if should_stop():
            logging.info("Synthesis interrupted by user (after audio generation).")
            break
        frames_written = write_audio_stream(chapter_wav_path, audio_chunks)
        if frames_written > 0:
            if enable_silence_trimming:
                trimmed_path = chapter_wav_path.with_suffix('.trimmed.wav')
                remove_silence_from_audio(
                    chapter_wav_path,
                    trimmed_path,
                    silence_thresh=silence_thresh,
                    min_silence_len=min_silence_len,
                    keep_silence=keep_silence
                )
                # Replace original with trimmed
                os.remove(chapter_wav_path)
                os.rename(trimmed_path, chapter_wav_path)

            end_time = time.time()
            delta_seconds = end_time - start_time
            chars_per_sec = len(text) / delta_seconds if delta_seconds else 0
            logging.info('Chapter written to %s', chapter_wav_path)
            if post_event and hasattr(chapter, "chapter_index"):
                post_event('CORE_CHAPTER_FINISHED', chapter_index=chapter.chapter_index)
            logging.info(f'Chapter {i} read in {delta_seconds:.2f} seconds ({chars_per_sec:.0f} characters per second)')
            if per_chapter_export:
                final_audio_path = convert_chapter_wav_to_m4a(chapter_wav_path)
                chapter_exports.append(
                    {
                        "sequence": len(chapter_exports) + 1,
                        "chapter_index": getattr(chapter, "chapter_index", i - 1),
                        "chapter_name": chapter.get_name(),
                        "audio_path": str(final_audio_path),
                        "file_name": Path(final_audio_path).name,
                    }
                )
                continue
        else:
            logging.warning(f'Warning: No audio generated for chapter {i}')
            if include_in_concat and chapter_wav_path in chapter_wav_files:
                chapter_wav_files.remove(chapter_wav_path)

    if per_chapter_export:
        if not chapter_exports:
            logging.error("No chapter exports were produced.")
            if post_event:
                post_event('CORE_ERROR', message="No chapter exports were produced.")
            allow_sleep()
            return
        manifest_path = Path(output_folder) / CHAPTER_MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps({"chapters": chapter_exports}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logging.info("Per-chapter exports saved (%d files).", len(chapter_exports))
        if post_event:
            post_event('CORE_FINISHED')
        allow_sleep()
        return

    if not chapter_wav_files:
        logging.error("No audio chapters were generated. Cannot create audiobook.")
        if post_event:
            post_event('CORE_ERROR', message="No audio chapters were generated.")
        allow_sleep()
        return

    original_name = Path(filename).with_suffix('').name  # removes old suffix
    parts = original_name.split('--')
    if len(parts) > 2:
        new_dir_name = f"{parts[0]}--{parts[1]}".strip()
        output_folder = Path(output_folder) / new_dir_name
        output_folder.mkdir(parents=True, exist_ok=True)

    if has_ffmpeg:
        create_index_file(title, creator, chapter_wav_files, output_folder)
        try:
            concat_file_path = concat_wavs_with_ffmpeg(chapter_wav_files, output_folder, filename,
                                                       post_event=post_event, should_stop=should_stop)
            if should_stop() or concat_file_path is None:
                logging.info("Synthesis interrupted before or during FFmpeg concat.")
                allow_sleep()
                return
            create_m4b(concat_file_path, filename, cover_image, output_folder, post_event=post_event, should_stop=should_stop)
            if should_stop():
                logging.info("Synthesis interrupted before or during FFmpeg m4b creation.")
                allow_sleep()
                return
            if post_event: post_event('CORE_FINISHED')
        except RuntimeError as e:
            logging.error(f"Audiobook creation failed: {e}")
            if post_event:
                post_event('CORE_ERROR', message=str(e))
    logging.info('Ended at: %s', time.strftime('%H:%M:%S'))

    all_files = os.listdir(output_folder)
    wav_files = [os.path.join(output_folder, f) for f in all_files if f.lower().endswith('.wav')]

    for wav_file in wav_files:
        try:
            os.remove(wav_file)
            logging.debug(f"Deleted: {wav_file}")
        except Exception as e:
            logging.debug(f"Failed to delete {wav_file}: {e}")

    allow_sleep()



def batch_sentences_intelligently(sentences, min_chars=BATCH_MIN_CHARS, max_chars=BATCH_MAX_CHARS):
    """
    Batch sentences into reasonable chunks for TTS processing.

    Notes:
    - Smaller batches mitigate early-EOS/truncation from the TTS model.
    - Sentences longer than `max_chars` are further split on word boundaries
      to guarantee no text is dropped.
    """
    batches: List[str] = []
    current_batch: List[str] = []
    current_length = 0

    for sent in sentences:
        raw_sent = sent.text.strip()
        if not raw_sent or len(raw_sent) < 2:
            continue

        # Ensure overlong sentences are broken down so we never drop text.
        candidate_segments = [raw_sent]
        if len(raw_sent) > max_chars:
            candidate_segments = split_sentence_by_length(raw_sent, max_chars)

        for segment in candidate_segments:
            seg_len = len(segment)
            if seg_len == 0:
                continue

            # If adding this segment would exceed max_chars, flush current batch first.
            if current_length > 0 and (current_length + seg_len + 1) > max_chars:
                batches.append(' '.join(current_batch))
                current_batch = []
                current_length = 0

            current_batch.append(segment)
            current_length += seg_len + (1 if len(current_batch) > 1 else 0)  # +1 for space between segments

            # If we've reached a healthy size and ended on punctuation, flush.
            if current_length >= min_chars and segment.endswith(('.', '!', '?', '"', "'")):
                batches.append(' '.join(current_batch))
                current_batch = []
                current_length = 0

    # Don't forget the last batch
    if current_batch:
        batches.append(' '.join(current_batch))

    return batches


def coalesce_short_batches(
    batches: List[str],
    *,
    min_chars: int = BATCH_MIN_CHARS,
    max_chars: int = BATCH_MAX_CHARS,
) -> List[str]:
    """
    Merge adjacent tiny batches while keeping a strict `max_chars` cap.

    This reduces the number of serial TTS calls when sentence segmentation
    generates many short chunks.
    """
    if len(batches) < 2:
        return [batch for batch in batches if (batch or "").strip()]

    target_min = max(80, int(min_chars * 0.85))
    merged: List[str] = []
    i = 0
    while i < len(batches):
        current = (batches[i] or "").strip()
        if not current:
            i += 1
            continue
        j = i
        while j + 1 < len(batches):
            nxt = (batches[j + 1] or "").strip()
            if not nxt:
                j += 1
                continue
            # Stop merging when both sides are already "healthy".
            if len(current) >= target_min and len(nxt) >= target_min:
                break
            projected = len(current) + 1 + len(nxt)
            if projected > max_chars:
                break
            current = f"{current} {nxt}"
            j += 1
            if len(current) >= min_chars and current.endswith(('.', '!', '?', '"', "'")):
                break
        merged.append(current)
        i = j + 1
    return merged


def find_cover(book):
    def is_image(item):
        return item is not None and item.media_type.startswith('image/')

    for item in book.get_items_of_type(ebooklib.ITEM_COVER):
        if is_image(item):
            return item

    for meta in book.get_metadata('OPF', 'cover'):
        if is_image(item := book.get_item_with_id(meta[1]['content'])):
            return item

    if is_image(item := book.get_item_with_id('cover')):
        return item

    for item in book.get_items_of_type(ebooklib.ITEM_IMAGE):
        if 'cover' in item.get_name().lower() and is_image(item):
            return item

    return None


def print_selected_chapters(document_chapters, chapters):
    ok = 'X' if platform.system() == 'Windows' else '✅'
    logging.info("\n" + tabulate([
        [i, c.get_name(), len(c.extracted_text), ok if c in chapters else '', chapter_beginning_one_liner(c)]
        for i, c in enumerate(document_chapters, start=1)
    ], headers=['#', 'Chapter', 'Text Length', 'Selected', 'First words']))


def write_audio_stream(chapter_wav_path: Path, chunks) -> int:
    """
    Write generated audio chunks directly to disk to avoid keeping entire chapters in memory.
    Returns the number of samples written.
    """
    samples_written = 0
    with soundfile.SoundFile(
        chapter_wav_path,
        mode="w",
        samplerate=sample_rate,
        channels=1,
        subtype="PCM_16",
    ) as wav_file:
        for chunk in chunks:
            if chunk is None:
                continue
            array = np.asarray(chunk, dtype=np.float32).flatten()
            if array.size == 0:
                continue
            wav_file.write(array)
            samples_written += array.shape[0]
    return samples_written


def convert_chapter_wav_to_m4a(source_path: Path) -> Path:
    """
    Convert the generated WAV file for a chapter into a compressed M4A file
    to lower disk usage before the next chapter starts.
    """
    destination = source_path.with_suffix(".m4a")
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-i",
        str(source_path),
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        str(destination),
    ]
    logging.info("Converting %s -> %s", source_path.name, destination.name)
    subprocess.run(ffmpeg_cmd, check=True)
    source_path.unlink(missing_ok=True)
    return destination


def gen_audio_segments(
    tts_resources,
    nlp,
    text,
    speed,
    stats=None,
    max_sentences=None,
    post_event=None,
    should_stop=None,
    repetition_penalty=1.2,
    min_p=0.05,
    top_p=1.0,
    top_k=None,
    exaggeration=0.5,
    cfg_weight=0.5,
    temperature=0.8,
    use_multilingual=False,
    language_id="en",
    audio_prompt_wav=None,
    sentence_gap_ms=0,
    question_gap_ms=0,
    force_sentence_gaps=True,
):  # Use spacy to split into sentences
    # NOTE (pipeline reminder):
    # 1) spacy segmenta in frasi.
    # 2) batch_sentences_intelligently compone batch di testo rispettando
    #    BATCH_MIN/MAX_CHARS e, se una singola frase eccede il max, la spezza
    #    ulteriormente su spazi (split_sentence_by_length).
    # 3) sentence_gap_ms e question_gap_ms vengono applicati tra i batch
    #    emessi, quindi valgono anche tra segmenti generati da una frase lunga.

    if should_stop is None:
        should_stop = lambda: False

    engine = tts_resources.get("engine", "qwen")

    doc = nlp(text)
    sentences = list(doc.sents)
    batch_min_chars = BATCH_MIN_CHARS
    batch_max_chars = BATCH_MAX_CHARS
    num_candidates = 3
    batches = batch_sentences_intelligently(
        sentences,
        min_chars=batch_min_chars,
        max_chars=batch_max_chars
    )
    raw_batch_total = len(batches)
    batches = coalesce_short_batches(
        batches,
        min_chars=batch_min_chars,
        max_chars=batch_max_chars,
    )
    if max_sentences and int(max_sentences) > 0:
        batches = batches[: int(max_sentences)]

    global _QWEN_MICROBATCH_DISABLED
    total_batches = len(batches)
    configured_microbatch_size = max(1, int(QWEN_MICROBATCH_SIZE))
    effective_microbatch_size = 1 if _QWEN_MICROBATCH_DISABLED else configured_microbatch_size
    logging.info(
        "BATCH_CONFIG min_chars=%s max_chars=%s microbatch_size=%s sentence_gap_ms=%s question_gap_ms=%s force_sentence_gaps=%s",
        batch_min_chars,
        batch_max_chars,
        effective_microbatch_size,
        sentence_gap_ms,
        question_gap_ms,
        force_sentence_gaps,
    )
    if raw_batch_total != len(batches):
        logging.info(
            "BATCH_COALESCE reduced_batches from=%s to=%s",
            raw_batch_total,
            len(batches),
        )
    logging.info(f"Split {len(sentences)} sentences into {total_batches} batches")

    # Show some batch examples
    for i, batch in enumerate(batches[:3]):
        logging.info(f"  Batch {i + 1} ({len(batch)} chars): {batch[:80]}{'...' if len(batch) > 80 else ''}")
    if total_batches > 3:
        logging.info(f"  ... and {total_batches - 3} more batches")

    if engine != "qwen":
        raise RuntimeError(f"Engine non supportato su worker_v3_qwen3_5tts: {engine}")

    chunk_index = 0
    microbatch_size = effective_microbatch_size
    while chunk_index < total_batches:
        if should_stop():
            logging.info("Synthesis interrupted by user (batch loop).")
            return

        group: List[Tuple[int, str]] = []
        while chunk_index < total_batches and len(group) < microbatch_size:
            batch_text = batches[chunk_index].strip()
            if batch_text:
                group.append((chunk_index, batch_text))
            chunk_index += 1
        if not group:
            continue

        texts = [item[1] for item in group]
        group_chars = sum(len(item) for item in texts)
        group_start = time.perf_counter()
        try:
            wav_arrays = synthesize_many_with_qwen(
                tts_resources,
                texts,
                language_id=language_id,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                top_k=top_k,
            )
        except Exception as exc:
            if len(texts) > 1 and not _QWEN_MICROBATCH_DISABLED:
                _QWEN_MICROBATCH_DISABLED = True
                microbatch_size = 1
                logging.warning(
                    "QWEN_MICROBATCH disabled for this worker process after runtime failure: %s",
                    exc,
                )
            logging.warning(
                "QWEN_MICROBATCH fallback to single-call mode (size=%s): %s",
                len(texts),
                exc,
            )
            wav_arrays = [
                synthesize_with_qwen(
                    tts_resources,
                    text,
                    language_id=language_id,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    top_k=top_k,
                )
                for text in texts
            ]
        group_synth_seconds = max(0.0, time.perf_counter() - group_start)
        group_audio_seconds = sum(
            float(np.asarray(wav).size) / float(sample_rate) for wav in wav_arrays
        )
        group_chars_per_second = (group_chars / group_synth_seconds) if group_synth_seconds > 0 else 0.0
        group_rtf = (group_audio_seconds / group_synth_seconds) if group_synth_seconds > 0 else 0.0
        if len(group) > 1:
            logging.info(
                "MICROBATCH_METRICS start=%s size=%s chars=%s synth_s=%.2f audio_s=%.2f char_per_s=%.2f rtf=%.2f",
                group[0][0] + 1,
                len(group),
                group_chars,
                group_synth_seconds,
                group_audio_seconds,
                group_chars_per_second,
                group_rtf,
            )

        for local_idx, (original_idx, batch_text) in enumerate(group):
            wav_array = wav_arrays[local_idx]
            current_batch = original_idx + 1
            chunk_chars = len(batch_text)
            audio_seconds = float(np.asarray(wav_array).size) / float(sample_rate) if wav_array is not None else 0.0
            if group_audio_seconds > 0:
                synth_seconds = group_synth_seconds * (audio_seconds / group_audio_seconds)
            else:
                synth_seconds = group_synth_seconds / max(1, len(group))
            chars_per_second = (chunk_chars / synth_seconds) if synth_seconds > 0 else 0.0
            realtime_factor = (audio_seconds / synth_seconds) if synth_seconds > 0 else 0.0
            logging.info(
                "CHUNK_METRICS current=%s total=%s chars=%s synth_s=%.2f audio_s=%.2f char_per_s=%.2f rtf=%.2f microbatch=%s",
                current_batch,
                total_batches,
                chunk_chars,
                synth_seconds,
                audio_seconds,
                chars_per_second,
                realtime_factor,
                len(group),
            )
            yield wav_array
            remaining_batches = max(0, total_batches - current_batch)
            logging.info(
                "CHUNK_PROGRESS current=%s total=%s remaining=%s",
                current_batch,
                total_batches,
                remaining_batches,
            )

            gap_duration = 0
            if force_sentence_gaps:
                gap_duration = sentence_gap_ms
            if question_gap_ms > 0 and batch_text.rstrip().endswith("?"):
                gap_duration = max(gap_duration, question_gap_ms)

            if gap_duration > 0 and current_batch < total_batches:
                gap_samples = int(sample_rate * (gap_duration / 1000.0))
                if gap_samples > 0:
                    yield np.zeros(gap_samples, dtype=np.float32)

            # Update statistics based on batch size
            if stats:
                update_stats(stats, len(batch_text))
                if post_event:
                    post_event('CORE_PROGRESS', stats=stats)
    return


def extract_chapter_number(chapter_name):
    """
    Extracts the chapter number from a chapter name like 'Text/Chapter_18.xhtml'.
    Returns the integer number, or a large number if not found, to sort non-matching
    chapters last.
    """
    match = re.search(r'(\d+)', chapter_name)
    if match:
        return int(match.group(1))
    return float('inf') # Return a large number for chapters that don't match

def find_document_chapters_and_extract_texts(book):
    """Returns every chapter that is an ITEM_DOCUMENT and enriches each chapter with extracted_text."""
    document_chapters = []
    for chapter in book.get_items():
        if chapter.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        xml = chapter.get_body_content()
        soup = BeautifulSoup(xml, features='lxml')
        chapter.extracted_text = ''
        html_content_tags = ['p', 'h1', 'h2', 'h3', 'h4', 'li']
        seen_body_content = False
        for node in soup.find_all(html_content_tags):
            text = node.text.strip() if node.text else ''
            if not text:
                continue
            if not seen_body_content and node.name in {'h1', 'h2', 'h3', 'h4'}:
                logging.info("Skipping leading chapter heading from spoken text: %s", text[:120])
                continue
            if node.name in {'p', 'li'}:
                seen_body_content = True
            if not text.endswith('.'):
                text += '.'
            chapter.extracted_text += text + '\n'
        document_chapters.append(chapter)

    # Sort chapters numerically based on their names
    document_chapters.sort(key=lambda c: extract_chapter_number(c.get_name()))

    for i, c in enumerate(document_chapters):
        c.chapter_index = i
    return document_chapters


class SimpleDocumentChapter:
    """Lightweight stand-in for ebooklib chapters so PDFs can share the same pipeline."""
    def __init__(self, name, extracted_text, chapter_index=0):
        self._name = name
        self.extracted_text = extracted_text
        self.chapter_index = chapter_index

    def get_name(self):
        return self._name

    def get_type(self):
        # Mimic ebooklib chapters so downstream filters keep working.
        return ebooklib.ITEM_DOCUMENT


def extract_pdf_chapters(file_path, title):
    """Read a PDF and return pseudo chapters (one per page) with extracted text."""
    chapters = []
    try:
        reader = PdfReader(file_path)
    except Exception as exc:
        logging.error("Failed to read PDF %s: %s", file_path, exc)
        raise

    for idx, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:
            logging.warning("Failed to extract text from page %s: %s", idx, exc)
            page_text = ""
        page_text = page_text.strip()
        if not page_text:
            continue
        chapter_name = f"{title}_page_{idx:03d}"
        chapters.append(SimpleDocumentChapter(chapter_name, page_text, chapter_index=len(chapters)))

    if not chapters:
        logging.warning("No text extracted from PDF %s; creating empty placeholder chapter.", file_path)
        chapters.append(SimpleDocumentChapter(f"{title}_full_document", "", chapter_index=0))

    for i, chapter in enumerate(chapters):
        chapter.chapter_index = i
    return chapters


def extract_txt_chapters(file_path: str, title: str):
    """Wrap a plain text file into a single SimpleDocumentChapter."""
    try:
        text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        logging.error("Failed to read TXT %s: %s", file_path, exc)
        raise
    text = text.strip()
    chapter = SimpleDocumentChapter(f"{title}_full_text", text, chapter_index=0)
    return [chapter]


def is_chapter(c):
    name = c.get_name().lower()
    has_min_len = len(c.extracted_text) > 100
    title_looks_like_chapter = bool(
        'chapter' in name.lower()
        or re.search(r'part_?\d{1,3}', name)
        or re.search(r'split_?\d{1,3}', name)
        or re.search(r'ch_?\d{1,3}', name)
        or re.search(r'chap_?\d{1,3}', name)
    )
    return has_min_len and title_looks_like_chapter


def chapter_beginning_one_liner(c, chars=20):
    s = c.extracted_text[:chars].strip().replace('\n', ' ').replace('\r', ' ')
    return s + '…' if len(s) > 0 else ''


def find_good_chapters(document_chapters):
    chapters = [c for c in document_chapters if c.get_type() == ebooklib.ITEM_DOCUMENT and is_chapter(c)]
    if len(chapters) == 0:
        logging.info('Not easy to recognize the chapters, defaulting to all non-empty documents.')
        chapters = [c for c in document_chapters if
                    c.get_type() == ebooklib.ITEM_DOCUMENT and len(c.extracted_text) > 10]
    return chapters


def pick_chapters(chapters):
    chapters_by_names = {
        f'{c.get_name()}\t({len(c.extracted_text)} chars)\t[{chapter_beginning_one_liner(c, 50)}]': c
        for c in chapters}
    title = 'Select which chapters to read in the audiobook'
    ret = pick(list(chapters_by_names.keys()), title, multiselect=True, min_selection_count=1)
    selected_chapters_out_of_order = [chapters_by_names[r[0]] for r in ret]
    selected_chapters = [c for c in chapters if c in selected_chapters_out_of_order]
    return selected_chapters


def strfdelta(tdelta, fmt='{D:02}d {H:02}h {M:02}m {S:02}s'):
    remainder = int(tdelta)
    f = Formatter()
    desired_fields = [field_tuple[1] for field_tuple in f.parse(fmt)]
    possible_fields = ('W', 'D', 'H', 'M', 'S')
    constants = {'W': 604800, 'D': 86400, 'H': 3600, 'M': 60, 'S': 1}
    values = {}
    for field in possible_fields:
        if field in desired_fields and field in constants:
            values[field], remainder = divmod(remainder, constants[field])
    return f.format(fmt, **values)


def enqueue_output(stream, queue_obj):
    """Helper function to read from a stream and put lines into a queue."""
    for line in iter(stream.readline, ''):
        queue_obj.put(line)
    stream.close()

MAX_PATH_LEN = 240

def safe_concat_path(output_folder: str, filename: str) -> Path:
    folder_path = Path(output_folder)
    name_part = Path(filename).stem
    suffix = Path(filename).suffix

    candidate = folder_path / f"{name_part}{suffix}"

    # If too long, truncate the base name until it fits
    while len(str(candidate.resolve())) > MAX_PATH_LEN and len(name_part) > 1:
        name_part = name_part[:-1]  # progressively shorten the name
        candidate = folder_path / f"{name_part}{suffix}"


    return candidate

def concat_wavs_with_ffmpeg(chapter_files, output_folder, filename, post_event=None, should_stop=None):
    base_filename_stem = Path(filename).stem
    wav_list_txt = Path(output_folder) / f"{base_filename_stem}_wav_list.txt"
    with open(wav_list_txt, 'w') as f:
        for wav_file in chapter_files:
            f.write(f"file '{str(wav_file)}'\n")

    concat_file_path = Path(output_folder) / f"{base_filename_stem}.tmp.mp4"

    ffmpeg_concat_cmd = [
        'ffmpeg',
        '-y',
        '-nostdin',  # <--- ADD THIS LINE
        '-f', 'concat',
        '-safe', '0',
        '-i', str(wav_list_txt),
        '-c:a', 'aac',
        '-b:a', '64k',
        '-progress', 'pipe:1',
        '-nostats',
        str(concat_file_path)
    ]

    logging.info(f"Running FFmpeg concat command: {' '.join(ffmpeg_concat_cmd)}")

    total_duration_seconds = sum(probe_duration(wav_file) for wav_file in chapter_files if wav_file.exists())
    logging.info(f"Concatenation Total Duration: {total_duration_seconds:.2f} seconds")

    process = subprocess.Popen(
        ffmpeg_concat_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    if should_stop is None:
        should_stop = lambda: False

    q_stdout = queue.Queue()
    q_stderr = queue.Queue()

    t_stdout = threading.Thread(target=enqueue_output, args=(process.stdout, q_stdout))
    t_stderr = threading.Thread(target=enqueue_output, args=(process.stderr, q_stderr))
    t_stdout.daemon = True
    t_stderr.daemon = True
    t_stdout.start()
    t_stderr.start()

    initial_stderr_lines = []
    # Drain initial STDERR output for a limited time or until queue is empty
    # This prevents blocking on large initial stderr bursts
    timeout_start = time.time()
    while (t_stderr.is_alive() or not q_stderr.empty()) and (
            time.time() - timeout_start < 5):  # DRAIN for max 5 seconds
        try:
            line = q_stderr.get_nowait().strip()
            if line:
                initial_stderr_lines.append(line)
                logging.error(f"FFmpeg CONCAT Initial STDERR: {line}")
        except queue.Empty:
            time.sleep(0.01)  # Small pause to yield CPU

    current_time_seconds = 0.0
    concat_error_output = initial_stderr_lines  # Start collecting from here

    try:
        while process.poll() is None or not q_stdout.empty() or not q_stderr.empty():
            if should_stop():
                logging.info("Synthesis interrupted by user (ffmpeg concat). Terminating FFmpeg process.")
                process.terminate()
                process.wait()
                return None
            # Process stdout for progress
            try:
                line_stdout = q_stdout.get(timeout=0.05)
                line_stdout = line_stdout.strip()
                # print(f"FFmpeg CONCAT STDOUT: {line_stdout}") # Debugging stdout output
                if "=" in line_stdout:
                    key, value = line_stdout.split("=", 1)
                    if key == "out_time":
                        try:
                            h, m, s = map(float, value.split(':'))
                            current_time_seconds = h * 3600 + m * 60 + s
                            if total_duration_seconds > 0:
                                progress = int((current_time_seconds / total_duration_seconds) * 100)
                                if post_event:
                                    stats_obj = SimpleNamespace(progress=progress, stage="concat", eta=strfdelta(
                                        total_duration_seconds - current_time_seconds))
                                    post_event('CORE_PROGRESS', stats=stats_obj)
                                # print(f"CONCAT Progress: {progress}% (Time: {current_time_seconds:.2f})") # More debugging
                        except ValueError:
                            pass
                    elif key == "progress" and value == "end":
                        break
            except queue.Empty:
                pass

            # Process stderr for errors/warnings
            try:
                line_stderr = q_stderr.get(timeout=0.05)
                stripped_line = line_stderr.strip()
                if stripped_line:
                    logging.error(f"FFmpeg CONCAT STDERR: {stripped_line}")
                    concat_error_output.append(stripped_line)
            except queue.Empty:
                pass

            time.sleep(0.001)  # Small sleep to avoid busy-waiting

    finally:
        # Final drain of queues
        while not q_stdout.empty():
            line_stdout = q_stdout.get_nowait().strip()
            if "=" in line_stdout:  # Still try to process any last progress updates
                key, value = line_stdout.split("=", 1)
                if key == "out_time":
                    try:
                        h, m, s = map(float, value.split(':'))
                        current_time_seconds = h * 3600 + m * 60 + s
                        if total_duration_seconds > 0:
                            progress = int((current_time_seconds / total_duration_seconds) * 100)
                            if post_event:
                                stats_obj = SimpleNamespace(progress=progress, stage="concat", eta=strfdelta(
                                    total_duration_seconds - current_time_seconds))
                                post_event('CORE_PROGRESS', stats=stats_obj)
                    except ValueError:
                        pass
        while not q_stderr.empty():
            stripped_line = q_stderr.get_nowait().strip()
            if stripped_line:
                logging.error(f"FFmpeg CONCAT STDERR (Post-loop): {stripped_line}")
                concat_error_output.append(stripped_line)

        process.wait()

    Path(wav_list_txt).unlink()

    if process.returncode != 0:
        error_message = f"FFmpeg concatenation failed with error code {process.returncode}.\nDetails:\n" + "\n".join(
            concat_error_output[-50:])
        logging.error(error_message)
        raise RuntimeError(error_message)

    return concat_file_path


def create_m4b(concat_file_path, filename, cover_image, output_folder, post_event=None, should_stop=None):
    logging.info('Creating M4B file...')

    original_name = Path(filename).with_suffix('').name  # removes old suffix


    new_name = f"{original_name}.m4b"

    final_filename = safe_concat_path(output_folder,new_name)
    chapters_txt_path = Path(output_folder) / "chapters.txt"
    logging.info('Creating M4B file...')

    ffmpeg_command = [
        'ffmpeg',
        '-y',
        '-nostdin',  # <--- ADD THIS LINE
        '-i', str(concat_file_path),
        '-i', str(chapters_txt_path),
    ]

    if cover_image:
        cover_file_path = Path(output_folder) / 'cover'
        with open(cover_file_path, 'wb') as f:
            f.write(cover_image)
        ffmpeg_command.extend([
            '-i', str(cover_file_path),
        ])
        map_video_index = '2:v'
        map_metadata_index = '2'
        map_chapters_index = '2'
    else:
        map_video_index = None
        map_metadata_index = '1'
        map_chapters_index = '1'

    ffmpeg_command.extend([
        '-map', '0:a',
        '-c:a', 'aac',
        '-b:a', '64k',
    ])

    if map_video_index:
        ffmpeg_command.extend([
            '-map', map_video_index,
            '-metadata:s:v', 'title="Album cover"',
            '-metadata:s:v', 'comment="Cover (front)"',
            '-disposition:v:0', 'attached_pic',
            '-c:v', 'copy'
        ])

    ffmpeg_command.extend([
        '-map_metadata', map_metadata_index,
        '-map_chapters', map_chapters_index,
        '-f', 'mp4',
        '-progress', 'pipe:1',
        '-nostats',
        str(final_filename)
    ])

    logging.info(f"Running FFmpeg command:\n{' '.join(ffmpeg_command)}\n")

    total_duration_seconds = probe_duration(concat_file_path)  # Changed to use Path object directly
    logging.info(f"M4B Conversion Total Duration: {total_duration_seconds:.2f} seconds")

    process = subprocess.Popen(
        ffmpeg_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    if should_stop is None:
        should_stop = lambda: False

    q_stdout = queue.Queue()
    q_stderr = queue.Queue()

    t_stdout = threading.Thread(target=enqueue_output, args=(process.stdout, q_stdout))
    t_stderr = threading.Thread(target=enqueue_output, args=(process.stderr, q_stderr))
    t_stdout.daemon = True
    t_stderr.daemon = True
    t_stdout.start()
    t_stderr.start()

    initial_stderr_lines = []
    # Drain initial STDERR output for a limited time or until queue is empty
    timeout_start = time.time()
    while (t_stderr.is_alive() or not q_stderr.empty()) and (
            time.time() - timeout_start < 5):  # DRAIN for max 5 seconds
        try:
            line = q_stderr.get_nowait().strip()
            if line:
                initial_stderr_lines.append(line)
                logging.error(f"FFmpeg M4B Initial STDERR: {line}")
        except queue.Empty:
            time.sleep(0.01)

    current_time_seconds = 0.0
    ffmpeg_error_output = initial_stderr_lines

    try:
        while process.poll() is None or not q_stdout.empty() or not q_stderr.empty():
            if should_stop():
                logging.info("Synthesis interrupted by user (ffmpeg m4b). Terminating FFmpeg process.")
                process.terminate()
                process.wait()
                return
            # Process stdout for progress
            try:
                line_stdout = q_stdout.get(timeout=0.05)
                line_stdout = line_stdout.strip()
                # print(f"FFmpeg M4B STDOUT: {line_stdout}") # Debugging stdout output
                if "=" in line_stdout:
                    key, value = line_stdout.split("=", 1)
                    if key == "out_time":
                        try:
                            h, m, s = map(float, value.split(':'))
                            current_time_seconds = h * 3600 + m * 60 + s
                            if total_duration_seconds > 0:
                                progress = int((current_time_seconds / total_duration_seconds) * 100)
                                if post_event:
                                    stats_obj = SimpleNamespace(progress=progress, stage="ffmpeg", eta=strfdelta(
                                        total_duration_seconds - current_time_seconds))
                                    post_event('CORE_PROGRESS', stats=stats_obj)
                                # print(f"M4B Progress: {progress}% (Time: {current_time_seconds:.2f})") # More debugging
                        except ValueError:
                            pass
                    elif key == "progress" and value == "end":
                        break
            except queue.Empty:
                pass

            # Process stderr for errors/warnings
            try:
                line_stderr = q_stderr.get(timeout=0.05)
                stripped_line = line_stderr.strip()
                if stripped_line:
                    logging.error(f"FFmpeg M4B STDERR: {stripped_line}")
                    ffmpeg_error_output.append(stripped_line)
            except queue.Empty:
                pass

            time.sleep(0.001)

    finally:
        # Final drain of queues
        while not q_stdout.empty():
            line_stdout = q_stdout.get_nowait().strip()
            if "=" in line_stdout:
                key, value = line_stdout.split("=", 1)
                if key == "out_time":
                    try:
                        h, m, s = map(float, value.split(':'))
                        current_time_seconds = h * 3600 + m * 60 + s
                        if total_duration_seconds > 0:
                            progress = int((current_time_seconds / total_duration_seconds) * 100)
                            if post_event:
                                stats_obj = SimpleNamespace(progress=progress, stage="ffmpeg", eta=strfdelta(
                                    total_duration_seconds - current_time_seconds))
                                post_event('CORE_PROGRESS', stats=stats_obj)
                    except ValueError:
                        pass
        while not q_stderr.empty():
            stripped_line = q_stderr.get_nowait().strip()
            if stripped_line:
                logging.error(f"FFmpeg M4B STDERR (Post-loop): {stripped_line}")
                ffmpeg_error_output.append(stripped_line)

        process.wait()

    Path(concat_file_path).unlink()
    if process.returncode == 0:
        logging.info(f'{final_filename} created. Enjoy your audiobook.')
    else:
        error_message = f"FFmpeg process exited with error code {process.returncode}.\nDetails:\n" + "\n".join(
            ffmpeg_error_output[-50:])
        logging.error(error_message)
        raise RuntimeError(error_message)


def probe_duration(file_name):
    # Check if the file exists before probing, to prevent errors if file was not created
    if not Path(file_name).exists():
        logging.warning(f"Warning: File not found for ffprobe duration: {file_name}")
        return 0.0

    args = ['ffprobe', '-i', str(file_name), '-show_entries', 'format=duration', '-v', 'quiet', '-of',
            'default=noprint_wrappers=1:nokey=1']
    try:
        # Using CREATE_NO_WINDOW on Windows to prevent console flashing for ffprobe
        creation_flags = subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
        proc = subprocess.run(args, capture_output=True, text=True, check=True, creationflags=creation_flags)
        duration = float(proc.stdout.strip())
        return duration
    except subprocess.CalledProcessError as e:
        logging.error(f"Error probing duration for {file_name}: {e.stderr}")
        return 0.0
    except ValueError:  # Occurs if stdout is not a float (e.g., empty or error message)
        logging.error(f"Could not parse duration from ffprobe output for {file_name}: '{proc.stdout.strip()}'")
        return 0.0


def create_index_file(title, creator, chapter_mp3_files, output_folder):
    with open(Path(output_folder) / "chapters.txt", "w", encoding="ascii", newline="\n") as f:
        f.write(f";FFMETADATA1\ntitle={title}\nartist={creator}\n\n")
        start = 0
        i = 0
        for c in chapter_mp3_files:
            duration = probe_duration(c)
            end = start + (int)(duration * 1000)
            f.write(f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={start}\nEND={end}\ntitle=Chapter {i}\n\n")
            i += 1
            start = end


def unmark_element(element, stream=None):
    if stream is None:
        stream = StringIO()
    if element.text:
        stream.write(element.text)
    for sub in element:
        unmark_element(sub, stream)
    if element.tail:
        stream.write(element.tail)
    return stream.getvalue()


def unmark(text):
    return text
