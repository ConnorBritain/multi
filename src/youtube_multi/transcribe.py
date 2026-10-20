"""Optional, local-only speech recognition. No paid API adapters."""
from __future__ import annotations

import importlib
import json
import math
from pathlib import Path

from .emit import sha256_file
from .fetch import aggregate_cues, cues_path_for, cues_to_chunks, write_cues_file, write_transcript_file
from .models import Cue


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def source_path_for(transcript: Path) -> Path:
    return transcript.with_suffix(".source.json")


def save_transcript(cues: list[Cue], path: Path, chunk_seconds: float, source: dict) -> dict:
    if not cues or not any(c.text.strip() for c in cues):
        raise ValueError("No speech/captions found; no transcript was saved")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    write_transcript_file(aggregate_cues(cues_to_chunks(cues), chunk_seconds), temp)
    cue_path = cues_path_for(path)
    cue_temp = cue_path.with_suffix(".json.tmp")
    write_cues_file(cues, cue_temp)
    # The source record is the commit marker and includes hashes of both files.
    temp.replace(path)
    cue_temp.replace(cue_path)
    record = {**source, "transcript_sha256": sha256_file(path), "cues_sha256": sha256_file(cue_path)}
    atomic_json(source_path_for(path), record)
    return record


def read_source(path: Path) -> dict:
    record_path = source_path_for(path)
    if not record_path.exists():
        return {"method": "supplied_or_legacy", "timing": "unknown"}
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError(f"Invalid transcript provenance: {record_path}")
    if record.get("transcript_sha256") != sha256_file(path):
        return {"method": "edited", "original_method": record.get("method"), "timing": "estimated"}
    return record


def transcribe_audio(
    audio: Path, transcript: Path, *, model_name: str = "small.en", device: str = "cpu",
    compute_type: str = "int8", language: str | None = "en", cpu_threads: int = 4,
    chunk_seconds: float = 30, local_files_only: bool = False,
) -> dict:
    if not audio.is_file():
        raise ValueError(f"Audio file does not exist: {audio}")
    signature = {
        "method": "faster_whisper", "model": model_name, "device": device,
        "compute_type": compute_type, "language": language,
        "audio_sha256": sha256_file(audio), "chunk_seconds": chunk_seconds,
        "beam_size": 5, "vad_filter": True,
    }
    cue_path = cues_path_for(transcript)
    if transcript.is_file() and cue_path.is_file():
        old = read_source(transcript)
        if all(old.get(k) == v for k, v in signature.items()) and old.get("cues_sha256") == sha256_file(cue_path):
            return old
    try:
        backend = importlib.import_module("faster_whisper")
    except ImportError as exc:
        raise RuntimeError("Local transcription requires: uv sync --extra whisper") from exc
    model = backend.WhisperModel(
        model_name, device=device, compute_type=compute_type, cpu_threads=cpu_threads,
        local_files_only=local_files_only,
    )
    segments, info = model.transcribe(str(audio), language=language, beam_size=5, vad_filter=True)
    cues = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        start, end = float(segment.start), float(segment.end)
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
            raise ValueError("Speech recognizer returned invalid timestamps")
        cues.append(Cue(len(cues), start, end - start, text))
    source = {
        **signature, "timing": "asr_segments", "duration": float(info.duration),
        "detected_language": info.language, "api_cost_usd": 0,
    }
    return save_transcript(cues, transcript, chunk_seconds, source)
