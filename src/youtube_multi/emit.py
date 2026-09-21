"""Writers: paired.md, paired.json, manifest.json."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .models import Chunk, hms


def write_markdown(
    chunks: list[Chunk],
    out_path: Path,
    source_url: str,
    video_id: str,
) -> None:
    lines: list[str] = [
        f"# YouTube paired transcript — {video_id}",
        "",
        f"Source: {source_url}",
        "",
        "---",
        "",
    ]
    for chunk in chunks:
        lines.append(f"## {hms(chunk.start_seconds)}")
        lines.append("")
        lines.append(chunk.text)
        lines.append("")
        for scene in chunk.scenes:
            rel = scene.image_path.relative_to(out_path.parent).as_posix()
            lines.append(f"![scene {scene.idx:04d} @ {hms(scene.t_seconds)}]({rel})")
            if scene.ocr:
                lines.append(f"> OCR: {scene.ocr}")
            lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_json(
    chunks: list[Chunk],
    out_path: Path,
    source_url: str,
    video_id: str,
) -> None:
    payload = {
        "source_url": source_url,
        "video_id": video_id,
        "chunks": [
            {
                "start_seconds": c.start_seconds,
                "start_hms": hms(c.start_seconds),
                "text": c.text,
                "frames": [
                    {
                        "idx": s.idx,
                        "t_seconds": round(s.t_seconds, 2),
                        "image": s.image_path.relative_to(out_path.parent).as_posix(),
                        "ocr": s.ocr,
                    }
                    for s in c.scenes
                ],
            }
            for c in chunks
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# --- manifest -----------------------------------------------------------------

_LIBS = ["yt-dlp", "scenedetect", "opencv-python", "pytesseract", "pillow", "youtube-transcript-api"]


def _lib_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _tool_version(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    text = (out.stdout or out.stderr).strip().splitlines()
    return text[0] if text else None


def _tesseract_version() -> str | None:
    try:
        import pytesseract

        return str(pytesseract.get_tesseract_version())
    except Exception:
        return None


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def build_manifest(
    *,
    args: dict[str, Any],
    video_id: str,
    source_url: str,
    video_path: Path | None,
    ocr_enabled: bool,
) -> dict[str, Any]:
    video: dict[str, Any] | None = None
    if video_path is not None and video_path.is_file():
        video = {
            "path": video_path.name,
            "sha256": sha256_file(video_path),
            "bytes": video_path.stat().st_size,
        }
    return {
        "tool": "multi",
        "version": _lib_version("youtube-multi"),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "video_id": video_id,
        "source_url": source_url,
        "args": _jsonable(args),
        "video": video,
        "libs": {name: _lib_version(name) for name in _LIBS},
        "tools": {
            "ffmpeg": _tool_version(["ffmpeg", "-version"]),
            "tesseract": _tesseract_version() if ocr_enabled else None,
        },
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def write_manifest(manifest: dict[str, Any], out_path: Path) -> None:
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
