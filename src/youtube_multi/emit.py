"""Writers: paired.md, paired.json, manifest.json."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .models import Chunk, Scene, hms


def _rel(path: Path | None, out_dir: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.relative_to(out_dir).as_posix()
    except ValueError:
        return path.as_posix()


def _scene_md_lines(scene: Scene, out_dir: Path) -> list[str]:
    rel = _rel(scene.image_path, out_dir)
    lines = [f"![scene {scene.idx:04d} @ {hms(scene.t_seconds)}]({rel})"]
    meta = f"> kind: {scene.kind} ({scene.kind_confidence:.2f})"
    if scene.visible_to > scene.visible_from:
        meta += f" · visible {hms(scene.visible_from)}–{hms(scene.visible_to)}"
    lines.append(meta)
    if scene.ocr:
        lines.append(f"> OCR: {scene.ocr}")
    if scene.diff is not None:
        d = scene.diff
        refs = ", ".join(p for p in (_rel(d.text_path, out_dir), _rel(d.image_path, out_dir)) if p)
        parts = [f"> diff vs {d.vs_idx:04d}: +{d.added} −{d.removed} lines"]
        if d.changed_region:
            parts.append(f"{d.changed_frac:.0%} of pixels changed")
        line = ", ".join(parts)
        if refs:
            line += f" ({refs})"
        lines.append(line)
    lines.append("")
    return lines


def write_markdown(
    chunks: list[Chunk],
    out_path: Path,
    source_url: str,
    video_id: str,
) -> None:
    out_dir = out_path.parent
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
            lines.extend(_scene_md_lines(scene, out_dir))
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _frame_json(s: Scene, out_dir: Path) -> dict[str, Any]:
    d: dict[str, Any] = {
        "idx": s.idx,
        "t_seconds": round(s.t_seconds, 2),
        "image": _rel(s.image_path, out_dir),
        "ocr": s.ocr,
        "ocr_lines": s.ocr_lines,
        "ocr_confidence": s.ocr_confidence,
        "kind": s.kind,
        "kind_confidence": s.kind_confidence,
        "phash": f"{s.phash:016x}",
        "visible_from": round(s.visible_from, 2),
        "visible_to": round(s.visible_to, 2),
        "diff": None,
    }
    if s.diff is not None:
        d["diff"] = {
            "vs": s.diff.vs_idx,
            "text": _rel(s.diff.text_path, out_dir),
            "image": _rel(s.diff.image_path, out_dir),
            "added": s.diff.added,
            "removed": s.diff.removed,
            "changed_region": list(s.diff.changed_region) if s.diff.changed_region else None,
            "changed_frac": s.diff.changed_frac,
        }
    return d


def frame_stats(all_scenes: list[Scene]) -> dict[str, Any]:
    kept = [s for s in all_scenes if s.kept]
    dropped_by_kind = Counter(s.kind for s in all_scenes if s.dropped_reason and s.dropped_reason.startswith("kind:"))
    return {
        "extracted": len(all_scenes),
        "duplicates": sum(1 for s in all_scenes if s.dropped_reason == "duplicate"),
        "dropped_by_kind": dict(sorted(dropped_by_kind.items())),
        "kept": len(kept),
        "kept_by_kind": dict(sorted(Counter(s.kind for s in kept).items())),
    }


def write_json(
    chunks: list[Chunk],
    out_path: Path,
    source_url: str,
    video_id: str,
    all_scenes: list[Scene] | None = None,
) -> None:
    out_dir = out_path.parent
    all_scenes = all_scenes if all_scenes is not None else [s for c in chunks for s in c.scenes]
    payload: dict[str, Any] = {
        "source_url": source_url,
        "video_id": video_id,
        "manifest": "manifest.json",
        "frame_stats": frame_stats(all_scenes),
        "chunks": [
            {
                "start_seconds": c.start_seconds,
                "start_hms": hms(c.start_seconds),
                "text": c.text,
                "frames": [_frame_json(s, out_dir) for s in c.scenes],
            }
            for c in chunks
        ],
        "dropped_frames": [
            {
                "idx": s.idx,
                "t_seconds": round(s.t_seconds, 2),
                "reason": s.dropped_reason,
                "duplicate_of": s.duplicate_of,
                "kind": s.kind,
                "image": _rel(s.image_path, out_dir) if s.image_path.exists() else None,
            }
            for s in all_scenes
            if not s.kept
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# --- manifest -----------------------------------------------------------------

_LIBS = ["yt-dlp", "scenedetect", "opencv-python", "numpy", "pytesseract", "pillow", "youtube-transcript-api"]


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
