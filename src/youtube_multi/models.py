"""Core data types and pure helpers shared across the multi pipeline."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Scene:
    idx: int
    t_seconds: float
    image_path: Path
    ocr: str = ""


@dataclass
class Chunk:
    start_seconds: int
    text: str
    scenes: list[Scene] = field(default_factory=list)


_VIDEO_ID_RE = re.compile(r"(?:v=|youtu\.be/|/embed/|/shorts/)([A-Za-z0-9_-]{11})")


def extract_video_id(url: str) -> str:
    m = _VIDEO_ID_RE.search(url)
    if not m:
        raise ValueError(f"could not extract video id from URL: {url}")
    return m.group(1)


def hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def short_ms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60:02d}m{s % 60:02d}s"
