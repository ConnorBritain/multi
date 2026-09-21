"""Core data types and pure helpers shared across the multi pipeline."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

KINDS = ("slide", "code", "terminal", "diagram", "talking-head", "whiteboard", "other")


@dataclass
class Diff:
    """Delta between a frame and the previous kept frame of the same kind."""

    vs_idx: int
    text_path: Path | None = None
    image_path: Path | None = None
    added: int = 0
    removed: int = 0
    changed_region: tuple[int, int, int, int] | None = None  # x, y, w, h
    changed_frac: float = 0.0


@dataclass
class Scene:
    idx: int
    t_seconds: float
    image_path: Path
    ocr: str = ""
    # --- Phase 1 enrichment ---
    ocr_lines: list[str] = field(default_factory=list)
    ocr_confidence: float = 0.0
    phash: int = 0
    dhash: int = 0
    visible_from: float = 0.0
    visible_to: float = 0.0
    kind: str = "other"
    kind_confidence: float = 0.0
    features: dict[str, float] = field(default_factory=dict)
    diff: Diff | None = None
    dropped_reason: str | None = None  # "duplicate" | "kind:<kind>"
    duplicate_of: int | None = None

    @property
    def kept(self) -> bool:
        return self.dropped_reason is None


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
