"""Navigation layer: chapters, token estimates, frame scoring and budget tiers."""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import Any

from .enrich import hamming
from .models import Chunk, Scene

# --- token estimates ----------------------------------------------------------------------
# Rough, provider-agnostic: ~4 chars per text token; images ~ w*h/750 tokens
# (Anthropic's published rule of thumb), capped at what a 1568px-fitted image
# costs. Good enough for relative budgeting, not for billing.

IMAGE_TOKEN_CAP = 1600


def estimate_text_tokens(text: str) -> int:
    return math.ceil(len(text) / 4) if text else 0


def estimate_image_tokens(width: int, height: int) -> int:
    if width <= 0 or height <= 0:
        return 0
    return min(IMAGE_TOKEN_CAP, math.ceil(width * height / 750))


def frame_heading(scene: Scene, max_len: int = 80) -> str:
    for line in scene.ocr_lines:
        line = line.strip()
        if len(line) >= 3:
            return line if len(line) <= max_len else line[: max_len - 1] + "…"
    return scene.kind


# --- scoring / tiers ---------------------------------------------------------------------------


def score_frames(kept: list[Scene], chunks: list[Chunk]) -> None:
    """score = 0.5*novelty + 0.3*ocr_density + 0.2*coverage.

    novelty: pHash distance from the previous kept frame (first frame = 1).
    ocr_density: OCR chars / 1500, clipped.
    coverage: 1 for the best frame (by the other two terms) in each chunk, so
    every chunk keeps at least one representative near the top.
    """
    prev: Scene | None = None
    base: dict[int, float] = {}
    for s in kept:
        novelty = 1.0 if prev is None else min(1.0, hamming(prev.phash, s.phash) / 64.0 * 2.0)
        density = min(1.0, len(s.ocr) / 1500.0)
        base[s.idx] = 0.5 * novelty + 0.3 * density
        prev = s
    for chunk in chunks:
        best = max((s for s in chunk.scenes if s.kept), key=lambda s: base.get(s.idx, 0.0), default=None)
        for s in chunk.scenes:
            if not s.kept:
                continue
            coverage = 1.0 if s is best else 0.0
            s.score = round(base.get(s.idx, 0.0) + 0.2 * coverage, 3)
    # frames not attached to any chunk (no transcript) keep base only
    for s in kept:
        if s.idx in base and s.score == 0.0:
            s.score = round(base[s.idx], 3)


def assign_tiers(kept: list[Scene], budget: int | None) -> dict[str, Any]:
    """Greedy by score: tier 1 while cumulative tokens <= budget, tier 2 while
    <= 2*budget, else 3. No budget -> everything is tier 1."""
    if budget is None:
        for s in kept:
            s.tier = 1
    else:
        total = 0
        for s in sorted(kept, key=lambda s: (-s.score, s.t_seconds)):
            total += s.tokens_est
            s.tier = 1 if total <= budget else 2 if total <= 2 * budget else 3
    by_tier = {1: 0, 2: 0, 3: 0}
    counts = {1: 0, 2: 0, 3: 0}
    for s in kept:
        by_tier[s.tier] += s.tokens_est
        counts[s.tier] += 1
    return {
        "budget": budget,
        "tokens_by_tier": {str(k): v for k, v in by_tier.items()},
        "frames_by_tier": {str(k): v for k, v in counts.items()},
    }


# --- chapters -------------------------------------------------------------------------------------


@dataclass
class Chapter:
    index: int
    title: str
    start: float
    end: float
    source: str  # "youtube" | "synthesized"
    chunk_indices: list[int] = field(default_factory=list)
    frames: list[Scene] = field(default_factory=list)
    tokens_text: int = 0
    tokens_images: int = 0

    @property
    def tokens_total(self) -> int:
        return self.tokens_text + self.tokens_images


SYNTH_CHAPTER_SECONDS = 180.0
SINGLE_CHAPTER_MAX_SECONDS = 360.0


def _title_from(chunk: Chunk, words: int = 7) -> str:
    for s in chunk.scenes:
        if s.kept and s.ocr_lines:
            return frame_heading(s, max_len=60)
    parts = chunk.text.split()
    return " ".join(parts[:words]) + ("…" if len(parts) > words else "")


def build_chapters(metadata: dict[str, Any] | None, chunks: list[Chunk], duration: float) -> list[Chapter]:
    raw = (metadata or {}).get("chapters") or []
    chapters: list[Chapter] = []
    if raw:
        for i, ch in enumerate(raw):
            start = float(ch.get("start_time") or 0.0)
            end = float(ch.get("end_time") or (raw[i + 1].get("start_time") if i + 1 < len(raw) else duration) or duration)
            chapters.append(Chapter(index=i, title=str(ch.get("title") or f"Chapter {i + 1}"), start=start, end=end, source="youtube"))
    elif duration <= SINGLE_CHAPTER_MAX_SECONDS or len(chunks) < 2:
        title = (metadata or {}).get("title") or "Full video"
        chapters.append(Chapter(index=0, title=title, start=0.0, end=duration, source="synthesized"))
    else:
        # Cut at chunk boundaries roughly every SYNTH_CHAPTER_SECONDS.
        starts = [float(c.start_seconds) for c in chunks]
        cut_at = 0.0
        idx = 0
        bounds: list[tuple[float, Chunk]] = []
        while cut_at < duration:
            i = bisect.bisect_left(starts, cut_at)
            if i >= len(chunks):
                break
            if not bounds or chunks[i] is not bounds[-1][1]:
                bounds.append((starts[i], chunks[i]))
            cut_at = starts[i] + SYNTH_CHAPTER_SECONDS
        for (start, chunk), nxt in zip(bounds, bounds[1:] + [(duration, None)]):
            chapters.append(Chapter(index=idx, title=_title_from(chunk), start=start, end=float(nxt[0]), source="synthesized"))
            idx += 1
    if chapters:
        chapters[0].start = 0.0
        chapters[-1].end = max(chapters[-1].end, duration)
    return chapters


def populate_chapters(chapters: list[Chapter], chunks: list[Chunk]) -> None:
    """Attach chunks (by chunk start) and kept frames (by frame time) to chapters; sum tokens.

    A chunk that straddles a chapter boundary is listed under the chapter it starts in,
    but its frames go to the chapter they actually appear in."""
    if not chapters:
        return
    starts = [c.start for c in chapters]

    def _chapter_for(t: float) -> Chapter:
        return chapters[max(0, bisect.bisect_right(starts, t) - 1)]

    for ci, chunk in enumerate(chunks):
        ch = _chapter_for(float(chunk.start_seconds))
        ch.chunk_indices.append(ci)
        ch.tokens_text += estimate_text_tokens(chunk.text)
        for s in chunk.scenes:
            if s.kept:
                target = _chapter_for(s.t_seconds)
                target.frames.append(s)
                target.tokens_images += s.tokens_est
    for ch in chapters:
        ch.frames.sort(key=lambda s: s.t_seconds)


def chapter_json(ch: Chapter, chunks: list[Chunk], video_id: str, deep_link) -> dict[str, Any]:
    return {
        "index": ch.index,
        "title": ch.title,
        "start": round(ch.start, 2),
        "end": round(ch.end, 2),
        "url": deep_link(video_id, ch.start),
        "source": ch.source,
        "chunk_ids": ch.chunk_indices,
        "chunk_starts": [chunks[i].start_seconds for i in ch.chunk_indices],
        "frame_ids": [s.idx for s in ch.frames],
        "tokens": {"text": ch.tokens_text, "images": ch.tokens_images, "total": ch.tokens_total},
        "tokens_tier1": ch.tokens_text + sum(s.tokens_est for s in ch.frames if s.tier == 1),
        "visual_states": [
            {
                "frame_idx": s.idx,
                "kind": s.kind,
                "kind_confidence": s.kind_confidence,
                "heading": frame_heading(s),
                "visible_from": round(s.visible_from, 2),
                "visible_to": round(s.visible_to, 2),
                "url": deep_link(video_id, s.t_seconds),
                "tokens_est": s.tokens_est,
                "score": s.score,
                "tier": s.tier,
                "image": None,  # filled by emit with the relative path
            }
            for s in ch.frames
        ],
    }
