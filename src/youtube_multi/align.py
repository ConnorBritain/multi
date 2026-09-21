"""Alignment: attach frames to transcript chunks, sentences and cues by time."""
from __future__ import annotations

import bisect
import re

from .models import Chunk, Cue, Scene, Sentence


def pair_scenes(scenes: list[Scene], chunks: list[Chunk]) -> list[Chunk]:
    """Append each scene to the last chunk whose start is <= the scene time.

    Scenes earlier than the first chunk go into the first chunk.
    """
    if not chunks:
        return []
    starts = [c.start_seconds for c in chunks]
    for scene in scenes:
        idx = 0
        for i, s in enumerate(starts):
            if scene.t_seconds >= s:
                idx = i
            else:
                break
        chunks[idx].scenes.append(scene)
    return chunks


# --- sentences ---------------------------------------------------------------------------

_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "eg", "ie",
    "u.s", "u.k", "no", "fig", "inc", "ltd", "co", "approx", "min", "max", "sec",
}
# Split at whitespace that follows end punctuation (optionally a closing quote /
# bracket) and precedes a capital or digit. Two fixed-width lookbehinds keep the
# closing quote attached to the sentence it ends.
_SPLIT_RE = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"')\]]))\s+(?=[\"'(\[]?[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation followed by whitespace and a capital
    or digit, re-joining pieces that end in a common abbreviation."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    pieces = _SPLIT_RE.split(text)
    out: list[str] = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if out:
            last_word = out[-1].rsplit(" ", 1)[-1].rstrip(".!?").lower()
            if last_word in _ABBREVIATIONS or (len(last_word) == 1 and last_word.isalpha()):
                out[-1] = f"{out[-1]} {piece}"
                continue
        out.append(piece)
    return out


def assign_cues(cues: list[Cue], chunks: list[Chunk]) -> None:
    """Attach each cue to the last chunk whose start is <= the cue start."""
    if not chunks:
        return
    starts = [c.start_seconds for c in chunks]
    for cue in cues:
        idx = max(0, bisect.bisect_right(starts, cue.start) - 1)
        chunks[idx].cues.append(cue)


def _time_at_fraction(cues: list[Cue], frac: float, fallback_start: float, fallback_end: float) -> float:
    """Map a 0..1 position through the chunk's text onto the cue timeline
    (piecewise by cue character counts), or linearly across the fallback span."""
    frac = max(0.0, min(1.0, frac))
    if not cues:
        return fallback_start + frac * (fallback_end - fallback_start)
    total = sum(len(c.text) + 1 for c in cues)
    target = frac * total
    acc = 0.0
    for c in cues:
        n = len(c.text) + 1
        if target <= acc + n or c is cues[-1]:
            inner = (target - acc) / n if n else 0.0
            return c.start + max(0.0, min(1.0, inner)) * c.duration
        acc += n
    return cues[-1].end


def time_sentences(chunks: list[Chunk], duration: float) -> None:
    """Split each chunk into sentences and give each a start time interpolated
    across the chunk's cues (or its time span when there are no cues)."""
    for i, chunk in enumerate(chunks):
        span_end = float(chunks[i + 1].start_seconds) if i + 1 < len(chunks) else max(duration, float(chunk.start_seconds))
        sentences = split_sentences(chunk.text)
        total_chars = max(sum(len(s) + 1 for s in sentences), 1)
        chunk.sentences = []
        offset = 0
        for j, s in enumerate(sentences):
            t = float(chunk.start_seconds) if j == 0 else _time_at_fraction(chunk.cues, offset / total_chars, chunk.start_seconds, span_end)
            chunk.sentences.append(Sentence(index=j, start=round(max(t, float(chunk.start_seconds)), 2), text=s))
            offset += len(s) + 1


def place_scenes(chunks: list[Chunk], all_cues: list[Cue]) -> None:
    """Set scene.sentence_index (sentence the frame appeared during) and
    scene.cue_ids_visible (cues overlapping the frame's visible range)."""
    cue_starts = [c.start for c in all_cues]
    for chunk in chunks:
        starts = [s.start for s in chunk.sentences]
        for scene in chunk.scenes:
            scene.sentence_index = bisect.bisect_right(starts, scene.t_seconds) - 1 if starts else -1
            lo, hi = scene.visible_from, max(scene.visible_to, scene.visible_from)
            # cues that start before the frame disappears and end after it appears
            first = bisect.bisect_right(cue_starts, hi)
            scene.cue_ids_visible = [c.id for c in all_cues[:first] if c.end > lo and (c.start < hi or hi == lo)]


def align(scenes: list[Scene], chunks: list[Chunk], cues: list[Cue], duration: float) -> list[Chunk]:
    """Full Phase 2 alignment: chunks <- scenes, cues, sentences; scenes <- sentence/cue refs."""
    pair_scenes(scenes, chunks)
    assign_cues(cues, chunks)
    time_sentences(chunks, duration)
    place_scenes(chunks, cues)
    return chunks
