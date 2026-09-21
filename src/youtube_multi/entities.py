"""Entity glossary: identifiers, commands, URLs, paths and product names seen in OCR and transcript."""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import Chunk, Cue, Scene, deep_link, hms
from .reconstruct import extract_commands

_URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s<>\"')\]]+|\b[\w-]+(?:\.[\w-]+)+\.(?:com|org|io|dev|net|ai|co|app|sh)(?:/[^\s<>\"')\]]*)?", re.I)
_PATH_RE = re.compile(r"(?<![\w:])(?:~?/[\w.-]+(?:/[\w.-]+)+|\.{1,2}/[\w.-]+(?:/[\w.-]+)*|[A-Z]:\\[\w\\.-]+|(?:src|lib|app|tests?|docs?|scripts?|config)/[\w./-]+)")
_IDENT_RE = re.compile(r"\b(?:[a-z][a-z0-9]*(?:_[a-z0-9]+)+|[a-z][a-z0-9]*(?:[A-Z][a-z0-9]+)+|(?:[A-Z][a-z0-9]+){2,}|[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+(?:\(\))?)\b")
_PRODUCT_RE = re.compile(r"\b[A-Z][a-zA-Z0-9+#]{2,}(?:\s[A-Z][a-zA-Z0-9]+)?\b|\b[A-Z]{2,6}\b")
_SPOKEN_CMD_RE = re.compile(r"\b((?:npm|npx|pnpm|yarn|pip|pipx|uv|git|docker|kubectl|cargo|go|make|brew|apt|curl|python|node|bun|deno)\s+[a-z][\w-]*)\b")

_STOP_CAPS = {
    "The", "This", "That", "These", "Those", "There", "Then", "They", "Them", "Their", "And", "But", "For", "Not",
    "Now", "How", "What", "When", "Where", "Which", "Who", "Why", "With", "You", "Your", "Our", "One", "Two", "Three",
    "First", "Next", "Last", "Also", "Just", "Like", "Let", "Okay", "Yes", "Yeah", "See", "Good", "Great", "Here",
    "Well", "Right", "Left", "Top", "Bottom", "Because", "Before", "After", "All", "Any", "Some", "Most", "Every",
    "Each", "Both", "Even", "Only", "Very", "Really", "Again", "Still", "Once", "Maybe", "Sure", "Thanks", "Thank",
    "Hello", "Today", "Tomorrow", "Yesterday", "Something", "Nothing", "Everything", "Anything", "Instead", "Actually",
    "OCR", "URL", "OK", "AND", "THE", "FOR", "NOT", "YOU", "ALL", "NEW", "USE",
}
_MIN_MENTIONS_PRODUCT = 2


@dataclass
class Entity:
    name: str
    type: str  # identifier | command | url | path | product
    mentions: int = 0
    first_spoken: float | None = None
    first_on_screen: float | None = None
    first_frame: int | None = None
    frames: list[int] = field(default_factory=list)
    spoken_in_chunks: list[int] = field(default_factory=list)


def _candidates(text: str, *, source: str) -> list[tuple[str, str]]:
    """(name, type) candidates in one piece of text. Higher-precision types first
    so a URL isn't also reported as a dotted identifier."""
    found: list[tuple[str, str]] = []
    taken: list[tuple[int, int]] = []

    def take(m: re.Match[str], typ: str, name: str | None = None) -> None:
        span = m.span()
        if any(a <= span[0] < b or a < span[1] <= b for a, b in taken):
            return
        taken.append(span)
        found.append(((name or m.group(0)).rstrip(".,;:"), typ))

    for m in _URL_RE.finditer(text):
        take(m, "url")
    for m in _PATH_RE.finditer(text):
        take(m, "path")
    if source == "ocr":
        for cmd in extract_commands([text]):
            found.append((cmd, "command"))
    else:
        for m in _SPOKEN_CMD_RE.finditer(text):
            take(m, "command", m.group(1))
    for m in _IDENT_RE.finditer(text):
        name = m.group(0)
        if len(name) < 4 or name.replace(".", "").isdigit():
            continue
        take(m, "identifier")
    for m in _PRODUCT_RE.finditer(text):
        name = m.group(0)
        if name in _STOP_CAPS or name.split()[0] in _STOP_CAPS:
            continue
        take(m, "product")
    return found


def _sentence_start_words(text: str) -> set[str]:
    """Capitalised words at sentence starts are usually not product names."""
    words = set()
    for m in re.finditer(r"(?:^|[.!?]\s+)([A-Z][a-zA-Z]+)", text):
        words.add(m.group(1))
    return words


def build_glossary(chunks: list[Chunk], cues: list[Cue], scenes: list[Scene]) -> list[Entity]:
    entities: dict[tuple[str, str], Entity] = {}
    product_counts: Counter[str] = Counter()

    def get(name: str, typ: str) -> Entity:
        key = (name.lower() if typ != "identifier" else name, typ)
        if key not in entities:
            entities[key] = Entity(name=name, type=typ)
        return entities[key]

    # Spoken: cues carry the precise time; chunk membership for navigation.
    chunk_starts = [c.start_seconds for c in chunks]
    all_text = " ".join(c.text for c in chunks)
    sentence_starts = _sentence_start_words(all_text)
    for cue in cues:
        ci = max(0, sum(1 for s in chunk_starts if s <= cue.start) - 1) if chunk_starts else -1
        for name, typ in _candidates(cue.text, source="spoken"):
            if typ == "product":
                if name in sentence_starts and product_counts[name] == 0 and cue.text.startswith(name):
                    continue
                product_counts[name] += 1
            e = get(name, typ)
            e.mentions += 1
            if e.first_spoken is None or cue.start < e.first_spoken:
                e.first_spoken = round(cue.start, 2)
            if ci >= 0 and ci not in e.spoken_in_chunks:
                e.spoken_in_chunks.append(ci)

    # On screen: kept frames in time order.
    for s in sorted((s for s in scenes if s.kept), key=lambda s: s.t_seconds):
        seen_here: set[tuple[str, str]] = set()
        for line in s.ocr_lines:
            for name, typ in _candidates(line, source="ocr"):
                if typ == "product":
                    product_counts[name] += 1
                key = (name, typ)
                if key in seen_here:
                    continue
                seen_here.add(key)
                e = get(name, typ)
                e.mentions += 1
                if e.first_on_screen is None or s.t_seconds < e.first_on_screen:
                    e.first_on_screen = round(s.t_seconds, 2)
                    e.first_frame = s.idx
                if s.idx not in e.frames:
                    e.frames.append(s.idx)

    out = [
        e for e in entities.values()
        if e.type != "product" or product_counts[e.name] >= _MIN_MENTIONS_PRODUCT
    ]

    def first_time(e: Entity) -> float:
        times = [t for t in (e.first_spoken, e.first_on_screen) if t is not None]
        return min(times) if times else 0.0

    out.sort(key=lambda e: (first_time(e), e.type, e.name.lower()))
    return out


def write_entities(entities: list[Entity], out_dir: Path, video_id: str) -> None:
    payload: list[dict[str, Any]] = []
    for e in entities:
        payload.append({
            "name": e.name,
            "type": e.type,
            "mentions": e.mentions,
            "first_spoken": e.first_spoken,
            "first_spoken_url": deep_link(video_id, e.first_spoken) if e.first_spoken is not None else None,
            "first_on_screen": e.first_on_screen,
            "first_on_screen_url": deep_link(video_id, e.first_on_screen) if e.first_on_screen is not None else None,
            "first_frame": e.first_frame,
            "frames": e.frames,
            "spoken_in_chunks": e.spoken_in_chunks,
        })
    (out_dir / "entities.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Entities",
        "",
        "Identifiers, commands, URLs, paths and product names found in the transcript and on screen, "
        "with the first time each was spoken and first shown. Times link to the video.",
        "",
        "| Entity | Type | Mentions | First spoken | First on screen | Frames |",
        "|---|---|---|---|---|---|",
    ]
    for e in entities:
        name = e.name.replace("|", "\\|")
        spoken = f"[{hms(e.first_spoken)}]({deep_link(video_id, e.first_spoken)})" if e.first_spoken is not None else "–"
        screen = (
            f"[{hms(e.first_on_screen)}]({deep_link(video_id, e.first_on_screen)}) (frame {e.first_frame:04d})"
            if e.first_on_screen is not None else "–"
        )
        frames = ", ".join(f"{i:04d}" for i in e.frames[:6]) + (" …" if len(e.frames) > 6 else "") or "–"
        lines.append(f"| `{name}` | {e.type} | {e.mentions} | {spoken} | {screen} | {frames} |")
    (out_dir / "entities.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
