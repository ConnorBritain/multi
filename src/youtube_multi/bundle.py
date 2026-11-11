"""agent/: a small, fixed-shape bundle for handing a video to an agent.

    agent/README.md       what the video is, the file list, how to read it
    agent/transcript.md   ## HH:MM:SS blocks, with "→ slide NNN" lines where each visual appeared
    agent/slides.md       every kept non-talking-head frame: image, time on screen, OCR, speech while visible
    agent/slides.json     the same, as data
    agent/slides/         NNN_HH-MM-SS.jpg copies of those frames

The full pack (paired.*, index.*, frames/, diffs, entities) stays next to it for tools; the bundle
repeats only what an agent reading for content needs.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable

from .models import Chunk, Cue, Scene, hms

# Frames of these kinds carry no on-screen content of their own.
_NON_VISUAL_KINDS = frozenset({"talking-head"})


def _said(cues: list[Cue], start: float, end: float) -> str:
    return " ".join(c.text.strip() for c in cues if c.start < end and c.end > start)


def write_bundle(
    out_dir: Path,
    *,
    chunks: list[Chunk],
    cues: list[Cue],
    kept: list[Scene],
    title: str,
    source_display: str,
    link: Callable[[float], str | None],
    transcript_method: str | None,
) -> Path:
    """Rebuild agent/ from scratch (it is generated output; hand edits do not survive a re-run)."""
    agent = out_dir / "agent"
    if agent.exists():
        shutil.rmtree(agent)
    (agent / "slides").mkdir(parents=True)
    slides = _copy_slides(agent, kept, cues, link)
    _write_slides(agent, slides)
    _write_transcript(agent, chunks, slides, transcript_method)
    _write_readme(agent, title, source_display, len(slides), link(0), transcript_method)
    return agent


def _copy_slides(agent: Path, kept: list[Scene], cues: list[Cue], link) -> list[dict[str, Any]]:
    visuals = sorted((s for s in kept if s.kind not in _NON_VISUAL_KINDS), key=lambda s: s.t_seconds)
    slides: list[dict[str, Any]] = []
    for n, s in enumerate(visuals, 1):
        start, end = s.visible_from, max(s.visible_to, s.visible_from + 1)
        rel = f"slides/{n:03d}_{hms(start).replace(':', '-')}.jpg"
        shutil.copy2(s.image_path, agent / rel)
        slides.append({
            "n": n, "image": rel, "frame_idx": s.idx, "kind": s.kind,
            "start": round(start, 2), "end": round(end, 2), "start_hms": hms(start), "end_hms": hms(end),
            "url": link(start), "ocr": s.ocr, "spoken_while_visible": _said(cues, start, end),
        })
    return slides


def _write_slides(agent: Path, slides: list[dict[str, Any]]) -> None:
    (agent / "slides.json").write_text(json.dumps(slides, indent=1, ensure_ascii=False), encoding="utf-8")
    md = ["# Slides", "",
          f"{len(slides)} on-screen visuals in time order (slides, code, diagrams; talking-head frames left out). "
          "Each: image, time on screen, slide text (OCR), and what was said while it was up.", ""]
    for x in slides:
        md += [f"## Slide {x['n']:03d} · {x['start_hms']} – {x['end_hms']}", "", f"![slide {x['n']:03d}]({x['image']})", ""]
        if x["ocr"].strip():
            md += ["**On slide (OCR):** " + " ".join(x["ocr"].split()), ""]
        if x["spoken_while_visible"]:
            md += ["**Said while visible:** " + x["spoken_while_visible"], ""]
    (agent / "slides.md").write_text("\n".join(md), encoding="utf-8")


def _write_transcript(agent: Path, chunks: list[Chunk], slides: list[dict[str, Any]], method: str | None) -> None:
    tr = ["# Transcript", "",
          f"Timestamped transcript ({method or 'unknown source'}). "
          "`→ slide NNN` lines mark when a visual appeared; see slides.md.", ""]
    for i, c in enumerate(chunks):
        nxt = chunks[i + 1].start_seconds if i + 1 < len(chunks) else float("inf")
        tr += [f"## {hms(c.start_seconds)}", "", c.text, ""]
        marks = [x for x in slides if c.start_seconds <= x["start"] < nxt or (i == 0 and x["start"] < c.start_seconds)]
        if marks:
            tr += [f"→ slide {x['n']:03d} at {x['start_hms']} ([image]({x['image']}))" for x in marks] + [""]
    (agent / "transcript.md").write_text("\n".join(tr), encoding="utf-8")


def _write_readme(agent: Path, title: str, source_display: str, n_slides: int, sample_link: str | None,
                  method: str | None) -> None:
    time_note = (f"Times are deep-linkable (e.g. {sample_link})." if sample_link
                 else "Times are seconds into the source recording; this source has no web link.")
    readme = f"""# {title} — agent bundle

Source: {source_display}

## Files

| File | What it is |
|---|---|
| `README.md` | This file. Read first. |
| `transcript.md` | Full transcript in `## HH:MM:SS` blocks; `→ slide NNN` lines show when each visual appeared. |
| `slides.md` | All {n_slides} on-screen visuals in order: image, time on screen, slide text, speech while visible. |
| `slides.json` | Same as `slides.md`, as data. |
| `slides/` | Images named `NNN_HH-MM-SS.jpg` (slide number, start time). |

## How to read

1. Skim `slides.md` for the structure: section titles, frameworks, examples.
2. Read `transcript.md` for the full reasoning; use the slide markers to move between the two.
3. Open an image only when the picture matters; the slide text is often enough.

## Caveats

- Slide text is OCR and can be garbled on small or stylised text; check the image when it matters.
- The transcript may be machine-generated ({method or 'unknown'}); names can be misspelled.
- {time_note}
- The full pipeline output (every frame, diffs, entities) is in the parent folder; you do not need it.
"""
    (agent / "README.md").write_text(readme, encoding="utf-8")
