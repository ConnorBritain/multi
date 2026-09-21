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

from .models import Chunk, Cue, Scene, deep_link, hms
from .navigate import Chapter, chapter_json, estimate_text_tokens


def _rel(path: Path | None, out_dir: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.relative_to(out_dir).as_posix()
    except ValueError:
        return path.as_posix()


def _scene_md_lines(scene: Scene, out_dir: Path, video_id: str) -> list[str]:
    rel = _rel(scene.image_path, out_dir)
    lines = [f"![scene {scene.idx:04d} @ {hms(scene.t_seconds)}]({rel})"]
    meta = f"> kind: {scene.kind} ({scene.kind_confidence:.2f})"
    if scene.visible_to > scene.visible_from:
        meta += f" · visible {hms(scene.visible_from)}–{hms(scene.visible_to)}"
    meta += f" · [▶ {hms(scene.t_seconds)}]({deep_link(video_id, scene.t_seconds)})"
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
        # Heading line kept exactly as in the original format; the deep link sits below it.
        lines.append(f"## {hms(chunk.start_seconds)}")
        lines.append("")
        lines.append(f"[▶ {hms(chunk.start_seconds)}]({deep_link(video_id, chunk.start_seconds)})")
        lines.append("")
        lines.extend(_chunk_body_md(chunk, out_dir, video_id))
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _chunk_body_md(chunk: Chunk, out_dir: Path, video_id: str) -> list[str]:
    """Transcript sentences with each frame inserted after the sentence it
    appeared during (frames before the first sentence come first)."""
    if not chunk.sentences:
        lines = [chunk.text, ""]
        for scene in chunk.scenes:
            lines.extend(_scene_md_lines(scene, out_dir, video_id))
        return lines

    by_sentence: dict[int, list[Scene]] = {}
    for scene in chunk.scenes:
        by_sentence.setdefault(scene.sentence_index, []).append(scene)

    lines: list[str] = []
    for scene in by_sentence.get(-1, []):
        lines.extend(_scene_md_lines(scene, out_dir, video_id))
    paragraph: list[str] = []
    for sentence in chunk.sentences:
        paragraph.append(sentence.text)
        scenes_here = by_sentence.get(sentence.index)
        if scenes_here:
            lines.append(" ".join(paragraph))
            lines.append("")
            paragraph = []
            for scene in scenes_here:
                lines.extend(_scene_md_lines(scene, out_dir, video_id))
    if paragraph:
        lines.append(" ".join(paragraph))
        lines.append("")
    return lines


def _frame_json(s: Scene, out_dir: Path, video_id: str) -> dict[str, Any]:
    d: dict[str, Any] = {
        "idx": s.idx,
        "t_seconds": round(s.t_seconds, 2),
        "url": deep_link(video_id, s.t_seconds),
        "image": _rel(s.image_path, out_dir),
        "ocr": s.ocr,
        "ocr_lines": s.ocr_lines,
        "ocr_confidence": s.ocr_confidence,
        "kind": s.kind,
        "kind_confidence": s.kind_confidence,
        "phash": f"{s.phash:016x}",
        "visible_from": round(s.visible_from, 2),
        "visible_to": round(s.visible_to, 2),
        "sentence_index": s.sentence_index,
        "cue_ids_visible": list(s.cue_ids_visible),
        "width": s.width,
        "height": s.height,
        "tokens_est": s.tokens_est,
        "score": s.score,
        "tier": s.tier,
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
    cues: list[Cue] | None = None,
    metadata: dict[str, Any] | None = None,
    chapters: list[Chapter] | None = None,
    budget_info: dict[str, Any] | None = None,
) -> None:
    out_dir = out_path.parent
    all_scenes = all_scenes if all_scenes is not None else [s for c in chunks for s in c.scenes]
    cues = cues if cues is not None else [cue for c in chunks for cue in c.cues]
    metadata = metadata or {}
    payload: dict[str, Any] = {
        "source_url": source_url,
        "video_id": video_id,
        "title": metadata.get("title"),
        "description": metadata.get("description"),
        "duration": metadata.get("duration"),
        "channel": metadata.get("channel") or metadata.get("uploader"),
        "manifest": "manifest.json",
        "index": "index.json",
        "skill": "SKILL.md",
        "budget": budget_info or {"budget": None},
        "chapters": [
            {k: v for k, v in chapter_json(ch, chunks, video_id, deep_link).items() if k != "visual_states"}
            for ch in (chapters or [])
        ],
        "frame_stats": frame_stats(all_scenes),
        "chunks": [
            {
                "start_seconds": c.start_seconds,
                "start_hms": hms(c.start_seconds),
                "url": deep_link(video_id, c.start_seconds),
                "text": c.text,
                "sentences": [
                    {"index": s.index, "start": s.start, "url": deep_link(video_id, s.start), "text": s.text}
                    for s in c.sentences
                ],
                "cue_ids": [cue.id for cue in c.cues],
                "frames": [_frame_json(s, out_dir, video_id) for s in c.scenes],
            }
            for c in chunks
        ],
        "cues": [
            {
                "id": cue.id,
                "start": round(cue.start, 3),
                "duration": round(cue.duration, 3),
                "url": deep_link(video_id, cue.start),
                "text": cue.text,
            }
            for cue in cues
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


# --- index.json / index.md --------------------------------------------------------


def _fmt_range(a: float, b: float) -> str:
    return f"{hms(a)}–{hms(b)}"


def write_index(
    chapters: list[Chapter],
    chunks: list[Chunk],
    all_scenes: list[Scene],
    *,
    out_dir: Path,
    video_id: str,
    source_url: str,
    metadata: dict[str, Any] | None,
    duration: float,
    budget_info: dict[str, Any],
) -> None:
    metadata = metadata or {}
    title = metadata.get("title") or video_id
    kept = [s for s in all_scenes if s.kept]
    stats = frame_stats(all_scenes)
    text_tokens = sum(estimate_text_tokens(c.text) for c in chunks)
    image_tokens = sum(s.tokens_est for s in kept)
    tier1_tokens = text_tokens + sum(s.tokens_est for s in kept if s.tier == 1)

    ch_json = []
    for ch in chapters:
        d = chapter_json(ch, chunks, video_id, deep_link)
        for vs, s in zip(d["visual_states"], ch.frames):
            vs["image"] = _rel(s.image_path, out_dir)
        ch_json.append(d)

    index = {
        "video_id": video_id,
        "source_url": source_url,
        "title": title,
        "duration": round(duration, 2),
        "channel": metadata.get("channel") or metadata.get("uploader"),
        "chapters_source": chapters[0].source if chapters else None,
        "frame_stats": stats,
        "tokens": {"text": text_tokens, "images": image_tokens, "total": text_tokens + image_tokens, "tier1": tier1_tokens},
        "budget": budget_info,
        "files": {"paired_md": "paired.md", "paired_json": "paired.json", "skill": "SKILL.md", "manifest": "manifest.json"},
        "chapters": ch_json,
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")

    lines: list[str] = [
        f"# {title} — index",
        "",
        f"Source: {source_url} · duration {hms(duration)} · {len(chapters)} chapters "
        f"({chapters[0].source if chapters else 'none'}) · {stats['kept']} frames kept of {stats['extracted']} extracted",
        "",
        f"Estimated tokens: text {text_tokens:,} · images {image_tokens:,} · total {text_tokens + image_tokens:,} · "
        f"text + tier-1 images {tier1_tokens:,}"
        + (f" (image budget {budget_info['budget']:,})" if budget_info.get("budget") else ""),
        "",
        "Read `SKILL.md` first, then pick a chapter below and open only the `## HH:MM:SS` sections of `paired.md` "
        "listed under it. Open images by tier: 1 first, then 2, then 3.",
        "",
        "| # | Chapter | Time | Chunks | Frames | Tokens text / images | Tier-1 tokens |",
        "|---|---|---|---|---|---|---|",
    ]
    for ch, d in zip(chapters, ch_json):
        lines.append(
            f"| {ch.index + 1} | [{ch.title}](#{_anchor(ch)}) | [{_fmt_range(ch.start, ch.end)}]({d['url']}) | "
            f"{len(ch.chunk_indices)} | {len(ch.frames)} | {ch.tokens_text:,} / {ch.tokens_images:,} | {d['tokens_tier1']:,} |"
        )
    lines.append("")
    for ch, d in zip(chapters, ch_json):
        lines.append(f"## {ch.index + 1}. {ch.title}")
        lines.append("")
        chunk_refs = ", ".join(f"`## {hms(chunks[i].start_seconds)}`" for i in ch.chunk_indices) or "none"
        lines.append(f"[▶ {_fmt_range(ch.start, ch.end)}]({d['url']}) · paired.md sections: {chunk_refs}")
        lines.append("")
        if not ch.frames:
            lines.append("_No frames kept in this chapter._")
            lines.append("")
            continue
        lines.append("| Frame | Time | Kind | Heading | Visible | Tokens | Tier | Image |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for vs, s in zip(d["visual_states"], ch.frames):
            heading = vs["heading"].replace("|", "\\|")
            lines.append(
                f"| {s.idx:04d} | [{hms(s.t_seconds)}]({vs['url']}) | {s.kind} ({s.kind_confidence:.2f}) | {heading} | "
                f"{_fmt_range(s.visible_from, s.visible_to)} | {s.tokens_est:,} | {s.tier} | [{Path(vs['image']).name}]({vs['image']}) |"
            )
        lines.append("")
    (out_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")


def _anchor(ch: Chapter) -> str:
    import re

    text = f"{ch.index + 1}. {ch.title}".lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"\s+", "-", text.strip())


# --- SKILL.md ------------------------------------------------------------------------


def write_skill(
    *,
    out_dir: Path,
    video_id: str,
    source_url: str,
    metadata: dict[str, Any] | None,
    chapters: list[Chapter],
    all_scenes: list[Scene],
    chunks: list[Chunk],
    budget_info: dict[str, Any],
    has_diffs: bool,
) -> None:
    metadata = metadata or {}
    title = metadata.get("title") or video_id
    stats = frame_stats(all_scenes)
    kept = [s for s in all_scenes if s.kept]
    image_tokens = sum(s.tokens_est for s in kept)
    text_tokens = sum(estimate_text_tokens(c.text) for c in chunks)
    tier1 = sum(s.tokens_est for s in kept if s.tier == 1)
    kinds = ", ".join(f"{k} ×{v}" for k, v in stats["kept_by_kind"].items()) or "none"
    budget_line = (
        f"A token budget of {budget_info['budget']:,} was requested: tier-1 frames total ≈{tier1:,} image tokens."
        if budget_info.get("budget")
        else "No token budget was requested, so every kept frame is tier 1."
    )
    body = f"""---
name: video-context-pack-{video_id}
description: Context pack for the YouTube video "{title}" ({video_id}) — transcript, deduplicated and classified frames, OCR, frame diffs, chapters and token estimates. Read this file first, then index.md.
---

# How to read this context pack

This directory is a curated, navigable representation of {source_url} ("{title}").
It was generated by `multi`; nothing here was written by a person. Read progressively —
summary → chapter → chunk → frame — rather than opening every image.

## Layout

```
SKILL.md        this file
index.md/json   chapters → visual states (one row per kept frame) with time ranges, headings and token estimates
paired.md/json  the transcript, chunk by chunk (## HH:MM:SS), with frames inserted after the sentence they appeared during
manifest.json   how this pack was produced (args, versions, video hash)
metadata.json   raw yt-dlp metadata (title, description, chapters) when it could be fetched
frames/         kept frames only: NNNN_tMMmSSs.jpg (NNNN = extraction index; gaps are dropped frames)
frames/diffs/   for consecutive frames of the same kind: <a>_<b>.txt (OCR line diff) and <a>_<b>.jpg (crop of the changed region)
```

## Recommended reading order

1. **index.md** — pick the chapter(s) relevant to the question. Each chapter row gives its time range, the
   `## HH:MM:SS` sections of paired.md it spans, and its token cost.
2. **paired.md, only those sections** — read the transcript text. Every frame line looks like
   `![scene NNNN @ HH:MM:SS](frames/...)` followed by `> kind`, `> OCR` and `> diff` notes. The OCR
   text and diff summary are often enough; open the image only when the visual matters.
3. **Frames by tier** — open tier-1 images first (`tier` in index.json / paired.json), escalate to tier 2
   and 3 only if the answer still depends on the visual. {budget_line}
4. **frames/diffs/** — when two consecutive frames are the same kind (e.g. a code file being edited), read
   the `.txt` diff instead of both images; the `.jpg` is just the changed region.
5. **paired.json / index.json** — for programmatic access (exact seconds, cue ids, hashes, scores).

## Field reference

- `kind` (`slide | code | terminal | diagram | talking-head | whiteboard | other`) with `kind_confidence` 0–1:
  a heuristic classification (OCR density, monospace layout, faces, colour, boxes). Trust high-confidence
  values; treat low ones as hints. Frames of the kinds listed in manifest.json `args.drop` were removed
  (default: talking-head) and are listed in paired.json `dropped_frames`.
- `visible_from` / `visible_to` (seconds): the frame was on screen for this whole range; consecutive
  near-duplicates were collapsed into it (`frame_stats.duplicates`).
- `sentence_index`: which sentence of the chunk was being spoken when the frame appeared (-1 = before the
  chunk's first sentence). `cue_ids_visible`: raw caption cues (paired.json `cues`) spoken while it was visible.
- `url`: a `https://youtu.be/{video_id}?t=<seconds>` deep link — cite these when pointing at a moment.
- `ocr` / `ocr_lines` / `ocr_confidence`: Tesseract output (flat, per line, mean word confidence 0–100).
  Expect noise on small or stylised text.
- `tokens_est`: approximate cost of the image (≈ width×height/750, capped) plus its OCR text.
  `score` blends visual novelty, OCR density and chunk coverage; `tier` is derived from score and the budget.
- `diff`: `vs` is the previous frame's index; `added`/`removed` count OCR lines; `changed_region` is
  `[x, y, w, h]` in the frame's pixels; `changed_frac` is the fraction of pixels that changed.

## This pack at a glance

- Chapters: {len(chapters)} ({chapters[0].source if chapters else 'none'})
- Frames: {stats['kept']} kept of {stats['extracted']} extracted ({stats['duplicates']} duplicates, {sum(stats['dropped_by_kind'].values())} dropped by kind); kept kinds: {kinds}
- Estimated tokens: transcript ≈{text_tokens:,}, all kept images ≈{image_tokens:,}, tier-1 images ≈{tier1:,}
- Diffs: {'present in frames/diffs/' if has_diffs else 'none'}

## Caveats

- Classification and OCR are heuristic, offline and cheap; verify anything load-bearing against the image.
- Sentence timing is interpolated from caption cues; it is accurate to a few seconds, not to the word.
- Frame indices are not contiguous; that is expected.
"""
    (out_dir / "SKILL.md").write_text(body, encoding="utf-8")


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
