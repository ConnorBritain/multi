"""Fetching: video download and caption/transcript handling."""
from __future__ import annotations

import json
import re
from pathlib import Path

from yt_dlp import YoutubeDL
from youtube_transcript_api import YouTubeTranscriptApi

from .models import Chunk, Cue, hms


def download_video(url: str, out_dir: Path) -> Path:
    target = out_dir / "video.mp4"
    if target.exists():
        print(f"  [skip] {target.name} already exists")
        return target
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": "b[ext=mp4][height<=720]/b[height<=720]/b[ext=mp4]/b",
        "outtmpl": str(out_dir / "video.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    with YoutubeDL(opts) as ydl:
        ydl.download([url])
    if not target.exists():
        produced = list(out_dir.glob("video.*"))
        if produced:
            produced[0].rename(target)
    return target


# --- cues -------------------------------------------------------------------------


def fetch_cues_youtube(video_id: str, languages: list[str]) -> list[Cue]:
    """Raw caption cues with float start/duration, whitespace-normalised."""
    api = YouTubeTranscriptApi()
    fetched = api.fetch(video_id, languages=languages)
    cues: list[Cue] = []
    for snippet in fetched.snippets:
        text = re.sub(r"\s+", " ", snippet.text).strip()
        if text:
            cues.append(Cue(id=len(cues), start=float(snippet.start), duration=float(snippet.duration), text=text))
    return cues


def cues_to_chunks(cues: list[Cue]) -> list[Chunk]:
    return [Chunk(start_seconds=round(c.start), text=c.text) for c in cues]


def fetch_transcript_youtube(video_id: str, languages: list[str]) -> list[Chunk]:
    """Backward-compatible wrapper: one Chunk per cue with a rounded start."""
    return cues_to_chunks(fetch_cues_youtube(video_id, languages))


def cues_path_for(transcript_path: Path) -> Path:
    return transcript_path.with_name(transcript_path.stem + ".cues.json")


def write_cues_file(cues: list[Cue], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [{"id": c.id, "start": round(c.start, 3), "duration": round(c.duration, 3), "text": c.text} for c in cues]
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")


def read_cues_file(path: Path) -> list[Cue]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Cue(id=int(c.get("id", i)), start=float(c["start"]), duration=float(c["duration"]), text=str(c["text"])) for i, c in enumerate(raw)]


def synthesize_cues(chunks: list[Chunk], duration: float) -> list[Cue]:
    """One cue per chunk when no raw cues exist (hand-written transcripts)."""
    cues: list[Cue] = []
    for i, c in enumerate(chunks):
        end = float(chunks[i + 1].start_seconds) if i + 1 < len(chunks) else max(duration, float(c.start_seconds))
        cues.append(Cue(id=i, start=float(c.start_seconds), duration=max(0.0, end - float(c.start_seconds)), text=c.text))
    return cues


# --- transcript text file --------------------------------------------------------------


def aggregate_cues(cues: list[Chunk], target_seconds: float) -> list[Chunk]:
    if target_seconds <= 0 or not cues:
        return cues
    out: list[Chunk] = []
    bucket_start = cues[0].start_seconds
    bucket_text: list[str] = []
    for c in cues:
        if c.start_seconds - bucket_start >= target_seconds and bucket_text:
            out.append(Chunk(start_seconds=bucket_start, text=" ".join(bucket_text)))
            bucket_start = c.start_seconds
            bucket_text = [c.text]
        else:
            bucket_text.append(c.text)
    if bucket_text:
        out.append(Chunk(start_seconds=bucket_start, text=" ".join(bucket_text)))
    return out


def write_transcript_file(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for c in chunks:
        lines.append(hms(c.start_seconds))
        lines.append(c.text)
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


_TS_RE = re.compile(r"^(\d{1,2}):(\d{2}):(\d{2})$")


def parse_transcript(path: Path) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8").strip()
    blocks = re.split(r"\n\s*\n", raw)
    chunks: list[Chunk] = []
    for block in blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue
        m = _TS_RE.match(lines[0].strip())
        if not m:
            continue
        h, mi, s = (int(g) for g in m.groups())
        start = h * 3600 + mi * 60 + s
        text = " ".join(line.strip() for line in lines[1:]).strip()
        if not text:
            continue
        chunks.append(Chunk(start_seconds=start, text=text))
    chunks.sort(key=lambda c: c.start_seconds)
    return chunks
