"""Input sources: a YouTube URL, another yt-dlp URL (Vimeo, Wistia, ...), or a local video file.

Only this module knows how a source names itself and links to a moment; the rest of the pipeline
asks the Source for its id, its display URL and `link(seconds)`.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from .models import deep_link, extract_video_id

_VIMEO_RE = re.compile(r"(?:^|[/.])vimeo\.com/(?:.*/)?(?:video/)?(\d{6,})")
_WISTIA_RE = re.compile(r"(?:wistia\.com|wistia\.net|wi\.st)/(?:medias|embed/iframe|embed/medias)/([A-Za-z0-9]{10})")


@dataclass(frozen=True)
class Source:
    kind: str  # "youtube" | "vimeo" | "wistia" | "url" | "local"
    id: str  # names output/<id>/ and transcripts/<id>.*
    url: str | None  # page URL for yt-dlp and display; None for local files
    path: Path | None = None  # local video file

    @property
    def display(self) -> str:
        return self.url or str(self.path)

    @property
    def is_youtube(self) -> bool:
        return self.kind == "youtube"

    @property
    def noun(self) -> str:
        return {"youtube": "YouTube video", "local": "local video"}.get(self.kind, "video")

    def link(self, seconds: float) -> str | None:
        """Deep link to a moment, or None when the source has no linkable timeline."""
        s = int(seconds)
        if self.kind == "youtube":
            return deep_link(self.id, s)
        if self.kind == "vimeo":
            return f"{_without_fragment(self.url)}#t={s}s"
        if self.kind == "wistia":
            sep = "&" if "?" in _without_fragment(self.url) else "?"
            return f"{_without_fragment(self.url)}{sep}wtime={s}s"
        return None


def _without_fragment(url: str | None) -> str:
    parts = urlsplit(url or "")
    return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))


def _slug(text: str, limit: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:limit].rstrip("-") or "video"


def _file_digest(path: Path, n_bytes: int = 1 << 20) -> str:
    """Short, stable id fragment from the file size and its first MiB (hashing a multi-GB file per run is wasteful)."""
    h = hashlib.sha256(str(path.stat().st_size).encode())
    with path.open("rb") as f:
        h.update(f.read(n_bytes))
    return h.hexdigest()[:8]


def resolve_source(value: str) -> Source:
    """Classify a --source value. Existing paths are local files; anything else must be an http(s) URL."""
    path = Path(value).expanduser()
    if path.is_file():
        return Source("local", f"{_slug(path.stem)}-{_file_digest(path)}", None, path.resolve())
    if not re.match(r"^https?://", value):
        raise ValueError(f"--source is neither an existing file nor an http(s) URL: {value}")
    try:
        return Source("youtube", extract_video_id(value), value)
    except ValueError:
        pass
    if m := _VIMEO_RE.search(value):
        return Source("vimeo", f"vimeo-{m.group(1)}", value)
    if m := _WISTIA_RE.search(value):
        return Source("wistia", f"wistia-{m.group(1)}", value)
    host = urlsplit(value).netloc
    return Source("url", f"{_slug(host, 24)}-{hashlib.sha256(value.encode()).hexdigest()[:8]}", value)
