from __future__ import annotations

import json
from pathlib import Path

from youtube_multi.entities import build_glossary, write_entities
from youtube_multi.models import Chunk, Cue, Scene


def _frame(idx: int, t: float, lines: list[str]) -> Scene:
    s = Scene(idx=idx, t_seconds=t, image_path=Path(f"frames/{idx:04d}.jpg"))
    s.kind, s.ocr_lines = "code", lines
    return s


def test_glossary_types_and_first_times() -> None:
    chunks = [Chunk(0, "Now we run uv sync. Then open parse_transcript in Figma. Figma is great."),
              Chunk(30, "Visit https://example.com/docs for the rest.")]
    cues = [
        Cue(0, 0.0, 5, "Now we run uv sync."),
        Cue(1, 5.0, 5, "Then open parse_transcript in Figma."),
        Cue(2, 10.0, 5, "Figma is great."),
        Cue(3, 30.0, 5, "Visit https://example.com/docs for the rest."),
    ]
    frames = [
        _frame(1, 3.0, ["$ uv sync", "def parse_transcript(path):", "src/youtube_multi/fetch.py"]),
        _frame(2, 40.0, ["def parse_transcript(path):", "https://example.com/docs"]),
    ]
    glossary = {(e.name, e.type): e for e in build_glossary(chunks, cues, frames)}

    cmd = glossary[("uv sync", "command")]
    assert cmd.first_spoken == 0.0 and cmd.first_on_screen == 3.0 and cmd.first_frame == 1 and cmd.mentions == 2

    ident = glossary[("parse_transcript", "identifier")]
    assert ident.first_spoken == 5.0 and ident.first_on_screen == 3.0 and ident.frames == [1, 2]
    assert ident.spoken_in_chunks == [0]

    url = glossary[("https://example.com/docs", "url")]
    assert url.first_spoken == 30.0 and url.first_on_screen == 40.0

    path = glossary[("src/youtube_multi/fetch.py", "path")]
    assert path.first_spoken is None and path.first_on_screen == 3.0

    product = glossary[("Figma", "product")]
    assert product.mentions == 2 and product.first_spoken == 5.0

    names = {n for n, _ in glossary}
    assert "Now" not in names and "Then" not in names and "Visit" not in names  # stopwords / sentence starts
    assert "example.com" not in names  # not double-counted as an identifier


def test_glossary_sorted_by_first_appearance_and_products_need_two_mentions() -> None:
    chunks = [Chunk(0, "Kubernetes once. Later we use camelCaseName twice, camelCaseName again.")]
    cues = [Cue(0, 0, 3, "Kubernetes once."), Cue(1, 3, 3, "Later we use camelCaseName twice, camelCaseName again.")]
    glossary = build_glossary(chunks, cues, [])
    assert [e.name for e in glossary] == ["camelCaseName"]  # single-mention product dropped
    assert glossary[0].mentions == 2


def test_write_entities(tmp_path: Path) -> None:
    chunks = [Chunk(0, "Run git status now.")]
    cues = [Cue(0, 2.5, 3, "Run git status now.")]
    frames = [_frame(7, 9.0, ["$ git status"])]
    glossary = build_glossary(chunks, cues, frames)
    write_entities(glossary, tmp_path, "VIDEOID12345")
    data = json.loads((tmp_path / "entities.json").read_text())
    assert data[0]["name"] == "git status" and data[0]["type"] == "command"
    assert data[0]["first_spoken_url"] == "https://youtu.be/VIDEOID12345?t=2"
    assert data[0]["first_on_screen_url"] == "https://youtu.be/VIDEOID12345?t=9" and data[0]["first_frame"] == 7
    md = (tmp_path / "entities.md").read_text()
    assert "| `git status` | command | 2 | [00:00:02](https://youtu.be/VIDEOID12345?t=2) | [00:00:09](https://youtu.be/VIDEOID12345?t=9) (frame 0007) | 0007 |" in md
