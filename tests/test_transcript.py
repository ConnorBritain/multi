from __future__ import annotations

from pathlib import Path

from youtube_multi.fetch import aggregate_cues, parse_transcript, write_transcript_file
from youtube_multi.models import Chunk, extract_video_id, hms, short_ms


def test_parse_transcript_basic(tmp_path: Path) -> None:
    p = tmp_path / "t.txt"
    p.write_text("00:00:00\nhello there\n\n00:00:30\nsecond block\nspans lines\n", encoding="utf-8")
    chunks = parse_transcript(p)
    assert [(c.start_seconds, c.text) for c in chunks] == [
        (0, "hello there"),
        (30, "second block spans lines"),
    ]


def test_parse_transcript_sorts_and_skips_bad_blocks(tmp_path: Path) -> None:
    p = tmp_path / "t.txt"
    p.write_text(
        "01:02:03\nlate\n\nnot a timestamp\nignored\n\n00:00:05\nearly\n\n00:00:09\n\n\n\n",
        encoding="utf-8",
    )
    chunks = parse_transcript(p)
    assert [(c.start_seconds, c.text) for c in chunks] == [(5, "early"), (3723, "late")]


def test_parse_transcript_single_digit_hour(tmp_path: Path) -> None:
    p = tmp_path / "t.txt"
    p.write_text("1:00:00\nx y\n", encoding="utf-8")
    assert parse_transcript(p)[0].start_seconds == 3600


def test_aggregate_cues_buckets() -> None:
    cues = [Chunk(0, "a"), Chunk(10, "b"), Chunk(29, "c"), Chunk(30, "d"), Chunk(65, "e")]
    out = aggregate_cues(cues, 30)
    assert [(c.start_seconds, c.text) for c in out] == [(0, "a b c"), (30, "d"), (65, "e")]


def test_aggregate_cues_passthrough_and_empty() -> None:
    cues = [Chunk(0, "a"), Chunk(1, "b")]
    assert aggregate_cues(cues, 0) is cues
    assert aggregate_cues([], 30) == []
    assert aggregate_cues([Chunk(7, "only")], 30) == [Chunk(7, "only")]


def test_write_then_parse_roundtrip(tmp_path: Path) -> None:
    chunks = [Chunk(0, "first"), Chunk(45, "second one"), Chunk(3600, "hour")]
    p = tmp_path / "sub" / "t.txt"
    write_transcript_file(chunks, p)
    back = parse_transcript(p)
    assert [(c.start_seconds, c.text) for c in back] == [(c.start_seconds, c.text) for c in chunks]


def test_helpers() -> None:
    assert hms(3723.9) == "01:02:03"
    assert short_ms(125) == "02m05s"
    assert extract_video_id("https://youtu.be/EcbgbKtOELY") == "EcbgbKtOELY"
    assert extract_video_id("https://www.youtube.com/watch?v=EcbgbKtOELY&t=1s") == "EcbgbKtOELY"
    assert extract_video_id("https://www.youtube.com/shorts/EcbgbKtOELY") == "EcbgbKtOELY"
