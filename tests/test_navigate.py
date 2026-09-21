from __future__ import annotations

from pathlib import Path

from youtube_multi.models import Chunk, Scene
from youtube_multi.navigate import (
    IMAGE_TOKEN_CAP,
    assign_tiers,
    build_chapters,
    estimate_image_tokens,
    estimate_text_tokens,
    frame_heading,
    populate_chapters,
    score_frames,
)


def _scene(idx: int, t: float, phash: int = 0, ocr: str = "", tokens: int = 1000) -> Scene:
    s = Scene(idx=idx, t_seconds=t, image_path=Path(f"frames/{idx:04d}.jpg"))
    s.phash = phash
    s.ocr = ocr
    s.ocr_lines = [ocr] if ocr else []
    s.tokens_est = tokens
    s.width, s.height = 1280, 720
    return s


def test_token_estimates() -> None:
    assert estimate_text_tokens("") == 0
    assert estimate_text_tokens("abcd" * 10) == 10
    assert estimate_text_tokens("abcde") == 2
    assert estimate_image_tokens(1280, 720) == 1229
    assert estimate_image_tokens(320, 240) == 103
    assert estimate_image_tokens(4000, 3000) == IMAGE_TOKEN_CAP
    assert estimate_image_tokens(0, 0) == 0


def test_frame_heading() -> None:
    s = _scene(1, 0)
    s.ocr_lines = ["a", "  ", "Why hierarchy matters", "second"]
    assert frame_heading(s) == "Why hierarchy matters"
    s.ocr_lines = ["x" * 100]
    assert frame_heading(s).endswith("…") and len(frame_heading(s)) == 80
    s.ocr_lines = []
    s.kind = "diagram"
    assert frame_heading(s) == "diagram"


def test_score_frames_novelty_density_coverage() -> None:
    A, B = 0, 0xFFFFFFFFFFFFFFFF
    s1 = _scene(1, 0, A, "")  # first frame: novelty 1
    s2 = _scene(2, 5, A, "x" * 1500)  # identical hash: novelty 0, density 1
    s3 = _scene(3, 10, B, "")  # max novelty
    chunks = [Chunk(0, "c1", scenes=[s1, s2]), Chunk(8, "c2", scenes=[s3])]
    score_frames([s1, s2, s3], chunks)
    assert s1.score == 0.7  # 0.5 novelty + 0.2 coverage (best in chunk 1)
    assert s2.score == 0.3  # 0.3 density, not best
    assert s3.score == 0.7  # novelty + coverage


def test_assign_tiers_with_and_without_budget() -> None:
    frames = [_scene(i, i, tokens=1000) for i in range(1, 6)]
    for f, sc in zip(frames, [0.9, 0.1, 0.8, 0.5, 0.3]):
        f.score = sc
    info = assign_tiers(frames, None)
    assert [f.tier for f in frames] == [1, 1, 1, 1, 1]
    assert info["budget"] is None and info["frames_by_tier"] == {"1": 5, "2": 0, "3": 0}

    info = assign_tiers(frames, 2000)
    # by score: 1 (0.9), 3 (0.8) fit in 2000; 4 (0.5), 5 (0.3) in 4000; 2 (0.1) is tier 3
    assert {f.idx: f.tier for f in frames} == {1: 1, 3: 1, 4: 2, 5: 2, 2: 3}
    assert info["tokens_by_tier"] == {"1": 2000, "2": 2000, "3": 1000}
    assert info["frames_by_tier"] == {"1": 2, "2": 2, "3": 1}


def test_build_chapters_from_metadata() -> None:
    meta = {"chapters": [{"start_time": 0, "end_time": 60, "title": "Intro"}, {"start_time": 60, "end_time": None, "title": "Body"}]}
    ch = build_chapters(meta, [Chunk(0, "a")], duration=500.0)
    assert [(c.index, c.title, c.start, c.end, c.source) for c in ch] == [(0, "Intro", 0.0, 60.0, "youtube"), (1, "Body", 60.0, 500.0, "youtube")]


def test_build_chapters_single_for_short_video() -> None:
    ch = build_chapters({"title": "T"}, [Chunk(0, "a"), Chunk(30, "b")], duration=300.0)
    assert len(ch) == 1 and ch[0].title == "T" and ch[0].source == "synthesized" and ch[0].end == 300.0
    assert build_chapters(None, [], duration=10.0)[0].title == "Full video"


def test_build_chapters_synthesized_every_three_minutes() -> None:
    chunks = [Chunk(t, f"words for chunk at {t} seconds go here now") for t in range(0, 600, 30)]
    chunks[0].scenes.append(_scene(1, 0, ocr="Opening Slide"))
    ch = build_chapters(None, chunks, duration=600.0)
    assert [c.start for c in ch] == [0.0, 180.0, 360.0, 540.0]
    assert [c.end for c in ch] == [180.0, 360.0, 540.0, 600.0]
    assert ch[0].title == "Opening Slide"  # OCR heading of the first kept frame
    assert ch[1].title.startswith("words for chunk at 180")
    assert all(c.source == "synthesized" for c in ch)


def test_populate_chapters_assigns_chunks_frames_tokens() -> None:
    chunks = [Chunk(0, "a" * 40), Chunk(30, "b" * 80), Chunk(70, "c" * 4)]
    f1, f2, f3 = _scene(1, 0, tokens=100), _scene(2, 35, tokens=200), _scene(3, 40, tokens=300)
    f3.dropped_reason = "duplicate"
    chunks[0].scenes.append(f1)
    chunks[1].scenes.extend([f2, f3])
    ch = build_chapters({"chapters": [{"start_time": 0, "end_time": 60, "title": "A"}, {"start_time": 60, "end_time": 90, "title": "B"}]}, chunks, 90.0)
    populate_chapters(ch, chunks)
    assert ch[0].chunk_indices == [0, 1] and ch[1].chunk_indices == [2]
    assert [s.idx for s in ch[0].frames] == [1, 2]  # dropped frame excluded
    assert (ch[0].tokens_text, ch[0].tokens_images, ch[0].tokens_total) == (30, 300, 330)
    assert (ch[1].tokens_text, ch[1].tokens_images) == (1, 0)


def test_populate_chapters_frames_follow_their_own_time() -> None:
    chunks = [Chunk(0, "a"), Chunk(4, "b")]
    late = _scene(2, 6.5, tokens=50)  # in the chunk starting at 4, but inside chapter 2 (6-12)
    chunks[1].scenes.extend([_scene(1, 5.0, tokens=10), late])
    ch = build_chapters({"chapters": [{"start_time": 0, "end_time": 6, "title": "A"}, {"start_time": 6, "end_time": 12, "title": "B"}]}, chunks, 12.0)
    populate_chapters(ch, chunks)
    assert ch[0].chunk_indices == [0, 1] and ch[1].chunk_indices == []
    assert [s.idx for s in ch[0].frames] == [1] and [s.idx for s in ch[1].frames] == [2]
    assert ch[1].tokens_images == 50
