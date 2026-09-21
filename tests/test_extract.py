from __future__ import annotations

import re
from pathlib import Path

from youtube_multi.extract import detect_scenes, extract_interval_frames

from .conftest import SEGMENT_SECONDS, SEGMENTS

NAME_RE = re.compile(r"^\d{4}_t\d{2}m\d{2}s\.jpg$")


def test_interval_frames(fixture_video: Path, tmp_path: Path) -> None:
    scenes = extract_interval_frames(fixture_video, tmp_path / "frames", 1.0)
    duration = SEGMENT_SECONDS * len(SEGMENTS)
    assert duration - 1 <= len(scenes) <= duration + 1
    assert [s.idx for s in scenes] == list(range(1, len(scenes) + 1))
    for s in scenes:
        assert NAME_RE.match(s.image_path.name), s.image_path.name
        assert s.image_path.is_file()


def test_scene_detection(fixture_video: Path, tmp_path: Path) -> None:
    scenes = detect_scenes(fixture_video, tmp_path / "frames", threshold=27.0, min_scene_len=1.5)
    assert len(scenes) == len(SEGMENTS)
    times = [s.t_seconds for s in scenes]
    assert times[0] == 0.0
    for i, t in enumerate(times):
        assert abs(t - i * SEGMENT_SECONDS) < 0.5
    for s in scenes:
        assert NAME_RE.match(s.image_path.name)
        assert s.image_path.is_file()
    assert not list((tmp_path / "frames").glob("*.jpg")) == [], "frames written"
    leftovers = [p.name for p in (tmp_path / "frames").iterdir() if not NAME_RE.match(p.name)]
    assert leftovers == []
