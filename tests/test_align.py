from __future__ import annotations

from pathlib import Path

from youtube_multi.align import pair_scenes
from youtube_multi.models import Chunk, Scene


def _scene(idx: int, t: float) -> Scene:
    return Scene(idx=idx, t_seconds=t, image_path=Path(f"frames/{idx:04d}.jpg"))


def test_pair_scenes_places_by_time() -> None:
    chunks = [Chunk(0, "a"), Chunk(30, "b"), Chunk(60, "c")]
    scenes = [_scene(1, 0.0), _scene(2, 29.99), _scene(3, 30.0), _scene(4, 45), _scene(5, 999)]
    pair_scenes(scenes, chunks)
    assert [s.idx for s in chunks[0].scenes] == [1, 2]
    assert [s.idx for s in chunks[1].scenes] == [3, 4]
    assert [s.idx for s in chunks[2].scenes] == [5]


def test_pair_scenes_before_first_chunk_goes_to_first() -> None:
    chunks = [Chunk(10, "a"), Chunk(20, "b")]
    pair_scenes([_scene(1, 2.0)], chunks)
    assert [s.idx for s in chunks[0].scenes] == [1]


def test_pair_scenes_empty_chunks() -> None:
    assert pair_scenes([_scene(1, 0.0)], []) == []
