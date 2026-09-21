from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from youtube_multi.enrich import dedupe_frames, dhash, hamming, hash_scenes, phash
from youtube_multi.models import Scene

from . import synth


def _jpeg_roundtrip(img: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    assert ok
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def test_phash_is_64_bits_and_stable_under_recompression() -> None:
    img = synth.to_bgr(synth.slide())
    h = phash(img)
    assert 0 <= h < 2**64
    assert hamming(h, phash(_jpeg_roundtrip(img, 40))) <= 4
    assert hamming(dhash(img), dhash(_jpeg_roundtrip(img, 40))) <= 4


def test_phash_separates_different_content() -> None:
    a = phash(synth.to_bgr(synth.slide()))
    b = phash(synth.to_bgr(synth.terminal()))
    c = phash(synth.to_bgr(synth.diagram()))
    inverted = phash(255 - synth.to_bgr(synth.slide()))
    assert hamming(a, b) >= 20
    assert hamming(a, c) >= 12
    assert hamming(a, inverted) >= 20
    assert hamming(a, a) == 0


def _scene(idx: int, t: float, h: int) -> Scene:
    s = Scene(idx=idx, t_seconds=t, image_path=Path(f"{idx}.jpg"))
    s.phash = h
    return s


def test_dedupe_marks_duplicates_and_extends_visibility() -> None:
    A, B = 0x0F0F0F0F0F0F0F0F, 0xF0F0F0F0F0F0F0F0
    scenes = [_scene(1, 0.0, A), _scene(2, 3.0, A ^ 0b11), _scene(3, 6.0, A ^ 0b111), _scene(4, 9.0, B), _scene(5, 12.0, B)]
    dupes = dedupe_frames(scenes, max_distance=6, duration=15.0)
    assert dupes == 3
    assert [s.kept for s in scenes] == [True, False, False, True, False]
    assert scenes[1].duplicate_of == 1 and scenes[2].duplicate_of == 1 and scenes[4].duplicate_of == 4
    assert (scenes[0].visible_from, scenes[0].visible_to) == (0.0, 9.0)
    assert (scenes[3].visible_from, scenes[3].visible_to) == (9.0, 15.0)
    # dropped frames keep their own raw range
    assert (scenes[1].visible_from, scenes[1].visible_to) == (3.0, 6.0)


def test_dedupe_disabled_and_exact_only() -> None:
    A = 0x1234
    scenes = [_scene(1, 0.0, A), _scene(2, 1.0, A), _scene(3, 2.0, A ^ 1)]
    assert dedupe_frames(scenes, None, 3.0) == 0
    assert all(s.kept for s in scenes)
    assert scenes[2].visible_to == 3.0
    assert dedupe_frames(scenes, 0, 3.0) == 1
    assert [s.kept for s in scenes] == [True, False, True]


def test_dedupe_empty_and_single() -> None:
    assert dedupe_frames([], 6, 10.0) == 0
    s = [_scene(1, 4.0, 1)]
    assert dedupe_frames(s, 6, 3.0) == 0
    assert s[0].visible_to == 4.0  # never before visible_from


def test_hash_scenes_from_disk(tmp_path: Path) -> None:
    p = synth.save(synth.slide(), tmp_path / "a.jpg")
    s = Scene(idx=1, t_seconds=0, image_path=p)
    hash_scenes([s])
    assert s.phash == phash(cv2.imread(str(p)))
    assert s.dhash != 0
