from __future__ import annotations

from pathlib import Path

import cv2

from youtube_multi.enrich import diff_consecutive, diff_scenes, pixel_diff_region
from youtube_multi.models import Scene

from . import synth


def test_pixel_diff_region_localizes_change() -> None:
    a = synth.to_bgr(synth.moved_rectangle(0))
    b = synth.to_bgr(synth.moved_rectangle(40))
    region, frac = pixel_diff_region(a, b)
    assert region is not None
    x, y, w, h = region
    # union of old (100..180) and new (140..220) rectangle, plus dilation slack
    assert 80 <= x <= 100 and 80 <= y <= 100
    assert 120 <= w <= 160 and 60 <= h <= 90
    assert 0.005 < frac < 0.05
    assert pixel_diff_region(a, a) == (None, 0.0)


def test_diff_scenes_writes_text_and_crop(tmp_path: Path) -> None:
    prev = Scene(1, 0.0, synth.save(synth.moved_rectangle(0), tmp_path / "1.jpg"))
    cur = Scene(2, 5.0, synth.save(synth.moved_rectangle(40), tmp_path / "2.jpg"))
    prev.ocr_lines = ["def main():", "    return 0"]
    cur.ocr_lines = ["def main():", "    print('hi')", "    return 0"]
    d = diff_scenes(prev, cur, tmp_path / "diffs")
    assert d.vs_idx == 1
    assert (d.added, d.removed) == (1, 0)
    assert d.text_path == tmp_path / "diffs" / "0001_0002.txt"
    assert "+    print('hi')" in d.text_path.read_text()
    assert d.image_path == tmp_path / "diffs" / "0001_0002.jpg"
    crop = cv2.imread(str(d.image_path))
    assert d.changed_region is not None
    assert crop.shape[1] == d.changed_region[2] and crop.shape[0] == d.changed_region[3]
    assert crop.shape[0] < 200 and crop.shape[1] < 300


def test_diff_scenes_skips_crop_when_whole_frame_changes(tmp_path: Path) -> None:
    prev = Scene(1, 0.0, synth.save(synth.slide(), tmp_path / "1.jpg"))
    cur = Scene(2, 5.0, synth.save(synth.terminal(), tmp_path / "2.jpg"))
    d = diff_scenes(prev, cur, tmp_path / "diffs")
    assert d.image_path is None
    assert d.text_path is None  # no OCR lines on either side
    assert d.changed_frac > 0.5


def test_diff_consecutive_same_kind_only(tmp_path: Path) -> None:
    imgs = [synth.moved_rectangle(i * 10) for i in range(4)]
    scenes = [Scene(i + 1, float(i), synth.save(im, tmp_path / f"{i}.jpg")) for i, im in enumerate(imgs)]
    scenes[0].kind = scenes[1].kind = "slide"
    scenes[2].kind = "code"
    scenes[3].kind = "code"
    scenes[3].dropped_reason = "duplicate"
    n = diff_consecutive(scenes, tmp_path / "diffs")
    assert n == 1
    assert scenes[1].diff is not None and scenes[1].diff.vs_idx == 1
    assert scenes[2].diff is None  # kind changed
    assert scenes[3].diff is None  # dropped
