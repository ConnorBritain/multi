from __future__ import annotations

from pathlib import Path

import pytest

from youtube_multi.enrich import (
    classify_features,
    classify_scene,
    drop_by_kind,
    image_features,
    ocr_scene,
    parse_drop_kinds,
    text_features,
)
from youtube_multi.models import Scene

from . import synth
from .conftest import tesseract_available

# --- rule table, with hand-built features ------------------------------------------

BASE = {
    "n_lines": 0.0, "n_chars": 0.0, "megapixels": 0.23, "char_width_cv": 1.0, "indent_levels": 0.0,
    "dark_frac": 0.05, "light_frac": 0.8, "mean_sat": 20.0, "sat_frac": 0.1, "n_faces": 0.0, "face_frac": 0.0,
    "edge_density": 0.01, "n_hough_lines": 0.0, "n_rects": 0.0, "code_ratio": 0.0, "prompt_ratio": 0.0,
}


def feats(**over: float) -> dict[str, float]:
    return {**BASE, **over}


@pytest.mark.parametrize(
    "override, expected",
    [
        ({"n_faces": 1, "face_frac": 0.06, "light_frac": 0.2, "mean_sat": 90}, "talking-head"),
        ({"n_faces": 1, "face_frac": 0.06, "n_lines": 2, "n_chars": 40, "light_frac": 0.3}, "talking-head"),  # lower-third caption
        ({"n_lines": 10, "n_chars": 400, "dark_frac": 0.85, "light_frac": 0.02, "char_width_cv": 0.05, "prompt_ratio": 0.4}, "terminal"),
        ({"n_lines": 14, "n_chars": 500, "dark_frac": 0.1, "char_width_cv": 0.05, "indent_levels": 4, "code_ratio": 0.5}, "code"),
        ({"n_lines": 14, "n_chars": 500, "dark_frac": 0.9, "light_frac": 0.0, "char_width_cv": 0.05, "indent_levels": 4, "code_ratio": 0.6}, "code"),
        ({"n_lines": 6, "n_chars": 220, "char_width_cv": 0.35}, "slide"),
        ({"n_rects": 6, "n_hough_lines": 25, "n_lines": 3, "n_chars": 25}, "diagram"),
        ({"light_frac": 0.9, "mean_sat": 8, "edge_density": 0.05, "n_lines": 1, "n_chars": 8}, "whiteboard"),
        ({}, "other"),
    ],
)
def test_rule_table(override: dict[str, float], expected: str) -> None:
    kind, conf = classify_features(feats(**override))
    assert kind == expected, (kind, conf)
    assert 0.0 <= conf <= 1.0


def test_pip_face_over_dense_slide_prefers_slide() -> None:
    # A small picture-in-picture face over a text-heavy slide should stay a slide.
    kind, _ = classify_features(feats(n_faces=1, face_frac=0.01, n_lines=12, n_chars=900, char_width_cv=0.4))
    assert kind == "slide"


def test_text_features() -> None:
    tf = text_features(["$ ls -la", "total 12", "def main():", "    return 0"])
    assert tf["prompt_ratio"] == pytest.approx(0.25)
    assert tf["code_ratio"] == pytest.approx(0.5)
    assert text_features([]) == {"code_ratio": 0.0, "prompt_ratio": 0.0}


# --- rendered synthetic images through the real feature extractors -------------------


def _scene_for(img, tmp_path: Path, name: str) -> Scene:
    p = synth.save(img, tmp_path / f"{name}.png")
    return Scene(idx=1, t_seconds=0.0, image_path=p)


def test_image_features_no_false_faces_on_slide(tmp_path: Path) -> None:
    f = image_features(synth.to_bgr(synth.slide()))
    assert f["n_faces"] == 0
    assert f["light_frac"] > 0.7
    f2 = image_features(synth.to_bgr(synth.terminal()))
    assert f2["dark_frac"] > 0.8


def test_image_features_diagram_has_boxes_and_lines() -> None:
    f = image_features(synth.to_bgr(synth.diagram()))
    assert f["n_rects"] >= 4
    assert f["n_hough_lines"] >= 8


@pytest.mark.skipif(not tesseract_available(), reason="tesseract not installed")
@pytest.mark.parametrize(
    "maker, expected",
    [(synth.slide, "slide"), (synth.terminal, "terminal"), (synth.code, "code"), (synth.diagram, "diagram")],
)
def test_classify_rendered(maker, expected: str, tmp_path: Path) -> None:
    s = _scene_for(maker(), tmp_path, expected)
    ocr_scene(s, min_chars=5)
    classify_scene(s)
    assert s.kind == expected, (s.kind, s.kind_confidence, s.features, s.ocr_lines)
    assert s.kind_confidence > 0.2


@pytest.mark.skipif(not tesseract_available(), reason="tesseract not installed")
def test_ocr_scene_lines_and_confidence(tmp_path: Path) -> None:
    s = _scene_for(synth.terminal(), tmp_path, "term")
    ocr_scene(s, min_chars=5)
    assert any(line.startswith("$ uv sync") for line in s.ocr_lines), s.ocr_lines
    assert s.ocr == " ".join(s.ocr_lines)
    assert 50 < s.ocr_confidence <= 100
    assert s.features["n_lines"] >= 8
    assert s.features["char_width_cv"] < 0.22  # monospace


# --- drop -----------------------------------------------------------------------------


def test_parse_drop_kinds() -> None:
    assert parse_drop_kinds(None) == ["talking-head"]
    assert parse_drop_kinds(["none"]) == []
    assert parse_drop_kinds(["slide,code", "Terminal"]) == ["slide", "code", "terminal"]
    with pytest.raises(ValueError):
        parse_drop_kinds(["faces"])


def test_drop_by_kind() -> None:
    a = Scene(1, 0.0, Path("a")); a.kind = "talking-head"
    b = Scene(2, 1.0, Path("b")); b.kind = "slide"
    c = Scene(3, 2.0, Path("c")); c.kind = "talking-head"; c.dropped_reason = "duplicate"
    assert drop_by_kind([a, b, c], ["talking-head"]) == 1
    assert a.dropped_reason == "kind:talking-head" and b.kept and c.dropped_reason == "duplicate"
