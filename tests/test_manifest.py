from __future__ import annotations

import json
from pathlib import Path

from youtube_multi.emit import build_manifest, sha256_file, write_manifest


def test_manifest_shape(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"not really a video")
    m = build_manifest(
        args={"url": "u", "out": tmp_path, "lang": None, "interval": 2.0},
        video_id="abc",
        source_url="u",
        video_path=video,
        ocr_enabled=False,
    )
    assert m["tool"] == "multi"
    assert m["video_id"] == "abc"
    assert m["args"]["out"] == str(tmp_path)  # Paths stringified
    assert m["video"] == {"path": "video.mp4", "sha256": sha256_file(video), "bytes": 18}
    assert set(m["libs"]) >= {"yt-dlp", "scenedetect", "opencv-python", "pytesseract"}
    assert m["tools"]["tesseract"] is None  # OCR disabled -> not probed
    assert m["created_at"].endswith("+00:00")
    out = tmp_path / "manifest.json"
    write_manifest(m, out)
    assert json.loads(out.read_text())["video"]["sha256"] == m["video"]["sha256"]
