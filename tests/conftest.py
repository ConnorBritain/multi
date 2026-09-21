"""Shared fixtures: a synthetic short video built with ffmpeg (no network)."""
from __future__ import annotations

import glob
import shutil
import subprocess
from pathlib import Path

import pytest

# Four 3-second segments, each a distinct background colour with a distinct
# large label, so scene detection and OCR both have something to find.
# Colours are chosen so consecutive pairs differ in saturation/value, not just
# hue (ContentDetector averages H/S/V deltas; hue-only changes score too low).
SEGMENTS = [
    ("white", "ALPHA ONE"),
    ("yellow", "BRAVO TWO"),
    ("skyblue", "CHARLIE THREE"),
    ("pink", "DELTA FOUR"),
]
SEGMENT_SECONDS = 3
FPS = 10
SIZE = "320x240"


def _font_arg() -> str:
    for pattern in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/**/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/**/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ):
        hits = glob.glob(pattern, recursive=True)
        if hits:
            return f"fontfile={hits[0]}:"
    return ""


def build_fixture_video(target: Path) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg not installed")
    font = _font_arg()
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
    for colour, _ in SEGMENTS:
        cmd += ["-f", "lavfi", "-i", f"color=c={colour}:s={SIZE}:r={FPS}:d={SEGMENT_SECONDS}"]
    chains = []
    for i, (_, label) in enumerate(SEGMENTS):
        chains.append(
            f"[{i}]drawtext={font}text='{label}':fontsize=36:fontcolor=black:"
            f"x=(w-text_w)/2:y=(h-text_h)/2[v{i}]"
        )
    inputs = "".join(f"[v{i}]" for i in range(len(SEGMENTS)))
    chains.append(f"{inputs}concat=n={len(SEGMENTS)}:v=1:a=0[out]")
    cmd += ["-filter_complex", ";".join(chains), "-map", "[out]", "-pix_fmt", "yuv420p", str(target)]
    subprocess.run(cmd, check=True)
    return target


@pytest.fixture(scope="session")
def fixture_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_fixture_video(tmp_path_factory.mktemp("video") / "video.mp4")


FIXTURE_TRANSCRIPT = """00:00:00
This is the first block of the transcript. It covers the alpha segment.

00:00:04
Now we are in the second block, talking about bravo.

00:00:07
Third block. Charlie is on screen now, and we keep talking.

00:00:10
Final block for delta.
"""


@pytest.fixture()
def fixture_transcript(tmp_path: Path) -> Path:
    p = tmp_path / "transcript.txt"
    p.write_text(FIXTURE_TRANSCRIPT, encoding="utf-8")
    return p


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None
