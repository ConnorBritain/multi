"""Phase 0 golden test: refactored CLI output is byte-identical to the legacy
single-file CLI (tests/_legacy_cli.py, a verbatim copy of the pre-refactor
src/youtube_multi/cli.py) for the same inputs.

The fixture video is pre-placed as video.mp4 in the output dir so both CLIs take
download_video's existing "already exists" path and never touch the network.
"""
from __future__ import annotations

import filecmp
import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

from youtube_multi import cli as new_cli

from .conftest import tesseract_available

LEGACY_PATH = Path(__file__).parent / "_legacy_cli.py"
URL = "https://youtu.be/AAAAAAAAAAA"


def _load_legacy():
    spec = importlib.util.spec_from_file_location("_legacy_cli", LEGACY_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod  # dataclasses need the module registered to resolve annotations
    spec.loader.exec_module(mod)
    return mod


def _run(main, argv: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["multi", *argv])
    assert main() == 0


@pytest.mark.parametrize(
    "extra",
    [
        pytest.param(["--no-ocr"], id="scene-noocr"),
        pytest.param(["--interval", "2", "--no-ocr"], id="interval-noocr"),
        pytest.param(
            [],
            id="scene-ocr",
            marks=pytest.mark.skipif(not tesseract_available(), reason="tesseract not installed"),
        ),
    ],
)
def test_outputs_byte_identical(
    fixture_video: Path,
    fixture_transcript: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra: list[str],
) -> None:
    legacy = _load_legacy()
    outs = {}
    for name, main in (("legacy", legacy.main), ("new", new_cli.main)):
        out = tmp_path / name
        out.mkdir()
        shutil.copy(fixture_video, out / "video.mp4")
        _run(main, ["--url", URL, "--transcript", str(fixture_transcript), "--out", str(out), *extra], monkeypatch)
        outs[name] = out

    for fname in ("paired.md", "paired.json"):
        a, b = outs["legacy"] / fname, outs["new"] / fname
        assert a.is_file() and b.is_file()
        assert filecmp.cmp(a, b, shallow=False), f"{fname} differs:\n{a.read_text()}\n---\n{b.read_text()}"

    legacy_frames = sorted(p.name for p in (outs["legacy"] / "frames").iterdir())
    new_frames = sorted(p.name for p in (outs["new"] / "frames").iterdir())
    assert legacy_frames == new_frames

    # Phase 0 additions: manifest exists only for the new CLI.
    assert (outs["new"] / "manifest.json").is_file()
    assert not (outs["legacy"] / "manifest.json").exists()
