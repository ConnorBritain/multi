"""End-to-end on the synthetic fixture (video pre-placed, no network) plus the
schema-compat contract: every Phase 0 key in paired.json/paired.md survives."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from youtube_multi import cli

from .conftest import SEGMENTS, tesseract_available

URL = "https://youtu.be/AAAAAAAAAAA"

P0_TOP_KEYS = {"source_url": str, "video_id": str, "chunks": list}
P0_CHUNK_KEYS = {"start_seconds": int, "start_hms": str, "text": str, "frames": list}
P0_FRAME_KEYS = {"idx": int, "t_seconds": (int, float), "image": str, "ocr": str}


def _run(fixture_video: Path, transcript: Path, out: Path, *extra: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(fixture_video, out / "video.mp4")
    # --no-fetch-cues keeps the suite offline; the fixture transcript has no cues.json.
    assert cli.main(["--url", URL, "--transcript", str(transcript), "--out", str(out), "--no-fetch-cues", *extra]) == 0
    return json.loads((out / "paired.json").read_text(encoding="utf-8"))


def _assert_schema_compat(data: dict, md: str) -> None:
    for k, t in P0_TOP_KEYS.items():
        assert isinstance(data[k], t), k
    for c in data["chunks"]:
        for k, t in P0_CHUNK_KEYS.items():
            assert isinstance(c[k], t), k
        for f in c["frames"]:
            for k, t in P0_FRAME_KEYS.items():
                assert isinstance(f[k], t), k
    assert md.startswith("# YouTube paired transcript — ")
    assert re.search(r"^## \d\d:\d\d:\d\d$", md, re.M)
    assert re.search(r"^!\[scene \d{4} @ \d\d:\d\d:\d\d\]\(frames/\d{4}_t\d\dm\d\ds\.jpg\)$", md, re.M)
    # Phase 2: every chunk and frame carries a deep link
    assert re.search(r"^\[▶ \d\d:\d\d:\d\d\]\(https://youtu\.be/AAAAAAAAAAA\?t=\d+\)$", md, re.M)
    for c in data["chunks"]:
        assert c["url"] == f"https://youtu.be/AAAAAAAAAAA?t={c['start_seconds']}"
        assert isinstance(c["sentences"], list) and isinstance(c["cue_ids"], list)
        for f in c["frames"]:
            assert f["url"].startswith("https://youtu.be/AAAAAAAAAAA?t=")
            assert isinstance(f["sentence_index"], int) and isinstance(f["cue_ids_visible"], list)
    assert isinstance(data["cues"], list) and data["cues"], "cues synthesized when no cues.json exists"
    # Phase 3: navigation layer
    assert data["index"] == "index.json" and data["skill"] == "SKILL.md"
    assert isinstance(data["chapters"], list) and data["chapters"]
    for f in (f for c in data["chunks"] for f in c["frames"]):
        assert f["tokens_est"] > 0 and 0.0 <= f["score"] <= 1.0 and f["tier"] in (1, 2, 3)
        assert f["width"] > 0 and f["height"] > 0


def _check_index(out: Path, data: dict) -> dict:
    index = json.loads((out / "index.json").read_text(encoding="utf-8"))
    paired_frames = {f["idx"]: f for c in data["chunks"] for f in c["frames"]}
    seen: list[int] = []
    for ch in index["chapters"]:
        assert ch["start"] < ch["end"]
        for vs in ch["visual_states"]:
            assert vs["frame_idx"] in paired_frames
            assert (out / vs["image"]).is_file()
            assert vs["tier"] == paired_frames[vs["frame_idx"]]["tier"]
            seen.append(vs["frame_idx"])
        assert ch["frame_ids"] == [vs["frame_idx"] for vs in ch["visual_states"]]
        for ci in ch["chunk_ids"]:
            assert 0 <= ci < len(data["chunks"])
    assert sorted(seen) == sorted(paired_frames), "every kept frame appears in exactly one chapter"
    assert index["tokens"]["total"] == index["tokens"]["text"] + index["tokens"]["images"]
    md = (out / "index.md").read_text(encoding="utf-8")
    assert md.startswith("# ") and "| # | Chapter |" in md
    skill = (out / "SKILL.md").read_text(encoding="utf-8")
    assert skill.startswith("---\nname: video-context-pack-AAAAAAAAAAA\n")
    assert "## Recommended reading order" in skill
    return index


@pytest.mark.parametrize("ocr", [pytest.param(False, id="noocr"), pytest.param(True, id="ocr", marks=pytest.mark.skipif(not tesseract_available(), reason="no tesseract"))])
def test_interval_dedupe_end_to_end(fixture_video: Path, fixture_transcript: Path, tmp_path: Path, ocr: bool) -> None:
    out = tmp_path / "out"
    data = _run(fixture_video, fixture_transcript, out, "--interval", "1", "--no-metadata", *([] if ocr else ["--no-ocr"]))
    md = (out / "paired.md").read_text(encoding="utf-8")
    _assert_schema_compat(data, md)
    index = _check_index(out, data)
    assert index["chapters_source"] == "synthesized" and len(index["chapters"]) == 1  # 12s video: one chapter
    assert data["budget"]["budget"] is None
    assert all(f["tier"] == 1 for c in data["chunks"] for f in c["frames"])

    stats = data["frame_stats"]
    assert stats["extracted"] in (12, 13)
    assert stats["kept"] == len(SEGMENTS)
    assert stats["duplicates"] == stats["extracted"] - len(SEGMENTS)
    assert stats["dropped_by_kind"] == {}

    frames = [f for c in data["chunks"] for f in c["frames"]]
    assert [f["idx"] for f in frames] == [1, 4, 7, 10]
    assert frames[0]["visible_from"] == 0.0 and frames[0]["visible_to"] == 3.0
    assert frames[-1]["visible_to"] == pytest.approx(12.0, abs=0.2)
    for f in frames:
        assert re.fullmatch(r"[0-9a-f]{16}", f["phash"])
        assert f["kind"] in cli.KINDS
        assert (out / f["image"]).is_file()
        assert isinstance(f["ocr_lines"], list)
    if ocr:
        assert [f["ocr"] for f in frames] == [label for _, label in SEGMENTS]

    on_disk = sorted(p.name for p in (out / "frames").glob("*.jpg"))
    assert on_disk == sorted(Path(f["image"]).name for f in frames), "dropped frames deleted from frames/"
    assert not (out / "frames" / "dropped").exists()
    assert len(data["dropped_frames"]) == stats["duplicates"]
    assert all(d["reason"] == "duplicate" and d["image"] is None for d in data["dropped_frames"])
    assert data["manifest"] == "manifest.json" and (out / "manifest.json").is_file()
    # Consecutive kept frames of the same kind carry a diff; others don't.
    for prev, cur in zip(frames, frames[1:]):
        if prev["kind"] == cur["kind"]:
            assert cur["diff"] is not None and cur["diff"]["vs"] == prev["idx"]
            for key in ("text", "image"):
                if cur["diff"][key]:
                    assert (out / cur["diff"][key]).is_file()
        else:
            assert cur["diff"] is None


def test_keep_dropped_and_drop_kind(fixture_video: Path, fixture_transcript: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    # Drop every kind that shows up so the drop path is exercised regardless of classifier output.
    data = _run(fixture_video, fixture_transcript, out, "--interval", "1", "--no-ocr", "--keep-dropped",
                "--drop", ",".join(cli.KINDS))
    stats = data["frame_stats"]
    assert stats["kept"] == 0
    assert sum(stats["dropped_by_kind"].values()) == len(SEGMENTS)
    kept_dir = sorted(p.name for p in (out / "frames").glob("*.jpg"))
    assert kept_dir == []
    dropped_dir = sorted(p.name for p in (out / "frames" / "dropped").glob("*.jpg"))
    assert len(dropped_dir) == stats["extracted"]
    for d in data["dropped_frames"]:
        assert d["image"].startswith("frames/dropped/")
        assert (out / d["image"]).is_file()


def test_no_dedupe_keeps_all(fixture_video: Path, fixture_transcript: Path, tmp_path: Path) -> None:
    data = _run(fixture_video, fixture_transcript, tmp_path / "out", "--interval", "2", "--no-ocr", "--no-dedupe", "--drop", "none", "--no-diff")
    stats = data["frame_stats"]
    assert stats["duplicates"] == 0 and stats["kept"] == stats["extracted"]
    assert all(f["diff"] is None for c in data["chunks"] for f in c["frames"])


def test_budget_tiers_and_metadata_chapters(fixture_video: Path, fixture_transcript: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    # Pre-seed the metadata cache so no network is needed and chapters come from "youtube".
    (out / "metadata.json").write_text(json.dumps({
        "title": "Fixture Video", "description": "desc", "duration": 12,
        "chapters": [{"start_time": 0, "end_time": 6, "title": "First half"}, {"start_time": 6, "end_time": 12, "title": "Second half"}],
    }), encoding="utf-8")
    # 4 kept frames of ~103 image tokens each (320x240): budget 250 -> 2 tier-1, 2 tier-2.
    data = _run(fixture_video, fixture_transcript, out, "--interval", "1", "--no-ocr", "--budget", "250")
    assert data["title"] == "Fixture Video"
    assert [c["title"] for c in data["chapters"]] == ["First half", "Second half"]
    index = _check_index(out, data)
    assert index["chapters_source"] == "youtube"
    assert [len(c["visual_states"]) for c in index["chapters"]] == [2, 2]
    tiers = sorted(f["tier"] for c in data["chunks"] for f in c["frames"])
    assert tiers == [1, 1, 2, 2]
    assert data["budget"]["budget"] == 250 and index["budget"]["frames_by_tier"] == {"1": 2, "2": 2, "3": 0}
    assert index["tokens"]["tier1"] == index["tokens"]["text"] + index["budget"]["tokens_by_tier"]["1"]
    skill = (out / "SKILL.md").read_text(encoding="utf-8")
    assert "Fixture Video" in skill and "budget of 250" in skill


def test_bad_drop_kind_exits_2(fixture_transcript: Path, tmp_path: Path) -> None:
    assert cli.main(["--url", URL, "--transcript", str(fixture_transcript), "--out", str(tmp_path), "--drop", "faces"]) == 2
