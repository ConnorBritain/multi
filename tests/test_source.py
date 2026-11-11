"""Source-agnostic input (local files, Vimeo, Wistia, other URLs), split audio, and the agent/ bundle."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from youtube_multi import cli, transcribe
from youtube_multi.emit import write_markdown
from youtube_multi.models import Chunk, Cue, Scene
from youtube_multi.source import Source, resolve_source


def fail(*args, **kwargs):
    raise AssertionError("network or YouTube-only path must not be called for this source")


def _offline(monkeypatch):
    for name in ("download_video", "download_audio", "fetch_metadata", "fetch_cues_youtube"):
        monkeypatch.setattr(cli, name, fail)


@pytest.mark.parametrize(
    ("url", "kind", "id_", "link_10"),
    [
        ("https://www.youtube.com/watch?v=EcbgbKtOELY", "youtube", "EcbgbKtOELY", "https://youtu.be/EcbgbKtOELY?t=10"),
        ("https://vimeo.com/76979871", "vimeo", "vimeo-76979871", "https://vimeo.com/76979871#t=10s"),
        ("https://player.vimeo.com/video/76979871?h=abc#t=3s", "vimeo", "vimeo-76979871",
         "https://player.vimeo.com/video/76979871?h=abc#t=10s"),
        ("https://acme.wistia.com/medias/abcde12345", "wistia", "wistia-abcde12345",
         "https://acme.wistia.com/medias/abcde12345?wtime=10s"),
        ("https://fast.wistia.net/embed/iframe/abcde12345?videoFoam=true", "wistia", "wistia-abcde12345",
         "https://fast.wistia.net/embed/iframe/abcde12345?videoFoam=true&wtime=10s"),
        ("https://example.com/talk.mp4", "url", None, None),
    ],
)
def test_resolve_source_classifies_urls_and_builds_timestamp_links(url, kind, id_, link_10):
    # Why: deep links are how agents cite moments; a wrong format sends readers to 0:00 or a dead page.
    src = resolve_source(url)
    assert src.kind == kind
    if id_ is not None:
        assert src.id == id_
    assert re.fullmatch(r"[A-Za-z0-9_-]+", src.id), "id names output/<id>/ and must be filesystem-safe"
    assert src.link(10.9) == link_10


def test_resolve_source_local_file_is_stable_and_has_no_web_link(tmp_path):
    # Why: re-runs must land in the same output/<id>/ (caching), and a local file must never get a fake web URL.
    video = tmp_path / "My Talk (720p).mp4"
    video.write_bytes(b"\x00" * 4096)
    a, b = resolve_source(str(video)), resolve_source(str(video))
    assert a.kind == "local" and a.id == b.id and a.id.startswith("my-talk-720p-")
    assert a.path == video.resolve() and a.url is None and a.link(5) is None


def test_resolve_source_rejects_missing_path():
    # Why: a typo in a file path must fail loudly, not be sent to yt-dlp as a "URL".
    with pytest.raises(ValueError, match="neither an existing file"):
        resolve_source("/no/such/file.mp4")


def test_unlinkable_source_writes_plain_times_not_broken_links(tmp_path):
    # Why: a youtu.be link on a non-YouTube pack sends the reader to the wrong video (or none) when they cite a moment.
    chunk = Chunk(start_seconds=5, text="Hello there.")
    chunk.scenes.append(Scene(idx=1, t_seconds=6, image_path=tmp_path / "frames" / "0001_t00m06s.jpg"))
    out = tmp_path / "paired.md"
    write_markdown([chunk], out, "/videos/talk.mp4", "talk-1234",
                   source=Source("local", "talk-1234", None, Path("/videos/talk.mp4")))
    md = out.read_text()
    assert md.startswith("# Paired transcript — talk-1234")
    assert "youtu.be" not in md and "](None)" not in md
    assert re.search(r"^▶ 00:00:05$", md, re.M)


def test_local_video_with_split_audio_runs_offline_and_writes_agent_bundle(fixture_video, tmp_path, monkeypatch):
    # Why: split downloads (video-only file + audio file) are the real-world case that forced a fake YouTube URL;
    # the pack must build from local files alone and hand agents a predictable agent/ folder.
    _offline(monkeypatch)
    audio = tmp_path / "talk-audio.m4a"
    audio.write_bytes(b"audio")
    used: list[Path] = []

    def recognize(audio_path, text, **kwargs):
        used.append(Path(audio_path))
        cues = [Cue(0, 0.0, 4.0, "Alpha is on screen."), Cue(1, 4.0, 4.0, "Now bravo."), Cue(2, 8.0, 4.0, "Charlie and delta.")]
        return transcribe.save_transcript(cues, text, 30, {"method": "faster_whisper", "timing": "asr_segments"})

    monkeypatch.setattr(cli, "transcribe_audio", recognize)
    out = tmp_path / "pack"
    assert cli.main(["--source", str(fixture_video), "--audio-file", str(audio), "--transcribe", "missing",
                     "--transcript", str(tmp_path / "t.txt"), "--out", str(out), "--no-ocr", "--drop", "none"]) == 0
    assert used == [audio], "ASR must read the separate audio file, not the silent video"

    paired = (out / "paired.md").read_text()
    assert "youtu.be" not in paired
    data = json.loads((out / "paired.json").read_text())
    assert all(c["url"] is None for c in data["chunks"])
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["source"]["kind"] == "local" and manifest["source"]["audio_file"] == str(audio)

    agent = out / "agent"
    assert sorted(p.name for p in agent.iterdir()) == ["README.md", "slides", "slides.json", "slides.md", "transcript.md"]
    slides = json.loads((agent / "slides.json").read_text())
    assert slides and [s["n"] for s in slides] == list(range(1, len(slides) + 1))
    for s in slides:
        assert (agent / s["image"]).is_file() and re.fullmatch(r"slides/\d{3}_\d\d-\d\d-\d\d\.jpg", s["image"])
    transcript = (agent / "transcript.md").read_text()
    assert "## 00:00:00" in transcript and "→ slide 001" in transcript
    assert "Alpha is on screen." in (agent / "slides.md").read_text()


def test_bundle_leaves_out_talking_heads(fixture_video, fixture_transcript, tmp_path, monkeypatch):
    # Why: the bundle is the "slides" layer; webcam-only frames waste an agent's image budget and add no content.
    _offline(monkeypatch)
    real_classify = cli.classify_scene

    def classify(scene):
        real_classify(scene)
        if scene.t_seconds < 3:
            scene.kind = "talking-head"

    monkeypatch.setattr(cli, "classify_scene", classify)
    out = tmp_path / "pack"
    assert cli.main(["--source", str(fixture_video), "--transcript", str(fixture_transcript), "--out", str(out),
                     "--no-ocr", "--drop", "none"]) == 0
    slides = json.loads((out / "agent" / "slides.json").read_text())
    assert slides and all(s["kind"] != "talking-head" and s["start"] >= 3 for s in slides)


def test_non_youtube_source_without_transcript_explains_instead_of_calling_youtube(tmp_path, monkeypatch):
    # Why: the caption API is YouTube-only; asking it about a Vimeo id fails confusingly or fetches the wrong video.
    _offline(monkeypatch)
    rc = cli.main(["--source", "https://vimeo.com/76979871", "--transcript", str(tmp_path / "none.txt"),
                   "--out", str(tmp_path / "pack"), "--transcript-only", "--no-metadata"])
    assert rc == 3


def test_url_alias_still_works_for_youtube(tmp_path, monkeypatch):
    # Why: existing scripts and batch.py call `--url`; renaming the flag must not break them.
    monkeypatch.setattr(cli, "fetch_cues_youtube", fail)
    text = tmp_path / "t.txt"
    text.write_text("00:00:00\nHello.\n")
    out = tmp_path / "pack"
    assert cli.main(["--url", "https://youtu.be/AAAAAAAAAAA", "--transcript", str(text), "--out", str(out),
                     "--transcript-only", "--no-metadata", "--no-fetch-cues"]) == 0
    assert json.loads((out / "paired.json").read_text())["chunks"][0]["url"] == "https://youtu.be/AAAAAAAAAAA?t=0"
    assert (out / "agent" / "transcript.md").is_file()


def test_local_source_without_transcript_explains_instead_of_calling_youtube(fixture_video, tmp_path, monkeypatch):
    # Why: a local file has no caption API; without a clear exit the run would hit YouTube with a made-up id.
    _offline(monkeypatch)
    rc = cli.main(["--source", str(fixture_video), "--transcript", str(tmp_path / "none.txt"),
                   "--out", str(tmp_path / "pack"), "--no-ocr"])
    assert rc == 3


def test_non_youtube_download_makes_one_attempt_without_youtube_client_args(tmp_path, monkeypatch):
    # Why: the YouTube player-client retry is meaningless on Vimeo/Wistia and would double every failed download.
    from youtube_multi import fetch

    calls: list[dict] = []

    class FakeYDL:
        def __init__(self, opts):
            calls.append(opts)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download):
            raise fetch.DownloadError("refused")

    monkeypatch.setattr(fetch, "YoutubeDL", FakeYDL)
    with pytest.raises(fetch.DownloadError):
        fetch.download_video("https://vimeo.com/76979871", tmp_path, youtube=False)
    assert len(calls) == 1 and "extractor_args" not in calls[0]
