from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from youtube_multi import batch, catalog, cli, transcribe
from youtube_multi.models import Cue
from youtube_multi.fetch import cues_path_for

URL = "https://youtu.be/AAAAAAAAAAA"


def fail(*args, **kwargs):
    raise AssertionError("Unexpected network/media work")


def test_text_pack_has_evidence_without_video_ocr_or_metadata(tmp_path, monkeypatch):
    text = tmp_path / "source.txt"
    transcribe.save_transcript([Cue(0, 0, 4, "An offer starts with a customer."), Cue(1, 4, 6, "Measure the result.")],
                              text, 30, {"method": "youtube_captions", "timing": "caption_cues", "chunk_seconds": 30})
    for name in ("download_video", "download_audio", "fetch_metadata", "fetch_cues_youtube", "configure_tesseract", "detect_scenes"):
        monkeypatch.setattr(cli, name, fail)
    out = tmp_path / "pack"
    assert cli.main(["--url", URL, "--transcript", str(text), "--out", str(out), "--transcript-only",
                     "--no-metadata", "--no-fetch-cues"]) == 0
    data = json.loads((out / "paired.json").read_text())
    assert data["duration"] == 10
    assert data["transcript"]["duration_source"] == "last_cue_end"
    assert data["frame_stats"]["extracted"] == 0
    rows = [json.loads(line) for line in (out / "segments.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and rows[1]["start"] == 4 and rows[1]["end"] == 10
    assert rows[1]["url"].endswith("?t=4")
    assert json.loads((out / "manifest.json").read_text())["video"] is None
    assert not (out / "frames").exists()
    assert "Entities: not generated" not in (out / "SKILL.md").read_text()


def test_edited_transcript_does_not_reuse_stale_cues(tmp_path, monkeypatch):
    path = tmp_path / "text.txt"
    transcribe.save_transcript([Cue(0, 0, 5, "OLD CAPTION")], path, 30,
                              {"method": "youtube_captions", "timing": "caption_cues"})
    path.write_text("00:00:00\nCorrected new text.\n")
    monkeypatch.setattr(cli, "fetch_cues_youtube", fail)
    out = tmp_path / "out"
    assert cli.main(["--url", URL, "--transcript", str(path), "--out", str(out), "--transcript-only", "--no-metadata"]) == 0
    data = json.loads((out / "paired.json").read_text())
    assert data["cues"][0]["text"] == "Corrected new text."
    assert data["transcript"]["method"] == "edited"
    assert data["transcript"]["timing"] == "estimated"


def test_no_captions_default_does_not_download_audio(tmp_path, monkeypatch):
    def absent(*args):
        raise cli.TranscriptsDisabled("AAAAAAAAAAA")
    monkeypatch.setattr(cli, "fetch_cues_youtube", absent)
    monkeypatch.setattr(cli, "download_audio", fail)
    assert cli.main(["--url", URL, "--transcript", str(tmp_path / "t.txt"), "--out", str(tmp_path / "out"),
                     "--transcript-only", "--no-metadata"]) == 3


def test_caption_block_is_reported_without_audio_fallback(tmp_path, monkeypatch):
    from youtube_transcript_api._errors import RequestBlocked
    def blocked(*args):
        raise RequestBlocked("AAAAAAAAAAA")
    monkeypatch.setattr(cli, "fetch_cues_youtube", blocked)
    monkeypatch.setattr(cli, "download_audio", fail)
    assert cli.main(["--url", URL, "--transcript", str(tmp_path / "t.txt"), "--out", str(tmp_path / "out"),
                     "--transcribe", "missing", "--transcript-only", "--no-metadata"]) == 5


def test_asr_local_cache_and_config_invalidation(tmp_path, monkeypatch):
    calls = []
    class Model:
        def __init__(self, name, **kwargs):
            calls.append((name, kwargs))
        def transcribe(self, path, **kwargs):
            assert kwargs["vad_filter"] is True
            return iter([SimpleNamespace(start=0.25, end=2.5, text=" Test sentence. ")]), SimpleNamespace(duration=4, language="en")
    monkeypatch.setattr(transcribe.importlib, "import_module", lambda name: SimpleNamespace(WhisperModel=Model))
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"first")
    text = tmp_path / "t.txt"
    first = transcribe.transcribe_audio(audio, text, model_name="tiny.en", local_files_only=True)
    assert first["timing"] == "asr_segments"
    assert calls[0][1]["local_files_only"] is True
    assert json.loads(cues_path_for(text).read_text())[0]["start"] == 0.25
    assert transcribe.transcribe_audio(audio, text, model_name="tiny.en") == first
    assert len(calls) == 1
    transcribe.transcribe_audio(audio, text, model_name="small.en")
    assert len(calls) == 2
    audio.write_bytes(b"changed audio")
    transcribe.transcribe_audio(audio, text, model_name="small.en")
    assert len(calls) == 3
    cues_path_for(text).write_text("[]")
    transcribe.transcribe_audio(audio, text, model_name="small.en")
    assert len(calls) == 4


def test_local_asr_cli_skips_captions_and_download(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "fetch_cues_youtube", fail)
    monkeypatch.setattr(cli, "download_audio", fail)
    def recognize(audio, text, **kwargs):
        return transcribe.save_transcript([Cue(0, 0, 3, "Local input.")], text, 30,
                                         {"method": "faster_whisper", "timing": "asr_segments", "duration": 8})
    monkeypatch.setattr(cli, "transcribe_audio", recognize)
    out = tmp_path / "pack"
    assert cli.main(["--url", URL, "--transcript", str(tmp_path / "t.txt"), "--out", str(out),
                     "--audio-file", str(tmp_path / "local.wav"), "--transcribe", "always", "--transcript-only", "--no-metadata"]) == 0
    assert json.loads((out / "paired.json").read_text())["duration"] == 8


def test_empty_transcript_is_not_a_success(tmp_path):
    text = tmp_path / "empty.txt"
    text.write_text("")
    assert cli.main(["--url", URL, "--transcript", str(text), "--out", str(tmp_path / "pack"),
                     "--transcript-only", "--no-metadata", "--no-fetch-cues"]) == 2


def test_batch_real_subprocess_resume_and_changed_input(tmp_path):
    text = tmp_path / "source.txt"
    text.write_text("00:00:00\nAn original business explanation.\n")
    manifest = tmp_path / "input.json"
    manifest.write_text(json.dumps({"videos": [{"url": URL, "transcript": "source.txt"},
                                              {"url": "https://youtu.be/BBBBBBBBBBB", "transcript": "source.txt"}]}))
    jobs = batch.load_jobs(manifest)
    root = tmp_path / "batch"
    opts = ["--transcript-only", "--no-metadata", "--no-fetch-cues"]
    assert batch.execute(jobs, root, opts, workers=2, delay=0)["completed"] == 2
    assert batch.execute(jobs, root, opts, workers=2, delay=0)["completed"] == 2
    state = json.loads((root / "batch.json").read_text())
    assert all(v["attempts"] == 1 for v in state["jobs"].values())
    (root / "AAAAAAAAAAA" / "paired.json").write_text("broken")
    batch.execute(jobs, root, opts, workers=2, delay=0)
    state = json.loads((root / "batch.json").read_text())
    assert state["jobs"]["AAAAAAAAAAA"]["attempts"] == 2
    assert state["jobs"]["BBBBBBBBBBB"]["attempts"] == 1
    text.write_text("00:00:00\nChanged business explanation.\n")
    batch.execute(jobs, root, opts, workers=2, delay=0)
    data = json.loads((root / "BBBBBBBBBBB" / "paired.json").read_text())
    assert data["chunks"][0]["text"] == "Changed business explanation."


def test_batch_failure_requires_retry_flag(tmp_path, monkeypatch):
    calls = []
    def failed(*args):
        calls.append(1)
        return {"status": "failed", "exit_code": 5}
    monkeypatch.setattr(batch, "run_one", failed)
    jobs = [{"url": URL, "video_id": "AAAAAAAAAAA"}]
    assert batch.execute(jobs, tmp_path, [], delay=0)["failed"] == 1
    batch.execute(jobs, tmp_path, [], delay=0)
    assert len(calls) == 1
    batch.execute(jobs, tmp_path, [], retry_failed=True, delay=0)
    assert len(calls) == 2


def test_batch_lock_prevents_duplicate_runners(tmp_path):
    with batch.batch_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another batch"):
            with batch.batch_lock(tmp_path):
                pytest.fail("Second writer acquired the lock")
    with batch.batch_lock(tmp_path):
        pass


def test_batch_deduplicates_input_and_retains_unavailable(tmp_path):
    manifest = tmp_path / "input.json"
    manifest.write_text(json.dumps([URL, URL, {"url": "https://youtu.be/BBBBBBBBBBB", "availability": "unavailable"}]))
    jobs = batch.load_jobs(manifest)
    assert len(jobs) == 2
    assert batch.execute(jobs[1:], tmp_path / "out", [], delay=0)["unavailable"] == 1


def test_catalog_paginates_and_retains_unavailable(monkeypatch):
    seen = []
    def request(resource, params, key):
        seen.append((resource, params))
        if resource == "channels":
            return {"items": [{"id": "UCtest", "contentDetails": {"relatedPlaylists": {"uploads": "UUtest"}}, "statistics": {"videoCount": "3"}}]}
        if resource == "playlistItems":
            if "pageToken" not in params:
                return {"items": [{"contentDetails": {"videoId": "AAAAAAAAAAA"}}], "nextPageToken": "page2"}
            return {"items": [{"contentDetails": {"videoId": "BBBBBBBBBBB"}}]}
        return {"items": [{"id": "AAAAAAAAAAA", "contentDetails": {"duration": "PT1H2M3S"}, "status": {"privacyStatus": "public"}}]}
    monkeypatch.setattr(catalog, "_request", request)
    result = catalog.inventory("https://www.youtube.com/@Example/videos", "fake")
    assert result["enumeration_complete"] is True
    assert result["known_duration_seconds"] == 3723
    assert result["unknown_duration_count"] == 1
    assert result["videos"][1]["availability"] == "unavailable"
    assert seen[0][1]["forHandle"] == "@Example"
    assert catalog.inventory("@Example", "fake", limit=1)["enumeration_complete"] is False


def test_missing_api_key_does_not_call_network(tmp_path, monkeypatch):
    monkeypatch.delenv("MULTI_TEST_KEY", raising=False)
    monkeypatch.setattr(catalog, "_request", fail)
    assert catalog.main(["--channel", "@Example", "--out", str(tmp_path / "cat.json"), "--key-env", "MULTI_TEST_KEY"]) == 2


@pytest.mark.parametrize("channel", ["https://www.youtube.com/c/SomeName",
                                    "https://www.youtube.com/user/SomeName",
                                    "https://www.youtube.com/watch?v=AAAAAAAAAAA", "UCinvalid"])
@pytest.mark.parametrize("backend", ["api", "yt-dlp"])
def test_invalid_channel_selector_cannot_catalog_the_wrong_channel(channel, backend, monkeypatch):
    monkeypatch.setattr(catalog, "_request", fail)
    monkeypatch.setattr(catalog, "YoutubeDL", fail)
    with pytest.raises(ValueError, match="Use an @handle"):
        if backend == "api":
            catalog.inventory(channel, "unused")
        else:
            catalog.inventory_ytdlp(channel)


def test_key_free_catalog_interleaves_formats_and_reports_partial(monkeypatch):
    class YDL:
        def __init__(self, options):
            assert options["skip_download"] and options["extract_flat"]
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def extract_info(self, url, download):
            assert download is False
            tab = url.rsplit("/", 1)[-1]
            char = {"videos": "A", "shorts": "B", "streams": "C"}[tab]
            return {"channel_id": "UCtest", "channel": "Example",
                    "entries": [{"id": char * 11, "duration": 10, "title": tab},
                                {"id": char * 10 + "x", "duration": 20, "title": tab}]}
    monkeypatch.setattr(catalog, "YoutubeDL", YDL)
    result = catalog.inventory_ytdlp("@Example", limit=3)
    assert [v["tab"] for v in result["videos"]] == ["videos", "shorts", "streams"]
    assert result["known_duration_seconds"] == 30
    assert result["enumeration_complete"] is False
    assert catalog.inventory_ytdlp("@Example")["enumeration_complete"] is True


def test_configured_certificate_bundle_keeps_tls_verification(monkeypatch):
    from youtube_multi.fetch import network_options
    monkeypatch.setenv("SSL_CERT_FILE", "/configured/ca.pem")
    options = network_options()
    assert options["compat_opts"] == {"no-certifi"}
    assert "nocheckcertificate" not in options


def test_optional_asr_audio_decoder_compatibility(tmp_path):
    # A real decoder call catches incompatible PyAV releases; no weights/network.
    audio = pytest.importorskip("faster_whisper.audio")
    import wave
    source = tmp_path / "silence.wav"
    with wave.open(str(source), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\0\0" * 16000)
    assert len(audio.decode_audio(str(source))) == 16000


@pytest.mark.parametrize("options", [["--interval", "0"], ["--duration-seconds", "nan"], ["--chunk-seconds", "-1"]])
def test_invalid_processing_options_fail_before_network(tmp_path, monkeypatch, options):
    monkeypatch.setattr(cli, "fetch_cues_youtube", fail)
    assert cli.main(["--url", URL, "--out", str(tmp_path / "pack"), *options]) == 2
