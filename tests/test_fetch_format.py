from __future__ import annotations

from yt_dlp import YoutubeDL

from youtube_multi.fetch import VIDEO_FORMAT


def _fmt(fid: str, *, height: int, vcodec: str, acodec: str, ext: str) -> dict:
    return {"format_id": fid, "height": height, "width": height * 16 // 9, "vcodec": vcodec, "acodec": acodec, "ext": ext, "url": f"https://example.invalid/{fid}", "protocol": "https"}


def _pick(formats: list[dict]) -> list[str]:
    selector = YoutubeDL({"quiet": True}).build_format_selector(VIDEO_FORMAT)
    ctx = {"formats": formats, "incomplete_formats": False, "has_merged_format": False}
    return [f["format_id"] for f in selector(ctx)]


# Why: YouTube now serves most resolutions only as separate video and audio streams. If the selector still
# required a combined file, downloads fail with "Requested format is not available" and multi produces nothing.
def test_picks_video_only_h264_up_to_1080p_when_no_combined_file_exists() -> None:
    formats = [
        _fmt("140", height=0, vcodec="none", acodec="mp4a.40.2", ext="m4a"),
        _fmt("137", height=1080, vcodec="avc1.640028", acodec="none", ext="mp4"),
        _fmt("248", height=1080, vcodec="vp9", acodec="none", ext="webm"),
        _fmt("401", height=2160, vcodec="av01.0.12M.08", acodec="none", ext="mp4"),
    ]
    assert _pick(formats) == ["137"]


# Why: OpenCV and PySceneDetect decode H.264 on every platform; if only VP9/AV1 exists the run must still
# get a video stream rather than fail outright.
def test_falls_back_to_any_codec_when_no_h264_stream() -> None:
    formats = [
        _fmt("248", height=1080, vcodec="vp9", acodec="none", ext="webm"),
        _fmt("401", height=2160, vcodec="av01.0.12M.08", acodec="none", ext="mp4"),
    ]
    assert _pick(formats) == ["248"]


# Why: some videos (and other sites) only expose combined files; those must still download.
def test_falls_back_to_combined_file() -> None:
    formats = [_fmt("18", height=360, vcodec="avc1.42001E", acodec="mp4a.40.2", ext="mp4")]
    assert _pick(formats) == ["18"]
