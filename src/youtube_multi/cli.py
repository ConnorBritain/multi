"""multi: pair YouTube scene-change frames with timestamped transcript chunks."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import pytesseract
from PIL import Image
from scenedetect import ContentDetector, SceneManager, open_video
from scenedetect.scene_manager import save_images
from yt_dlp import YoutubeDL


_TESSERACT_FALLBACKS = [
    Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Tesseract-OCR" / "tesseract.exe",
]


def _configure_tesseract() -> None:
    from shutil import which
    if which("tesseract"):
        return
    for candidate in _TESSERACT_FALLBACKS:
        if candidate.is_file():
            pytesseract.pytesseract.tesseract_cmd = str(candidate)
            return
    raise RuntimeError(
        "tesseract not found on PATH or in known install locations. "
        "Install with: winget install UB-Mannheim.TesseractOCR"
    )


@dataclass
class Scene:
    idx: int
    t_seconds: float
    image_path: Path
    ocr: str = ""


@dataclass
class Chunk:
    start_seconds: int
    text: str
    scenes: list[Scene] = field(default_factory=list)


_VIDEO_ID_RE = re.compile(r"(?:v=|youtu\.be/|/embed/|/shorts/)([A-Za-z0-9_-]{11})")


def extract_video_id(url: str) -> str:
    m = _VIDEO_ID_RE.search(url)
    if not m:
        raise ValueError(f"could not extract video id from URL: {url}")
    return m.group(1)


def hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def short_ms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60:02d}m{s % 60:02d}s"


def download_video(url: str, out_dir: Path) -> Path:
    target = out_dir / "video.mp4"
    if target.exists():
        print(f"  [skip] {target.name} already exists")
        return target
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": "b[ext=mp4][height<=720]/b[height<=720]/b[ext=mp4]/b",
        "outtmpl": str(out_dir / "video.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    with YoutubeDL(opts) as ydl:
        ydl.download([url])
    if not target.exists():
        produced = list(out_dir.glob("video.*"))
        if produced:
            produced[0].rename(target)
    return target


def detect_scenes(
    video_path: Path,
    frames_dir: Path,
    threshold: float,
    min_scene_len: float,
) -> list[Scene]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    video = open_video(str(video_path))
    fps = video.frame_rate
    min_len_frames = max(1, int(min_scene_len * fps))
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_len_frames))
    sm.detect_scenes(video, show_progress=False)
    scene_list = sm.get_scene_list()

    image_filenames = save_images(
        scene_list=scene_list,
        video=video,
        num_images=1,
        image_name_template="$SCENE_NUMBER",
        output_dir=str(frames_dir),
        show_progress=False,
    )

    scenes: list[Scene] = []
    for i, (start, _end) in enumerate(scene_list, start=1):
        t = start.get_seconds()
        candidates = image_filenames.get(i - 1) or image_filenames.get(i) or []
        if not candidates:
            continue
        raw_path = frames_dir / candidates[0]
        if not raw_path.exists():
            continue
        new_name = f"{i:04d}_t{short_ms(t)}.jpg"
        new_path = frames_dir / new_name
        if raw_path != new_path:
            if new_path.exists():
                new_path.unlink()
            raw_path.rename(new_path)
        scenes.append(Scene(idx=i, t_seconds=t, image_path=new_path))
    return scenes


def extract_interval_frames(
    video_path: Path,
    frames_dir: Path,
    interval_seconds: float,
) -> list[Scene]:
    if interval_seconds <= 0:
        raise ValueError("interval must be > 0")
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0.0

    scenes: list[Scene] = []
    t = 0.0
    idx = 1
    try:
        while t <= duration:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            name = f"{idx:04d}_t{short_ms(t)}.jpg"
            out_path = frames_dir / name
            cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            scenes.append(Scene(idx=idx, t_seconds=t, image_path=out_path))
            idx += 1
            t += interval_seconds
    finally:
        cap.release()
    return scenes


def ocr_frame(image_path: Path, min_chars: int) -> str:
    try:
        with Image.open(image_path) as img:
            text = pytesseract.image_to_string(img)
    except Exception as e:
        print(f"  [ocr-warn] {image_path.name}: {e}", file=sys.stderr)
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) >= min_chars else ""


_TS_RE = re.compile(r"^(\d{1,2}):(\d{2}):(\d{2})$")


def parse_transcript(path: Path) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8").strip()
    blocks = re.split(r"\n\s*\n", raw)
    chunks: list[Chunk] = []
    for block in blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue
        m = _TS_RE.match(lines[0].strip())
        if not m:
            continue
        h, mi, s = (int(g) for g in m.groups())
        start = h * 3600 + mi * 60 + s
        text = " ".join(line.strip() for line in lines[1:]).strip()
        if not text:
            continue
        chunks.append(Chunk(start_seconds=start, text=text))
    chunks.sort(key=lambda c: c.start_seconds)
    return chunks


def pair_scenes(scenes: list[Scene], chunks: list[Chunk]) -> list[Chunk]:
    if not chunks:
        return []
    starts = [c.start_seconds for c in chunks]
    for scene in scenes:
        idx = 0
        for i, s in enumerate(starts):
            if scene.t_seconds >= s:
                idx = i
            else:
                break
        chunks[idx].scenes.append(scene)
    return chunks


def write_markdown(
    chunks: list[Chunk],
    out_path: Path,
    source_url: str,
    video_id: str,
) -> None:
    lines: list[str] = [
        f"# YouTube paired transcript — {video_id}",
        "",
        f"Source: {source_url}",
        "",
        "---",
        "",
    ]
    for chunk in chunks:
        lines.append(f"## {hms(chunk.start_seconds)}")
        lines.append("")
        lines.append(chunk.text)
        lines.append("")
        for scene in chunk.scenes:
            rel = scene.image_path.relative_to(out_path.parent).as_posix()
            lines.append(f"![scene {scene.idx:04d} @ {hms(scene.t_seconds)}]({rel})")
            if scene.ocr:
                lines.append(f"> OCR: {scene.ocr}")
            lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_json(
    chunks: list[Chunk],
    out_path: Path,
    source_url: str,
    video_id: str,
) -> None:
    payload = {
        "source_url": source_url,
        "video_id": video_id,
        "chunks": [
            {
                "start_seconds": c.start_seconds,
                "start_hms": hms(c.start_seconds),
                "text": c.text,
                "frames": [
                    {
                        "idx": s.idx,
                        "t_seconds": round(s.t_seconds, 2),
                        "image": s.image_path.relative_to(out_path.parent).as_posix(),
                        "ocr": s.ocr,
                    }
                    for s in c.scenes
                ],
            }
            for c in chunks
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(prog="multi", description=__doc__)
    ap.add_argument("--url", required=True, help="YouTube video URL")
    ap.add_argument("--transcript", required=True, type=Path, help="Path to transcript .txt")
    ap.add_argument("--out", type=Path, help="Output dir (default: output/<video_id>)")
    ap.add_argument(
        "--interval",
        type=float,
        default=None,
        help="Grab a frame every N seconds (e.g. 0.5, 1, 2, 5). "
        "When set, scene detection is skipped.",
    )
    ap.add_argument("--threshold", type=float, default=27.0, help="PySceneDetect content threshold (scene-detect mode)")
    ap.add_argument("--min-scene-len", type=float, default=1.5, help="Minimum scene length in seconds (scene-detect mode)")
    ap.add_argument("--ocr-min-chars", type=int, default=5, help="Drop OCR results shorter than this")
    ap.add_argument("--no-ocr", action="store_true", help="Skip OCR step")
    args = ap.parse_args()

    if not args.transcript.is_file():
        print(f"transcript not found: {args.transcript}", file=sys.stderr)
        return 2

    video_id = extract_video_id(args.url)
    out_dir: Path = args.out or Path("output") / video_id
    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] downloading video ->{out_dir / 'video.mp4'}")
    video_path = download_video(args.url, out_dir)

    if args.interval is not None:
        print(f"[2/5] grabbing frames every {args.interval}s (interval mode)")
        scenes = extract_interval_frames(video_path, frames_dir, args.interval)
    else:
        print(f"[2/5] detecting scenes (threshold={args.threshold}, min_len={args.min_scene_len}s)")
        scenes = detect_scenes(video_path, frames_dir, args.threshold, args.min_scene_len)
    print(f"      ->{len(scenes)} frames")

    if args.no_ocr:
        print("[3/5] OCR skipped (--no-ocr)")
    else:
        _configure_tesseract()
        print(f"[3/5] OCR over {len(scenes)} frames")
        for scene in scenes:
            scene.ocr = ocr_frame(scene.image_path, args.ocr_min_chars)

    print(f"[4/5] parsing transcript: {args.transcript}")
    chunks = parse_transcript(args.transcript)
    print(f"      ->{len(chunks)} chunks")
    pair_scenes(scenes, chunks)

    md_path = out_dir / "paired.md"
    json_path = out_dir / "paired.json"
    print(f"[5/5] writing {md_path.name}, {json_path.name}")
    write_markdown(chunks, md_path, args.url, video_id)
    write_json(chunks, json_path, args.url, video_id)

    paired_count = sum(1 for c in chunks for _ in c.scenes)
    print(f"done. {paired_count}/{len(scenes)} scenes paired into {len(chunks)} chunks.")
    print(f"  markdown: {md_path}")
    print(f"  json:     {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
