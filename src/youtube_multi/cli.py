"""multi: pair YouTube scene-change frames with timestamped transcript chunks."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

from .align import pair_scenes
from .emit import build_manifest, write_json, write_manifest, write_markdown
from .enrich import configure_tesseract, ocr_frame
from .extract import detect_scenes, extract_interval_frames
from .fetch import (
    aggregate_cues,
    download_video,
    fetch_transcript_youtube,
    parse_transcript,
    write_transcript_file,
)
from .models import extract_video_id


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="multi", description=__doc__)
    ap.add_argument("--url", required=True, help="YouTube video URL")
    ap.add_argument(
        "--transcript",
        type=Path,
        default=None,
        help="Path to transcript .txt. If omitted, transcripts/<video_id>.txt is used "
        "(auto-fetched from YouTube captions if missing).",
    )
    ap.add_argument("--out", type=Path, help="Output dir (default: output/<video_id>)")
    ap.add_argument(
        "--lang",
        action="append",
        default=None,
        help="Caption language(s) to try when auto-fetching, in priority order. "
        "Repeat for multiple. Default: en, en-US, en-GB.",
    )
    ap.add_argument(
        "--chunk-seconds",
        type=float,
        default=30.0,
        help="When auto-fetching, aggregate cues into ~N-second blocks. "
        "Set 0 to keep raw per-cue granularity (default: 30).",
    )
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
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    video_id = extract_video_id(args.url)
    out_dir: Path = args.out or Path("output") / video_id
    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    transcript_path: Path = args.transcript or Path("transcripts") / f"{video_id}.txt"
    if not transcript_path.is_file():
        languages = args.lang or ["en", "en-US", "en-GB"]
        print(f"[0/5] fetching captions for {video_id} (langs={','.join(languages)})")
        try:
            cues = fetch_transcript_youtube(video_id, languages)
        except (TranscriptsDisabled, NoTranscriptFound) as e:
            print(
                f"\nNo YouTube captions available for {video_id} "
                f"({type(e).__name__}). Provide a transcript file via --transcript "
                f"or pick another --lang.",
                file=sys.stderr,
            )
            return 3
        except VideoUnavailable as e:
            print(f"\nVideo unavailable: {e}", file=sys.stderr)
            return 4
        chunks_to_save = aggregate_cues(cues, args.chunk_seconds)
        write_transcript_file(chunks_to_save, transcript_path)
        print(f"      ->{len(cues)} cues, saved {len(chunks_to_save)} chunks to {transcript_path}")

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
        configure_tesseract()
        print(f"[3/5] OCR over {len(scenes)} frames")
        for scene in scenes:
            scene.ocr = ocr_frame(scene.image_path, args.ocr_min_chars)

    print(f"[4/5] parsing transcript: {transcript_path}")
    chunks = parse_transcript(transcript_path)
    print(f"      ->{len(chunks)} chunks")
    pair_scenes(scenes, chunks)

    md_path = out_dir / "paired.md"
    json_path = out_dir / "paired.json"
    manifest_path = out_dir / "manifest.json"
    print(f"[5/5] writing {md_path.name}, {json_path.name}, {manifest_path.name}")
    write_markdown(chunks, md_path, args.url, video_id)
    write_json(chunks, json_path, args.url, video_id)
    write_manifest(
        build_manifest(
            args=vars(args),
            video_id=video_id,
            source_url=args.url,
            video_path=video_path,
            ocr_enabled=not args.no_ocr,
        ),
        manifest_path,
    )

    paired_count = sum(1 for c in chunks for _ in c.scenes)
    print(f"done. {paired_count}/{len(scenes)} scenes paired into {len(chunks)} chunks.")
    print(f"  markdown: {md_path}")
    print(f"  json:     {json_path}")
    print(f"  manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
