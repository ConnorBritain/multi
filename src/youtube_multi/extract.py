"""Frame extraction: scene-change detection or fixed-interval sampling."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
from scenedetect import ContentDetector, SceneManager, open_video
from scenedetect.scene_manager import save_images

from .models import Scene, short_ms


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

    # save_images returns {scene_index: [paths]} keyed 0-based, one entry per
    # scene in scene_list (scenedetect 0.7). We index it with i-1 only; the old
    # `.get(i-1) or .get(i)` fallback could silently borrow the *next* scene's
    # image when a scene produced none, mis-pairing frames with timestamps.
    image_filenames = save_images(
        scene_list=scene_list,
        video=video,
        num_images=1,
        image_name_template="$SCENE_NUMBER",
        output_dir=str(frames_dir),
        show_progress=False,
    )
    if image_filenames and min(image_filenames) != 0:
        raise RuntimeError(
            f"unexpected save_images key basis: min key = {min(image_filenames)}, expected 0"
        )

    scenes: list[Scene] = []
    for i, (start, _end) in enumerate(scene_list, start=1):
        t = start.seconds
        candidates = image_filenames.get(i - 1) or []
        if not candidates:
            print(f"  [warn] scene {i} @ {t:.2f}s produced no image; skipping", file=sys.stderr)
            continue
        raw_path = frames_dir / candidates[0]
        if not raw_path.exists():
            print(f"  [warn] scene {i} image missing on disk: {raw_path.name}; skipping", file=sys.stderr)
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
