"""Alignment: attach extracted frames to transcript chunks by time."""
from __future__ import annotations

from .models import Chunk, Scene


def pair_scenes(scenes: list[Scene], chunks: list[Chunk]) -> list[Chunk]:
    """Append each scene to the last chunk whose start is <= the scene time.

    Scenes earlier than the first chunk go into the first chunk.
    """
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
