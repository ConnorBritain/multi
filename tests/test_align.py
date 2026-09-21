from __future__ import annotations

from pathlib import Path

import pytest

from youtube_multi.align import align, assign_cues, pair_scenes, place_scenes, split_sentences, time_sentences
from youtube_multi.fetch import synthesize_cues
from youtube_multi.models import Chunk, Cue, Scene, deep_link


def _scene(idx: int, t: float, visible_to: float | None = None) -> Scene:
    s = Scene(idx=idx, t_seconds=t, image_path=Path(f"frames/{idx:04d}.jpg"))
    s.visible_from = t
    s.visible_to = visible_to if visible_to is not None else t
    return s


def test_pair_scenes_places_by_time() -> None:
    chunks = [Chunk(0, "a"), Chunk(30, "b"), Chunk(60, "c")]
    scenes = [_scene(1, 0.0), _scene(2, 29.99), _scene(3, 30.0), _scene(4, 45), _scene(5, 999)]
    pair_scenes(scenes, chunks)
    assert [s.idx for s in chunks[0].scenes] == [1, 2]
    assert [s.idx for s in chunks[1].scenes] == [3, 4]
    assert [s.idx for s in chunks[2].scenes] == [5]


def test_pair_scenes_before_first_chunk_goes_to_first() -> None:
    chunks = [Chunk(10, "a"), Chunk(20, "b")]
    pair_scenes([_scene(1, 2.0)], chunks)
    assert [s.idx for s in chunks[0].scenes] == [1]


def test_pair_scenes_empty_chunks() -> None:
    assert pair_scenes([_scene(1, 0.0)], []) == []


# --- sentences -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hello world. This is two. And three!", ["Hello world.", "This is two.", "And three!"]),
        ("Talk to Dr. Smith today. He agrees.", ["Talk to Dr. Smith today.", "He agrees."]),
        ("Use e.g. a container. Then toggle.", ["Use e.g. a container.", "Then toggle."]),
        ("Is it done? Yes.   Really.", ["Is it done?", "Yes.", "Really."]),
        ("no capitals here. so no split", ["no capitals here. so no split"]),
        ("Version 2.0 shipped. Then 3.1 came out.", ["Version 2.0 shipped.", "Then 3.1 came out."]),
        ('He said "Go." Then left.', ['He said "Go."', "Then left."]),
        ("", []),
        ("   ", []),
    ],
)
def test_split_sentences(text: str, expected: list[str]) -> None:
    assert split_sentences(text) == expected


def test_assign_cues() -> None:
    chunks = [Chunk(0, "a"), Chunk(30, "b")]
    cues = [Cue(0, 0.0, 2, "x"), Cue(1, 29.5, 2, "y"), Cue(2, 30.0, 2, "z"), Cue(3, 100, 2, "w")]
    assign_cues(cues, chunks)
    assert [c.id for c in chunks[0].cues] == [0, 1]
    assert [c.id for c in chunks[1].cues] == [2, 3]


def test_time_sentences_interpolates_across_cues() -> None:
    # Two cues of equal text length: first half of the chunk text spans 30-40, second 40-50.
    chunk = Chunk(30, "First sentence here. Second sentence here. Third one now.")
    chunk.cues = [Cue(0, 30.0, 10.0, "First sentence here. Second"), Cue(1, 40.0, 10.0, "sentence here. Third one now.")]
    time_sentences([chunk], duration=60.0)
    starts = [s.start for s in chunk.sentences]
    assert starts[0] == 30.0
    assert 36.0 <= starts[1] <= 38.5  # ~21/59 of the way through the text
    assert 44.0 <= starts[2] <= 46.5


def test_time_sentences_without_cues_uses_chunk_span() -> None:
    chunks = [Chunk(0, "One. Two. Three. Four."), Chunk(40, "Last.")]
    time_sentences(chunks, duration=100.0)
    starts = [s.start for s in chunks[0].sentences]
    assert starts[0] == 0.0
    assert starts == sorted(starts)
    # character-proportional across 0-40: offsets 5/23, 10/23, 17/23
    assert 7 <= starts[1] <= 11 and 16 <= starts[2] <= 19 and 28 <= starts[3] <= 31
    assert chunks[1].sentences[0].start == 40.0
    assert [s.index for s in chunks[0].sentences] == [0, 1, 2, 3]


def test_place_scenes_sentence_index_and_visible_cues() -> None:
    chunks = [Chunk(30, "Alpha alpha alpha. Beta beta beta. Gamma gamma gamma."), Chunk(60, "Delta.")]
    cues = [Cue(0, 30, 10, "Alpha alpha alpha."), Cue(1, 40, 10, "Beta beta beta."), Cue(2, 50, 10, "Gamma gamma gamma."), Cue(3, 60, 5, "Delta.")]
    scenes = [_scene(1, 25.0, 30.0), _scene(2, 41.0, 55.0), _scene(3, 59.0, 70.0)]
    align(scenes, chunks, cues, duration=80.0)
    assert scenes[0].sentence_index == -1  # before the chunk's first sentence
    assert scenes[0].cue_ids_visible == []  # visible 25-30 overlaps no cue (cue 0 starts at 30)
    assert scenes[1].sentence_index == 1  # during "Beta"
    assert scenes[1].cue_ids_visible == [1, 2]
    assert scenes[2].sentence_index == 2
    assert scenes[2].cue_ids_visible == [2, 3]
    assert [c.id for c in chunks[0].cues] == [0, 1, 2]


def test_place_scenes_zero_length_visibility_matches_containing_cue() -> None:
    chunks = [Chunk(0, "A. B.")]
    cues = [Cue(0, 0, 5, "A."), Cue(1, 5, 5, "B.")]
    scenes = [_scene(1, 6.0)]  # visible_to == visible_from
    align(scenes, chunks, cues, duration=10.0)
    assert scenes[0].cue_ids_visible == [1]


def test_synthesize_cues_and_deep_link() -> None:
    cues = synthesize_cues([Chunk(0, "a"), Chunk(30, "b")], duration=45.0)
    assert [(c.id, c.start, c.duration, c.text) for c in cues] == [(0, 0.0, 30.0, "a"), (1, 30.0, 15.0, "b")]
    assert deep_link("EcbgbKtOELY", 61.9) == "https://youtu.be/EcbgbKtOELY?t=61"
