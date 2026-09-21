# multi

Turn a YouTube video into a curated context pack an LLM agent can navigate: deduplicated, classified frames paired with the timestamped transcript, plus the derived data an agent can't compute well on its own (frame diffs, frame kinds, OCR).

Given a YouTube URL and a timestamped transcript file, `multi` produces a directory containing:

- `frames/` — JPGs captured either at scene-change moments or at a fixed time interval, with near-duplicates and talking-head shots removed
- `frames/diffs/` — for consecutive frames of the same kind, a text diff of the OCR and a cropped image of the region that changed
- `paired.md` — the transcript with image references interleaved at the right timestamps, each with its kind, visible time range, OCR text and diff summary
- `paired.json` — the same data in structured form, plus the list of dropped frames and why
- `manifest.json` — how this output was produced (CLI args, library and tool versions, video hash, timestamp)

Designed for tutorial/lecture content where the visual and the spoken word need to be reasoned about together.

## Requirements

- Python 3.13 and [uv](https://docs.astral.sh/uv/)
- ffmpeg
- Tesseract OCR (only needed if you don't pass `--no-ocr`)

| Platform | ffmpeg | Tesseract |
|---|---|---|
| Windows | `winget install Gyan.FFmpeg` | `winget install UB-Mannheim.TesseractOCR` |
| macOS | `brew install ffmpeg` | `brew install tesseract` |
| Debian/Ubuntu | `apt install ffmpeg` | `apt install tesseract-ocr` |

Tesseract is found via `$TESSERACT_CMD`, then `PATH`, then the default install location for your platform.

## Install

```sh
git clone https://github.com/ConnorBritain/multi.git
cd multi
uv sync
```

## Usage

Just give it a URL. Captions are fetched from YouTube on the first run and cached in `transcripts/<video_id>.txt`.

```sh
# scene-change mode (default) — content-driven, variable spacing
uv run multi --url https://youtu.be/EcbgbKtOELY

# fixed-interval mode — predictable spacing
uv run multi --url https://youtu.be/EcbgbKtOELY --interval 2

# every half second, no OCR (for dense sampling on long videos)
uv run multi --url <URL> --interval 0.5 --no-ocr
```

Output lands in `output/<video_id>/` by default, or pass `--out <path>` to override.

### Transcript handling

On every run, `multi` looks for `transcripts/<video_id>.txt` (or whatever `--transcript` points at):

- **File exists** → use it as-is.
- **File missing** → fetch YouTube captions via `youtube-transcript-api`, aggregate cues into ~30s blocks (override with `--chunk-seconds`), and save to that path.

This means re-runs are cached and you can hand-edit the transcript file between runs to fix bad auto-captions.

The raw per-cue captions (float start, duration, text) are saved next to the transcript as `transcripts/<video_id>.cues.json` and are what sentence-level timing is built from. If that file is missing but the `.txt` exists (a transcript from an older run, or one you wrote by hand), `multi` fetches the cues once and caches them; pass `--no-fetch-cues` to stay offline, in which case each chunk becomes a single cue and sentence times are interpolated across the chunk. Hand edits to the `.txt` never touch the cues file.

### Sentence-level placement and deep links

Each chunk's text is split into sentences, and each sentence gets a start time by mapping its character offset onto the chunk's cue timeline. A frame is then inserted in `paired.md` right after the sentence that was being spoken when it appeared (`sentence_index` in JSON; `-1` means before the chunk's first sentence). Each frame also records `cue_ids_visible`, the cues spoken while it was on screen (over its `visible_from`–`visible_to` range).

Every timestamp in `paired.md` and `paired.json` carries a `https://youtu.be/<id>?t=<seconds>` link (`url` fields; `[▶ HH:MM:SS](...)` lines in markdown) so an agent can cite or jump to the exact moment.

If a video has no captions at all, you can supply your own:

```
00:00:00
First paragraph of transcript text...

00:00:27
Next paragraph...
```

Drop that in `transcripts/<video_id>.txt` (or pass `--transcript <path>`) and `multi` will skip the fetch step.

## Two extraction modes

| Mode | When to use | Flag |
|---|---|---|
| **Scene-change** (default) | Slide decks, tutorials, demos with discrete visual states | `--threshold 27.0 --min-scene-len 1.5` |
| **Fixed interval** | Continuous-motion content, or you want predictable density | `--interval 0.5` / `1` / `2` / `5` |

### How scene-change is determined

PySceneDetect's `ContentDetector` converts each frame to HSV, computes the average per-pixel delta from the previous frame across H/S/V, and weights them into a single content score. When that score exceeds `--threshold` (default `27.0`), a new scene boundary is recorded. `--min-scene-len` (default `1.5s`) suppresses flicker. Lower threshold = more frames; higher = fewer.

When `--interval` is set, scene detection is skipped entirely and `--threshold` / `--min-scene-len` are ignored.

## Dedupe, classify, drop, diff

After extraction every frame goes through four cheap passes (no models, no network):

1. **Dedupe.** Each frame gets a 64-bit perceptual hash (DCT pHash, computed with OpenCV). Consecutive frames whose hashes are within `--dedupe-distance` (default `6`) of the current survivor are dropped as duplicates; the survivor's `visible_from`/`visible_to` range is extended to cover them. `--no-dedupe` keeps everything. This is what makes `--interval` modes usable: a 1-second grab of a static slide collapses to one frame with a visible range.
2. **OCR** runs only on surviving frames and now keeps line structure (`ocr_lines`) and mean word confidence (`ocr_confidence`) alongside the flat `ocr` string.
3. **Classify.** Each frame is labelled `slide | code | terminal | diagram | talking-head | whiteboard | other` with a confidence in `[0, 1]`, from OCR character density, monospace/indent structure, code and shell-prompt tokens, Haar face detection, luminance/saturation stats, and line/box detection. It is a rule table, not a model: expect it to be right on the obvious cases and low-confidence on the rest.
4. **Drop.** Frames whose kind is in `--drop` are removed. The default drops `talking-head`; pass `--drop none` to keep everything or `--drop talking-head --drop other` (or `--drop talking-head,other`) to drop more. Dropped files are deleted unless `--keep-dropped` moves them to `frames/dropped/`. Every dropped frame is still listed in `paired.json` under `dropped_frames` with its reason.
5. **Diff.** For consecutive kept frames of the same kind, `frames/diffs/<prev>_<cur>.txt` is a unified diff of the OCR lines and `frames/diffs/<prev>_<cur>.jpg` is a crop of the region that changed (only written when the change covers less than 60% of the frame). `paired.md` summarises each diff under the frame; `paired.json` carries the paths, line counts, changed region and changed-pixel fraction.

## Flags

- `--url URL` (required) — YouTube URL
- `--transcript PATH` — path to a transcript `.txt` (default: `transcripts/<video_id>.txt`, auto-fetched if missing)
- `--out PATH` — output directory (default: `output/<video_id>`)
- `--lang LANG` — caption language to try when auto-fetching; repeat for fallbacks (default: `en`, `en-US`, `en-GB`)
- `--chunk-seconds FLOAT` — when auto-fetching, aggregate cues into ~N-second blocks (default `30`, `0` keeps raw per-cue granularity)
- `--interval FLOAT` — grab a frame every N seconds (e.g. `0.5`, `1`, `2`, `5`); disables scene detection
- `--threshold FLOAT` — PySceneDetect content threshold (default `27.0`, scene mode only)
- `--min-scene-len FLOAT` — minimum scene length in seconds (default `1.5`, scene mode only)
- `--ocr-min-chars INT` — drop OCR results shorter than this (default `5`)
- `--no-ocr` — skip OCR (classification then relies on image features only)
- `--dedupe-distance INT` — max pHash hamming distance for consecutive frames to count as duplicates (default `6`, `0` = exact only)
- `--no-dedupe` — keep near-duplicate frames
- `--drop KIND` — drop frames of this kind; repeatable or comma-separated (default `talking-head`; `none` keeps all)
- `--keep-dropped` — move dropped frames to `frames/dropped/` instead of deleting them
- `--no-diff` — skip OCR/pixel diffs between consecutive frames
- `--no-fetch-cues` — never contact YouTube for raw cues when the transcript file already exists

## Output shape

```
output/<video_id>/
├── video.mp4          # downloaded source (gitignored)
├── frames/
│   ├── 0001_t00m00s.jpg   # kept frames only; idx gaps are dropped frames
│   ├── 0004_t00m03s.jpg
│   ├── diffs/
│   │   ├── 0001_0004.txt  # unified diff of OCR lines
│   │   └── 0001_0004.jpg  # crop of the changed region
│   └── dropped/           # only with --keep-dropped
├── paired.md          # transcript with embedded ![](frames/...) references
├── paired.json        # structured, see below
└── manifest.json      # run provenance: args, lib/tool versions, video sha256, created_at
```

`paired.md` shape:

```markdown
# YouTube paired transcript — <video_id>

Source: https://youtu.be/...

---

## 00:00:00

[▶ 00:00:00](https://youtu.be/<video_id>?t=0)

First sentence of the transcript. Second sentence, during which the first frame appeared.

![scene 0001 @ 00:00:05](frames/0001_t00m05s.jpg)
> kind: slide (0.81) · visible 00:00:05–00:00:12 · [▶ 00:00:05](https://youtu.be/<video_id>?t=5)
> OCR: extracted text from the slide

Third sentence. Fourth sentence.

![scene 0004 @ 00:00:12](frames/0004_t00m12s.jpg)
> kind: slide (0.77) · visible 00:00:12–00:00:31 · [▶ 00:00:12](https://youtu.be/<video_id>?t=12)
> OCR: extracted text from the next slide
> diff vs 0001: +3 −1 lines, 14% of pixels changed (frames/diffs/0001_0004.txt, frames/diffs/0001_0004.jpg)

## 00:00:27

[▶ 00:00:27](https://youtu.be/<video_id>?t=27)

Next paragraph...
```

`paired.json` shape (fields from the original release are unchanged; new ones are additive):

```jsonc
{
  "source_url": "...", "video_id": "...", "manifest": "manifest.json",
  "frame_stats": { "extracted": 80, "duplicates": 22, "dropped_by_kind": { "talking-head": 9 }, "kept": 49, "kept_by_kind": { "slide": 40, "code": 9 } },
  "chunks": [{
    "start_seconds": 0, "start_hms": "00:00:00", "url": "https://youtu.be/<id>?t=0", "text": "...",
    "sentences": [{ "index": 0, "start": 0.0, "url": "...", "text": "First sentence." }],
    "cue_ids": [0, 1, 2],
    "frames": [{
      "idx": 1, "t_seconds": 5.0, "url": "https://youtu.be/<id>?t=5", "image": "frames/0001_t00m05s.jpg", "ocr": "...",
      "ocr_lines": ["..."], "ocr_confidence": 91.3,
      "kind": "slide", "kind_confidence": 0.81, "phash": "ebd1940e6bf1940e",
      "visible_from": 5.0, "visible_to": 12.0,
      "sentence_index": 1, "cue_ids_visible": [1, 2],
      "diff": null | { "vs": 1, "text": "frames/diffs/0001_0004.txt", "image": "frames/diffs/0001_0004.jpg", "added": 3, "removed": 1, "changed_region": [x, y, w, h], "changed_frac": 0.14 }
    }]
  }],
  "cues": [{ "id": 0, "start": 0.0, "duration": 3.2, "url": "...", "text": "raw caption cue" }],
  "dropped_frames": [{ "idx": 2, "t_seconds": 1.0, "reason": "duplicate", "duplicate_of": 1, "kind": "other", "image": null }]
}
```

## Feeding it to an LLM

The whole `output/<video_id>/` directory is the artifact. Image paths in `paired.md` are relative to `paired.md`'s location, so the directory is portable — copy or zip it and it still works.

**Agentic coding CLIs** (Claude Code, Cursor, Aider, Codex CLI, etc.): point the agent at the directory and ask it to read `paired.md`. Any agent with a file-read + image-read tool will load the JPGs on demand when the visuals matter.

**Direct multimodal API calls** (Anthropic, OpenAI, Google Gemini, local vision models via Ollama / llama.cpp / LM Studio, etc.): iterate `paired.json`'s `chunks[]`, sending each chunk's `text` as a text content block and `frames[].image` paths as image content blocks. The JSON is shaped so you can stream chunk-by-chunk to stay under context limits on long videos.

**Quick human eyeball:** open `paired.md` in any markdown viewer that renders relative-path images (VS Code's preview works out of the box).

## Tuning the frame count

Rough numbers for a 9-minute video, before dedupe/drop:

| Mode | Frames extracted |
|---|---|
| Scene-detect (defaults) | ~80 |
| `--interval 5` | ~110 |
| `--interval 2` | ~280 |
| `--interval 1` | ~550 |
| `--interval 0.5` | ~1100 |

Dedupe and the default `--drop talking-head` then remove whatever is visually redundant, so the kept count depends on the content, not the mode. `frame_stats` in `paired.json` reports extracted / duplicates / dropped-by-kind / kept for each run. As a rule of thumb a 1280×720 frame costs an LLM roughly 1,200 tokens (about width × height / 750), so every dropped frame is a real saving.

Start with scene-detect or `--interval 5`; bump density only if the agent is missing visual context. Dense intervals are far cheaper than they used to be because static stretches collapse to a single frame with a visible range.

## Development

The code lives in `src/youtube_multi/`, one module per pipeline stage:

| Module | Responsibility |
|---|---|
| `fetch.py` | video download, caption/cue fetch, transcript and cues file read/write |
| `extract.py` | scene-change detection and fixed-interval frame grabs |
| `enrich.py` | tesseract discovery, OCR, pHash dedupe, kind classification, diffs |
| `align.py` | pairing frames to chunks, sentence splitting/timing, cue overlap |
| `emit.py` | `paired.md`, `paired.json`, `manifest.json` writers |
| `models.py` | `Scene`/`Chunk` dataclasses and time/URL helpers |
| `cli.py` | argument parsing and orchestration |

Tests need ffmpeg (to synthesize a short fixture video, so no network is required) and optionally tesseract; tests that need a missing tool skip themselves.

```sh
uv sync
uv run pytest
```
