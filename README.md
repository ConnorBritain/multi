# multi

Turn a YouTube video into a curated context pack an LLM agent can navigate: deduplicated, classified frames paired with the timestamped transcript, plus the derived data an agent can't compute well on its own (frame diffs, frame kinds, OCR).

Given a YouTube URL and a timestamped transcript file, `multi` produces a directory containing:

- `SKILL.md` — how to read the pack and in what order; the file to hand an agent first
- `index.md` / `index.json` — chapters (from YouTube metadata, or synthesized) → visual states (one row per kept frame: kind, OCR heading, time range, tier) with token estimates per chapter and per frame
- `frames/` — JPGs captured either at scene-change moments or at a fixed time interval, with near-duplicates and talking-head shots removed
- `frames/diffs/` — for consecutive frames of the same kind, a text diff of the OCR and a cropped image of the region that changed
- `paired.md` — the transcript with image references interleaved after the sentence they appeared during, each with its kind, visible time range, OCR text, deep link and diff summary
- `paired.json` — the same data in structured form, plus raw caption cues, chapters, dropped frames and why
- `reconstructed/` — best-effort text folded from code and terminal frames: one file per filename seen in the OCR (or `unknown_N.<ext>`), `commands.sh` for shell commands, with per-line uncertainty markers and a README of confidences
- `entities.md` / `entities.json` — identifiers, commands, URLs, paths and product names from OCR and transcript, each with the time it was first spoken and first shown on screen
- `manifest.json` — how this output was produced (CLI args, library and tool versions, video hash, timestamp)
- `metadata.json` — cached yt-dlp metadata (title, description, chapters)

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
- `--no-metadata` — skip the yt-dlp metadata fetch (title/description/chapters); chapters are then synthesized
- `--budget TOKENS` — image-token budget for tiering: tier 1 fits in `TOKENS`, tier 2 in `2×TOKENS`, the rest is tier 3
- `--no-reconstruct` — skip `reconstructed/`
- `--no-entities` — skip `entities.md` / `entities.json`

## Output shape

```
output/<video_id>/
├── SKILL.md           # read me first: layout, reading order, field reference
├── index.md           # chapters → frames table with tiers and token estimates
├── index.json         # same, structured
├── metadata.json      # yt-dlp title/description/chapters (cached)
├── entities.md        # glossary with first-spoken / first-on-screen links
├── entities.json
├── reconstructed/
│   ├── README.md      # per-file frames, confidence, uncertain lines
│   ├── app.py         # folded from code frames that showed "app.py"
│   ├── unknown_1.js   # filename not visible; language guessed
│   └── commands.sh    # prompt lines from terminal frames, timestamped
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
  "source_url": "...", "video_id": "...", "title": "...", "description": "...", "duration": 540, "channel": "...",
  "manifest": "manifest.json", "index": "index.json", "skill": "SKILL.md", "entities": "entities.json",
  "reconstructed": [{ "path": "reconstructed/app.py", "kind": "code", "frames": [12, 15, 18], "lines": 41, "confidence": 0.87, "uncertain_lines": [9, 30], "filename_source": "ocr" }],
  "budget": { "budget": 20000, "tokens_by_tier": { "1": 19660, "2": 18400, "3": 22100 }, "frames_by_tier": { "1": 16, "2": 15, "3": 18 } },
  "chapters": [{ "index": 0, "title": "Intro", "start": 0, "end": 95, "url": "...", "source": "youtube", "chunk_ids": [0, 1, 2], "chunk_starts": [0, 30, 60], "frame_ids": [1, 4, 9], "tokens": { "text": 410, "images": 3687, "total": 4097 }, "tokens_tier1": 1639 }],
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
      "width": 1280, "height": 720, "tokens_est": 1260, "score": 0.74, "tier": 1, "reconstructed": ["reconstructed/app.py"],
      "diff": null | { "vs": 1, "text": "frames/diffs/0001_0004.txt", "image": "frames/diffs/0001_0004.jpg", "added": 3, "removed": 1, "changed_region": [x, y, w, h], "changed_frac": 0.14 }
    }]
  }],
  "cues": [{ "id": 0, "start": 0.0, "duration": 3.2, "url": "...", "text": "raw caption cue" }],
  "dropped_frames": [{ "idx": 2, "t_seconds": 1.0, "reason": "duplicate", "duplicate_of": 1, "kind": "other", "image": null }]
}
```

## Navigation layer: chapters, tiers, SKILL.md

The pack is meant to be read progressively rather than linearly:

1. `SKILL.md` explains the layout, the reading order and every field. It is generated per pack with the video's title and counts filled in, and uses the same frontmatter agent skill loaders expect (`name`, `description`).
2. `index.md` lists chapters with their time range, the `## HH:MM:SS` sections of `paired.md` they span, and token estimates (text / images / tier-1). Under each chapter is one row per kept frame: kind, OCR heading, visible range, tokens, tier, image link. Chapters come from YouTube's metadata (`metadata.json`, fetched via yt-dlp and cached). Videos without chapters get one synthesized chapter if they are ≤ 6 minutes, otherwise ~3-minute chapters cut at chunk boundaries and titled from the first OCR heading in range.
3. `paired.md` is then read one chapter's sections at a time; `paired.json` / `index.json` carry the same data for programmatic use.

**Token estimates** are deliberately rough: text ≈ characters / 4, images ≈ width × height / 750 (capped at 1,600), which tracks how most vision APIs price a 720p frame. Use them for relative decisions, not billing.

**`--budget TOKENS`** scores every kept frame (`score` = 0.5 × visual novelty vs. the previous frame + 0.3 × OCR density + 0.2 × chunk coverage, where the best frame in each chunk gets the coverage bonus so no chunk goes unrepresented) and assigns `tier` greedily by score: tier 1 until the cumulative image tokens exceed `TOKENS`, tier 2 until `2 × TOKENS`, tier 3 after. A caller includes tier 1 first and escalates. Without `--budget`, scores are still emitted and every frame is tier 1.

## Reconstruction and cross-reference

**`reconstructed/`.** Consecutive kept `code` frames are grouped into runs; a run's filename comes from an editor tab or title bar seen in the OCR (`app.py`, `src/routes.ts` → `routes.ts`), otherwise the language is guessed from token patterns and the file is named `unknown_N.<ext>`. Each run's OCR snapshots are folded into one buffer with `difflib`: the later snapshot wins on changed lines, lines that vanish from the middle count as deletions, lines that vanish at the top or bottom count as scrolling and are kept. The output file starts with a comment naming the contributing frames and the mean OCR confidence; lines that appeared in only one of three or more snapshots, or came from a low-confidence frame, carry a trailing `??` marker (in the language's comment syntax; JSON gets no inline markers). `terminal` frames contribute to `commands.sh`: every prompt-prefixed line becomes a command, preceded by `# t=HH:MM:SS frame NNNN confidence=0.xx`, and low-confidence commands are commented out with `# ??`. `reconstructed/README.md` tabulates all of it. In `paired.json`, `reconstructed[]` lists the files and each frame's `reconstructed` field names the files it fed. This is best-effort by design: it will get indentation, quotes and look-alike glyphs wrong sometimes, and it says so.

**`entities.md` / `entities.json`.** A glossary of `identifier` (snake_case, camelCase, PascalCase, dotted paths), `command` (prompt lines on screen; `npm|pip|uv|git|docker|…` verbs in speech), `url`, `path` and `product` (capitalised names seen at least twice, excluding sentence starts and stopwords). Each entry has `mentions`, `first_spoken` (from the raw cue timeline), `first_on_screen` with the frame index, the list of frames it appears in, and deep links for both first times. Sorted by first appearance.

`--no-reconstruct` and `--no-entities` skip either step.

## Feeding it to an LLM

The whole `output/<video_id>/` directory is the artifact. Image paths are relative, so the directory is portable — copy or zip it and it still works.

**Agentic coding CLIs** (Claude Code, Cursor, Aider, Codex CLI, etc.): point the agent at the directory and ask it to read `SKILL.md`, or drop the directory into the agent's skills folder. It will then use `index.md` to pick chapters and open frames by tier when the visuals matter.

**Direct multimodal API calls** (Anthropic, OpenAI, Google Gemini, local vision models via Ollama / llama.cpp / LM Studio, etc.): iterate `paired.json`'s `chunks[]`, sending each chunk's `text` as a text content block and the `frames[].image` paths whose `tier` is within your budget as image content blocks. Use `index.json`'s per-chapter `tokens` to decide how much of the video fits, and `chapters[].chunk_ids` to send only the chapters that matter.

**Quick human eyeball:** open `index.md` or `paired.md` in any markdown viewer that renders relative-path images (VS Code's preview works out of the box).

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
| `navigate.py` | chapters, token estimates, frame scoring and budget tiers |
| `reconstruct.py` | folding code/terminal OCR into `reconstructed/` files and `commands.sh` |
| `entities.py` | the entity glossary (`entities.md/json`) |
| `emit.py` | `paired.md`, `paired.json`, `index.md/json`, `SKILL.md`, `manifest.json` writers |
| `models.py` | `Scene`/`Chunk` dataclasses and time/URL helpers |
| `cli.py` | argument parsing and orchestration |

Tests need ffmpeg (to synthesize a short fixture video, so no network is required) and optionally tesseract; tests that need a missing tool skip themselves.

```sh
uv sync
uv run pytest
```

Where this is going (local recordings, Whisper fallback, serving a pack as an MCP server): see [docs/roadmap.md](docs/roadmap.md).
