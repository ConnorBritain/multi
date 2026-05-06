# multi

Pair a YouTube video's frames with its timestamped transcript so a multimodal LLM can *see* the visuals while reading the words.

Given a YouTube URL and a timestamped transcript file, `multi` produces a directory containing:

- `frames/` — JPGs captured either at scene-change moments or at a fixed time interval
- `paired.md` — the transcript with image references interleaved at the right timestamps (and OCR'd text per frame)
- `paired.json` — the same data in structured form

Designed for tutorial/lecture content where the visual and the spoken word need to be reasoned about together.

## Requirements

- Windows 11 (paths are Windows-flavored; macOS/Linux work with minor tesseract path tweaks)
- Python 3.13
- [uv](https://docs.astral.sh/uv/)
- ffmpeg — `winget install Gyan.FFmpeg`
- Tesseract OCR — `winget install UB-Mannheim.TesseractOCR` (only needed if you don't pass `--no-ocr`)

## Install

```powershell
git clone https://github.com/ConnorBritain/multi.git
cd multi
uv sync
```

## Usage

Just give it a URL. Captions are fetched from YouTube on the first run and cached in `transcripts/<video_id>.txt`.

```powershell
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
- `--no-ocr` — skip OCR

## Output shape

```
output/<video_id>/
├── video.mp4          # downloaded source (gitignored)
├── frames/
│   ├── 0001_t00m00s.jpg
│   ├── 0002_t00m02s.jpg
│   └── ...
├── paired.md          # transcript with embedded ![](frames/...) references
└── paired.json        # structured: { source_url, video_id, chunks: [{ start_seconds, text, frames: [...] }] }
```

`paired.md` shape:

```markdown
# YouTube paired transcript — <video_id>

Source: https://youtu.be/...

---

## 00:00:00

First paragraph of the transcript...

![scene 0001 @ 00:00:00](frames/0001_t00m00s.jpg)
> OCR: extracted text from the slide

![scene 0002 @ 00:00:02](frames/0002_t00m02s.jpg)

## 00:00:27

Next paragraph...
```

## Feeding it to an LLM

The whole `output/<video_id>/` directory is the artifact. Image paths in `paired.md` are relative to `paired.md`'s location, so the directory is portable — copy or zip it and it still works.

**Agentic coding CLIs** (Claude Code, Cursor, Aider, Codex CLI, etc.): point the agent at the directory and ask it to read `paired.md`. Any agent with a file-read + image-read tool will load the JPGs on demand when the visuals matter.

**Direct multimodal API calls** (Anthropic, OpenAI, Google Gemini, local vision models via Ollama / llama.cpp / LM Studio, etc.): iterate `paired.json`'s `chunks[]`, sending each chunk's `text` as a text content block and `frames[].image` paths as image content blocks. The JSON is shaped so you can stream chunk-by-chunk to stay under context limits on long videos.

**Quick human eyeball:** open `paired.md` in any markdown viewer that renders relative-path images (VS Code's preview works out of the box).

## Tuning the frame count

Rough numbers for a 9-minute video:

| Mode | Frames |
|---|---|
| Scene-detect (defaults) | ~80 |
| `--interval 5` | ~110 |
| `--interval 2` | ~280 |
| `--interval 1` | ~550 |
| `--interval 0.5` | ~1100 |

For LLM consumption, more frames = more tokens. Start with scene-detect or `--interval 5`; bump density only if the agent is missing visual context.
