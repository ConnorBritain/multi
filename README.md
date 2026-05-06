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

Drop a transcript file in `transcripts/<video_id>.txt` using this format (one block per timestamp, blank line between blocks):

```
00:00:00
First paragraph of transcript text...

00:00:27
Next paragraph...
```

Then run:

```powershell
# scene-change mode (default) — content-driven, variable spacing
uv run multi --url https://youtu.be/EcbgbKtOELY --transcript transcripts/EcbgbKtOELY.txt

# fixed-interval mode — predictable spacing
uv run multi --url https://youtu.be/EcbgbKtOELY --transcript transcripts/EcbgbKtOELY.txt --interval 2

# every half second, no OCR (for very dense sampling on long videos)
uv run multi --url <URL> --transcript <PATH> --interval 0.5 --no-ocr
```

Output lands in `output/<video_id>/` by default, or pass `--out <path>` to override.

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
- `--transcript PATH` (required) — path to the timestamped `.txt` file
- `--out PATH` — output directory (default: `output/<video_id>`)
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

The whole `output/<video_id>/` directory is the artifact. Image paths in `paired.md` are relative to `paired.md`'s location, so the directory is portable.

**Claude Code:** `cd output/<video_id>` and start a session, then ask Claude to `Read paired.md`. Claude will load individual frames via `Read` when relevant.

**Anthropic API or other multimodal SDKs:** read `paired.json`, send `chunks[].text` as text content blocks and `chunks[].frames[].image` paths as image content blocks. The JSON is structured so you can chunk-by-chunk send transcript + frames together.

**Quick eyeball:** open `paired.md` in any markdown viewer that renders relative-path images (VS Code preview works).

## Tuning the frame count

Some rough numbers from a 9-minute UX/UI tutorial (`EcbgbKtOELY`):

| Mode | Frames |
|---|---|
| Scene-detect (defaults) | 81 |
| `--interval 5` | ~110 |
| `--interval 2` | 282 |
| `--interval 1` | ~550 |
| `--interval 0.5` | ~1100 |

For LLM consumption, more frames = more tokens. Start with scene-detect or `--interval 5`; bump density only if the agent is missing visual context.
