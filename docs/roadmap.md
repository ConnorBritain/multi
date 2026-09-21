# Roadmap: source-agnostic input and an MCP server over a pack

Design notes only; nothing here is built yet. Both items build on invariants the
current output already has, so they should be additive.

## Invariants the current pack already gives us

- A pack is self-contained: every path in `paired.json` / `index.json` is relative to the pack directory, every time is in seconds, every timestamp carries a deep link. Nothing in the pack depends on YouTube after it is written.
- `paired.json` is the single source of truth; `paired.md`, `index.*`, `SKILL.md`, `entities.*` and `reconstructed/` are derived from the same in-memory objects. Any new input source only has to produce those objects.
- The pipeline is already split by stage: `fetch` (video, cues, metadata) → `extract` → `enrich` → `align` → `navigate` → `emit`. Only `fetch` knows anything about YouTube.

## 1. Source-agnostic input

### Goal

`multi --source <PATH|URL>` accepts a local `.mp4`/`.mkv`/`.mov`, a Loom share link, a Microsoft Teams / SharePoint / Stream recording, or a YouTube URL, and produces the same pack. `--url` stays as an alias for YouTube.

### Shape

A small `Source` protocol in `fetch.py`, with one implementation per kind:

```python
class Source(Protocol):
    id: str                                   # stable id used for output/<id>/ and transcripts/<id>.*
    display_url: str | None                   # what deep links point at, or None
    def video_path(self, out_dir: Path) -> Path: ...          # download or copy-in
    def cues(self, languages: list[str]) -> list[Cue] | None: ...  # captions if the source has them, else None
    def metadata(self) -> dict | None: ...    # title / description / chapters / duration when available
    def deep_link(self, seconds: float) -> str | None: ...
```

- `YouTubeSource`: the current code, moved behind the protocol.
- `LocalFileSource`: `id` = first 12 hex chars of the file's sha256 (already computed for the manifest); `video_path` copies or symlinks the file into the pack; `cues()` looks for a sidecar `.vtt`/`.srt` next to the file (Zoom, OBS and Teams exports produce these) and parses it; `metadata()` reads container tags via ffprobe (`title`, `comment`, chapters from `-show_chapters`); `deep_link` returns `None`, and emit falls back to plain `HH:MM:SS` text where a link would have been.
- `LoomSource`, `TeamsSource`: yt-dlp's generic extractor already handles Loom share pages and many SharePoint/Stream URLs when a cookie file is supplied, so these are thin `YouTubeSource` variants that pass `--cookies-from-browser` / `--cookies` through and use the page's own URL for deep links (Loom supports `?t=<seconds>`; Stream supports `&st=<seconds>`). If yt-dlp cannot fetch the page, fall back to asking for a downloaded file (`LocalFileSource`).
- `deep_link` and `id` live on the source so nothing else in the pipeline changes; `manifest.json` gains a `source` block (`kind`, original path/URL, sha256).

### Whisper fallback when no captions exist

When `cues()` returns `None` and no `transcripts/<id>.txt` exists, transcribe locally:

- Dependency: `faster-whisper` as an optional extra (`uv sync --extra whisper`); it is CTranslate2-based, runs on CPU acceptably for `small`/`base`, and returns word- or segment-level timestamps that map directly onto `Cue(start, duration, text)`. Never a hard dependency: the base install stays pure-Python + ffmpeg + tesseract.
- Audio is extracted once with ffmpeg (`-vn -ac 1 -ar 16000 -f wav`), transcribed with a configurable `--whisper-model` (default `small`) and `--lang`, and the segments are written as the same `transcripts/<id>.cues.json` + aggregated `.txt` the YouTube path writes. Everything downstream is unchanged, and re-runs are cached exactly like fetched captions.
- The manifest records the model name, compute type and that the transcript is machine-generated; `SKILL.md` adds a caveat line so the agent knows the transcript is Whisper output rather than author-provided captions.
- Word-level timestamps, when available, replace the character-proportional sentence timing in `align.time_sentences` with exact sentence starts; the function already takes cues, so this is a precision upgrade, not a redesign.

### Order of work

1. Extract `Source` protocol and `YouTubeSource` (no behaviour change; tests keep passing).
2. `LocalFileSource` with `.vtt`/`.srt` sidecars and ffprobe metadata; the pytest fixture video becomes a first-class input, which also lets the e2e test run without network.
3. `faster-whisper` extra and the audio path.
4. Loom / Teams via yt-dlp with cookies; document the cookie requirement.

## 2. Serving a pack as an MCP server

### Goal

`multi serve output/<id>/` exposes a pack to any MCP client (Claude Code, Cursor, etc.) so the agent navigates by tool calls instead of reading files, and only the parts it asks for enter its context.

### Shape

- Dependency: the `mcp` Python SDK as an optional extra (`uv sync --extra mcp`), stdio transport by default. The server reads only `index.json` and `paired.json` (plus files they reference) at startup and keeps them in memory; a pack is a few hundred KB of JSON.
- Tools (all return compact JSON; images are returned as MCP image content so the client can show them):
  - `chapters()` → `index.json.chapters` without `visual_states` (title, range, chunk starts, token totals).
  - `chapter(index)` → one chapter with its visual states and the chunk texts it spans.
  - `chunk(start_seconds)` → the paired chunk: sentences, frames (metadata only), cue ids.
  - `frame_at(t, kind=None)` → the kept frame visible at `t` (via `visible_from`/`visible_to`), its OCR lines, diff summary, and the image; `kind` filters (e.g. only `code`).
  - `frame(idx, image=True)` → a specific frame by index, optionally with the JPEG.
  - `search(q, scope="all")` → hits across transcript sentences, OCR lines and entities, ranked by simple BM25 over the pre-tokenised text, each hit with its time, chunk, frame and deep link. `scope` narrows to `transcript` / `ocr` / `entities` / `reconstructed`.
  - `entity(name)` → the glossary entry with first-spoken / first-on-screen and frames.
  - `reconstructed(path=None)` → the list of reconstructed files, or one file's text with its uncertain-line numbers.
  - `budget(tokens)` → re-tiers frames on the fly for a given budget and returns the tier-1 set, so a client with a different context size does not need a re-run.
- Resources: `pack://SKILL.md`, `pack://index.md`, `pack://paired.md` for clients that prefer to read whole documents.
- Multiple packs: `multi serve output/` serves every subdirectory that has an `index.json`, with a `packs()` tool and a `pack` argument on the others.

### Why this is cheap

Every tool is a lookup or filter over data that `emit.py` already computes; the server adds no new analysis. The one piece of new code with any depth is the tokenised search index, which can start as substring matching over sentences and OCR lines and be upgraded later. The `frame_at` / `budget` tools reuse `navigate.assign_tiers` and the `visible_*` ranges directly.

### Order of work

1. `serve` subcommand with `chapters`, `chapter`, `chunk`, `frame`, `frame_at` and the three resources.
2. `search` over sentences + OCR + entities.
3. `budget`, `reconstructed`, `entity`, multi-pack mode.
