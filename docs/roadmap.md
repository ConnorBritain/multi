# Roadmap: source-agnostic input and an MCP server over a pack

Status update (October 2026): transcript-only packs, catalog discovery, resumable
batches, and optional local Whisper transcription are implemented; see
[corpus.md](corpus.md). Source-agnostic input (`--source` with local files, Vimeo, Wistia and other
yt-dlp URLs, plus `--audio-file` for split downloads) is implemented in `source.py`; sidecar
`.vtt`/`.srt` parsing, ffprobe chapters, Loom/Teams cookies and the MCP server remain proposals.

## Invariants the current pack already gives us

- A pack is self-contained: every path in `paired.json` / `index.json` is relative to the pack directory, every time is in seconds, every timestamp carries a deep link. Nothing in the pack depends on YouTube after it is written.
- `paired.json` is the single source of truth; `paired.md`, `index.*`, `SKILL.md`, `entities.*` and `reconstructed/` are derived from the same in-memory objects. Any new input source only has to produce those objects.
- The pipeline is already split by stage: `fetch` (video, cues, metadata) → `extract` → `enrich` → `align` → `navigate` → `emit`. Only `fetch` knows anything about YouTube.

## 1. Source-agnostic input

### Shipped (November 2026)

`multi --source <PATH|URL>` (`--url` is an alias) accepts a local video file, a YouTube URL, a Vimeo or Wistia
URL, or any other URL yt-dlp can fetch. `source.py` holds a frozen `Source` dataclass with `kind`
(`youtube | vimeo | wistia | url | local`), `id`, `url`, `path` and `link(seconds)`; the writers take the
`Source` and nothing else in the pipeline knows the provider.

- Ids: YouTube video id; `vimeo-<n>`; `wistia-<hash>`; `<host-slug>-<8 hex of the URL>`; local files
  `<file-name-slug>-<8 hex of size + first MiB>`.
- Deep links: `youtu.be/<id>?t=<s>`, Vimeo `#t=<s>s`, Wistia `?wtime=<s>s`; generic URLs and local files get
  `None`, so markdown shows plain `▶ HH:MM:SS` and JSON `url` fields are `null`.
- Captions are fetched only for YouTube. Other sources need `--transcript` or `--transcribe missing|always`
  (faster-whisper); `--audio-file` supplies a separate audio track for split downloads.
- `manifest.json` gains a `source` block (`kind`, `id`, `url`, `path`, `audio_file`, `linkable`).
- Every run writes `agent/` (README, transcript with slide markers, slides.md/json, slides/) for agents.

### Still planned

- Sidecar `.vtt`/`.srt` next to a local file as its transcript, and ffprobe container tags / chapters as metadata.
- Loom / Teams / Stream via yt-dlp with `--cookies-from-browser`, using each page's own time parameter.
- Word-level ASR timestamps replacing character-proportional sentence timing in `align.time_sentences`.

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
