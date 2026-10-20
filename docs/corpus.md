# Corpus processing with no paid model API

The default path fetches captions and performs local processing. Optional speech
recognition runs on your hardware. There are no paid transcription, embedding,
or LLM endpoints in this implementation. Electricity, hardware, bandwidth, and
any cloud machine you choose to run still have their own costs.

## 1. Discover sources

For a small sample without an API key:

```sh
uv run multi catalog --channel @AlexHormozi --backend yt-dlp --limit 30 --out catalog.json
```

This reads the videos, Shorts, and streams tabs without downloading media. Limited
results are interleaved across those tabs. Durations may be unknown, particularly
for Shorts; `known_duration_seconds` is a lower bound. Read `warnings`,
`unknown_duration_count`, and `enumeration_complete`. Omitting `--limit` requests
the entire accessible tab inventory; it cannot discover private/deleted uploads
or appearances on other channels. Key-free extraction depends on YouTube's web
interface and may stop working or be blocked.

For the official API, enable YouTube Data API v3 in a Google project and supply
the key in the `YOUTUBE_API_KEY` environment variable through your preferred secret
manager or shell configuration. Do not put it in the catalog or Git.

```sh
uv run multi catalog --channel @AlexHormozi --out catalog.json
```

The API backend resolves the uploads playlist, follows every page, and hydrates
metadata in groups of 50. It records unavailable entries. A limited scan never
claims completeness. This uses the API's quota; it is not a paid AI call or a
reason to open a new Cloud trial. Caption downloading is separate: the official
caption API requires video-edit permission, so catalog credentials do not unlock
third-party captions.

## 2. Build text packs first

```sh
uv run multi batch --input catalog.json --out output/pilot --workers 2
```

The default batch path is transcript-only. Catalog metadata is cached per video.
Each pack contains `paired.md/json`, `index.md/json`, `SKILL.md`, `manifest.json`,
and `segments.jsonl` with stable content-based IDs and source-time links.
Caption cues and provenance are cached with the transcript. No video is
downloaded and no frame extraction or OCR runs in this mode.

You can also provide a text file of one YouTube URL per line (blank lines and `#`
comments are ignored). Duplicate URLs are processed once. A JSON list or a JSON
object with a `videos` array can include per-video input files:

```json
{
  "videos": [
    {"url": "https://youtu.be/AAAAAAAAAAA", "transcript": "inputs/transcript.txt"},
    {"url": "https://youtu.be/BBBBBBBBBBB", "audio_file": "inputs/recording.wav"}
  ]
}
```

Relative file paths resolve from the manifest's directory. Transcript files use
the existing `HH:MM:SS`, newline, paragraph format. An `audio_file` entry explicitly
selects local ASR for that row and requires the optional Whisper dependencies.
Each source URL must correspond to the provided media; Multi cannot verify that
association. Ordinary batch options follow `--`; per-video URLs and paths belong
in the manifest.

For an offline supplied-transcript run, also disable metadata and cue fetching:

```sh
uv run multi batch --input local-inputs.json --out output/local -- --no-metadata --no-fetch-cues
```

## 3. Transcribe only the gaps

```sh
uv sync --extra whisper
uv run --extra whisper multi batch --input catalog.json --out output/pilot --retry-failed -- --transcribe missing
```

`--transcribe never` is the default. `missing` downloads audio and runs
faster-whisper only when captions are disabled or unavailable in the requested
languages. Network errors, blocks, and unavailable videos remain failures; they
do not silently trigger another acquisition route.

Use `always` to prefer local ASR even if captions exist, or to provide your own
permitted audio/video file:

```sh
uv run --extra whisper multi --url https://youtu.be/AAAAAAAAAAA --transcript-only --transcribe always --audio-file recording.wav --whisper-model small.en --no-metadata
```

Defaults are CPU, int8, four CPU threads, English, and `small.en`. Multilingual
audio needs a multilingual model such as `small` and the appropriate
`--whisper-language` (or `auto`). `--lang` controls caption language independently.
GPU selection is explicit through `--whisper-device cuda`; compatible NVIDIA
libraries are required. On Apple Silicon this adapter uses CPU; it is not a Metal
backend. Try the pilot before choosing larger models or more workers.

The first model use downloads weights; `--local-model-only` requires them to be
cached or a local model directory to be supplied. ASR reuse checks the input-audio
hash, model/configuration, and transcript/cue hashes. Changing those invalidates
the cache. Model names can resolve to updated upstream weights; pin a local model
directory for reproducible research. The optional dependencies bound PyAV below
19 because its removed `metadata_errors` argument breaks faster-whisper 1.2.

## 4. Resume, inspect failures, and bound concurrency

`output/pilot/batch.json` is written atomically after dispatch and completion.
It records attempts, input/configuration/code fingerprints, output checksums,
elapsed worker times, and failures. Logs are in each video's `worker.log`.
Re-running the same command skips verified completed packs. Modified inputs,
configuration, code, or missing/altered output files trigger regeneration. Raw
caption/ASR caches can still be reused during regeneration.

Failed jobs require `--retry-failed`; there is no unbounded retry loop. Jobs left
running after an interruption are retried on the next invocation. OS-held locks
protect both the batch and individual packs, including a surviving worker after
its parent exits. One process owns each output root. Do not run distributed
workers against an ordinary shared/network filesystem with these locks.

`--workers` bounds active subprocesses (default 1); `--timeout` bounds an individual
worker (default 3,600 seconds); `--delay` spaces submissions (default one second).
Start with 1–2 workers for source access. `workers × cpu-threads` is the potential
ASR CPU load and each ASR worker holds its own model. Use `--limit` for a small run.
Exit 0 means all selected items completed, 1 indicates failed/unavailable items,
and 2 indicates an invalid invocation or batch setup failure.

This is whole-video resume with cached acquisition, not per-stage checkpoints for
frame extraction/OCR. Model calls, hybrid search, cross-video semantic dedupe,
knowledge-graph extraction, and business workflows remain downstream work.

## 5. Add visuals selectively

```sh
uv run multi batch --input selected-visuals.json --out output/visual --with-visuals -- --interval 5 --no-reconstruct
```

Use a separate output root for this pass. Preserve important overlays/whiteboards
by checking the heuristic filter on a few examples. The existing `--budget` is an
image inclusion heuristic, not a dollar spending limit. No LLM automatically reads
the frames in this release.

## Evidence precision and access

`paired.json.transcript` and the manifest record source method and timing type.
ASR segments, caption cues, and estimated intervals are distinguishable. A text
edit invalidates managed cue alignment instead of citing stale text. Sentence
times remain interpolated. A duration inferred from the last cue is not an exact
video duration; a transcript without cue ends gets a marked estimate. Supply
`--duration-seconds` when the exact duration is known.

Access and redistribution permissions are separate from processing capabilities.
Do not use blocked/private sources without authorization. Keep private source
archives out of public code repositories. Corporate proxy users can configure a
trusted CA with `SSL_CERT_FILE` (yt-dlp and the catalog API) and
`REQUESTS_CA_BUNDLE` (caption fetching); neither turns off TLS checks.

## Primary references

- [YouTube uploads inventory](https://developers.google.com/youtube/v3/guides/implementation/videos)
- [Caption endpoint permissions](https://developers.google.com/youtube/v3/docs/captions/download)
- [Transcript library and access limitations](https://github.com/jdepoix/youtube-transcript-api)
- [Local faster-whisper usage](https://github.com/SYSTRAN/faster-whisper)
- [Gemini public-video input and free-tier limit](https://ai.google.dev/gemini-api/docs/video-understanding)

Gemini's free tier can be evaluated later for derived concepts and selected visual
analysis. Its documented YouTube free-tier limit is eight source hours per day,
alongside model quotas. This repository does not implement Gemini calls.
