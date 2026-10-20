"""Single-host resumable batches with bounded subprocess workers and a durable ledger."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from .emit import sha256_file
from .models import extract_video_id
from .transcribe import atomic_json


@contextmanager
def batch_lock(root: Path, filename: str = ".batch.lock"):
    """OS-released advisory lock: a killed runner cannot leave a stale lease."""
    with (root / filename).open("a+b") as lock:
        lock.seek(0)
        if sys.platform == "win32":
            import msvcrt
            if not lock.read(1):
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            acquire = lambda: msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            release = lambda: msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            acquire = lambda: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            release = lambda: fcntl.flock(lock, fcntl.LOCK_UN)
        try:
            acquire()
        except OSError:
            raise RuntimeError(f"Another batch is using {root}") from None
        try:
            yield
        finally:
            lock.seek(0)
            release()


def load_jobs(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        payload = json.loads(text)
        rows = payload.get("videos", []) if isinstance(payload, dict) else payload
    else:
        rows = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not isinstance(rows, list) or not rows:
        raise ValueError("Input must contain a nonempty URL list or catalog videos array")
    jobs = {}
    for item in rows:
        row = {"url": item} if isinstance(item, str) else dict(item)
        video_id = extract_video_id(row["url"])
        row["url"] = f"https://youtu.be/{video_id}"
        row["video_id"] = video_id
        for field in ("transcript", "audio_file"):
            if row.get(field):
                source = Path(row[field])
                source = (path.parent / source).resolve() if not source.is_absolute() else source
                if not source.is_file():
                    raise ValueError(f"Missing {field} file: {source}")
                row[field] = str(source)
        if video_id in jobs and row != jobs[video_id]:
            raise ValueError(f"Conflicting entries for {video_id}")
        jobs[video_id] = row
    return list(jobs.values())


def fingerprint(row: dict, options: list[str], code_hash: str) -> str:
    inputs = {field: sha256_file(Path(row[field])) for field in ("transcript", "audio_file") if row.get(field)}
    if row.get("transcript"):
        source = Path(row["transcript"])
        for suffix in (".cues.json", ".source.json"):
            sidecar = source.with_suffix(suffix)
            if sidecar.is_file():
                inputs[suffix] = sha256_file(sidecar)
    data = json.dumps([row, options, code_hash, inputs], sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()


def pack_hashes(out: Path) -> dict[str, str]:
    required = ["paired.json", "paired.md", "index.json", "index.md", "SKILL.md", "manifest.json"]
    if not all((out / name).is_file() for name in required):
        raise ValueError("Worker did not produce a complete context pack")
    data = json.loads((out / "paired.json").read_text(encoding="utf-8"))
    if not data.get("chunks"):
        raise ValueError("Worker produced an empty context pack")
    files = set(required)
    for path in out.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".jsonl", ".md", ".txt", ".jpg", ".py", ".sh"}:
            files.add(path.relative_to(out).as_posix())
    return {name: sha256_file(out / name) for name in sorted(files)}


def outputs_valid(out: Path, hashes: dict) -> bool:
    return bool(hashes) and all((out / name).is_file() and sha256_file(out / name) == digest for name, digest in hashes.items())


def run_one(row: dict, root: Path, options: list[str], timeout: float) -> dict:
    out = root / row["video_id"]
    out.mkdir(parents=True, exist_ok=True)
    transcript = row.get("transcript") or str(out / "transcript.txt")
    command = [sys.executable, "-m", "youtube_multi.cli", "--url", row["url"], "--out", str(out),
               "--transcript", transcript, *options]
    if row.get("audio_file"):
        command += ["--audio-file", row["audio_file"], "--transcribe", "always"]
    # Preserve API catalog metadata without another yt-dlp metadata request.
    if row.get("title"):
        atomic_json(out / "metadata.json", {k: row[k] for k in
                    ("title", "description", "duration", "channel", "published_at") if k in row})
    started = time.monotonic()
    try:
        with (out / "worker.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        if result.returncode:
            return {"status": "failed", "exit_code": result.returncode, "error": "See worker.log",
                    "elapsed_seconds": round(time.monotonic() - started, 2)}
        hashes = pack_hashes(out)
        return {"status": "completed", "exit_code": 0, "artifacts": hashes,
                "elapsed_seconds": round(time.monotonic() - started, 2), "api_cost_usd": 0}
    except subprocess.TimeoutExpired:
        return {"status": "failed", "error": f"Worker exceeded {timeout:g}s", "exit_code": 124}
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "exit_code": 2}


def execute(jobs: list[dict], root: Path, options: list[str], *, workers: int = 1,
            retry_failed: bool = False, timeout: float = 3600, delay: float = 1) -> dict:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    code_hash = hashlib.sha256(b"".join(p.name.encode() + p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py")))).hexdigest()
    with batch_lock(root):
        ledger_path = root / "batch.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {"schema_version": 1, "jobs": {}}
        records = ledger["jobs"]
        # Validate and fingerprint all input files before starting any worker.
        keyed = [(row, fingerprint(row, options, code_hash)) for row in jobs]
        todo = []
        for row, key in keyed:
            vid = row["video_id"]
            old = records.get(vid, {})
            if row.get("availability") in {"unavailable", "private"}:
                records[vid] = {"status": "unavailable", "fingerprint": key}
                continue
            same = old.get("fingerprint") == key
            if same and old.get("status") == "completed" and outputs_valid(root / vid, old.get("artifacts", {})):
                continue
            if same and old.get("status") == "failed" and not retry_failed:
                continue
            todo.append((row, key))
        atomic_json(ledger_path, ledger)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {}
            remaining = iter(todo)
            exhausted = False
            while pending or not exhausted:
                while len(pending) < workers and not exhausted:
                    item = next(remaining, None)
                    if item is None:
                        exhausted = True
                        break
                    row, key = item
                    vid = row["video_id"]
                    old = records.get(vid, {})
                    records[vid] = {"status": "running", "fingerprint": key, "attempts": old.get("attempts", 0) + 1}
                    atomic_json(ledger_path, ledger)
                    pending[pool.submit(run_one, row, root, options, timeout)] = vid
                    if delay:
                        time.sleep(delay)
                if not pending:
                    break
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    vid = pending.pop(future)
                    records[vid].update(future.result())
                    atomic_json(ledger_path, ledger)
                    print(f"{vid}: {records[vid]['status']}")
        selected = [records[row["video_id"]] for row in jobs]
        ledger["summary"] = {status: sum(r["status"] == status for r in selected)
                             for status in ("completed", "failed", "unavailable", "running")}
        ledger["summary"]["api_cost_usd"] = 0
        atomic_json(ledger_path, ledger)
        return ledger["summary"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="multi batch", description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("output"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--with-visuals", action="store_true", help="Default is transcript-only")
    parser.add_argument("--timeout", type=float, default=3600, help="Seconds per worker")
    parser.add_argument("--delay", type=float, default=1, help="Seconds between job submissions")
    parser.add_argument("options", nargs=argparse.REMAINDER, help="Single-video options after --")
    args = parser.parse_args(argv)
    options = args.options[1:] if args.options[:1] == ["--"] else args.options
    if any(opt.split("=", 1)[0] in {"--url", "--out", "--transcript", "--audio-file"} for opt in options):
        parser.error("Per-video paths/URLs belong in the JSON input, not shared options")
    if not args.with_visuals:
        options = ["--transcript-only", *options]
    from .cli import build_parser
    build_parser().parse_args(["--url", "https://youtu.be/AAAAAAAAAAA", *options])
    import math
    if args.workers < 1 or args.workers > 32 or args.timeout <= 0 or args.delay < 0 or not all(map(math.isfinite, [args.timeout, args.delay])):
        parser.error("workers must be 1–32, timeout positive, delay nonnegative, and times finite")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    try:
        jobs = load_jobs(args.input)
        result = execute(jobs[:args.limit], args.out, options, workers=args.workers,
                         retry_failed=args.retry_failed, timeout=args.timeout, delay=args.delay)
        print(json.dumps(result, sort_keys=True))
        return 1 if result["failed"] or result["unavailable"] else 0
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Batch failed: {exc}", file=sys.stderr)
        return 2
